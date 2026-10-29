"""Thin orchestration layer shared by the CLIs and the Streamlit dashboard.

Heavy lifting lives elsewhere (dsp.py, model.py, train.py, inference.py, ...).
This module only wires the pieces together and owns their lifecycles:

* :func:`open_store` / :func:`profile_dsp` - pick the right profile directory.
* :func:`create_audio` - real :class:`AudioEngine` or :class:`SimulatedAudioEngine`.
* :class:`LiveSession` - audio + model + recognizer + action executor + worker thread.
* :class:`TrainingJob` - training on a background thread with a pollable progress log.
* :class:`AppRuntime` - long-lived holder for the dashboard (lives in ``st.cache_resource``).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

from actions import Action, ActionExecutor
from audio_stream import AudioDeviceError, AudioEngine, AudioSource
from config import SOURCE_REAL, AudioConfig, ConfigMismatchError, DSPConfig, GestureSpec, InferenceConfig, TrainConfig
from inference import GestureRecognizer, InferenceWorker
from model import CheckpointError, load_checkpoint
from profiles import ProfileError, ProfileStore
from simulator import SimulatedAudioEngine

logger = logging.getLogger(__name__)


class EngineError(RuntimeError):
    """User-facing error with an actionable message."""


def open_store(profile: str, simulate: bool) -> ProfileStore:
    """Profile store in ``sim/`` (simulate) or ``profiles/`` (real)."""
    return ProfileStore(profile, simulated=simulate)


def profile_dsp(store: ProfileStore, require_calibration: bool = True) -> DSPConfig:
    """The profile's DSP config; real profiles must be calibrated first."""
    if not store.exists():
        if store.simulated:
            return DSPConfig()
        raise EngineError(
            f"Profile '{store.name}' does not exist yet. Run `python doctor.py --profile {store.name}` first."
        )
    if require_calibration and not store.simulated and not store.calibration:
        raise EngineError(
            f"Profile '{store.name}' is not calibrated. Run `python doctor.py --profile {store.name}` first."
        )
    return store.dsp_config()


def create_audio(
    dsp: DSPConfig,
    volume: float,
    simulate: bool,
    input_device: int | str | None = None,
    output_device: int | str | None = None,
    realtime: bool = True,
    auto_demo: bool = False,
    demo_gestures: Sequence[str] = ("push", "pull", "wave"),
    scenario: str = "normal",
    seed: int = 0,
) -> AudioSource:
    audio_cfg = AudioConfig(volume=volume)
    if simulate:
        return SimulatedAudioEngine(
            dsp,
            audio_cfg,
            realtime=realtime,
            seed=seed,
            scenario=scenario,
            auto_demo=auto_demo,
            demo_gestures=tuple(demo_gestures),
        )
    return AudioEngine(dsp, audio_cfg, input_device=input_device, output_device=output_device)


class LiveSession:
    """Everything needed for live recognition, started/stopped as one unit."""

    def __init__(
        self,
        engine: AudioSource,
        worker: InferenceWorker,
        recognizer: GestureRecognizer,
        meta: dict[str, Any],
        owns_engine: bool = True,
    ) -> None:
        self.engine = engine
        self.worker = worker
        self.recognizer = recognizer
        self.meta = meta
        self._owns_engine = owns_engine

    @property
    def class_names(self) -> list[str]:
        return list(self.recognizer.class_names)

    @classmethod
    def open(
        cls,
        profile: str,
        simulate: bool,
        actions: dict[str, Action],
        cfg: InferenceConfig | None = None,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        control: bool = False,
        dry_run: bool = True,
        auto_demo: bool = False,
        engine: AudioSource | None = None,
        realtime: bool = True,
    ) -> LiveSession:
        """Load the profile's model with all guardrails and build the live pipeline.

        Raises :class:`EngineError` if the model is missing, simulated (outside
        ``simulate``), or was trained with different DSP settings.
        """
        cfg = cfg or InferenceConfig()
        try:
            store = open_store(profile, simulate)
            dsp = profile_dsp(store)
            model, meta = load_checkpoint(store.model_path, current_dsp=dsp, allow_simulated=simulate)
        except (CheckpointError, ConfigMismatchError, ProfileError) as exc:
            raise EngineError(str(exc)) from exc
        if engine is not None and bool(engine.simulated) != bool(simulate):
            raise EngineError("Audio engine and mode disagree (simulated vs real). Restart audio.")
        if engine is not None and (engine.sample_rate != dsp.sample_rate or engine.channels != dsp.channels):
            raise EngineError("The running audio engine does not match the model's sample rate/channels.")
        recognizer = GestureRecognizer(
            model, meta["class_names"], dsp, cfg, gate_db=meta.get("extra", {}).get("gate_db")
        )
        simulated_model = meta.get("source") != SOURCE_REAL
        executor = ActionExecutor(actions, enabled=False, dry_run=dry_run, force_dry_run=simulate or simulated_model)
        if control:
            executor.set_enabled(True)
        owns = engine is None
        if engine is None:
            demo = [c for c in meta["class_names"] if c != "idle"]
            engine = create_audio(
                dsp,
                store.volume,
                simulate,
                input_device,
                output_device,
                realtime=realtime,
                auto_demo=auto_demo,
                demo_gestures=demo,
            )
        elif abs(engine.carrier_hz - dsp.carrier_hz) > 1e-6:
            engine.set_frequency(dsp.carrier_hz)
        worker = InferenceWorker(engine, recognizer, executor, cfg)
        return cls(engine, worker, recognizer, meta, owns_engine=owns)

    def start(self) -> None:
        try:
            self.engine.start()
        except AudioDeviceError as exc:
            raise EngineError(str(exc)) from exc
        self.worker.start()

    def stop(self) -> None:
        self.worker.stop()
        if self._owns_engine:
            self.engine.stop()

    def __enter__(self) -> LiveSession:
        self.start()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.stop()


@dataclass
class TrainingJob:
    """Runs :func:`train.train_profile` on a background thread; the UI polls :meth:`progress`."""

    status: str = "idle"  # idle | running | done | error | stopped
    events: list[dict[str, Any]] = field(default_factory=list)
    result: Any = None
    error: str | None = None
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start(
        self,
        store: ProfileStore,
        gestures: Sequence[GestureSpec],
        cfg: TrainConfig | None = None,
        allow_simulated: bool = False,
    ) -> None:
        from train import train_profile

        with self._lock:
            if self.status == "running":
                raise EngineError("Training is already running.")
            self.status, self.events, self.result, self.error = "running", [], None, None
            self._stop.clear()

        def _target() -> None:
            try:
                res = train_profile(
                    store,
                    list(gestures),
                    cfg,
                    allow_simulated=allow_simulated,
                    callback=self._on_event,
                    stop_event=self._stop,
                )
                with self._lock:
                    self.result = res
                    self.status = "stopped" if res.stopped_early else "done"
            except Exception as exc:
                logger.exception("Training failed")
                with self._lock:
                    self.error = str(exc)
                    self.status = "error"

        self._thread = threading.Thread(target=_target, name="training", daemon=True)
        self._thread.start()

    def _on_event(self, ev: dict[str, Any]) -> None:
        with self._lock:
            self.events.append(ev)

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def progress(self) -> dict[str, Any]:
        with self._lock:
            return {"status": self.status, "events": list(self.events), "result": self.result, "error": self.error}


class AppRuntime:
    """Long-lived state for the dashboard: one audio engine, one live session, one training job.

    Streamlit reruns the script on every interaction; this object (kept in
    ``st.cache_resource``) survives reruns so audio never restarts unexpectedly.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.engine: AudioSource | None = None
        self.engine_key: tuple[Any, ...] | None = None
        self.live: LiveSession | None = None
        self.training = TrainingJob()
        self.enrollment: Any = None  # enroll.EnrollmentController, created lazily by the UI

    def ensure_audio(
        self,
        dsp: DSPConfig,
        volume: float,
        simulate: bool,
        input_device: int | str | None,
        output_device: int | str | None,
    ) -> AudioSource:
        """Return a running engine matching the arguments, restarting it if they changed."""
        key = (dsp.sample_rate, dsp.channels, dsp.carrier_hz, volume, simulate, input_device, output_device)
        with self._lock:
            if self.engine is not None and self.engine_key == key and self.engine.running:
                return self.engine
            self.stop_audio()
            eng = create_audio(dsp, volume, simulate, input_device, output_device)
            try:
                eng.start()
            except AudioDeviceError as exc:
                raise EngineError(str(exc)) from exc
            self.engine, self.engine_key = eng, key
            return eng

    def stop_live(self) -> None:
        with self._lock:
            if self.live is not None:
                self.live.stop()
                self.live = None

    def stop_audio(self) -> None:
        with self._lock:
            self.stop_live()
            if self.engine is not None:
                self.engine.stop()
                self.engine, self.engine_key = None, None

    def shutdown(self) -> None:
        self.training.stop()
        self.stop_audio()
