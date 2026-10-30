"""Live gesture recognition.

    python inference.py --profile default                    # recognize, print only (control OFF)
    python inference.py --enable-control --dry-run           # print the actions that would run
    python inference.py --enable-control                     # actually press keys
    python inference.py --simulate                           # demo on the simulator

Pipeline per hop (default 100 ms) over the newest 1.0 s of the ring buffer:

    window --STFT crop--> motion GATE (Doppler sideband energy) --open?--> CNN
           --> confidence threshold --> N consistent predictions --> trigger --> cooldown

The gate keeps the classifier idle (and false triggers low) while nothing moves.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from actions import ActionBackendError, ActionConfigError, ActionExecutor, environment_notes, load_actions
from audio_stream import AudioSource
from config import DSPConfig, InferenceConfig
from dsp import as_channels_first, features_from_stft, motion_score_db, snr_at_frequency, stft_crop
from model import GestureCNN

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Gate / smoothing / cooldown (pure logic, clock injected for testability)
# --------------------------------------------------------------------------- #
class MotionGate:
    """Open while the window's Doppler motion score exceeds ``threshold_db``.

    Stays open for ``hold_seconds`` after the score drops, so the tail of a gesture
    is still classified.
    """

    def __init__(self, threshold_db: float, hold_seconds: float = 0.3) -> None:
        self.threshold_db = threshold_db
        self.hold_seconds = hold_seconds
        self._last_above: float | None = None

    def update(self, score_db: float, now: float) -> bool:
        if score_db >= self.threshold_db:
            self._last_above = now
            return True
        return self._last_above is not None and now - self._last_above <= self.hold_seconds

    def reset(self) -> None:
        self._last_above = None


class PredictionSmoother:
    """Fire only after ``n`` consecutive confident predictions of the same non-idle class."""

    def __init__(self, n: int, confidence: float, reject_label: str = "idle") -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.confidence = confidence
        self.reject_label = reject_label
        self._hist: deque[str | None] = deque(maxlen=n)

    def update(self, label: str, prob: float) -> str | None:
        self._hist.append(label if prob >= self.confidence and label != self.reject_label else None)
        if len(self._hist) == self.n and self._hist[0] is not None and all(h == self._hist[0] for h in self._hist):
            return self._hist[0]
        return None

    def reset(self) -> None:
        self._hist.clear()


class Cooldown:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._until = float("-inf")

    def active(self, now: float) -> bool:
        return now < self._until

    def trigger(self, now: float) -> None:
        self._until = now + self.seconds


@dataclass
class StepResult:
    """Output of one recognizer step."""

    motion_db: float
    gate_open: bool
    classified: bool = False
    probs: dict[str, float] = field(default_factory=dict)
    label: str | None = None
    confidence: float = 0.0
    triggered: str | None = None
    cooldown: bool = False


class GestureRecognizer:
    """Stateful gate + classifier + smoothing + cooldown over successive windows."""

    def __init__(
        self,
        model: GestureCNN,
        class_names: Sequence[str],
        dsp: DSPConfig,
        cfg: InferenceConfig | None = None,
        gate_db: float | None = None,
    ) -> None:
        self.cfg = cfg or InferenceConfig()
        self.model = model.eval()
        self.class_names = list(class_names)
        self.dsp = dsp
        threshold = self.cfg.gate_db if self.cfg.gate_db is not None else gate_db
        self.gate = MotionGate(
            threshold if threshold is not None else self.cfg.default_gate_db, self.cfg.gate_hold_seconds
        )
        self.smoother = PredictionSmoother(self.cfg.smoothing, self.cfg.confidence)
        self.cooldown = Cooldown(self.cfg.cooldown_seconds)
        self._window = dsp.samples(dsp.window_seconds)

    def classify(self, z: np.ndarray) -> np.ndarray:
        feats = features_from_stft(z, self.dsp)
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(feats)[None])
        return torch.softmax(logits, dim=1)[0].numpy()

    def step(self, audio: np.ndarray, now: float) -> StepResult:
        """Process the newest window of ``audio`` (``(n, C)`` or ``(C, n)``) at time ``now``."""
        x = as_channels_first(audio)
        if x.shape[1] < self._window:
            return StepResult(0.0, False)
        z = stft_crop(x[:, -self._window :], self.dsp)
        score = motion_score_db(z, self.dsp)
        gate_open = self.gate.update(score, now)
        res = StepResult(score, gate_open, cooldown=self.cooldown.active(now))
        if not gate_open:
            self.smoother.reset()
            return res
        probs = self.classify(z)
        k = int(np.argmax(probs))
        res.classified = True
        res.probs = {c: float(p) for c, p in zip(self.class_names, probs)}
        res.label, res.confidence = self.class_names[k], float(probs[k])
        if res.cooldown:
            self.smoother.reset()
            return res
        fired = self.smoother.update(res.label, res.confidence)
        if fired is not None:
            res.triggered = fired
            self.cooldown.trigger(now)
            self.smoother.reset()
            res.cooldown = True
        return res


# --------------------------------------------------------------------------- #
# Background worker
# --------------------------------------------------------------------------- #
@dataclass
class TriggerEvent:
    time: float
    wall_time: str
    gesture: str
    confidence: float
    action: str


@dataclass
class LiveState:
    running: bool = False
    gate_open: bool = False
    motion_db: float = 0.0
    probs: dict[str, float] = field(default_factory=dict)
    label: str | None = None
    confidence: float = 0.0
    cooldown: bool = False
    snr_db: float | None = None
    carrier_lost: bool = False
    control_enabled: bool = False
    dry_run: bool = True
    steps: int = 0
    classified_steps: int = 0
    avg_step_ms: float = 0.0
    events: list[TriggerEvent] = field(default_factory=list)
    error: str | None = None


class InferenceWorker:
    """Runs a :class:`GestureRecognizer` over an audio source on a background thread.

    The UI (CLI or Streamlit) only reads :meth:`snapshot`; it never blocks the worker.
    """

    def __init__(
        self,
        engine: AudioSource,
        recognizer: GestureRecognizer,
        executor: ActionExecutor,
        cfg: InferenceConfig | None = None,
        max_events: int = 50,
        on_trigger: Any = None,
    ) -> None:
        self.engine = engine
        self.recognizer = recognizer
        self.executor = executor
        self.cfg = cfg or recognizer.cfg
        self._state = LiveState(control_enabled=executor.enabled, dry_run=executor.effective_dry_run)
        self._events: deque[TriggerEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_snr = float("-inf")
        self._carrier_lost = False
        self._on_trigger = on_trigger

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="inference", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        with self._lock:
            self._state.running = False

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_control(self, enabled: bool) -> None:
        self.executor.set_enabled(enabled)
        with self._lock:
            self._state.control_enabled = self.executor.enabled

    def snapshot(self) -> LiveState:
        with self._lock:
            s = LiveState(**{k: v for k, v in vars(self._state).items() if k != "events"})
            s.probs = dict(self._state.probs)
            s.events = list(self._events)
        return s

    def process_once(self, now: float | None = None) -> StepResult:
        """Run one recognizer step on the newest audio (also used by tests)."""
        now = time.monotonic() if now is None else now
        dsp = self.recognizer.dsp
        t0 = time.perf_counter()
        audio = self.engine.get_latest(dsp.window_seconds)
        snr = None
        if now - self._last_snr >= self.cfg.snr_interval_seconds and len(audio) >= 8192:
            self._last_snr = now
            try:
                snr = snr_at_frequency(audio, dsp.sample_rate, dsp.carrier_hz)
            except ValueError:
                snr = None
            if snr is not None:
                lost = snr < self.cfg.min_carrier_snr_db
                if lost != self._carrier_lost:
                    if lost:
                        logger.warning(
                            "Carrier tone lost (SNR %.0f dB): is the speaker muted or the volume too low? "
                            "Recognition is paused until it returns.",
                            snr,
                        )
                    else:
                        logger.info("Carrier tone is back (SNR %.0f dB); recognition resumed.", snr)
                self._carrier_lost = lost
        if self._carrier_lost:
            # Without the carrier there is no echo to classify; the window is only noise.
            self.recognizer.smoother.reset()
            res = StepResult(0.0, False)
        else:
            res = self.recognizer.step(audio, now)
        dt_ms = (time.perf_counter() - t0) * 1000
        event = None
        if res.triggered:
            try:
                line = self.executor.execute(res.triggered)
            except ActionBackendError as exc:
                line = f"{res.triggered}: action failed ({exc})"
                logger.error(line)
            event = TriggerEvent(now, time.strftime("%H:%M:%S"), res.triggered, res.confidence, line)
            logger.info("Gesture: %s (p=%.2f) -> %s", res.triggered, res.confidence, line)
        with self._lock:
            st = self._state
            st.running = True
            st.dry_run = self.executor.effective_dry_run
            st.gate_open, st.motion_db, st.cooldown = res.gate_open, res.motion_db, res.cooldown
            if res.classified:
                st.probs, st.label, st.confidence = res.probs, res.label, res.confidence
                st.classified_steps += 1
            elif not res.gate_open:
                st.label, st.confidence = None, 0.0
            if snr is not None:
                st.snr_db = snr
            st.carrier_lost = self._carrier_lost
            st.steps += 1
            st.avg_step_ms = 0.9 * st.avg_step_ms + 0.1 * dt_ms if st.steps > 1 else dt_ms
            if event is not None:
                self._events.append(event)
        if event is not None and self._on_trigger is not None:
            self._on_trigger(event)
        return res

    def _run(self) -> None:
        hop = self.cfg.hop_seconds
        next_t = time.monotonic()
        while not self._stop.is_set():
            try:
                self.process_once()
            except Exception as exc:  # keep running; surface the error to the UI
                logger.exception("Inference step failed")
                with self._lock:
                    self._state.error = repr(exc)
            next_t += hop
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_t = time.monotonic()
        with self._lock:
            self._state.running = False


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    from cli_common import add_common_args, print_banner_if_simulated, setup_logging
    from engine import EngineError, LiveSession

    parser = argparse.ArgumentParser(description="EchoTouch live gesture recognition")
    add_common_args(parser, audio=True)
    parser.add_argument(
        "--enable-control", action="store_true", help="allow gestures to trigger OS actions (OFF by default)"
    )
    parser.add_argument("--dry-run", action="store_true", help="print actions instead of executing them")
    parser.add_argument("--actions", default=None, help="actions YAML (default: actions.yaml)")
    parser.add_argument("--confidence", type=float, default=InferenceConfig.confidence)
    parser.add_argument("--smoothing", type=int, default=InferenceConfig.smoothing)
    parser.add_argument("--cooldown", type=float, default=InferenceConfig.cooldown_seconds)
    parser.add_argument("--gate-db", type=float, default=None, help="override the learned motion-gate threshold")
    parser.add_argument("--demo", action="store_true", help="with --simulate: perform random simulated gestures")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    print_banner_if_simulated(args.simulate)
    if args.list_devices:
        from audio_stream import AudioDeviceError, list_devices_text

        try:
            print(list_devices_text())
        except AudioDeviceError as exc:
            print(f"ERROR: {exc}")
            return 2
        return 0
    try:
        actions = load_actions(args.actions)
    except ActionConfigError as exc:
        print(f"ERROR in actions file: {exc}")
        return 2
    if args.enable_control and not args.dry_run:
        for note in environment_notes():
            print(f"NOTE: {note}")
    cfg = InferenceConfig(
        confidence=args.confidence, smoothing=args.smoothing, cooldown_seconds=args.cooldown, gate_db=args.gate_db
    )
    try:
        session = LiveSession.open(
            profile=args.profile,
            simulate=args.simulate,
            input_device=args.input_device,
            output_device=args.output_device,
            actions=actions,
            cfg=cfg,
            control=args.enable_control,
            dry_run=args.dry_run,
            auto_demo=args.demo,
        )
    except (EngineError, ActionBackendError) as exc:
        print(f"ERROR: {exc}")
        return 2
    dry = session.worker.executor.effective_dry_run
    mode = "OFF (recognition only)" if not args.enable_control else ("DRY-RUN (print only)" if dry else "ON")
    print(
        f"Classes: {', '.join(session.class_names)} | gate {session.recognizer.gate.threshold_db:.1f} dB | "
        f"OS control: {mode}"
    )
    print("Listening... press Ctrl+C to stop.")
    try:
        with session:
            last_status = 0.0
            while True:
                time.sleep(0.25)
                st = session.worker.snapshot()
                if st.error:
                    print(f"ERROR: {st.error}")
                    return 1
                if time.monotonic() - last_status > 5.0:
                    last_status = time.monotonic()
                    snr = f"{st.snr_db:.0f} dB" if st.snr_db is not None else "?"
                    print(f"  [status] carrier SNR {snr} | motion {st.motion_db:.1f} dB | step {st.avg_step_ms:.1f} ms")
                    if st.carrier_lost:
                        print(
                            "  !! NO CARRIER: EchoTouch can't hear its own tone, so it can't see gestures.\n"
                            "     Check the speaker isn't muted and the system volume is where you calibrated it.\n"
                            "     (Lower a video's volume in the player, not the Windows master volume.)"
                        )
    except KeyboardInterrupt:
        print("\nStopped.")
    except EngineError as exc:
        print(f"ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
