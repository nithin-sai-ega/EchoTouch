"""Physics-based acoustic Doppler simulator: FOR TESTS, CI AND DEMOS ONLY.

SIMULATED MODE: not real data. Nothing produced here may train a model used for
real control. Simulated clips are written only to ``sim/`` and are tagged
``source: "simulated"``; simulated models are refused by live control.

Model: a speaker S and one or more mics M sit on the laptop (screen plane z=0).
The mic receives
  * direct speaker->mic leakage (loud, static),
  * a few static reflections (body, desk),
  * the echo off the moving hand at position p(t), with path length
    L(t) = |p - S| + |p - M| and amplitude ~ 1 / (|p - S| |p - M|),
    rendered exactly as cos(2 pi f0 (t - L(t)/c)), so Doppler emerges naturally,
  * white noise, keyboard clicks, distant movers, etc. for the idle sub-types.

Because the geometry is bistatic and asymmetric, left/right and up/down swipes
are only weakly distinguishable here, which is also the honest expectation for
real single-mic hardware.

CLI: ``python simulator.py --profile demo`` writes a small demo dataset to sim/demo.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

import numpy as np

from audio_stream import RingBuffer
from config import (
    IDLE_SUBTYPES,
    SIMULATED_BANNER,
    SOURCE_SIMULATED,
    AudioConfig,
    DSPConfig,
)

logger = logging.getLogger(__name__)

SPEED_OF_SOUND = 343.0
IDLE_SUBTYPE_NAMES = tuple(name for name, _ in IDLE_SUBTYPES)
SCENARIOS = ("normal", "noise_suppression", "no_tone", "clipping")


# --------------------------------------------------------------------------- #
# Geometry and acoustics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SimAcoustics:
    """Fixed per-"laptop" parameters (metres, linear amplitudes at volume 0.3)."""

    speaker: tuple[float, float, float] = (0.12, -0.12, 0.02)
    mics: tuple[tuple[float, float, float], ...] = ((0.0, 0.20, 0.0), (0.05, 0.20, 0.0))
    leakage: float = 0.2
    hand_reflectivity: float = 4e-4
    noise_std: float = 2e-3
    #: Speaker/mic roll-off above 18 kHz (dB per kHz) and a resonance notch.
    rolloff_db_per_khz: float = 4.0
    notch_hz: float = 19500.0
    notch_db: float = 18.0
    notch_width_hz: float = 250.0
    reference_volume: float = 0.3

    def gain(self, freq: float) -> float:
        db = -max(0.0, freq - 18000.0) / 1000.0 * self.rolloff_db_per_khz
        db -= self.notch_db * np.exp(-0.5 * ((freq - self.notch_hz) / self.notch_width_hz) ** 2)
        return float(10 ** (db / 20.0))


def _ease(u: np.ndarray) -> np.ndarray:
    """Raised-cosine ease 0->1 (zero velocity at both ends)."""
    u = np.clip(u, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * u)


def _name_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


@dataclass
class SimEvent:
    """A gesture or idle activity occurring at absolute time ``start`` (seconds)."""

    gesture: str
    start: float
    duration: float
    params: dict[str, Any] = field(default_factory=dict)
    idle_subtype: str | None = None
    pre: float = 0.25
    post: float = 0.25

    @property
    def end(self) -> float:
        return self.start + self.duration + self.post

    @property
    def begin(self) -> float:
        return self.start - self.pre


def _hand_path(ev: SimEvent, t: np.ndarray) -> np.ndarray:
    """Hand position ``(n, 3)`` for gesture events at absolute times ``t``."""
    p = ev.params
    u = (t - ev.start) / ev.duration
    e = _ease(u)[:, None]
    g = ev.gesture
    a = np.asarray(p["a"], dtype=np.float64)
    b = np.asarray(p["b"], dtype=np.float64)
    if g == "wave":
        cycles, amp = p["cycles"], p["amp"]
        uc = np.clip(u, 0.0, 1.0)
        env = np.sin(np.pi * uc)  # amplitude grows then decays
        pos = np.repeat(a[None, :], len(t), axis=0)
        pos[:, 0] += amp * env * np.sin(2 * np.pi * cycles * uc)
        pos[:, 2] += 0.25 * amp * env * np.sin(2 * np.pi * cycles * uc + p["phase"])
        return pos
    return a[None, :] + e * (b - a)[None, :]


def gesture_params(gesture: str, rng: np.random.Generator) -> tuple[float, dict[str, Any]]:
    """Random duration and trajectory parameters for one gesture instance."""
    speed = rng.uniform(0.75, 1.35)
    cx, cy = rng.normal(0.0, 0.03), rng.normal(0.05, 0.03)
    near, far = rng.uniform(0.06, 0.11), rng.uniform(0.25, 0.38)
    mid = rng.uniform(0.12, 0.2)
    span = rng.uniform(0.2, 0.3)
    dur = rng.uniform(0.35, 0.55) / speed
    if gesture == "push":
        return dur, {"a": (cx, cy, far), "b": (cx, cy, near)}
    if gesture == "pull":
        return dur, {"a": (cx, cy, near), "b": (cx, cy, far)}
    if gesture == "wave":
        cycles = rng.uniform(2.0, 3.0)
        return cycles / rng.uniform(2.5, 4.0), {
            "a": (cx, cy, mid),
            "b": (cx, cy, mid),
            "cycles": cycles,
            "amp": rng.uniform(0.07, 0.12),
            "phase": rng.uniform(0, 2 * np.pi),
        }
    if gesture == "swipe_left":
        return dur, {"a": (span, cy, mid), "b": (-span, cy, mid)}
    if gesture == "swipe_right":
        return dur, {"a": (-span, cy, mid), "b": (span, cy, mid)}
    if gesture == "swipe_up":
        return dur, {"a": (cx, -0.15, mid), "b": (cx, 0.3, mid)}
    if gesture == "swipe_down":
        return dur, {"a": (cx, 0.3, mid), "b": (cx, -0.15, mid)}
    # Custom gesture: a consistent (name-seeded) straight path plus per-instance jitter.
    crng = np.random.default_rng(_name_seed(gesture))
    a = crng.uniform([-0.2, -0.1, 0.08], [0.2, 0.3, 0.35]) + rng.normal(0, 0.02, 3)
    b = crng.uniform([-0.2, -0.1, 0.08], [0.2, 0.3, 0.35]) + rng.normal(0, 0.02, 3)
    return dur, {"a": tuple(a), "b": tuple(b)}


def make_event(gesture: str, start: float, rng: np.random.Generator, idle_subtype: str | None = None) -> SimEvent:
    """Create a randomized :class:`SimEvent` for ``gesture`` (or an idle sub-type)."""
    if gesture == "idle":
        sub = idle_subtype or str(rng.choice(IDLE_SUBTYPE_NAMES))
        if sub not in IDLE_SUBTYPE_NAMES:
            raise ValueError(f"unknown idle subtype {sub!r}")
        params: dict[str, Any] = {"seed": int(rng.integers(1 << 31))}
        if sub == "walking_by":
            z = rng.uniform(1.2, 2.5)
            direction = rng.choice([-1.0, 1.0])
            params.update(a=(-1.5 * direction, 0.3, z), b=(1.5 * direction, 0.3, z))
        elif sub == "head_movement":
            params.update(
                a=(rng.normal(0, 0.05), 0.35, rng.uniform(0.45, 0.65)),
                amp=rng.uniform(0.02, 0.05),
                rate=rng.uniform(0.4, 1.2),
            )
        return SimEvent("idle", start, rng.uniform(1.0, 2.5), params, idle_subtype=sub, pre=0.0, post=0.0)
    dur, params = gesture_params(gesture, rng)
    params["refl"] = rng.uniform(0.6, 1.5)
    params["phase0"] = rng.uniform(0, 2 * np.pi)
    return SimEvent(gesture, start, dur, params)


# --------------------------------------------------------------------------- #
# Scene
# --------------------------------------------------------------------------- #
class SimScene:
    """Renders mic signals for absolute sample ranges; deterministic given its events.

    Rendering is stateless in time (apart from the noise RNG), so consecutive blocks
    join seamlessly and Doppler shifts arise exactly from the time-varying path length.
    """

    def __init__(
        self,
        dsp: DSPConfig,
        acoustics: SimAcoustics | None = None,
        volume: float = 0.3,
        seed: int = 0,
        scenario: str = "normal",
        noise: bool = True,
    ) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        if dsp.channels > len((acoustics or SimAcoustics()).mics):
            raise ValueError("simulator has fewer mics than requested channels")
        self.dsp = dsp
        self.ac = acoustics or SimAcoustics()
        self.sr = dsp.sample_rate
        self.frequency = dsp.carrier_hz
        self.volume = volume
        self.tone_on = True
        self.scenario = scenario
        self.noise = noise
        self._rng = np.random.default_rng(seed)
        srng = np.random.default_rng(seed + 1)
        self._static = [
            (
                np.array([srng.uniform(-0.3, 0.3), srng.uniform(-0.3, 0.1), srng.uniform(0.4, 0.8)]),
                srng.uniform(0.5, 2.0) * 4e-4,
                srng.uniform(0, 2 * np.pi),
            )
            for _ in range(3)
        ]
        self.events: list[SimEvent] = []
        self._lock = threading.Lock()

    def add_event(self, ev: SimEvent) -> None:
        with self._lock:
            self.events.append(ev)

    def prune(self, before: float) -> None:
        with self._lock:
            self.events = [e for e in self.events if e.end >= before]

    def active_gesture(self, t: float) -> str | None:
        with self._lock:
            for ev in self.events:
                if ev.begin <= t <= ev.end:
                    return ev.gesture if ev.gesture != "idle" else f"idle:{ev.idle_subtype}"
        return None

    # -- rendering ------------------------------------------------------------ #
    def _reflector(self, pos: np.ndarray, t: np.ndarray, amp: np.ndarray | float, phase: float) -> np.ndarray:
        """Echo from a point reflector at ``pos`` (n,3) or (3,), for each mic -> (n, C)."""
        spk = np.asarray(self.ac.speaker)
        out = np.empty((len(t), self.dsp.channels))
        d_s = np.maximum(np.linalg.norm(pos - spk, axis=-1), 0.04)
        for c in range(self.dsp.channels):
            mic = np.asarray(self.ac.mics[c])
            d_m = np.maximum(np.linalg.norm(pos - mic, axis=-1), 0.04)
            delay = (d_s + d_m) / SPEED_OF_SOUND
            out[:, c] = amp / (d_s * d_m) * np.cos(2 * np.pi * self.frequency * (t - delay) + phase)
        return out

    def render(self, start_sample: int, n: int) -> np.ndarray:
        """Return mic signal ``(n, channels)`` float32 for samples ``[start, start+n)``."""
        t = (start_sample + np.arange(n)) / self.sr
        t0, t1 = t[0], t[-1]
        level = (self.volume / self.ac.reference_volume) * self.ac.gain(self.frequency)
        if not self.tone_on or self.scenario == "no_tone":
            level = 0.0
        out = np.zeros((n, self.dsp.channels))
        with self._lock:
            events = [e for e in self.events if e.begin <= t1 and e.end >= t0]
        if level > 0:
            # Everything that is a reflection of the carrier scales with the tone level.
            spk = np.asarray(self.ac.speaker)
            for c in range(self.dsp.channels):
                d = np.linalg.norm(np.asarray(self.ac.mics[c]) - spk)
                out[:, c] += self.ac.leakage * np.cos(2 * np.pi * self.frequency * (t - d / SPEED_OF_SOUND))
            for pos, refl, ph in self._static:
                out += self._reflector(pos, t, refl, ph)
            for ev in events:
                if ev.gesture == "idle":
                    out += self._render_idle(ev, t)
                    continue
                pos = _hand_path(ev, t)
                env = np.clip((t - ev.begin) / 0.1, 0, 1) * np.clip((ev.end - t) / 0.1, 0, 1)
                amp = self.ac.hand_reflectivity * ev.params["refl"] * env
                out += self._reflector(pos, t, amp, ev.params["phase0"])
            out *= level
        noise_scale = 1.0
        for ev in events:
            if ev.idle_subtype == "typing":
                out += self._clicks(ev, t)
            elif ev.idle_subtype == "ambient":
                noise_scale = 2.0
        if self.scenario == "noise_suppression":
            # OS "voice enhancement" gating the stationary tone on and off.
            phase = np.mod(t, 1.6)
            out *= np.where(phase < 0.45, 0.03, 1.0)[:, None]
        if self.noise:
            out += self._rng.normal(0.0, self.ac.noise_std * noise_scale, size=out.shape)
        if self.scenario == "clipping":
            out = np.clip(out * 8.0, -1.0, 1.0)
        return out.astype(np.float32)

    def _render_idle(self, ev: SimEvent, t: np.ndarray) -> np.ndarray:
        """Idle activities that reflect the carrier (returned before global level scaling)."""
        p = ev.params
        n = len(t)
        inside = ((t >= ev.start) & (t <= ev.start + ev.duration)).astype(np.float64)
        if not inside.any():
            return np.zeros((n, self.dsp.channels))
        if ev.idle_subtype == "walking_by":
            u = (t - ev.start) / ev.duration
            a, b = np.asarray(p["a"]), np.asarray(p["b"])
            pos = a[None, :] + np.clip(u, 0, 1)[:, None] * (b - a)[None, :]
            return self._reflector(pos, t, 6e-3 * inside, 0.3)
        if ev.idle_subtype == "head_movement":
            a = np.asarray(p["a"])
            pos = np.repeat(a[None, :], n, axis=0)
            pos[:, 2] += p["amp"] * np.sin(2 * np.pi * p["rate"] * (t - ev.start))
            pos[:, 0] += 0.5 * p["amp"] * np.sin(2 * np.pi * 0.7 * p["rate"] * (t - ev.start))
            return self._reflector(pos, t, 3e-3 * inside, 1.1)
        if ev.idle_subtype == "typing":
            # Fingers/wrists jiggle a few mm above the keyboard.
            rng = np.random.default_rng(p["seed"])
            f1, f2 = rng.uniform(3, 7, 2)
            pos = np.repeat(np.array([[0.0, -0.1, 0.06]]), n, axis=0)
            pos[:, 2] += 0.004 * np.sin(2 * np.pi * f1 * t) + 0.003 * np.sin(2 * np.pi * f2 * t + 1.0)
            return self._reflector(pos, t, 1.5e-4 * inside, 0.5)
        return np.zeros((n, self.dsp.channels))

    def _clicks(self, ev: SimEvent, t: np.ndarray) -> np.ndarray:
        """Keyboard clicks: short decaying broadband bursts at deterministic times."""
        rng = np.random.default_rng(ev.params["seed"])
        n_clicks = int(ev.duration * rng.uniform(5, 9))
        times = ev.start + np.sort(rng.uniform(0, ev.duration, n_clicks))
        amps = rng.uniform(0.003, 0.02, n_clicks)
        out = np.zeros((len(t), self.dsp.channels))
        for k, (tc, amp) in enumerate(zip(times, amps)):
            sel = (t >= tc) & (t < tc + 0.004)
            if sel.any():
                local = t[sel] - tc
                burst_rng = np.random.default_rng(ev.params["seed"] + k + 1)
                full = burst_rng.normal(0, 1, int(0.004 * self.sr) + 2)
                idx = np.minimum((local * self.sr).astype(int), len(full) - 1)
                out[sel, :] += (amp * full[idx] * np.exp(-local / 0.001))[:, None]
        return out


# --------------------------------------------------------------------------- #
# Clip generation
# --------------------------------------------------------------------------- #
def generate_clip(
    gesture: str,
    dsp: DSPConfig,
    rng: np.random.Generator,
    idle_subtype: str | None = None,
    acoustics: SimAcoustics | None = None,
    volume: float = 0.3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render one ``dsp.clip_seconds`` clip ``(n, channels)`` containing ``gesture``."""
    scene = SimScene(dsp, acoustics, volume=volume, seed=int(rng.integers(1 << 31)))
    clip_s = dsp.clip_seconds
    probe = make_event(gesture, 0.0, rng, idle_subtype)
    if gesture == "idle":
        probe.start = rng.uniform(-0.5, 0.3)
        probe.duration = max(probe.duration, clip_s + 0.5)
    else:
        latest = max(0.15, clip_s - probe.duration - 0.15)
        probe.start = rng.uniform(0.15, latest) if latest > 0.15 else 0.15
    scene.add_event(probe)
    audio = scene.render(0, dsp.samples(clip_s))
    meta = {"simulated_gesture": gesture, "onset_s": round(probe.start, 3)}
    if probe.idle_subtype:
        meta["idle_subtype"] = probe.idle_subtype
    return audio, meta


def simulate_dataset(
    store: Any,
    gestures: list[str],
    clips_per_gesture: int = 20,
    sessions: int = 2,
    rounds: int = 2,
    seed: int = 0,
    dsp: DSPConfig | None = None,
    volume: float = 0.3,
) -> int:
    """Write a simulated dataset into ``store`` (which must be a simulated store under sim/).

    ``clips_per_gesture`` clips are recorded per gesture *per round*. Returns the
    number of clips written.
    """
    from dsp import quality_check
    from profiles import SimulatedDataError

    if not getattr(store, "simulated", False):
        raise SimulatedDataError("simulate_dataset() only writes to simulated (sim/) profiles")
    dsp = dsp or DSPConfig()
    store.ensure(dsp, volume, {"input": "simulated mic", "output": "simulated speaker"})
    dsp = store.dsp_config()
    rng = np.random.default_rng(seed)
    acoustics = SimAcoustics()
    written = 0
    for _ in range(sessions):
        sid = store.new_session()
        for _r in range(rounds):
            rnd = store.start_round(sid)
            for g in gestures:
                for i in range(clips_per_gesture):
                    sub = IDLE_SUBTYPE_NAMES[i % len(IDLE_SUBTYPE_NAMES)] if g == "idle" else None
                    audio, meta = generate_clip(g, dsp, rng, sub, acoustics, volume)
                    q = quality_check(audio, dsp, expects_motion=(g != "idle"))
                    store.add_clip(audio, g, sid, rnd, SOURCE_SIMULATED, q.to_dict(), meta)
                    written += 1
    logger.info("Wrote %d simulated clips to %s", written, store.root)
    return written


# --------------------------------------------------------------------------- #
# Simulated audio engine (same interface as audio_stream.AudioEngine)
# --------------------------------------------------------------------------- #
class SimulatedAudioEngine:
    """Drop-in replacement for :class:`audio_stream.AudioEngine` backed by :class:`SimScene`.

    ``realtime=True`` renders blocks on a background thread at wall-clock pace (for the
    dashboard and live demo). ``realtime=False`` renders on demand (``record`` /
    ``advance``), which makes tests fast and deterministic.
    """

    simulated = True

    def __init__(
        self,
        dsp: DSPConfig,
        audio: AudioConfig | None = None,
        realtime: bool = True,
        seed: int = 0,
        scenario: str = "normal",
        auto_demo: bool = False,
        demo_gestures: tuple[str, ...] = ("push", "pull", "wave"),
        noise: bool = True,
    ) -> None:
        self._cfg = audio or AudioConfig()
        self.sample_rate = dsp.sample_rate
        self.channels = dsp.channels
        self.realtime = realtime
        self.scene = SimScene(dsp, volume=self._cfg.volume, seed=seed, scenario=scenario, noise=noise)
        self._ring = RingBuffer(int(self._cfg.buffer_seconds * dsp.sample_rate), dsp.channels)
        self._rng = np.random.default_rng(seed + 7)
        self._cursor = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._new_data = threading.Condition()
        self._render_lock = threading.Lock()
        self.auto_demo = auto_demo
        self.demo_gestures = demo_gestures
        self._next_demo = 1.5
        self.input_name = "simulated mic"
        self.output_name = "simulated speaker"
        self.performed: list[tuple[float, str]] = []

    # -- lifecycle ------------------------------------------------------------- #
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop.clear()
        if self.realtime:
            self._thread = threading.Thread(target=self._run, name="sim-audio", daemon=True)
            self._thread.start()
        logger.warning(SIMULATED_BANNER)

    def stop(self) -> None:
        self._running = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> SimulatedAudioEngine:
        self.start()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        return self._running

    def _run(self) -> None:
        block = self._cfg.blocksize
        period = block / self.sample_rate
        next_t = time.monotonic()
        while not self._stop.is_set():
            self._render_block(block)
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            elif delay < -1.0:  # fell far behind (e.g. laptop suspended): resync
                next_t = time.monotonic()

    def _render_block(self, n: int) -> None:
        with self._render_lock:
            now = self._cursor / self.sample_rate
            if self.auto_demo and now >= self._next_demo:
                g = str(self._rng.choice(self.demo_gestures))
                self._schedule(g, now + 0.2)
                self._next_demo = now + self._rng.uniform(2.5, 4.5)
            data = self.scene.render(self._cursor, n)
            self._cursor += n
            if self._cursor % (self.sample_rate * 10) < n:
                self.scene.prune(now - 5.0)
        self._ring.write(data)
        with self._new_data:
            self._new_data.notify_all()

    # -- controls ------------------------------------------------------------- #
    @property
    def carrier_hz(self) -> float:
        return self.scene.frequency

    def set_frequency(self, frequency: float) -> None:
        self.scene.frequency = float(frequency)

    def set_volume(self, volume: float) -> None:
        self.scene.volume = float(volume)

    def set_tone_enabled(self, enabled: bool) -> None:
        self.scene.tone_on = enabled

    @property
    def now(self) -> float:
        """Current simulated time in seconds."""
        return self._cursor / self.sample_rate

    def _schedule(self, gesture: str, at: float, idle_subtype: str | None = None) -> SimEvent:
        ev = make_event(gesture, at, self._rng, idle_subtype)
        self.scene.add_event(ev)
        self.performed.append((at, gesture))
        return ev

    def perform(self, gesture: str, delay: float = 0.3, idle_subtype: str | None = None) -> SimEvent:
        """Schedule a simulated gesture ``delay`` seconds from now."""
        return self._schedule(gesture, self.now + delay, idle_subtype)

    # -- data access ------------------------------------------------------------- #
    @property
    def samples_written(self) -> int:
        return self._ring.total_written

    def advance(self, seconds: float) -> None:
        """Non-realtime mode: render ``seconds`` of audio into the ring buffer now."""
        n = int(round(seconds * self.sample_rate))
        block = self._cfg.blocksize
        while n > 0:
            k = min(block, n)
            self._render_block(k)
            n -= k

    def get_latest(self, seconds: float) -> np.ndarray:
        return self._ring.read_latest(int(round(seconds * self.sample_rate)))

    def record(self, seconds: float, timeout: float | None = None) -> np.ndarray:
        if not self._running:
            raise RuntimeError("Simulated engine is not running; call start() first.")
        n = int(round(seconds * self.sample_rate))
        start = self._ring.total_written
        if not self.realtime:
            self.advance(seconds)
        else:
            deadline = time.monotonic() + (timeout if timeout is not None else seconds * 2 + 2.0)
            with self._new_data:
                while self._ring.total_written - start < n:
                    if time.monotonic() > deadline:
                        raise TimeoutError("simulated audio stalled")
                    self._new_data.wait(0.1)
        data, _ = self._ring.read_since(start)
        return data[:n]

    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "simulated": True,
            "input": self.input_name,
            "output": self.output_name,
            "sample_rate": self.sample_rate,
            "carrier_hz": self.carrier_hz,
            "tone_on": self.scene.tone_on,
            "scenario": self.scene.scenario,
            "input_overflows": 0,
            "output_underflows": 0,
            "callback_errors": 0,
            "last_error": None,
        }


def main(argv: list[str] | None = None) -> int:
    from config import enabled_gestures, load_gestures
    from profiles import ProfileStore

    parser = argparse.ArgumentParser(description="Generate a SIMULATED demo dataset in sim/<profile>")
    parser.add_argument("--profile", default="demo")
    parser.add_argument("--clips", type=int, default=20, help="clips per gesture per round")
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gestures-file", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(f"*** {SIMULATED_BANNER} ***")
    gestures = [g.name for g in enabled_gestures(load_gestures(args.gestures_file))]
    store = ProfileStore(args.profile, simulated=True)
    n = simulate_dataset(store, gestures, args.clips, args.sessions, args.rounds, args.seed)
    print(f"Wrote {n} simulated clips to {store.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
