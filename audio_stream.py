"""Full-duplex audio engine: play a continuous carrier tone while recording the mic.

The engine is thread-safe and never raises from inside the PortAudio callback:
xruns are counted and reported, not fatal. ``sounddevice`` is imported lazily so
the rest of EchoTouch (and the test-suite) works on machines without PortAudio.

Run ``python audio_stream.py --list-devices`` to see available devices.
"""

from __future__ import annotations

import argparse
import logging
import math
import threading
import time
from types import ModuleType, TracebackType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from config import AudioConfig, DSPConfig

logger = logging.getLogger(__name__)


class AudioDeviceError(RuntimeError):
    """Raised with a human-readable explanation when audio devices are unusable."""


def import_sounddevice() -> ModuleType:
    """Import ``sounddevice`` lazily with a helpful error message."""
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio shared library missing
        raise AudioDeviceError(
            "PortAudio could not be loaded, so no audio devices are available.\n"
            "  - Linux: sudo apt install libportaudio2\n"
            "  - Windows/macOS: pip install --force-reinstall sounddevice\n"
            f"  (details: {exc})"
        ) from exc
    except ImportError as exc:
        raise AudioDeviceError("sounddevice is not installed: pip install sounddevice") from exc
    return sd


# --------------------------------------------------------------------------- #
# Ring buffer
# --------------------------------------------------------------------------- #
class RingBuffer:
    """Fixed-capacity, lock-protected multi-channel ring buffer of float32 samples."""

    def __init__(self, capacity: int, channels: int = 1) -> None:
        if capacity <= 0 or channels <= 0:
            raise ValueError("capacity and channels must be positive")
        self._buf = np.zeros((capacity, channels), dtype=np.float32)
        self._capacity = capacity
        self._channels = channels
        self._write_pos = 0
        self._total = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def total_written(self) -> int:
        """Total samples (frames) ever written; monotonically increasing."""
        with self._lock:
            return self._total

    @property
    def available(self) -> int:
        with self._lock:
            return min(self._total, self._capacity)

    def write(self, block: np.ndarray) -> None:
        """Append ``(n, channels)`` or ``(n,)`` samples, overwriting the oldest data."""
        data = np.asarray(block, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        if data.shape[1] != self._channels:
            raise ValueError(f"expected {self._channels} channels, got {data.shape[1]}")
        n = data.shape[0]
        if n == 0:
            return
        with self._lock:
            if n >= self._capacity:
                self._buf[:] = data[-self._capacity :]
                self._write_pos = 0
            else:
                end = self._write_pos + n
                if end <= self._capacity:
                    self._buf[self._write_pos : end] = data
                else:
                    first = self._capacity - self._write_pos
                    self._buf[self._write_pos :] = data[:first]
                    self._buf[: n - first] = data[first:]
                self._write_pos = end % self._capacity
            self._total += n

    def read_latest(self, n: int) -> np.ndarray:
        """Return a chronological copy of the newest ``min(n, available)`` samples ``(n, C)``."""
        with self._lock:
            n = min(n, self._total, self._capacity)
            start = (self._write_pos - n) % self._capacity
            if start + n <= self._capacity:
                return self._buf[start : start + n].copy()
            return np.concatenate([self._buf[start:], self._buf[: (start + n) - self._capacity]])

    def read_since(self, total_index: int) -> tuple[np.ndarray, int]:
        """Return samples written since absolute index ``total_index`` and the new index.

        If the reader fell behind by more than the capacity, the oldest samples are lost
        and only the retained ones are returned.
        """
        with self._lock:
            total = self._total
        missing = total - total_index
        return (self.read_latest(missing) if missing > 0 else np.zeros((0, self._channels), np.float32)), total

    def clear(self) -> None:
        with self._lock:
            self._buf[:] = 0.0
            self._write_pos = 0
            self._total = 0


# --------------------------------------------------------------------------- #
# Tone generator
# --------------------------------------------------------------------------- #
class ToneGenerator:
    """Phase-continuous sine generator with linear fade-in/out (no clicks).

    Frequency and volume changes are glitch-free: phase is accumulated, and the
    amplitude ramps towards its target over ``fade_seconds``.
    """

    def __init__(self, sample_rate: int, frequency: float, volume: float, fade_seconds: float = 0.05) -> None:
        self.sample_rate = sample_rate
        self._freq = float(frequency)
        self._phase = 0.0
        self._amp = 0.0
        self._target = float(volume)
        self._volume = float(volume)
        self._step = 1.0 / max(1, int(fade_seconds * sample_rate))
        self._lock = threading.Lock()

    @property
    def frequency(self) -> float:
        return self._freq

    @property
    def amplitude(self) -> float:
        return self._amp

    def set_frequency(self, frequency: float) -> None:
        with self._lock:
            self._freq = float(frequency)

    def set_volume(self, volume: float) -> None:
        with self._lock:
            self._volume = float(np.clip(volume, 0.0, 1.0))
            if self._target > 0:
                self._target = self._volume

    def fade_in(self) -> None:
        with self._lock:
            self._target = self._volume

    def fade_out(self) -> None:
        with self._lock:
            self._target = 0.0

    @property
    def silent(self) -> bool:
        return self._amp == 0.0 and self._target == 0.0

    def next_block(self, n: int) -> np.ndarray:
        """Generate the next ``n`` samples (float32, mono)."""
        with self._lock:
            freq, target = self._freq, self._target
            inc = 2.0 * math.pi * freq / self.sample_rate
            phases = self._phase + inc * np.arange(n, dtype=np.float64)
            self._phase = float((self._phase + inc * n) % (2.0 * math.pi))
            direction = np.sign(target - self._amp)
            ramp = self._amp + direction * self._step * np.arange(1, n + 1)
            ramp = np.minimum(ramp, target) if direction > 0 else np.maximum(ramp, target)
            self._amp = float(ramp[-1]) if n else self._amp
        return (ramp * np.sin(phases)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Device helpers
# --------------------------------------------------------------------------- #
def list_devices_text(sd: ModuleType | None = None) -> str:
    """Return a human-readable table of audio devices."""
    sd = sd or import_sounddevice()
    devices = sd.query_devices()
    try:
        default_in, default_out = sd.default.device
    except Exception:  # pragma: no cover - defensive
        default_in = default_out = -1
    lines = ["idx  in  out  default  name"]
    for idx, dev in enumerate(devices):
        mark = ("I" if idx == default_in else " ") + ("O" if idx == default_out else " ")
        lines.append(
            f"{idx:>3}  {dev['max_input_channels']:>2}  {dev['max_output_channels']:>3}  "
            f"{mark:^7}  {dev['name']}  ({dev.get('default_samplerate', 0):.0f} Hz)"
        )
    return "\n".join(lines)


def resolve_device(spec: int | str | None, kind: str, sd: ModuleType | None = None) -> int | None:
    """Resolve a device given as ``None`` (system default), index, or name substring.

    ``kind`` is ``"input"`` or ``"output"``. Raises :class:`AudioDeviceError` listing
    the available devices if nothing matches.
    """
    if kind not in ("input", "output"):
        raise ValueError("kind must be 'input' or 'output'")
    sd = sd or import_sounddevice()
    devices = sd.query_devices()
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    if not any(d[key] > 0 for d in devices):
        raise AudioDeviceError(
            f"No audio {kind} device found. Check that a microphone/speaker is connected and "
            "enabled in the OS sound settings (and that the app has microphone permission)."
        )
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return None
    if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().isdigit()):
        idx = int(spec)
        if not 0 <= idx < len(devices) or devices[idx][key] <= 0:
            raise AudioDeviceError(f"Device index {idx} is not a valid {kind} device.\n{list_devices_text(sd)}")
        return idx
    needle = spec.lower()
    matches = [i for i, d in enumerate(devices) if needle in d["name"].lower() and d[key] > 0]
    if not matches:
        raise AudioDeviceError(f"No {kind} device matching {spec!r}.\n{list_devices_text(sd)}")
    if len(matches) > 1:
        logger.info("Several %s devices match %r; using #%d (%s)", kind, spec, matches[0], devices[matches[0]]["name"])
    return matches[0]


def device_name(index: int | None, kind: str, sd: ModuleType | None = None) -> str:
    sd = sd or import_sounddevice()
    try:
        info = sd.query_devices(index, kind) if index is not None else sd.query_devices(kind=kind)
        return str(info["name"])
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
@runtime_checkable
class AudioSource(Protocol):
    """Interface shared by the real :class:`AudioEngine` and the simulated engine."""

    sample_rate: int
    channels: int
    simulated: bool

    def start(self) -> None: ...
    def stop(self) -> None: ...
    @property
    def running(self) -> bool: ...
    @property
    def carrier_hz(self) -> float: ...
    def set_frequency(self, frequency: float) -> None: ...
    def set_volume(self, volume: float) -> None: ...
    def set_tone_enabled(self, enabled: bool) -> None: ...
    def get_latest(self, seconds: float) -> np.ndarray: ...
    def record(self, seconds: float, timeout: float | None = None) -> np.ndarray: ...
    @property
    def samples_written(self) -> int: ...
    def status(self) -> dict[str, Any]: ...


class AudioEngine:
    """Thread-safe full-duplex engine (carrier out, mic in -> ring buffer).

    Use as a context manager::

        with AudioEngine(dsp_cfg) as eng:
            clip = eng.record(1.5)
    """

    simulated = False

    def __init__(
        self,
        dsp: DSPConfig,
        audio: AudioConfig | None = None,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        play_tone: bool = True,
    ) -> None:
        self._dsp = dsp
        self._cfg = audio or AudioConfig()
        self.sample_rate = dsp.sample_rate
        self.channels = dsp.channels
        self._input_spec = input_device
        self._output_spec = output_device
        self._tone = ToneGenerator(dsp.sample_rate, dsp.carrier_hz, self._cfg.volume, self._cfg.fade_seconds)
        self._tone_enabled = play_tone
        self._ring = RingBuffer(int(self._cfg.buffer_seconds * dsp.sample_rate), dsp.channels)
        self._stream: Any = None
        self._lock = threading.Lock()
        self._new_data = threading.Condition()
        self._input_overflows = 0
        self._output_underflows = 0
        self._callback_errors = 0
        self._last_error: str | None = None
        self._out_channels = 1
        self.input_name = "?"
        self.output_name = "?"

    # -- lifecycle ------------------------------------------------------------ #
    def start(self) -> None:
        with self._lock:
            if self._stream is not None:
                return
            sd = import_sounddevice()
            in_idx = resolve_device(self._input_spec, "input", sd)
            out_idx = resolve_device(self._output_spec, "output", sd)
            in_info = sd.query_devices(in_idx, "input") if in_idx is not None else sd.query_devices(kind="input")
            out_info = sd.query_devices(out_idx, "output") if out_idx is not None else sd.query_devices(kind="output")
            if in_info["max_input_channels"] < self.channels:
                raise AudioDeviceError(
                    f"Input device '{in_info['name']}' has {in_info['max_input_channels']} channel(s), "
                    f"but {self.channels} were requested."
                )
            self._out_channels = max(1, min(2, int(out_info["max_output_channels"])))
            self.input_name, self.output_name = str(in_info["name"]), str(out_info["name"])
            try:
                stream = sd.Stream(
                    samplerate=self.sample_rate,
                    blocksize=self._cfg.blocksize,
                    device=(in_idx, out_idx),
                    channels=(self.channels, self._out_channels),
                    dtype="float32",
                    latency=self._cfg.latency,
                    callback=self._callback,
                )
                if self._tone_enabled:
                    self._tone.fade_in()
                else:
                    self._tone.fade_out()
                stream.start()
            except Exception as exc:
                raise AudioDeviceError(
                    f"Could not open full-duplex stream at {self.sample_rate} Hz on "
                    f"in='{self.input_name}' / out='{self.output_name}': {exc}\n"
                    "Try another --sample-rate (44100/48000), other devices (--list-devices), "
                    "or close apps holding the device exclusively."
                ) from exc
            self._stream = stream
            logger.info("Audio started: in=%s out=%s sr=%d", self.input_name, self.output_name, self.sample_rate)

    def stop(self) -> None:
        """Fade the tone out, then close the stream. Safe to call repeatedly."""
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is None:
            return
        self._tone.fade_out()
        deadline = time.monotonic() + self._cfg.fade_seconds + 0.2
        while not self._tone.silent and time.monotonic() < deadline:
            time.sleep(0.01)
        try:
            stream.stop()
            stream.close()
        except Exception as exc:  # pragma: no cover - backend specific
            logger.warning("Error while closing audio stream: %s", exc)
        logger.info("Audio stopped")

    def __enter__(self) -> AudioEngine:
        self.start()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        return self._stream is not None

    # -- callback ------------------------------------------------------------- #
    def _callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, _time: Any, status: Any) -> None:
        """PortAudio callback: must never raise or block for long."""
        try:
            if status:
                if getattr(status, "input_overflow", False):
                    self._input_overflows += 1
                if getattr(status, "output_underflow", False):
                    self._output_underflows += 1
            block = self._tone.next_block(frames)
            outdata[:] = block[:, None]
            self._ring.write(indata)
            with self._new_data:
                self._new_data.notify_all()
        except Exception as exc:  # pragma: no cover - defensive
            self._callback_errors += 1
            self._last_error = repr(exc)
            outdata.fill(0)

    # -- controls --------------------------------------------------------------- #
    @property
    def carrier_hz(self) -> float:
        return self._tone.frequency

    def set_frequency(self, frequency: float) -> None:
        self._tone.set_frequency(frequency)

    def set_volume(self, volume: float) -> None:
        self._tone.set_volume(volume)

    def set_tone_enabled(self, enabled: bool) -> None:
        self._tone_enabled = enabled
        if enabled:
            self._tone.fade_in()
        else:
            self._tone.fade_out()

    # -- data access ------------------------------------------------------------ #
    @property
    def samples_written(self) -> int:
        return self._ring.total_written

    def get_latest(self, seconds: float) -> np.ndarray:
        """Newest ``seconds`` of audio as ``(n, channels)`` float32 (may be shorter at start-up)."""
        return self._ring.read_latest(int(round(seconds * self.sample_rate)))

    def record(self, seconds: float, timeout: float | None = None) -> np.ndarray:
        """Block until ``seconds`` of *new* audio arrived, then return it ``(n, channels)``."""
        if not self.running:
            raise AudioDeviceError("Audio engine is not running; call start() first.")
        n = int(round(seconds * self.sample_rate))
        if n > self._ring.capacity:
            raise ValueError("requested recording is longer than the ring buffer")
        start = self._ring.total_written
        deadline = time.monotonic() + (timeout if timeout is not None else seconds * 2 + 2.0)
        with self._new_data:
            while self._ring.total_written - start < n:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AudioDeviceError(
                        "Timed out waiting for microphone audio. The input stream stalled: "
                        "check the device is not in use or disconnected."
                    )
                self._new_data.wait(min(remaining, 0.1))
        data, _ = self._ring.read_since(start)
        return data[:n]

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "simulated": False,
            "input": self.input_name,
            "output": self.output_name,
            "sample_rate": self.sample_rate,
            "carrier_hz": self.carrier_hz,
            "tone_on": self._tone_enabled,
            "input_overflows": self._input_overflows,
            "output_underflows": self._output_underflows,
            "callback_errors": self._callback_errors,
            "last_error": self._last_error,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EchoTouch audio engine utilities")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    parser.add_argument("--test", type=float, metavar="SECONDS", help="play the carrier and record for N seconds")
    parser.add_argument("--carrier", type=float, default=DSPConfig().carrier_hz)
    parser.add_argument("--volume", type=float, default=AudioConfig().volume)
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--output-device", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        if args.list_devices or not args.test:
            print(list_devices_text())
            return 0
        from dsp import snr_at_frequency

        dsp_cfg = DSPConfig(carrier_hz=args.carrier)
        with AudioEngine(dsp_cfg, AudioConfig(volume=args.volume), args.input_device, args.output_device) as eng:
            data = eng.record(args.test)
            snr = snr_at_frequency(data, eng.sample_rate, args.carrier)
            print(f"Recorded {len(data)} samples; carrier SNR = {snr:.1f} dB")
            print(eng.status())
    except AudioDeviceError as exc:
        print(f"ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
