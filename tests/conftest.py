"""Shared fixtures. No test needs audio hardware: sounddevice is replaced by a fake."""

from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

from config import DSPConfig

FAKE_DEVICES = [
    {"name": "Fake Mic Array", "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 44100.0},
    {"name": "Fake Speakers", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 44100.0},
    {"name": "Bluetooth Headset", "max_input_channels": 1, "max_output_channels": 2, "default_samplerate": 16000.0},
]


class FakeStatus:
    def __init__(self, input_overflow: bool = False, output_underflow: bool = False) -> None:
        self.input_overflow = input_overflow
        self.output_underflow = output_underflow

    def __bool__(self) -> bool:
        return self.input_overflow or self.output_underflow


class FakeStream:
    """Loops the played tone back into the 'mic' on a thread, faster than real time."""

    xrun_every = 5
    instances: list[FakeStream] = []

    def __init__(self, samplerate, blocksize, device, channels, dtype, latency, callback) -> None:
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.ch_in, self.ch_out = channels
        self.callback = callback
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.blocks = 0
        self.closed = False
        FakeStream.instances.append(self)

    def _run(self) -> None:
        rng = np.random.default_rng(0)
        prev = np.zeros((self.blocksize, self.ch_out), np.float32)
        while not self._stop.is_set():
            indata = np.repeat(0.5 * prev[:, :1], self.ch_in, axis=1) + rng.normal(
                0, 1e-4, (self.blocksize, self.ch_in)
            )
            out = np.zeros((self.blocksize, self.ch_out), np.float32)
            status = FakeStatus(input_overflow=self.blocks % self.xrun_every == self.xrun_every - 1)
            self.callback(indata.astype(np.float32), out, self.blocksize, None, status)
            prev = out
            self.blocks += 1
            time.sleep(0.001)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(2)

    def close(self) -> None:
        self.closed = True


def make_fake_sd(devices: list[dict] | None = None) -> types.ModuleType:
    devs = FAKE_DEVICES if devices is None else devices
    mod = types.ModuleType("sounddevice")

    def query_devices(device=None, kind=None):
        if device is None and kind is None:
            return list(devs)
        if device is None:
            key = "max_input_channels" if kind == "input" else "max_output_channels"
            for d in devs:
                if d[key] > 0:
                    return d
            raise ValueError("no default device")
        return devs[device]

    mod.query_devices = query_devices
    mod.default = types.SimpleNamespace(device=(0, 1))
    mod.Stream = FakeStream
    return mod


@pytest.fixture
def fake_sd(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    mod = make_fake_sd()
    monkeypatch.setitem(sys.modules, "sounddevice", mod)
    FakeStream.instances.clear()
    return mod


@pytest.fixture
def dsp_cfg() -> DSPConfig:
    return DSPConfig()


@pytest.fixture
def isolated_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point default profiles/ and sim/ directories at a temp dir."""
    import profiles

    monkeypatch.setattr(profiles, "REPO_ROOT", tmp_path)
    return tmp_path


def tone(
    dsp: DSPConfig, seconds: float, freq: float | None = None, amp: float = 0.2, noise: float = 1e-3, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(dsp.samples(seconds)) / dsp.sample_rate
    x = amp * np.sin(2 * np.pi * (freq or dsp.carrier_hz) * t) + rng.normal(0, noise, t.size)
    return x.astype(np.float32)
