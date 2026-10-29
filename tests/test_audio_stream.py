import threading

import numpy as np
import pytest

from audio_stream import AudioDeviceError, AudioEngine, RingBuffer, ToneGenerator, list_devices_text, resolve_device
from config import AudioConfig, DSPConfig
from tests.conftest import make_fake_sd


def test_ring_buffer_order_and_wrap():
    rb = RingBuffer(10)
    rb.write(np.arange(7))
    rb.write(np.arange(7, 15))
    assert rb.total_written == 15 and rb.available == 10
    np.testing.assert_array_equal(rb.read_latest(4)[:, 0], [11, 12, 13, 14])
    np.testing.assert_array_equal(rb.read_latest(100)[:, 0], np.arange(5, 15))
    rb.write(np.arange(100, 130))  # larger than capacity
    np.testing.assert_array_equal(rb.read_latest(10)[:, 0], np.arange(120, 130))


def test_ring_buffer_read_since():
    rb = RingBuffer(8)
    rb.write(np.arange(5))
    idx = rb.total_written
    rb.write(np.arange(5, 8))
    data, new_idx = rb.read_since(idx)
    np.testing.assert_array_equal(data[:, 0], [5, 6, 7])
    assert new_idx == 8
    empty, _ = rb.read_since(new_idx)
    assert len(empty) == 0


def test_ring_buffer_channel_mismatch():
    rb = RingBuffer(8, channels=2)
    with pytest.raises(ValueError):
        rb.write(np.zeros((4, 3)))


def test_ring_buffer_thread_safety():
    """Concurrent writer/readers: every read is a contiguous, increasing run of samples."""
    rb = RingBuffer(4096)
    stop = threading.Event()
    errors: list[str] = []

    def writer() -> None:
        v = 0
        while not stop.is_set() and v < 2_000_000:
            n = 97
            rb.write(np.arange(v, v + n, dtype=np.float32))
            v += n

    def reader() -> None:
        while not stop.is_set():
            d = rb.read_latest(1000)[:, 0]
            if len(d) > 1 and not np.all(np.diff(d) == 1):
                errors.append("non-contiguous read")
                return

    threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    threads[0].join(10)
    stop.set()
    for t in threads[1:]:
        t.join(5)
    assert not errors
    assert rb.total_written > 100_000


def test_tone_generator_phase_continuity():
    sr, f = 44100, 18500.0
    a = ToneGenerator(sr, f, 0.5, fade_seconds=0.0)
    a._amp = 0.5  # skip fade for this test
    blocks = np.concatenate([a.next_block(n) for n in (100, 333, 1024, 7)])
    t = np.arange(len(blocks)) / sr
    np.testing.assert_allclose(blocks, 0.5 * np.sin(2 * np.pi * f * t), atol=1e-4)


def test_tone_generator_fades():
    g = ToneGenerator(44100, 1000, 0.4, fade_seconds=0.01)
    g.fade_in()
    first = g.next_block(441)
    assert abs(first[0]) == 0 and abs(first[:10]).max() <= 10 / 441 + 1e-6  # ramps up from silence: no click
    assert g.amplitude == pytest.approx(0.4)
    g.fade_out()
    g.next_block(441)
    assert g.silent
    assert np.abs(g.next_block(100)).max() == 0


def test_resolve_device(fake_sd):
    assert resolve_device(None, "input") is None
    assert resolve_device("mic", "input") == 0
    assert resolve_device("speakers", "output") == 1
    assert resolve_device(1, "output") == 1
    assert resolve_device("1", "output") == 1
    with pytest.raises(AudioDeviceError, match="No input device matching"):
        resolve_device("nonexistent", "input")
    with pytest.raises(AudioDeviceError):
        resolve_device(1, "input")  # speakers have no inputs
    assert "Fake Mic Array" in list_devices_text()


def test_no_devices(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "sounddevice", make_fake_sd([]))
    with pytest.raises(AudioDeviceError, match="No audio input device"):
        resolve_device(None, "input")


def test_engine_full_duplex(fake_sd):
    cfg = DSPConfig()
    eng = AudioEngine(cfg, AudioConfig(volume=0.3, blocksize=512))
    with eng:
        assert eng.running
        data = eng.record(0.5, timeout=10)
        assert data.shape == (cfg.samples(0.5), 1)
        st = eng.status()
        assert st["input"] == "Fake Mic Array" and st["output"] == "Fake Speakers"
        assert st["input_overflows"] >= 1  # xruns are counted, not fatal
        latest = eng.get_latest(0.1)
        assert latest.shape[1] == 1
        eng.set_frequency(19000.0)
        assert eng.carrier_hz == 19000.0
    assert not eng.running
    assert fake_sd.Stream.instances[-1].closed
    # The loopback must contain the carrier tone (played by the engine).
    from dsp import snr_at_frequency

    assert snr_at_frequency(data, cfg.sample_rate, cfg.carrier_hz) > 30
    eng.stop()  # idempotent


def test_engine_record_requires_start(fake_sd):
    eng = AudioEngine(DSPConfig())
    with pytest.raises(AudioDeviceError):
        eng.record(0.1)


def test_engine_too_many_channels(fake_sd):
    eng = AudioEngine(DSPConfig(channels=3))
    with pytest.raises(AudioDeviceError, match="channel"):
        eng.start()
