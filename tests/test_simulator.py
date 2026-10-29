from pathlib import Path

import numpy as np
import pytest

from config import SOURCE_SIMULATED, DSPConfig
from dsp import snr_at_frequency
from profiles import ProfileStore, SimulatedDataError
from simulator import SimScene, SimulatedAudioEngine, generate_clip, make_event, simulate_dataset


def test_generate_clip_deterministic(dsp_cfg: DSPConfig):
    a, ma = generate_clip("push", dsp_cfg, np.random.default_rng(3))
    b, mb = generate_clip("push", dsp_cfg, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)
    assert ma == mb
    assert a.shape == (dsp_cfg.samples(dsp_cfg.clip_seconds), 1)
    assert a.dtype == np.float32


@pytest.mark.parametrize(
    "g", ["idle", "push", "pull", "wave", "swipe_left", "swipe_right", "swipe_up", "swipe_down", "my_custom"]
)
def test_all_gestures_render(dsp_cfg: DSPConfig, g: str):
    audio, meta = generate_clip(g, dsp_cfg, np.random.default_rng(0))
    assert np.isfinite(audio).all() and np.abs(audio).max() < 1.0


def test_render_is_continuous_across_blocks(dsp_cfg: DSPConfig):
    scene = SimScene(dsp_cfg, noise=False)
    scene.add_event(make_event("wave", 0.2, np.random.default_rng(0)))
    whole = scene.render(0, 30000)
    parts = np.concatenate([scene.render(0, 12345), scene.render(12345, 30000 - 12345)])
    np.testing.assert_allclose(whole, parts, atol=1e-6)


def test_scenarios(dsp_cfg: DSPConfig):
    def snr(scenario: str) -> float:
        eng = SimulatedAudioEngine(dsp_cfg, realtime=False, scenario=scenario)
        with eng:
            return snr_at_frequency(eng.record(1.0), dsp_cfg.sample_rate, dsp_cfg.carrier_hz)

    assert snr("normal") > 40
    assert snr("no_tone") < 15


def test_engine_interface(dsp_cfg: DSPConfig):
    eng = SimulatedAudioEngine(dsp_cfg, realtime=False)
    with pytest.raises(RuntimeError):
        eng.record(0.1)
    with eng:
        eng.perform("push", delay=0.1)
        clip = eng.record(1.0)
        assert clip.shape == (dsp_cfg.sample_rate, 1)
        assert eng.samples_written == dsp_cfg.sample_rate
        eng.set_frequency(19000)
        assert eng.carrier_hz == 19000
        assert eng.status()["simulated"] is True
        assert eng.get_latest(0.5).shape[0] == dsp_cfg.samples(0.5)


def test_realtime_engine_produces_audio(dsp_cfg: DSPConfig):
    with SimulatedAudioEngine(dsp_cfg, realtime=True) as eng:
        data = eng.record(0.2, timeout=5)
    assert data.shape[0] == dsp_cfg.samples(0.2)


def test_simulate_dataset_only_in_sim(tmp_path: Path, dsp_cfg: DSPConfig):
    store = ProfileStore("demo", simulated=True, base_dir=tmp_path / "sim")
    n = simulate_dataset(store, ["idle", "push"], clips_per_gesture=2, sessions=1, rounds=1)
    assert n == 4
    assert all(r["source"] == SOURCE_SIMULATED for r in store.clips())
    assert store.root.parent.name == "sim"
    real = ProfileStore("demo", simulated=False, base_dir=tmp_path / "profiles")
    with pytest.raises(SimulatedDataError):
        simulate_dataset(real, ["idle", "push"], clips_per_gesture=1)
    assert not (tmp_path / "profiles").exists()
