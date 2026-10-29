from pathlib import Path

import pytest

from config import (
    DEFAULT_GESTURES_FILE,
    ConfigError,
    ConfigMismatchError,
    DSPConfig,
    enabled_gestures,
    load_gestures,
    validate_profile_name,
)


def test_dsp_derived_values():
    cfg = DSPConfig()
    assert cfg.bin_hz == pytest.approx(44100 / 4096)
    assert cfg.freq_bins == 2 * cfg.half_bins + 1
    assert cfg.half_bins == round(300 / cfg.bin_hz)
    assert cfg.window_frames == (44100 - 4096) // 512 + 1
    assert cfg.clip_frames > cfg.window_frames
    freqs = cfg.relative_freqs()
    assert len(freqs) == cfg.freq_bins and freqs[cfg.half_bins] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 22050},
        {"n_fft": 3000},
        {"carrier_hz": 21900},
        {"window_seconds": 2.0},
        {"background": "bogus"},
        {"crop_hz": 40.0},
    ],
)
def test_dsp_invalid(kwargs):
    with pytest.raises(ConfigError):
        DSPConfig(**kwargs)


def test_dsp_roundtrip_and_mismatch():
    a = DSPConfig()
    assert DSPConfig.from_dict(a.to_dict()) == a
    b = a.with_overrides(carrier_hz=19000.0)
    assert any("carrier_hz" in d for d in a.diff(b))
    a.require_compatible(DSPConfig())
    with pytest.raises(ConfigMismatchError, match="carrier_hz"):
        a.require_compatible(b)


def test_48k_supported():
    cfg = DSPConfig(sample_rate=48000)
    assert cfg.freq_bins == 2 * round(300 / (48000 / 4096)) + 1


def test_repo_gestures_file_loads():
    specs = load_gestures(DEFAULT_GESTURES_FILE)
    names = [s.name for s in specs]
    assert names[:4] == ["idle", "push", "pull", "wave"]
    assert {s.name for s in specs if s.tier == "advanced"} == {"swipe_left", "swipe_right", "swipe_up", "swipe_down"}
    assert all(s.experimental for s in specs if s.tier == "advanced")
    assert "idle" in [s.name for s in enabled_gestures(specs)]


def test_custom_gestures(tmp_path: Path):
    p = tmp_path / "g.yaml"
    p.write_text(
        "gestures:\n"
        "  - {name: idle, motion: false}\n"
        "  - {name: double_tap, tier: custom}\n"
        "  - {name: push, enabled: false}\n"
    )
    specs = load_gestures(p)
    assert [s.name for s in enabled_gestures(specs)] == ["idle", "double_tap"]


@pytest.mark.parametrize(
    "body",
    [
        "gestures:\n  - {name: push}\n",  # no idle
        "gestures:\n  - {name: idle, motion: false}\n  - {name: Bad-Name}\n",
        "gestures:\n  - {name: idle, motion: false}\n  - {name: push}\n  - {name: push}\n",
        "gestures:\n  - {name: idle, motion: false}\n  - {name: x, tier: legendary}\n",
        "gestures: []\n",
    ],
)
def test_invalid_gesture_files(tmp_path: Path, body: str):
    p = tmp_path / "g.yaml"
    p.write_text(body)
    with pytest.raises(ConfigError):
        load_gestures(p)


def test_profile_names():
    assert validate_profile_name("my-laptop_1") == "my-laptop_1"
    for bad in ["", "../x", "a/b", "x" * 100]:
        with pytest.raises(ConfigError):
            validate_profile_name(bad)
