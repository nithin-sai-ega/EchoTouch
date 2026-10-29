import json
from pathlib import Path

import numpy as np
import pytest

from config import DSPConfig
from profiles import ProfileError, ProfileStore, SimulatedDataError, list_profiles


def make_store(tmp_path: Path, simulated: bool = False) -> ProfileStore:
    s = ProfileStore("p1", simulated=simulated, base_dir=tmp_path / ("sim" if simulated else "profiles"))
    s.create(DSPConfig(), 0.3, {"input": "mic", "output": "spk"})
    return s


def test_roundtrip_and_manifest(tmp_path: Path):
    s = make_store(tmp_path)
    sid = s.new_session()
    rnd = s.start_round(sid)
    audio = np.random.default_rng(0).normal(0, 0.1, (66150, 1)).astype(np.float32)
    rec = s.add_clip(audio, "push", sid, rnd, "real", {"snr_db": 40})
    assert rec["gesture"] == "push" and rec["round"] == 1 and rec["source"] == "real"
    loaded = s.load_clip(rec)
    np.testing.assert_array_equal(loaded[0], audio[:, 0])  # float32 WAV is lossless
    m = json.loads(s.manifest_path.read_text())
    assert m["mode"] == "real" and m["clips"][0]["session_id"] == sid
    assert m["dsp"]["carrier_hz"] == DSPConfig().carrier_hz
    assert s.counts()["push"] == 1
    assert list_profiles(base_dir=tmp_path / "profiles") == ["p1"]
    # A fresh store instance sees the same data (resumable).
    again = ProfileStore("p1", base_dir=tmp_path / "profiles")
    assert again.counts()["push"] == 1


def test_undo_moves_to_trash(tmp_path: Path):
    s = make_store(tmp_path)
    sid = s.new_session()
    r1 = s.add_clip(np.zeros(1000, np.float32), "push", sid, 1, "real")
    r2 = s.add_clip(np.zeros(1000, np.float32), "pull", sid, 1, "real")
    undone = s.undo_last(sid)
    assert undone["filename"] == r2["filename"]
    assert not (s.root / r2["filename"]).exists()
    assert (s.root / "_trash" / r2["filename"]).exists()
    assert [c["filename"] for c in s.clips()] == [r1["filename"]]
    assert s.undo_last("other-session") is None
    # New clips never reuse a filename after undo.
    r3 = s.add_clip(np.zeros(1000, np.float32), "pull", sid, 1, "real")
    assert r3["filename"] != r2["filename"]


def test_guardrails(tmp_path: Path):
    with pytest.raises(SimulatedDataError):
        ProfileStore("x", simulated=True, base_dir=tmp_path / "profiles")
    with pytest.raises(SimulatedDataError):
        ProfileStore("x", simulated=False, base_dir=tmp_path / "sim")
    real = make_store(tmp_path)
    with pytest.raises(SimulatedDataError):
        real.add_clip(np.zeros(100, np.float32), "push", "s", 1, "simulated")
    sim = make_store(tmp_path, simulated=True)
    with pytest.raises(SimulatedDataError):
        sim.add_clip(np.zeros(100, np.float32), "push", "s", 1, "real")


def test_manifest_mode_mismatch(tmp_path: Path):
    s = make_store(tmp_path)
    m = json.loads(s.manifest_path.read_text())
    m["mode"] = "simulated"
    s.manifest_path.write_text(json.dumps(m))
    with pytest.raises(SimulatedDataError):
        _ = ProfileStore("p1", base_dir=tmp_path / "profiles").manifest


def test_calibration_change_needs_force(tmp_path: Path):
    s = make_store(tmp_path)
    s.set_calibration({"carrier_hz": 18500.0}, DSPConfig(), 0.3)  # no clips yet: fine
    s.add_clip(np.zeros(100, np.float32), "push", "s", 1, "real")
    new = DSPConfig(carrier_hz=19500.0)
    with pytest.raises(ProfileError, match="invalidates"):
        s.set_calibration({"carrier_hz": 19500.0}, new, 0.3)
    with pytest.raises(ProfileError):
        s.set_calibration({"carrier_hz": 18500.0}, DSPConfig(), 0.6)  # volume change too
    warnings = s.set_calibration({"carrier_hz": 19500.0}, new, 0.3, force=True)
    assert warnings and s.dsp_config().carrier_hz == 19500.0


def test_missing_profile(tmp_path: Path):
    s = ProfileStore("nope", base_dir=tmp_path / "profiles")
    assert not s.exists()
    with pytest.raises(ProfileError, match="not found"):
        _ = s.manifest


def test_wrong_channel_count(tmp_path: Path):
    s = make_store(tmp_path)
    with pytest.raises(ProfileError, match="channels"):
        s.add_clip(np.zeros((100, 2), np.float32), "push", "s", 1, "real")
