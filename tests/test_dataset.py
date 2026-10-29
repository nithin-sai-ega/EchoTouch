from pathlib import Path

import numpy as np
import pytest
import torch

from config import AugmentConfig, DSPConfig
from dataset import (
    DatasetError,
    GestureDataset,
    check_sources,
    group_key,
    load_profile_data,
    split_records,
)
from profiles import ProfileStore, SimulatedDataError
from simulator import simulate_dataset

CLASSES = ["idle", "push", "pull"]


def records(sessions: int, rounds: int, per: int = 5, classes=CLASSES, source="real"):
    out = []
    t = 0
    for s in range(sessions):
        for r in range(1, rounds + 1):
            for c in classes:
                for _ in range(per):
                    t += 1
                    out.append(
                        {
                            "gesture": c,
                            "session_id": f"s{s}",
                            "round": r,
                            "timestamp": f"2026-01-01T00:{t:05d}",
                            "source": source,
                            "filename": f"{t}.wav",
                        }
                    )
    return out


def assert_no_leakage(recs, tr, va, strategy):
    if strategy == "chronological":
        return
    tg = {group_key(recs[i], strategy) for i in tr}
    vg = {group_key(recs[i], strategy) for i in va}
    assert not tg & vg, f"groups leaked: {tg & vg}"
    # Also never share a (session, round) block, whatever the strategy.
    assert not {group_key(recs[i], "round") for i in tr} & {group_key(recs[i], "round") for i in va}


def test_session_split():
    recs = records(sessions=3, rounds=2)
    tr, va, info = split_records(recs, CLASSES, 0.2)
    assert info.strategy == "session"
    assert set(tr).isdisjoint(va) and len(tr) + len(va) == len(recs)
    assert_no_leakage(recs, tr, va, "session")
    assert {recs[i]["session_id"] for i in va} == {"s2"}  # newest session held out
    assert not info.warnings


def test_round_split_single_session_warns():
    recs = records(sessions=1, rounds=4)
    tr, va, info = split_records(recs, CLASSES, 0.2)
    assert info.strategy == "round"
    assert_no_leakage(recs, tr, va, "round")
    assert any("OPTIMISTIC" in w for w in info.warnings)
    assert {recs[i]["round"] for i in va} == {4}
    for c in CLASSES:  # every class in both splits
        assert any(recs[i]["gesture"] == c for i in tr) and any(recs[i]["gesture"] == c for i in va)


def test_chronological_fallback():
    recs = records(sessions=1, rounds=1, per=10)
    tr, va, info = split_records(recs, CLASSES, 0.2)
    assert info.strategy == "chronological" and info.warnings
    for c in CLASSES:  # validation is the LAST block of each class
        idx = [i for i, r in enumerate(recs) if r["gesture"] == c]
        assert set(idx[-2:]) <= set(va) and set(idx[:-2]) <= set(tr)


def test_partial_session_falls_back_to_rounds():
    # The second session only has 'push' clips: holding it out would leave others unvalidated.
    recs = records(sessions=1, rounds=3) + [
        {**r, "session_id": "s9", "timestamp": "2027" + r["timestamp"][4:]} for r in records(1, 1, classes=["push"])
    ]
    tr, va, info = split_records(recs, CLASSES, 0.2)
    assert info.strategy == "round"
    assert_no_leakage(recs, tr, va, "round")


def test_missing_class():
    with pytest.raises(DatasetError, match="pull"):
        split_records(records(2, 1, classes=["idle", "push"]), CLASSES)


def test_check_sources():
    assert check_sources(records(1, 1), allow_simulated=False) == "real"
    sim = records(1, 1, source="simulated")
    with pytest.raises(SimulatedDataError, match="allow-simulated"):
        check_sources(sim, allow_simulated=False)
    assert check_sources(sim, allow_simulated=True) == "simulated"
    with pytest.raises(SimulatedDataError, match="mixes"):
        check_sources(records(1, 1) + sim, allow_simulated=True)
    with pytest.raises(DatasetError):
        check_sources([], allow_simulated=True)


def test_load_and_dataset(tmp_path: Path):
    store = ProfileStore("d", simulated=True, base_dir=tmp_path / "sim")
    simulate_dataset(store, CLASSES, clips_per_gesture=2, sessions=1, rounds=2)
    with pytest.raises(SimulatedDataError):
        load_profile_data(store, CLASSES, allow_simulated=False)
    recs, stfts, source = load_profile_data(store, CLASSES, allow_simulated=True)
    dsp = DSPConfig()
    assert source == "simulated" and len(recs) == len(stfts) == 12
    assert stfts[0].shape == (1, dsp.freq_bins, dsp.clip_frames)
    labels = [CLASSES.index(r["gesture"]) for r in recs]
    val = GestureDataset(stfts, labels, dsp, None)
    x, y = val[0]
    assert isinstance(x, torch.Tensor) and x.shape == (1, dsp.freq_bins, dsp.window_frames)
    torch.testing.assert_close(val[0][0], x)  # deterministic validation view
    train = GestureDataset(stfts, labels, dsp, AugmentConfig(), seed=1)
    a, _ = train[0]
    train.set_epoch(1)
    b, _ = train[0]
    assert a.shape == b.shape and not torch.equal(a, b)  # new augmentation each epoch
    assert np.isfinite(a.numpy()).all()
