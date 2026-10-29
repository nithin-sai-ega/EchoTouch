"""End-to-end on CPU: simulate -> enroll-style dataset -> train -> live inference on a simulated stream."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import train
from actions import load_actions
from config import InferenceConfig, TrainConfig, load_gestures
from engine import EngineError, LiveSession, TrainingJob
from profiles import ProfileStore
from simulator import SimulatedAudioEngine, simulate_dataset

CORE = ["idle", "push", "pull", "wave"]


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("repo")
    store = ProfileStore("e2e", simulated=True, base_dir=root / "sim")
    simulate_dataset(store, CORE, clips_per_gesture=10, sessions=2, rounds=2, seed=1)
    specs = [s for s in load_gestures() if s.name in CORE]
    cfg = replace(TrainConfig(), epochs=12, patience=100, width=16, embed_dim=64, seed=0)
    events: list[dict] = []
    result = train.train_profile(store, specs, cfg, allow_simulated=True, callback=events.append)
    return root, store, result, events


@pytest.mark.slow
def test_training_outputs(trained):
    root, store, result, events = trained
    assert result.split.strategy == "session"
    assert store.model_path.exists() and store.report_path.exists()
    assert result.confusion_path is not None and result.confusion_path.exists()
    report = json.loads(store.report_path.read_text())
    assert report["source"] == "simulated"
    assert report["metrics"]["accuracy"] >= 0.8, train.format_report(report)
    assert set(report["metrics"]["per_class"]) == set(CORE)
    assert report["gate_db"] > 0
    assert any(e["type"] == "epoch" for e in events) and events[-1]["type"] == "done"
    assert len(result.history) == 12


@pytest.mark.slow
def test_live_inference_on_simulated_stream(trained, monkeypatch: pytest.MonkeyPatch):
    import profiles

    root, store, _, _ = trained
    monkeypatch.setattr(profiles, "REPO_ROOT", root)
    dsp = store.dsp_config()
    eng = SimulatedAudioEngine(dsp, realtime=False, seed=11)
    session = LiveSession.open("e2e", simulate=True, actions=load_actions(), engine=eng, cfg=InferenceConfig())
    assert session.worker.executor.force_dry_run  # simulated models can never drive the OS
    rng = np.random.default_rng(0)
    plan = ["push", "pull", "wave"] * 3 + ["idle"] * 3
    rng.shuffle(plan)
    outcomes = []
    with eng:
        eng.advance(1.2)
        for g in plan:
            before = len(session.worker.snapshot().events)
            eng.perform(g, delay=0.2, idle_subtype="silence" if g == "idle" else None)
            for _ in range(30):
                eng.advance(0.1)
                session.worker.process_once(now=eng.now)
            fired = [e.gesture for e in session.worker.snapshot().events[before:]]
            outcomes.append((g, fired))
    gestures = [(g, f) for g, f in outcomes if g != "idle"]
    hits = sum(1 for g, f in gestures if f == [g])
    wrong = [(g, f) for g, f in outcomes if any(x != g for x in f)]
    assert hits >= 6, outcomes  # most gestures recognized
    assert len(wrong) <= 1, outcomes  # (almost) no false or wrong triggers
    assert all(not f for g, f in outcomes if g == "idle" and (g, f) not in wrong)


@pytest.mark.slow
def test_simulated_model_refused_for_real_mode(trained, monkeypatch: pytest.MonkeyPatch):
    import shutil

    import profiles

    root, store, _, _ = trained
    monkeypatch.setattr(profiles, "REPO_ROOT", root)
    # Even if someone copies a simulated model into a real profile, live control refuses it.
    real = ProfileStore("e2e", simulated=False, base_dir=root / "profiles")
    real.create(store.dsp_config(), 0.3)
    real.set_calibration({"carrier_hz": store.dsp_config().carrier_hz}, store.dsp_config(), 0.3)
    shutil.copy(store.model_path, real.model_path)
    with pytest.raises(EngineError, match="SIMULATED"):
        LiveSession.open("e2e", simulate=False, actions=load_actions())
    # And a model whose DSP settings differ from the current calibration is refused clearly.
    shutil.copytree(store.root, root / "sim" / "e2e_moved")
    moved = ProfileStore("e2e_moved", simulated=True, base_dir=root / "sim")
    new_dsp = store.dsp_config().with_overrides(carrier_hz=19000.0)
    moved.set_calibration({"carrier_hz": 19000.0}, new_dsp, 0.3, force=True)
    with pytest.raises(EngineError, match="carrier_hz"):
        LiveSession.open(
            "e2e_moved", simulate=True, actions=load_actions(), engine=SimulatedAudioEngine(new_dsp, realtime=False)
        )


@pytest.mark.slow
def test_training_job_background(tmp_path: Path):
    store = ProfileStore("job", simulated=True, base_dir=tmp_path / "sim")
    simulate_dataset(store, ["idle", "push"], clips_per_gesture=4, sessions=2, rounds=1, seed=3)
    specs = [s for s in load_gestures() if s.name in ("idle", "push")]
    job = TrainingJob()
    job.start(store, specs, replace(TrainConfig(), epochs=2, width=8, embed_dim=16), allow_simulated=True)
    job.join(120)
    p = job.progress()
    assert p["status"] == "done", p["error"]
    assert [e for e in p["events"] if e["type"] == "epoch"]
    # Without allow_simulated the job fails with a clear error instead of training.
    job2 = TrainingJob()
    job2.start(store, specs, replace(TrainConfig(), epochs=1), allow_simulated=False)
    job2.join(60)
    assert job2.progress()["status"] == "error" and "allow-simulated" in job2.progress()["error"]


def test_train_cli_requires_allow_simulated(isolated_repo, capsys: pytest.CaptureFixture[str]):
    assert train.main(["--simulate", "--profile", "x"]) == 2
    assert "allow-simulated" in capsys.readouterr().out


def test_gate_threshold_suggestion():
    th, note = train.suggest_gate_threshold([0.5, 1.0, 2.0], [7.0, 8.0, 10.0], 6.0)
    assert 2.0 < th < 7.0 and "learned" in note
    th, note = train.suggest_gate_threshold([5.0, 9.0], [4.0, 6.0], 6.0)
    assert "overlapping" in note
    assert train.suggest_gate_threshold([], [1.0], 6.0)[0] == 6.0


def test_class_weights():
    w = train.class_weights([0, 0, 0, 1], 3)
    assert w[1] > w[0] and w[2] == 0
