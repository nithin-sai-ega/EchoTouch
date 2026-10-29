from pathlib import Path

import numpy as np
import pytest

from config import DSPConfig, GestureSpec, load_gestures
from doctor import FAIL, PASS, WARN, CalibrationConfig, bluetooth_suspect, choose_carrier, run_doctor
from enroll import EnrollmentController
from profiles import ProfileStore, SimulatedDataError
from simulator import SimAcoustics, SimulatedAudioEngine

FAST_CAL = CalibrationConfig(settle_seconds=0.1, measure_seconds=0.4, stability_seconds=2.0)


def statuses(report) -> dict[str, str]:
    return {r.name: r.status for r in report.results}


def test_doctor_normal_picks_good_carrier(dsp_cfg: DSPConfig):
    with SimulatedAudioEngine(dsp_cfg, realtime=False) as eng:
        rep = run_doctor(eng, dsp_cfg, FAST_CAL)
    st = statuses(rep)
    assert st["carrier detection"] == PASS and st["carrier stability"] == PASS and st["clipping"] == PASS
    assert rep.overall in (PASS, WARN)
    assert len(rep.sweep) == len(FAST_CAL.candidate_carriers)
    # The simulator has a resonance notch at 19.5 kHz: it must not be chosen.
    assert abs(rep.carrier_hz - SimAcoustics().notch_hz) > 1
    snrs = dict(rep.sweep)
    assert snrs[rep.carrier_hz] >= max(snrs.values()) - FAST_CAL.prefer_high_within_db
    cal = rep.calibration()
    assert cal["carrier_hz"] == rep.carrier_hz and cal["sweep"]
    assert "OVERALL" in rep.format()


def test_doctor_detects_missing_tone(dsp_cfg: DSPConfig):
    with SimulatedAudioEngine(dsp_cfg, realtime=False, scenario="no_tone") as eng:
        rep = run_doctor(eng, dsp_cfg, FAST_CAL, sweep=False)
    assert statuses(rep)["carrier detection"] == FAIL and rep.overall == FAIL


def test_doctor_detects_noise_suppression(dsp_cfg: DSPConfig):
    with SimulatedAudioEngine(dsp_cfg, realtime=False, scenario="noise_suppression") as eng:
        rep = run_doctor(eng, dsp_cfg, FAST_CAL, sweep=False)
    assert statuses(rep)["carrier stability"] == WARN


def test_doctor_detects_clipping(dsp_cfg: DSPConfig):
    with SimulatedAudioEngine(dsp_cfg, realtime=False, scenario="clipping") as eng:
        rep = run_doctor(eng, dsp_cfg, FAST_CAL, sweep=False)
    assert statuses(rep)["clipping"] == FAIL


def test_choose_carrier_prefers_high():
    sweep = [(17000.0, 50.0), (18000.0, 49.0), (19000.0, 48.0), (20000.0, 30.0)]
    f, _ = choose_carrier(sweep, CalibrationConfig())
    assert f == 19000.0
    f, why = choose_carrier([(17000.0, 8.0), (18000.0, 5.0)], CalibrationConfig())
    assert f == 17000.0 and "weak" in why


def test_bluetooth_hint():
    assert bluetooth_suspect("Headset (WH-1000XM4 Hands-Free AG Audio)")
    assert not bluetooth_suspect("Microphone Array (Realtek(R) Audio)")


# --------------------------------------------------------------------------- #
def sim_store(tmp_path: Path) -> ProfileStore:
    store = ProfileStore("e", simulated=True, base_dir=tmp_path / "sim")
    store.create(DSPConfig(), 0.3)
    return store


def test_enrollment_controller(tmp_path: Path, dsp_cfg: DSPConfig):
    specs = [s for s in load_gestures() if s.name in ("idle", "push", "pull")]
    store = sim_store(tmp_path)
    with SimulatedAudioEngine(dsp_cfg, realtime=False) as eng:
        ctrl = EnrollmentController(store, eng, specs)
        sid = ctrl.begin_session()
        rnd, variation = ctrl.begin_round()
        assert rnd == 1 and variation
        a = ctrl.record("push")
        assert a.quality.accepted, a.quality.messages
        rec = ctrl.save(a)
        assert rec["session_id"] == sid and rec["round"] == 1 and rec["source"] == "simulated"
        b = ctrl.record("idle", "typing")
        rec_idle = ctrl.save(b)
        assert rec_idle["idle_subtype"] == "typing"
        assert ctrl.counts() == {"idle": 1, "push": 1, "pull": 0}
        assert ctrl.undo_last()["gesture"] == "idle"
        assert ctrl.counts()["idle"] == 0
        snr, motion = ctrl.meter(0.5)
        assert snr > 30 and np.isfinite(motion)
        rnd2, var2 = ctrl.begin_round()
        assert rnd2 == 2 and var2 != variation
    # Resumable: a new controller on the same profile sees existing clips and a new session id.
    with SimulatedAudioEngine(dsp_cfg, realtime=False) as eng:
        ctrl2 = EnrollmentController(ProfileStore("e", simulated=True, base_dir=tmp_path / "sim"), eng, specs)
        assert ctrl2.counts()["push"] == 1
        assert ctrl2.begin_session() != sid


def test_enrollment_rejects_still_hand_for_gestures(tmp_path: Path, dsp_cfg: DSPConfig):
    store = sim_store(tmp_path)
    specs = [GestureSpec("idle", "core", True, False), GestureSpec("push", "core", True, True)]
    with SimulatedAudioEngine(dsp_cfg, realtime=False) as eng:
        ctrl = EnrollmentController(store, eng, specs)
        eng.perform = lambda *a, **k: None  # the "user" does nothing
        attempt = ctrl.record("push")
    assert not attempt.quality.accepted and "no_motion" in attempt.quality.flags


def test_enrollment_idle_subtypes_cycle(tmp_path: Path, dsp_cfg: DSPConfig):
    with SimulatedAudioEngine(dsp_cfg, realtime=False) as eng:
        ctrl = EnrollmentController(sim_store(tmp_path), eng, load_gestures())
        subs = [ctrl.idle_subtype(i, 25)[0] for i in range(25)]
    assert subs[0] == "silence" and subs[-1] == "ambient"
    assert len(set(subs)) == 5


def test_enrollment_refuses_mixing(tmp_path: Path, dsp_cfg: DSPConfig):
    real = ProfileStore("r", simulated=False, base_dir=tmp_path / "profiles")
    real.create(DSPConfig(), 0.3)
    with pytest.raises(SimulatedDataError):
        EnrollmentController(real, SimulatedAudioEngine(dsp_cfg, realtime=False), load_gestures())
