import numpy as np
import pytest
import torch

from actions import Action, ActionExecutor
from config import DSPConfig, InferenceConfig
from inference import Cooldown, GestureRecognizer, InferenceWorker, MotionGate, PredictionSmoother
from simulator import SimulatedAudioEngine

CLASSES = ["idle", "push", "pull"]


class FixedModel(torch.nn.Module):
    """Stub classifier: always predicts ``label`` with probability ~``p``."""

    def __init__(self, label: int, p: float = 0.9) -> None:
        super().__init__()
        logits = torch.full((len(CLASSES),), float(np.log((1 - p) / (len(CLASSES) - 1))))
        logits[label] = float(np.log(p))
        self.register_buffer("logits", logits)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.logits.expand(x.shape[0], -1)


def test_gate_hysteresis():
    g = MotionGate(6.0, hold_seconds=0.3)
    assert not g.update(2.0, 0.0)
    assert g.update(8.0, 1.0)
    assert g.update(1.0, 1.2)  # still within hold
    assert not g.update(1.0, 1.5)


def test_smoother_requires_consistency():
    s = PredictionSmoother(3, 0.7)
    assert s.update("push", 0.9) is None
    assert s.update("push", 0.9) is None
    assert s.update("push", 0.9) == "push"
    s.reset()
    for label, p in [("push", 0.9), ("pull", 0.9), ("push", 0.9)]:
        assert s.update(label, p) is None
    s.reset()
    for _ in range(5):
        assert s.update("push", 0.5) is None  # below confidence
    s.reset()
    for _ in range(5):
        assert s.update("idle", 0.99) is None  # idle never fires


def test_cooldown():
    c = Cooldown(1.0)
    assert not c.active(0.0)
    c.trigger(5.0)
    assert c.active(5.5) and not c.active(6.01)


def _recognizer(label: int, cfg: InferenceConfig | None = None) -> GestureRecognizer:
    return GestureRecognizer(FixedModel(label), CLASSES, DSPConfig(), cfg or InferenceConfig(), gate_db=5.0)


def test_gate_skips_classifier_when_still():
    dsp = DSPConfig()
    rec = _recognizer(1)
    with SimulatedAudioEngine(dsp, realtime=False) as eng:
        eng.advance(1.5)
        for _ in range(10):
            eng.advance(0.1)
            res = rec.step(eng.get_latest(1.0), eng.now)
            assert not res.gate_open and not res.classified and res.triggered is None
    assert rec.model.calls == 0


def test_trigger_then_cooldown():
    dsp = DSPConfig()
    rec = _recognizer(1, InferenceConfig(smoothing=3, cooldown_seconds=1.5))
    triggers = []
    with SimulatedAudioEngine(dsp, realtime=False, seed=1) as eng:
        eng.advance(1.2)
        eng.perform("push", delay=0.2)
        for _ in range(25):
            eng.advance(0.1)
            res = rec.step(eng.get_latest(1.0), eng.now)
            if res.triggered:
                triggers.append((eng.now, res.triggered))
    assert len(triggers) == 1, triggers  # one trigger, then cooldown blocks repeats
    assert triggers[0][1] == "push"


def test_short_audio_is_ignored():
    rec = _recognizer(1)
    res = rec.step(np.zeros((100, 1), np.float32), 0.0)
    assert not res.gate_open


def test_worker_process_once_and_dry_run(capsys: pytest.CaptureFixture[str]):
    dsp = DSPConfig()
    rec = _recognizer(1, InferenceConfig(smoothing=2))
    actions = {"push": Action("push", "key", {"key": "space"})}
    ex = ActionExecutor(actions, enabled=True, dry_run=True)
    with SimulatedAudioEngine(dsp, realtime=False, seed=2) as eng:
        worker = InferenceWorker(eng, rec, ex)
        eng.advance(1.2)
        eng.perform("push", delay=0.1)
        for _ in range(20):
            eng.advance(0.1)
            worker.process_once(now=eng.now)
        snap = worker.snapshot()
    assert snap.events and snap.events[0].gesture == "push"
    assert "[dry-run] push: press space" in capsys.readouterr().out
    assert snap.steps == 20 and snap.snr_db is not None and snap.snr_db > 30


def test_worker_thread_runs(dsp_cfg: DSPConfig):
    rec = _recognizer(0)
    with SimulatedAudioEngine(dsp_cfg, realtime=True) as eng:
        worker = InferenceWorker(eng, rec, ActionExecutor({}), InferenceConfig(hop_seconds=0.05))
        worker.start()
        import time

        time.sleep(1.5)
        worker.stop()
    snap = worker.snapshot()
    assert snap.steps > 3 and snap.error is None and not worker.running


def test_carrier_loss_pauses_recognition_and_recovers():
    """A muted speaker must be reported as 'no carrier', not silently classified as idle."""
    dsp = DSPConfig()
    rec = _recognizer(1, InferenceConfig(smoothing=2, cooldown_seconds=0.5))
    worker = InferenceWorker(SimulatedAudioEngine(dsp, realtime=False, seed=4), rec, ActionExecutor({}))
    eng = worker.engine

    def run(seconds: float) -> list[str]:
        fired = []
        for _ in range(int(seconds * 10)):
            eng.advance(0.1)
            if worker.process_once(now=eng.now).triggered:
                fired.append("push")
        return fired

    with eng:
        eng.advance(1.2)
        run(1.0)
        assert not worker.snapshot().carrier_lost
        eng.set_tone_enabled(False)  # e.g. a "mute" key press or volume at 0
        run(2.5)
        assert worker.snapshot().carrier_lost
        calls_before = rec.model.calls
        eng.perform("push", delay=0.1)
        assert run(2.0) == []  # no guessing from noise
        assert rec.model.calls == calls_before
        eng.set_tone_enabled(True)
        run(2.5)
        assert not worker.snapshot().carrier_lost
        eng.perform("push", delay=0.1)
        assert "push" in run(2.0)  # recognition resumes once the tone is back
