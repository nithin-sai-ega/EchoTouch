from pathlib import Path

import pytest
import torch

from config import ConfigMismatchError, DSPConfig
from model import CheckpointError, GestureCNN, SimulatedModelError, load_checkpoint, read_checkpoint, save_checkpoint

CLASSES = ["idle", "push", "pull", "wave"]


def test_forward_and_embed_shapes():
    m = GestureCNN(4, in_channels=1, embed_dim=64, width=8).eval()
    x = torch.randn(3, 1, 57, 79)
    assert m(x).shape == (3, 4)
    assert m.embed(x).shape == (3, 64)
    # Global pooling: other time lengths and multi-channel input work too.
    assert m(torch.randn(2, 1, 57, 120)).shape == (2, 4)
    m2 = GestureCNN(3, in_channels=2, width=8).eval()
    assert m2(torch.randn(2, 2, 53, 86)).shape == (2, 3)
    with pytest.raises(ValueError):
        m.embed(torch.randn(57, 79))


def test_checkpoint_roundtrip(tmp_path: Path):
    dsp = DSPConfig()
    m = GestureCNN(4, width=8).eval()
    p = save_checkpoint(tmp_path / "model.pth", m, CLASSES, dsp, "real", {"gate_db": 5.0})
    loaded, meta = load_checkpoint(p, current_dsp=dsp)
    x = torch.randn(2, 1, dsp.freq_bins, dsp.window_frames)
    torch.testing.assert_close(loaded(x), m(x))
    assert meta["class_names"] == CLASSES and meta["source"] == "real"
    assert meta["carrier_hz"] == dsp.carrier_hz and meta["sample_rate"] == dsp.sample_rate
    assert meta["extra"]["gate_db"] == 5.0
    assert "state_dict" not in meta


def test_checkpoint_dsp_mismatch(tmp_path: Path):
    p = save_checkpoint(tmp_path / "m.pth", GestureCNN(4, width=8), CLASSES, DSPConfig(), "real")
    for other in (DSPConfig(carrier_hz=19500.0), DSPConfig(sample_rate=48000), DSPConfig(n_fft=2048)):
        with pytest.raises(ConfigMismatchError):
            load_checkpoint(p, current_dsp=other)


def test_simulated_model_refused_for_live(tmp_path: Path):
    p = save_checkpoint(tmp_path / "m.pth", GestureCNN(4, width=8), CLASSES, DSPConfig(), "simulated")
    with pytest.raises(SimulatedModelError, match="SIMULATED"):
        load_checkpoint(p, current_dsp=DSPConfig())
    model, meta = load_checkpoint(p, current_dsp=DSPConfig(), allow_simulated=True)
    assert meta["source"] == "simulated"


def test_bad_checkpoints(tmp_path: Path):
    with pytest.raises(CheckpointError, match="No model"):
        read_checkpoint(tmp_path / "missing.pth")
    p = save_checkpoint(tmp_path / "m.pth", GestureCNN(4, width=8), CLASSES, DSPConfig(), "real")
    ckpt = torch.load(p, weights_only=True)
    ckpt["source"] = None
    torch.save(ckpt, p)
    with pytest.raises(CheckpointError, match="source"):
        load_checkpoint(p)
    ckpt["source"], ckpt["model_version"] = "real", 999
    torch.save(ckpt, p)
    with pytest.raises(CheckpointError, match="model_version"):
        load_checkpoint(p)
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / "x.pth", GestureCNN(4, width=8), CLASSES[:3], DSPConfig(), "real")
