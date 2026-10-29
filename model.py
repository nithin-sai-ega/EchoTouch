"""GestureCNN and checkpoint handling.

Architecture: ``encoder -> embedding -> linear classifier``. The encoder is a
small 2-D CNN over the carrier-relative spectrogram ``(C, F, T)``. A coordinate
channel encoding each row's frequency offset from the carrier is appended to the
input, so the network knows above-carrier (approaching) from below-carrier
(receding) energy even after global pooling, which keeps it input-size agnostic.

``.embed(x)`` exposes the embedding so prototype/few-shot onboarding can be added
later without changing the architecture (not implemented yet; see README roadmap).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from config import MODEL_VERSION, SOURCE_REAL, SOURCE_SIMULATED, ConfigError, DSPConfig

logger = logging.getLogger(__name__)


class CheckpointError(RuntimeError):
    """Raised for unreadable or incompatible checkpoints."""


class SimulatedModelError(CheckpointError):
    """Raised when a simulated model would be used for real live control."""


def _block(c_in: int, c_out: int, dropout: float, double: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    ]
    if double:
        layers += [nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(2, ceil_mode=True), nn.Dropout2d(dropout * 0.5)]
    return nn.Sequential(*layers)


class GestureCNN(nn.Module):
    """2-D CNN gesture classifier with an exposed embedding.

    Args:
        n_classes: number of output classes.
        in_channels: number of microphone channels in the features.
        embed_dim: size of the embedding returned by :meth:`embed`.
        width: channel count of the first conv block (doubles twice).
        dropout: dropout before the classifier.
    """

    def __init__(
        self, n_classes: int, in_channels: int = 1, embed_dim: int = 96, width: int = 24, dropout: float = 0.3
    ) -> None:
        super().__init__()
        if n_classes < 2:
            raise ConfigError("need at least two classes")
        self.hparams = {
            "n_classes": n_classes,
            "in_channels": in_channels,
            "embed_dim": embed_dim,
            "width": width,
            "dropout": dropout,
        }
        w = width
        self.encoder = nn.Sequential(
            _block(in_channels + 1, w, dropout, double=False),  # cheap at full resolution
            _block(w, 2 * w, dropout),
            _block(2 * w, 4 * w, dropout),
        )
        self.embedding = nn.Sequential(
            nn.Linear(2 * 4 * w, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, n_classes))

    @staticmethod
    def _with_coords(x: torch.Tensor) -> torch.Tensor:
        b, _, f, t = x.shape
        coord = torch.linspace(-1.0, 1.0, f, device=x.device, dtype=x.dtype).view(1, 1, f, 1)
        return torch.cat([x, coord.expand(b, 1, f, t)], dim=1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return the ``(batch, embed_dim)`` embedding for features ``(batch, C, F, T)``."""
        if x.dim() != 4:
            raise ValueError(f"expected (batch, C, F, T), got {tuple(x.shape)}")
        h = self.encoder(self._with_coords(x))
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.embedding(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: Path | str,
    model: GestureCNN,
    class_names: list[str],
    dsp: DSPConfig,
    source: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save weights plus everything inference needs to reproduce the features."""
    if source not in (SOURCE_REAL, SOURCE_SIMULATED):
        raise ValueError(f"invalid source {source!r}")
    if len(class_names) != model.hparams["n_classes"]:
        raise ValueError("class_names does not match the model's output size")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": MODEL_VERSION,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "hparams": dict(model.hparams),
        "class_names": list(class_names),
        "dsp": dsp.to_dict(),
        "carrier_hz": dsp.carrier_hz,
        "sample_rate": dsp.sample_rate,
        "source": source,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    logger.info("Saved %s model to %s", source, path)
    return path


def read_checkpoint(path: Path | str) -> dict[str, Any]:
    """Load the raw checkpoint dict (weights-only safe loading)."""
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"No model found at {path}. Train one first with train.py.")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"Could not read checkpoint {path}: {exc}") from exc
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise CheckpointError(f"{path} is not an EchoTouch checkpoint")
    if ckpt.get("model_version") != MODEL_VERSION:
        raise CheckpointError(
            f"{path} has model_version {ckpt.get('model_version')}, this code expects {MODEL_VERSION}. Retrain."
        )
    return ckpt


def load_checkpoint(
    path: Path | str,
    current_dsp: DSPConfig | None = None,
    allow_simulated: bool = False,
) -> tuple[GestureCNN, dict[str, Any]]:
    """Load a model for inference with all safety checks.

    Args:
        path: checkpoint file.
        current_dsp: the current calibration's DSP config. If given, it must match the
            checkpoint exactly (carrier, sample rate, FFT, crop, channels...).
        allow_simulated: only True in ``--simulate`` mode. Simulated models are
            never loaded for real live control.

    Returns:
        ``(model in eval mode, checkpoint metadata without weights)``.
    """
    ckpt = read_checkpoint(path)
    source = ckpt.get("source")
    if source not in (SOURCE_REAL, SOURCE_SIMULATED):
        raise CheckpointError(f"{path} has no valid 'source' tag; refusing to load it")
    if source == SOURCE_SIMULATED and not allow_simulated:
        raise SimulatedModelError(
            f"{path} was trained on SIMULATED data and cannot be used for real live control. "
            "Record your own data with enroll.py and train on it (or run with --simulate for a demo)."
        )
    stored = DSPConfig.from_dict(ckpt["dsp"])
    if current_dsp is not None:
        stored.require_compatible(current_dsp, what="model")
    model = GestureCNN(**ckpt["hparams"])
    try:
        model.load_state_dict(ckpt["state_dict"])
    except RuntimeError as exc:
        raise CheckpointError(f"Checkpoint weights do not match the architecture: {exc}") from exc
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta
