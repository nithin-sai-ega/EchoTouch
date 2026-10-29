"""Train a GestureCNN on a profile's recorded clips.

    python train.py --profile default

Key points:
  * Splits by session/round (never randomly) so validation numbers are honest.
  * Class-balanced loss, AdamW, cosine LR, early stopping, best-checkpoint saving.
  * Writes model.pth, report.json and confusion_matrix.png into the profile and
    flags weak classes (especially experimental swipes).
  * Refuses simulated data unless --allow-simulated, and never mixes sources.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import (
    SIMULATED_BANNER,
    SOURCE_SIMULATED,
    DSPConfig,
    GestureSpec,
    TrainConfig,
    enabled_gestures,
    load_gestures,
)
from dataset import GestureDataset, SplitInfo, load_profile_data, split_records
from dsp import motion_score_db
from model import GestureCNN, save_checkpoint
from profiles import ProfileStore

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]


class TrainingStopped(RuntimeError):
    """Raised internally when a stop was requested."""


@dataclass
class TrainResult:
    model_path: Path
    report_path: Path
    confusion_path: Path | None
    history: list[dict[str, float]]
    report: dict[str, Any]
    split: SplitInfo
    best_epoch: int
    stopped_early: bool = False
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def class_weights(labels: Sequence[int], n_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes).astype(np.float64)
    weights = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(weights, dtype=torch.float32)


def suggest_gate_threshold(
    idle_scores: Sequence[float], gesture_scores: Sequence[float], default: float
) -> tuple[float, str]:
    """Pick a motion-gate threshold (dB) between idle and gesture motion scores.

    Biased towards recall (closer to idle) because the classifier rejects idle anyway;
    the gate's job is mainly to save compute and suppress obvious non-events.
    """
    if not idle_scores or not gesture_scores:
        return default, "default (not enough data to learn a gate threshold)"
    idle_hi = float(np.percentile(idle_scores, 90))
    gest_lo = float(np.percentile(gesture_scores, 10))
    if gest_lo > idle_hi:
        th = idle_hi + 0.4 * (gest_lo - idle_hi)
        return float(np.clip(th, 2.0, 15.0)), f"learned (idle p90={idle_hi:.1f} dB, gesture p10={gest_lo:.1f} dB)"
    th = max(2.0, gest_lo - 1.0)
    return th, (
        f"overlapping (idle p90={idle_hi:.1f} dB >= gesture p10={gest_lo:.1f} dB): gate set low; "
        "the classifier must reject idle motion"
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    stop_event: threading.Event | None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    train = optimizer is not None
    model.train(train)
    total_loss, n = 0.0, 0
    preds, targets = [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            if stop_event is not None and stop_event.is_set():
                raise TrainingStopped
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total_loss += float(loss.item()) * len(y)
            n += len(y)
            preds.append(logits.argmax(1).numpy())
            targets.append(y.numpy())
    p = np.concatenate(preds) if preds else np.zeros(0, int)
    t = np.concatenate(targets) if targets else np.zeros(0, int)
    acc = float((p == t).mean()) if len(t) else float("nan")
    return total_loss / max(n, 1), acc, p, t


def _balanced_accuracy(p: np.ndarray, t: np.ndarray, n_classes: int) -> float:
    recalls = [float((p[t == c] == c).mean()) for c in range(n_classes) if (t == c).any()]
    return float(np.mean(recalls)) if recalls else float("nan")


def evaluation_report(
    preds: np.ndarray, targets: np.ndarray, classes: Sequence[str], specs: dict[str, GestureSpec], weak_recall: float
) -> dict[str, Any]:
    """Confusion matrix + per-class precision/recall, flagging weak classes."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    labels = list(range(len(classes)))
    cm = confusion_matrix(targets, preds, labels=labels)
    prec, rec, f1, sup = precision_recall_fscore_support(targets, preds, labels=labels, zero_division=0)
    per_class = {}
    weak = []
    for i, c in enumerate(classes):
        spec = specs.get(c)
        per_class[c] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(sup[i]),
            "tier": spec.tier if spec else "custom",
        }
        if sup[i] > 0 and (rec[i] < weak_recall or prec[i] < weak_recall):
            weak.append(c)
    return {
        "accuracy": round(float((preds == targets).mean()), 4) if len(targets) else None,
        "balanced_accuracy": round(_balanced_accuracy(preds, targets, len(classes)), 4) if len(targets) else None,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "weak_classes": weak,
    }


def save_confusion_png(cm: Sequence[Sequence[int]], classes: Sequence[str], path: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(cm, dtype=float)
    norm = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * len(classes), 0.8 + 0.7 * len(classes)), dpi=120)
    im = ax.imshow(norm, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                f"{int(arr[i, j])}",
                ha="center",
                va="center",
                color="black" if norm[i, j] > 0.6 else "white",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_profile(
    store: ProfileStore,
    gestures: Sequence[GestureSpec],
    cfg: TrainConfig | None = None,
    allow_simulated: bool = False,
    callback: ProgressCallback | None = None,
    stop_event: threading.Event | None = None,
    default_gate_db: float = 6.0,
) -> TrainResult:
    """Train on ``store``'s clips for the given (enabled) ``gestures``.

    ``callback`` receives dicts with ``type`` in {"info", "warning", "epoch", "done"}.
    Setting ``stop_event`` stops training after the current batch; the best model so
    far is still saved.
    """
    cfg = cfg or TrainConfig()
    emit: ProgressCallback = callback or (lambda _e: None)
    classes = [g.name for g in gestures]
    specs = {g.name: g for g in gestures}
    if "idle" not in classes:
        raise ValueError("'idle' must be one of the trained classes")
    torch.manual_seed(cfg.seed)
    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)

    records, stfts, source = load_profile_data(store, classes, allow_simulated)
    dsp: DSPConfig = store.dsp_config()
    warnings: list[str] = []
    if source == SOURCE_SIMULATED:
        warnings.append(SIMULATED_BANNER + " This model is for tests/demos only.")
    labels = [classes.index(r["gesture"]) for r in records]
    train_idx, val_idx, split = split_records(records, classes, cfg.val_fraction)
    warnings += split.warnings
    for w in warnings:
        (logger.debug if callback else logger.warning)(w)  # the callback shows it already
        emit({"type": "warning", "message": w})
    emit({"type": "info", "message": f"{len(train_idx)} train / {len(val_idx)} val clips, split by {split.strategy}"})

    train_ds = GestureDataset([stfts[i] for i in train_idx], [labels[i] for i in train_idx], dsp, cfg.augment, cfg.seed)
    val_ds = GestureDataset([stfts[i] for i in val_idx], [labels[i] for i in val_idx], dsp, None)
    gen = torch.Generator().manual_seed(cfg.seed)
    # Class-balanced sampling (with replacement). Small profiles get at least
    # min_samples_per_epoch augmented draws per epoch so the optimizer takes enough steps.
    sample_weights = class_weights(train_ds.labels, len(classes))[torch.as_tensor(train_ds.labels)]
    sampler = WeightedRandomSampler(
        sample_weights.double(), max(len(train_ds), cfg.min_samples_per_epoch), replacement=True, generator=gen
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False)
    eval_loader = (
        val_loader
        if len(val_ds)
        else DataLoader(GestureDataset(train_ds.stfts, train_ds.labels, dsp, None), batch_size=cfg.batch_size * 2)
    )

    model = GestureCNN(len(classes), dsp.channels, cfg.embed_dim, cfg.width, cfg.dropout)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)  # classes are already balanced by the sampler
    eval_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.02)

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_key = (-1.0, float("-inf"))
    best_epoch, since_best, stopped = 0, 0, False
    t_start = time.monotonic()
    try:
        for epoch in range(1, cfg.epochs + 1):
            train_ds.set_epoch(epoch)
            tr_loss, tr_acc, _, _ = _run_epoch(model, train_loader, loss_fn, optimizer, stop_event)
            va_loss, _va_acc, p, t = _run_epoch(model, eval_loader, eval_loss_fn, None, stop_event)
            va_bal = _balanced_accuracy(p, t, len(classes))
            scheduler.step()
            key = (round(va_bal, 4), -va_loss)
            improved = key > best_key
            if improved:
                best_key, best_epoch, since_best = key, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                since_best += 1
            row = {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": float((p == t).mean()) if len(t) else float("nan"),
                "val_balanced_acc": va_bal,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_s": time.monotonic() - t_start,
            }
            history.append(row)
            emit({"type": "epoch", "epochs": cfg.epochs, "best_epoch": best_epoch, "improved": improved, **row})
            if since_best >= cfg.patience:
                emit({"type": "info", "message": f"Early stopping at epoch {epoch} (best epoch {best_epoch})."})
                break
    except TrainingStopped:
        stopped = True
        emit({"type": "warning", "message": "Training stopped by user; saving the best model so far."})
        if not history:
            raise

    model.load_state_dict(best_state)
    _, _, preds, targets = _run_epoch(model, eval_loader, eval_loss_fn, None, None)
    report = evaluation_report(preds, targets, classes, specs, cfg.weak_class_recall)

    # Learn a motion-gate threshold from the training clips (whole-clip motion scores).
    idle_scores = [motion_score_db(stfts[i], dsp) for i in train_idx if records[i]["gesture"] == "idle"]
    gest_scores = [motion_score_db(stfts[i], dsp) for i in train_idx if records[i]["gesture"] != "idle"]
    gate_db, gate_note = suggest_gate_threshold(idle_scores, gest_scores, default_gate_db)

    weak_msgs = []
    for c in report["weak_classes"]:
        tier = specs[c].tier if c in specs else "custom"
        pc = report["per_class"][c]
        weak_msgs.append(
            f"WEAK class '{c}' ({tier}{', experimental' if tier != 'core' else ''}): "
            f"recall {pc['recall']:.2f}, precision {pc['precision']:.2f}. Record more varied rounds "
            + ("or consider disabling it." if tier != "core" else "for it.")
        )
    for w in weak_msgs:
        (logger.debug if callback else logger.warning)(w)
        emit({"type": "warning", "message": w})
    warnings += weak_msgs

    full_report = {
        "profile": store.name,
        "source": source,
        "classes": classes,
        "n_clips": len(records),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "split": asdict(split),
        "best_epoch": best_epoch,
        "evaluated_on": "validation" if len(val_ds) else "training (no validation data!)",
        "metrics": report,
        "gate_db": round(gate_db, 2),
        "gate_note": gate_note,
        "history": history,
        "train_config": {**asdict(cfg), "augment": asdict(cfg.augment)},
        "dsp": dsp.to_dict(),
        "warnings": warnings,
        "stopped_by_user": stopped,
    }
    model_path = save_checkpoint(
        store.model_path,
        model,
        classes,
        dsp,
        source,
        extra={
            "profile": store.name,
            "gate_db": gate_db,
            "best_epoch": best_epoch,
            "val_accuracy": report["accuracy"],
            "val_balanced_accuracy": report["balanced_accuracy"],
            "split_strategy": split.strategy,
            "volume": store.volume,
        },
    )
    with open(store.report_path, "w", encoding="utf-8") as fh:
        json.dump(full_report, fh, indent=2)
    title = f"{store.name} ({source}) - {full_report['evaluated_on']}"
    confusion = None
    try:
        confusion = save_confusion_png(report["confusion_matrix"], classes, store.confusion_path, title)
    except Exception as exc:  # plotting must never lose a trained model
        logger.warning("Could not save confusion matrix image: %s", exc)
    emit({"type": "done", "report": full_report, "model_path": str(model_path)})
    return TrainResult(
        model_path, store.report_path, confusion, history, full_report, split, best_epoch, stopped, warnings
    )


def format_report(report: dict[str, Any]) -> str:
    """Pretty text version of a training report."""
    m = report["metrics"]
    lines = [
        f"Evaluated on {report['evaluated_on']} ({report['n_val']} clips, split by {report['split']['strategy']})",
        f"Accuracy: {m['accuracy']}   balanced accuracy: {m['balanced_accuracy']}",
        "",
        f"{'class':<14}{'tier':<10}{'precision':>10}{'recall':>8}{'support':>9}",
    ]
    for c, pc in m["per_class"].items():
        flag = "  <-- WEAK" if c in m["weak_classes"] else ""
        lines.append(f"{c:<14}{pc['tier']:<10}{pc['precision']:>10.2f}{pc['recall']:>8.2f}{pc['support']:>9}{flag}")
    lines += [
        "",
        "Confusion matrix (rows = true, cols = predicted):",
        "    " + " ".join(f"{c[:6]:>6}" for c in report["classes"]),
    ]
    for c, row in zip(report["classes"], m["confusion_matrix"]):
        lines.append(f"{c[:4]:<4}" + " ".join(f"{v:>6}" for v in row))
    lines.append(f"\nMotion gate threshold: {report['gate_db']} dB ({report['gate_note']})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from cli_common import add_common_args, print_banner_if_simulated, setup_logging

    parser = argparse.ArgumentParser(description="Train the EchoTouch gesture model on your recorded profile")
    add_common_args(parser)
    parser.add_argument("--allow-simulated", action="store_true", help="permit training on simulated clips (sim/ only)")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--val-fraction", type=float, default=TrainConfig.val_fraction)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    print_banner_if_simulated(args.simulate)
    if args.simulate and not args.allow_simulated:
        print("ERROR: --simulate trains on SIMULATED clips; add --allow-simulated to confirm this is a demo/test.")
        return 2

    from tqdm import tqdm

    from profiles import ProfileError

    gestures = enabled_gestures(load_gestures(args.gestures_file))
    store = ProfileStore(args.profile, simulated=args.simulate)
    cfg = replace(
        TrainConfig(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    bar = tqdm(total=cfg.epochs, desc="training", unit="epoch")

    def on_progress(ev: dict[str, Any]) -> None:
        if ev["type"] == "epoch":
            bar.update(1)
            bar.set_postfix(loss=f"{ev['train_loss']:.3f}", val_acc=f"{ev['val_acc']:.3f}", best=ev["best_epoch"])
        elif ev["type"] in ("info", "warning"):
            bar.write(("WARNING: " if ev["type"] == "warning" else "") + ev["message"])

    try:
        result = train_profile(store, gestures, cfg, allow_simulated=args.allow_simulated, callback=on_progress)
    except (ProfileError, RuntimeError, ValueError) as exc:
        bar.close()
        print(f"ERROR: {exc}")
        return 2
    bar.close()
    print()
    print(format_report(result.report))
    print(f"\nSaved model:   {result.model_path}\nSaved report:  {result.report_path}")
    if result.confusion_path:
        print(f"Confusion PNG: {result.confusion_path}")
    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
