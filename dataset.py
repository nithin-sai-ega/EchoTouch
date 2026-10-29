"""Dataset loading and leakage-free train/validation splitting.

Back-to-back clips of the same gesture are near-identical, so a random split
yields fake ~99% validation accuracy. We therefore split by *recording block*:

1. ``session`` split - hold out entire enrollment sessions (best; >= 2 sessions).
2. ``round`` split   - hold out entire (session, round) blocks. With a single session
   this is optimistic (same day, same setup) and a warning is emitted.
3. ``chronological`` - last resort with a single round: the last clips of each
   gesture, as one contiguous block. Strongly warned.

A split is only accepted if every class has training data; classes without any
validation clips are warned about.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from config import SOURCE_REAL, SOURCE_SIMULATED, AugmentConfig, DSPConfig
from dsp import augment_features, center_features, stft_crop
from profiles import ProfileError, ProfileStore, SimulatedDataError

logger = logging.getLogger(__name__)


class DatasetError(RuntimeError):
    """Raised when the profile's data cannot be used for training."""


@dataclass
class SplitInfo:
    strategy: str
    train_groups: list[str]
    val_groups: list[str]
    warnings: list[str] = field(default_factory=list)


def group_key(rec: dict[str, Any], strategy: str) -> str:
    if strategy == "session":
        return str(rec["session_id"])
    if strategy == "round":
        return f"{rec['session_id']}/r{int(rec['round']):02d}"
    raise ValueError(strategy)


def check_sources(records: Sequence[dict[str, Any]], allow_simulated: bool) -> str:
    """Return the single data source of ``records`` or raise.

    Never mixes real and simulated clips; simulated clips need ``allow_simulated``.
    """
    sources = {r.get("source") for r in records}
    if not sources:
        raise DatasetError("No clips recorded yet. Run enroll.py first.")
    if sources - {SOURCE_REAL, SOURCE_SIMULATED}:
        raise DatasetError(f"Clips with unknown source tags: {sources}")
    if len(sources) > 1:
        raise SimulatedDataError("Refusing to train: profile mixes real and simulated clips.")
    source = sources.pop()
    if source == SOURCE_SIMULATED and not allow_simulated:
        raise SimulatedDataError(
            "These clips are SIMULATED. Refusing to train without --allow-simulated "
            "(simulated models are for tests/demos only and can never control your computer)."
        )
    return source


def _try_group_split(
    records: Sequence[dict[str, Any]], classes: Sequence[str], strategy: str, val_fraction: float
) -> tuple[list[int], list[int], SplitInfo] | None:
    """Hold out the most recent groups until each class has ~val_fraction in validation."""
    groups: dict[str, list[int]] = defaultdict(list)
    first_seen: dict[str, str] = {}
    for i, r in enumerate(records):
        k = group_key(r, strategy)
        groups[k].append(i)
        first_seen.setdefault(k, r.get("timestamp", ""))
        first_seen[k] = min(first_seen[k], r.get("timestamp", ""))
    if len(groups) < 2:
        return None
    order = sorted(groups, key=lambda k: (first_seen[k], k), reverse=True)  # newest first
    total = Counter(r["gesture"] for r in records)
    val_counts: Counter[str] = Counter()
    train_counts = Counter(total)
    val_groups: list[str] = []
    for k in order:
        if all(val_counts[c] >= val_fraction * total[c] for c in classes if total[c]):
            break
        gc = Counter(records[i]["gesture"] for i in groups[k])
        # Never remove the last training data of any class.
        if any(train_counts[c] - gc[c] <= 0 for c in gc):
            continue
        val_groups.append(k)
        val_counts.update(gc)
        train_counts.subtract(gc)
    if not val_groups:
        return None
    val_set = set(val_groups)
    val_idx = [i for k in val_groups for i in groups[k]]
    train_idx = [i for k in order if k not in val_set for i in groups[k]]
    info = SplitInfo(strategy, sorted(set(groups) - val_set), sorted(val_set))
    missing_val = [c for c in classes if total[c] and val_counts[c] == 0]
    if missing_val:
        info.warnings.append(
            f"No validation clips for: {', '.join(missing_val)} (record another round to evaluate them)."
        )
    return sorted(train_idx), sorted(val_idx), info


def split_records(
    records: Sequence[dict[str, Any]], classes: Sequence[str], val_fraction: float = 0.2
) -> tuple[list[int], list[int], SplitInfo]:
    """Split clip records into train/val *indices* by session, round, or chronology."""
    if not records:
        raise DatasetError("No clips to split.")
    missing = [c for c in classes if not any(r["gesture"] == c for r in records)]
    if missing:
        raise DatasetError(f"No clips recorded for: {', '.join(missing)}. Record them or disable them.")
    sessions = {r["session_id"] for r in records}
    if len(sessions) >= 2:
        res = _try_group_split(records, classes, "session", val_fraction)
        if res is not None and not res[2].warnings:
            return res
    res = _try_group_split(records, classes, "round", val_fraction)
    if res is not None:
        if len(sessions) < 2:
            res[2].warnings.insert(
                0,
                "Only one enrollment session: validation holds out later ROUNDS of the same session, "
                "so accuracy will be OPTIMISTIC. Record another session (different time/posture) "
                "for an honest estimate.",
            )
        else:
            res[2].warnings.insert(0, "Session-level split was not possible; held out whole rounds instead.")
        return res
    # Chronological fallback: last block of each gesture.
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_class[r["gesture"]].append(i)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for _c, idxs in by_class.items():
        idxs = sorted(idxs, key=lambda i: (records[i].get("timestamp", ""), i))
        n_val = int(round(len(idxs) * val_fraction)) if len(idxs) >= 5 else 0
        train_idx += idxs[: len(idxs) - n_val]
        val_idx += idxs[len(idxs) - n_val :]
    info = SplitInfo(
        "chronological",
        ["(early clips)"],
        ["(late clips)"],
        [
            "Only ONE round recorded: validation = the last clips of each gesture. This is VERY "
            "optimistic. Record at least 2 rounds, ideally 2+ sessions."
        ],
    )
    return sorted(train_idx), sorted(val_idx), info


def load_profile_data(
    store: ProfileStore, classes: Sequence[str], allow_simulated: bool = False
) -> tuple[list[dict[str, Any]], list[np.ndarray], str]:
    """Load the clip records + cropped complex STFTs for ``classes``.

    Returns ``(records, stfts, source)``; each STFT is ``(C, F, T_clip)`` complex64.
    """
    records = store.clips(list(classes))
    source = check_sources(store.clips(), allow_simulated)  # check the whole profile, not just classes
    dsp = store.dsp_config()
    if not records:
        raise DatasetError("No clips for the enabled gestures.")
    stfts = []
    min_samples = dsp.samples(dsp.window_seconds)
    for rec in records:
        if (
            int(rec.get("sample_rate", dsp.sample_rate)) != dsp.sample_rate
            or abs(float(rec.get("carrier_hz", dsp.carrier_hz)) - dsp.carrier_hz) > 1e-6
        ):
            raise ProfileError(
                f"Clip {rec['filename']} was recorded with different carrier/sample rate than the profile."
            )
        audio = store.load_clip(rec)
        if audio.shape[1] < min_samples:
            raise ProfileError(f"Clip {rec['filename']} is shorter than the classifier window.")
        stfts.append(stft_crop(audio, dsp))
    return records, stfts, source


class GestureDataset(Dataset):
    """Torch dataset over precomputed cropped STFTs with optional augmentation."""

    def __init__(
        self,
        stfts: Sequence[np.ndarray],
        labels: Sequence[int],
        dsp: DSPConfig,
        augment: AugmentConfig | None = None,
        seed: int = 0,
    ) -> None:
        if len(stfts) != len(labels):
            raise ValueError("stfts and labels differ in length")
        self.stfts = list(stfts)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.dsp = dsp
        self.augment = augment
        self._seed = seed
        self._epoch = 0
        self._draws = 0
        self._cache: list[np.ndarray | None] = [None] * len(self.stfts)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._draws = 0

    def __len__(self) -> int:
        return len(self.stfts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if self.augment is None:
            feats = self._cache[idx]
            if feats is None:
                feats = center_features(self.stfts[idx], self.dsp)
                self._cache[idx] = feats
        else:
            # Every draw gets a fresh view, even when the sampler repeats an index.
            self._draws += 1
            rng = np.random.default_rng((self._seed, self._epoch, idx, self._draws))
            feats = augment_features(self.stfts[idx], self.dsp, self.augment, rng)
        return torch.from_numpy(np.ascontiguousarray(feats)), int(self.labels[idx])
