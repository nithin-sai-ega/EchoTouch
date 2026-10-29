"""Profile storage: raw clips + ``manifest.json``.

A profile is one person's acoustic signature on one laptop::

    profiles/<name>/
        manifest.json        devices, DSP params, calibration, per-clip records
        clips/<gesture>/*.wav  raw float32 WAV (all mic channels), ~1.5 s each
        model.pth            trained model (train.py)
        report.json          training report, confusion_matrix.png

Real profiles live in ``profiles/``; simulated ones live ONLY in ``sim/``. The
store enforces this: a simulated store cannot be opened under ``profiles/`` and a
real one cannot be opened under ``sim/``, and clip ``source`` must match the mode.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

from config import (
    MANIFEST_SCHEMA_VERSION,
    PROFILES_DIRNAME,
    REPO_ROOT,
    SIM_DIRNAME,
    SOURCE_REAL,
    SOURCE_SIMULATED,
    DSPConfig,
    validate_profile_name,
)

logger = logging.getLogger(__name__)


class ProfileError(RuntimeError):
    """Generic profile problem (missing, corrupt, inconsistent)."""


class SimulatedDataError(ProfileError):
    """Raised when simulated and real data would be mixed or misplaced."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_base_dir(simulated: bool) -> Path:
    return REPO_ROOT / (SIM_DIRNAME if simulated else PROFILES_DIRNAME)


def list_profiles(simulated: bool = False, base_dir: Path | None = None) -> list[str]:
    base = Path(base_dir) if base_dir is not None else default_base_dir(simulated)
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "manifest.json").exists())


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


class ProfileStore:
    """Read/write access to one profile directory. Thread-safe within a process."""

    def __init__(self, name: str, simulated: bool = False, base_dir: Path | str | None = None) -> None:
        self.name = validate_profile_name(name)
        self.simulated = simulated
        base = Path(base_dir) if base_dir is not None else default_base_dir(simulated)
        if simulated and base.name != SIM_DIRNAME:
            raise SimulatedDataError(
                f"Simulated data may only be stored under a '{SIM_DIRNAME}/' directory, not {base}"
            )
        if not simulated and base.name == SIM_DIRNAME:
            raise SimulatedDataError(f"Real profiles cannot be stored under '{SIM_DIRNAME}/' ({base})")
        self.base_dir = base
        self.root = base / self.name
        self._lock = threading.RLock()
        self._manifest: dict[str, Any] | None = None

    # -- paths ---------------------------------------------------------------- #
    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def clips_dir(self) -> Path:
        return self.root / "clips"

    @property
    def model_path(self) -> Path:
        return self.root / "model.pth"

    @property
    def report_path(self) -> Path:
        return self.root / "report.json"

    @property
    def confusion_path(self) -> Path:
        return self.root / "confusion_matrix.png"

    @property
    def mode(self) -> str:
        return SOURCE_SIMULATED if self.simulated else SOURCE_REAL

    def exists(self) -> bool:
        return self.manifest_path.exists()

    # -- manifest ------------------------------------------------------------- #
    @property
    def manifest(self) -> dict[str, Any]:
        with self._lock:
            if self._manifest is None:
                self._manifest = self._read_manifest()
            return self._manifest

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise ProfileError(
                f"Profile '{self.name}' not found at {self.root}. Run doctor.py / enroll.py first"
                + (" (with --simulate)." if self.simulated else ".")
            )
        try:
            with open(self.manifest_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ProfileError(f"Corrupt manifest {self.manifest_path}: {exc}") from exc
        if data.get("mode") != self.mode:
            raise SimulatedDataError(
                f"Profile at {self.root} is tagged mode={data.get('mode')!r} but was opened as {self.mode!r}"
            )
        return data

    def reload(self) -> None:
        with self._lock:
            self._manifest = None

    def save(self) -> None:
        with self._lock:
            if self._manifest is None:
                return
            self.root.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(self.manifest_path, self._manifest)

    def create(self, dsp: DSPConfig, volume: float, devices: dict[str, str] | None = None) -> None:
        """Create a new empty profile (no-op if it already exists)."""
        with self._lock:
            if self.exists():
                return
            self.root.mkdir(parents=True, exist_ok=True)
            self._manifest = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "profile": self.name,
                "mode": self.mode,
                "created": now_iso(),
                "dsp": dsp.to_dict(),
                "audio": {"volume": volume},
                "devices": devices or {},
                "calibration": None,
                "sessions": {},
                "clips": [],
            }
            self.save()
            logger.info("Created %s profile '%s' at %s", self.mode, self.name, self.root)

    def ensure(self, dsp: DSPConfig, volume: float, devices: dict[str, str] | None = None) -> None:
        if not self.exists():
            self.create(dsp, volume, devices)

    def dsp_config(self) -> DSPConfig:
        return DSPConfig.from_dict(self.manifest["dsp"])

    @property
    def volume(self) -> float:
        return float(self.manifest.get("audio", {}).get("volume", 0.3))

    @property
    def calibration(self) -> dict[str, Any] | None:
        return self.manifest.get("calibration")

    @property
    def devices(self) -> dict[str, str]:
        return dict(self.manifest.get("devices", {}))

    def set_calibration(
        self,
        calibration: dict[str, Any],
        dsp: DSPConfig,
        volume: float,
        devices: dict[str, str] | None = None,
        force: bool = False,
    ) -> list[str]:
        """Store calibration results. Returns warnings.

        Changing carrier/sample rate/channels or volume when clips already exist would
        silently invalidate them, so it requires ``force=True``.
        """
        with self._lock:
            self.ensure(dsp, volume, devices)
            m = self.manifest
            old = DSPConfig.from_dict(m["dsp"])
            warnings: list[str] = []
            diffs = old.diff(dsp)
            vol_changed = abs(float(m["audio"].get("volume", volume)) - volume) > 1e-6
            if (diffs or vol_changed) and m["clips"]:
                msg = (
                    f"Profile '{self.name}' already has {len(m['clips'])} clips recorded with "
                    f"different settings ({'; '.join(diffs) or 'volume changed'}). Changing carrier "
                    "or volume invalidates existing clips and trained models."
                )
                if not force:
                    raise ProfileError(msg + " Use a new --profile, or pass --force to overwrite anyway.")
                warnings.append(msg)
            m["dsp"] = dsp.to_dict()
            m["audio"]["volume"] = volume
            if devices:
                m["devices"] = devices
            m["calibration"] = {**calibration, "updated": now_iso()}
            self.save()
            return warnings

    # -- sessions --------------------------------------------------------------- #
    def new_session(self) -> str:
        """Start a new enrollment session and return its id."""
        with self._lock:
            sid = datetime.now().strftime("s%Y%m%d-%H%M%S-") + secrets.token_hex(2)
            self.manifest["sessions"][sid] = {"started": now_iso(), "rounds": 0}
            self.save()
            return sid

    def start_round(self, session_id: str) -> int:
        """Increment and return the round number for ``session_id`` (1-based)."""
        with self._lock:
            sess = self.manifest["sessions"].setdefault(session_id, {"started": now_iso(), "rounds": 0})
            sess["rounds"] = int(sess.get("rounds", 0)) + 1
            self.save()
            return sess["rounds"]

    # -- clips ------------------------------------------------------------------ #
    def clips(self, gestures: list[str] | tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        recs = list(self.manifest.get("clips", []))
        if gestures is not None:
            wanted = set(gestures)
            recs = [r for r in recs if r["gesture"] in wanted]
        return recs

    def counts(self) -> Counter[str]:
        return Counter(r["gesture"] for r in self.manifest.get("clips", []))

    def add_clip(
        self,
        audio: np.ndarray,
        gesture: str,
        session_id: str,
        round_no: int,
        source: str,
        quality: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Save raw audio ``(n,)`` or ``(n, channels)`` and append a clip record."""
        if source not in (SOURCE_REAL, SOURCE_SIMULATED):
            raise ValueError(f"invalid source {source!r}")
        if source != self.mode:
            raise SimulatedDataError(
                f"Refusing to store a {source!r} clip in a {self.mode!r} profile ({self.root}). "
                "Simulated and real data are never mixed."
            )
        data = np.asarray(audio, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        with self._lock:
            m = self.manifest
            dsp = DSPConfig.from_dict(m["dsp"])
            if data.shape[1] != dsp.channels:
                raise ProfileError(f"clip has {data.shape[1]} channels, profile expects {dsp.channels}")
            idx = int(m.get("next_clip_index", len(m["clips"])))
            rel = Path("clips") / gesture / f"{session_id}_r{round_no:02d}_{idx:05d}.wav"
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(path, dsp.sample_rate, data)
            record = {
                "filename": rel.as_posix(),
                "gesture": gesture,
                "session_id": session_id,
                "round": int(round_no),
                "timestamp": now_iso(),
                "source": source,
                "sample_rate": dsp.sample_rate,
                "carrier_hz": dsp.carrier_hz,
                "channels": int(data.shape[1]),
                "duration_s": round(data.shape[0] / dsp.sample_rate, 4),
                "quality": quality or {},
            }
            if extra:
                record.update(extra)
            m["clips"].append(record)
            m["next_clip_index"] = idx + 1
            self.save()
            return record

    def undo_last(self, session_id: str | None = None) -> dict[str, Any] | None:
        """Remove the most recent clip (optionally only from ``session_id``).

        The WAV is moved to ``_trash/`` rather than deleted, so nothing is lost by accident.
        """
        with self._lock:
            clips = self.manifest["clips"]
            for i in range(len(clips) - 1, -1, -1):
                if session_id is None or clips[i]["session_id"] == session_id:
                    rec = clips.pop(i)
                    src = self.root / rec["filename"]
                    if src.exists():
                        dst = self.root / "_trash" / rec["filename"]
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(src), str(dst))
                    self.save()
                    return rec
            return None

    def load_clip(self, record: dict[str, Any]) -> np.ndarray:
        """Load a clip as float32 ``(channels, samples)``."""
        path = self.root / record["filename"]
        if not path.exists():
            raise ProfileError(f"Missing clip file {path} (manifest and files out of sync)")
        sr, data = wavfile.read(path)
        dsp = self.dsp_config()
        if sr != dsp.sample_rate:
            raise ProfileError(f"{path} has sample rate {sr}, profile expects {dsp.sample_rate}")
        data = np.asarray(data)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        data = data.astype(np.float32)
        if data.ndim == 1:
            data = data[:, None]
        return np.ascontiguousarray(data.T)
