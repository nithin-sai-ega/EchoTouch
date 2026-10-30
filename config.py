"""Central configuration for EchoTouch.

Everything tunable lives here as immutable dataclasses. Gesture definitions are
loaded from ``gestures.yaml`` (falling back to the built-in defaults below), so
users can enable/disable gestures or add their own without touching code.

Rule of thumb: anything that changes what the microphone "sees" (sample rate,
carrier frequency, FFT size, crop width, channels) is part of :class:`DSPConfig`
and is stored in every profile manifest and every model checkpoint. Inference
refuses to run when these do not match.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

#: Repository root (all default paths are relative to it, not to the CWD).
REPO_ROOT = Path(__file__).resolve().parent

#: Real, user-recorded profiles live here. Never contains simulated data.
PROFILES_DIRNAME = "profiles"
#: Simulated data and models live here. Never contains real data.
SIM_DIRNAME = "sim"

DEFAULT_PROFILE = "default"
DEFAULT_GESTURES_FILE = REPO_ROOT / "gestures.yaml"
DEFAULT_ACTIONS_FILE = REPO_ROOT / "actions.yaml"

#: Bumped whenever the checkpoint layout or feature pipeline changes incompatibly.
MODEL_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1

SIMULATED_BANNER = "SIMULATED MODE: not real data."

SOURCE_REAL = "real"
SOURCE_SIMULATED = "simulated"


class ConfigError(ValueError):
    """Raised for invalid configuration values or files."""


class ConfigMismatchError(RuntimeError):
    """Raised when stored DSP parameters do not match the current ones."""


# --------------------------------------------------------------------------- #
# DSP
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DSPConfig:
    """Signal-processing parameters. Must be identical at train and inference time."""

    sample_rate: int = 44100
    carrier_hz: float = 18500.0
    n_fft: int = 4096
    hop: int = 512
    crop_hz: float = 300.0
    channels: int = 1
    #: ``median`` or ``mean`` of each frequency row over time (log domain), or ``none``.
    background: str = "median"
    #: ``robust`` (median/MAD) or ``zscore`` per-sample normalization.
    normalization: str = "robust"
    #: Bins within +/- guard_hz of the carrier count as "static" for the motion gate.
    guard_hz: float = 25.0
    #: Length of each recorded clip (seconds). Raw audio is stored at this length.
    clip_seconds: float = 1.5
    #: Length of the classifier window (seconds). Training randomly crops to this.
    window_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.sample_rate not in (44100, 48000, 96000):
            raise ConfigError(f"sample_rate must be 44100, 48000 or 96000, got {self.sample_rate}")
        if self.n_fft <= 0 or self.n_fft & (self.n_fft - 1):
            raise ConfigError(f"n_fft must be a positive power of two, got {self.n_fft}")
        if not 0 < self.hop <= self.n_fft:
            raise ConfigError(f"hop must be in (0, n_fft], got {self.hop}")
        if self.crop_hz <= 2 * self.guard_hz:
            raise ConfigError("crop_hz must be larger than 2 * guard_hz")
        nyquist = self.sample_rate / 2
        if not 1000.0 <= self.carrier_hz - self.crop_hz or self.carrier_hz + self.crop_hz >= nyquist:
            raise ConfigError(
                f"carrier {self.carrier_hz:.0f} Hz +/- {self.crop_hz:.0f} Hz does not fit below "
                f"Nyquist ({nyquist:.0f} Hz) for sample rate {self.sample_rate}"
            )
        if self.channels < 1:
            raise ConfigError("channels must be >= 1")
        if self.background not in ("median", "mean", "none"):
            raise ConfigError(f"background must be median|mean|none, got {self.background!r}")
        if self.normalization not in ("robust", "zscore"):
            raise ConfigError(f"normalization must be robust|zscore, got {self.normalization!r}")
        if self.window_seconds > self.clip_seconds:
            raise ConfigError("window_seconds must be <= clip_seconds")
        if self.samples(self.window_seconds) < self.n_fft:
            raise ConfigError("window_seconds is shorter than one FFT frame")

    # -- derived quantities ------------------------------------------------- #
    @property
    def bin_hz(self) -> float:
        """Frequency resolution of one FFT bin."""
        return self.sample_rate / self.n_fft

    @property
    def carrier_bin(self) -> int:
        return int(round(self.carrier_hz / self.bin_hz))

    @property
    def half_bins(self) -> int:
        """Number of bins kept on each side of the carrier."""
        return int(round(self.crop_hz / self.bin_hz))

    @property
    def freq_bins(self) -> int:
        return 2 * self.half_bins + 1

    @property
    def guard_bins(self) -> int:
        return max(1, int(round(self.guard_hz / self.bin_hz)))

    def samples(self, seconds: float) -> int:
        return int(round(seconds * self.sample_rate))

    def frames_for(self, n_samples: int) -> int:
        """Number of STFT frames produced for ``n_samples`` (no padding)."""
        if n_samples < self.n_fft:
            return 0
        return 1 + (n_samples - self.n_fft) // self.hop

    @property
    def window_frames(self) -> int:
        return self.frames_for(self.samples(self.window_seconds))

    @property
    def clip_frames(self) -> int:
        return self.frames_for(self.samples(self.clip_seconds))

    def relative_freqs(self) -> list[float]:
        """Frequency (Hz, relative to carrier) of each cropped bin."""
        return [(k - self.half_bins) * self.bin_hz for k in range(self.freq_bins)]

    # -- (de)serialization --------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DSPConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def with_overrides(self, **kwargs: Any) -> DSPConfig:
        return replace(self, **kwargs)

    def diff(self, other: DSPConfig) -> list[str]:
        """Human-readable list of fields that differ (``self`` = stored, ``other`` = current)."""
        out = []
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if isinstance(a, float) or isinstance(b, float):
                if abs(float(a) - float(b)) > 1e-6:
                    out.append(f"{f.name}: stored={a} current={b}")
            elif a != b:
                out.append(f"{f.name}: stored={a} current={b}")
        return out

    def require_compatible(self, current: DSPConfig, what: str = "model") -> None:
        """Raise :class:`ConfigMismatchError` if ``current`` differs from this stored config."""
        differences = self.diff(current)
        if differences:
            raise ConfigMismatchError(
                f"The {what} was created with different audio/DSP settings than the current "
                "calibration:\n  - "
                + "\n  - ".join(differences)
                + "\nFix: re-run calibration with the original settings (doctor.py), or retrain "
                "the model (train.py). Changing carrier/sample rate invalidates trained models."
            )


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AudioConfig:
    """Runtime audio-engine settings."""

    #: Linear output amplitude of the carrier tone (0..1). Keep LOW: some people and
    #: pets can hear 17-19 kHz.
    volume: float = 0.3
    fade_seconds: float = 0.05
    buffer_seconds: float = 6.0
    blocksize: int = 1024
    latency: str = "low"

    def __post_init__(self) -> None:
        if not 0.0 <= self.volume <= 1.0:
            raise ConfigError("volume must be within [0, 1]")


# --------------------------------------------------------------------------- #
# Calibration / doctor
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CalibrationConfig:
    candidate_carriers: tuple[float, ...] = (
        17000.0,
        17500.0,
        18000.0,
        18500.0,
        19000.0,
        19500.0,
        20000.0,
        20500.0,
        21000.0,
    )
    settle_seconds: float = 0.3
    measure_seconds: float = 0.8
    stability_seconds: float = 4.0
    snr_fail_db: float = 10.0
    snr_warn_db: float = 20.0
    #: Prefer the highest (least audible) carrier within this many dB of the best SNR.
    prefer_high_within_db: float = 3.0
    dropout_db: float = 10.0
    drift_db: float = 6.0
    clip_level: float = 0.99


# --------------------------------------------------------------------------- #
# Enrollment
# --------------------------------------------------------------------------- #
IDLE_SUBTYPES: tuple[tuple[str, str], ...] = (
    ("silence", "Stay still. Keep hands off the keyboard (silence / ambient)."),
    ("typing", "Type normally on the keyboard as you usually would."),
    ("head_movement", "Move your head, lean in and out, shift in your seat."),
    ("walking_by", "Have someone walk past behind the laptop (or walk past yourself)."),
    ("ambient", "Normal background: talk, play music, or just sit naturally."),
)

ROUND_VARIATIONS: tuple[str, ...] = (
    "Natural speed, about 15-20 cm from the screen, centred over the keyboard.",
    "FAST movements, close to the laptop (about 10 cm).",
    "SLOW movements, further away (30-40 cm).",
    "Hand slightly to the LEFT side of the keyboard.",
    "Hand slightly to the RIGHT side of the keyboard.",
    "Use your OTHER hand, natural speed.",
    "Sit back / change posture; vary hand height above the keyboard.",
    "Mix it up: change speed and distance on every clip.",
)


@dataclass(frozen=True)
class EnrollConfig:
    clips_per_round: int = 25
    countdown_seconds: int = 3
    #: Clips without at least this much Doppler motion (dB above static) are rejected.
    min_motion_db: float = 4.0
    #: Fraction of samples at/above full scale that marks a clip as clipped.
    max_clip_fraction: float = 0.001
    low_snr_db: float = 15.0
    review_seconds: float = 1.2


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AugmentConfig:
    gain_db: float = 3.0
    noise_db_range: tuple[float, float] = (-70.0, -45.0)
    noise_prob: float = 0.5
    stretch_range: tuple[float, float] = (0.85, 1.15)
    stretch_prob: float = 0.5


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-2
    patience: int = 8
    val_fraction: float = 0.2
    seed: int = 0
    embed_dim: int = 96
    width: int = 24
    dropout: float = 0.3
    #: Classes whose validation recall is below this are flagged as weak.
    weak_class_recall: float = 0.75
    #: Minimum augmented samples drawn per epoch (matters only for small profiles).
    min_samples_per_epoch: int = 512
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    num_threads: int = 0  # 0 = torch default


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InferenceConfig:
    hop_seconds: float = 0.1
    #: Motion gate threshold in dB. ``None`` = use the value learned at training time.
    gate_db: float | None = None
    default_gate_db: float = 6.0
    confidence: float = 0.55
    smoothing: int = 3
    cooldown_seconds: float = 1.2
    #: Keep classifying for this long after the gate closes.
    gate_hold_seconds: float = 0.3
    snr_interval_seconds: float = 1.0
    #: Below this carrier SNR the tone is considered lost (muted speaker, volume too low,
    #: OS noise suppression); classification pauses instead of guessing from noise.
    min_carrier_snr_db: float = 15.0


# --------------------------------------------------------------------------- #
# Gestures
# --------------------------------------------------------------------------- #
TIERS = ("core", "advanced", "custom")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


@dataclass(frozen=True)
class GestureSpec:
    name: str
    tier: str = "custom"
    enabled: bool = True
    #: Whether the gesture is expected to contain hand motion (False only for idle).
    motion: bool = True
    instruction: str = ""

    @property
    def experimental(self) -> bool:
        return self.tier in ("advanced", "custom")

    @property
    def label(self) -> str:
        suffix = " (experimental)" if self.experimental else ""
        return f"{self.name}{suffix}"


DEFAULT_GESTURES: tuple[GestureSpec, ...] = (
    GestureSpec("idle", "core", True, False, "Do NOT gesture. Behave naturally (prompted per sub-type)."),
    GestureSpec("push", "core", True, True, "Move your open hand TOWARD the screen, then stop."),
    GestureSpec("pull", "core", True, True, "Move your open hand AWAY from the screen, then stop."),
    GestureSpec("wave", "core", True, True, "Wave your hand side to side 2-3 times in front of the screen."),
    GestureSpec("swipe_left", "advanced", False, True, "Swipe your hand from right to left across the screen."),
    GestureSpec("swipe_right", "advanced", False, True, "Swipe your hand from left to right across the screen."),
    GestureSpec("swipe_up", "advanced", False, True, "Swipe your hand upward in front of the screen."),
    GestureSpec("swipe_down", "advanced", False, True, "Swipe your hand downward in front of the screen."),
)


def _parse_gesture(raw: dict[str, Any]) -> GestureSpec:
    if not isinstance(raw, dict) or "name" not in raw:
        raise ConfigError(f"Each gesture entry needs at least a 'name': {raw!r}")
    name = str(raw["name"]).strip()
    if not _NAME_RE.match(name):
        raise ConfigError(f"Invalid gesture name {name!r}: use lowercase letters, digits and underscores")
    tier = str(raw.get("tier", "custom"))
    if tier not in TIERS:
        raise ConfigError(f"Gesture {name!r}: tier must be one of {TIERS}, got {tier!r}")
    return GestureSpec(
        name=name,
        tier=tier,
        enabled=bool(raw.get("enabled", True)),
        motion=bool(raw.get("motion", name != "idle")),
        instruction=str(raw.get("instruction", "")),
    )


def load_gestures(path: Path | None = None) -> tuple[GestureSpec, ...]:
    """Load gesture definitions from YAML (or return defaults if the file is missing)."""
    path = DEFAULT_GESTURES_FILE if path is None else Path(path)
    if not path.exists():
        logger.debug("No gestures file at %s; using built-in defaults", path)
        return DEFAULT_GESTURES
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    entries = data.get("gestures")
    if not isinstance(entries, list) or not entries:
        raise ConfigError(f"{path}: expected a non-empty 'gestures:' list")
    specs = tuple(_parse_gesture(e) for e in entries)
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ConfigError(f"{path}: duplicate gesture names in {names}")
    idle = [s for s in specs if s.name == "idle"]
    if not idle or not idle[0].enabled:
        raise ConfigError(f"{path}: an enabled 'idle' gesture is required (it is the reject class)")
    if idle[0].motion:
        raise ConfigError(f"{path}: 'idle' must have motion: false")
    return specs


def enabled_gestures(specs: tuple[GestureSpec, ...] | None = None) -> tuple[GestureSpec, ...]:
    specs = load_gestures() if specs is None else specs
    return tuple(s for s in specs if s.enabled)


def gesture_map(specs: tuple[GestureSpec, ...]) -> dict[str, GestureSpec]:
    return {s.name: s for s in specs}


def validate_profile_name(name: str) -> str:
    """Profile names become directory names, so keep them boring."""
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$", name or ""):
        raise ConfigError(f"Invalid profile name {name!r}: use letters, digits, '_', '-', '.' (max 64 chars)")
    return name
