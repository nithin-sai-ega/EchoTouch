"""Doppler signal processing.

Pipeline (identical at training and inference time)::

    raw audio (C, N) --Hann STFT--> complex (C, n_fft/2+1, T)
        --crop carrier +/- crop_hz--> complex (C, F, T)          [stft_crop]
        --|.|, log--> log-magnitude
        --subtract per-frequency background (median over time)-->
        --per-sample robust normalization--> float32 (C, F, T)   [features_from_stft]

Working in the cropped *complex* STFT domain lets training augment cheaply and
exactly: time shifts are frame slices, gain is a complex multiply, and white
noise is complex Gaussian (the STFT is linear). We never resample raw audio for
time-stretching because that would also shift the carrier frequency and destroy
the Doppler information; stretching happens on the time axis of the magnitude.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field

import numpy as np

from config import AugmentConfig, DSPConfig, EnrollConfig

logger = logging.getLogger(__name__)

_EPS = 1e-10


class DSPError(ValueError):
    """Raised for audio that cannot be processed with the given DSP settings."""


# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #
def as_channels_first(audio: np.ndarray) -> np.ndarray:
    """Return ``audio`` as float32 ``(channels, samples)``.

    Accepts ``(samples,)``, ``(samples, channels)`` (sounddevice layout) or
    ``(channels, samples)`` when channels < samples.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        return x[None, :]
    if x.ndim != 2:
        raise DSPError(f"audio must be 1-D or 2-D, got shape {x.shape}")
    if x.shape[0] > x.shape[1]:  # (samples, channels)
        x = x.T
    return np.ascontiguousarray(x)


def _frame(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Frame ``(C, N)`` into ``(C, T, n_fft)`` without copying."""
    n = x.shape[-1]
    t = 1 + (n - n_fft) // hop
    c_stride, s_stride = x.strides
    return np.lib.stride_tricks.as_strided(
        x, shape=(x.shape[0], t, n_fft), strides=(c_stride, hop * s_stride, s_stride), writeable=False
    )


def crop_slice(cfg: DSPConfig) -> slice:
    """Slice of rFFT bins kept around the carrier (always ``cfg.freq_bins`` long)."""
    lo = cfg.carrier_bin - cfg.half_bins
    hi = cfg.carrier_bin + cfg.half_bins + 1
    if lo < 0 or hi > cfg.n_fft // 2 + 1:
        raise DSPError("carrier crop falls outside the spectrum")
    return slice(lo, hi)


def stft(audio: np.ndarray, cfg: DSPConfig) -> np.ndarray:
    """Full Hann-window STFT, ``(C, n_fft//2 + 1, T)`` complex64 (no centre padding)."""
    x = as_channels_first(audio)
    if x.shape[1] < cfg.n_fft:
        raise DSPError(f"need at least n_fft={cfg.n_fft} samples, got {x.shape[1]}")
    frames = _frame(x, cfg.n_fft, cfg.hop) * np.hanning(cfg.n_fft).astype(np.float32)
    spec = np.fft.rfft(frames, axis=-1)  # (C, T, bins)
    return np.ascontiguousarray(np.swapaxes(spec, 1, 2)).astype(np.complex64)


def stft_crop(audio: np.ndarray, cfg: DSPConfig) -> np.ndarray:
    """Carrier-relative complex STFT, ``(C, F, T)`` with ``F = cfg.freq_bins``.

    Only the cropped bins are computed (a small DFT matrix product), which is far
    cheaper than a full FFT when F << n_fft.
    """
    x = as_channels_first(audio)
    if x.shape[0] != cfg.channels:
        raise DSPError(f"expected {cfg.channels} channel(s), got {x.shape[0]}")
    if x.shape[1] < cfg.n_fft:
        raise DSPError(f"need at least n_fft={cfg.n_fft} samples, got {x.shape[1]}")
    sl = crop_slice(cfg)
    basis = _crop_basis(cfg.n_fft, sl.start, sl.stop)  # (n_fft, F)
    frames = _frame(x, cfg.n_fft, cfg.hop)  # (C, T, n_fft)
    out = frames @ basis  # (C, T, F)
    return np.ascontiguousarray(np.swapaxes(out, 1, 2)).astype(np.complex64)


@functools.lru_cache(maxsize=8)
def _crop_basis(n_fft: int, lo: int, hi: int) -> np.ndarray:
    """Hann-windowed DFT basis for bins ``lo..hi-1`` (cached, read-only)."""
    n = np.arange(n_fft)[:, None]
    k = np.arange(lo, hi)[None, :]
    basis = (np.hanning(n_fft)[:, None] * np.exp(-2j * np.pi * n * k / n_fft)).astype(np.complex64)
    basis.setflags(write=False)
    return basis


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def log_magnitude(z_or_mag: np.ndarray) -> np.ndarray:
    mag = np.abs(z_or_mag) if np.iscomplexobj(z_or_mag) else z_or_mag
    return np.log(mag.astype(np.float32) + _EPS)


def subtract_background(logmag: np.ndarray, mode: str) -> np.ndarray:
    """Remove the static background (carrier leakage + static reflections).

    The background is estimated per frequency row as the median (or mean) over
    time, in the log domain. Static components become ~0; moving reflections
    (Doppler-shifted energy) stand out as positive values.
    """
    if mode == "none":
        return logmag
    if mode == "median":
        bg = np.median(logmag, axis=-1, keepdims=True)
    elif mode == "mean":
        bg = np.mean(logmag, axis=-1, keepdims=True)
    else:
        raise DSPError(f"unknown background mode {mode!r}")
    return logmag - bg


def normalize(x: np.ndarray, mode: str) -> np.ndarray:
    """Per-sample normalization.

    ``robust`` centres on the median and scales by the MAD, so the noise floor has
    unit scale while real motion keeps a large value (motion strength is
    preserved, unlike z-scoring a clip that is mostly motion).
    """
    if mode == "zscore":
        return (x - x.mean()) / (x.std() + 1e-6)
    if mode == "robust":
        med = np.median(x)
        mad = np.median(np.abs(x - med)) * 1.4826
        return np.clip((x - med) / (mad + 1e-6), -10.0, 50.0)
    raise DSPError(f"unknown normalization {mode!r}")


def features_from_mag(mag: np.ndarray, cfg: DSPConfig) -> np.ndarray:
    """Magnitude ``(C, F, T)`` -> normalized float32 features ``(C, F, T)``."""
    x = subtract_background(log_magnitude(mag), cfg.background)
    return normalize(x, cfg.normalization).astype(np.float32)


def features_from_stft(z: np.ndarray, cfg: DSPConfig) -> np.ndarray:
    """Complex cropped STFT ``(C, F, T)`` -> normalized float32 features ``(C, F, T)``."""
    return features_from_mag(np.abs(z), cfg)


def extract_features(audio: np.ndarray, cfg: DSPConfig) -> np.ndarray:
    """Raw audio -> model input ``(C, F, T)`` (deterministic)."""
    return features_from_stft(stft_crop(audio, cfg), cfg)


def window_features(audio: np.ndarray, cfg: DSPConfig) -> np.ndarray:
    """Features for the most recent ``cfg.window_seconds`` of ``audio``."""
    x = as_channels_first(audio)
    n = cfg.samples(cfg.window_seconds)
    if x.shape[1] < n:
        raise DSPError(f"need {n} samples for one window, got {x.shape[1]}")
    return extract_features(x[:, -n:], cfg)


def display_spectrogram(audio: np.ndarray, cfg: DSPConfig, floor_db: float = 0.0) -> np.ndarray:
    """Background-subtracted carrier-band spectrogram in dB for visualization ``(F, T)``.

    Channels are averaged. Values are dB above the static background (clipped at
    ``floor_db``), so the plot is dark when nothing moves and lights up with motion.
    """
    z = stft_crop(audio, cfg)
    power_db = 20.0 * np.log10(np.abs(z).mean(axis=0) + _EPS)
    rel = power_db - np.median(power_db, axis=-1, keepdims=True)
    return np.maximum(rel, floor_db).astype(np.float32)


# --------------------------------------------------------------------------- #
# Motion, SNR and quality metrics
# --------------------------------------------------------------------------- #
def sideband_ratio_db(z: np.ndarray, cfg: DSPConfig) -> np.ndarray:
    """Per-frame sideband/carrier power ratio in dB, ``(T,)``.

    Sideband = cropped bins further than ``guard_hz`` from the carrier, i.e.
    Doppler-shifted energy. Dividing by the carrier power makes the metric
    independent of volume and mic gain.
    """
    power = (np.abs(z) ** 2).sum(axis=0)  # (F, T), summed over channels
    centre, guard = cfg.half_bins, cfg.guard_bins
    carrier = power[centre - guard : centre + guard + 1].sum(axis=0)
    side = power.sum(axis=0) - carrier
    return 10.0 * np.log10((side + _EPS) / (carrier + _EPS))


def motion_score_db(z: np.ndarray, cfg: DSPConfig) -> float:
    """How far (dB) the strongest-motion frame rises above the window's static level."""
    r = sideband_ratio_db(z, cfg)
    return float(r.max() - np.median(r))


def doppler_direction(z: np.ndarray, cfg: DSPConfig) -> float:
    """Signed balance of above- vs below-carrier excess energy in ``[-1, 1]``.

    Positive = approaching (push), negative = receding (pull). For diagnostics
    only; the classifier learns this itself.
    """
    power = (np.abs(z) ** 2).sum(axis=0)
    excess = np.maximum(power - np.median(power, axis=-1, keepdims=True), 0.0)
    centre, guard = cfg.half_bins, cfg.guard_bins
    up = excess[centre + guard + 1 :].sum()
    down = excess[: centre - guard].sum()
    return float((up - down) / (up + down + _EPS))


def snr_at_frequency(
    audio: np.ndarray,
    sample_rate: int,
    freq_hz: float,
    n_fft: int = 8192,
    exclude_hz: float = 150.0,
    band_hz: float = 1500.0,
) -> float:
    """SNR (dB) of a tone at ``freq_hz`` vs the median noise floor around it.

    Uses an averaged Hann periodogram. Noise is the median bin power within
    ``band_hz`` of the tone, excluding +/- ``exclude_hz`` (tone + window leakage).
    Returns a large negative number for silent input.
    """
    x = as_channels_first(audio).mean(axis=0)
    if x.size < n_fft:
        n_fft = 1 << int(np.floor(np.log2(max(x.size, 256))))
    frames = _frame(x[None, :], n_fft, n_fft // 2)[0]
    psd = (np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=-1)) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    dist = np.abs(freqs - freq_hz)
    tone = psd[dist <= 2.5 * sample_rate / n_fft].max(initial=0.0)
    noise_mask = (dist > exclude_hz) & (dist <= band_hz)
    if not noise_mask.any():
        raise DSPError("no noise bins available for SNR estimate")
    noise = np.median(psd[noise_mask])
    return float(10.0 * np.log10((tone + _EPS) / (noise + _EPS)))


def rms_dbfs(audio: np.ndarray) -> float:
    x = np.asarray(audio, dtype=np.float64)
    return float(20.0 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12))


def clipping_fraction(audio: np.ndarray, level: float = 0.99) -> float:
    x = np.asarray(audio)
    return float(np.mean(np.abs(x) >= level)) if x.size else 0.0


@dataclass
class QualityReport:
    """Result of checking one enrollment clip."""

    accepted: bool
    snr_db: float
    motion_db: float
    clipping: float
    direction: float
    flags: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "snr_db": round(self.snr_db, 2),
            "motion_db": round(self.motion_db, 2),
            "clipping": round(self.clipping, 5),
            "direction": round(self.direction, 3),
            "flags": list(self.flags),
        }


def quality_check(
    audio: np.ndarray, cfg: DSPConfig, expects_motion: bool, thresholds: EnrollConfig | None = None
) -> QualityReport:
    """Check a recorded clip: clipping, carrier SNR, and (for gestures) Doppler motion."""
    th = thresholds or EnrollConfig()
    x = as_channels_first(audio)
    z = stft_crop(x, cfg)
    snr = snr_at_frequency(x, cfg.sample_rate, cfg.carrier_hz)
    motion = motion_score_db(z, cfg)
    clip = clipping_fraction(x)
    rep = QualityReport(True, snr, motion, clip, doppler_direction(z, cfg))
    if clip > th.max_clip_fraction:
        rep.accepted = False
        rep.flags.append("clipping")
        rep.messages.append("Clipping detected: lower the carrier volume or mic gain.")
    if snr < th.low_snr_db:
        rep.flags.append("low_snr")
        rep.messages.append(f"Low carrier SNR ({snr:.0f} dB): check volume / run doctor.py.")
    if expects_motion and motion < th.min_motion_db:
        rep.accepted = False
        rep.flags.append("no_motion")
        rep.messages.append(
            f"No clear Doppler motion detected ({motion:.1f} dB < {th.min_motion_db:.1f} dB). "
            "Move a bit closer / faster."
        )
    return rep


# --------------------------------------------------------------------------- #
# Augmentation (training only)
# --------------------------------------------------------------------------- #
def stretch_time(mag: np.ndarray, out_frames: int) -> np.ndarray:
    """Linearly resample the time axis of ``(C, F, T)`` to ``out_frames`` frames."""
    t_in = mag.shape[-1]
    if t_in == out_frames:
        return mag
    src = np.linspace(0.0, t_in - 1, out_frames)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, t_in - 1)
    w = (src - lo).astype(mag.dtype)
    return mag[..., lo] * (1 - w) + mag[..., hi] * w


def noise_std_for_level(z: np.ndarray, level_db: float, n_fft: int) -> float:
    """Complex STFT noise std equivalent to white noise ``level_db`` below the carrier amplitude.

    For white noise of std s and a Hann window, each STFT bin is complex Gaussian
    with variance s^2 * sum(w^2) = s^2 * 3 n_fft / 8.
    """
    carrier_amp = float(np.abs(z).max()) / (n_fft / 4.0)  # Hann coherent gain = n_fft/2 * 1/2
    s = carrier_amp * 10 ** (level_db / 20.0)
    return s * np.sqrt(3.0 * n_fft / 8.0)


def augment_features(z_clip: np.ndarray, cfg: DSPConfig, aug: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    """Random training view of one clip: time crop, gain, noise, time-stretch -> features.

    ``z_clip`` is the cropped complex STFT of the full clip ``(C, F, T_clip)``.
    No frequency-axis operations are applied (they would destroy Doppler meaning).
    """
    t_clip, t_win = z_clip.shape[-1], cfg.window_frames
    stretch = 1.0
    if rng.random() < aug.stretch_prob:
        stretch = float(rng.uniform(*aug.stretch_range))
    src_len = int(np.clip(round(t_win / stretch), 4, t_clip))
    start = int(rng.integers(0, t_clip - src_len + 1))
    z = z_clip[..., start : start + src_len]
    z = z * np.float32(10 ** (rng.uniform(-aug.gain_db, aug.gain_db) / 20.0))
    if rng.random() < aug.noise_prob:
        std = noise_std_for_level(z, float(rng.uniform(*aug.noise_db_range)), cfg.n_fft)
        noise = rng.normal(0.0, std / np.sqrt(2), size=z.shape + (2,)).astype(np.float32)
        z = z + (noise[..., 0] + 1j * noise[..., 1])
    mag = stretch_time(np.abs(z), t_win)
    return features_from_mag(mag, cfg)


def center_features(z_clip: np.ndarray, cfg: DSPConfig) -> np.ndarray:
    """Deterministic (validation) view: centre ``window_frames`` crop of the clip."""
    t_clip, t_win = z_clip.shape[-1], cfg.window_frames
    if t_clip < t_win:
        raise DSPError(f"clip has {t_clip} frames, window needs {t_win}")
    start = (t_clip - t_win) // 2
    return features_from_stft(z_clip[..., start : start + t_win], cfg)
