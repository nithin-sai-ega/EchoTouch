import numpy as np
import pytest

import dsp
from config import AugmentConfig, DSPConfig
from simulator import generate_clip
from tests.conftest import tone


def test_stft_crop_matches_full_stft(dsp_cfg: DSPConfig):
    x = tone(dsp_cfg, 1.0)
    full = dsp.stft(x, dsp_cfg)
    crop = dsp.stft_crop(x, dsp_cfg)
    sl = dsp.crop_slice(dsp_cfg)
    assert crop.shape == (1, dsp_cfg.freq_bins, dsp_cfg.window_frames)
    np.testing.assert_allclose(np.abs(crop), np.abs(full[:, sl, :]), rtol=1e-3, atol=1e-3)


def test_carrier_is_centred(dsp_cfg: DSPConfig):
    z = dsp.stft_crop(tone(dsp_cfg, 1.0), dsp_cfg)
    peak = np.abs(z[0]).mean(axis=1).argmax()
    assert peak == dsp_cfg.half_bins


@pytest.mark.parametrize("sr", [44100, 48000])
def test_feature_shape(sr: int):
    cfg = DSPConfig(sample_rate=sr)
    f = dsp.extract_features(tone(cfg, cfg.window_seconds), cfg)
    assert f.shape == (1, cfg.freq_bins, cfg.window_frames)
    assert f.dtype == np.float32


def test_multichannel_shapes():
    cfg = DSPConfig(channels=2)
    x = np.stack([tone(cfg, 1.0, seed=0), tone(cfg, 1.0, seed=1)], axis=1)  # (n, 2) sounddevice layout
    assert dsp.extract_features(x, cfg).shape == (2, cfg.freq_bins, cfg.window_frames)
    with pytest.raises(dsp.DSPError):
        dsp.stft_crop(tone(cfg, 1.0), cfg)  # wrong channel count


def test_background_subtraction_removes_static_tone(dsp_cfg: DSPConfig):
    z = dsp.stft_crop(tone(dsp_cfg, 1.0, noise=1e-4), dsp_cfg)
    raw = dsp.log_magnitude(z)
    sub = dsp.subtract_background(raw, "median")
    # the carrier row dominates before, but is ~0 after background subtraction
    assert raw[0, dsp_cfg.half_bins].mean() > raw.mean() + 5
    assert abs(np.median(sub[0, dsp_cfg.half_bins])) < 1e-5
    assert np.abs(np.median(sub, axis=-1)).max() < 1e-5


def test_determinism(dsp_cfg: DSPConfig):
    x = tone(dsp_cfg, 1.2)
    np.testing.assert_array_equal(dsp.window_features(x, dsp_cfg), dsp.window_features(x.copy(), dsp_cfg))


def test_too_short_raises(dsp_cfg: DSPConfig):
    with pytest.raises(dsp.DSPError):
        dsp.stft_crop(np.zeros(100, np.float32), dsp_cfg)


def test_doppler_direction_and_motion(dsp_cfg: DSPConfig):
    rng = np.random.default_rng(0)
    push, _ = generate_clip("push", dsp_cfg, rng)
    pull, _ = generate_clip("pull", dsp_cfg, rng)
    idle, _ = generate_clip("idle", dsp_cfg, rng, idle_subtype="silence")
    zp, zl, zi = (dsp.stft_crop(a, dsp_cfg) for a in (push, pull, idle))
    assert dsp.doppler_direction(zp, dsp_cfg) > 0.3  # approaching -> above carrier
    assert dsp.doppler_direction(zl, dsp_cfg) < -0.3  # receding -> below carrier
    assert dsp.motion_score_db(zp, dsp_cfg) > dsp.motion_score_db(zi, dsp_cfg) + 3


def test_snr_estimate(dsp_cfg: DSPConfig):
    loud = dsp.snr_at_frequency(tone(dsp_cfg, 1.0, noise=1e-3), dsp_cfg.sample_rate, dsp_cfg.carrier_hz)
    quiet = dsp.snr_at_frequency(tone(dsp_cfg, 1.0, amp=1e-3, noise=1e-3), dsp_cfg.sample_rate, dsp_cfg.carrier_hz)
    absent = dsp.snr_at_frequency(tone(dsp_cfg, 1.0, amp=0.0), dsp_cfg.sample_rate, dsp_cfg.carrier_hz)
    assert loud > quiet + 30
    assert absent < 15


def test_quality_check(dsp_cfg: DSPConfig):
    rng = np.random.default_rng(1)
    push, _ = generate_clip("push", dsp_cfg, rng)
    still = tone(dsp_cfg, dsp_cfg.clip_seconds)
    assert dsp.quality_check(push, dsp_cfg, expects_motion=True).accepted
    rep = dsp.quality_check(still, dsp_cfg, expects_motion=True)
    assert not rep.accepted and "no_motion" in rep.flags
    assert dsp.quality_check(still, dsp_cfg, expects_motion=False).accepted  # fine for idle
    clipped = np.clip(still * 20, -1, 1)
    rep = dsp.quality_check(clipped, dsp_cfg, expects_motion=False)
    assert not rep.accepted and "clipping" in rep.flags
    low = dsp.quality_check(tone(dsp_cfg, 1.5, amp=2e-4, noise=2e-3), dsp_cfg, expects_motion=False)
    assert "low_snr" in low.flags


def test_augmentation_shape_and_direction(dsp_cfg: DSPConfig):
    rng = np.random.default_rng(2)
    push, _ = generate_clip("push", dsp_cfg, rng)
    z = dsp.stft_crop(push, dsp_cfg)
    aug = AugmentConfig(noise_prob=1.0, stretch_prob=1.0)
    for seed in range(5):
        f = dsp.augment_features(z, dsp_cfg, aug, np.random.default_rng(seed))
        assert f.shape == (1, dsp_cfg.freq_bins, dsp_cfg.window_frames)
        assert np.isfinite(f).all()
        # No frequency flips: motion energy must stay above the carrier for a push.
        c = dsp_cfg.half_bins
        assert f[0, c + 3 :].max() > f[0, : c - 3].max()


def test_center_features_deterministic(dsp_cfg: DSPConfig):
    z = dsp.stft_crop(tone(dsp_cfg, dsp_cfg.clip_seconds), dsp_cfg)
    a, b = dsp.center_features(z, dsp_cfg), dsp.center_features(z, dsp_cfg)
    assert a.shape[-1] == dsp_cfg.window_frames
    np.testing.assert_array_equal(a, b)


def test_stretch_time():
    m = np.arange(10, dtype=np.float32)[None, None, :]
    out = dsp.stretch_time(m, 19)
    assert out.shape == (1, 1, 19)
    assert out[0, 0, 0] == 0 and out[0, 0, -1] == 9


def test_display_spectrogram_nonnegative(dsp_cfg: DSPConfig):
    s = dsp.display_spectrogram(tone(dsp_cfg, 2.0), dsp_cfg)
    assert s.shape[0] == dsp_cfg.freq_bins and (s >= 0).all()
