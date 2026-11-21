"""Tests for thresholds from phase-randomised null searches."""

import numpy as np
import pytest
import torch
from scipy.special import gammainc

from leopard_em.analysis import (
    NullPeaks,
    distinct_peaks,
    empirical_zscore_cutoff,
)
from leopard_em.analysis.empirical_null import (
    counted_rank,
    poisson_interval,
    valid_region,
)
from leopard_em.analysis.zscore_metric import find_peaks_from_zscore


def _smooth_field(shape, seed, sigma=2.0):
    rng = np.random.default_rng(seed)
    noise = np.fft.rfft2(rng.standard_normal(shape))
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.rfftfreq(shape[1])[None, :]
    field = np.fft.irfft2(noise * np.exp(-2 * (np.pi * sigma) ** 2 * (ky**2 + kx**2)))
    return (field / field.std()).astype(np.float32)


@pytest.mark.parametrize("radius", [3.0, 10.0])
def test_distinct_peaks_equal_find_peaks_from_zscore(radius):
    z = _smooth_field((160, 140), seed=2)
    peaks = distinct_peaks(z, 1.0, mask_radius=radius)
    pos_y, pos_x = find_peaks_from_zscore(torch.from_numpy(z), 1.0, radius)
    ours = set(zip(peaks.pos_y.tolist(), peaks.pos_x.tolist(), strict=False))
    theirs = set(zip(pos_y.tolist(), pos_x.tolist(), strict=False))
    assert ours == theirs
    assert np.all(np.diff(peaks.zscore) <= 0)
    assert peaks.valid_area == z.size


def test_cutoff_is_inclusive():
    z = np.zeros((30, 30), dtype=np.float32)
    z[10, 10] = 5.0
    z[20, 20] = 4.0
    assert distinct_peaks(z, 4.0, mask_radius=3).zscore.tolist() == [5.0, 4.0]


def test_excluded_pixels_neither_count_nor_suppress():
    z = np.zeros((40, 40), dtype=np.float32)
    z[2, 20] = 9.0  # inside the border
    z[6, 20] = 5.0  # valid, within the suppression radius of the border peak
    mask = np.zeros_like(z, dtype=bool)
    mask[30, 30] = True
    z[30, 30] = 8.0  # masked
    peaks = distinct_peaks(z, 1.0, mask_radius=10, border=4, exclude_mask=mask)
    assert peaks.zscore.tolist() == [5.0]
    assert peaks.valid_area == valid_region((40, 40), 4, mask).sum() == 32 * 32 - 1


def test_counted_rank_rules():
    assert counted_rank(5.0, "mean") == 5
    assert counted_rank(0.4, "mean") == 0
    assert counted_rank(2.0, "confidence") is None  # needs >= 2.30 at 0.9
    for allowed in (3.0, 10.0, 40.0):
        rank = counted_rank(allowed, "confidence")
        assert gammainc(rank, allowed) >= 0.9 > gammainc(rank + 1, allowed)
    with pytest.raises(ValueError, match="Unknown rule"):
        counted_rank(3.0, "median")  # type: ignore[arg-type]


def test_cutoff_is_the_mth_pooled_peak():
    nulls = [np.array([9.0, 7.0, 6.5]), np.array([8.0, 7.5, 6.0])]
    # R = 2: 1 FP per micrograph allows 2 pooled peaks -> the 2nd highest
    assert empirical_zscore_cutoff(nulls, 1.0) == 8.0
    # 0.25 FP allows 0.5 peaks -> just above the highest
    assert empirical_zscore_cutoff(nulls, 0.25) == np.nextafter(9.0, np.inf)
    with pytest.raises(ValueError, match="lower the peak floor"):
        empirical_zscore_cutoff(nulls, 5.0)


@pytest.mark.parametrize("rule", ["mean", "confidence"])
def test_cutoff_delivers_target_on_poisson_nulls(rule):
    """Null peaks above t at rate 1000 exp(-(t - 6) / 0.2) per micrograph."""
    rng = np.random.default_rng(0)
    scale, base_rate, target, copies = 0.2, 1000.0, 1.0, 10
    realised = []
    for _ in range(400):
        nulls = [
            6.0 + rng.exponential(scale, size=rng.poisson(base_rate))
            for _ in range(copies)
        ]
        t = empirical_zscore_cutoff(nulls, target, rule=rule)
        realised.append(base_rate * np.exp(-(t - 6.0) / scale))
    realised = np.asarray(realised)
    if rule == "mean":
        assert realised.mean() == pytest.approx(target, rel=0.05)
    else:
        assert np.mean(realised <= target) >= 0.87


def test_null_peaks_rate_and_round_trip():
    peaks = [
        distinct_peaks(_smooth_field((120, 120), seed=s), 2.0, border=5)
        for s in range(4)
    ]
    nulls = NullPeaks.from_peaks(peaks, seeds=[1, 2, 3, 4])
    curve = nulls.rate([2.0, 2.5, 3.0])
    assert curve.counts[0] == sum(len(p.zscore) for p in peaks)
    assert np.all(np.diff(curve.rate) <= 0)
    assert np.all(curve.rate_lo <= curve.rate) and np.all(curve.rate <= curve.rate_hi)
    assert curve.rate[0] == pytest.approx(curve.counts[0] / 4)
    double = nulls.rate([2.0], area=2 * peaks[0].valid_area)
    assert double.rate[0] == pytest.approx(2 * curve.rate[0])
    again = NullPeaks.from_dict(nulls.as_dict())
    assert again.zscore_cutoff(2.0) == nulls.zscore_cutoff(2.0)
    with pytest.raises(ValueError, match="peak floor"):
        nulls.rate([1.0])


def test_poisson_interval_known_values():
    lo, hi = poisson_interval(np.array([0, 10]), conf=0.95)
    np.testing.assert_allclose(lo, [0.0, 4.795389], atol=1e-5)
    np.testing.assert_allclose(hi, [3.688879, 18.390356], atol=1e-5)
