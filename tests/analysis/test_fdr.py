"""Tests for the Benjamini-Hochberg false discovery rate cut."""

import numpy as np
import pytest
from scipy.special import ndtr

from leopard_em.analysis import (
    NullPeaks,
    bh_cutoff,
    nominal_null_rate,
    null_rate,
)


def _step_up(z, rate, q):
    """Benjamini-Hochberg by the definition: the largest k with rate(z_(k)) <= q k."""
    z = np.sort(np.asarray(z, float))[::-1]
    best = None
    for k in range(1, z.size + 1):
        if rate(np.array([z[k - 1]]))[0] <= q * k:
            best = k
    return None if best is None else float(z[best - 1])


def _exponential_rate(scale=2000.0, slope=6.0, start=6.0):
    def rate(z):
        return scale * np.exp(-slope * (np.asarray(z, float) - start))

    return rate


def test_bh_cutoff_is_the_step_up_rule():
    rng = np.random.default_rng(0)
    rate = _exponential_rate()
    for q in (0.01, 0.05, 0.2):
        z = np.concatenate([6 + rng.exponential(1 / 6, 2000), rng.normal(9, 0.8, 300)])
        cut = bh_cutoff(z, rate, q)
        assert cut.bh_threshold == _step_up(z, rate, q)
        assert cut.n_picks == int((z >= cut.threshold).sum())
        assert cut.expected_false == pytest.approx(rate(np.array([cut.threshold]))[0])
        assert cut.estimated_fdr <= q + 1e-12


def test_realised_fdr_is_at_most_q():
    """Null peaks from a Poisson process with a known rate, true peaks above: the
    mean false discovery proportion of the BH picks is at most q."""
    rng = np.random.default_rng(1)
    rate = _exponential_rate(scale=3000.0)
    q = 0.05
    fdp = []
    for _ in range(300):
        n_null = rng.poisson(rate(np.array([6.0]))[0])
        null = 6.0 + rng.exponential(1 / 6, n_null)  # lambda(t) = 3000 exp(-6 (t - 6))
        true = rng.normal(7.6, 0.6, 150)
        cut = bh_cutoff(np.concatenate([null, true]), rate, q)
        if cut.threshold is not None and cut.n_picks:
            fdp.append((null >= cut.threshold).sum() / cut.n_picks)
    assert 0.0 < np.mean(fdp) <= q + 0.005


def test_no_peak_qualifies_and_the_floor():
    rate = _exponential_rate()
    cut = bh_cutoff([6.1, 6.2], rate, 0.05)
    assert cut.threshold is None and cut.n_picks == 0 and cut.estimated_fdr is None
    z = np.concatenate([np.full(50, 9.0), np.linspace(7.0, 8.9, 400)])
    free = bh_cutoff(z, rate, 0.2)
    capped = bh_cutoff(z, rate, 0.2, floor=free.bh_threshold + 0.3)
    assert capped.capped and capped.threshold == pytest.approx(free.bh_threshold + 0.3)
    assert capped.n_picks < free.n_picks
    with pytest.raises(ValueError):
        bh_cutoff(z, rate, 1.0)


def test_null_rate_sources():
    assert nominal_null_rate(1e9)(np.array([7.0]))[0] == pytest.approx(1e9 * ndtr(-7))
    rng = np.random.default_rng(2)
    peaks = NullPeaks(
        zscores=[np.sort(6 + rng.exponential(0.2, 400))[::-1] for _ in range(4)],
        valid_areas=[1000, 1000, 1200, 800],
        cutoff=6.0,
        mask_radius=10.0,
    )
    counted = null_rate(peaks)
    t = np.array([6.0, 6.3, 6.8])
    np.testing.assert_allclose(counted(t), peaks.rate(t).rate)
    assert np.isinf(counted(np.array([5.9]))[0])
    assert counted(np.array([20.0]))[0] == 0.0
    np.testing.assert_allclose(null_rate(peaks, area_scale=2.0)(t), 2 * counted(t))
    grid = np.round(np.arange(5.0, 9.0, 0.01), 3)
    curve = 500 * np.exp(-5 * (grid - 6))
    predicted = null_rate((grid, curve))
    assert predicted(np.array([6.505]))[0] == pytest.approx(500 * np.exp(-2.525))
    assert np.isinf(predicted(np.array([4.0]))[0])

    class Prediction:  # pylint: disable=too-few-public-methods
        thresholds = grid
        rate = curve

    assert null_rate(Prediction())(np.array([7.0]))[0] == pytest.approx(
        500 * np.exp(-5)
    )
    with pytest.raises(TypeError):
        null_rate(3.0)
