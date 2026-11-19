"""False discovery rate: Benjamini-Hochberg thresholds from a null peak rate.

On one micrograph, with lambda(t) the expected number of null (false) distinct peaks at
or above t and N(t) the distinct peaks found at or above t, the false discovery rate of
a cut at t is estimated as lambda(t) / N(t) (pi_0 = 1: the null is taken over the whole
searched area, so the estimate is slightly conservative). The Benjamini-Hochberg step-up
rule keeps the k* highest peaks,

    k* = max{k : lambda(z_(k)) <= q k},   z_(1) >= z_(2) >= ...,

and its threshold is z_(k*) (:func:`bh_cutoff`). The rule needs lambda(t) where the
peaks are, typically a few z below the one-false-positive threshold:

* :func:`nominal_null_rate`: the independent-trials formula N Q(t). It over-counts false
  peaks (the search's trials are correlated), so the realised rate is below q;
* :func:`null_rate` of a :class:`~.empirical_null.NullPeaks` (phase-randomised copies),
  an :class:`~.analytic_null.AnalyticNullPrediction`, or a
  :class:`~.analytic_null.RefinedNullPrediction` (for refined z-scores of the search's
  candidates).

Engine-agnostic: the peaks and the null can come from any search (Panther-EM uses it).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import ndtr  # pylint: disable=no-name-in-module

from .empirical_null import NullPeaks

NullRate = Callable[[np.ndarray], np.ndarray]
"""Expected null distinct peaks at or above each z-score, per micrograph."""


def nominal_null_rate(n_points: float) -> NullRate:
    """The independent-trials null, lambda(t) = N Q(t).

    Parameters
    ----------
    n_points : float
        Search points: valid positions x hypotheses per position.
    """

    def rate(z: np.ndarray) -> np.ndarray:
        return float(n_points) * ndtr(-np.asarray(z, float))

    return rate


def _curve_rate(thresholds: np.ndarray, rate: np.ndarray, scale: float) -> NullRate:
    """Log-linear interpolation of a decreasing rate curve.

    Infinite below the curve's first threshold (no picks there); the last value above
    its last one.
    """
    t = np.asarray(thresholds, float)
    log_rate = np.log(np.maximum(np.asarray(rate, float), 1e-300))

    def interpolated(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, float)
        out = scale * np.exp(np.interp(z, t, log_rate))
        return np.where(z < t[0] - 1e-9, np.inf, out)

    return interpolated


def null_rate(source: Any, area_scale: float = 1.0) -> NullRate:
    """lambda(t) per micrograph from a null.

    Parameters
    ----------
    source : NullPeaks | AnalyticNullPrediction | RefinedNullPrediction | tuple
        * :class:`~.empirical_null.NullPeaks`: the pooled null peaks at or above each
          t per copy area (a step function; infinite below the copies' peak floor,
          zero above their highest peak);
        * a prediction (any object with ``thresholds`` and ``rate``), or a
          ``(thresholds, rate)`` pair: log-linear between thresholds, infinite below
          the first.
    area_scale : float
        Multiplies the rate: the micrograph's searched area over the null's, when
        they differ (e.g. another mask). Default 1.

    Returns
    -------
    NullRate
    """
    if isinstance(source, NullPeaks):
        pooled = np.sort(np.concatenate(source.zscores))[::-1]
        unit = float(np.mean(source.valid_areas))
        per_peak = area_scale * unit / float(sum(source.valid_areas))
        floor = float(source.cutoff)

        def counted(z: np.ndarray) -> np.ndarray:
            z = np.asarray(z, float)
            n = np.searchsorted(-pooled, -z, side="right")
            return np.where(z < floor - 1e-9, np.inf, per_peak * n)

        return counted
    if isinstance(source, tuple) and len(source) == 2:
        return _curve_rate(np.asarray(source[0]), np.asarray(source[1]), area_scale)
    if hasattr(source, "thresholds") and hasattr(source, "rate"):
        return _curve_rate(
            np.asarray(source.thresholds), np.asarray(source.rate), area_scale
        )
    raise TypeError(f"No null rate from {type(source).__name__}.")


@dataclass(frozen=True)
class FdrCutoff:  # pylint: disable=too-many-instance-attributes
    """A Benjamini-Hochberg cut on one micrograph's peaks.

    Attributes
    ----------
    q : float
        Target false discovery rate.
    threshold : float | None
        The applied cut: the BH threshold, raised to ``floor`` if below it. None: no
        peak qualifies (no picks).
    bh_threshold : float | None
        The BH threshold z_(k*) before the floor.
    floor : float | None
        The lowest cut allowed (e.g. where the peaks or the null stop).
    capped : bool
        Whether the BH threshold was below the floor.
    n_peaks : int
        Peaks considered.
    n_picks : int
        Peaks at or above the applied cut.
    expected_false : float | None
        lambda at the applied cut: the expected false picks.
    """

    q: float
    threshold: float | None
    bh_threshold: float | None
    floor: float | None
    capped: bool
    n_peaks: int
    n_picks: int
    expected_false: float | None

    @property
    def estimated_fdr(self) -> float | None:
        """Expected false picks over picks at the applied cut."""
        if self.threshold is None or self.n_picks == 0 or self.expected_false is None:
            return None
        return self.expected_false / self.n_picks

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly dict."""
        return {
            "q": self.q,
            "threshold": self.threshold,
            "bh_threshold": self.bh_threshold,
            "floor": self.floor,
            "capped": self.capped,
            "n_peaks": self.n_peaks,
            "n_picks": self.n_picks,
            "expected_false": self.expected_false,
            "estimated_fdr": self.estimated_fdr,
        }


def bh_cutoff(
    peak_z: Sequence[float] | np.ndarray,
    rate: NullRate,
    q: float,
    *,
    floor: float | None = None,
) -> FdrCutoff:
    """Benjamini-Hochberg cut of one micrograph's distinct peaks at FDR ``q``.

    Parameters
    ----------
    peak_z : Sequence[float] | np.ndarray
        The micrograph's distinct-peak z-scores (any order; non-finite ignored).
    rate : NullRate
        Expected null distinct peaks >= z per micrograph (:func:`null_rate`,
        :func:`nominal_null_rate`).
    q : float
        False discovery rate, in (0, 1).
    floor : float | None
        Lowest cut allowed: a BH threshold below it is raised to it (``capped``).

    Returns
    -------
    FdrCutoff
    """
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0, 1).")
    z = np.asarray(peak_z, float).reshape(-1)
    z = np.sort(z[np.isfinite(z)])[::-1]
    t_bh: float | None = None
    if z.size:
        k = np.arange(1, z.size + 1)
        ok = np.flatnonzero(np.asarray(rate(z), float) <= q * k)
        if ok.size:
            t_bh = float(z[ok[-1]])
    capped = False
    threshold = t_bh
    if t_bh is not None and floor is not None and t_bh < floor:
        capped, threshold = True, float(floor)
    n_picks = 0 if threshold is None else int((z >= threshold).sum())
    expected = (
        None
        if threshold is None
        else float(np.asarray(rate(np.array([threshold])), float)[0])
    )
    return FdrCutoff(
        q=float(q),
        threshold=threshold,
        bh_threshold=t_bh,
        floor=None if floor is None else float(floor),
        capped=bool(capped),
        n_peaks=int(z.size),
        n_picks=n_picks,
        expected_false=expected,
    )
