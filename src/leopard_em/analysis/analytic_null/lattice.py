"""Expected null peak counts from field correlations on the search lattice.

A search reports the maximum of the correlation field over a grid of search points at
each pixel, normalised per pixel, then keeps distinct peaks. With the field's
covariance known (:mod:`.field`), the expected number of null peaks at or above z per
micrograph follows in three parts:

1. **Lattice local maxima** (Worsley 2005, "discrete local maxima"): the expected
   number of grid points at or above u that are higher than both neighbours along
   every axis, :func:`lattice_maxima_rate`. Axes are treated as conditionally
   independent given the centre value; the correlations at one and two steps along
   each axis give the probability that both neighbours are lower.
2. **Crowding** (:func:`crowding_correction`): distinct peaks are kept only if no
   higher one lies within the suppression radius. For Poisson-distributed maxima the
   expected count is (1 - exp(-beta L)) / beta, beta = pi r^2 / area.
3. **Per-pixel normalisation** (:func:`normalization_factor`): the z-score divides by
   the standard deviation over the search at each pixel. A null peak raises its own
   pixel's standard deviation, and the standard deviation fluctuates between pixels;
   to first order in kappa (:func:`.field.estimate_kappa`) the tail is multiplied by
   F(u) = exp(-kappa (u^4 / 4 - u^2 / 2)).

The continuous (Euler characteristic) approximation of random field theory does not
apply to 2DTM grids: neighbouring grid points correlate at only 0.5-0.9, so most
continuous maxima fall between grid points. The lattice version does apply.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr, owens_t  # pylint: disable=no-name-in-module

from .field import LatticeSamples

Z_STEP = 0.002
Z_GRID = np.arange(2.0, 11.0, Z_STEP)
_CLIP = 0.999999


def _both_below(h: np.ndarray, r: np.ndarray) -> np.ndarray:
    """P(X < h, Y < h) for a standard bivariate normal with correlation r (h >= 0)."""
    a = np.sqrt(np.clip((1 - r) / (1 + r), 0, None))
    return ndtr(h) - 2 * owens_t(h, a)


def _axis_probability(
    z: np.ndarray, r1: np.ndarray, r2: np.ndarray, sided: np.ndarray
) -> np.ndarray:
    """P(neighbours along one axis are below the centre | centre = z).

    ``r1``, ``r2``, ``sided`` have shape (S,), ``z`` (Z,); returns (S, Z).
    """
    r1 = np.clip(r1, -_CLIP, _CLIP)[:, None]
    h = z[None, :] * np.sqrt((1 - r1) / (1 + r1))
    one = ndtr(h)
    conditional = np.clip((r2[:, None] - r1**2) / (1 - r1**2), -_CLIP, _CLIP)
    two = _both_below(h, conditional)
    sided = sided[:, None]
    return np.where(sided >= 2, two, np.where(sided >= 1, one, 1.0))


def _plane_axis(
    plane_correlation: np.ndarray | None, n_planes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One- and two-step correlations and sidedness along the defocus planes."""
    k = 0 if plane_correlation is None else plane_correlation.shape[0]
    r1 = np.zeros((k, n_planes))
    r2 = np.zeros((k, n_planes))
    sided = np.zeros((k, n_planes))
    if plane_correlation is None or n_planes < 2:
        return r1, r2, sided
    cf = plane_correlation
    for j in range(n_planes):
        lo, hi = j - 1, j + 1
        if lo >= 0 and hi < n_planes:
            r1[:, j] = 0.5 * (cf[:, j, lo] + cf[:, j, hi])
            r2[:, j] = cf[:, lo, hi]
            sided[:, j] = 2
        else:
            r1[:, j] = cf[:, j, lo if lo >= 0 else hi]
            sided[:, j] = 1
    return r1, r2, sided


def lattice_maxima_rate(
    samples: LatticeSamples,
    n_points: float,
    thresholds: np.ndarray,
    variance_scale: float | None = None,
) -> np.ndarray:
    """Expected number of lattice local maxima >= u over the search.

    Parameters
    ----------
    samples : LatticeSamples
        Field correlations at sampled grid points (:func:`.field.sample_lattice`).
    n_points : float
        Number of search points: valid pixels x orientations x planes.
    thresholds : np.ndarray
        Thresholds u (in units of the field's mean standard deviation).
    variance_scale : float | None
        Mean field variance to normalise the samples' variances by. Default None: the
        samples' own mean (so the mean variance is 1).

    Returns
    -------
    np.ndarray
        Expected lattice maxima >= u, same shape as ``thresholds``.
    """
    variance = samples.variance
    k, n_planes = variance.shape
    v = variance / (variance.mean() if variance_scale is None else variance_scale)
    pdf = np.exp(-(Z_GRID**2) / 2) / np.sqrt(2 * np.pi)
    prod = np.ones((k * n_planes, Z_GRID.size))
    for arr in samples.axes.values():
        prod *= _axis_probability(
            Z_GRID,
            arr[..., 0].reshape(-1),
            arr[..., 1].reshape(-1),
            arr[..., 2].reshape(-1),
        )
    r1, r2, sided = _plane_axis(samples.plane_correlation, n_planes)
    if r1.size:
        prod *= _axis_probability(
            Z_GRID, r1.reshape(-1), r2.reshape(-1), sided.reshape(-1)
        )
    density = pdf[None, :] * prod
    # Tail integral from each z to the top of the grid (trapezoid rule)
    pieces = (density[:, 1:] + density[:, :-1]) / 2 * Z_STEP
    tail = np.concatenate(
        [np.cumsum(pieces[:, ::-1], axis=1)[:, ::-1], np.zeros((k * n_planes, 1))],
        axis=1,
    )
    scaled = np.asarray(thresholds, float)[None, :] / np.sqrt(v.reshape(-1))[:, None]
    total = np.zeros(np.asarray(thresholds).shape, dtype=float)
    for i in range(k * n_planes):
        total += np.interp(scaled[i], Z_GRID, tail[i])
    return n_points * total / (k * n_planes)


def crowding_correction(
    rate: np.ndarray, area: float, mask_radius: float
) -> np.ndarray:
    """Distinct peaks left after suppression within ``mask_radius`` of a higher one.

    A maximum at height z survives if no higher maximum lies within the disc; for
    Poisson-distributed maxima the expected count above u is
    (1 - exp(-beta L(u))) / beta, beta = pi r^2 / area.
    """
    beta = np.pi * mask_radius**2 / area
    return (1 - np.exp(-beta * np.asarray(rate, float))) / beta


def normalization_factor(kappa: float, thresholds: np.ndarray) -> np.ndarray:
    """First-order effect of the per-pixel standard deviation on the null tail.

    F(u) = exp(-kappa (u^4 / 4 - u^2 / 2)): the pixel-to-pixel fluctuation of the
    standard deviation fattens the tail (+kappa u^4 / 4), and a null peak raising its
    own pixel's standard deviation thins it (-kappa u^2 (u^2 - 1) / 2).
    """
    u = np.asarray(thresholds, float)
    return np.exp(-kappa * (u**4 / 4 - u**2 / 2))


def nominal_rate(n_points: float, thresholds: np.ndarray) -> np.ndarray:
    """Independent-trials expectation N Q(u), the formula behind nominal thresholds."""
    return n_points * ndtr(-np.asarray(thresholds, float))


def crossing(thresholds: np.ndarray, rate: np.ndarray, target: float = 1.0) -> float:
    """Threshold u at which a decreasing ``rate`` falls to ``target`` (log-linear).

    Returns NaN when the rate does not cross the target within ``thresholds``.
    """
    u = np.asarray(thresholds, float)
    log_rate = np.log(np.maximum(np.asarray(rate, float), 1e-300)) - np.log(target)
    above = np.flatnonzero(log_rate >= 0)
    if above.size == 0 or above.max() + 1 >= u.size:
        return float("nan")
    j = int(above.max())
    frac = log_rate[j] / (log_rate[j] - log_rate[j + 1])
    return float(u[j] + frac * (u[j + 1] - u[j]))


@dataclass(frozen=True)
class AnalyticNullPrediction:  # pylint: disable=too-many-instance-attributes
    """Predicted null peak rate of a search: distinct null peaks >= t per micrograph.

    Attributes
    ----------
    thresholds : np.ndarray
        Z-score thresholds t.
    rate : np.ndarray
        Predicted distinct null peaks >= t per micrograph, on the search's own
        (per-pixel normalised) z-score.
    distinct_rate : np.ndarray
        As ``rate``, before the per-pixel normalisation factor (a global-sd z-score).
    lattice_rate : np.ndarray
        Lattice local maxima >= t, before crowding and normalisation.
    nominal_rate : np.ndarray
        The independent-trials expectation N Q(t).
    normalization_factor : np.ndarray
        F(t) (:func:`normalization_factor`).
    n_points : float
        Search points: valid pixels x orientations x planes.
    valid_area : float
        Valid pixels the peaks are counted in.
    mask_radius : float
        Peak suppression radius, pixels.
    kappa, kappa_se : float
        Mean squared correlation of random search points, and its standard error.
    rho_bar : float
        Mean-template share of the projections' power.
    variance_sd : float
        Relative standard deviation of the field variance between search points.
    neighbour_correlations : dict[str, list[float]]
        Median correlations at one and two steps along each axis.
    details : dict
        Engine-specific extras (sample sizes, timings, configuration).
    """

    thresholds: np.ndarray
    rate: np.ndarray
    distinct_rate: np.ndarray
    lattice_rate: np.ndarray
    nominal_rate: np.ndarray
    normalization_factor: np.ndarray
    n_points: float
    valid_area: float
    mask_radius: float
    kappa: float
    kappa_se: float
    rho_bar: float
    variance_sd: float
    neighbour_correlations: dict[str, list[float]]
    details: dict = field(default_factory=dict)

    def zscore_cutoff(self, false_positives: float = 1.0) -> float:
        """Predicted threshold for ``false_positives`` null peaks per micrograph."""
        return crossing(self.thresholds, self.rate, false_positives)

    def nominal_zscore_cutoff(self, false_positives: float = 1.0) -> float:
        """The independent-trials (nominal) threshold, for comparison."""
        return crossing(self.thresholds, self.nominal_rate, false_positives)

    def rate_at(self, threshold: float) -> float:
        """Predicted null peaks >= ``threshold`` per micrograph (log-linear)."""
        log_rate = np.log(np.maximum(self.rate, 1e-300))
        return float(np.exp(np.interp(threshold, self.thresholds, log_rate)))

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly dict, with the 1-false-positive thresholds."""
        out: dict[str, object] = {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in self.__dict__.items()
        }
        out["zscore_cutoff_1fp"] = self.zscore_cutoff(1.0)
        out["nominal_zscore_cutoff_1fp"] = self.nominal_zscore_cutoff(1.0)
        return out


def predict_null_rate(
    samples: LatticeSamples,
    kappa: tuple[float, float],
    rho_bar: float,
    n_points: float,
    valid_area: float,
    mask_radius: float,
    thresholds: np.ndarray,
    details: dict | None = None,
) -> AnalyticNullPrediction:
    """Combine the three parts into the predicted null peak rate.

    Parameters
    ----------
    samples : LatticeSamples
        Field correlations at sampled grid points.
    kappa : tuple[float, float]
        kappa and its standard error (:func:`.field.estimate_kappa`).
    rho_bar : float
        Mean-template share (:class:`.field.MeanTemplate`).
    n_points : float
        Search points: valid pixels x orientations x planes.
    valid_area : float
        Valid pixels the peaks are counted in.
    mask_radius : float
        Peak suppression radius, pixels.
    thresholds : np.ndarray
        Thresholds to evaluate.
    details : dict | None
        Extras to keep with the prediction.

    Returns
    -------
    AnalyticNullPrediction
    """
    u = np.asarray(thresholds, float)
    lattice = lattice_maxima_rate(samples, n_points, u)
    distinct = crowding_correction(lattice, valid_area, mask_radius)
    factor = normalization_factor(kappa[0], u)
    v = samples.variance / samples.variance.mean()
    return AnalyticNullPrediction(
        thresholds=u,
        rate=distinct * factor,
        distinct_rate=distinct,
        lattice_rate=lattice,
        nominal_rate=nominal_rate(n_points, u),
        normalization_factor=factor,
        n_points=float(n_points),
        valid_area=float(valid_area),
        mask_radius=float(mask_radius),
        kappa=float(kappa[0]),
        kappa_se=float(kappa[1]),
        rho_bar=float(rho_bar),
        variance_sd=float(np.std(v)),
        neighbour_correlations=samples.median_correlations(),
        details=dict(details or {}),
    )
