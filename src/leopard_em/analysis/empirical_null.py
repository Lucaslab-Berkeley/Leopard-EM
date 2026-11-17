"""Detection thresholds from null searches of phase-randomised copies of a micrograph.

The nominal threshold (:func:`~leopard_em.analysis.gaussian_noise_zscore_cutoff`)
treats every correlation of a search as an independent Gaussian trial. They are not:
neighbouring orientations, defocus planes and pixels are correlated, and the per-pixel
z-score normalisation changes the tail. The number of false positives the nominal
threshold allows is therefore only roughly what it promises.

A null search measures it instead. Search phase-randomised copies of the micrograph
(:func:`~leopard_em.utils.phase_randomization.phase_randomize_image`) with exactly the
same template, settings and valid region, collect their distinct peaks
(:func:`distinct_peaks`), and set the threshold from the pooled null peaks
(:func:`empirical_zscore_cutoff`). The functions here are engine-agnostic: they take
z-score maps and peak values, so they work for any 2DTM search engine.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from scipy.special import gammainc  # pylint: disable=no-name-in-module
from scipy.stats import chi2

ONE_SIGMA = 0.6826894921370859  # coverage of +-1 standard deviation

ThresholdRule = Literal["mean", "confidence"]


# --------------------------------------------------------------------------- #
# Distinct peaks
# --------------------------------------------------------------------------- #


def valid_region(
    shape: tuple[int, int],
    border: int = 0,
    exclude_mask: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray:
    """Boolean map of the pixels peaks are counted in (True = valid).

    Parameters
    ----------
    shape : tuple[int, int]
        Shape of the z-score map.
    border : int
        Pixels within ``border`` of any map edge are excluded. Default 0.
    exclude_mask : np.ndarray | torch.Tensor | None
        Boolean exclusion mask of ``shape`` (True = excluded), e.g. carbon or flat
        regions. Default None.

    Returns
    -------
    np.ndarray
        Boolean array of ``shape``.
    """
    if border < 0:
        raise ValueError("border must be >= 0.")
    valid = np.zeros(shape, dtype=bool)
    height, width = shape
    if 2 * border < height and 2 * border < width:
        valid[border : height - border, border : width - border] = True
    if exclude_mask is not None:
        if isinstance(exclude_mask, torch.Tensor):
            exclude_mask = exclude_mask.detach().cpu().numpy()
        mask = np.asarray(exclude_mask, dtype=bool)
        if mask.shape != tuple(shape):
            raise ValueError(f"Mask shape {mask.shape} != map shape {tuple(shape)}.")
        valid &= ~mask
    return valid


def _greedy_peaks(  # pylint: disable=too-many-locals
    z_map: np.ndarray, cutoff: float, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """The rule of :func:`~leopard_em.analysis.zscore_metric.find_peaks_from_zscore`.

    Visit the pixels with z > ``cutoff`` in descending z; keep a pixel unless it lies
    at a float32 distance < ``radius`` from an already kept one. Each kept pixel
    stamps its disc into a padded "taken" map, so a visit is one lookup and the cost
    is linear in the number of candidates. Ties in z are broken in raster order.

    Returns the kept pixels' (y, x), by descending z.
    """
    z = np.ascontiguousarray(z_map, dtype=np.float32)
    width = z.shape[1]
    flat = z.ravel()
    candidates = np.flatnonzero(flat > np.float32(cutoff))
    if candidates.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    candidates = candidates[np.argsort(-flat[candidates], kind="stable")]
    pos_y, pos_x = np.divmod(candidates, width)

    r = max(int(np.ceil(radius)), 0)
    dy, dx = np.mgrid[-r : r + 1, -r : r + 1]
    inside = np.sqrt((dy * dy + dx * dx).astype(np.float32)) < np.float32(radius)
    row = width + 2 * r
    offsets = (dy * row + dx)[inside].astype(np.int64)
    taken = np.zeros((z.shape[0] + 2 * r) * row, dtype=np.uint8)
    seen = memoryview(taken)  # fast scalar reads of the stamps written via ``taken``
    keep = []
    for i, p in enumerate(((pos_y + r) * row + (pos_x + r)).tolist()):
        if seen[p]:
            continue
        keep.append(i)
        taken[p + offsets] = 1
    idx = np.asarray(keep, dtype=np.int64)
    return pos_y[idx], pos_x[idx]


@dataclass(frozen=True)
class DistinctPeaks:
    """Distinct peaks of one z-score map at or above a cutoff, in a valid region.

    Attributes
    ----------
    pos_y, pos_x : np.ndarray
        Peak positions in the map, by descending z.
    zscore : np.ndarray
        Peak z-scores, descending.
    valid_area : int
        Number of valid pixels the peaks were found in.
    cutoff : float
        Every peak with z >= ``cutoff`` is listed.
    mask_radius : float
        Suppression radius of the peak finder, pixels.
    """

    pos_y: np.ndarray
    pos_x: np.ndarray
    zscore: np.ndarray
    valid_area: int
    cutoff: float
    mask_radius: float


def distinct_peaks(
    zscore_map: np.ndarray | torch.Tensor,
    cutoff: float,
    mask_radius: float = 10.0,
    border: int = 0,
    exclude_mask: np.ndarray | torch.Tensor | None = None,
) -> DistinctPeaks:
    """Distinct peaks of a z-score map with z >= ``cutoff``.

    The peaks are those of
    :func:`~leopard_em.analysis.zscore_metric.find_peaks_from_zscore` (greedy
    suppression in descending z: a pixel is a peak unless a higher peak lies closer
    than ``mask_radius``), found in time linear in the number of candidates, so a low
    cutoff on a 4k map takes about a second. Pixels outside the valid region are set
    to -inf before the search, so an excluded pixel can neither be a peak nor suppress
    a valid one.

    Because suppression runs in descending z, the peaks >= t found from a lower cutoff
    are exactly the peaks a search with cutoff t finds. So one call at a low cutoff
    serves every threshold above it.

    Parameters
    ----------
    zscore_map : np.ndarray | torch.Tensor
        2-D z-score map (Leopard-EM's ``scaled_mip``).
    cutoff : float
        Lowest z-score to list (inclusive).
    mask_radius : float
        Suppression radius, pixels. Default 10, as ``find_peaks_from_zscore``.
    border : int
        Exclude pixels within ``border`` of the map edge. Default 0.
    exclude_mask : np.ndarray | torch.Tensor | None
        Boolean exclusion mask (True = excluded), same shape as the map.

    Returns
    -------
    DistinctPeaks
    """
    if isinstance(zscore_map, torch.Tensor):
        z = zscore_map.detach().to("cpu", torch.float32).numpy()
    else:
        z = np.asarray(zscore_map, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"Expected a 2-D z-score map, got shape {z.shape}.")
    valid = valid_region((int(z.shape[0]), int(z.shape[1])), border, exclude_mask)
    masked = np.where(valid, z, np.float32(-np.inf))
    # The finder keeps z > cutoff; step one float32 ulp down so that z == cutoff counts.
    below = float(np.nextafter(np.float32(cutoff), np.float32(-np.inf)))
    pos_y, pos_x = _greedy_peaks(masked, below, float(mask_radius))
    values = z[pos_y, pos_x].astype(np.float64)
    order = np.argsort(-values, kind="stable")
    return DistinctPeaks(
        pos_y=pos_y[order],
        pos_x=pos_x[order],
        zscore=values[order],
        valid_area=int(valid.sum()),
        cutoff=float(cutoff),
        mask_radius=float(mask_radius),
    )


# --------------------------------------------------------------------------- #
# Null peak rate and thresholds
# --------------------------------------------------------------------------- #


def poisson_interval(
    counts: np.ndarray | int, conf: float = ONE_SIGMA
) -> tuple[np.ndarray, np.ndarray]:
    """Exact (Garwood) central confidence interval of a Poisson mean.

    ``lo = chi2.ppf(a/2, 2k) / 2`` (0 for k = 0) and ``hi = chi2.ppf(1 - a/2, 2k + 2)
    / 2``, with ``a = 1 - conf``.

    Parameters
    ----------
    counts : np.ndarray | int
        Observed counts k.
    conf : float
        Coverage of the interval. Default 0.6827 (+-1 standard deviation).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Lower and upper bounds.
    """
    if not 0.0 < conf < 1.0:
        raise ValueError("conf must lie in (0, 1).")
    k = np.asarray(counts, dtype=np.float64)
    alpha = 1.0 - conf
    lo = np.where(k > 0, chi2.ppf(alpha / 2, 2 * np.maximum(k, 1)) / 2, 0.0)
    hi = chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2
    return lo, hi


def counted_rank(
    allowed: float, rule: ThresholdRule = "mean", confidence: float = 0.9
) -> int | None:
    """1-based rank of the pooled null peak a counted threshold sits at.

    With R null copies and a target of ``f`` false positives per micrograph,
    ``allowed = R * f`` null peaks may pass. If the null peaks above a threshold form a
    Poisson process, R times the rate above the r-th highest pooled peak is
    Gamma(r, 1) distributed.

    Parameters
    ----------
    allowed : float
        R x target false positives per micrograph.
    rule : {"mean", "confidence"}
        ``"mean"``: rank m = floor(allowed), unbiased in the mean (delivers the target
        on average; its median is a little below). ``"confidence"``: the largest rank
        r with P(Gamma(r, 1) <= allowed) >= ``confidence``, so the delivered rate is
        at most the target with that probability.
    confidence : float
        Probability for ``rule="confidence"``. Default 0.9.

    Returns
    -------
    int | None
        The rank, 0 when fewer than one null peak may pass (the threshold then goes
        just above the highest null peak), or None when the ``"confidence"`` rule has
        no valid rank (it needs ``allowed >= -ln(1 - confidence)``, 2.3 at 0.9).
    """
    m = int(np.floor(allowed + 1e-9))
    if rule == "mean":
        return m
    if rule == "confidence":
        rank = None
        for j in range(1, m + 2):  # P(Gamma(j, 1) <= allowed) falls with j
            if gammainc(j, allowed) >= confidence:
                rank = j
            else:
                break
        return rank
    raise ValueError(f"Unknown rule {rule!r}; use 'mean' or 'confidence'.")


def empirical_zscore_cutoff(
    null_peaks: Sequence[np.ndarray],
    false_positives: float = 1.0,
    rule: ThresholdRule = "mean",
    confidence: float = 0.9,
) -> float:
    """Z-score threshold for a target number of false positives per micrograph.

    Pool the distinct peaks of R null copies and sort them, t_1 >= t_2 >= ...; with
    ``m = floor(R x false_positives)``, the threshold is t_m and peaks with z >= t_m
    pass (m pooled null peaks). This delivers the target on average. If m is 0 the
    threshold is placed just above t_1, the most conservative choice the nulls allow.

    Use R >= 5 null copies for a 1-false-positive target: the threshold's standard
    deviation is about 0.1 / sqrt(R) z. For targets of 10 or more false positives per
    micrograph, one or two copies are enough.

    Parameters
    ----------
    null_peaks : Sequence[np.ndarray]
        Distinct-peak z-scores of each null copy (one array per copy), each listing
        every peak above a common floor that lies below the threshold sought.
    false_positives : float
        Target false positives per micrograph. Default 1.
    rule : {"mean", "confidence"}
        See :func:`counted_rank`. Default ``"mean"``.
    confidence : float
        Probability for ``rule="confidence"``. Default 0.9.

    Returns
    -------
    float
        The threshold; NaN when ``rule="confidence"`` has too few allowed peaks.

    Raises
    ------
    ValueError
        If the rank lies below the lowest pooled peak, i.e. the copies' peak floor is
        too high for this target.
    """
    if len(null_peaks) == 0:
        raise ValueError("Need at least one null copy.")
    pooled = np.sort(np.concatenate([np.asarray(p, float) for p in null_peaks]))[::-1]
    allowed = false_positives * len(null_peaks)
    rank = counted_rank(allowed, rule, confidence)
    if rank is None:
        return float("nan")
    if rank == 0:
        if pooled.size == 0:
            raise ValueError("No null peaks: lower the peak floor.")
        return float(np.nextafter(pooled[0], np.inf))
    if rank > pooled.size:
        raise ValueError(
            f"The threshold is the {rank}-th highest null peak, but only "
            f"{pooled.size} were collected: lower the peak floor."
        )
    return float(pooled[rank - 1])


@dataclass(frozen=True)
class NullRateCurve:
    """lambda_null(t): expected distinct null peaks >= t per micrograph.

    Attributes
    ----------
    thresholds : np.ndarray
        Thresholds t, ascending.
    counts : np.ndarray
        Distinct null peaks >= t, summed over the null copies.
    n_copies : int
        Number of null copies pooled.
    area : float
        Valid area (pixels) the rate is expressed for.
    rate : np.ndarray
        Expected null peaks >= t per ``area`` pixels.
    rate_lo, rate_hi : np.ndarray
        Exact Poisson interval of ``rate`` at ``conf``.
    conf : float
        Coverage of ``rate_lo`` .. ``rate_hi``.
    """

    thresholds: np.ndarray
    counts: np.ndarray
    n_copies: int
    area: float
    rate: np.ndarray
    rate_lo: np.ndarray
    rate_hi: np.ndarray
    conf: float

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly dict (lists for the arrays)."""
        return {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class NullPeaks:
    """Distinct peaks of the null copies of one search configuration.

    Attributes
    ----------
    zscores : list[np.ndarray]
        Distinct-peak z-scores of each null copy, descending.
    valid_areas : list[int]
        Valid area (pixels) of each copy's peak search.
    cutoff : float
        Every peak with z >= ``cutoff`` is listed.
    mask_radius : float
        Suppression radius of the peak finder, pixels.
    seeds : list[int]
        Phase-randomisation seed of each copy (informational).
    """

    zscores: list[np.ndarray]
    valid_areas: list[int]
    cutoff: float
    mask_radius: float
    seeds: list[int] = field(default_factory=list)

    @classmethod
    def from_peaks(
        cls, peaks: Sequence[DistinctPeaks], seeds: Sequence[int] = ()
    ) -> "NullPeaks":
        """Collect the :func:`distinct_peaks` results of the null copies."""
        if len(peaks) == 0:
            raise ValueError("Need at least one null copy.")
        if len({p.mask_radius for p in peaks}) != 1:
            raise ValueError("All copies must share the same mask radius.")
        return cls(
            zscores=[np.asarray(p.zscore, float) for p in peaks],
            valid_areas=[int(p.valid_area) for p in peaks],
            cutoff=max(p.cutoff for p in peaks),
            mask_radius=peaks[0].mask_radius,
            seeds=[int(s) for s in seeds],
        )

    @property
    def n_copies(self) -> int:
        """Number of null copies."""
        return len(self.zscores)

    def rate(
        self,
        thresholds: Sequence[float] | np.ndarray,
        area: float | None = None,
        conf: float = ONE_SIGMA,
    ) -> NullRateCurve:
        """Expected distinct null peaks >= t per micrograph, with Poisson intervals.

        Parameters
        ----------
        thresholds : Sequence[float] | np.ndarray
            Thresholds t (at or above ``cutoff``).
        area : float | None
            Valid area (pixels) of the micrograph the rate is for. Default None: the
            copies' mean valid area (the same micrograph).
        conf : float
            Coverage of the interval. Default +-1 standard deviation.

        Returns
        -------
        NullRateCurve
        """
        t = np.sort(np.atleast_1d(np.asarray(thresholds, dtype=np.float64)))
        if t.size and t[0] < self.cutoff - 1e-9:
            raise ValueError(f"Thresholds below the peak floor {self.cutoff}.")
        pooled = np.concatenate(self.zscores) if self.zscores else np.zeros(0)
        counts = (pooled[None, :] >= t[:, None]).sum(axis=1).astype(np.int64)
        total_area = float(sum(self.valid_areas))
        unit = float(np.mean(self.valid_areas)) if area is None else float(area)
        scale = unit / total_area
        lo, hi = poisson_interval(counts, conf)
        return NullRateCurve(
            thresholds=t,
            counts=counts,
            n_copies=self.n_copies,
            area=unit,
            rate=counts * scale,
            rate_lo=lo * scale,
            rate_hi=hi * scale,
            conf=float(conf),
        )

    def zscore_cutoff(
        self,
        false_positives: float = 1.0,
        rule: ThresholdRule = "mean",
        confidence: float = 0.9,
    ) -> float:
        """Threshold for a target false positives per micrograph.

        See :func:`empirical_zscore_cutoff`.
        """
        return empirical_zscore_cutoff(
            self.zscores, false_positives, rule=rule, confidence=confidence
        )

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly dict."""
        return {
            "zscores": [z.tolist() for z in self.zscores],
            "valid_areas": list(self.valid_areas),
            "cutoff": self.cutoff,
            "mask_radius": self.mask_radius,
            "seeds": list(self.seeds),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NullPeaks":
        """Inverse of :meth:`as_dict`."""
        return cls(
            zscores=[np.asarray(z, float) for z in data["zscores"]],
            valid_areas=[int(a) for a in data["valid_areas"]],
            cutoff=float(data["cutoff"]),
            mask_radius=float(data["mask_radius"]),
            seeds=[int(s) for s in data.get("seeds", [])],
        )
