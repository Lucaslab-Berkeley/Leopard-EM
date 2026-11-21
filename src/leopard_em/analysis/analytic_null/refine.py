"""Refined null: the null peak rate after local refinement of a search's candidates.

Refinement (e.g. Leopard-EM's refine_template) re-scores each candidate of a search on
a finer local grid around its best search point: orientation offsets composed with the
candidate's orientation, defocus offsets about its plane, and a few pixels about its
position. The refined z-score is the best of those scores, so a null candidate of
height u gains G = max(window) - u (>= 0 when the window holds the candidate's own
score; an exact re-score of an approximate search can lose).

Only the search's candidates are refined (its distinct peaks at or above a candidate
threshold t_c), so the expected number of refined null peaks at or above z is

    lambda_ref(z) = integral_{u >= t_c} l(u) P(G >= z - u | u) du,

where l(u) = -d lambda / du is the density of the search's null peaks over height
(:func:`.lattice.predict_null_rate`). The gain's distribution follows from the same
Gaussian field (:mod:`.field`). Around a sampled grid point, the scores refinement
looks at and the search's own grid points inside its window are jointly Gaussian, with
a covariance fixed by the projections and the image power spectrum. A null candidate
of height u is that grid point conditioned on two facts the search established:

1. its score is u, which fixes every other score's mean (rho u) and lowers its
   variance (to 1 - rho^2, for correlation rho with the candidate);
2. it beat every search grid point in the window (it is the search's best score there).

Drawing the window under (1) and keeping the draws that satisfy (2) gives the
distribution of G (:func:`sample_refinement_gain`).

Counting every local maximum of refinement's fine grid over the whole search space
instead (as if refinement searched everywhere) overcounts below the threshold: it
counts maxima away from any candidate, and maxima at pixels whose best search score
fell below t_c, which refinement never sees.
"""

import math
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import torch

from .field import MeanTemplate, PowerWeighting
from .lattice import AnalyticNullPrediction, crossing

DEFAULT_HEIGHTS = np.round(np.arange(5.0, 9.01, 0.5), 3)
DEFAULT_GAIN_EDGES = np.round(np.arange(-3.0, 3.0001, 0.005), 4)
_JITTERS = (1e-6, 1e-5, 1e-4, 1e-3)


@dataclass
class RefinementWindow:
    """The scores refinement looks at around one candidate, and the points it beat.

    All projections are on the lag grid (:meth:`.field.PowerWeighting.transform`),
    shape (n, Ly, Lx // 2 + 1), with the mean projection not yet removed.

    Attributes
    ----------
    window : torch.Tensor
        The projections refinement scores (orientation x defocus offsets), each scored
        at every position within the shift radius.
    centre : int
        Index in ``window`` of the candidate's own search point (zero offsets).
    search : torch.Tensor
        The search's grid points around the candidate: its neighbours one step along
        every search axis and its adjacent defocus planes. The candidate beat each of
        them, and its own search point, at every position within the condition radius.
    own : torch.Tensor | None
        The candidate's own search template, shape (Ly, Lx // 2 + 1), when the search
        scores a different template from refinement's: an approximate search (e.g.
        Panther-EM's SVD templates) refined by an exact one. The candidate's score is
        then ``own`` at zero shift, and ``search`` holds the search's own templates
        of its neighbours. None (default): the search scores ``window[centre]``.
    """

    window: torch.Tensor
    centre: int
    search: torch.Tensor
    own: torch.Tensor | None = None


def _survival(counts: np.ndarray) -> np.ndarray:
    """P(G >= edge_j) from counts per bin [edge_j, edge_j+1) (last bin open)."""
    total = counts.sum(axis=-1, keepdims=True)
    tail = np.cumsum(counts[..., ::-1], axis=-1)[..., ::-1]
    return np.divide(tail, total, out=np.ones_like(tail, float), where=total > 0)


@dataclass(frozen=True)
class RefinementGain:
    """Distribution of the refinement gain G of a null candidate, by its height.

    Attributes
    ----------
    heights : np.ndarray
        Candidate heights u (search z-scores), shape (n_u,).
    gain_edges : np.ndarray
        Gains g, shape (n_g,), increasing. A refinement that re-scores with the
        search's own templates gains G >= 0; one that re-scores an approximate
        search exactly can lose (G < 0).
    survival : np.ndarray
        P(G >= g | u), shape (n_u, n_g).
    accepted : np.ndarray
        Fraction of draws at each height in which the candidate beat every search
        point in the window (the draws kept; windows weighted as pooled), shape
        (n_u,).
    n_windows : int
        Sampled candidates (grid points).
    n_draws : int
        Gaussian draws per candidate.
    scores_per_window : int
        Scores in each window's joint covariance (projections x positions).
    details : dict
        Window sizes, Cholesky jitter, timings.
    """

    heights: np.ndarray
    gain_edges: np.ndarray
    survival: np.ndarray
    accepted: np.ndarray
    n_windows: int
    n_draws: int
    scores_per_window: int
    details: dict = field(default_factory=dict)

    def survival_at(
        self, height: float | np.ndarray, gain: float | np.ndarray
    ) -> np.ndarray:
        """P(G >= gain | u = height), linear in both.

        1 below the tabulated gains, 0 above them. Heights outside the sampled range
        take the nearest sampled height.
        """
        u, g = np.broadcast_arrays(np.asarray(height, float), np.asarray(gain, float))
        flat_u, flat_g = u.reshape(-1), g.reshape(-1)
        per_height = np.stack(
            [
                np.interp(flat_g, self.gain_edges, s, left=1.0, right=0.0)
                for s in self.survival
            ]
        )  # (n_u, m)
        if len(self.heights) == 1:
            out = per_height[0]
        else:
            hu = np.clip(flat_u, self.heights[0], self.heights[-1])
            j = np.clip(np.searchsorted(self.heights, hu) - 1, 0, len(self.heights) - 2)
            frac = (hu - self.heights[j]) / (self.heights[j + 1] - self.heights[j])
            cols = np.arange(flat_u.size)
            out = (1 - frac) * per_height[j, cols] + frac * per_height[j + 1, cols]
        return np.asarray(out).reshape(u.shape)

    def quantile(self, height: float, p: float) -> float:
        """The gain G with P(G < gain | u = height) = p (smallest edge reaching it)."""
        s = self.survival_at(height, self.gain_edges)
        below = 1.0 - s
        j = int(np.searchsorted(below, p, side="left"))
        return float(self.gain_edges[min(j, len(self.gain_edges) - 1)])

    def summary(self) -> dict[str, dict[str, float]]:
        """Median, 90th and 99th percentile gain and acceptance at each height."""
        return {
            f"{u:g}": {
                "median": self.quantile(u, 0.5),
                "p90": self.quantile(u, 0.9),
                "p99": self.quantile(u, 0.99),
                "accepted": float(a),
            }
            for u, a in zip(self.heights, self.accepted, strict=True)
        }

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly dict."""
        return {
            "heights": self.heights.tolist(),
            "gain_edges": self.gain_edges.tolist(),
            "survival": self.survival.tolist(),
            "accepted": self.accepted.tolist(),
            "n_windows": self.n_windows,
            "n_draws": self.n_draws,
            "scores_per_window": self.scores_per_window,
            "summary": self.summary(),
            "details": self.details,
        }


@dataclass(frozen=True)
class ScoreLayout:
    """Which projection at which position each score of a window's covariance is.

    Scores are grouped by position (row-major over dy, then dx, from -r to r, with
    r the larger radius); at each position the window projections come first (within
    the shift radius), then the search points (within the condition radius), then the
    candidate's own projection where only the search scored it (outside the shift
    radius).

    Attributes
    ----------
    projection : np.ndarray
        Index into ``cat(window, search)`` of each score.
    dy, dx : np.ndarray
        Position of each score, pixels.
    in_window : np.ndarray
        Scores refinement looks at (its maximum is the refined score).
    beaten : np.ndarray
        Search scores the candidate beat: every search point within the condition
        radius and the candidate's own projection away from zero shift.
    centre : int
        Index of the candidate's own score (its projection at zero shift).
    """

    projection: np.ndarray
    dy: np.ndarray
    dx: np.ndarray
    in_window: np.ndarray
    beaten: np.ndarray
    centre: int


def score_layout(  # pylint: disable=too-many-locals,too-many-arguments
    n_window: int,
    n_search: int,
    centre: int,
    shift_radius: int,
    condition_radius: int,
    separate_own: bool = False,
) -> ScoreLayout:
    """The scores of a window (:class:`ScoreLayout`).

    With ``separate_own``, the candidate's own search template is projection
    ``n_window + n_search`` (:attr:`RefinementWindow.own`): it is scored at every
    position within the condition radius, and the candidate is it at zero shift.
    """
    rw, rc = int(shift_radius), int(condition_radius)
    r = max(rw, rc)
    own_index = n_window + n_search if separate_own else centre
    projection: list[int] = []
    dys: list[int] = []
    dxs: list[int] = []
    kinds: list[str] = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            in_shift = abs(dy) <= rw and abs(dx) <= rw
            if in_shift:
                projection += list(range(n_window))
                kinds += ["window"] * n_window
            if abs(dy) <= rc and abs(dx) <= rc:
                projection += list(range(n_window, n_window + n_search))
                kinds += ["search"] * n_search
                if separate_own or not in_shift:
                    projection.append(own_index)
                    kinds.append("own")
            added = len(projection) - len(dys)
            dys += [dy] * added
            dxs += [dx] * added
    projection_a = np.asarray(projection)
    dy_a, dx_a = np.asarray(dys), np.asarray(dxs)
    kind = np.asarray(kinds)
    in_window = kind == "window"
    at_zero = (dy_a == 0) & (dx_a == 0)
    if separate_own:
        candidate = (kind == "own") & at_zero
        beaten = (kind == "search") | ((kind == "own") & ~at_zero)
    else:
        own_in_window = in_window & (projection_a == centre)
        candidate = own_in_window & at_zero
        beaten = (kind == "search") | (kind == "own") | (own_in_window & ~at_zero)
    centre_index = int(np.flatnonzero(candidate)[0])
    return ScoreLayout(projection_a, dy_a, dx_a, in_window, beaten, centre_index)


def _lag_grams(
    q: torch.Tensor, rows: list[int], weighting: PowerWeighting, rw: int, r: int
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[tuple[int, int], torch.Tensor]]:
    """Real parts of the power-weighted Gram matrices of ``q`` at every lag.

    ``full`` (every projection pair) for lags within +-2 ``rw`` (window positions
    against window positions); ``partial`` (``rows`` against every projection) for
    every lag within +-2 ``r``.
    """
    device = q.device
    q_conj = q.conj().T.contiguous()
    weights = weighting.weights.reshape(-1).double().clamp(min=0)
    ky = weighting.ky.expand(weighting.weights.shape).reshape(-1).double()
    kx = weighting.kx.expand(weighting.weights.shape).reshape(-1).double()
    row_index = torch.as_tensor(rows, device=device)
    full: dict[tuple[int, int], torch.Tensor] = {}
    partial: dict[tuple[int, int], torch.Tensor] = {}
    for dy in range(-2 * r, 2 * r + 1):
        for dx in range(-2 * r, 2 * r + 1):
            phase = weights * torch.exp(-2j * math.pi * (ky * dy + kx * dx))
            if max(abs(dy), abs(dx)) > 2 * rw:
                partial[(dy, dx)] = ((q[row_index] * phase) @ q_conj).real
                continue
            if (-dy, -dx) in full:  # G(-d) = G(d)^T
                full[(dy, dx)] = full[(-dy, -dx)].T
            else:
                full[(dy, dx)] = ((q * phase) @ q_conj).real
            partial[(dy, dx)] = full[(dy, dx)][row_index]
    return full, partial


def _assemble(  # pylint: disable=too-many-locals
    layout: ScoreLayout,
    full: dict[tuple[int, int], torch.Tensor],
    partial: dict[tuple[int, int], torch.Tensor],
    row_of: dict[int, int],
    device: torch.device,
) -> torch.Tensor:
    """The covariance of ``layout``'s scores from the lag Gram matrices."""
    # each position's scores are contiguous
    keys = list(zip(layout.dy.tolist(), layout.dx.tolist(), strict=True))
    groups: dict[tuple[int, int], slice] = {}
    for i, key in enumerate(keys):
        start = groups[key].start if key in groups else i
        groups[key] = slice(start, i + 1)
    cov = torch.empty((len(keys), len(keys)), device=device, dtype=torch.float64)
    for (ya, xa), sa in groups.items():
        ta = layout.projection[sa]
        a_search = not bool(layout.in_window[sa].any())
        a_rows = [row_of[int(t)] for t in ta] if a_search else None
        for (yb, xb), sb in groups.items():
            tb = layout.projection[sb]
            lag = (ya - yb, xa - xb)
            if a_rows is not None:
                cov[sa, sb] = partial[lag][a_rows][:, tb]
            elif not bool(layout.in_window[sb].any()):
                back = (-lag[0], -lag[1])
                cov[sa, sb] = partial[back][[row_of[int(t)] for t in tb]][:, ta].T
            else:
                cov[sa, sb] = full[lag][ta][:, tb]
    return cov


def window_covariance(  # pylint: disable=too-many-locals
    window: RefinementWindow,
    weighting: PowerWeighting,
    mean_template: MeanTemplate | None,
    shift_radius: int,
    condition_radius: int | None = None,
) -> tuple[torch.Tensor, ScoreLayout]:
    """Joint covariance of a window's scores (:func:`score_layout`).

    Units: the field's mean variance after the mean projection is removed
    (``mean_template.sigma2``), so the scores are on the search's z scale. Computed in
    float64: neighbouring offsets give near-duplicate scores. Negative lag-grid
    weights (a small fraction of the image's autocorrelation spectrum) are set to
    zero, which keeps the matrix positive semi-definite.

    Parameters
    ----------
    window : RefinementWindow
        The window's projections.
    weighting : PowerWeighting
        Lag-grid power weighting of the image.
    mean_template : MeanTemplate | None
        The search's mean projection and field variance. None: the projections are
        the scores' own templates already (mean removed, in z units), as an engine
        whose search and refinement normalise differently supplies them.
    shift_radius : int
        Positions refinement scores, +-pixels.
    condition_radius : int | None
        Positions at which the candidate beat the search points, +-pixels. Default:
        ``shift_radius``.

    Returns
    -------
    tuple[torch.Tensor, ScoreLayout]
        The covariance (float64, on the weighting's device) and its layout.
    """
    rw = int(shift_radius)
    rc = rw if condition_radius is None else int(condition_radius)
    device = weighting.weights.device
    parts = [window.window, window.search]
    if window.own is not None:
        parts.append(window.own[None])
    projections = torch.cat([x.to(device) for x in parts])
    n_window, n_all = int(window.window.shape[0]), int(projections.shape[0])
    n_search = int(window.search.shape[0])
    separate = window.own is not None
    layout = score_layout(n_window, n_search, int(window.centre), rw, rc, separate)
    if mean_template is not None:
        projections = projections - mean_template.mean.to(device)
    q = projections.reshape(n_all, -1).to(torch.complex128)
    # rows of the search points and the candidate's own projection, at any lag
    own_row = n_all - 1 if separate else int(window.centre)
    rows = [own_row, *range(n_window, n_window + n_search)]
    full, partial = _lag_grams(q, rows, weighting, rw, max(rw, rc))
    cov = _assemble(layout, full, partial, {t: i for i, t in enumerate(rows)}, device)
    if mean_template is not None:
        cov /= mean_template.sigma2
    return cov, layout


# pylint: disable=too-many-locals,too-many-arguments,too-many-statements
def sample_refinement_gain(
    windows: Iterable[RefinementWindow],
    weighting: PowerWeighting,
    mean_template: MeanTemplate | None,
    *,
    shift_radius: int = 1,
    condition_radius: int = 2,
    heights: np.ndarray | None = None,
    gain_edges: np.ndarray | None = None,
    n_draws: int = 200_000,
    chunk_size: int = 10_000,
    max_scores: int = 30_000,
    seed: int = 0,
) -> RefinementGain:
    """Sample the refinement gain of null candidates from the field's covariance.

    For each window (one sampled grid point, :class:`RefinementWindow`): the joint
    covariance of its scores (:func:`window_covariance`: the window's projections
    within ``shift_radius``, the search points within ``condition_radius``); the
    scores conditioned on the candidate's being u (mean rho u, covariance
    C - c c^T / c0); ``n_draws`` Gaussian draws, kept where the candidate beats every
    search point (and its own projection at the other positions); the gain
    G = max(window) - u of each kept draw. Windows are pooled in proportion to the
    density of their candidate's score at u (a score of variance v: phi(u / sqrt(v))
    / sqrt(v)): search points whose scores vary more (e.g. defocus planes of higher
    noise variance) supply more candidates at a given height.

    Parameters
    ----------
    windows : Iterable[RefinementWindow]
        One per sampled candidate (sample grid points uniformly over the search).
    weighting : PowerWeighting
        Lag-grid power weighting of the image.
    mean_template : MeanTemplate | None
        The search's mean projection and field variance
        (:func:`.field.estimate_mean_template`). None: the windows' projections are
        the scores' own templates already (:func:`window_covariance`).
    shift_radius : int
        Positions refinement scores about the candidate, +-pixels. Default 1: on
        xenon_131 the window's maximum lies within +-1 px (+-2 px gives the same
        gains).
    condition_radius : int
        Positions at which the candidate is known to beat the search points,
        +-pixels (it is a distinct peak of the search's z-score map). Default 2.
        Only the search points are added there, so a larger radius is cheap.
    heights : np.ndarray | None
        Candidate heights u. Default 5 to 9 in steps of 0.5.
    gain_edges : np.ndarray | None
        Gains to tabulate. Default -3 to 3 in steps of 0.005.
    n_draws : int
        Gaussian draws per window. Default 200,000.
    chunk_size : int
        Draws per batch. Default 10,000.
    max_scores : int
        Largest joint covariance allowed (projections x positions); its float64 copy
        and Cholesky factor take 16 bytes per entry. Default 30,000 (~14 GB).
    seed : int
        Seed of the draws. Default 0.

    Returns
    -------
    RefinementGain
    """
    start = time.time()
    heights = DEFAULT_HEIGHTS if heights is None else np.asarray(heights, float)
    edges = DEFAULT_GAIN_EDGES if gain_edges is None else np.asarray(gain_edges, float)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("gain_edges must increase.")
    r = int(shift_radius)
    rc = max(r, int(condition_radius))
    device = weighting.weights.device
    edges_t = torch.as_tensor(edges, device=device, dtype=torch.float32)
    counts = np.zeros((len(heights), len(edges)))
    kept = np.zeros(len(heights))
    weight_total = np.zeros(len(heights))
    total = 0
    n_windows = 0
    jitters: list[float] = []
    sizes: list[int] = []
    generator = torch.Generator(device=device).manual_seed(int(seed))
    for window in windows:
        n_window = int(window.window.shape[0])
        n_search = int(window.search.shape[0])
        separate = window.own is not None
        size = len(
            score_layout(n_window, n_search, int(window.centre), r, rc, separate).dy
        )
        if size > max_scores:
            raise ValueError(
                f"The refinement window has {size} scores, more than max_scores="
                f"{max_scores}. Use a coarser defocus step or a smaller shift radius."
            )
        cov, layout = window_covariance(window, weighting, mean_template, r, rc)
        c0 = layout.centre
        keep = torch.ones(size, dtype=torch.bool, device=device)
        keep[c0] = False
        c = cov[keep, c0]
        c00 = float(cov[c0, c0])
        regression = (c / c00).float()  # each score's mean is regression * u
        # a search point whose score has variance c00 supplies candidates at height u
        # in proportion to its density there: weight the window by it (relative to 1)
        density = np.exp(-(heights**2) / 2 * (1 / c00 - 1)) / np.sqrt(c00)
        conditional = cov[keep][:, keep]
        del cov
        conditional.addr_(c, c, alpha=-1.0 / c00)  # C - c c^T / c0
        info = 1
        factor = None
        added = 0.0
        for jitter in _JITTERS:
            conditional.diagonal().add_(jitter - added)
            added = jitter
            factor, info_t = torch.linalg.cholesky_ex(  # pylint: disable=not-callable
                conditional
            )
            info = int(info_t)
            if info == 0:
                jitters.append(jitter)
                break
        del conditional
        if info != 0 or factor is None:
            raise RuntimeError("The conditional window covariance is not positive.")
        factor32 = factor.float()
        del factor
        own_in_window = bool(layout.in_window[c0])
        kept_scores = np.delete(np.arange(size), c0)
        in_window = torch.as_tensor(layout.in_window[kept_scores], device=device)
        beaten = torch.as_tensor(layout.beaten[kept_scores], device=device)
        for _ in range(max(1, n_draws // chunk_size)):
            noise = torch.randn(
                (size - 1, chunk_size), device=device, generator=generator
            )
            base = factor32 @ noise
            for i, u in enumerate(heights):
                draws = base + float(u) * regression[:, None]
                ok = (draws[beaten] < u).all(dim=0)
                gain = draws[in_window].max(dim=0).values - float(u)
                if own_in_window:  # the candidate's own score (u) is in the window
                    gain = gain.clamp(min=0)
                gain = gain[ok]
                kept[i] += density[i] * int(ok.sum())
                weight_total[i] += density[i] * chunk_size
                if gain.numel():
                    # below the first edge counts in the first bin (survival there 1)
                    bins = torch.bucketize(gain, edges_t, right=True) - 1
                    counts[i] += density[i] * (
                        torch.bincount(bins.clamp(min=0), minlength=len(edges))
                        .cpu()
                        .numpy()
                    )
            total += chunk_size
        del factor32
        n_windows += 1
        sizes.append(size)
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    if n_windows == 0:
        raise ValueError("No refinement windows.")
    return RefinementGain(
        heights=heights,
        gain_edges=edges,
        survival=_survival(counts),
        accepted=kept / weight_total,
        n_windows=n_windows,
        n_draws=total // n_windows,
        scores_per_window=int(max(sizes)),
        details={
            "shift_radius": r,
            "condition_radius": rc,
            "cholesky_jitter": jitters,
            "seed": int(seed),
            "seconds": time.time() - start,
        },
    )


@dataclass(frozen=True)
class RefinedNullPrediction:  # pylint: disable=too-many-instance-attributes
    """Predicted null peak rate after refining a search's candidates.

    Attributes
    ----------
    thresholds : np.ndarray
        Refined z-score thresholds t.
    rate : np.ndarray
        Predicted refined null peaks >= t per micrograph.
    candidate_threshold : float
        Search z-score at or above which candidates are refined.
    candidates : float
        Expected null candidates per micrograph (the search's rate there).
    search_rate : np.ndarray
        The search's own predicted rate at ``thresholds`` (no refinement).
    gain : RefinementGain
        The refinement gain distribution.
    details : dict
        Engine-specific extras (window, sample sizes, timings).
    """

    thresholds: np.ndarray
    rate: np.ndarray
    candidate_threshold: float
    candidates: float
    search_rate: np.ndarray
    gain: RefinementGain
    details: dict = field(default_factory=dict)

    def zscore_cutoff(self, false_positives: float = 1.0) -> float:
        """Predicted refined threshold for ``false_positives`` per micrograph."""
        return crossing(self.thresholds, self.rate, false_positives)

    def search_zscore_cutoff(self, false_positives: float = 1.0) -> float:
        """The search's own predicted threshold, for comparison."""
        return crossing(self.thresholds, self.search_rate, false_positives)

    def rate_at(self, threshold: float) -> float:
        """Predicted refined null peaks >= ``threshold`` per micrograph."""
        log_rate = np.log(np.maximum(self.rate, 1e-300))
        return float(np.exp(np.interp(threshold, self.thresholds, log_rate)))

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly dict, with the 1-false-positive thresholds."""
        return {
            "thresholds": self.thresholds.tolist(),
            "rate": self.rate.tolist(),
            "search_rate": self.search_rate.tolist(),
            "candidate_threshold": self.candidate_threshold,
            "candidates": self.candidates,
            "zscore_cutoff_1fp": self.zscore_cutoff(1.0),
            "search_zscore_cutoff_1fp": self.search_zscore_cutoff(1.0),
            "gain": self.gain.as_dict(),
            "details": self.details,
        }


def predict_refined_null_rate(
    search: AnalyticNullPrediction,
    gain: RefinementGain,
    candidate_threshold: float,
    thresholds: np.ndarray | None = None,
    details: dict | None = None,
) -> RefinedNullPrediction:
    """Combine the search's null rate with the refinement gain.

    lambda_ref(t) = sum over candidate heights u >= ``candidate_threshold`` of
    l(u) du P(G >= t - u | u), l(u) = -d lambda / du from ``search``.

    Parameters
    ----------
    search : AnalyticNullPrediction
        The search's predicted null rate (fine thresholds, e.g. steps of 0.01).
    gain : RefinementGain
        From :func:`sample_refinement_gain`.
    candidate_threshold : float
        Search z-score at or above which candidates are refined.
    thresholds : np.ndarray | None
        Refined thresholds to evaluate. Default: the search's.
    details : dict | None
        Extras to keep with the prediction.

    Returns
    -------
    RefinedNullPrediction
    """
    t = np.asarray(search.thresholds, float)
    density = -np.gradient(np.asarray(search.rate, float), t)
    sel = t >= candidate_threshold - 1e-9
    if sel.sum() < 2:
        raise ValueError(
            "The candidate threshold is above the search prediction's thresholds."
        )
    u = t[sel]
    step = np.gradient(u)  # trapezoid-like weights on a (near) uniform grid
    step[0] /= 2
    step[-1] /= 2
    weight = density[sel] * step
    out_t = t if thresholds is None else np.asarray(thresholds, float)
    survival = gain.survival_at(u[:, None], out_t[None, :] - u[:, None])
    rate = (weight[:, None] * survival).sum(axis=0)
    return RefinedNullPrediction(
        thresholds=out_t,
        rate=rate,
        candidate_threshold=float(candidate_threshold),
        candidates=float(np.interp(candidate_threshold, t, search.rate)),
        search_rate=np.interp(out_t, t, search.rate),
        gain=gain,
        details=dict(details or {}),
    )
