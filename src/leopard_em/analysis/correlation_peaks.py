"""Orientation-aware peak finding over a sparse correlation table.

The per-pixel statistics maps written by ``match_template`` keep only the single best
scoring hypothesis at each position, so two particles that overlap in projection --
the near and far wall of a microtubule, or two crossing filaments -- annihilate each
other, and every sub-maximal orientation is discarded.

A :class:`~leopard_em.pydantic_models.results.correlation_table.CorrelationTable` keeps
every hypothesis that crossed the search threshold, which makes it possible to pick
peaks in the joint space of position *and* orientation instead. The rule implemented
here is that one detection suppresses another only when they are close in **both**
position and orientation; things that overlap in ``(x, y)`` but disagree about angle
survive as separate peaks.

Defocus is deliberately not part of the suppression neighbourhood. At the step sizes
2DTM typically searches (hundreds of Angstroms) it does not resolve objects that
overlap in projection, so it is maximised over within a peak and reported as a spread.
"""

from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd
import roma
import torch

from leopard_em.analysis.zscore_metric import gaussian_noise_zscore_cutoff

AngularMetric = Literal["phi", "so3", "axis"]

#: Columns describing the cluster of detections each peak was picked out of. A real
#: detection produces a compact lobe of many above-threshold entries; noise produces
#: isolated singletons, so these are a model-independent confidence measure.
CLUSTER_STATISTIC_COLUMNS = [
    "n_detections",
    "sum_z_score",
    "phi_spread_deg",
    "theta_spread_deg",
    "psi_spread_deg",
    "defocus_spread",
    "n_defocus_planes",
]

__all__ = [
    "CLUSTER_STATISTIC_COLUMNS",
    "constrained_zscore_cutoff",
    "find_peaks_orientation_aware",
    "peaks_to_match_template_dataframe",
]


def constrained_zscore_cutoff(
    num_orientations_map: np.ndarray | torch.Tensor,
    num_defocus_map: np.ndarray | torch.Tensor | None = None,
    num_pixel_sizes: int = 1,
    false_positives: float = 1.0,
) -> float:
    """Z-score cutoff for a *constrained* search, from the eligible search size.

    A constrained search only allows some (pixel, orientation, defocus) combinations to
    win, so the number of cross-correlograms that actually competed is far smaller than
    the full grid. Deriving the cutoff from the full grid is needlessly conservative.

    Parameters
    ----------
    num_orientations_map : np.ndarray | torch.Tensor
        Per-pixel count of eligible orientations, shape (H, W). This is the
        ``n_orientations_map`` a ``FilamentConstraint`` installs on the manager.
    num_defocus_map : np.ndarray | torch.Tensor | None
        Per-pixel count of eligible defocus planes, shape (H, W). If None, every pixel
        is assumed to allow a single defocus plane.
    num_pixel_sizes : int
        Number of pixel sizes (Cs values) searched.
    false_positives : float
        Expected number of false positives across the whole search.

    Returns
    -------
    float
        Z-score above which a detection is expected to arise by chance fewer than
        ``false_positives`` times.
    """
    orientations = _to_numpy(num_orientations_map)
    defocus = (
        np.ones_like(orientations)
        if num_defocus_map is None
        else _to_numpy(num_defocus_map)
    )

    num_ccg = float((orientations.astype(np.float64) * defocus).sum()) * num_pixel_sizes
    if num_ccg <= 0:
        raise ValueError(
            "The constraint permits no (pixel, orientation, defocus) combinations, so "
            "no z-score cutoff can be derived."
        )

    return float(gaussian_noise_zscore_cutoff(num_ccg, false_positives))


def _to_numpy(array: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _circular_distance_deg(
    a: np.ndarray | float, b: np.ndarray | float
) -> np.ndarray | float:
    """Smallest absolute angular difference in degrees, wrapping at 360."""
    return np.abs((np.subtract(a, b) + 180.0) % 360.0 - 180.0)


class _AngularNeighbourhood:
    """Decides whether two orientations count as pointing the same way.

    Works on the *unique* orientations present in the detection set rather than on
    every detection: a search grid has far fewer distinct orientations than a
    correlation table has rows, so this keeps the rotation-matrix work small.
    """

    def __init__(
        self, unique_angles: np.ndarray, metric: AngularMetric, radius_deg: float
    ) -> None:
        if metric not in ("phi", "so3", "axis"):
            raise ValueError(
                f"Unknown angular_metric '{metric}'; expected 'phi', 'so3' or 'axis'."
            )

        self.metric = metric
        self.radius_deg = radius_deg
        self.phi = unique_angles[:, 0]
        self._cos_radius = float(np.cos(np.radians(radius_deg)))

        self.rotations: np.ndarray | None = None
        self.axes: np.ndarray | None = None
        if metric in ("so3", "axis"):
            rotations = roma.euler_to_rotmat(
                "ZYZ", torch.as_tensor(unique_angles, dtype=torch.float64), degrees=True
            )
            self.rotations = rotations.numpy().astype(np.float64)
            # The slice extractor rotates the sampling grid, so the object is rotated
            # by the transpose and the template z-axis maps to the third *row*. For a
            # filament template that row is the filament axis direction.
            self.axes = self.rotations[:, 2, :]

    def close_mask(self, index: int, others: np.ndarray) -> np.ndarray:
        """Which of ``others`` lie within the angular radius of ``index``."""
        if self.metric == "phi":
            return np.asarray(
                _circular_distance_deg(self.phi[others], self.phi[index])
                <= self.radius_deg
            )

        if self.metric == "axis":
            assert self.axes is not None
            return np.asarray(self.axes[others] @ self.axes[index] >= self._cos_radius)

        assert self.rotations is not None
        # trace(Ri^T Rj) is the Frobenius inner product; the geodesic angle between two
        # rotations is arccos((trace - 1) / 2).
        trace = np.einsum("nij,ij->n", self.rotations[others], self.rotations[index])
        return np.asarray((trace - 1.0) / 2.0 >= self._cos_radius)

    def is_close(self, index: int, other: int) -> bool:
        """Whether two orientations lie within the angular radius of each other."""
        return bool(self.close_mask(index, np.array([other]))[0])


#: Which of (phi, theta, psi) each metric actually distinguishes. Binning during the
#: pre-reduction must use exactly these, or orientations the metric would have kept
#: apart get silently merged.
_METRIC_ANGLE_COLUMNS: dict[AngularMetric, tuple[int, ...]] = {
    "phi": (0,),
    "so3": (0, 1, 2),
    "axis": (1, 2),
}


def _reduce_to_local_maxima(
    x: np.ndarray,
    y: np.ndarray,
    angles: np.ndarray,
    score: np.ndarray,
    xy_radius_px: float,
    angular_radius_deg: float,
    metric: AngularMetric,
) -> np.ndarray:
    """Keep only the best-scoring detection in each fine (position, angle) cell.

    Greedy suppression over millions of detections is dominated by comparisons between
    near-duplicates that could never win. Binning at *half* the suppression radius
    removes them cheaply and safely: two detections sharing a cell are at most
    ``sqrt(2)/2`` of a radius apart in position and half a radius in each binned angle,
    so the loser would have been suppressed anyway.

    Parameters
    ----------
    x, y : np.ndarray
        Per-detection position in pixels.
    angles : np.ndarray
        Per-detection (phi, theta, psi) in degrees, shape (n, 3). Only the components
        the chosen metric responds to are binned.
    score : np.ndarray
        Per-detection score; the largest in each cell is the one kept.
    xy_radius_px : float
        Positional suppression radius; cells are half this wide.
    angular_radius_deg : float
        Angular suppression radius; cells are half this wide.
    metric : AngularMetric
        Which angular components to bin on, via ``_METRIC_ANGLE_COLUMNS``.

    Returns
    -------
    np.ndarray
        Indices of the retained detections, ascending.

    Notes
    -----
    Binning Euler components independently is not the same as measuring geodesic
    distance, and the two diverge near the theta = 0 gimbal degeneracy where phi and
    psi become redundant. Filament searches sit at theta ~ 90, well away from it.
    """
    tiny = float(np.finfo(np.float64).eps)
    xy_cell = max(xy_radius_px / 2.0, tiny)
    angle_cell = max(angular_radius_deg / 2.0, tiny)

    key_columns = [
        np.floor(x / xy_cell).astype(np.int64),
        np.floor(y / xy_cell).astype(np.int64),
    ]
    key_columns.extend(
        np.floor(angles[:, column] / angle_cell).astype(np.int64)
        for column in _METRIC_ANGLE_COLUMNS[metric]
    )

    keys = np.stack(key_columns, axis=1)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()

    # Highest score wins each cell: visit ascending so better scores overwrite worse.
    order = np.argsort(score, kind="stable")
    winners = np.empty(int(inverse.max()) + 1, dtype=np.int64)
    winners[inverse[order]] = order

    return np.sort(winners)


def _greedy_suppression(
    x: np.ndarray,
    y: np.ndarray,
    score: np.ndarray,
    angle_ids: np.ndarray,
    angular: _AngularNeighbourhood,
    xy_radius_px: float,
    max_peaks: int | None,
) -> np.ndarray:
    """Accept detections best-first, skipping any close in both space and angle."""
    radius_squared = xy_radius_px**2
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    accepted: list[int] = []

    cell_x_all = np.floor(x / xy_radius_px).astype(np.int64)
    cell_y_all = np.floor(y / xy_radius_px).astype(np.int64)

    for raw_index in np.argsort(-score):
        i = int(raw_index)
        cell_x, cell_y = int(cell_x_all[i]), int(cell_y_all[i])

        neighbours: list[int] = []
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                neighbours.extend(grid.get((cell_x + offset_x, cell_y + offset_y), ()))

        if neighbours:
            candidates = np.asarray(neighbours, dtype=np.int64)
            close_in_space = (x[candidates] - x[i]) ** 2 + (
                y[candidates] - y[i]
            ) ** 2 <= radius_squared
            if close_in_space.any():
                contenders = candidates[close_in_space]
                close_in_angle = angular.close_mask(
                    int(angle_ids[i]), angle_ids[contenders]
                )
                if close_in_angle.any():
                    continue

        grid[(cell_x, cell_y)].append(i)
        accepted.append(i)
        if max_peaks is not None and len(accepted) >= max_peaks:
            break

    return np.asarray(accepted, dtype=np.int64)


def _cluster_statistics(
    peaks: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    angle_ids: np.ndarray,
    angles: np.ndarray,
    defocus: np.ndarray,
    z_score: np.ndarray,
    angular: _AngularNeighbourhood,
    xy_radius_px: float,
) -> dict[str, np.ndarray]:
    """Summarise the detections each accepted peak was picked out of.

    Runs against the *full* detection set, not the reduced one, so the counts reflect
    everything the search actually found near that peak.
    """
    radius_squared = xy_radius_px**2

    # Bucket every detection by grid cell once, then look up the 3x3 neighbourhood of
    # each peak with searchsorted rather than building a Python dict of millions of
    # entries.
    cell_x = np.floor(x / xy_radius_px).astype(np.int64)
    cell_y = np.floor(y / xy_radius_px).astype(np.int64)
    stride = int(cell_y.max() - cell_y.min()) + 3
    cell_id = (cell_x - cell_x.min()) * stride + (cell_y - cell_y.min())

    order = np.argsort(cell_id, kind="stable")
    sorted_cell_id = cell_id[order]

    stats: dict[str, list[float]] = {name: [] for name in CLUSTER_STATISTIC_COLUMNS}

    for peak in peaks:
        peak_cell_x = int(cell_x[peak] - cell_x.min())
        peak_cell_y = int(cell_y[peak] - cell_y.min())

        blocks = []
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                target = (peak_cell_x + offset_x) * stride + (peak_cell_y + offset_y)
                lo = np.searchsorted(sorted_cell_id, target, side="left")
                hi = np.searchsorted(sorted_cell_id, target, side="right")
                if hi > lo:
                    blocks.append(order[lo:hi])

        candidates = np.concatenate(blocks) if blocks else np.empty(0, dtype=np.int64)
        if candidates.size:
            in_space = (x[candidates] - x[peak]) ** 2 + (
                y[candidates] - y[peak]
            ) ** 2 <= radius_squared
            candidates = candidates[in_space]
        if candidates.size:
            candidates = candidates[
                angular.close_mask(int(angle_ids[peak]), angle_ids[candidates])
            ]

        member_angles = angles[angle_ids[candidates]]
        peak_angles = angles[angle_ids[peak]]

        stats["n_detections"].append(float(candidates.size))
        stats["sum_z_score"].append(float(z_score[candidates].sum()))
        for column, axis in (
            ("phi_spread_deg", 0),
            ("theta_spread_deg", 1),
            ("psi_spread_deg", 2),
        ):
            deviations = _circular_distance_deg(
                member_angles[:, axis], peak_angles[axis]
            )
            stats[column].append(float(np.sqrt(np.mean(np.square(deviations)))))

        member_defocus = defocus[candidates]
        stats["defocus_spread"].append(
            float(member_defocus.max() - member_defocus.min())
        )
        stats["n_defocus_planes"].append(float(np.unique(member_defocus).size))

    return {name: np.asarray(values) for name, values in stats.items()}


# pylint: disable=too-many-locals
def find_peaks_orientation_aware(
    detections: pd.DataFrame,
    xy_radius_px: float = 12.0,
    angular_radius_deg: float = 15.0,
    score_column: str = "z_score",
    angular_metric: AngularMetric = "so3",
    score_threshold: float | None = None,
    max_peaks: int | None = None,
    compute_cluster_statistics: bool = True,
) -> pd.DataFrame:
    """Pick peaks in position *and* orientation from a table of detections.

    A detection is suppressed only when an already-accepted, better-scoring peak lies
    within ``xy_radius_px`` **and** within ``angular_radius_deg`` of it. Detections that
    share a position but disagree about orientation therefore both survive, which is
    what keeps the near and far wall of a filament -- or two crossing filaments -- from
    annihilating each other the way they do in a per-pixel MIP.

    Parameters
    ----------
    detections : pd.DataFrame
        One row per detection, as returned by
        :meth:`CorrelationTable.to_detections_dataframe`. Must contain ``x``, ``y``,
        ``phi``, ``theta``, ``psi`` and ``score_column``; ``relative_defocus`` is used
        for the defocus spread statistics when present.
    xy_radius_px : float
        Suppression radius in pixels. Bracketed from below by the width of a single
        particle's correlation lobe (split it and one particle becomes several peaks)
        and from above by the closest spacing of distinct particles you need to
        resolve. Note the lobe width is set by the template's *resolution content*,
        not its box size -- a 600-pixel microtubule template still produces a lobe
        only a few pixels across. See ``TemplateLatticeGeometry`` in
        ``leopard_em.analysis.filament_lattice`` for deriving this from a lattice.
    angular_radius_deg : float
        Suppression radius in degrees, interpreted according to ``angular_metric``.
        Bracketed the same way: above the angular width of one particle's lobe, below
        the angular separation of the distinct orientations you need to keep apart
        (for a filament, the azimuthal spacing between protofilaments).
    score_column : str
        Column ranked and suppressed on. Defaults to the z-score.
    angular_metric : {"phi", "so3", "axis"}
        How angular closeness is measured. ``"phi"`` (default) compares only the phi
        angle, which for a filament template is the azimuth around the tube and is
        therefore the axis that separates near wall from far wall. ``"so3"`` uses the
        full geodesic distance between orientations. ``"axis"`` compares only the
        direction the template z-axis projects to, ignoring roll about it.
    score_threshold : float | None
        Discard detections at or below this score before picking. See
        :func:`constrained_zscore_cutoff` for deriving one for a constrained search.
    max_peaks : int | None
        Stop after accepting this many peaks.
    compute_cluster_statistics : bool
        Whether to summarise the detections behind each peak. See
        :data:`CLUSTER_STATISTIC_COLUMNS`.

    Returns
    -------
    pd.DataFrame
        The accepted rows of ``detections``, best-scoring first, with a reset index and
        (optionally) the cluster statistic columns appended.

    Raises
    ------
    ValueError
        If required columns are missing or the radii are not positive.
    """
    required = {"x", "y", "phi", "theta", "psi", score_column}
    missing = sorted(required - set(detections.columns))
    if missing:
        raise ValueError(f"detections is missing required columns: {missing}")
    if xy_radius_px <= 0 or angular_radius_deg <= 0:
        raise ValueError(
            "xy_radius_px and angular_radius_deg must both be positive, got "
            f"{xy_radius_px} and {angular_radius_deg}."
        )

    working = detections
    if score_threshold is not None:
        working = working[working[score_column] > score_threshold]
    if working.empty:
        return working.head(0).reset_index(drop=True)

    x = working["x"].to_numpy(dtype=np.float64)
    y = working["y"].to_numpy(dtype=np.float64)
    score = working[score_column].to_numpy(dtype=np.float64)
    angles_per_detection = working[["phi", "theta", "psi"]].to_numpy(dtype=np.float64)
    defocus = (
        working["relative_defocus"].to_numpy(dtype=np.float64)
        if "relative_defocus" in working.columns
        else np.zeros(len(working))
    )

    unique_angles, angle_ids = np.unique(
        angles_per_detection, axis=0, return_inverse=True
    )
    angle_ids = angle_ids.ravel()
    angular = _AngularNeighbourhood(unique_angles, angular_metric, angular_radius_deg)

    reduced = _reduce_to_local_maxima(
        x,
        y,
        angles_per_detection,
        score,
        xy_radius_px,
        angular_radius_deg,
        angular_metric,
    )
    accepted_in_reduced = _greedy_suppression(
        x[reduced],
        y[reduced],
        score[reduced],
        angle_ids[reduced],
        angular,
        xy_radius_px,
        max_peaks,
    )
    peaks = reduced[accepted_in_reduced]

    result = working.iloc[peaks].reset_index(drop=True)

    if compute_cluster_statistics:
        stats = _cluster_statistics(
            peaks,
            x,
            y,
            angle_ids,
            unique_angles,
            defocus,
            score,
            angular,
            xy_radius_px,
        )
        for name, values in stats.items():
            result[name] = values
        result["n_detections"] = result["n_detections"].astype(int)
        result["n_defocus_planes"] = result["n_defocus_planes"].astype(int)

    return result


def peaks_to_match_template_dataframe(
    peaks: pd.DataFrame,
    manager,
    total_correlations: int | None = None,
    half_template_width_pos_shift: bool = True,
) -> pd.DataFrame:
    """Turn orientation-aware peaks into a table ``refine_template`` accepts.

    Peaks picked from a ``CorrelationTable`` carry positions and orientations but none
    of the run metadata -- optics, CTF, the paths of the statistics maps -- that a
    particle stack needs in order to re-extract and re-score each particle. This renames
    the detection columns to their match-template equivalents and asks the manager for
    the rest, so the result loads as a ``ParticleStackCSV`` and can be handed straight
    to ``RefineTemplateManager`` with no further conversion.

    The cluster statistics are carried through as extra columns rather than dropped,
    since they are the model-independent confidence measure for each peak.

    Parameters
    ----------
    peaks : pd.DataFrame
        Output of :func:`find_peaks_orientation_aware`.
    manager : MatchTemplateManager
        The manager for the run these detections came from; supplies the optics group,
        template and micrograph paths, and statistics-map paths.
    total_correlations : int | None
        Size of the search space. Defaults to the manager's recorded total projections.
    half_template_width_pos_shift : bool
        Passed through to the manager; leave True when peak positions refer to the
        top-left corner of the template, as correlation-table positions do.

    Returns
    -------
    pd.DataFrame
        Columns in ``MULTIPEAK_DF_COLUMN_ORDER``.

    Raises
    ------
    ValueError
        If ``peaks`` is missing columns the conversion needs.
    """
    from leopard_em.pydantic_models.formats import (
        MULTIPEAK_DF_COLUMN_ORDER,
    )

    required = {"x", "y", "phi", "theta", "psi", "correlation_value"}
    missing = sorted(required - set(peaks.columns))
    if missing:
        raise ValueError(f"peaks is missing required columns: {missing}")

    converted = peaks.copy()
    converted["pos_x"] = converted["x"]
    converted["pos_y"] = converted["y"]
    converted["mip"] = converted["correlation_value"]
    if "z_score" in converted.columns:
        converted["scaled_mip"] = converted["z_score"]

    if total_correlations is None:
        total_correlations = int(
            getattr(manager.match_template_result, "total_projections", 0) or 0
        )
    converted["total_correlations"] = total_correlations

    return manager.annotate_dataframe_metadata(
        converted,
        half_template_width_pos_shift=half_template_width_pos_shift,
        column_order=MULTIPEAK_DF_COLUMN_ORDER,
    )
