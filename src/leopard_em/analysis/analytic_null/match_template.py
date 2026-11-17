"""Analytic null of a Leopard-EM match_template search.

:func:`predict_match_template_null` predicts the expected number of distinct null
peaks at or above each z-score per micrograph, and so the threshold for a target
number of false positives, from the search's own inputs: the template, the CTF and
whitening filters, the orientation and defocus grids, and the preprocessed image's
power spectrum. No search is run. The projections are built with the same backend
steps the search uses (:func:`~leopard_em.backend.cross_correlation.
project_filtered_templates` and
:func:`~leopard_em.backend.utils.normalize_template_projection`).

The input is the dictionary returned by
:meth:`~leopard_em.pydantic_models.managers.MatchTemplateManager.
make_backend_core_function_kwargs`; this module does not import the managers.
"""

import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import roma
import torch

from leopard_em.backend.cross_correlation import project_filtered_templates
from leopard_em.backend.utils import normalize_template_projection

from .field import (
    AxisNeighbours,
    PowerWeighting,
    estimate_kappa,
    estimate_mean_template,
    sample_lattice,
)
from .lattice import AnalyticNullPrediction, predict_null_rate

DEFAULT_THRESHOLDS = np.round(np.arange(2.0, 10.0, 0.01), 3)


def _grid_step(values: np.ndarray) -> float:
    """Smallest spacing between distinct values of a grid axis (degrees)."""
    unique = np.unique(np.round(values, 4))
    if unique.size < 2:
        return 0.0
    return float(np.diff(unique).min())


class _LeopardGrid:  # pylint: disable=too-few-public-methods
    """Leopard's uniform ZYZ grid: neighbours of grid rows along psi, theta, phi."""

    def __init__(self, euler: np.ndarray):
        self.euler = euler
        theta = np.round(euler[:, 1], 4)
        self.theta_values, inverse = np.unique(theta, return_inverse=True)
        ring_size = np.array(
            [
                len(np.unique(np.round(euler[inverse == i, 0], 4)))
                for i in range(len(self.theta_values))
            ]
        )
        self.phi_step = {
            float(t): 360.0 / n
            for t, n in zip(self.theta_values, ring_size, strict=False)
        }
        self.psi_values = np.unique(np.round(euler[:, 2], 4))
        self.psi_step = _grid_step(euler[:, 2])
        self.theta_step = _grid_step(euler[:, 1])
        self.psi_wraps = (
            self.psi_values.max() - self.psi_values.min() + self.psi_step > 359.0
        )
        self.theta_range = (
            float(self.theta_values.min()),
            float(self.theta_values.max()),
        )
        full_theta = self.theta_range[0] <= 0.0 and self.theta_range[1] >= 180.0
        usable = np.ones(len(euler), bool)
        if full_theta and self.theta_step > 0:
            # avoid rows next to the poles, where the phi ring degenerates
            usable = np.sin(np.radians(euler[:, 1])) > np.sin(
                np.radians(2 * self.theta_step)
            )
        self.usable = np.flatnonzero(usable)

    def _in_range(self, column: int, values: np.ndarray) -> np.ndarray:
        if column == 0 or (column == 2 and self.psi_wraps):
            return np.ones(len(values), bool)
        if column == 1:
            lo, hi = self.theta_range
        else:
            lo, hi = float(self.psi_values.min()), float(self.psi_values.max())
        return (values >= lo - 1e-3) & (values <= hi + 1e-3)

    def neighbours(self, rows: np.ndarray) -> list[AxisNeighbours]:
        """Grid rows one and two steps away along psi, theta and phi."""
        out = []
        for name, column in (("psi", 2), ("theta", 1), ("phi", 0)):
            if column == 0:
                step = np.array(
                    [self.phi_step[float(np.round(t, 4))] for t in rows[:, 1]]
                )
            else:
                step = np.full(
                    len(rows), self.psi_step if column == 2 else self.theta_step
                )
            if not np.any(step > 0):
                continue
            steps = []
            valid = []
            for m in (1, -1, 2, -2):
                shifted = rows.copy()
                shifted[:, column] += m * step
                steps.append(shifted)
                valid.append(self._in_range(column, shifted[:, column]))
            out.append(
                AxisNeighbours(
                    name=name,
                    steps=(steps[0], steps[1], steps[2], steps[3]),
                    valid_plus=valid[0],
                    valid_minus=valid[1],
                )
            )
        return out


def _batches(rows: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


# pylint: disable=too-many-locals,too-many-arguments
def predict_match_template_null(
    backend_kwargs: dict[str, Any],
    valid_area: float,
    *,
    mask_radius: float = 10.0,
    n_mean_samples: int = 3000,
    n_kappa_pairs: int = 64,
    n_lattice_samples: int = 128,
    batch_size: int = 4,
    thresholds: np.ndarray | None = None,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> AnalyticNullPrediction:
    """Predict the null peak rate of a match_template search, without a search.

    Parameters
    ----------
    backend_kwargs : dict[str, Any]
        The dictionary from ``MatchTemplateManager.make_backend_core_function_kwargs``
        (keys ``image_dft``, ``template_dft``, ``ctf_filters``,
        ``whitening_filter_template``, ``euler_angles``, ``defocus_values``,
        ``pixel_values`` and ``mag_matrix``).
    valid_area : float
        Number of valid pixels peaks are counted in (the z-score map's area, less any
        border or masked region).
    mask_radius : float
        Peak suppression radius, pixels. Default 10.
    n_mean_samples : int
        Orientations sampled for the mean template. Default 3000.
    n_kappa_pairs : int
        Pairs of orientation batches sampled for kappa. Default 64.
    n_lattice_samples : int
        Grid points sampled for the lattice correlations. Default 128.
    batch_size : int
        Orientations per batch (each uses every defocus plane). Default 4.
    thresholds : np.ndarray | None
        Thresholds to evaluate. Default 2 to 10 in steps of 0.01.
    seed : int
        Seed for sampling orientations. Default 0.
    device : torch.device | str
        Device for the projections and inner products. Default "cpu".

    Returns
    -------
    AnalyticNullPrediction
        Call ``.zscore_cutoff(false_positives)`` for the predicted threshold.
    """
    start = time.time()
    pixel_values = backend_kwargs.get("pixel_values")
    if pixel_values is not None and torch.as_tensor(pixel_values).numel() > 1:
        raise NotImplementedError(
            "The analytic null does not model a pixel size search yet."
        )
    rng = np.random.default_rng(seed)
    image_dft = backend_kwargs["image_dft"]
    image_shape = (int(image_dft.shape[-2]), 2 * (int(image_dft.shape[-1]) - 1))
    template_dft = backend_kwargs["template_dft"].to(device)
    box = (int(template_dft.shape[1]), int(template_dft.shape[2]) * 2 - 2)
    filters = (
        backend_kwargs["ctf_filters"][:1] * backend_kwargs["whitening_filter_template"]
    ).to(device)  # (1, planes, h, w // 2 + 1)
    mag_matrix = backend_kwargs.get("mag_matrix")
    if mag_matrix is not None:
        mag_matrix = mag_matrix.to(device)
    euler = backend_kwargs["euler_angles"].detach().cpu().double().numpy()
    n_planes = int(filters.shape[1])
    weighting = PowerWeighting(image_dft, image_shape, box).to(device)

    def project(rows: np.ndarray) -> torch.Tensor:
        angles = torch.from_numpy(np.asarray(rows, np.float64)).float().to(device)
        rot = roma.euler_to_rotmat("ZYZ", angles, degrees=True)
        p = project_filtered_templates(template_dft, rot, filters, mag_matrix)[0]
        p = normalize_template_projection(p, box, image_shape)
        return weighting.transform(p)

    sub = max(1, 16 // n_planes)
    n_mean = min(n_mean_samples, len(euler))
    mean_rows = euler[rng.choice(len(euler), size=n_mean, replace=False)]
    mean_template = estimate_mean_template(project, weighting, _batches(mean_rows, sub))

    def pairs() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for _ in range(n_kappa_pairs):
            a = euler[rng.choice(len(euler), size=batch_size, replace=False)]
            b = euler[rng.choice(len(euler), size=batch_size, replace=False)]
            yield a, b

    kappa = estimate_kappa(project, weighting, mean_template.mean, pairs())

    grid = _LeopardGrid(euler)
    k = min(n_lattice_samples, grid.usable.size)
    lattice_rows = euler[rng.choice(grid.usable, size=k, replace=False)]
    lattice_batch = batch_size if n_planes > 1 else 2 * batch_size
    samples = sample_lattice(
        project,
        weighting,
        mean_template.mean,
        _batches(lattice_rows, lattice_batch),
        grid.neighbours,
    )

    n_points = float(valid_area) * len(euler) * n_planes
    return predict_null_rate(
        samples,
        kappa,
        mean_template.rho_bar,
        n_points,
        valid_area,
        mask_radius,
        DEFAULT_THRESHOLDS if thresholds is None else thresholds,
        details={
            "engine": "leopard_em.match_template",
            "orientations": len(euler),
            "planes": n_planes,
            "image_shape": list(image_shape),
            "box_shape": list(box),
            "psi_step": grid.psi_step,
            "theta_step": grid.theta_step,
            "n_mean_samples": n_mean,
            "n_kappa_pairs": n_kappa_pairs,
            "n_lattice_samples": k,
            "seed": seed,
            "weighting_negative_fraction": weighting.negative_fraction,
            "seconds": time.time() - start,
        },
    )
