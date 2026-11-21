"""Analytic null of a Leopard-EM search followed by refine_template.

:func:`predict_refine_template_null` predicts the expected number of refined null
peaks at or above each refined z-score per micrograph, for candidates of a
match_template search at or above a candidate threshold refined with refine_template's
local grid (:mod:`.refine`). The refinement window is refine_template's own:

* orientation offsets from :class:`~leopard_em.pydantic_models.config.
  RefineOrientationConfig`, composed with the candidate's rotation as refine_template
  composes them (offset @ R);
* defocus offsets about the candidate's plane from the refine's
  :class:`~leopard_em.pydantic_models.config.DefocusSearchConfig` (at a coarser step
  than refine's own when asked: 40 and 80 A steps give the same threshold within
  0.02 z on xenon_131, where refine steps 20 A);
* positions within +-``shift_radius`` pixels.

:func:`refine_window` reads them from a refine_template configuration.
"""

import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import roma
import torch
import yaml  # type: ignore

from leopard_em.pydantic_models.config.defocus_search import DefocusSearchConfig
from leopard_em.pydantic_models.config.orientation_search import (
    RefineOrientationConfig,
)
from leopard_em.utils.ctf_utils import calculate_ctf_filter_stack

from .lattice import AnalyticNullPrediction
from .match_template import _LeopardGrid, _LeopardSearch
from .refine import (
    RefinedNullPrediction,
    RefinementWindow,
    predict_refined_null_rate,
    sample_refinement_gain,
)


def refine_window(
    config: dict[str, Any] | str | Path,
    defocus_step: float | None = 40.0,
) -> tuple[torch.Tensor, np.ndarray]:
    """Orientation and defocus offsets of a refine_template configuration.

    Parameters
    ----------
    config : dict[str, Any] | str | Path
        A refine_template YAML (path) or its dictionary. Only its
        ``orientation_refinement_config``, ``defocus_refinement_config`` and
        ``pixel_size_refinement_config`` are read, so its particle stack need not
        exist yet.
    defocus_step : float | None
        Defocus step for the model, A, when coarser than the refine's own. Default
        40. None: the refine's own step.

    Returns
    -------
    tuple[torch.Tensor, np.ndarray]
        ZYZ Euler offsets (n, 3) in degrees, and relative defocus offsets in A.
    """
    if not isinstance(config, dict):
        with open(config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    assert isinstance(config, dict)
    pixel = config.get("pixel_size_refinement_config") or {}
    if pixel.get("enabled", False):
        raise NotImplementedError(
            "The analytic refined null does not model a pixel size refinement."
        )
    orientation = RefineOrientationConfig(
        **(config.get("orientation_refinement_config") or {})
    )
    defocus = DefocusSearchConfig(**(config.get("defocus_refinement_config") or {}))
    offsets = orientation.euler_angles_offsets.detach().cpu().double()
    if not defocus.enabled:
        values = np.zeros(1)
    elif defocus_step is not None and defocus_step > defocus.defocus_step:
        # a coarser step over the same range, symmetric about zero
        step = float(defocus_step)
        lo = math.ceil(defocus.defocus_min / step - 1e-9)
        hi = math.floor(defocus.defocus_max / step + 1e-9)
        values = step * np.arange(min(lo, 0), max(hi, 0) + 1, dtype=float)
    else:
        values = defocus.defocus_values.detach().cpu().double().numpy().reshape(-1)
    return offsets, values


def _zero_index(values: np.ndarray, what: str) -> int:
    j = int(np.argmin(np.abs(values).reshape(len(values), -1).sum(axis=1)))
    if float(np.abs(values[j]).sum()) > 1e-6:
        raise ValueError(f"The refinement's {what} offsets must include zero.")
    return j


# pylint: disable=too-many-locals,too-many-arguments
def predict_refine_template_null(
    backend_kwargs: dict[str, Any],
    optics_group: Any,
    search: AnalyticNullPrediction,
    orientation_offsets: torch.Tensor | np.ndarray,
    defocus_offsets: np.ndarray,
    candidate_threshold: float,
    *,
    shift_radius: int = 1,
    condition_radius: int = 2,
    n_candidates: int = 8,
    n_draws: int = 200_000,
    n_mean_samples: int = 3000,
    heights: np.ndarray | None = None,
    batch_size: int = 11,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> RefinedNullPrediction:
    """Predict the null peak rate after refine_template, without a null search.

    Parameters
    ----------
    backend_kwargs : dict[str, Any]
        The match_template search's
        ``MatchTemplateManager.make_backend_core_function_kwargs`` dictionary.
    optics_group : OpticsGroup
        The micrograph's optics group (for the CTF at the window's defocus offsets).
    search : AnalyticNullPrediction
        The search's prediction (:func:`.match_template.predict_match_template_null`).
    orientation_offsets : torch.Tensor | np.ndarray
        Refinement's ZYZ Euler offsets (n, 3), degrees, including zero
        (:func:`refine_window`).
    defocus_offsets : np.ndarray
        Refinement's relative defocus offsets, A, including zero.
    candidate_threshold : float
        Search z-score at or above which candidates are refined.
    shift_radius : int
        Positions refinement scores about each candidate, +-pixels. Default 1.
    condition_radius : int
        Positions at which the candidate beat the search's grid points, +-pixels.
        Default 2.
    n_candidates : int
        Grid points sampled as candidates, spread evenly over the defocus planes.
        Default 8.
    n_draws : int
        Gaussian draws per candidate. Default 200,000.
    n_mean_samples : int
        Orientations for the mean template. Default 3000 (with the same ``seed``,
        the search prediction's mean template).
    heights : np.ndarray | None
        Candidate heights to sample the gain at. Default 5 to 9 in steps of 0.5.
    batch_size : int
        Orientations projected per batch. Default 11.
    seed : int
        Seed for sampling. Default 0.
    device : torch.device | str
        Device. Default "cpu"; a GPU is ~100x faster.

    Returns
    -------
    RefinedNullPrediction
        Call ``.zscore_cutoff(false_positives)`` for the refined threshold.
    """
    start = time.time()
    rng = np.random.default_rng(seed)
    setup = _LeopardSearch(backend_kwargs, device)
    mean_template = setup.mean_template(rng, n_mean_samples)
    grid = _LeopardGrid(setup.euler)
    k = min(n_candidates, grid.usable.size)
    rows = setup.euler[rng.choice(grid.usable, size=k, replace=False)]

    offsets = torch.as_tensor(np.asarray(orientation_offsets, np.float64))
    df = np.asarray(defocus_offsets, float).reshape(-1)
    zero_offset = _zero_index(offsets.numpy(), "orientation")
    zero_df = _zero_index(df, "defocus")
    offset_rot = roma.euler_to_rotmat("ZYZ", offsets.float(), degrees=True).to(device)
    n_offsets = int(offsets.shape[0])

    def rotations(row: np.ndarray) -> torch.Tensor:
        angles = torch.as_tensor(np.asarray(row, np.float64)).float().reshape(-1, 3)
        return roma.euler_to_rotmat("ZYZ", angles.to(device), degrees=True)

    def windows() -> Iterator[RefinementWindow]:
        for i, row in enumerate(rows):
            plane = (i * setup.n_planes) // len(rows)  # spread over the planes
            r0 = rotations(row)[0]
            # refine_template composes the offsets on the left: offset @ R
            window_rot = roma.rotmat_composition(
                (offset_rot, r0[None].expand(n_offsets, 3, 3))
            )
            ctf = calculate_ctf_filter_stack(
                template_shape=(setup.box[0], setup.box[0]),
                optics_group=optics_group,
                defocus_offsets=torch.as_tensor(
                    setup.defocus_values[plane] + df, dtype=torch.float32
                ),
                pixel_size_offsets=torch.tensor([0.0], dtype=torch.float32),
            )
            filters = ctf[:1].to(device) * setup.whitening  # (1, n_df, h, w // 2 + 1)
            window = torch.cat(
                [
                    setup.project_rotations(window_rot[j : j + batch_size], filters)
                    for j in range(0, n_offsets, batch_size)
                ],
                dim=1,
            )  # (n_df, n_offsets, ...): index df * n_offsets + offset
            window = window.reshape(-1, *window.shape[2:])
            neighbour_rows = [
                step[0]
                for axis in grid.neighbours(row[None])
                for step, valid in (
                    (axis.steps[0], axis.valid_plus),
                    (axis.steps[1], axis.valid_minus),
                )
                if valid is None or bool(valid[0])
            ]
            parts = []
            if neighbour_rows:
                parts.append(setup.project(np.array(neighbour_rows))[plane])
            adjacent = [p for p in (plane - 1, plane + 1) if 0 <= p < setup.n_planes]
            if adjacent:
                parts.append(setup.project(row[None])[adjacent, 0])
            yield RefinementWindow(
                window=window,
                centre=zero_df * n_offsets + zero_offset,
                search=torch.cat(parts) if parts else window[:0],
            )

    gain = sample_refinement_gain(
        windows(),
        setup.weighting,
        mean_template,
        shift_radius=shift_radius,
        condition_radius=condition_radius,
        heights=heights,
        n_draws=n_draws,
        seed=seed,
    )
    return predict_refined_null_rate(
        search,
        gain,
        candidate_threshold,
        details={
            "engine": "leopard_em.refine_template",
            "orientation_offsets": n_offsets,
            "defocus_offsets": df.tolist(),
            "shift_radius": int(shift_radius),
            "condition_radius": int(condition_radius),
            "n_candidates": k,
            "n_mean_samples": mean_template.n_samples // setup.n_planes,
            "seed": seed,
            "seconds": time.time() - start,
        },
    )
