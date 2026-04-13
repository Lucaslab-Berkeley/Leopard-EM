"""Program for running sectored SO(3) search using 2D template matching.

The sphere is split into HEALPix sectors via ``get_sectored_euler_angles`` from
``torch-so3`` (>= 0.2.2).  Each sector runs an independent masked-MIP
``core_match_template`` call so that every orientation in the sector contributes
to the per-pixel mean and variance (for proper z-score normalisation), while
orientations that fall outside the user-defined or symmetry-derived angular window
are masked from the MIP and from the peak-significance multiplicity.

Results from all sectors are merged with ``merge_runs_independent_zscore``
(using *full* per-sector totals for z-score normalisation) and peak extraction
uses the *eligible-only* total (summed across sectors) as the multiplicity.

Requires: torch-so3 >= 0.2.2
"""

import time
from typing import Any

import torch
from torch_so3 import get_sectored_euler_angles

from leopard_em.backend.core_match_template import core_match_template
from leopard_em.backend.process_results import (
    decode_global_search_index,
    merge_runs_independent_zscore,
)
from leopard_em.pydantic_models.config.orientation_search import OrientationSearchConfig
from leopard_em.pydantic_models.managers import MatchTemplateManager

#######################################
### Editable parameters for program ###
#######################################

# Edit your YAML file to configure the match template program.
YAML_CONFIG_PATH = "/path/to/match-template-configuration.yaml"

# Path where the picked peaks will be written.
DATAFRAME_OUTPUT_PATH = "/path/to/sectored-match-template-results.csv"

# Number of orientations to cross-correlate simultaneously.
ORIENTATION_BATCH_SIZE = 8

# HEALPix coarse resolution: n_sectors = 12 * NSIDE_COARSE**2
# Examples: NSIDE_COARSE=1 → 12 sectors, 2 → 48, 4 → 192
NSIDE_COARSE = 1

# HEALPix fine resolution inside each sector.
# If None, inferred automatically from OrientationSearchConfig.theta_step.
NSIDE_FINE: int | None = None

##############################################################
### Main function called to run the match template program ###
##############################################################


def _build_sector_euler_tables(
    cfg: OrientationSearchConfig,
    nside_coarse: int,
    nside_fine: int | None,
) -> torch.Tensor:
    """Call get_sectored_euler_angles using bounds from the orientation config.

    Parameters
    ----------
    cfg : OrientationSearchConfig
        Orientation search configuration providing angular bounds and step sizes.
    nside_coarse : int
        HEALPix nside for the coarse sector grid.
    nside_fine : int or None
        HEALPix nside for the fine grid within each sector.  ``None`` infers
        the value from ``cfg.theta_step``.

    Returns
    -------
    torch.Tensor
        Shape ``(n_kept, n_per_sector, 3)`` with columns ``(phi, theta, psi)``.
    """
    phi_min, phi_max, theta_min, theta_max, psi_min, psi_max = (
        cfg.effective_euler_bounds
    )
    return get_sectored_euler_angles(
        nside_coarse=nside_coarse,
        nside_fine=nside_fine,
        theta_step=cfg.theta_step,
        psi_step=cfg.psi_step,
        psi_min=psi_min,
        psi_max=psi_max,
        theta_min=theta_min,
        theta_max=theta_max,
        phi_min=phi_min,
        phi_max=phi_max,
    )


def _decode_poses_and_defocus_from_sectors(
    winner: torch.Tensor,
    bgi: torch.Tensor,
    sector_euler_tables: list[torch.Tensor],
    pixel_values: torch.Tensor,
    defocus_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct per-pixel phi/theta/psi/defocus from the winning sector runs.

    Uses :func:`decode_global_search_index` with each sector's euler table so
    results match a single ``core_match_template`` call (same index layout).

    Parameters
    ----------
    winner : torch.Tensor
        ``winner_run_index`` from ``merge_runs_independent_zscore``,
        shape ``(H, W)`` int32.
    bgi : torch.Tensor
        ``best_global_index`` from the merge, shape ``(H, W)`` int32.
    sector_euler_tables : list[torch.Tensor]
        Per-sector euler angle tables, each shape ``(n_per_sector, 3)`` with
        columns ``(phi, theta, psi)``.
    pixel_values : torch.Tensor
        Same ``pixel_values`` passed to ``core_match_template`` (shape ``(num_Cs,)``).
    defocus_values : torch.Tensor
        Same ``defocus_values`` passed to ``core_match_template``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(phi_map, theta_map, psi_map, defocus_map)`` each shape ``(H, W)``
        float32.  ``defocus_map`` is in Angstroms (relative defocus offsets),
        same as ``MatchTemplateResult.relative_defocus`` from a normal run.
    """
    phi_map = torch.zeros_like(bgi, dtype=torch.float32)
    theta_map = torch.zeros_like(bgi, dtype=torch.float32)
    psi_map = torch.zeros_like(bgi, dtype=torch.float32)
    defocus_map = torch.zeros_like(bgi, dtype=torch.float32)

    for s, euler_s in enumerate(sector_euler_tables):
        mask = winner == s
        if not mask.any():
            continue
        sub_bgi = bgi[mask].long()
        phi, theta, psi, defocus = decode_global_search_index(
            sub_bgi,
            pixel_values,
            defocus_values,
            euler_s,
        )
        phi_map[mask] = phi.to(torch.float32)
        theta_map[mask] = theta.to(torch.float32)
        psi_map[mask] = psi.to(torch.float32)
        defocus_map[mask] = defocus.to(torch.float32)

    return phi_map, theta_map, psi_map, defocus_map


def main() -> None:  # pylint: disable=too-many-locals
    """Run the sectored match template search."""
    manager = MatchTemplateManager.from_yaml(YAML_CONFIG_PATH)

    # Sectored mode requires a plain OrientationSearchConfig (not Multiple).
    if not isinstance(manager.orientation_search_config, OrientationSearchConfig):
        raise TypeError(
            "Sectored match template requires a plain OrientationSearchConfig, "
            "not MultipleOrientationConfig."
        )
    cfg = manager.orientation_search_config

    print("Loaded configuration.")
    print(f"Building sectored euler angles (nside_coarse={NSIDE_COARSE})...")
    sectored = _build_sector_euler_tables(cfg, NSIDE_COARSE, NSIDE_FINE)
    n_kept, n_per_sector, _ = sectored.shape
    print(f"  {n_kept} sectors, {n_per_sector} orientations each.")

    # Build shared preprocessing kwargs (image DFT, template DFT, CTF stack, etc.).
    # We call with the manager's default backend (non-masked) so no
    # orientation_eligible validation fires here; we will inject euler_angles
    # and orientation_eligible per-sector below and call core_match_template
    # directly with the masked-MIP backend.
    print("Preprocessing image and template...")
    base_kwargs: dict[str, Any] = manager.make_backend_core_function_kwargs()

    print("Starting sectored search...")
    start_time = time.time()

    runs: list[dict[str, Any]] = []
    sector_euler_tables: list[torch.Tensor] = []
    full_totals: list[int] = []
    eligible_totals: list[int] = []

    for s in range(n_kept):  # iterate over each sector
        euler_s = sectored[s].float()  # (n_per_sector, 3) — (phi, theta, psi)
        eligible_s = cfg.orientation_eligible_mask(euler_s)  # (n_per_sector,) bool

        # Build per-sector kwargs: inherit all preprocessing, override angles.
        sector_kwargs = dict(base_kwargs)
        sector_kwargs["euler_angles"] = euler_s
        sector_kwargs["orientation_eligible"] = eligible_s.float()

        n_eligible = int(eligible_s.sum())
        print(
            f"  Sector {s + 1}/{n_kept}: "
            f"{n_eligible}/{n_per_sector} eligible orientations."
        )

        results = core_match_template(
            **sector_kwargs,
            orientation_batch_size=ORIENTATION_BATCH_SIZE,
            num_cuda_streams=manager.computational_config.num_cpus,
            backend="streamed_masked_mip",
        )
        runs.append(results)
        sector_euler_tables.append(euler_s)
        full_totals.append(results["total_projections"])
        eligible_totals.append(results["total_mip_eligible_projections"])

    elapsed = time.time() - start_time
    print(f"Sector loop done in {time.strftime('%H:%M:%S', time.gmtime(elapsed))}.")

    # --- Merge ---
    # MUST use full totals: correlation_sum was accumulated over ALL sector
    # orientations, so mean = csum / n_full is the only correct normalisation.
    print("Merging sector results...")
    merged = merge_runs_independent_zscore(
        runs,
        total_correlation_positions_per_run=full_totals,
    )

    winner = merged["winner_run_index"]  # (H, W) int32
    bgi = merged["best_global_index"]  # (H, W) int32, local to winning sector

    # Decode phi/theta/psi/defocus using the winning sector's euler table and
    # the same index layout as core_match_template (see decode_global_search_index).
    phi_map, theta_map, psi_map, defocus_map = _decode_poses_and_defocus_from_sectors(
        winner,
        bgi,
        sector_euler_tables,
        base_kwargs["pixel_values"],
        base_kwargs["defocus_values"],
    )

    # --- Populate MatchTemplateResult ---
    result = manager.match_template_result
    result.mip = merged["mip"]
    result.scaled_mip = merged["scaled_mip"]
    result.correlation_average = merged["correlation_mean"]
    result.correlation_variance = merged["correlation_variance"]
    result.orientation_phi = phi_map
    result.orientation_theta = theta_map
    result.orientation_psi = psi_map

    # total_projections: full search (used for record-keeping only).
    result.total_projections = sum(full_totals)
    # total_mip_eligible_projections: peak multiplicity — only orientations that
    # were actual MIP candidates (eligible, across all sectors).
    result.total_mip_eligible_projections = sum(eligible_totals)

    result.relative_defocus = defocus_map

    # Apply valid-mode cropping (remove template edge artefacts).
    if manager.template_volume is not None:
        nx = manager.template_volume.shape[-1]
        result.apply_valid_cropping((nx, nx))

    # Export MRC statistics files.
    result.export_results()

    # --- Peak extraction ---
    # locate_peaks automatically uses total_mip_eligible_projections as the
    # multiplicity when it is > 0 (see MatchTemplateResult.locate_peaks).
    print("Exporting results...")
    df = manager.results_to_dataframe()
    df.to_csv(DATAFRAME_OUTPUT_PATH, index=True)

    print(
        f"Done! "
        f"Total eligible projections (peak multiplicity): {sum(eligible_totals):,}  "
        f"(full search: {sum(full_totals):,})"
    )


# NOTE: `if __name__ == "__main__"` guard is required for multiprocessing.
if __name__ == "__main__":
    main()
