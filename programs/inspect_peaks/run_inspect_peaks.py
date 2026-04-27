"""Example: local cross-correlation maps over orientation (peak inspection).

Peak inspection runs the same correlation engine as ``refine_template``, but returns
**all** local maps instead of only the best peak. This script runs the full stack,
then reports the best few orientations for the top-scoring particle.
"""

# NOTE: The ``if __name__ == "__main__"`` guard is required for multiprocessing.
import time
from typing import Literal

import pandas as pd
import torch

from leopard_em.pydantic_models.managers import PeakInspectionManager

#######################################
### Editable parameters for program ###
#######################################

# Edit the YAML the same way you would for refine template (see example config next
# to this script, identical schema to `refine_template_example_config.yaml`).
YAML_CONFIG_PATH = "/path/to/inspect_peaks_configuration.yaml"

# Batched orientations per GPU call — lower if you run out of memory.
CORRELATION_BATCH_SIZE = 32

# Output mode for inspect backend.
# - "cross_correlation": returns (N, n_px, n_def, n_orient, H, W)
# - "frc": returns (frc_tensor, frequency_bins), where frc_tensor is
#   (N, n_px, n_def, n_orient, n_freq)
OUTPUT_MODE: Literal["cross_correlation", "frc"] = "cross_correlation"

# How many orientation hypotheses to list for the “top” particle
TOP_K_ORIENTATIONS = 5

# How we pick the “top” row in the particle CSV (try in order, higher = better)
SCORE_COLUMNS = (
    "refined_scaled_mip",
    "refined_mip",
    "scaled_mip",
    "mip",
)

##########################################################################
### Intuition: shape of the tensor returned by ``run_peak_inspection`` ###
##########################################################################
# Let ``T`` = ``manager.run_peak_inspection(...)``.
#
#   T.shape = (N, n_px, n_def, n_orient, H, W)
#             |   |     |     |         |  |
#             |   |     |     |         |  +-- valid same-size CC map (x)
#             |   |     |     |         +----- valid same-size CC map (y)
#             |   |     |     +---------------- local Euler *offsets* (phi,θ,ψ)
#             |   |     +------------------------ relative defocus search index
#             |   +------------------------------ relative pixel-size search index
#             +---------------------------------- particle (row in stack / CSV)
#
# Values are **real-space cross-correlations** (not z-scored) to match the
# non-reducing “inspect” path. Offsets in ``orientation_refinement_config`` define
# the ``n_orient`` grid; ``H``/``W`` are the “valid” map size (template in image).
#
# This example: take one particle (best score in the CSV), then for each
# orientation index take the best CC over (pixel offset, defocus, y, x) and show
# the top K orientations.
##########################################################################


def _stack_row_for_best_particle(df: pd.DataFrame) -> int:
    for col in SCORE_COLUMNS:
        if col in df.columns:
            return int(df[col].to_numpy().argmax())
    raise ValueError(f"No known score column found; expected one of {SCORE_COLUMNS!r}.")


def _print_top_orientations(
    inspection: torch.Tensor,
    euler_offsets: torch.Tensor,
    stack_row: int,
) -> None:
    """For one particle, print the best K orientations by max cross-correlation."""
    # (n_px, n_def, n_orient, H, W) for this particle
    one = inspection[stack_row]
    # Best CC for each local orientation, pooling over (pixel, defocus, y, x)
    best_cc_per_orient = one.amax(dim=(0, 1, 3, 4)).float().cpu()
    n_or = best_cc_per_orient.numel()
    k = min(TOP_K_ORIENTATIONS, n_or)
    top_vals, top_idx = torch.topk(best_cc_per_orient, k=k, largest=True, sorted=True)
    euler = euler_offsets.detach().float().cpu()

    print(
        f"\nTop {k} local-orientation hypotheses (ZYZ offset degrees) by max CC — "
        f"stack row {stack_row}\n"
    )
    for rank, (v, oi) in enumerate(
        zip(top_vals.tolist(), top_idx.tolist(), strict=True),
        1,
    ):
        phi = euler[oi, 0].item()
        theta = euler[oi, 1].item()
        psi = euler[oi, 2].item()
        print(
            f"  {rank}. orient_idx={oi:4d}  "
            f"max_cc={v:.6f}  "
            f"Δφ={phi:7.3f}°  Δθ={theta:7.3f}°  Δψ={psi:7.3f}°"
        )


def _print_top_orientations_frc(
    frc_tensor: torch.Tensor,
    frequency_bins: torch.Tensor,
    euler_offsets: torch.Tensor,
    stack_row: int,
) -> None:
    """For one particle, print top K orientations by max FRC across frequencies."""
    # (n_px, n_def, n_orient, n_freq) for this particle
    one = frc_tensor[stack_row]
    best_frc_per_orient = one.amax(dim=(0, 1, 3)).float().cpu()
    n_or = best_frc_per_orient.numel()
    k = min(TOP_K_ORIENTATIONS, n_or)
    top_vals, top_idx = torch.topk(best_frc_per_orient, k=k, largest=True, sorted=True)
    euler = euler_offsets.detach().float().cpu()
    freq = frequency_bins.detach().float().cpu()

    print(
        f"\nTop {k} local-orientation hypotheses (ZYZ offset degrees) by max FRC — "
        f"stack row {stack_row}\n"
    )
    print(f"FRC frequency range: [{freq.min().item():.4f}, {freq.max().item():.4f}]")
    for rank, (v, oi) in enumerate(
        zip(top_vals.tolist(), top_idx.tolist(), strict=True),
        1,
    ):
        phi = euler[oi, 0].item()
        theta = euler[oi, 1].item()
        psi = euler[oi, 2].item()
        print(
            f"  {rank}. orient_idx={oi:4d}  "
            f"max_frc={v:.6f}  "
            f"Δφ={phi:7.3f}°  Δθ={theta:7.3f}°  Δψ={psi:7.3f}°"
        )


def main() -> None:
    """Run peak inspection and print a short summary for the best particle row."""
    manager = PeakInspectionManager.from_yaml(YAML_CONFIG_PATH)
    df = pd.read_csv(manager.particle_stack.df_path)
    n = len(df)
    if n == 0:
        raise ValueError("Particle dataframe is empty.")
    best_row = _stack_row_for_best_particle(df)

    print("Loaded configuration (same schema as refine template).")
    print("Running peak inspection (full stack; may take a while)…")

    start_time = time.time()
    inspection_result = manager.run_peak_inspection(
        correlation_batch_size=CORRELATION_BATCH_SIZE,
        prefer_refined_angles=True,
        output_mode=OUTPUT_MODE,
    )
    end_time = time.time()
    # Wall time and the actual 6D layout (see module docstring block above).
    print(f"Finished ``run_peak_inspection`` in {end_time - start_time:.1f} s")
    if OUTPUT_MODE == "cross_correlation":
        inspection = inspection_result
        if not isinstance(inspection, torch.Tensor) or inspection.ndim != 6:
            raise ValueError(
                "Expected tensor (N, n_px, n_def, n_orient, H, W); "
                f"got {type(inspection)}"
            )
        if int(inspection.shape[0]) != n:
            raise ValueError(
                f"Inspect output length {inspection.shape[0]} != dataframe rows {n}."
            )
        print(f"Tensor shape: {tuple(inspection.shape)}")
    else:
        if (
            not isinstance(inspection_result, tuple)
            or len(inspection_result) != 2
            or not isinstance(inspection_result[0], torch.Tensor)
            or not isinstance(inspection_result[1], torch.Tensor)
        ):
            raise ValueError("Expected (frc_tensor, frequency_bins) tuple in frc mode.")
        inspection, frequency_bins = inspection_result
        if inspection.ndim != 5:
            raise ValueError(
                "Expected FRC tensor (N, n_px, n_def, n_orient, n_freq); "
                f"got {tuple(inspection.shape)}"
            )
        if int(inspection.shape[0]) != n:
            raise ValueError(
                f"Inspect output length {inspection.shape[0]} != dataframe rows {n}."
            )
        print(f"FRC tensor shape: {tuple(inspection.shape)}")
        print(f"Frequency bins shape: {tuple(frequency_bins.shape)}")

    # (phi,θ,ψ) per local orientation row; used to label angles in the summary below.
    euler_offsets = manager.orientation_refinement_config.euler_angles_offsets
    if euler_offsets.device != inspection.device:
        euler_offsets = euler_offsets.to(inspection.device)

    # best_row = argmax score in the CSV; it matches inspection[best_row, ...] order.
    if "particle_index" in df.columns:
        pidx = int(df["particle_index"].iloc[best_row])
        print(f"\nBest-particle row in CSV: {best_row} (particle_index={pidx})")
    else:
        print(f"\nBest-particle row in CSV: {best_row}")
    if OUTPUT_MODE == "cross_correlation":
        # Pool CC over (pixel, defocus, map) per orientation, then show top K.
        _print_top_orientations(inspection, euler_offsets, best_row)
    else:
        # Pool FRC over (pixel, defocus, frequency) per orientation, then show top K.
        _print_top_orientations_frc(
            inspection,
            frequency_bins,
            euler_offsets,
            best_row,
        )
    print("\nDone!")


if __name__ == "__main__":
    main()
