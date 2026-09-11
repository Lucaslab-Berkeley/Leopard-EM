"""Run per-frame peak inspection and save the local score tensor per movie frame.

This mirrors ``run_inspect_peaks.py`` but scores each movie frame independently, so the
stored tensor carries a ``frame`` axis after ``particle``. The output ``.npz`` is
self-describing. Reload it with::

    from leopard_em.analysis import load_inspection_result

    result = load_inspection_result("results_frame_inspection.npz")
    result.scores  # main tensor; see result.axes for layout
    result.frame_index  # (T,) movie frame index for the ``frame`` axis
    result.euler_angle_offsets  # (n_orient, 3) ZYZ offsets per orientation index
    result.defocus_offsets  # (n_defocus,) relative defocus offsets (Angstrom)
    result.pixel_size_offsets  # (n_px,) relative pixel-size offsets
    result.base_euler_angles  # (N, 3) per-particle base ZYZ angles
    result.base_defocus  # (N, 3) per-particle base (defocus_u, defocus_v, angle)
    result.particle_index  # (N,) maps tensor rows to the particle stack
    result.frequency_bins  # (n_freq,) FRC frequencies (FRC mode only)

Stored tensor layout (``result.scores``):
- ``"cross_correlation"``: ``(N, T, n_px, n_defocus, n_orient, H, W)``
- ``"frc"``:               ``(N, T, n_px, n_defocus, n_orient, n_freq)``
"""

import time
from typing import Literal

from leopard_em.pydantic_models.managers import FrameInspectionManager

#######################################
### Editable parameters for program ###
#######################################

# Edit the YAML similarly to refine/inspect configs.
YAML_CONFIG_PATH = "/path/to/frame_inspection_configuration.yaml"

# Where to write the per-frame score tensor. A ``.npz`` suffix is appended if missing.
OUTPUT_PATH = "/path/to/results_frame_inspection.npz"

# Batched orientations per GPU call - lower if you run out of memory.
CORRELATION_BATCH_SIZE = 32

# Output mode for inspect backend.
# - "cross_correlation": saves (N, T, n_px, n_defocus, n_orient, H, W)
# - "frc":               saves (N, T, n_px, n_defocus, n_orient, n_freq) + freq bins
# NOTE: the per-frame cross-correlation tensor can be very large (an extra frame axis
# on top of the spatial inspection tensor). Subset the particle stack in the config for
# big runs.
OUTPUT_MODE: Literal["cross_correlation", "frc"] = "cross_correlation"

# Set True to apply cumulative template dose filtering per frame interval.
APPLY_TEMPLATE_DOSE_WEIGHTING = False


def main() -> None:
    """Run per-frame inspect workflow and save the score tensor."""
    manager = FrameInspectionManager.from_yaml(YAML_CONFIG_PATH)

    print("Loaded configuration (same schema as refine template).")
    print(
        f"Running per-frame peak inspection in {OUTPUT_MODE!r} mode "
        "(may take a while)..."
    )

    start_time = time.time()
    output_path = manager.run_and_save_peak_inspection_per_frame(
        output_path=OUTPUT_PATH,
        correlation_batch_size=CORRELATION_BATCH_SIZE,
        prefer_refined_angles=True,
        apply_projection_normalization=True,
        output_mode=OUTPUT_MODE,
        apply_template_dose_weighting=APPLY_TEMPLATE_DOSE_WEIGHTING,
    )
    elapsed = time.time() - start_time

    print(f"Finished per-frame peak inspection in {elapsed:.1f} s")
    print(f"Saved per-frame score tensor to: {output_path}")


if __name__ == "__main__":
    main()
