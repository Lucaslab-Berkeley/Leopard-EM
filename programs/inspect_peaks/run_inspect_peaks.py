"""Run peak inspection and save the local score tensor to a self-describing file.

The output ``.npz`` is self-describing. Reload it with::

    from leopard_em.analysis import load_inspection_result

    result = load_inspection_result("results_inspect_peaks.npz")
    result.scores  # main tensor; see result.axes for layout
    result.euler_angle_offsets  # (n_orient, 3) ZYZ offsets per orientation index
    result.defocus_offsets  # (n_defocus,) relative defocus offsets (Angstrom)
    result.pixel_size_offsets  # (n_px,) relative pixel-size offsets
    result.particle_index  # (N,) maps tensor rows to the particle stack
    result.frequency_bins  # (n_freq,) FRC frequencies (FRC mode only)

Stored tensor layout (``result.scores``):
- ``"cross_correlation"``: ``(N, n_px, n_defocus, n_orient, H, W)``
- ``"frc"``:               ``(N, n_px, n_defocus, n_orient, n_freq)``
"""

# NOTE: The ``if __name__ == "__main__"`` guard is required for multiprocessing.
import time
from typing import Literal

from leopard_em.pydantic_models.managers import PeakInspectionManager

#######################################
### Editable parameters for program ###
#######################################

# Edit the YAML the same way you would for refine template (see example config next
# to this script, identical schema to `refine_template_example_config.yaml`).
YAML_CONFIG_PATH = "/path/to/inspect_peaks_configuration.yaml"

# Where to write the score tensor. A ``.npz`` suffix is appended if missing.
OUTPUT_PATH = "/path/to/results_inspect_peaks.npz"

# Batched orientations per GPU call -- lower if you run out of memory.
CORRELATION_BATCH_SIZE = 32

# Output mode for inspect backend.
# - "cross_correlation": saves (N, n_px, n_defocus, n_orient, H, W)
# - "frc":               saves (N, n_px, n_defocus, n_orient, n_freq) + frequency bins
# NOTE: the cross-correlation tensor can be very large; subset the particle stack
# in the config for big runs.
# TODO: Automatically chunk particle stack within the InspectPeaksManager through an
#       option for managing memory.
OUTPUT_MODE: Literal["cross_correlation", "frc"] = "cross_correlation"


def main() -> None:
    """Run peak inspection over the full stack and save the score tensor."""
    manager = PeakInspectionManager.from_yaml(YAML_CONFIG_PATH)

    print("Loaded configuration (same schema as refine template).")
    print(f"Running peak inspection in {OUTPUT_MODE!r} mode (may take a while)...")

    start_time = time.time()
    output_path = manager.run_and_save_peak_inspection(
        output_path=OUTPUT_PATH,
        correlation_batch_size=CORRELATION_BATCH_SIZE,
        prefer_refined_angles=True,
        output_mode=OUTPUT_MODE,
    )
    elapsed = time.time() - start_time

    print(f"Finished peak inspection in {elapsed:.1f} s")
    print(f"Saved score tensor to: {output_path}")
    print(
        "Load it with `leopard_em.analysis.load_inspection_result` "
        "for inspection/plotting (see docs/examples)."
    )


if __name__ == "__main__":
    main()
