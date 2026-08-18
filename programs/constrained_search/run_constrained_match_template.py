"""Match template with a sidecar that subsets the YAML orientation search.

This is the standard match-template program. YAML_CONFIG_PATH defines the
FFT grid (and the default mean/variance space). If CONSTRAINT_YAML_PATH points at a
sidecar from ``napari_choose_constraint.py``, that file does **not** replace
``orientation_search_config``. It only subsets which (pixel, orientation,
defocus) tuples may win the MIP, and optionally mean/variance when
``STATS_FROM_VALID_ORIENTATIONS_DEFOCUS`` is True.
"""

import time

from leopard_em.pydantic_models.config import FilamentConstraint
from leopard_em.pydantic_models.managers import MatchTemplateManager

#######################################
### Editable parameters for program ###
#######################################

# Standard match-template YAML (optics, defocus, filters, output paths, ...).
YAML_CONFIG_PATH = "/path/to/match-template-configuration.yaml"

# Sidecar from napari_choose_constraint.py. Leave empty to run the YAML
# search with no per-pixel eligibility subset.
CONSTRAINT_YAML_PATH = "/path/to/filament_constraint.yaml"

# Path where the picked peaks from the match template search will be output.
DATAFRAME_OUTPUT_PATH = "/path/to/constrained-match-template-results.csv"

# Number of orientations to cross-correlate simultaneously.
ORIENTATION_BATCH_SIZE = 8

# How mean and variance (the z-score background) are accumulated.
#
# The MIP is always the best *allowed* (pixel, orientation, defocus) tuple:
# YAML angles outside the sidecar Euler box, pixels outside the spatial box,
# and defocus values outside the per-pixel HDF5 range cannot win.
#
# Mean / variance use the YAML search grid unless this flag is True:
#
#   False (default): every orientation and defocus in YAML_CONFIG_PATH, at
#       every pixel. This matches a normal match-template run.
#
#   True: only allowed (pixel, orientation, defocus) tuples, with a
#       per-pixel count divisor.
#
# Peak cutoff (num_ccg) always uses the sum of allowed tests:
# sum over pixels of n_orientations[y, x] * n_defocus[y, x].
STATS_FROM_VALID_ORIENTATIONS_DEFOCUS = False

##############################################################
### Main function called to run the match template program ###
##############################################################


def main() -> None:
    """Main function for running constrained match template."""
    mt_manager = MatchTemplateManager.from_yaml(YAML_CONFIG_PATH)

    if CONSTRAINT_YAML_PATH:
        constraint = FilamentConstraint.from_yaml(CONSTRAINT_YAML_PATH)
        constraint.stats_from_valid_orientations_defocus = (
            STATS_FROM_VALID_ORIENTATIONS_DEFOCUS
        )
        mt_manager.apply_filament_constraint(constraint)
        n_orient = int(mt_manager.orientation_search_config.euler_angles.shape[0])
        print(constraint.preview_text())
        print(
            f"YAML search: {n_orient} orientations "
            "(sidecar subsets MIP eligibility)."
        )
        if constraint.spatial_constraint_path:
            print(f"Spatial constraint: {constraint.spatial_constraint_path}")
        if STATS_FROM_VALID_ORIENTATIONS_DEFOCUS:
            print(
                "Mean/variance: allowed (pixel, orientation, defocus) tuples "
                "only (per-pixel count)."
            )
        else:
            print(
                "Mean/variance: all orientations and defocus searched in this "
                "run (MIP still restricted to allowed tuples)."
            )
    else:
        n_orient = int(mt_manager.orientation_search_config.euler_angles.shape[0])
        print(f"Using orientation_search_config from YAML ({n_orient} orientations).")

    print("Loaded configuration.")
    print("Running match template...")

    start_time = time.time()

    mt_manager.run_match_template(
        orientation_batch_size=ORIENTATION_BATCH_SIZE,
        do_result_export=True,
    )

    print("Finished core match_template call.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(f"Match Template wall time: {elapsed_time_str}")

    print("Exporting results...")
    df = mt_manager.results_to_dataframe()
    df.to_csv(DATAFRAME_OUTPUT_PATH, index=True)

    print("Done!")


if __name__ == "__main__":
    main()
