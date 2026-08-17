"""Match template with a filament-restricted (Euler-box) orientation search.

This is the standard match-template program. If CONSTRAINT_YAML_PATH points at
a sidecar written by ``napari_choose_constraint.py``, that file replaces
``orientation_search_config`` in the match-template YAML and, if present,
loads the per-pixel spatial constraint HDF5 for mask-before-max MIP updates
and the z-score ``num_ccg`` count.
"""

import time

from leopard_em.pydantic_models.config import FilamentConstraint
from leopard_em.pydantic_models.managers import MatchTemplateManager

#######################################
### Editable parameters for program ###
#######################################

# Standard match-template YAML (optics, defocus, filters, output paths, ...).
YAML_CONFIG_PATH = "/path/to/match-template-configuration.yaml"

# Sidecar from napari_choose_constraint.py. Leave empty to use the orientation
# search already specified in YAML_CONFIG_PATH.
CONSTRAINT_YAML_PATH = "/path/to/filament_constraint.yaml"

# Path where the picked peaks from the match template search will be output.
DATAFRAME_OUTPUT_PATH = "/path/to/constrained-match-template-results.csv"

# Number of orientations to cross-correlate simultaneously.
ORIENTATION_BATCH_SIZE = 8

# How mean and variance (the z-score background) are accumulated.
#
# The MIP is always the best *allowed* (pixel, orientation) pair: angles
# outside the Euler box, and pixels outside the spatial box, cannot win.
#
# Mean / variance are separate. They answer "what does a typical correlation
# look like at this pixel?" and are the divisor for scaled_mip =
# (mip - mean) / std.
#
#   False (default): use every orientation this run actually searched.
#       Today that is the Euler box (not full SO(3)). This matches a normal
#       match-template run, where mean/std come from the whole search grid.
#       Pixels outside the spatial box still contribute to the running sums,
#       but they are not used as peaks.
#
#   True: use only allowed (pixel, orientation) pairs, and divide by a
#       per-pixel count. Outside the spatial box, count is 0 so mean/std
#       are 0. This only differs from False once the search grid is larger
#       than the allowed set (e.g. a future full-SO(3) search with a
#       per-pixel Euler box). With the current Euler-box search, True and
#       False agree inside the spatial box.
#
# Peak cutoff (num_ccg) always uses the sum of allowed tests, independent of
# this flag: n_box_pixels * n_orientations * n_defocus.
STATS_FROM_VALID_ORIENTATIONS = False

##############################################################
### Main function called to run the match template program ###
##############################################################


def main() -> None:
    """Main function for running constrained match template."""
    mt_manager = MatchTemplateManager.from_yaml(YAML_CONFIG_PATH)

    if CONSTRAINT_YAML_PATH:
        constraint = FilamentConstraint.from_yaml(CONSTRAINT_YAML_PATH)
        constraint.stats_from_valid_orientations = STATS_FROM_VALID_ORIENTATIONS
        mt_manager.apply_filament_constraint(constraint)
        n_orient = int(mt_manager.orientation_search_config.euler_angles.shape[0])
        print(constraint.preview_text())
        print(f"Using filament constraint with {n_orient} orientations.")
        if constraint.spatial_constraint_path:
            print(f"Spatial constraint: {constraint.spatial_constraint_path}")
        if STATS_FROM_VALID_ORIENTATIONS:
            print(
                "Mean/variance: allowed (pixel, orientation) pairs only "
                "(per-pixel count)."
            )
        else:
            print(
                "Mean/variance: all orientations searched in this run "
                "(MIP still restricted to the spatial/Euler box)."
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
