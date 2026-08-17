"""Match template with a filament-restricted (Euler-box) orientation search.

This is the standard match-template program. If CONSTRAINT_YAML_PATH points at
a sidecar written by ``napari_choose_constraint.py``, that file replaces
``orientation_search_config`` in the match-template YAML.
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

##############################################################
### Main function called to run the match template program ###
##############################################################


def main() -> None:
    """Main function for running constrained match template."""
    mt_manager = MatchTemplateManager.from_yaml(YAML_CONFIG_PATH)

    if CONSTRAINT_YAML_PATH:
        constraint = FilamentConstraint.from_yaml(CONSTRAINT_YAML_PATH)
        mt_manager.orientation_search_config = constraint.to_orientation_config()
        n_orient = int(mt_manager.orientation_search_config.euler_angles.shape[0])
        print(constraint.preview_text())
        print(f"Using filament constraint with {n_orient} orientations.")
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
