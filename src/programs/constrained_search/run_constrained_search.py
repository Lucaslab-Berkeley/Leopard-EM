"""Constrained search around sets of match template results."""

import time

from programs.constrained_search import ConstrainedSearchManager

#######################################
### Editable parameters for program ###
#######################################

# Edit your YAML file to configure the refine template program.
# Needs to reference the outputs from a match template run.
# See online documentation for more information on editing this file.
YAML_CONFIG_PATH = "/path/to/constrained-search-configuration.yaml"

# Path to where the dataframe with refined peak parameters will be output.
DATAFRAME_OUTPUT_PATH = "/path/to/constrained-search-results.csv"

# Number of particles to refine simultaneously. Will need to tune this parameter
# based on the memory & computational resources available.
PARTICLE_BATCH_SIZE = 80

###############################################################
### Main function called to run the refine template program ###
###############################################################


def main() -> None:
    """Main function for running the refine template program."""
    cs_manager = ConstrainedSearchManager.from_yaml(YAML_CONFIG_PATH)

    print("Loaded configuration.")
    print("Running refine template...")

    start_time = time.time()

    cs_manager.run_constrained_search(DATAFRAME_OUTPUT_PATH, PARTICLE_BATCH_SIZE)

    print("Finished core call.")

    # Print the wall time of the search in HH:MM:SS
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(f"Wall time: {elapsed_time_str}")

    print("Done!")


# NOTE: Invoking  program under `if __name__ == "__main__"` necessary for multiprocesing
if __name__ == "__main__":
    main()
