"""Program for running whole-orientation search using 2D template matching."""

import time

from leopard_em.pydantic_models.managers import MatchTemplateManager

#######################################
### Editable parameters for program ###
#######################################

# Edit your YAML file to configure the match template program.
# See online documentation for more information on editing this file.
YAML_CONFIG_PATH = "/path/to/match-template-configuration.yaml"

# Path where the picked peaks from the match template search will be output.
# Can be passed to the refine_template & optimize_template programs
DATAFRAME_OUTPUT_PATH = "/path/to/match-template-results.csv"

# Number of orientations to cross-correlate simultaneously. Larger values may perform
# better on GPUs with more VRAM. Tuneable parameter to ensure GPUs don't run out of
# memory during the search
ORIENTATION_BATCH_SIZE = 8

##############################################################
### Main function called to run the match template program ###
##############################################################


def main() -> None:
    """Main function for running the match template program."""
    mt_manager = MatchTemplateManager.from_yaml(YAML_CONFIG_PATH)

    print("Loaded configuration.")
    print("Running match template...")

    start_time = time.time()

    mt_manager.run_match_template(
        orientation_batch_size=ORIENTATION_BATCH_SIZE,
        do_result_export=True,  # Saves the statistics immediately upon completion
    )

    print("Finished core match_template call.")

    # Print the wall time of the search in HH:MM:SS
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(f"Match Template wall time: {elapsed_time_str}")

    # Exporting the picked peaks to a CSV file
    print("Exporting results...")

    df = mt_manager.results_to_dataframe()
    df.to_csv(DATAFRAME_OUTPUT_PATH, index=True)

    print("Done!")


# NOTE: Invoking  program under `if __name__ == "__main__"` necessary for multiprocesing
if __name__ == "__main__":
    main()
