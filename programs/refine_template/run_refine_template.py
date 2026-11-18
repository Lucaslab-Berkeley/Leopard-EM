"""Local orientation refinement around sets of match template results."""

import time

from leopard_em.pydantic_models.managers import RefineTemplateManager

#######################################
### Editable parameters for program ###
#######################################

# Edit your YAML file to configure the refine template program.
# Needs to reference the outputs from a match template run.
# See online documentation for more information on editing this file.
YAML_CONFIG_PATH = "/path/to/refine-template-configuration.yaml"

# Path to where the dataframe with refined peak parameters will be output.
# May be a .csv or .h5 path -- output format matches the particle_stack back-end
# configured in the YAML file by default (df_path -> CSV, hdf5_path -> HDF5),
# or pass output_format="csv"/"hdf5" to RefineTemplateManager.run_refine_template()
# to override.
DATAFRAME_OUTPUT_PATH = "/path/to/refine-template-results.csv"

# Number of orientations simulated in refine template as once. Larger values require
# more GPU memory, but can sometimes be faster.
ORIENTATION_BATCH_SIZE = 80

# Number of particles each GPU moves to the device and refines at once. None moves all
# of a GPU's particles at once. Use to tune performance for large particle stacks.
PARTICLE_BATCH_SIZE = None

###############################################################
### Main function called to run the refine template program ###
###############################################################


def main() -> None:
    """Main function for running the refine template program."""
    rt_manager = RefineTemplateManager.from_yaml(YAML_CONFIG_PATH)

    print("Loaded configuration.")
    print("Running refine template...")

    start_time = time.time()

    rt_manager.run_refine_template(
        output_dataframe_path=DATAFRAME_OUTPUT_PATH,
        correlation_batch_size=ORIENTATION_BATCH_SIZE,
        particle_batch_size=PARTICLE_BATCH_SIZE,
    )

    print("Finished core refine_template call.")

    # Print the wall time of the search in HH:MM:SS
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(f"Refine template wall time: {elapsed_time_str}")

    print("Done!")


# NOTE: Invoking  program under `if __name__ == "__main__"` necessary for multiprocesing
if __name__ == "__main__":
    main()
