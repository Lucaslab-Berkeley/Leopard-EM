"""Program for setting a match_template threshold from null searches.

Runs the null searches configured in the YAML file and writes
``null_summary.json`` to its ``output_directory``. That file holds, for each target
number of false positives per micrograph, the nominal (independent-trials) threshold,
the threshold from phase-randomised null searches and the analytic prediction.
"""

import time

from leopard_em.pydantic_models.managers import NullCalibrationManager

#######################################
### Editable parameters for program ###
#######################################

# Edit your YAML file to configure the null calibration program.
# See online documentation for more information on editing this file.
YAML_CONFIG_PATH = "/path/to/null-calibration-configuration.yaml"

# Number of orientations to cross-correlate simultaneously in each null search.
ORIENTATION_BATCH_SIZE = 8

################################################################
### Main function called to run the null calibration program ###
################################################################


def main() -> None:
    """Main function for running the null calibration program."""
    manager = NullCalibrationManager.from_yaml(YAML_CONFIG_PATH)

    print(f"Loaded configuration; {len(manager.seeds)} phase-randomised copies.")
    start_time = time.time()

    summary = manager.run(orientation_batch_size=ORIENTATION_BATCH_SIZE)

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"Null calibration wall time: {elapsed}")
    for row in summary["thresholds"]:
        parts = [
            f"{key} {value:.3f}" for key, value in row.items() if value is not None
        ]
        print(" | ".join(parts))
    print(f"Summary written to {manager.output_directory}/null_summary.json")


# NOTE: Invoking  program under `if __name__ == "__main__"` necessary for multiprocesing
if __name__ == "__main__":
    main()
