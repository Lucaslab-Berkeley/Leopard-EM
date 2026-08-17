"""Program for 2D template matching with spatial real-space CTF pre-multiplication.

Each defocus search plane convolves the micrograph with a spatially varying PSF
grid (truncated kernel, circular convolution, not sum-normalized), then runs
2DTM with template CTF off. Whitening is taken from the raw micrograph. Image
DFTs are scaled so raw MIP is in Fourier-2DTM units.

Uniform fields (``grad_mag_angstrom: 0``) still use the PSF grid.
"""

from __future__ import annotations

import time

from leopard_em.pydantic_models.managers import SpatialCtfMatchTemplateManager

#######################################
### Editable parameters for program ###
#######################################

# Edit your YAML file to configure the spatial CTF match template program.
# See spatial_ctf_match_template_example.yaml in this directory.
YAML_CONFIG_PATH = "/path/to/spatial_ctf_match_template.yaml"

# Path where the picked peaks from the match template search will be output.
# Can be passed to the refine_template & optimize_template programs
DATAFRAME_OUTPUT_PATH = "/path/to/spatial-match-template-results.csv"

# Number of orientations to cross-correlate simultaneously. Larger values may perform
# better on GPUs with more VRAM. Tuneable parameter to ensure GPUs don't run out of
# memory during the search
ORIENTATION_BATCH_SIZE = 8

##############################################################
### Main function called to run the match template program ###
##############################################################


def main() -> None:
    """Main function for running the spatial CTF match template program."""
    mt_manager = SpatialCtfMatchTemplateManager.from_yaml(YAML_CONFIG_PATH)

    print("Loaded configuration.")
    print(f"GPUs: {[str(d) for d in mt_manager.computational_config.gpu_devices]}")
    print(f"spatial_model: {mt_manager.spatial_model}")
    print("Running spatial match template...")

    start_time = time.time()

    mt_manager.run_spatial_match_template(
        orientation_batch_size=ORIENTATION_BATCH_SIZE,
        do_result_export=True,  # Saves the statistics immediately upon completion
        # Backend already valid-crops statistic maps; do not crop again.
        do_valid_cropping=False,
        compute_correlation_table=False,
    )

    print("Finished core spatial match_template call.")

    # Print the wall time of the search in HH:MM:SS
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(f"Spatial match template wall time: {elapsed_time_str}")

    # Exporting the picked peaks to a CSV file
    print("Exporting results...")

    df = mt_manager.results_to_dataframe()
    df.to_csv(DATAFRAME_OUTPUT_PATH, index=True)

    print("Done!")


# NOTE: Invoking program under `if __name__ == "__main__"` necessary for multiprocessing
if __name__ == "__main__":
    main()
