"""Program for extracting peaks from match template results using z-score metric."""

import argparse

from leopard_em.analysis import (
    extract_peaks_and_statistics_zscore,
    match_template_peaks_to_dataframe,
)
from leopard_em.pydantic_models.managers import MatchTemplateManager
from leopard_em.utils.data_io import load_mrc_image

#######################################
### Editable parameters for program ###
#######################################

# Edit your YAML file to configure the match template program.
# See online documentation for more information on editing this file.
YAML_CONFIG_PATH = "leopard-em_intro/match_template_config_60S_intro_cropped.yaml"

# Path where the picked peaks from the z-score extraction will be output.
OUTPUT_CSV_PATH = "output_zscore_peaks.csv"

# Optional hardcoded paths for MIP and scaled MIP files.
# If not None, these will override the paths from the YAML config.
MIP_PATH = None  # e.g., "path/to/mip.mrc"
SCALED_MIP_PATH = None  # e.g., "path/to/scaled_mip.mrc"

##############################################################
### Main function called to run the z-score peak extraction ###
##############################################################


def main() -> None:
    """Main function for running the z-score peak extraction program."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Extract peaks from match template results using z-score metric."
    )
    parser.add_argument(
        "--false-positives",
        "-f",
        type=float,
        default=1.0,
        help="Number of false positives to allow (default: 1.0)",
    )
    parser.add_argument(
        "--z-score-cutoff",
        "-z",
        type=float,
        default=None,
        help=(
            "Z-score cutoff value. If not provided, calculated from "
            "false_positives (default: None)"
        ),
    )
    parser.add_argument(
        "--mask-radius",
        "-r",
        type=float,
        default=5.0,
        help="Minimum distance between peaks in pixels (default: 5.0)",
    )
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {YAML_CONFIG_PATH}...")
    mt_manager = MatchTemplateManager.from_yaml(YAML_CONFIG_PATH)
    print("Loaded configuration.")

    # Get MRC file paths from match_template_result
    result_config = mt_manager.match_template_result
    print("Loading MRC result files...")

    # Load all required MRC files
    # Use hardcoded paths if provided, otherwise use paths from config
    mip_path = (
        MIP_PATH if MIP_PATH is not None else result_config.mip_path
    )
    scaled_mip_path = (
        SCALED_MIP_PATH
        if SCALED_MIP_PATH is not None
        else result_config.scaled_mip_path
    )

    mip = load_mrc_image(mip_path)
    scaled_mip = load_mrc_image(scaled_mip_path)
    best_psi = load_mrc_image(result_config.orientation_psi_path)
    best_theta = load_mrc_image(result_config.orientation_theta_path)
    best_phi = load_mrc_image(result_config.orientation_phi_path)
    best_defocus = load_mrc_image(result_config.relative_defocus_path)
    correlation_average = load_mrc_image(result_config.correlation_average_path)
    correlation_variance = load_mrc_image(result_config.correlation_variance_path)

    print("Loaded all MRC result files.")

    # Calculate total correlation positions
    # Formula: num_orientations * num_defocus * num_pixel_sizes
    # Pixel sizes are typically 1 (no search by default)
    num_orientations = len(mt_manager.orientation_search_config.euler_angles)
    num_defocus = len(mt_manager.defocus_search_config.defocus_values)
    num_pixel_sizes = 1  # Default: no pixel size search
    total_correlation_positions = num_orientations * num_defocus * num_pixel_sizes

    print(f"Number of orientations: {num_orientations}")
    print(f"Number of defocus values: {num_defocus}")
    print(f"Total correlation positions: {total_correlation_positions}")

    # Extract peaks using z-score metric
    print("Extracting peaks using z-score metric...")
    peaks = extract_peaks_and_statistics_zscore(
        mip=mip,
        scaled_mip=scaled_mip,
        best_psi=best_psi,
        best_theta=best_theta,
        best_phi=best_phi,
        best_defocus=best_defocus,
        correlation_average=correlation_average,
        correlation_variance=correlation_variance,
        total_correlation_positions=total_correlation_positions,
        false_positives=args.false_positives,
        z_score_cutoff=args.z_score_cutoff,
        mask_radius=args.mask_radius,
    )

    print(f"Found {len(peaks.pos_y)} peaks.")

    # Convert to DataFrame and save
    print(f"Exporting results to {OUTPUT_CSV_PATH}...")
    df = match_template_peaks_to_dataframe(peaks)
    df.to_csv(OUTPUT_CSV_PATH, index=True)

    print("Done!")


if __name__ == "__main__":
    main()

