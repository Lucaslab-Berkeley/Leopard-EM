"""Processes results files from multiple directories sequentially."""

import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from leopard_em.analysis.zscore_metric import gaussian_noise_zscore_cutoff


def get_micrograph_id(filename: str) -> str:
    """Extract micrograph ID from filename.

    Parameters
    ----------
    filename : str
        Filename to extract micrograph ID from

    Returns
    -------
    micrograph_id : str
        Micrograph ID
    """
    base_name = os.path.basename(filename)
    # Extract the part before _results.csv
    parts = base_name.split("_results.csv")[0]
    return parts


def process_directories_sequentially(
    directory_list: list[str],
    output_base_dir: str,
    false_positive_rate: float = 0.005,
) -> dict[str, pd.DataFrame]:
    """
    Process directories sequentially.

    Parameters
    ----------
    directory_list : list
        Ordered list of directories to process
    output_base_dir : str
        Base directory to store output files
    false_positive_rate : float
        False positive rate to use for threshold calculation

    Returns
    -------
    all_particles : dict
        Dictionary of micrograph IDs as keys and df as values
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)

    # Dictionary to store particles from all steps
    # Key: micrograph_id
    # Value: DataFrame of particles
    all_particles = {}

    # Dictionary to track which particles were found in which step
    # Key: (micrograph_id, particle_index)
    # Value: last step where this particle was found
    particle_step_map = {}

    # Dictionary to track total correlations per micrograph
    # Key: micrograph_id
    # Value: total correlations for this micrograph across all steps
    micrograph_correlations = defaultdict(int)

    # Dictionary to track thresholds per micrograph
    # Key: micrograph_id
    # Value: threshold for this micrograph in the current step
    micrograph_thresholds = {}

    # Process each directory in order
    for step_idx, directory in enumerate(directory_list):
        step_num = step_idx + 1
        print(f"\nProcessing Step {step_num}: {directory}")

        # Create step output directory
        step_output_dir = os.path.join(output_base_dir, f"step_{step_num}")
        os.makedirs(step_output_dir, exist_ok=True)

        # Find all results.csv files in the directory
        results_files = glob.glob(
            os.path.join(directory, "**", "*_results.csv"), recursive=True
        )

        if not results_files:
            print(f"  Warning: No results files found in {directory}")
            continue

        print(f"  Found {len(results_files)} results files")

        # Dictionary to store parameters from each micrograph
        step_micrograph_parameters = {}

        # First, find and read all parameters files to update correlation counts
        for results_file in results_files:
            micrograph_id = get_micrograph_id(results_file)
            params_file = results_file.replace(
                "_results.csv", "_results_parameters.csv"
            )

            if os.path.exists(params_file):
                try:
                    # Read the parameters file
                    params_df = pd.read_csv(params_file)
                    if not params_df.empty:
                        step_micrograph_parameters[micrograph_id] = params_df.iloc[0]

                        # Add to total correlations for this micrograph
                        if "num_correlations" in params_df.columns:
                            correlations = int(params_df.iloc[0]["num_correlations"])
                            micrograph_correlations[micrograph_id] += correlations
                            print(
                                f"  {micrograph_id}: Added {correlations} correlations "
                                f"(total: {micrograph_correlations[micrograph_id]})"
                            )
                except Exception as e:
                    print(f"  Error reading parameters file {params_file}: {e}")
            else:
                print(f"  Warning: Parameters file not found for {results_file}")

        # Calculate threshold for each micrograph based on its cumulative correlations
        for micrograph_id, total_correlations in micrograph_correlations.items():
            threshold = gaussian_noise_zscore_cutoff(
                total_correlations, false_positive_rate
            )
            micrograph_thresholds[micrograph_id] = threshold
            print(
                f"  Threshold for {micrograph_id} in step {step_num}: {threshold:.4f} "
                f"(based on {total_correlations} total correlations)"
            )

        # Process each results file
        for results_file in results_files:
            micrograph_id = get_micrograph_id(results_file)

            try:
                # Read the results file
                results_df = pd.read_csv(results_file)

                if results_df.empty:
                    print(f"  Warning: Empty results file {results_file}")
                    continue

                # Get the threshold for this micrograph
                if micrograph_id not in micrograph_thresholds:
                    print(
                        f"  Warning: No correlation information for {micrograph_id}, "
                        "using default threshold"
                    )
                    # Try to use correlations from the current step's parameters file
                    if (
                        micrograph_id in step_micrograph_parameters
                        and "num_correlations"
                        in step_micrograph_parameters[micrograph_id]
                    ):
                        correlations = int(
                            step_micrograph_parameters[micrograph_id][
                                "num_correlations"
                            ]
                        )
                        micrograph_correlations[micrograph_id] = correlations
                        threshold = gaussian_noise_zscore_cutoff(
                            correlations, false_positive_rate
                        )
                        micrograph_thresholds[micrograph_id] = threshold
                        print(
                            f"  Using threshold {threshold:.4f} for {micrograph_id} "
                            f"based on {correlations} correlations"
                        )
                    else:
                        # If no information at all, use the median of other thresholds
                        # or a reasonable default
                        if micrograph_thresholds:
                            threshold = np.median(list(micrograph_thresholds.values()))
                            print(
                                f"  Using median threshold {threshold:.4f} for "
                                f"{micrograph_id}"
                            )
                        else:
                            #  Default if no other information is available
                            threshold = 5.0
                            print(
                                f"  Using default threshold {threshold:.4f} for "
                                f"{micrograph_id}"
                            )
                        micrograph_thresholds[micrograph_id] = threshold
                else:
                    threshold = micrograph_thresholds[micrograph_id]

                # Check if refined_scaled_mip column exists
                if "refined_scaled_mip" not in results_df.columns:
                    print(
                        f" Warning: refined_scaled_mip not found in {results_file},"
                        " using mip instead"
                    )
                    compare_col = "scaled_mip"
                else:
                    compare_col = "refined_scaled_mip"

                # Filter particles above threshold using the appropriate column
                above_threshold_df = results_df[
                    results_df[compare_col] > threshold
                ].copy()

                if above_threshold_df.empty:
                    print(f"  No particles above threshold in {results_file}")
                    continue

                # Print stats
                print(
                    f"{micrograph_id}: {len(above_threshold_df)} of {len(results_df)}"
                    f" particles above threshold (using {compare_col})"
                )

                # Add a step column to track which step this is from
                above_threshold_df["step"] = step_num

                # If this is the first step, just add all particles above threshold
                if step_num == 1:
                    all_particles[micrograph_id] = above_threshold_df

                    # Update particle step map
                    for idx in above_threshold_df["particle_index"]:
                        particle_step_map[(micrograph_id, idx)] = step_num
                else:
                    # If this micrograph was not seen before, add all particles
                    if micrograph_id not in all_particles:
                        all_particles[micrograph_id] = above_threshold_df

                        # Update particle step map
                        for idx in above_threshold_df["particle_index"]:
                            particle_step_map[(micrograph_id, idx)] = step_num
                    else:
                        # For existing micrographs, handle particles differently
                        existing_df = all_particles[micrograph_id]

                        # Create a new DataFrame to store updated particles
                        updated_df = existing_df.copy()

                        # For each particle in the new results
                        for _, particle in above_threshold_df.iterrows():
                            particle_idx = particle["particle_index"]

                            # Check if this particle exists previously
                            existing_particle = existing_df[
                                existing_df["particle_index"] == particle_idx
                            ]

                            if len(existing_particle) > 0:
                                # Particle exists, update parameters
                                # Find the index in the updated_df
                                idx_to_update = updated_df.index[
                                    updated_df["particle_index"] == particle_idx
                                ].tolist()[0]

                                # Check if original offset columns exist
                                offset_cols = [
                                    "original_offset_phi",
                                    "original_offset_theta",
                                    "original_offset_psi",
                                ]

                                # Add original offset columns from step 1
                                for col in offset_cols:
                                    if col not in updated_df.columns:
                                        updated_df[col] = 0.0

                                # Add offset values from current step
                                for col in offset_cols:
                                    # Add particle's offset to existing offset
                                    if col in particle and pd.notna(particle[col]):
                                        updated_df.at[idx_to_update, col] += particle[
                                            col
                                        ]

                                # Update other parameters
                                for col in particle.index:
                                    if col not in offset_cols and pd.notna(
                                        particle[col]
                                    ):
                                        updated_df.at[idx_to_update, col] = particle[
                                            col
                                        ]

                                # Update step
                                updated_df.at[idx_to_update, "step"] = step_num
                                particle_step_map[(micrograph_id, particle_idx)] = (
                                    step_num
                                )
                            else:
                                # New particle, add it to the DataFrame
                                updated_df = pd.concat(
                                    [updated_df, pd.DataFrame([particle])],
                                    ignore_index=True,
                                )
                                particle_step_map[(micrograph_id, particle_idx)] = (
                                    step_num
                                )

                        # Update the all_particles dictionary
                        all_particles[micrograph_id] = updated_df

            except Exception as e:
                print(f"  Error processing results file {results_file}: {e}")

        # Save intermediate results for this step
        for micrograph_id, particles_df in all_particles.items():
            # Only save particles found or updated in this step
            step_particles = particles_df[particles_df["step"] == step_num]

            if not step_particles.empty:
                output_file = os.path.join(
                    step_output_dir, f"{micrograph_id}_results_above_threshold.csv"
                )
                step_particles.to_csv(output_file, index=False)
                print(
                    f"  Saved {len(step_particles)} particles for {micrograph_id} "
                    f"in step {step_num}"
                )

    # Save final results after all steps
    final_output_dir = os.path.join(output_base_dir, "final_results")
    os.makedirs(final_output_dir, exist_ok=True)

    # Save summary of total particles per micrograph
    summary_data = []

    for micrograph_id, particles_df in all_particles.items():
        output_file = os.path.join(
            final_output_dir, f"{micrograph_id}_results_above_threshold.csv"
        )
        particles_df.to_csv(output_file, index=False)

        # Get the final threshold for this micrograph
        final_threshold = micrograph_thresholds.get(micrograph_id, "N/A")
        total_correlations = micrograph_correlations.get(micrograph_id, 0)

        # Create summary data
        n_particles = len(particles_df)
        summary_data.append(
            {
                "micrograph_id": micrograph_id,
                "total_particles": n_particles,
                "total_correlations": total_correlations,
                "final_threshold": final_threshold,
            }
        )

        print(
            f"Saved {n_particles} final particles for {micrograph_id} "
            f"(threshold: {final_threshold}, correlations: {total_correlations})"
        )

    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(
        os.path.join(final_output_dir, "processing_summary.csv"), index=False
    )

    print(f"\nProcessing complete. Final results saved to {final_output_dir}")

    # Print total particles
    total_particles = sum(len(df) for df in all_particles.values())
    print(f"Total particles across all micrographs: {total_particles}")

    return all_particles


def main() -> None:
    """Main function to process results files sequentially."""
    parser = argparse.ArgumentParser(
        description="Process results files from multiple directories sequentially"
    )
    parser.add_argument(
        "directories", nargs="+", help="Ordered list of directories to process"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--false-positive-rate",
        "-f",
        type=float,
        default=0.005,
        help="False positive rate for threshold calculation (default: 0.005)",
    )

    args = parser.parse_args()

    # Check if all directories exist
    for directory in args.directories:
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} does not exist!")
            return

    # Process directories
    process_directories_sequentially(
        args.directories, args.output, args.false_positive_rate
    )


if __name__ == "__main__":
    main()
