"""Processes results files from multiple directories sequentially."""

import argparse
import glob
import os
from collections import defaultdict
from typing import dict, tuple

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


def load_parameters_file(params_file: str) -> pd.DataFrame:
    """Load and validate parameters file.

    Parameters
    ----------
    params_file : str
        Path to parameters file

    Returns
    -------
    pd.DataFrame
        Parameters data if valid, empty DataFrame otherwise
    """
    try:
        params_df = pd.read_csv(params_file)
        return params_df if not params_df.empty else pd.DataFrame()
    except Exception as e:
        print(f"  Error reading parameters file {params_file}: {e}")
        return pd.DataFrame()


def calculate_micrograph_thresholds(
    micrograph_correlations: dict[str, int],
    false_positive_rate: float,
) -> dict[str, float]:
    """Calculate thresholds for each micrograph based on correlation counts.

    Parameters
    ----------
    micrograph_correlations : dict[str, int]
        Dictionary mapping micrograph IDs to their total correlations
    false_positive_rate : float
        False positive rate for threshold calculation

    Returns
    -------
    dict[str, float]
        Dictionary mapping micrograph IDs to their thresholds
    """
    thresholds = {}
    for micrograph_id, total_correlations in micrograph_correlations.items():
        threshold = gaussian_noise_zscore_cutoff(
            total_correlations, false_positive_rate
        )
        thresholds[micrograph_id] = threshold
    return thresholds


def process_results_file(
    results_file: str,
    micrograph_thresholds: dict[str, float],
    step_num: int,
) -> tuple[pd.DataFrame, str]:
    """Process a single results file and filter particles above threshold.

    Parameters
    ----------
    results_file : str
        Path to results file
    micrograph_thresholds : dict[str, float]
        Dictionary of thresholds per micrograph
    step_num : int
        Current step number

    Returns
    -------
    tuple[pd.DataFrame, str]
        DataFrame of particles above threshold and micrograph ID

    Raises
    ------
    ValueError
        If no threshold can be calculated for the micrograph
    """
    micrograph_id = get_micrograph_id(results_file)

    try:
        results_df = pd.read_csv(results_file)
        if results_df.empty:
            print(f"  Warning: Empty results file {results_file}")
            return pd.DataFrame(), micrograph_id

        # Get threshold for this micrograph
        if micrograph_id not in micrograph_thresholds:
            raise ValueError(
                f"No correlation data found for micrograph {micrograph_id}. "
                "Cannot calculate threshold. Please ensure the parameters file exists "
                "and contains correlation data."
            )

        threshold = micrograph_thresholds[micrograph_id]

        # Determine which column to use for comparison
        compare_col = (
            "refined_scaled_mip"
            if "refined_scaled_mip" in results_df.columns
            else "scaled_mip"
        )
        if compare_col != "refined_scaled_mip":
            print(
                f"  Warning: refined_scaled_mip not found in {results_file}, "
                f"using mip instead"
            )

        # Filter particles above threshold
        above_threshold_df = results_df[results_df[compare_col] > threshold].copy()
        if not above_threshold_df.empty:
            above_threshold_df["step"] = step_num
            print(
                f"{micrograph_id}: {len(above_threshold_df)} of {len(results_df)} "
                f"particles above threshold (using {compare_col})"
            )

        return above_threshold_df, micrograph_id

    except Exception as e:
        print(f"  Error processing results file {results_file}: {e}")
        return pd.DataFrame(), micrograph_id


def update_particle_data(
    all_particles: dict[str, pd.DataFrame],
    particle_step_map: dict[tuple[str, int], int],
    new_particles: pd.DataFrame,
    micrograph_id: str,
    step_num: int,
) -> None:
    """Update particle data with new results.

    Parameters
    ----------
    all_particles : dict[str, pd.DataFrame]
        Dictionary of all particles per micrograph
    particle_step_map : dict[tuple[str, int], int]
        Map of particles to their last seen step
    new_particles : pd.DataFrame
        New particles to add/update
    micrograph_id : str
        ID of the micrograph
    step_num : int
        Current step number
    """
    if new_particles.empty:
        return

    if micrograph_id not in all_particles:
        all_particles[micrograph_id] = new_particles
        for idx in new_particles["particle_index"]:
            particle_step_map[(micrograph_id, idx)] = step_num
        return

    existing_df = all_particles[micrograph_id]
    updated_df = existing_df.copy()

    for _, particle in new_particles.iterrows():
        particle_idx = particle["particle_index"]
        existing_particle = existing_df[existing_df["particle_index"] == particle_idx]

        if len(existing_particle) > 0:
            # Update existing particle
            idx_to_update = updated_df.index[
                updated_df["particle_index"] == particle_idx
            ].tolist()[0]

            # Handle offset columns
            offset_cols = [
                "original_offset_phi",
                "original_offset_theta",
                "original_offset_psi",
            ]
            for col in offset_cols:
                if col not in updated_df.columns:
                    updated_df[col] = 0.0
                if col in particle and pd.notna(particle[col]):
                    updated_df.at[idx_to_update, col] += particle[col]

            # Update other parameters
            for col in particle.index:
                if col not in offset_cols and pd.notna(particle[col]):
                    updated_df.at[idx_to_update, col] = particle[col]

            updated_df.at[idx_to_update, "step"] = step_num
        else:
            # Add new particle
            updated_df = pd.concat(
                [updated_df, pd.DataFrame([particle])], ignore_index=True
            )

        particle_step_map[(micrograph_id, particle_idx)] = step_num

    all_particles[micrograph_id] = updated_df


def save_step_results(
    all_particles: dict[str, pd.DataFrame],
    step_num: int,
    step_output_dir: str,
) -> None:
    """Save results for the current step.

    Parameters
    ----------
    all_particles : dict[str, pd.DataFrame]
        Dictionary of all particles per micrograph
    step_num : int
        Current step number
    step_output_dir : str
        Directory to save results
    """
    for micrograph_id, particles_df in all_particles.items():
        step_particles = particles_df[particles_df["step"] == step_num]
        if not step_particles.empty:
            output_file = os.path.join(
                step_output_dir, f"{micrograph_id}_results_above_threshold.csv"
            )
            step_particles.to_csv(output_file, index=False)
            print(
                f"  Saved {len(step_particles)} particles for {micrograph_id}"
                f" in step {step_num}"
            )


def save_final_results(
    all_particles: dict[str, pd.DataFrame],
    micrograph_thresholds: dict[str, float],
    micrograph_correlations: dict[str, int],
    output_base_dir: str,
) -> None:
    """Save final results and summary.

    Parameters
    ----------
    all_particles : dict[str, pd.DataFrame]
        Dictionary of all particles per micrograph
    micrograph_thresholds : dict[str, float]
        Dictionary of thresholds per micrograph
    micrograph_correlations : dict[str, int]
        Dictionary of correlations per micrograph
    output_base_dir : str
        Base directory for output
    """
    final_output_dir = os.path.join(output_base_dir, "final_results")
    os.makedirs(final_output_dir, exist_ok=True)

    summary_data = []
    for micrograph_id, particles_df in all_particles.items():
        output_file = os.path.join(
            final_output_dir, f"{micrograph_id}_results_above_threshold.csv"
        )
        particles_df.to_csv(output_file, index=False)

        final_threshold = micrograph_thresholds.get(micrograph_id, "N/A")
        total_correlations = micrograph_correlations.get(micrograph_id, 0)
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

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(
        os.path.join(final_output_dir, "processing_summary.csv"), index=False
    )
    print(f"\nProcessing complete. Final results saved to {final_output_dir}")


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
    os.makedirs(output_base_dir, exist_ok=True)

    all_particles = {}
    particle_step_map = {}
    micrograph_correlations = defaultdict(int)
    micrograph_thresholds = {}

    for step_idx, directory in enumerate(directory_list):
        step_num = step_idx + 1
        print(f"\nProcessing Step {step_num}: {directory}")

        step_output_dir = os.path.join(output_base_dir, f"step_{step_num}")
        os.makedirs(step_output_dir, exist_ok=True)

        results_files = glob.glob(
            os.path.join(directory, "**", "*_results.csv"), recursive=True
        )
        if not results_files:
            print(f"  Warning: No results files found in {directory}")
            continue

        print(f"  Found {len(results_files)} results files")
        step_micrograph_parameters = {}

        # Process parameters files
        for results_file in results_files:
            micrograph_id = get_micrograph_id(results_file)
            params_file = results_file.replace(
                "_results.csv", "_results_parameters.csv"
            )

            if os.path.exists(params_file):
                params_df = load_parameters_file(params_file)
                if not params_df.empty:
                    step_micrograph_parameters[micrograph_id] = params_df.iloc[0]
                    if "num_correlations" in params_df.columns:
                        correlations = int(params_df.iloc[0]["num_correlations"])
                        micrograph_correlations[micrograph_id] += correlations
                        print(
                            f"  {micrograph_id}: Added {correlations} correlations "
                            f"(total: {micrograph_correlations[micrograph_id]})"
                        )
            else:
                print(f"  Warning: Parameters file not found for {results_file}")

        # Calculate thresholds
        micrograph_thresholds = calculate_micrograph_thresholds(
            micrograph_correlations, false_positive_rate
        )

        # Process results files
        for results_file in results_files:
            above_threshold_df, micrograph_id = process_results_file(
                results_file,
                micrograph_thresholds,
                step_micrograph_parameters,
                false_positive_rate,
                step_num,
            )

            if not above_threshold_df.empty:
                update_particle_data(
                    all_particles,
                    particle_step_map,
                    above_threshold_df,
                    micrograph_id,
                    step_num,
                )

        # Save step results
        save_step_results(all_particles, step_num, step_output_dir)

    # Save final results
    save_final_results(
        all_particles, micrograph_thresholds, micrograph_correlations, output_base_dir
    )

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
