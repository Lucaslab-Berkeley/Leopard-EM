"""Calculates the center vector between two PDB structures."""

import argparse

import mmdf
import pandas as pd
import roma
import torch


def setup_argparse() -> argparse.Namespace:
    """Setup and return and argparse object for command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Calculate positional vector between PDB structures"
    )
    parser.add_argument("pdb_file1", help="PDB file of reference (larger) structure")
    parser.add_argument("pdb_file2", help="PDB file of constrained (smaller) structure")
    parser.add_argument("output_file", help="Output file for analysis results")
    parser.add_argument(
        "--num-rotations",
        type=int,
        default=5,
        help="Number of random rotations to test. Default is 5 rotations.",
    )

    return parser


def calculate_mean_position(df: pd.DataFrame) -> torch.Tensor:
    """Calculate the mean position of a PDB structure loaded into a dataframe."""
    coords = torch.tensor(df[["x", "y", "z"]].values, dtype=torch.float32)
    mean_pos = coords.mean(dim=0)

    return mean_pos


def calculate_relative_vectors(pdb_file1: str, pdb_file2: str) -> dict:
    """
    Calculate the relative position and orientation vectors between two PDB structures.

    Parameters
    ----------
    pdb_file1 : str
        Path to the first PDB file
    pdb_file2 : str
        Path to the second PDB file

    Returns
    -------
    dict
        Dictionary containing relative vector data including:
        - df1, df2: DataFrames for both PDB files
        - vector: Vector from PDB1 to PDB2
        - euler_angles: Phi, Theta, Psi angles
        - z_diff: Z height difference
        - defocus_description: Human-readable defocus description
    """
    # Parse PDB files using mmdf
    df1 = mmdf.read(pdb_file1)
    df2 = mmdf.read(pdb_file2)

    print(f"File 1: {pdb_file1} - {len(df1)} atoms")
    print(f"File 2: {pdb_file2} - {len(df2)} atoms")

    # Calculate mean positions at default orientation (0, 0, 0)
    mean_pos1 = calculate_mean_position(df1)
    mean_pos2 = calculate_mean_position(df2)

    # Calculate vector from PDB1 to PDB2
    vector = mean_pos2 - mean_pos1

    # Convert vector to Euler angles
    phi, theta, psi = roma.rotvec_to_euler(
        convention="ZYZ", rotvec=vector, degrees=True, as_tuple=True
    )

    # Calculate Z-height difference (defocus)
    z_diff = vector[2].item()
    defocus_description = (
        f"{abs(z_diff):.2f} Angstroms {'below' if z_diff < 0 else 'above'}"
    )

    # Print initial results
    initial_results = f"""Initial Analysis:
    Vector from PDB1 to PDB2: [{vector[0]:.6f}, {vector[1]:.6f}, {vector[2]:.6f}]
    Vector Euler angles (ZYZ, deg): Phi={phi:.2f}, Theta={theta:.2f}, Psi={psi:.2f}
    Z-height difference (defocus): {defocus_description}
    """
    print(initial_results)

    return {
        "df1": df1,
        "df2": df2,
        "vector": vector,
        "euler_angles": (phi, theta, psi),
        "z_diff": z_diff,
        "defocus_description": defocus_description,
    }


def process_rotations(vector: torch.Tensor, num_rotations: int) -> list:
    """
    Process each rotation and calculate the resulting defocus.

    Parameters
    ----------
    vector : torch.Tensor
        The original vector between structures
    num_rotations : int
        Number of random rotations to test (in addition to default orientation)

    Returns
    -------
    list
        List of dictionaries with defocus results for each rotation
    """
    print("\nDefocus changes for different rotations:")
    defocus_results = []

    for i in range(num_rotations + 1):
        if i == 0:
            rand_rotmat = torch.eye(3)
        else:
            rand_rotmat = roma.random_rotmat()

        rand_euler = roma.rotmat_to_euler("ZYZ", rand_rotmat, degrees=True)
        rotated_vector = rand_rotmat @ vector

        # Extract new z-component (defocus)
        new_z_diff = rotated_vector[2].item()
        new_defocus = (
            f"{abs(new_z_diff):.2f} Angstroms {'below' if new_z_diff < 0 else 'above'}"
        )
        print(f"Rotation #{i+1} - {rand_euler}: Defocus = {new_defocus}")

        defocus_results.append(
            {
                "rotation": i + 1,
                "euler_angles": [angle.item() for angle in rand_euler],
                "defocus": new_z_diff,
                "description": new_defocus,
            }
        )

    return defocus_results


def write_results_to_file(
    output_file: str,
    pdb_file1: str,
    pdb_file2: str,
    vector_info: dict,
    defocus_results: list,
) -> None:
    """
    Write analysis results to output file.

    Parameters
    ----------
    output_file : str
        Path to output file
    pdb_file1 : str
        Path to first PDB file
    pdb_file2 : str
        Path to second PDB file
    vector_info : dict
        Dictionary with vector data from calculate_relative_vectors
    defocus_results : list
        List of defocus results from process_rotations
    """
    vector = vector_info["vector"]
    phi, theta, psi = vector_info["euler_angles"]
    defocus_description = vector_info["defocus_description"]

    result_string = f"""# PDB Vector and Defocus Analysis
    Source PDB 1: {pdb_file1}
    Source PDB 2: {pdb_file2}

    ## Initial Vector Analysis
    Vector PDB1-PDB2: [{vector[0]:.6f}, {vector[1]:.6f}, {vector[2]:.6f}]
    Vector Eulers (ZYZ, deg): Phi={phi:.2f}, Theta={theta:.2f}, Psi={psi:.2f}
    Z-height difference (defocus): {defocus_description}

    ## Defocus changes for different rotations
    """
    for result in defocus_results:
        euler = result["euler_angles"]
        result_string += (
            f"    Rotation #{result['rotation']} - "
            f"    Euler({euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}): "
        )
        result_string += f"Defocus = {result['description']}\n"

    # Write results to file
    with open(output_file, "w") as f:
        f.write(result_string)

    print(f"\nAnalysis results written to {output_file}")


def main() -> None:
    """Main function to calculate the center vector between two PDB structures."""
    # Setup argparse
    parser = setup_argparse()
    args = parser.parse_args()

    vector_info = calculate_relative_vectors(args.pdb_file1, args.pdb_file2)
    defocus_results = process_rotations(vector_info["vector"], args.num_rotations)
    write_results_to_file(
        args.output_file, args.pdb_file1, args.pdb_file2, vector_info, defocus_results
    )


if __name__ == "__main__":
    main()
