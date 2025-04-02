"""Calculates the center vector between two PDB structures."""

import argparse

import mmdf
import numpy as np
import pandas as pd
import roma
import torch


def calculate_mean_position(df: pd.DataFrame) -> torch.Tensor:
    """
    Calculate the mean position of a PDB structure.

    Attributes
    ----------
    df: pd.DataFrame
        The mmdf DataFrame with the PDB structure.

    Returns
    -------
    torch.Tensor
        The mean position as a 3D vector [x, y, z].
    """
    # Extract coordinates and calculate mean
    coords = torch.tensor(df[["x", "y", "z"]].values, dtype=torch.float32)
    mean_pos = coords.mean(dim=0)

    return mean_pos


def rotate_vector(vector: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Rotate a vector using a rotation matrix.

    Attributes
    ----------
    vector: torch.Tensor
        The vector to rotate.
    rotation_matrix: torch.Tensor
        The rotation matrix to use.

    Returns
    -------
        Rotated vector
    """
    # Convert to tensor if not already
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector, dtype=torch.float32)

    # Apply rotation
    rotated_vector = rotation_matrix @ vector

    return rotated_vector


def vector_to_euler(
    vector: torch.Tensor,
) -> tuple[float, float, float]:
    """
    Convert a vector to Euler angles.

    Attributes
    ----------
    vector: torch.Tensor
        The vector to convert to Euler angles.

    Returns
    -------
    tuple[float, float, float]
        The Euler angles in degrees (phi, theta, psi).
    """
    # Normalize the vector
    norm_vector = vector / torch.norm(vector)

    # Calculate theta as the angle from the z-axis
    theta = torch.acos(norm_vector[2])

    # Calculate phi as the azimuthal angle in the x-y plane
    phi = torch.atan2(norm_vector[1], norm_vector[0])

    # For a direction vector, psi can be set to 0
    psi = torch.tensor(0.0)

    # Convert to degrees using np.rad2deg
    phi_deg = np.rad2deg(phi.item())
    theta_deg = np.rad2deg(theta.item())
    psi_deg = np.rad2deg(psi.item())

    return phi_deg, theta_deg, psi_deg


def main() -> None:
    """Main function to calculate the center vector between two PDB structures."""
    parser = argparse.ArgumentParser(
        description="Calculate vector between PDB structures"
    )
    parser.add_argument("pdb_file1", help="PDB file of larger structure")
    parser.add_argument("pdb_file2", help="PDB file of smaller structure")
    parser.add_argument(
        "--num-rotations",
        type=int,
        default=5,
        help="Number of random rotations to test",
    )
    parser.add_argument(
        "--output-file", default="defocus_analysis.txt", help="Output analysis file"
    )
    args = parser.parse_args()

    # Parse PDB files using mmdf
    df1 = mmdf.read(args.pdb_file1)
    df2 = mmdf.read(args.pdb_file2)

    print(f"File 1: {args.pdb_file1} - {len(df1)} atoms")
    print(f"File 2: {args.pdb_file2} - {len(df2)} atoms")

    # Calculate mean positions at default orientation (0,0,0)
    mean_pos1 = calculate_mean_position(df1)
    mean_pos2 = calculate_mean_position(df2)

    # Calculate vector from PDB1 to PDB2
    vector = mean_pos2 - mean_pos1

    # Convert vector to Euler angles
    phi, theta, psi = vector_to_euler(vector)

    # Calculate Z-height difference (defocus)
    z_diff = vector[2].item()
    defocus_description = (
        f"{abs(z_diff):.2f} Angstroms {'below' if z_diff < 0 else 'above'}"
    )

    # Generate random Euler angles for testing
    rotations = []
    for i in range(args.num_rotations):
        # Generate some sample rotations (you can adjust as needed)
        if i == 0:
            # Include default orientation (0,0,0)
            rotations.append([0.0, 0.0, 0.0])
        else:
            # Generate some random rotations
            phi_rand = np.random.uniform(-180, 180)
            theta_rand = np.random.uniform(0, 180)
            psi_rand = np.random.uniform(-180, 180)
            rotations.append([phi_rand, theta_rand, psi_rand])

    # Convert to tensor for batch processing
    rotations_tensor = torch.tensor(rotations, dtype=torch.float32)

    # Print initial results
    print("\nInitial Analysis:")
    print(
        f"Vector from PDB1 to PDB2: [{vector[0]:.6f}, {vector[1]:.6f}, {vector[2]:.6f}]"
    )
    print(
        f"Vector Euler angles (ZYZ): Phi={phi:.2f}°, Theta={theta:.2f}°, Psi={psi:.2f}°"
    )
    print(f"Z-height difference (defocus): {defocus_description}")

    # Calculate defocus changes for different rotations
    print("\nDefocus changes for different rotations:")

    defocus_results = []

    for i, euler in enumerate(rotations_tensor):
        # Convert degrees to radians using np.deg2rad
        phi_rad, theta_rad, psi_rad = (np.deg2rad(angle.item()) for angle in euler)

        # Create rotation matrix using RoMA (ZYZ intrinsic convention)
        euler_angles = torch.tensor([phi_rad, theta_rad, psi_rad])
        rotation_matrix = roma.rotvec_to_rotmat(
            roma.euler_to_rotvec(convention="ZYZ", angles=euler_angles)
        )

        # Rotate the vector
        rotated_vector = rotate_vector(vector, rotation_matrix)

        # Extract new z-component (defocus)
        new_z_diff = rotated_vector[2].item()
        new_defocus = (
            f"{abs(new_z_diff):.2f} Angstroms {'below' if new_z_diff < 0 else 'above'}"
        )

        print(
            f"Rotation #{i+1} - Euler({euler[0]:.2f}°, {euler[1]:.2f}°, "
            f"{euler[2]:.2f}°): Defocus = {new_defocus}"
        )

        defocus_results.append(
            {
                "rotation": i + 1,
                "euler_angles": [e.item() for e in euler],
                "defocus": new_z_diff,
                "description": new_defocus,
            }
        )

    # Write analysis to file
    with open(args.output_file, "w") as f:
        f.write("# PDB Vector and Defocus Analysis\n\n")
        f.write(f"Source PDB 1: {args.pdb_file1}\n")
        f.write(f"Source PDB 2: {args.pdb_file2}\n\n")

        f.write("## Initial Vector Analysis\n")
        f.write(
            f"Vector PDB1-PDB2: [{vector[0]:.6f}, {vector[1]:.6f}, {vector[2]:.6f}]\n"
        )
        f.write(
            f"Vector Eulers (ZYZ): Phi={phi:.2f}°, Theta={theta:.2f}°, Psi={psi:.2f}°\n"
        )
        f.write(f"Z-height difference (defocus): {defocus_description}\n\n")

        f.write("## Defocus changes for different rotations\n")
        for result in defocus_results:
            euler = result["euler_angles"]
            f.write(
                f"Rotation #{result['rotation']} - "
                f"Euler({euler[0]:.2f}°, {euler[1]:.2f}°, {euler[2]:.2f}°): "
            )
            f.write(f"Defocus = {result['description']}\n")

    print(f"\nAnalysis results written to {args.output_file}")


if __name__ == "__main__":
    main()
