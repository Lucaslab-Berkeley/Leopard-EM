"""For getting the final rotation of a small particle relative to a large particle."""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def euler_to_rotation_matrix(
    phi: float,
    theta: float,
    psi: float,
) -> np.ndarray:
    """Convert Euler angles to rotation matrix using scipy.

    Parameters
    ----------
    phi: float
        First Euler angle
    theta: float
        Second Euler angle
    psi: float
        Third Euler angle

    Returns
    -------
    np.ndarray: Rotation matrix
    """
    # Create rotation object from Euler angles (ZYZ convention)
    rot = Rotation.from_euler("zyz", [phi, theta, psi], degrees=True)
    return rot.as_matrix()


def recover_constrained_rotation_parameters(
    R_large: np.ndarray,
    R_small: np.ndarray,
    rotation_axis_euler: list[float],
) -> tuple[list[float], float]:
    """
    Recover the rotation parameters from a constrained rotation.

    This matches the strategy used in ConstrainedOrientationConfig where:
    R_small = R_axis @ R_z(phi,theta,psi) @ R_axis^T @ R_large

    Parameters
    ----------
    R_large: np.ndarray
        Rotation matrix for large particle
    R_small: np.ndarray
        Rotation matrix for small particle
    rotation_axis_euler: list[float]
        Euler angles (phi, theta, psi) defining the rotation axis

    Returns
    -------
    recovered_euler: The recovered Euler angles around the constrained axis
    angle: The total rotation angle in degrees (with sign preserved)
    """
    # Convert rotation axis Euler angles to matrix
    R_axis = euler_to_rotation_matrix(*rotation_axis_euler)

    # Calculate transform: R_small = R_transform @ R_large
    # So R_transform = R_small @ R_large^T
    R_transform = np.dot(R_small, R_large.T)

    # Now reverse the formula: R_transform = R_axis @ R_z @ R_axis^T
    # So R_z = R_axis^T @ R_transform @ R_axis
    R_axis_inv = R_axis.T  # For rotation matrices, inverse = transpose
    R_z = np.dot(R_axis_inv, np.dot(R_transform, R_axis))

    # Convert R_z to a rotation object
    rot_z = Rotation.from_matrix(R_z)

    # Get Euler angles
    recovered_euler = rot_z.as_euler("zyz", degrees=True)

    # Get rotation vector (axis * angle)
    rotvec = rot_z.as_rotvec()

    # Extract axis and angle with sign
    axis_norm = np.linalg.norm(rotvec)
    if axis_norm < 1e-10:
        # Effectively no rotation
        return recovered_euler, 0.0

    # Get the normalized axis and signed angle
    axis = rotvec / axis_norm

    # For constrained rotation around Z axis, the primary component should be Z
    # We can use the z-component to determine the sign
    sign = 1 if axis[2] >= 0 else -1
    signed_angle = sign * np.degrees(axis_norm)

    return recovered_euler, signed_angle


def euclidean_distance(
    euler1: list[float],
    euler2: list[float],
) -> float:
    """
    Calculate Euclidean distance between two sets of Euler angles.

    Note: This is a simple distance metric and doesn't account for
    the periodic nature of angles.

    Parameters
    ----------
    euler1: list[float]
        First set of Euler angles
    euler2: list[float]
        Second set of Euler angles

    Returns
    -------
    float: Euclidean distance between the two sets of Euler angles
    """
    return np.sqrt(np.sum(np.square(np.array(euler1) - np.array(euler2))))


def main(
    large_csv: str,
    small_csv: str,
    output_csv: str,
    rotation_axis_euler: list[float],
) -> None:
    """Main function to recover constrained rotation parameters.

    Parameters
    ----------
    large_csv: str
        Path to large particle refinement results CSV
    small_csv: str
        Path to small particle constrained search results CSV
    output_csv: str
        Path to output CSV file
    rotation_axis_euler: list[float]
        Euler angles (phi theta psi) defining the rotation axis

    Returns
    -------
    None
    """
    # Load dataframes
    df_large = pd.read_csv(large_csv)
    df_small = pd.read_csv(small_csv)

    # Look for the "particle_index" column
    index_col = "particle_index"

    # Check if both dataframes have the particle_index column
    if index_col not in df_large.columns:
        raise ValueError(f"Column '{index_col}' not found in the large particle CSV")
    if index_col not in df_small.columns:
        raise ValueError(f"Column '{index_col}' not found in the small particle CSV")

    print(f"Using column '{index_col}' as particle index in both CSVs")
    print(f"Using rotation axis Euler angles: {rotation_axis_euler}")

    # Find common particles
    common_particles = set(df_large[index_col]).intersection(set(df_small[index_col]))
    print(f"Found {len(common_particles)} common particles")

    # Initialize result dataframe
    result_data = []

    # Make sure we have the required Euler angle columns
    required_cols = ["refined_phi", "refined_theta", "refined_psi"]
    for col in required_cols:
        if col not in df_large.columns or col not in df_small.columns:
            raise ValueError(f"Missing required column {col} in one of the input files")

    # Process each common particle
    for particle_idx in common_particles:
        large_row = df_large[df_large[index_col] == particle_idx].iloc[0]
        small_row = df_small[df_small[index_col] == particle_idx].iloc[0]

        # Get Euler angles
        large_angles = [
            large_row["refined_phi"],
            large_row["refined_theta"],
            large_row["refined_psi"],
        ]
        small_angles = [
            small_row["refined_phi"],
            small_row["refined_theta"],
            small_row["refined_psi"],
        ]

        # Convert to rotation matrices
        R_large = euler_to_rotation_matrix(*large_angles)
        R_small = euler_to_rotation_matrix(*small_angles)

        # Recover rotation parameters
        recovered_euler, signed_angle = recover_constrained_rotation_parameters(
            R_large, R_small, rotation_axis_euler
        )

        # Calculate Euclidean distance between large and small Euler angles
        euler_distance = euclidean_distance(large_angles, small_angles)

        # Get refined_scaled_mip for small particle (with fallback if not present)
        refined_scaled_mip = small_row.get("refined_scaled_mip", np.nan)
        if "refined_scaled_mip" not in small_row:
            # Try to find a similar column
            mip_cols = [col for col in small_row.index if "mip" in col.lower()]
            if mip_cols:
                refined_scaled_mip = small_row[mip_cols[0]]
                print(f"Using '{mip_cols[0]}' as MIP score column")

        # Append to results
        result_data.append(
            {
                "particle_index": particle_idx,
                "refined_scaled_mip": refined_scaled_mip,
                "large_phi": large_angles[0],
                "large_theta": large_angles[1],
                "large_psi": large_angles[2],
                "small_phi": small_angles[0],
                "small_theta": small_angles[1],
                "small_psi": small_angles[2],
                "euler_distance": euler_distance,
                "recovered_phi": recovered_euler[0],
                "recovered_theta": recovered_euler[1],
                "recovered_psi": recovered_euler[2],
                "rotation_angle": signed_angle,  # Now with sign preserved
            }
        )

        # Print the Euler angles and recovered parameters for this particle
        print(f"Particle {particle_idx}:")
        print(
            f"  Large particle Euler angles: "
            f"phi={large_angles[0]:.2f}°, theta={large_angles[1]:.2f}°, "
            f"psi={large_angles[2]:.2f}°"
        )
        print(
            f"  Small particle Euler angles: "
            f"phi={small_angles[0]:.2f}°, theta={small_angles[1]:.2f}°, "
            f"psi={small_angles[2]:.2f}°"
        )
        print(f"  Euclidean distance between Euler angles: {euler_distance:.2f}°")
        print(
            f"  Constrained rotation parameters (ZYZ): "
            f"phi={recovered_euler[0]:.2f}°, theta={recovered_euler[1]:.2f}°, "
            f"psi={recovered_euler[2]:.2f}°, "
            f"total angle={signed_angle:.2f}° (signed)"  # Sign indication
        )
        print()

    # Create and save result dataframe
    if result_data:
        result_df = pd.DataFrame(result_data)
        result_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    else:
        print("No common particles found, no output file created")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recover constrained rotation parameters"
    )
    parser.add_argument(
        "--large", required=True, help="Path to large particle refinement results CSV"
    )
    parser.add_argument(
        "--small",
        required=True,
        help="Path to small particle constrained search results CSV",
    )
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument(
        "--rotation-axis",
        nargs=3,
        type=float,
        required=True,
        help="Rotation axis as Euler angles (phi theta psi)",
    )
    parser.add_argument(
        "--index-column",
        type=str,
        default="particle_index",
        help="Name of the column with particle indices (default: particle_index)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.large):
        raise FileNotFoundError(f"Large particle CSV not found: {args.large}")
    if not os.path.exists(args.small):
        raise FileNotFoundError(f"Small particle CSV not found: {args.small}")

    main(args.large, args.small, args.output, args.rotation_axis)
