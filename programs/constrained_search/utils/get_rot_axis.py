"""Calculate the rotation axis for a pair of PDB structures."""

import sys

import mmdf
import numpy as np
import roma
import torch
from scipy.spatial.transform import Rotation


def extract_rotation_axis_angle(
    rotmat: torch.Tensor | np.ndarray,
) -> tuple[np.ndarray, float]:
    """Extract rotation axis and angle from rotation matrix handling edge cases.

    Attributes
    ----------
    rotmat: torch.Tensor | np.ndarray
        The rotation matrix either as a torch tensor or numpy array.

    Returns
    -------
    tuple[np.ndarray, float]
        The rotation axis and angle with angle in units of radians.
    """
    rotmat = rotmat.numpy() if isinstance(rotmat, torch.Tensor) else rotmat

    rotation = Rotation.from_matrix(rotmat)
    rotvec = rotation.as_rotvec()

    angle = np.linalg.norm(rotvec)

    # Handle edge case for very small angles (near zero)
    if np.abs(angle) < 1e-6:
        return np.array([0.0, 0.0, 1.0]), angle

    # NOTE: Edge case for angles near 180 degrees handled by scipy internally
    axis = rotvec / angle

    return axis, angle


def calculate_axis_euler_angles(axis: torch.Tensor | np.ndarray) -> tuple[float, float]:
    """Calculate Euler angles (ZYZ) that for the rotation axis.

    Attributes
    ----------
    axis: torch.Tensor | np.ndarray
        The rotation axis.

    Returns
    -------
    tuple[float, float]
        The Euler angles in units of degrees
    """
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    axis = axis.numpy() if isinstance(axis, torch.Tensor) else axis

    # Edge case for axis already aligned with z-axis
    if np.linalg.norm(axis - z_axis) < 1e-6:
        return 0.0, 0.0

    # Edge case for axis anti-aligned with z-axis
    if np.linalg.norm(axis + z_axis) < 1e-6:
        return 0.0, 180.0

    # Calculate theta - angle from z-axis (polar angle)
    cos_theta = np.dot(axis, z_axis)
    theta = np.acos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi

    # Calculate phi - angle in xy plane (azimuthal angle)
    phi = np.atan2(axis[1], axis[0]) * 180 / np.pi
    if phi < 0:
        phi += 360.0  # Convert to 0-360 range

    return phi, theta


def process_pdb_files(
    pdb_file1: str, pdb_file2: str
) -> tuple[np.ndarray, float, float, float]:
    """Helper function to calculate the rotation axis and angle for two PDB files.

    Parameters
    ----------
    pdb_file1: str
        Path to the first PDB file.
    pdb_file2: str
        Path to the second PDB file.

    Returns
    -------
    tuple[np.ndarray, float, float, float]
        The rotation axis, rotation angle in radians, and Euler angles (phi, theta).
    """
    # Read PDB files
    df1 = mmdf.read(pdb_file1)
    df2 = mmdf.read(pdb_file2)

    # Extract coordinates
    coords1 = torch.tensor(df1[["x", "y", "z"]].values, dtype=torch.float32)
    coords2 = torch.tensor(df2[["x", "y", "z"]].values, dtype=torch.float32)

    # Center coordinates
    centroid1 = coords1.mean(dim=0)
    centroid2 = coords2.mean(dim=0)
    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2

    # Calculate rotation matrix
    rotation_matrix, _ = roma.rigid_points_registration(
        coords1_centered, coords2_centered
    )

    # Extract rotation axis and angle plus Euler angles
    rotation_axis, rotation_angle = extract_rotation_axis_angle(rotation_matrix)
    phi, theta = calculate_axis_euler_angles(rotation_axis)

    # radians to degrees
    rotation_angle = np.rad2deg(rotation_angle)

    return rotation_axis, rotation_angle, phi, theta


def write_results(
    output_file: str,
    pdb_file1: str,
    pdb_file2: str,
    rotation_axis: np.ndarray,
    rotation_angle: float,
    phi: float,
    theta: float,
) -> None:
    """Helper function to write the script results to a file."""
    suggested_range = min(30.0, max(10.0, rotation_angle / 2))
    results_string = f"""# PDB Rotation Analysis Results\n
    Source PDB: {pdb_file1}
    Target PDB: {pdb_file2}

    ## Rotation Parameters
    Axis: {rotation_axis[0]:.6f} {rotation_axis[1]:.6f} {rotation_axis[2]:.6f}
    Angle: {rotation_angle:.6f} degrees\n

    ## Axis Orientation Angles (input for constrained search config)
    rotation_axis_euler_angles: [{phi:.2f}, {theta:.2f}, 0.0]\n

    ## Example constrained search config
    orientation_refinement_config:
      enabled: true
      out_of_plane_step: 1.0   # Step size around the rotation axis
      in_plane_step: 0.5       # Step size for fine adjustment angles
      rotation_axis_euler_angles: [{phi:.2f}, {theta:.2f}, 0.0]
      phi_min: -{suggested_range:.1f}  # Search range for around the axis
      phi_max: {suggested_range:.1f}
      theta_min: -2.0  # Small adjustments perpendicular to axis (optional)
      theta_max: 2.0
      psi_min: -2.0    # Small in-plane adjustments (optional)
      psi_max: 2.0
    """

    # Print the script results to the console
    print(results_string)

    # And also write them to a file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(results_string)

    print(f"Rotation analysis written to: {output_file}")


def main() -> None:
    """Calculate rotation axis for a pair of PDB structures."""
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <pdb_file1> <pdb_file2> <output_file>")
        sys.exit(1)

    pdb_file1 = sys.argv[1]
    pdb_file2 = sys.argv[2]
    output_file = sys.argv[3]

    rotation_axis, rotation_angle, phi, theta = process_pdb_files(pdb_file1, pdb_file2)
    write_results(
        output_file,
        pdb_file1,
        pdb_file2,
        rotation_axis,
        rotation_angle,
        phi,
        theta,
    )


if __name__ == "__main__":
    main()
