"""Calculate the rotation axis for a pair of PDB structures."""

import math
import sys

import mmdf
import roma
import torch


def extract_rotation_axis_angle(rotmat: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Extract rotation axis and angle from rotation matrix.

    Attributes
    ----------
    rotmat: torch.Tensor
        The rotation matrix.

    Returns
    -------
    tuple[torch.Tensor, float]
        The rotation axis and angle.
    """
    # Calculate angle from trace
    trace = torch.trace(rotmat)
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    angle = torch.acos(cos_theta)

    # Handle special cases (very small angles or near 180 degrees)
    if torch.abs(angle) < 1e-6:
        return torch.tensor([0.0, 0.0, 1.0]), angle
    elif torch.abs(angle - math.pi) < 1e-6:
        diag = torch.diag(rotmat) + 1
        axis_idx = torch.argmax(diag)
        axis = rotmat[:, axis_idx].clone()
        axis = axis / torch.norm(axis)
        # Ensure the axis points in the positive z direction
        if axis[2] < 0:
            axis = -axis
        return axis, angle

    # Normal case - extract axis
    axis = torch.tensor(
        [
            rotmat[2, 1] - rotmat[1, 2],
            rotmat[0, 2] - rotmat[2, 0],
            rotmat[1, 0] - rotmat[0, 1],
        ]
    )

    # Normalize the axis
    axis = axis / torch.norm(axis)

    # Ensure the axis points in the positive z direction
    if axis[2] < 0:
        axis = -axis
        angle = -angle  # Also flip the angle when we flip the axis
        angle = angle + 2 * math.pi if angle < 0 else angle  # Keep angle positive

    return axis, angle


def calculate_axis_euler_angles(axis: torch.Tensor) -> tuple[float, float]:
    """Calculate Euler angles (ZYZ) that for the rotation axis.

    Attributes
    ----------
    axis: torch.Tensor
        The rotation axis.

    Returns
    -------
    tuple[float, float]
        The Euler angles.
    """
    # Z-axis unit vector
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    # Handle special cases
    if torch.norm(axis - z_axis) < 1e-6:
        return 0.0, 0.0  # Axis aligned with z-axis

    if torch.norm(axis + z_axis) < 1e-6:
        return 0.0, 180.0  # Axis anti-aligned with z-axis

    # Calculate theta - angle from z-axis (polar angle)
    cos_theta = torch.dot(axis, z_axis)
    theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0)) * 180 / math.pi

    # Calculate phi - angle in xy plane (azimuthal angle)
    phi = torch.atan2(axis[1], axis[0]) * 180 / math.pi
    if phi < 0:
        phi += 360.0  # Convert to 0-360 range

    return phi, theta


def main() -> None:
    """Calculate rotation axis for a pair of PDB structures."""
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <pdb_file1> <pdb_file2> <output_file>")
        sys.exit(1)

    pdb_file1 = sys.argv[1]
    pdb_file2 = sys.argv[2]
    output_file = sys.argv[3]

    # Parse PDB files
    df1 = mmdf.read(pdb_file1)
    df2 = mmdf.read(pdb_file2)

    # Check if atom counts match
    if len(df1) != len(df2):
        print("Warning: PDB files have different numbers of atoms:")
        print(f"  {pdb_file1}: {len(df1)} atoms")
        print(f"  {pdb_file2}: {len(df2)} atoms")
        print("\nAttempting to filter to CA (alpha carbon) atoms for alignment...")

        # Filter to CA atoms if available
        if "atom_name" in df1.columns and "atom_name" in df2.columns:
            df1_ca = df1[df1["atom_name"] == "CA"].copy()
            df2_ca = df2[df2["atom_name"] == "CA"].copy()

            if len(df1_ca) > 0 and len(df2_ca) > 0:
                if len(df1_ca) != len(df2_ca):
                    print(
                        "Error: Even after filtering to CA atoms, counts don't match:"
                    )
                    print(f"  {pdb_file1}: {len(df1_ca)} CA atoms")
                    print(f"  {pdb_file2}: {len(df2_ca)} CA atoms")
                    print(
                        "\nCannot perform rigid registration with "
                        "mismatched atom counts."
                    )
                    sys.exit(1)
                print(f"Using {len(df1_ca)} CA atoms for alignment.")
                df1 = df1_ca
                df2 = df2_ca
            else:
                print("Error: No CA atoms found in one or both PDB files.")
                print(
                    "Cannot perform rigid registration with " "mismatched atom counts."
                )
                sys.exit(1)
        else:
            print(
                "Error: Cannot filter atoms - 'atom_name' column not found "
                "in PDB data."
            )
            print("Cannot perform rigid registration with mismatched atom counts.")
            sys.exit(1)

    # Extract coordinates
    coords1 = torch.tensor(df1[["x", "y", "z"]].values, dtype=torch.float32)
    coords2 = torch.tensor(df2[["x", "y", "z"]].values, dtype=torch.float32)

    # Center coordinates
    centroid1 = coords1.mean(dim=0)
    centroid2 = coords2.mean(dim=0)
    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2

    # Calculate rotation matrix
    # rotation_matrix = calculate_rotation_matrix(coords1_centered, coords2_centered)
    rotation_matrix, _ = roma.rigid_points_registration(
        coords1_centered, coords2_centered
    )
    # Extract rotation axis and angle
    rotation_axis, rotation_angle = extract_rotation_axis_angle(rotation_matrix)

    # Calculate Euler angles for the rotation axis
    phi, theta = calculate_axis_euler_angles(rotation_axis)

    # Output results
    angle_degrees = float(rotation_angle) * 180 / math.pi

    print("\nRotation Analysis Results:")
    print(
        f"Rotation Axis: [{rotation_axis[0]:.6f}, "
        f"{rotation_axis[1]:.6f}, {rotation_axis[2]:.6f}]"
    )
    print(f"Rotation Angle: {angle_degrees:.2f} degrees")

    print("\nEuler Angles of the rotation axis (ZYZ convention):")
    print(f"  Phi:   {phi:.2f} degrees (range 0-360)")
    print(f"  Theta: {theta:.2f} degrees (range 0-180)")

    # Write results to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# PDB Rotation Analysis Results\n\n")
        f.write(f"Source PDB: {pdb_file1}\n")
        f.write(f"Target PDB: {pdb_file2}\n")
        f.write(f"Number of atoms: {len(df1)}\n\n")

        f.write("## Rotation Parameters\n")
        f.write(
            f"Axis: {rotation_axis[0]:.6f} "
            f"{rotation_axis[1]:.6f} {rotation_axis[2]:.6f}\n"
        )
        f.write(f"Angle: {angle_degrees:.6f} degrees\n\n")

        f.write("## Axis Orientation Angles (for constrained search config)\n")
        f.write(f"rotation_axis_euler_angles: [{phi:.2f}, {theta:.2f}, 0.0]\n\n")

        f.write("## Example constrained search config\n")
        f.write("orientation_refinement_config:\n")
        f.write("  enabled: true\n")
        f.write("  out_of_plane_step: 1.0   # Step size around the rotation axis\n")
        f.write("  in_plane_step: 0.5       # Step size for fine adjustment angles\n")
        f.write(f"  rotation_axis_euler_angles: [{phi:.2f}, {theta:.2f}, 0.0]\n")
        suggested_range = min(30.0, max(10.0, angle_degrees / 2))
        f.write(
            f"  phi_min: -{suggested_range:.1f}  # Search range for around the axis\n"
        )
        f.write(f"  phi_max: {suggested_range:.1f}\n")
        f.write(
            "  theta_min: -2.0  # Small adjustments perpendicular to axis (optional)\n"
        )
        f.write("  theta_max: 2.0\n")
        f.write("  psi_min: -2.0    # Small in-plane adjustments (optional)\n")
        f.write("  psi_max: 2.0\n")

    print(f"Rotation analysis written to: {output_file}")
    print("\nFor your constrained search config.yaml:")
    print(f"rotation_axis_euler_angles: [{phi:.2f}, {theta:.2f}, 0.0]")


if __name__ == "__main__":
    main()
