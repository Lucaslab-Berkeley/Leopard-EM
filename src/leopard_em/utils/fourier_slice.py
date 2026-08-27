"""Useful functions for extracting and filtering Fourier slices."""

import roma
import torch
from torch_fourier_slice import extract_central_slices_rfft_3d
from torch_fourier_slice.volume_utils import separable_sinc2_correction


def volume_to_rfft_fourier_slice(volume: torch.Tensor) -> torch.Tensor:
    """Prepares a 3D volume for Fourier slice extraction.

    Parameters
    ----------
    volume : torch.Tensor
        The input volume.

    Returns
    -------
    torch.Tensor
        The prepared volume in Fourier space ready for slice extraction.
    """
    assert volume.dim() == 3, "Volume must be 3D"

    sinc2 = separable_sinc2_correction(volume.shape, device=volume.device)
    volume = volume / sinc2

    # NOTE: There is an extra FFTshift step before the RFFT since, for some reason,
    # omitting this step will cause a 180 degree phase shift on odd (i, j, k)
    # structure factors in the Fourier domain. This just requires an extra
    # IFFTshift after converting a slice back to real-space (handled already).
    volume = torch.fft.fftshift(volume, dim=(0, 1, 2))  # pylint: disable=E1102
    volume_rfft = torch.fft.rfftn(volume, dim=(0, 1, 2))  # pylint: disable=E1102
    volume_rfft = torch.fft.fftshift(volume_rfft, dim=(0, 1))  # pylint: disable=E1102

    return volume_rfft


def _rfft_slices_to_real_projections(
    fourier_slices: torch.Tensor,
) -> torch.Tensor:
    """Convert Fourier slices to real-space projections.

    Parameters
    ----------
    fourier_slices : torch.Tensor
        The Fourier slices to convert. Inverse Fourier transform is applied
        across the last two dimensions.

    Returns
    -------
    torch.Tensor
        The real-space projections.
    """
    # pylint: disable=not-callable
    fourier_slices = torch.fft.fftshift(fourier_slices, dim=(-2,))
    # pylint: disable=not-callable
    projections = torch.fft.irfftn(fourier_slices, dim=(-2, -1))
    # pylint: disable=not-callable
    projections = torch.fft.ifftshift(projections, dim=(-2, -1))
    return projections


def get_rfft_slices_from_volume(
    volume: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
    psi: torch.Tensor,
    degrees: bool = True,
) -> torch.Tensor:
    """Helper function to get Fourier slices of a real-space volume.

    Parameters
    ----------
    volume : torch.Tensor
        The 3D volume to get Fourier slices from.
    phi : torch.Tensor
        The phi Euler angle.
    theta : torch.Tensor
        The theta Euler angle.
    psi : torch.Tensor
        The psi Euler angle.
    degrees : bool
        True if Euler angles are in degrees, False if in radians.

    Returns
    -------
    torch.Tensor
        The Fourier slices of the volume.

    """
    volume_rfft = volume_to_rfft_fourier_slice(volume)

    # Use roma to keep angles on same device
    rot_matrix = roma.euler_to_rotmat("ZYZ", (phi, theta, psi), degrees=degrees)

    # Use torch_fourier_slice to take the Fourier slice
    fourier_slices = extract_central_slices_rfft_3d(
        volume_rfft=volume_rfft,
        rotation_matrices=rot_matrix,
    )

    # Invert contrast to match image
    fourier_slices = -fourier_slices

    return fourier_slices


def get_real_space_projections_from_volume(
    volume: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
    psi: torch.Tensor,
    degrees: bool = True,
) -> torch.Tensor:
    """Real-space projections of a 3D volume.

    Note that Euler angles are in 'ZYZ' convention.

    Parameters
    ----------
    volume : torch.Tensor
        The 3D volume to get projections from.
    phi : torch.Tensor
        The phi Euler angle.
    theta : torch.Tensor
        The theta Euler angle.
    psi : torch.Tensor
        The psi Euler angle.
    degrees : bool
        True if Euler angles are in degrees, False if in radians.

    Returns
    -------
    torch.Tensor
        The real-space projections.
    """
    fourier_slices = get_rfft_slices_from_volume(
        volume=volume,
        phi=phi,
        theta=theta,
        psi=psi,
        degrees=degrees,
    )
    projections = _rfft_slices_to_real_projections(fourier_slices)

    return projections
