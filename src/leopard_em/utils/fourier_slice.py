"""Useful functions for extracting and filtering Fourier slices."""

import roma
import torch
import torch.nn.functional as F
from torch_fourier_slice import extract_central_slices_rfft_3d, transform_slice_2d
from torch_fourier_slice.volume_utils import (
    compute_cube_face_averages,
    separable_sinc2_correction,
)

# NOTE: Volume setup is duplicated here from 'torch_fourier_slice.project_3d_to_2d'
#       since we want to pre-compute this volume DFT once and re-use. Can be removed
#       once torch-fourier-slice exposes a volume-setup helper (see issue #156).

# Default zero-padding factor applied to template volumes before the 3D FFT.
DEFAULT_FOURIER_PAD_FACTOR = 2.0

# Width (in voxels) of the shell next to the volume faces whose mean is taken as the
# background "edge value". Same (arbitrary) choice as torch-fourier-slice.
_EDGE_SHELL_WIDTH = 4


def volume_to_rfft_fourier_slice(
    volume: torch.Tensor, pad_factor: float = DEFAULT_FOURIER_PAD_FACTOR
) -> torch.Tensor:
    """Prepares a 3D volume for Fourier slice extraction.

    Parameters
    ----------
    volume : torch.Tensor
        The input real-space volume with shape (d, h, w).
    pad_factor : float, optional
        Zero-padding factor applied to each dimension before the FFT, by default
        ``DEFAULT_FOURIER_PAD_FACTOR`` (2.0). A value of 1.0 disables padding.

    Returns
    -------
    torch.Tensor
        The prepared volume in Fourier space ready for slice extraction. Has shape
        (D, H, W // 2 + 1) where (D, H, W) is the padded real-space shape.

    Raises
    ------
    ValueError
        If ``pad_factor`` is less than 1.0.
    """
    assert volume.dim() == 3, "Volume must be 3D"
    if len(set(volume.shape)) != 1:
        raise ValueError(f"Volume must be cubic, got shape {tuple(volume.shape)}.")
    if pad_factor < 1.0:
        raise ValueError(f"pad_factor must be >= 1.0, got {pad_factor}.")

    # Subtract the edge value so the zero-padded region continues smoothly from zero
    volume = volume - _edge_value(volume)

    if pad_factor > 1.0:
        p = _pad_amount(volume.shape[-1], pad_factor)
        volume = F.pad(volume, pad=[p] * 6, mode="constant", value=0.0)

    # Divide by sinc^2 to correct for trilinear interpolation during slice extraction
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


def _pad_amount(size: int, pad_factor: float) -> int:
    """Voxels of zero-padding added to each side of a ``size``-long dimension."""
    return int((size * (pad_factor - 1.0)) // 2)


def _edge_value(volume: torch.Tensor) -> torch.Tensor:
    """Mean of the voxels within a thin shell next to the faces of a cubic volume."""
    n = max(1, min(_EDGE_SHELL_WIDTH, volume.shape[-1] // 2 - 1))
    edge_value: torch.Tensor = compute_cube_face_averages(volume, n=n)
    return edge_value


def projection_shape_from_filters(projective_filters: torch.Tensor) -> tuple[int, int]:
    """Real-space template projection shape (h, w) implied by RFFT-layout filters.

    Parameters
    ----------
    projective_filters : torch.Tensor
        Filters with RFFT layout in the last two dimensions, (..., h, w // 2 + 1).

    Returns
    -------
    tuple[int, int]
        The real-space projection shape (h, w). Assumes an even width.
    """
    return (
        int(projective_filters.shape[-2]),
        2 * (int(projective_filters.shape[-1]) - 1),
    )


def _crop_rfft_slices(
    fourier_slices: torch.Tensor,
    padded_shape: tuple[int, int],
    output_shape: tuple[int, int],
) -> torch.Tensor:
    """Crop RFFT slices of a padded volume back to the template size.

    Parameters
    ----------
    fourier_slices : torch.Tensor
        Slices with shape (..., H, W // 2 + 1), where (H, W) is ``padded_shape``.
    padded_shape : tuple[int, int]
        Real-space shape (H, W) of the padded projections.
    output_shape : tuple[int, int]
        Real-space shape (h, w) to crop to.

    Returns
    -------
    torch.Tensor
        Slices with shape (..., h, w // 2 + 1).
    """
    # pylint: disable=not-callable
    projections = torch.fft.ifftshift(fourier_slices, dim=(-2,))
    projections = torch.fft.irfftn(projections, s=padded_shape, dim=(-2, -1))
    projections = torch.fft.fftshift(projections, dim=(-2, -1))  # origin to center

    # Central crop, keeping the origin (index n // 2) at the center of the crop
    y0 = padded_shape[0] // 2 - output_shape[0] // 2
    x0 = padded_shape[1] // 2 - output_shape[1] // 2
    projections = projections[..., y0 : y0 + output_shape[0], x0 : x0 + output_shape[1]]

    projections = torch.fft.ifftshift(projections, dim=(-2, -1))  # center to origin
    cropped: torch.Tensor = torch.fft.rfftn(projections, dim=(-2, -1))
    cropped = torch.fft.fftshift(cropped, dim=(-2,))
    return cropped


def extract_template_fourier_slices(
    template_dft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    projection_shape: tuple[int, int],
    mag_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract template-size Fourier slices from a (possibly padded) template DFT.

    Parameters
    ----------
    template_dft : torch.Tensor
        Template volume prepared by :func:`volume_to_rfft_fourier_slice`, shape
        (D, H, W // 2 + 1).
    rotation_matrices : torch.Tensor
        Rotation matrices with shape (..., 3, 3).
    projection_shape : tuple[int, int]
        Real-space shape (h, w) of the unpadded template projections, e.g. from
        :func:`projection_shape_from_filters`.
    mag_matrix : torch.Tensor | None, optional
        Anisotropic magnification matrix of shape (2, 2). If None, no magnification
        transform is applied. Default is None.

    Returns
    -------
    torch.Tensor
        Fourier slices with shape (..., h, w // 2 + 1), RFFT with the zero frequency
        centered along dim -2 (same convention as ``extract_central_slices_rfft_3d``).

    Raises
    ------
    ValueError
        If ``projection_shape`` is larger than the volume's slice shape.
    """
    padded_shape = (int(template_dft.shape[-2]), 2 * (int(template_dft.shape[-1]) - 1))
    projection_shape = (int(projection_shape[0]), int(projection_shape[1]))
    if projection_shape[0] > padded_shape[0] or projection_shape[1] > padded_shape[1]:
        raise ValueError(
            f"Projection shape {projection_shape} is larger than the template volume "
            f"slice shape {padded_shape}."
        )

    fourier_slices = extract_central_slices_rfft_3d(
        volume_rfft=template_dft,
        rotation_matrices=rotation_matrices,
    )
    if mag_matrix is not None:
        fourier_slices = transform_slice_2d(
            projection_image_dfts=fourier_slices,
            rfft_shape=(int(template_dft.shape[-2]), int(template_dft.shape[-1])),
            stack_shape=tuple(rotation_matrices.shape[:-2]),
            transform_matrix=mag_matrix,
        )
    if padded_shape != projection_shape:
        fourier_slices = _crop_rfft_slices(
            fourier_slices, padded_shape, projection_shape
        )

    return fourier_slices


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
    pad_factor: float = DEFAULT_FOURIER_PAD_FACTOR,
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
    pad_factor : float, optional
        Zero-padding factor applied to the volume before the FFT, by default
        ``DEFAULT_FOURIER_PAD_FACTOR``. See :func:`volume_to_rfft_fourier_slice`.

    Returns
    -------
    torch.Tensor
        The Fourier slices of the volume, at the unpadded (h, w // 2 + 1) size.

    """
    volume_rfft = volume_to_rfft_fourier_slice(volume, pad_factor=pad_factor)

    # Use roma to keep angles on same device
    rot_matrix = roma.euler_to_rotmat("ZYZ", (phi, theta, psi), degrees=degrees)

    # Take the Fourier slices, cropped back to the unpadded projection size
    projection_shape = (int(volume.shape[-2]), int(volume.shape[-1]))
    fourier_slices = extract_template_fourier_slices(
        template_dft=volume_rfft,
        rotation_matrices=rot_matrix,
        projection_shape=projection_shape,
    )

    # Add back the subtracted edge value, projected through the volume depth. It only
    # touches the zero-frequency term, which sits at index h // 2 along dim -2.
    edge_value = _edge_value(volume)
    dc_value = edge_value * volume.shape[-3] * projection_shape[0] * projection_shape[1]
    fourier_slices = fourier_slices.clone()
    fourier_slices[..., projection_shape[0] // 2, 0] += dc_value

    # Invert contrast to match image
    fourier_slices = -fourier_slices

    return fourier_slices


def get_real_space_projections_from_volume(
    volume: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
    psi: torch.Tensor,
    degrees: bool = True,
    pad_factor: float = DEFAULT_FOURIER_PAD_FACTOR,
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
    pad_factor : float, optional
        Zero-padding factor applied to the volume before the FFT, by default
        ``DEFAULT_FOURIER_PAD_FACTOR``. See :func:`volume_to_rfft_fourier_slice`.

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
        pad_factor=pad_factor,
    )
    projections = _rfft_slices_to_real_projections(fourier_slices)

    return projections
