"""Search parameter utility functions."""

import torch


def get_cs_range(
    pixel_size: float,
    pixel_size_offsets: torch.Tensor,
    cs: float = 2.7,
) -> torch.Tensor:
    """Get the Cs values for a  range of pixel sizes.

    Parameters
    ----------
    pixel_size : float
        The nominal pixel size.
    pixel_size_offsets : torch.Tensor
        The pixel size offsets.
    cs : float, optional
        The Cs (spherical aberration) value, by default 2.7.

    Returns
    -------
    torch.Tensor
        The Cs values for the range of pixel sizes.
    """
    pixel_sizes = pixel_size + pixel_size_offsets
    cs_values = cs / torch.pow(pixel_sizes / pixel_size, 4)
    return cs_values


def cs_to_pixel_size(
    cs_vals: torch.Tensor,
    nominal_pixel_size: float,
    nominal_cs: float = 2.7,
) -> torch.Tensor:
    """Convert Cs values to pixel sizes.

    Parameters
    ----------
    cs_vals : torch.Tensor
        The Cs (spherical aberration) values.
    nominal_pixel_size : float
        The nominal pixel size.
    nominal_cs : float, optional
        The nominal Cs value, by default 2.7.

    Returns
    -------
    torch.Tensor
        The pixel sizes.
    """
    pixel_size = torch.pow(nominal_cs / cs_vals, 0.25) * nominal_pixel_size
    return pixel_size


def get_search_tensors(
    min_val: float,
    max_val: float,
    step_size: float,
    skip_enforce_zero: bool = False,
) -> torch.tensor:
    """Get the search tensors (pixel or defocus) for a given range and step size.

    Parameters
    ----------
    min_val : float
        The minimum value.
    max_val : float
        The maximum value.
    step_size : float
        The step size.
    skip_enforce_zero : bool, optional
        Whether to skip enforcing a zero value, by default False.

    Returns
    -------
    torch.tensor
        The search tensors.
    """
    vals = torch.arange(
        min_val,
        max_val + step_size,
        step_size,
        dtype=torch.float32,
    )

    if abs(torch.min(torch.abs(vals))) > 1e-6 and not skip_enforce_zero:
        vals = torch.cat([vals, torch.tensor([0.0])])
        # Re-sort pixel sizes
        vals = torch.sort(vals)[0]

    return vals
