"""Functions related to result processing after backend functions."""

import numpy as np
import torch


def aggregate_distributed_results(
    results: list[dict[str, torch.Tensor | np.ndarray]],
) -> dict[str, torch.Tensor]:
    """Combine the 2DTM results from multiple devices.

    NOTE: This assumes that all tensors have been passed back to the CPU and are in
    the form of numpy arrays.

    Parameters
    ----------
    results : list[dict[str, np.ndarray]]
        List of dictionaries containing the results from each device. Each dictionary
        contains the following keys:
            - "mip": Maximum intensity projection of the cross-correlation values.
            - "best_global_index": Best global search index
            - "correlation_sum": Sum of cross-correlation values for each pixel.
            - "correlation_squared_sum": Sum of squared cross-correlation values for
              each pixel.
    """
    # Ensure all the tensors are passed back to CPU as numpy arrays
    # Not sure why cannot sync across devices, but this is a workaround
    results = [
        {
            key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
            for key, value in result.items()
        }
        for result in results
    ]

    # Stack results from all devices into a single array. Dim 0 is device index
    mips = np.stack([result["mip"] for result in results], axis=0)
    best_index = np.stack([result["best_global_index"] for result in results], axis=0)

    # Find the maximum MIP across all devices, then decode the best index
    mip_max = mips.max(axis=0)
    mip_argmax = mips.argmax(axis=0)
    best_index = np.take_along_axis(best_index, mip_argmax[None, ...], axis=0)[0]

    # Sum the sums and squared sums of the cross-correlation values
    correlation_sum = np.stack(
        [result["correlation_sum"] for result in results], axis=0
    ).sum(axis=0)
    correlation_squared_sum = np.stack(
        [result["correlation_squared_sum"] for result in results], axis=0
    ).sum(axis=0)

    # Cast back to torch tensors on the CPU
    mip_max = torch.from_numpy(mip_max)
    best_index = torch.from_numpy(best_index)
    correlation_sum = torch.from_numpy(correlation_sum)
    correlation_squared_sum = torch.from_numpy(correlation_squared_sum)

    return {
        "mip": mip_max,
        "best_global_index": best_index,
        "correlation_sum": correlation_sum,
        "correlation_squared_sum": correlation_squared_sum,
    }


def decode_global_search_index(
    global_indices: torch.Tensor,  # integer tensor
    pixel_values: torch.Tensor,  # (num_cs,)
    defocus_values: torch.Tensor,  # (num_defocus,)
    euler_angles: torch.Tensor,  # (num_orientations, 3)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode flattened global indices back into (cs, defocus, orientation)."""
    _ = pixel_values  # Unused, but possible to add in future

    # num_cs = pixel_values.shape[0]
    num_defocus = defocus_values.shape[0]
    num_orientations = euler_angles.shape[0]

    stride_cs = num_defocus * num_orientations
    stride_defocus = num_orientations

    # Calculate the indexes for each "best" array
    # pixel_idx = global_indices // stride_cs
    rem = global_indices % stride_cs
    defocus_idx = rem // stride_defocus
    orientations_idx = rem % stride_defocus

    phi = euler_angles[orientations_idx, 0]
    theta = euler_angles[orientations_idx, 1]
    psi = euler_angles[orientations_idx, 2]
    defocus = defocus_values[defocus_idx]
    # pixels = pixel_values[pixel_idx]

    return phi, theta, psi, defocus


def correlation_sum_and_squared_sum_to_mean_and_variance(
    correlation_sum: torch.Tensor,
    correlation_squared_sum: torch.Tensor,
    total_correlation_positions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert the sum and squared sum of the correlation values to mean and variance.

    Parameters
    ----------
    correlation_sum : torch.Tensor
        Sum of the correlation values.
    correlation_squared_sum : torch.Tensor
        Sum of the squared correlation values.
    total_correlation_positions : int
        Total number cross-correlograms calculated.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple containing the mean and variance of the correlation values.
    """
    correlation_mean = correlation_sum / total_correlation_positions
    correlation_variance = correlation_squared_sum / total_correlation_positions
    correlation_variance -= correlation_mean**2
    correlation_variance = torch.sqrt(torch.clamp(correlation_variance, min=0))
    return correlation_mean, correlation_variance


def scale_mip(
    mip: torch.Tensor,
    mip_scaled: torch.Tensor,
    correlation_sum: torch.Tensor,
    correlation_squared_sum: torch.Tensor,
    total_correlation_positions: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale the MIP to Z-score map by the mean and variance of the correlation values.

    Z-score is accounting for the variation in image intensity and spurious correlations
    by subtracting the mean and dividing by the standard deviation pixel-wise. Since
    cross-correlation values are roughly normally distributed for pure noise, Z-score
    effectively becomes a measure of how unexpected (highly correlated to the reference
    template) a region is in the image. Note that we are looking at maxima of millions
    of Gaussian distributions, so Z-score has to be compared with a generalized extreme
    value distribution (GEV) to determine significance (done elsewhere).

    NOTE: This method also updates the correlation_sum and correlation_squared_sum
    tensors in-place into the mean and variance, respectively. Likely should reflect
    conversions in variable names...

    Parameters
    ----------
    mip : torch.Tensor
        MIP of the correlation values.
    mip_scaled : torch.Tensor
        Scaled MIP of the correlation values.
    correlation_sum : torch.Tensor
        Sum of the correlation values. Updated to mean of the correlation values.
    correlation_squared_sum : torch.Tensor
        Sum of the squared correlation values. Updated to variance of the correlation.
    total_correlation_positions : int
        Total number cross-correlograms calculated.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple containing, in order, the MIP, scaled MIP, correlation mean, and
        correlation variance.
    """
    corr_mean, corr_variance = correlation_sum_and_squared_sum_to_mean_and_variance(
        correlation_sum, correlation_squared_sum, total_correlation_positions
    )

    # Calculate normalized MIP
    mip_scaled = mip - corr_mean
    torch.where(
        corr_variance != 0,  # preventing zero division error, albeit unlikely
        mip_scaled / corr_variance,
        torch.zeros_like(mip_scaled),
        out=mip_scaled,
    )

    # # Update correlation_sum and correlation_squared_sum to mean and variance
    # correlation_sum.copy_(corr_mean)
    # correlation_squared_sum.copy_(corr_variance)

    return mip, mip_scaled, corr_mean, corr_variance
