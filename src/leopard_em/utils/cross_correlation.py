"""Core cross-correlation methods for single and stacks of image/templates."""

from typing import Literal

import torch


def handle_correlation_mode(
    cross_correlation: torch.Tensor,
    out_shape: tuple[int, ...],
    mode: Literal["valid", "same"],
) -> torch.Tensor:
    """Handle cropping for cross correlation mode.

     NOTE: 'full' mode is not implemented.

    Parameters
    ----------
    cross_correlation : torch.Tensor
        The cross correlation result.
    out_shape : tuple[int, ...]
        The desired shape of the output.
    mode : Literal["valid", "same"]
        The mode of the cross correlation. Either 'valid' or 'same'. See
        [numpy.correlate](https://numpy.org/doc/stable/reference/generated/
        numpy.convolve.html#numpy.convolve)
        for more details.
    """
    # Crop the result to the valid bounds
    if mode == "valid":
        slices = [slice(0, _out_s) for _out_s in out_shape]
        cross_correlation = cross_correlation[slices]
    elif mode == "same":
        pass
    elif mode == "full":
        raise NotImplementedError("Full mode not supported")
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return cross_correlation
