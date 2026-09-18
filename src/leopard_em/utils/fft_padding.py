"""Padding images up to fast FFT sizes and cropping results back down."""

from dataclasses import dataclass
from typing import Any

import torch
from torch_grid_utils import next_fft_size

DEFAULT_FFT_FACTORS = (2, 3)


class FFTPaddingWarning(UserWarning):
    """Warning raised when automatic FFT padding changes behavior materially."""


def next_even_fft_size(n: int, factors: tuple[int, ...] = DEFAULT_FFT_FACTORS) -> int:
    """Smallest **even** integer ``>= n`` whose prime factors all lie in ``factors``.

    Parameters
    ----------
    n : int
        Minimum required size, typically a micrograph edge length.
    factors : tuple[int, ...], optional
        Allowed prime factors. Must contain 2. Default is ``(2, 3)``.

    Returns
    -------
    int
        The smallest even ``factors``-smooth integer greater than or equal to ``n``.

    Raises
    ------
    ValueError
        If ``2`` is not present in ``factors``, in which case no even size could
        ever satisfy the smoothness constraint.

    Examples
    --------
    >>> next_even_fft_size(4000)
    4096
    >>> next_even_fft_size(2050)  # plain next_fft_size would return an odd 2187
    2304
    """
    if 2 not in factors:
        raise ValueError(
            f"'factors' must contain 2 so that an even size is "
            f"reachable, got {factors}."
        )
    if n <= 2:
        return 2

    return 2 * int(next_fft_size((n + 1) // 2, factors=factors))


def next_even_fft_shape(
    image_shape: tuple[int, int], factors: tuple[int, ...] = DEFAULT_FFT_FACTORS
) -> tuple[int, int]:
    """Apply :func:`next_even_fft_size` independently to each image dimension."""
    return (
        next_even_fft_size(image_shape[0], factors),
        next_even_fft_size(image_shape[1], factors),
    )


def pad_image_gaussian(
    image: torch.Tensor, padded_shape: tuple[int, int], seed: int = 0
) -> torch.Tensor:
    """Pad a 2D image at the bottom and right with Gaussian noise.

    Parameters
    ----------
    image : torch.Tensor
        Real-space 2D image of shape ``(h, w)``.
    padded_shape : tuple[int, int]
        Target real-space shape ``(H, W)``. Must be at least ``image.shape`` in both
        dimensions.
    seed : int, optional
        Seed for the noise. A local :class:`torch.Generator` is used so the global
        RNG state is left untouched. Default is 0.

    Returns
    -------
    torch.Tensor
        Padded image of shape ``padded_shape``, with the same dtype and device as
        ``image``. Returned unchanged (not copied) when no padding is required.

    Raises
    ------
    ValueError
        If ``image`` is not 2D, or ``padded_shape`` is smaller than the image in
        either dimension.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {tuple(image.shape)}.")

    height, width = int(image.shape[0]), int(image.shape[1])
    pad_height = padded_shape[0] - height
    pad_width = padded_shape[1] - width
    if pad_height < 0 or pad_width < 0:
        raise ValueError(
            f"'padded_shape' {padded_shape} is smaller than the image shape "
            f"{(height, width)} in at least one dimension."
        )
    if pad_height == 0 and pad_width == 0:
        return image

    generator = torch.Generator(device=image.device).manual_seed(seed)
    padded = torch.empty(padded_shape, dtype=image.dtype, device=image.device)
    padded[:height, :width] = image

    pad_mask = torch.ones(padded_shape, dtype=torch.bool, device=image.device)
    pad_mask[:height, :width] = False
    num_pad_values = int(pad_mask.sum())

    noise = torch.randn(
        num_pad_values,
        generator=generator,
        dtype=image.dtype,
        device=image.device,
    )
    if num_pad_values > 1:
        noise = (noise - noise.mean()) / noise.std()
    else:
        noise = torch.zeros_like(noise)
    padded[pad_mask] = noise * image.std() + image.mean()

    return padded


def crop_bottom_right(
    tensor: torch.Tensor, crop_shape: tuple[int, int]
) -> torch.Tensor:
    """Crop the trailing two dimensions of a tensor, keeping the top-left origin."""
    if crop_shape[0] > tensor.shape[-2] or crop_shape[1] > tensor.shape[-1]:
        raise ValueError(
            f"Cannot crop a tensor with trailing shape "
            f"{tuple(tensor.shape[-2:])} up to {crop_shape}."
        )
    return tensor[..., : crop_shape[0], : crop_shape[1]]


def filter_correlation_table(
    correlation_table: dict[str, Any], valid_shape: tuple[int, int]
) -> dict[str, Any]:
    """Drop correlation-table rows whose position lies outside the valid region.

    Parameters
    ----------
    correlation_table : dict[str, Any]
        Processed correlation table as returned by the backend.
    valid_shape : tuple[int, int]
        Shape ``(H, W)`` of the valid correlation region to retain.

    Returns
    -------
    dict[str, Any]
        Table containing only rows with ``y < H`` and ``x < W``. The input object is
        returned unchanged when every row is already inside the valid region.
    """
    valid_height, valid_width = valid_shape
    y_positions = correlation_table["y"]
    x_positions = correlation_table["x"]
    num_rows = len(y_positions)

    keep = [
        index
        for index, (y, x) in enumerate(zip(y_positions, x_positions, strict=True))
        if y < valid_height and x < valid_width
    ]
    if len(keep) == num_rows:
        return correlation_table

    return {
        key: (
            [value[index] for index in keep]
            if isinstance(value, list) and len(value) == num_rows
            else value
        )
        for key, value in correlation_table.items()
    }


@dataclass(frozen=True)
class FFTPaddingPlan:
    """Immutable record of how one micrograph is padded up to a fast FFT size.

    Attributes
    ----------
    original_shape : tuple[int, int]
        Real-space ``(H, W)`` of the unpadded micrograph.
    padded_shape : tuple[int, int]
        Real-space ``(H, W)`` actually handed to the backend.
    template_shape : tuple[int, int]
        Real-space ``(h, w)`` of a single template projection.
    noise_seed : int
        Seed for the Gaussian fill, so a plan fully reproduces a padded image.
    effective_backend : str
        Cross-correlation backend. Typically the requested backend, but for "zipfft" may
        fall back to PyTorch "streamed" if no supported image shape exists.

    Methods
    -------
    pad(image)
        Pad a real-space image according to this plan.
    unpad(tensor)
        Crop a padded result map back to the unpadded extent.
    """

    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    template_shape: tuple[int, int]
    noise_seed: int = 0
    effective_backend: str = "streamed"

    def __post_init__(self) -> None:
        """Validate that the recorded shapes are mutually consistent."""
        if (
            self.padded_shape[0] < self.original_shape[0]
            or self.padded_shape[1] < self.original_shape[1]
        ):
            raise ValueError(
                f"Padded shape {self.padded_shape} is smaller than the original "
                f"shape {self.original_shape}."
            )
        if (
            self.template_shape[0] > self.original_shape[0]
            or self.template_shape[1] > self.original_shape[1]
        ):
            raise ValueError(
                f"Template shape {self.template_shape} does not fit within the "
                f"image shape {self.original_shape}."
            )

    @property
    def is_padded(self) -> bool:
        """Whether this plan actually changes the image size."""
        return self.padded_shape != self.original_shape

    @property
    def pad_amount(self) -> tuple[int, int]:
        """Rows and columns added to the bottom and right, respectively."""
        return (
            self.padded_shape[0] - self.original_shape[0],
            self.padded_shape[1] - self.original_shape[1],
        )

    @property
    def area_growth(self) -> float:
        """Fractional increase in pixel count caused by padding."""
        original_area = self.original_shape[0] * self.original_shape[1]
        padded_area = self.padded_shape[0] * self.padded_shape[1]
        return padded_area / original_area - 1.0

    @property
    def unpadded_valid_shape(self) -> tuple[int, int]:
        """Shape of the "valid" correlation region for the *unpadded* image."""
        return (
            self.original_shape[0] - self.template_shape[0] + 1,
            self.original_shape[1] - self.template_shape[1] + 1,
        )

    def pad(self, image: torch.Tensor) -> torch.Tensor:
        """Pad a real-space image according to this plan.

        Parameters
        ----------
        image : torch.Tensor
            Real-space 2D image matching ``original_shape``.

        Returns
        -------
        torch.Tensor
            Image padded to ``padded_shape``.
        """
        return pad_image_gaussian(image, self.padded_shape, seed=self.noise_seed)

    def unpad(self, tensor: torch.Tensor) -> torch.Tensor:
        """Crop a padded result map back to what an unpadded search would produce.

        Parameters
        ----------
        tensor : torch.Tensor
            Result map whose trailing two dimensions track the padded image shape.

        Returns
        -------
        torch.Tensor
            Cropped view of ``tensor``.
        """
        pad_height, pad_width = self.pad_amount
        return crop_bottom_right(
            tensor,
            (tensor.shape[-2] - pad_height, tensor.shape[-1] - pad_width),
        )
