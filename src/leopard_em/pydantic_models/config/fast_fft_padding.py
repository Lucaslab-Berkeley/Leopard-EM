"""Configuration for automatically padding images up to fast FFT sizes."""

import warnings
from typing import Annotated

from pydantic import Field, field_validator

from leopard_em.pydantic_models.custom_types import BaseModel2DTM
from leopard_em.utils.fft_padding import (
    DEFAULT_FFT_FACTORS,
    FFTPaddingPlan,
    FFTPaddingWarning,
    next_even_fft_shape,
)
from leopard_em.utils.zipfft_support import ZIPFFT_AVAILABLE, snap_image_shape_to_zipfft

# Fractional increase to raise warning about
LARGE_AREA_GROWTH = 0.22


class FastFFTPaddingConfig(BaseModel2DTM):
    """Configuration for padding a micrograph up to an FFT-friendly size.

    Attributes
    ----------
    enabled : bool
        If True, pad the image up to a fast FFT size. Default is True.
    allowed_factors : list[int]
        Prime factors permitted in the padded image size. Must contain 2.
    target_shape : Optional[list[int]]
        Explicit padded shape ``[H, W]``, bypassing automatic size selection. Must be
        at least as large as the image in both dimensions. Default is None.
    noise_seed : int
        Seed for the Gaussian noise filling the padded region. Default is 0.

    Methods
    -------
    make_plan(image_shape, template_shape, backend)
        Build the immutable padding plan for a given image and template.
    """

    enabled: bool = True
    allowed_factors: Annotated[
        list[Annotated[int, Field(ge=2)]], Field(min_length=1)
    ] = Field(default_factory=lambda: list(DEFAULT_FFT_FACTORS))
    target_shape: (
        Annotated[list[Annotated[int, Field(ge=1)]], Field(min_length=2, max_length=2)]
        | None
    ) = None
    noise_seed: int = 0

    @field_validator("allowed_factors")  # type: ignore[misc]
    @classmethod
    def validate_allowed_factors(cls, value: list[int]) -> list[int]:
        """Ensure an even padded size is reachable.

        Parameters
        ----------
        value : list[int]
            Candidate list of allowed prime factors.

        Returns
        -------
        list[int]
            The validated list.

        Raises
        ------
        ValueError
            If 2 is not among the allowed factors.
        """
        if 2 not in value:
            raise ValueError(
                "'allowed_factors' must contain 2; padded image dimensions must be "
                "even for the backend's real-FFT shape conventions to hold."
            )
        return value

    def make_plan(
        self,
        image_shape: tuple[int, int],
        template_shape: tuple[int, int],
        backend: str,
    ) -> FFTPaddingPlan:
        """Determine how an image of ``image_shape`` should be padded.

        Parameters
        ----------
        image_shape : tuple[int, int]
            Real-space shape ``(H, W)`` of the unpadded micrograph.
        template_shape : tuple[int, int]
            Real-space projection shape ``(h, w)``.
        backend : str
            Cross-correlation backend from the computational config. When ``"zipfft"``
            the padded size is snapped to a compiled zipFFT shape where possible.

        Returns
        -------
        FFTPaddingPlan
            Immutable record of the original shape, padded shape, and noise seed.

        Raises
        ------
        ValueError
            If ``target_shape`` is smaller than the image in either dimension, or is odd
            on RFFT dimension.
        """
        if not self.enabled:
            return FFTPaddingPlan(
                original_shape=image_shape,
                padded_shape=image_shape,
                template_shape=template_shape,
                noise_seed=self.noise_seed,
                effective_backend=backend,
            )

        effective_backend = backend
        # Ensure padded shape is at least as large as the image
        if self.target_shape is not None:
            padded_shape = (int(self.target_shape[0]), int(self.target_shape[1]))
            if padded_shape[0] < image_shape[0] or padded_shape[1] < image_shape[1]:
                raise ValueError(
                    f"Configured 'target_shape' {padded_shape} is smaller than the "
                    f"image shape {image_shape}."
                )
            if padded_shape[1] % 2 != 0:
                raise ValueError(
                    f"Configured 'target_shape' {padded_shape} must be even on last "
                    f"dimensions for the backend's real-FFT shape conventions."
                )
        else:
            padded_shape, effective_backend = self._automatic_padded_shape(
                image_shape, template_shape, backend
            )

        plan = FFTPaddingPlan(
            original_shape=image_shape,
            padded_shape=padded_shape,
            template_shape=template_shape,
            noise_seed=self.noise_seed,
            effective_backend=effective_backend,
        )

        self._warn_if_padding_large(plan)

        return plan

    def _automatic_padded_shape(
        self,
        image_shape: tuple[int, int],
        template_shape: tuple[int, int],
        backend: str,
    ) -> tuple[tuple[int, int], str]:
        """Select a padded shape and backend, preferring a compiled zipFFT shape."""
        factors = tuple(self.allowed_factors)

        if backend != "zipfft":
            return next_even_fft_shape(image_shape, factors), backend

        if ZIPFFT_AVAILABLE:
            snapped = snap_image_shape_to_zipfft(image_shape, template_shape)
            if snapped is not None:
                return snapped, backend

        warnings.warn(
            f"The 'zipfft' backend cannot run an image of shape {image_shape} with "
            f"a {template_shape} template (no compiled configuration fits, or "
            f"zipFFT is not installed), so cross-correlation will fall back to the "
            f"'batched' backend at the next fast FFT size. Compile additional "
            f"zipFFT shapes or set 'fast_fft_padding.target_shape' explicitly to "
            f"keep using zipFFT.",
            FFTPaddingWarning,
            stacklevel=3,
        )

        return next_even_fft_shape(image_shape, factors), "batched"

    def _warn_if_padding_large(self, plan: FFTPaddingPlan) -> None:
        """Warn when padding grows the image enough to matter for GPU memory."""
        if plan.area_growth <= LARGE_AREA_GROWTH:
            return

        warnings.warn(
            f"Automatic FFT padding grew the micrograph from {plan.original_shape} "
            f"to {plan.padded_shape} (+{100 * plan.area_growth:.1f}% pixels), which "
            f"increases peak GPU memory roughly in proportion. Reduce "
            f"'orientation_batch_size' if you hit an out-of-memory error, or set "
            f"'fast_fft_padding.enabled: false' to disable padding.",
            FFTPaddingWarning,
            stacklevel=3,
        )
