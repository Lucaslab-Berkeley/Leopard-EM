"""Data structure for looking a local correlations around identified peaks."""

from typing import Any, ClassVar

import torch
import roma
import numpy as np
from pydantic import ConfigDict

from leopard_em.utils.data_io import read_mrc_to_tensor
from leopard_em.pydantic_models.custom_types import BaseModel2DTM, ExcludedTensor
from leopard_em.pydantic_models.data_structures import ParticleStack
from leopard_em.pydantic_models.utils import (
    setup_particle_backend_kwargs,
    volume_to_rfft_fourier_slice,
)
from leopard_em.pydantic_models.config import (
    ComputationalConfig,
    DefocusSearchConfig,
    PixelSizeSearchConfig,
    PreprocessingFilters,
    RefineOrientationConfig,
)
from leopard_em.analysis.inspect_peaks import inspect_correlation_peaks


class PeakInspection(BaseModel2DTM):
    """TODO: Docstring"""

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    particle_stack: ParticleStack
    pixel_size_offsets: PixelSizeSearchConfig
    defocus_offsets: DefocusSearchConfig
    orientation_offsets: RefineOrientationConfig
    preprocessing_filters: PreprocessingFilters

    def make_inspect_peaks_kwargs(self, prefer_refined_angles: bool) -> dict:
        """TODO: Docstring"""
        
        template = self.particle_stack["template_path"][0]
        template = read_mrc_to_tensor(template)

        # The set of "best" euler angles from match template search
        # Check if refined angles exist, otherwise use the original angles
        euler_angles = self.particle_stack.get_euler_angles(prefer_refined_angles)

        # The relative Euler angle offsets to search over
        euler_angle_offsets = self.orientation_offsets.euler_angles_offsets

        # The relative defocus values to search over
        defocus_offsets = self.defocus_offsets.defocus_values

        # The relative pixel size values to search over
        pixel_size_offsets = self.pixel_size_offsets.pixel_size_values

        # Use the common utility function to set up the backend kwargs
        # pylint: disable=duplicate-code
        kwargs = setup_particle_backend_kwargs(
            particle_stack=self.particle_stack,
            template=template,
            preprocessing_filters=self.preprocessing_filters,
            euler_angles=euler_angles,
            euler_angle_offsets=euler_angle_offsets,
            defocus_offsets=defocus_offsets,
            pixel_size_offsets=pixel_size_offsets,
            device_list=[None],  # Ignored parameter here
        )
        _ = kwargs.pop("device")  # Remove device key, not needed here
        _ = kwargs.pop("corr_mean")
        _ = kwargs.pop("corr_std")

        # Convert Euler angles into rotation matrices
        euler_angles = kwargs.pop("euler_angles")
        euler_angle_offsets = kwargs.pop("euler_angle_offsets")

        rotation_matrices = roma.euler_to_rotmat("ZYZ", euler_angles, degrees=True)
        rotation_offset_matrices = roma.euler_to_rotmat(
            "ZYZ", euler_angle_offsets, degrees=True
        )
        kwargs["rotation_matrices"] = rotation_matrices.float()
        kwargs["rotation_matrices_offset"] = rotation_offset_matrices.float()
        
        kwargs["pixel_size"] = [kwargs["ctf_kwargs"]["pixel_size"]] * self.particle_stack.num_particles

        return kwargs

    def _inspect_peaks(
        self,
        device: torch.device,
        pixel_size_batch_size: int,
        defocus_batch_size: int,
        orientation_batch_size: int,
        apply_projection_normalization: bool,
        return_valid_region: bool,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Inspect peaks function. TODO: Docstring"""
        return inspect_correlation_peaks(
            device=device,
            pixel_size_batch_size=pixel_size_batch_size,
            defocus_batch_size=defocus_batch_size,
            orientation_batch_size=orientation_batch_size,
            apply_projection_normalization=apply_projection_normalization,
            return_valid_region=return_valid_region,
            **kwargs,
        )

    def inspect_peaks(
        self,
        device: torch.device,
        pixel_size_batch_size: int = -1,
        defocus_batch_size: int = -1,
        orientation_batch_size: int = -1,
        prefer_refined_angles: bool = True,
        apply_projection_normalization: bool = True,
        return_valid_region: bool = True,
    ) -> torch.Tensor:
        """TODO: Docstring"""
        kwargs = self.make_inspect_peaks_kwargs(prefer_refined_angles)

        return self._inspect_peaks(
            device,
            pixel_size_batch_size,
            defocus_batch_size,
            orientation_batch_size,
            apply_projection_normalization,
            return_valid_region,
            **kwargs,
        )

    def inspect_peaks_alternate_template(
        self,
        alternate_template: torch.Tensor,
        device: torch.device,
        pixel_size_batch_size: int = -1,
        defocus_batch_size: int = -1,
        orientation_batch_size: int = -1,
        prefer_refined_angles: bool = True,
        apply_projection_normalization: bool = True,
        return_valid_region: bool = True,
    ) -> torch.Tensor:
        """Inspect peaks using a different template than in the particle stack."""
        template_dft = volume_to_rfft_fourier_slice(alternate_template)

        kwargs = self.make_inspect_peaks_kwargs(prefer_refined_angles)
        kwargs["template_dft"] = template_dft

        return self._inspect_peaks(
            device,
            pixel_size_batch_size,
            defocus_batch_size,
            orientation_batch_size,
            apply_projection_normalization,
            return_valid_region,
            **kwargs,
        )

    # def inspect_peaks_across_filters(
    #     self,
    #     alternate_fourier_filters: torch.Tensor,  # (l, H, W)
    #     **kwargs: dict[str, Any],
    # ) -> torch.Tensor:
    #     """Helper function to look at correlations across different Fourier filters."""
    #     _ = kwargs.pop("return_valid_region", None)
    #     kwargs["return_valid_region"] = True
        
    #     cc_stack = self.inspect_peaks(**kwargs)
    #     cc_stack_filtered = torch.empty(
    #         (alternate_fourier_filters.shape[0],) + cc_stack.shape,
    #         dtype=cc_stack.dtype,
    #         device=cc_stack.device,
    #     )
