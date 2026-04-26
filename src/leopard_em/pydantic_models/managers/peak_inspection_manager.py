"""Pydantic model for running local peak inspection."""

from typing import Any

import torch

from leopard_em.analysis.inspect_peaks import core_inspect_template
from leopard_em.pydantic_models.managers.refine_template_manager import (
    RefineTemplateManager,
)


class PeakInspectionManager(RefineTemplateManager):
    """Run refine-template correlations without reducing to the best peak."""

    def get_peak_inspection_result(
        self,
        backend_kwargs: dict[str, Any],
        correlation_batch_size: int = 32,
        apply_projection_normalization: bool = True,
    ) -> torch.Tensor:
        """Run peak inspection and return all local correlations."""
        return core_inspect_template(
            batch_size=correlation_batch_size,
            num_cuda_streams=self.computational_config.num_cpus,
            apply_projection_normalization=apply_projection_normalization,
            **backend_kwargs,
        )

    def run_peak_inspection(
        self,
        correlation_batch_size: int = 32,
        prefer_refined_angles: bool = True,
        apply_projection_normalization: bool = True,
        template_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run peak inspection using the configured or overridden template volume."""
        backend_kwargs = self.make_backend_core_function_kwargs(
            prefer_refined_angles=prefer_refined_angles,
            template_tensor=template_tensor,
        )
        return self.get_peak_inspection_result(
            backend_kwargs=backend_kwargs,
            correlation_batch_size=correlation_batch_size,
            apply_projection_normalization=apply_projection_normalization,
        )

    def inspect_peaks(
        self,
        correlation_batch_size: int = 32,
        prefer_refined_angles: bool = True,
        apply_projection_normalization: bool = True,
    ) -> torch.Tensor:
        """Backward-compatible alias for :meth:`run_peak_inspection`."""
        return self.run_peak_inspection(
            correlation_batch_size=correlation_batch_size,
            prefer_refined_angles=prefer_refined_angles,
            apply_projection_normalization=apply_projection_normalization,
        )

    def inspect_peaks_alternate_template(
        self,
        alternate_template: torch.Tensor,
        correlation_batch_size: int = 32,
        prefer_refined_angles: bool = True,
        apply_projection_normalization: bool = True,
    ) -> torch.Tensor:
        """Run peak inspection using a different template volume."""
        return self.run_peak_inspection(
            correlation_batch_size=correlation_batch_size,
            prefer_refined_angles=prefer_refined_angles,
            apply_projection_normalization=apply_projection_normalization,
            template_tensor=alternate_template,
        )
