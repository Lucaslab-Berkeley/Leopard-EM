"""Pydantic model for running local peak inspection."""

from typing import Any, Literal

import torch

from leopard_em.analysis.inspect_peaks import core_inspect_template
from leopard_em.pydantic_models.managers.refine_template_manager import (
    RefineTemplateManager,
)


class PeakInspectionManager(RefineTemplateManager):
    """Run refine-template search without best-peak reduction.

    This manager reuses the refine-template backend setup, but returns full local
    score tensors for inspection rather than only the argmax result.
    """

    def get_peak_inspection_result(
        self,
        backend_kwargs: dict[str, Any],
        correlation_batch_size: int = 32,
        apply_projection_normalization: bool = True,
        output_mode: Literal["cross_correlation", "frc"] = "cross_correlation",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the inspect backend and return scores for all local hypotheses.

        Parameters
        ----------
        backend_kwargs : dict[str, Any]
            Backend inputs from :meth:`make_backend_core_function_kwargs`.
        correlation_batch_size : int, optional
            Number of orientation offsets processed per backend batch.
        apply_projection_normalization : bool, optional
            Whether to normalize each projection before scoring.
        output_mode : Literal["cross_correlation", "frc"], optional
            Score mode. ``"cross_correlation"`` returns local CC maps; ``"frc"``
            returns local FRC spectra.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            - ``"cross_correlation"``: tensor with shape
              ``(N, n_px, n_defocus, n_orient, H, W)``.
            - ``"frc"``: ``(frc_tensor, frequency_bins)`` where
              ``frc_tensor`` has shape ``(N, n_px, n_defocus, n_orient, n_freq)``
              and ``frequency_bins`` has shape ``(n_freq,)``.
        """
        return core_inspect_template(
            batch_size=correlation_batch_size,
            num_cuda_streams=self.computational_config.num_cpus,
            apply_projection_normalization=apply_projection_normalization,
            output_mode=output_mode,
            **backend_kwargs,
        )

    def run_peak_inspection(
        self,
        correlation_batch_size: int = 32,
        prefer_refined_angles: bool = True,
        apply_projection_normalization: bool = True,
        template_tensor: torch.Tensor | None = None,
        output_mode: Literal["cross_correlation", "frc"] = "cross_correlation",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run peak inspection using configured data and optional template override.

        Parameters
        ----------
        correlation_batch_size : int, optional
            Number of orientation offsets processed per backend batch.
        prefer_refined_angles : bool, optional
            If True, use refined Euler angles from the particle stack when available.
        apply_projection_normalization : bool, optional
            Whether to normalize each projection before scoring.
        template_tensor : torch.Tensor | None, optional
            Optional template volume override.
        output_mode : Literal["cross_correlation", "frc"], optional
            Score mode. ``"cross_correlation"`` returns local CC maps; ``"frc"``
            returns local FRC spectra.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Inspect output tensor (CC mode) or ``(frc_tensor, frequency_bins)``
            tuple (FRC mode).
        """
        backend_kwargs = self.make_backend_core_function_kwargs(
            prefer_refined_angles=prefer_refined_angles,
            template_tensor=template_tensor,
        )
        return self.get_peak_inspection_result(
            backend_kwargs=backend_kwargs,
            correlation_batch_size=correlation_batch_size,
            apply_projection_normalization=apply_projection_normalization,
            output_mode=output_mode,
        )

    def inspect_peaks(
        self,
        correlation_batch_size: int = 32,
        prefer_refined_angles: bool = True,
        apply_projection_normalization: bool = True,
        output_mode: Literal["cross_correlation", "frc"] = "cross_correlation",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible alias for :meth:`run_peak_inspection`.

        Parameters
        ----------
        correlation_batch_size : int, optional
            Number of orientation offsets processed per backend batch.
        prefer_refined_angles : bool, optional
            If True, use refined Euler angles from the particle stack when available.
        apply_projection_normalization : bool, optional
            Whether to normalize each projection before scoring.
        output_mode : Literal["cross_correlation", "frc"], optional
            Score mode (CC maps or FRC spectra).

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Inspect output tensor (CC mode) or ``(frc_tensor, frequency_bins)``
            tuple (FRC mode).
        """
        return self.run_peak_inspection(
            correlation_batch_size=correlation_batch_size,
            prefer_refined_angles=prefer_refined_angles,
            apply_projection_normalization=apply_projection_normalization,
            output_mode=output_mode,
        )

    def inspect_peaks_alternate_template(
        self,
        alternate_template: torch.Tensor,
        correlation_batch_size: int = 32,
        prefer_refined_angles: bool = True,
        apply_projection_normalization: bool = True,
        output_mode: Literal["cross_correlation", "frc"] = "cross_correlation",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run peak inspection using an alternate template volume.

        Parameters
        ----------
        alternate_template : torch.Tensor
            Override template volume.
        correlation_batch_size : int, optional
            Number of orientation offsets processed per backend batch.
        prefer_refined_angles : bool, optional
            If True, use refined Euler angles from the particle stack when available.
        apply_projection_normalization : bool, optional
            Whether to normalize each projection before scoring.
        output_mode : Literal["cross_correlation", "frc"], optional
            Score mode (CC maps or FRC spectra).

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Inspect output tensor (CC mode) or ``(frc_tensor, frequency_bins)``
            tuple (FRC mode).
        """
        return self.run_peak_inspection(
            correlation_batch_size=correlation_batch_size,
            prefer_refined_angles=prefer_refined_angles,
            apply_projection_normalization=apply_projection_normalization,
            template_tensor=alternate_template,
            output_mode=output_mode,
        )
