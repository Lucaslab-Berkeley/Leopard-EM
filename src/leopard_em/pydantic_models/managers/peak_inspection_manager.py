"""Pydantic model for running local peak inspection."""

from pathlib import Path
from typing import Any, Literal

import torch

from leopard_em.analysis.inspect_peaks import core_inspect_template
from leopard_em.analysis.inspect_peaks_result import save_inspection_result
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

    def run_and_save_peak_inspection(
        self,
        output_path: str | Path,
        correlation_batch_size: int = 32,
        prefer_refined_angles: bool = True,
        apply_projection_normalization: bool = True,
        template_tensor: torch.Tensor | None = None,
        output_mode: Literal["cross_correlation", "frc"] = "cross_correlation",
    ) -> Path:
        """Run peak inspection and write the score tensor to a ``.npz`` file.

        Parameters
        ----------
        output_path : str | Path
            Destination path for the ``.npz`` file (suffix appended if missing).
        correlation_batch_size : int, optional
            Number of orientation offsets processed per backend batch.
        prefer_refined_angles : bool, optional
            If True, use refined Euler angles from the particle stack when available.
        apply_projection_normalization : bool, optional
            Whether to normalize each projection before scoring.
        template_tensor : torch.Tensor | None, optional
            Optional template volume override.
        output_mode : Literal["cross_correlation", "frc"], optional
            Score mode. ``"cross_correlation"`` saves local CC maps; ``"frc"``
            saves local FRC spectra plus the frequency bins.

        Returns
        -------
        Path
            The path the result was written to (with ``.npz`` suffix).
        """
        backend_kwargs = self.make_backend_core_function_kwargs(
            prefer_refined_angles=prefer_refined_angles,
            template_tensor=template_tensor,
        )
        result = self.get_peak_inspection_result(
            backend_kwargs=backend_kwargs,
            correlation_batch_size=correlation_batch_size,
            apply_projection_normalization=apply_projection_normalization,
            output_mode=output_mode,
        )

        # Pull the particle ordering from the source dataframe when available so
        # tensor rows can be mapped back to the original particle stack.
        df = self.particle_stack._df  # pylint: disable=protected-access
        particle_index = (
            df["particle_index"].to_numpy() if "particle_index" in df.columns else None
        )

        # Per-particle base astigmatic defocus (defocus_u, defocus_v, defocus_angle) so
        # absolute defocus can be reconstructed = base_defocus + defocus_offsets.
        base_defocus = torch.stack(
            [
                backend_kwargs["defocus_u"],
                backend_kwargs["defocus_v"],
                backend_kwargs["defocus_angle"],
            ],
            dim=-1,
        )

        return save_inspection_result(
            output_path,
            result=result,
            output_mode=output_mode,
            euler_angle_offsets=backend_kwargs["euler_angle_offsets"],
            defocus_offsets=backend_kwargs["defocus_offsets"],
            pixel_size_offsets=backend_kwargs["pixel_size_offsets"],
            base_euler_angles=backend_kwargs["euler_angles"],
            base_defocus=base_defocus,
            particle_index=particle_index,
            extra_metadata={
                "prefer_refined_angles": prefer_refined_angles,
                "apply_projection_normalization": apply_projection_normalization,
                "correlation_batch_size": correlation_batch_size,
            },
        )
