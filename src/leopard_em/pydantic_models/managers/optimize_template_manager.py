"""Pydantic model for running the optimize template program."""

import os
from typing import Any, ClassVar

import numpy as np
import torch
from pydantic import ConfigDict
from ttsim3d.models import Simulator

from leopard_em.backend.core_refine_template import core_refine_template
from leopard_em.pydantic_models.config import (
    ComputationalConfig,
    PixelSizeSearchConfig,
    PreprocessingFilters,
)
from leopard_em.pydantic_models.custom_types import BaseModel2DTM, ExcludedTensor
from leopard_em.pydantic_models.data_structures import ParticleStack
from leopard_em.pydantic_models.utils import setup_particle_backend_kwargs


class OptimizeTemplateManager(BaseModel2DTM):
    """Model holding parameters necessary for running the optimize template program.

    Attributes
    ----------
    particle_stack : ParticleStack
        Particle stack object containing particle data.
    pixel_size_coarse_search : PixelSizeSearchConfig
        Configuration for pixel size coarse search.
    pixel_size_fine_search : PixelSizeSearchConfig
        Configuration for pixel size fine search.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    computational_config : ComputationalConfig
        What computational resources to allocate for the program.
    simulator : Simulator
        The simulator object.

    Methods
    -------
    TODO serialization/import methods
    __init__(self, skip_mrc_preloads: bool = False, **data: Any)
        Initialize the optimize template manager.
    make_backend_core_function_kwargs(self) -> dict[str, Any]
        Create the kwargs for the backend optimize_template core function.
    run_optimize_template(self, output_text_path: str) -> None
        Run the optimize template program.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    particle_stack: ParticleStack
    pixel_size_coarse_search: PixelSizeSearchConfig
    pixel_size_fine_search: PixelSizeSearchConfig
    preprocessing_filters: PreprocessingFilters
    computational_config: ComputationalConfig
    simulator: Simulator

    # Excluded tensors
    template_volume: ExcludedTensor

    def make_backend_core_function_kwargs(
        self, prefer_refined_angles: bool = True
    ) -> dict[str, Any]:
        """Create the kwargs for the backend refine_template core function.

        Parameters
        ----------
        prefer_refined_angles : bool
            Whether to use refined angles or not. Defaults to True.
        """
        # simulate template volume
        template = self.simulator.run(device=self.computational_config.gpu_ids)

        # The set of "best" euler angles from match template search
        # Check if refined angles exist, otherwise use the original angles
        euler_angles = self.particle_stack.get_euler_angles(prefer_refined_angles)

        # The relative Euler angle offsets to search over (none for optimization)
        euler_angle_offsets = torch.zeros((1, 3))

        # The relative defocus values to search over (none for optimization)
        defocus_offsets = torch.tensor([0.0])

        # The relative pixel size values to search over (none for optimization)
        pixel_size_offsets = torch.tensor([0.0])

        # Use the common utility function to set up the backend kwargs
        # pylint: disable=duplicate-code
        return setup_particle_backend_kwargs(
            particle_stack=self.particle_stack,
            template=template,
            preprocessing_filters=self.preprocessing_filters,
            euler_angles=euler_angles,
            euler_angle_offsets=euler_angle_offsets,
            defocus_offsets=defocus_offsets,
            pixel_size_offsets=pixel_size_offsets,
            device_list=self.computational_config.gpu_devices,
        )

    def run_optimize_template(self, output_text_path: str) -> None:
        """Run the refine template program and saves the resultant DataFrame to csv.

        Parameters
        ----------
        output_text_path : str
            Path to save the optimized template pixel size.
        """
        if self.pixel_size_coarse_search.enabled:
            # Create a file for logging all iterations
            all_results_path = self._get_all_results_path(output_text_path)
            # Create the file and write header
            with open(all_results_path, "w", encoding="utf-8") as f:
                f.write("Pixel Size (Å),SNR\n")

            optimal_template_px = self.optimize_pixel_size(all_results_path)
            print(f"Optimal template px: {optimal_template_px:.3f} Å")
            # print this to the text file
            with open(output_text_path, "w", encoding="utf-8") as f:
                f.write(f"Optimal template px: {optimal_template_px:.3f} Å")

    def optimize_pixel_size(self, all_results_path: str) -> float:
        """Optimize the pixel size of the template volume.

        Parameters
        ----------
        all_results_path : str
            Path to the file for logging all iterations

        Returns
        -------
        float
            The optimal pixel size.
        """
        initial_template_px = self.simulator.pixel_spacing
        print(f"Initial template px: {initial_template_px:.3f} Å")

        best_snr = float("-inf")
        best_px = float(initial_template_px)

        print("Starting coarse search...")

        pixel_size_offsets_coarse = self.pixel_size_coarse_search.pixel_size_values
        coarse_px_values = pixel_size_offsets_coarse + initial_template_px

        consecutive_decreases = 0
        consecutive_threshold = 2
        previous_snr = float("-inf")
        for px in coarse_px_values:
            snr = self.evaluate_template_px(px=px.item())
            print(f"Pixel size: {px:.3f}, SNR: {snr:.3f}")

            # Log to file
            with open(all_results_path, "a", encoding="utf-8") as f:
                f.write(f"{px:.3f},{snr:.3f}\n")

            if snr > best_snr:
                best_snr = snr
                best_px = px.item()
            if snr > previous_snr:
                consecutive_decreases = 0
            else:
                consecutive_decreases += 1
                if consecutive_decreases >= consecutive_threshold:
                    print(
                        f"SNR decreased for {consecutive_threshold} iterations. "
                        f"Stopping coarse px search."
                    )
                    break
            previous_snr = snr

        if self.pixel_size_fine_search.enabled:
            pixel_size_offsets_fine = self.pixel_size_fine_search.pixel_size_values
            fine_px_values = pixel_size_offsets_fine + best_px

            consecutive_decreases = 0
            previous_snr = float("-inf")
            for px in fine_px_values:
                snr = self.evaluate_template_px(px=px.item())
                print(f"Pixel size: {px:.3f}, SNR: {snr:.3f}")

                # Log to file
                with open(all_results_path, "a", encoding="utf-8") as f:
                    f.write(f"{px:.3f},{snr:.3f}\n")

                if snr > best_snr:
                    best_snr = snr
                    best_px = px.item()
                if snr > previous_snr:
                    consecutive_decreases = 0
                else:
                    consecutive_decreases += 1
                    if consecutive_decreases >= consecutive_threshold:
                        print(
                            f"SNR decreased for {consecutive_threshold} iterations. "
                            "Stopping fine px search."
                        )
                        break
                previous_snr = snr

        return best_px

    def evaluate_template_px(self, px: float) -> float:
        """Evaluate the template pixel size.

        Parameters
        ----------
        px : float
            The pixel size to evaluate.

        Returns
        -------
        float
            The mean SNR of the template.
        """
        self.simulator.pixel_spacing = px
        backend_kwargs = self.make_backend_core_function_kwargs()
        result = self.get_correlation_result(backend_kwargs, 1)
        mean_snr = self.results_to_snr(result)
        return mean_snr

    def get_correlation_result(
        self, backend_kwargs: dict, orientation_batch_size: int = 64
    ) -> dict[str, np.ndarray]:
        """Get correlation result.

        Parameters
        ----------
        backend_kwargs : dict
            Keyword arguments for the backend processing
        orientation_batch_size : int
            Number of orientations to process at once. Defaults to 64.

        Returns
        -------
        dict[str, np.ndarray]
            The result of the refine template program.
        """
        # pylint: disable=duplicate-code
        result: dict[str, np.ndarray] = {}
        result = core_refine_template(
            batch_size=orientation_batch_size, **backend_kwargs
        )
        result = {k: v.cpu().numpy() for k, v in result.items()}

        return result

    def results_to_snr(self, result: dict[str, np.ndarray]) -> float:
        """Convert optimize template result to mean SNR.

        Parameters
        ----------
        result : dict[str, np.ndarray]
            The result of the optimize template program.

        Returns
        -------
        float
            The mean SNR of the template.
        """
        # Filter out any infinite or NaN values
        # NOTE: There should not be NaNs or infs, will follow up later
        refined_scaled_mip = result["refined_z_score"]
        refined_scaled_mip = refined_scaled_mip[np.isfinite(refined_scaled_mip)]

        # If more than n values, keep only the top n highest SNRs
        best_n = 6
        if len(refined_scaled_mip) > best_n:
            refined_scaled_mip = np.sort(refined_scaled_mip)[-best_n:]

        # Printing out the results to console
        print(
            f"max snr: {refined_scaled_mip.max()}, min snr: {refined_scaled_mip.min()}"
        )

        mean_snr = float(refined_scaled_mip.mean())

        return mean_snr

    def _get_all_results_path(self, output_text_path: str) -> str:
        """Generate the results file path from the output text path.

        Parameters
        ----------
        output_text_path : str
            Path to the output text file

        Returns
        -------
        str
            Path to the file with _all.txt extension
        """
        base, _ = os.path.splitext(output_text_path)
        return f"{base}_all.csv"
