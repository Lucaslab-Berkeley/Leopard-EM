"""Pydantic model for running the optimize template program."""

from typing import Any, ClassVar

import numpy as np
import torch
from pydantic import ConfigDict
from ttsim3d.models import Simulator

from leopard_em.backend.core_refine_template import core_refine_template
from leopard_em.pydantic_models.computational_config import ComputationalConfig
from leopard_em.pydantic_models.correlation_filters import PreprocessingFilters
from leopard_em.pydantic_models.particle_stack import ParticleStack
from leopard_em.pydantic_models.pixel_size_search import PixelSizeSearchConfig
from leopard_em.pydantic_models.types import BaseModel2DTM, ExcludedTensor


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

    def make_backend_core_function_kwargs(self) -> dict[str, Any]:
        """Create the kwargs for the backend refine_template core function."""
        device_list = self.computational_config.gpu_devices
        # Remove the single device limitation
        # if len(device) > 1:
        #     raise ValueError("Only single-device execution is currently supported.")
        # device = device[0]

        # simulate template volume
        template = self.simulator.run(gpu_ids=self.computational_config.gpu_ids)

        template_shape = template.shape[-2:]

        particle_images = self.particle_stack.construct_image_stack(
            pos_reference="center",
            padding_value=0.0,
            handle_bounds="pad",
            padding_mode="constant",
        )
        # pylint: disable=E1102
        particle_images_dft = torch.fft.rfftn(particle_images, dim=(-2, -1))
        particle_images_dft[..., 0, 0] = 0.0 + 0.0j  # Zero out DC component

        # Calculate and apply the filters for the particle image stack
        filter_stack = self.particle_stack.construct_filter_stack(
            self.preprocessing_filters, output_shape=particle_images_dft.shape[-2:]
        )
        particle_images_dft *= filter_stack

        # Normalize each particle image to mean zero variance 1
        squared_image_dft = torch.abs(particle_images_dft) ** 2
        squared_sum = torch.sum(squared_image_dft, dim=(-2, -1), keepdim=True)
        particle_images_dft /= torch.sqrt(squared_sum)

        # Normalize by the effective number of pixels in the particle images
        # (sum of the bandpass filter). See comments in 'match_template_manager.py'.
        bp_config = self.preprocessing_filters.bandpass_filter
        bp_filter_image = bp_config.calculate_bandpass_filter(
            particle_images_dft.shape[-2:]
        )
        dimensionality = bp_filter_image.sum()
        particle_images_dft *= dimensionality**0.5

        # Calculate the filters applied to each template (besides CTF)
        projective_filters = self.particle_stack.construct_filter_stack(
            self.preprocessing_filters,
            output_shape=(template_shape[-2], template_shape[-1] // 2 + 1),
        )

        # Calculate the DFT of the template to take Fourier slices from
        # NOTE: There is an extra FFTshift step before the RFFT since, for some reason,
        # omitting this step will cause a 180 degree phase shift on odd (i, j, k)
        # structure factors in the Fourier domain. This just requires an extra
        # IFFTshift after converting a slice back to real-space (handled already).
        # pylint: disable=E1102
        template_dft = torch.fft.fftshift(template, dim=(-3, -2, -1))
        # pylint: disable=E1102
        template_dft = torch.fft.rfftn(template_dft, dim=(-3, -2, -1))
        # pylint: disable=E1102
        template_dft = torch.fft.fftshift(template_dft, dim=(-3, -2))  # skip rfft dim

        # The set of "best" euler angles from match template search
        # Check if refined angles exist, otherwise use the original angles
        phi = (
            self.particle_stack["refined_phi"]
            if "refined_phi" in self.particle_stack.df_columns
            else self.particle_stack["phi"]
        )
        theta = (
            self.particle_stack["refined_theta"]
            if "refined_theta" in self.particle_stack.df_columns
            else self.particle_stack["theta"]
        )
        psi = (
            self.particle_stack["refined_psi"]
            if "refined_psi" in self.particle_stack.df_columns
            else self.particle_stack["psi"]
        )

        euler_angles = torch.stack(
            (
                torch.tensor(phi),
                torch.tensor(theta),
                torch.tensor(psi),
            ),
            dim=-1,
        )

        # The relative Euler angle offsets to search over
        euler_angle_offsets = torch.zeros((1, 3))

        # The best defocus values for each particle (+ astigmatism)
        defocus_u = self.particle_stack.absolute_defocus_u
        defocus_v = self.particle_stack.absolute_defocus_v
        defocus_angle = torch.tensor(self.particle_stack["astigmatism_angle"])

        # The relative defocus values to search over
        defocus_offsets = torch.tensor([0.0])

        # The relative pixel size values to search over
        pixel_size_offsets = torch.tensor([0.0])

        # Keyword arguments for the CTF filter calculation call
        # NOTE: We currently enforce the parameters (other than the defocus values) are
        # all the same. This could be updated in the future...
        part_stk = self.particle_stack
        assert part_stk["voltage"].nunique() == 1
        assert part_stk["spherical_aberration"].nunique() == 1
        assert part_stk["amplitude_contrast_ratio"].nunique() == 1
        assert part_stk["phase_shift"].nunique() == 1
        assert part_stk["ctf_B_factor"].nunique() == 1

        ctf_kwargs = {
            "voltage": part_stk["voltage"][0].item(),
            "spherical_aberration": part_stk["spherical_aberration"][0].item(),
            "amplitude_contrast_ratio": part_stk["amplitude_contrast_ratio"][0].item(),
            "ctf_B_factor": part_stk["ctf_B_factor"][0].item(),
            "phase_shift": part_stk["phase_shift"][0].item(),
            "pixel_size": part_stk["refined_pixel_size"].mean().item(),
            "template_shape": template_shape,
            "rfft": True,
            "fftshift": False,
        }

        return {
            "particle_stack_dft": particle_images_dft,  # Remove device specification
            "template_dft": template_dft,
            "euler_angles": euler_angles,
            "euler_angle_offsets": euler_angle_offsets,
            "defocus_u": defocus_u,
            "defocus_v": defocus_v,
            "defocus_angle": defocus_angle,
            "defocus_offsets": defocus_offsets,
            "pixel_size_offsets": pixel_size_offsets,
            "ctf_kwargs": ctf_kwargs,
            "projective_filters": projective_filters,
            "device": device_list,  # Pass all devices to core_refine_template
        }

    def run_optimize_template(self, output_text_path: str) -> None:
        """Run the refine template program and saves the resultant DataFrame to csv.

        Parameters
        ----------
        output_text_path : str
            Path to save the optimized template pixel size.
        """
        if self.pixel_size_coarse_search.enabled:
            optimal_template_px = self.optimize_pixel_size()
            print(f"Optimal template px: {optimal_template_px:.3f} Å")
            # print this to the text file
            with open(output_text_path, "w", encoding="utf-8") as f:
                f.write(f"Optimal template px: {optimal_template_px:.3f} Å")

    def optimize_pixel_size(self) -> float:
        """Optimize the pixel size of the template volume.

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
        previous_snr = float("-inf")
        for px in coarse_px_values:
            snr = self.evaluate_template_px(px=px.item())
            print(f"Pixel size: {px:.3f}, SNR: {snr:.3f}")
            if snr > best_snr:
                best_snr = snr
                best_px = px.item()
            if snr > previous_snr:
                consecutive_decreases = 0
            else:
                consecutive_decreases += 1
                if consecutive_decreases >= 2:
                    print(
                        "SNR decreased for two consecutive iterations. "
                        "Stopping coarse px search."
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
                if snr > best_snr:
                    best_snr = snr
                    best_px = px.item()
                if snr > previous_snr:
                    consecutive_decreases = 0
                else:
                    consecutive_decreases += 1
                    if consecutive_decreases >= 2:
                        print(
                            "SNR decreased for two consecutive iterations. "
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
        df_refined = self.particle_stack._df.copy()
        refined_mip = result["refined_cross_correlation"]
        refined_scaled_mip = refined_mip - df_refined["correlation_mean"]
        refined_scaled_mip = refined_scaled_mip / np.sqrt(
            df_refined["correlation_variance"]
        )
        mean_snr = float(refined_scaled_mip.mean())
        print(
            f"max snr: {refined_scaled_mip.max()}, min snr: {refined_scaled_mip.min()}"
        )
        return mean_snr
