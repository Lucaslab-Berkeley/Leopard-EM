"""Pydantic model for running the refine template program."""

from typing import Any, ClassVar

import numpy as np
import roma
import torch
from pydantic import ConfigDict, Field

from leopard_em.analysis.pick_match_template_peaks import gaussian_noise_zscore_cutoff
from leopard_em.backend.core_refine_template import core_refine_template
from leopard_em.pydantic_models.computational_config import ComputationalConfig
from leopard_em.pydantic_models.correlation_filters import PreprocessingFilters
from leopard_em.pydantic_models.defocus_search import DefocusSearchConfig
from leopard_em.pydantic_models.formats import CONSTRAINED_DF_COLUMN_ORDER
from leopard_em.pydantic_models.orientation_search import ConstrainedOrientationConfig
from leopard_em.pydantic_models.particle_stack import ParticleStack
from leopard_em.pydantic_models.types import BaseModel2DTM, ExcludedTensor
from leopard_em.utils.data_io import load_mrc_volume


class ConstrainedSearchManager(BaseModel2DTM):
    """Model holding parameters necessary for running the constrained search program.

    Attributes
    ----------
    template_volume_path : str
        Path to the template volume MRC file.
    centre_vector : list[float]
        The centre vector of the template volume.
    particle_stack_large : ParticleStack
        Particle stack object containing particle data large particles.
    particle_stack_small : ParticleStack
        Particle stack object containing particle data small particles.
    defocus_refinement_config : DefocusSearchConfig
        Configuration for defocus refinement.
    orientation_refinement_config : RefineOrientationConfig
        Configuration for orientation refinement.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    computational_config : ComputationalConfig
        What computational resources to allocate for the program.
    template_volume : ExcludedTensor
        The template volume tensor (excluded from serialization).
    false_positives : float
        The number of false positives to allow per particle.

    Methods
    -------
    TODO serialization/import methods
    __init__(self, skip_mrc_preloads: bool = False, **data: Any)
        Initialize the constrained search manager.
    make_backend_core_function_kwargs(self) -> dict[str, Any]
        Create the kwargs for the backend refine_template core function.
    run_constrained_search(self, orientation_batch_size: int = 64) -> None
        Run the constrained search program.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    template_volume_path: str  # In df per-particle, but ensure only one reference
    centre_vector: list[float] = Field(default=[0.0, 0.0, 0.0])

    particle_stack_large: ParticleStack
    particle_stack_small: ParticleStack
    defocus_refinement_config: DefocusSearchConfig
    orientation_refinement_config: ConstrainedOrientationConfig
    preprocessing_filters: PreprocessingFilters
    computational_config: ComputationalConfig

    false_positives: float = 0.001  # Default to 1 per 1000 particles

    # Excluded tensors
    template_volume: ExcludedTensor
    zdiffs: ExcludedTensor = torch.tensor([0.0])

    def __init__(self, skip_mrc_preloads: bool = False, **data: Any):
        super().__init__(**data)

        # Load the data from the MRC files
        if not skip_mrc_preloads:
            self.template_volume = load_mrc_volume(self.template_volume_path)

    def make_backend_core_function_kwargs(self) -> dict[str, Any]:
        """Create the kwargs for the backend refine_template core function."""
        device_list = self.computational_config.gpu_devices

        if self.template_volume is None:
            self.template_volume = load_mrc_volume(self.template_volume_path)
        if not isinstance(self.template_volume, torch.Tensor):
            template = torch.from_numpy(self.template_volume)
        else:
            template = self.template_volume

        template_shape = template.shape[-2:]

        particle_images = self.particle_stack_large.construct_image_stack(
            pos_reference="center",
            padding_value=0.0,
            handle_bounds="pad",
            padding_mode="constant",
        )
        particle_images_dft = torch.fft.rfftn(particle_images, dim=(-2, -1))
        particle_images_dft[..., 0, 0] = 0.0 + 0.0j  # Zero out DC component

        # Calculate and apply the filters for the particle image stack
        filter_stack = self.particle_stack_large.construct_filter_stack(
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
        projective_filters = self.particle_stack_large.construct_filter_stack(
            self.preprocessing_filters,
            output_shape=(template_shape[-2], template_shape[-1] // 2 + 1),
        )

        # Calculate the DFT of the template to take Fourier slices from
        # NOTE: There is an extra FFTshift step before the RFFT since, for some reason,
        # omitting this step will cause a 180 degree phase shift on odd (i, j, k)
        # structure factors in the Fourier domain. This just requires an extra
        # IFFTshift after converting a slice back to real-space (handled already).
        template_dft = torch.fft.fftshift(template, dim=(-3, -2, -1))
        template_dft = torch.fft.rfftn(template_dft, dim=(-3, -2, -1))
        template_dft = torch.fft.fftshift(template_dft, dim=(-3, -2))  # skip rfft dim

        # The set of "best" euler angles from match template search
        # Check if refined angles exist, otherwise use the original angles
        phi = (
            self.particle_stack_large["refined_phi"]
            if "refined_phi" in self.particle_stack_large._df.columns
            else self.particle_stack_large["phi"]
        )
        theta = (
            self.particle_stack_large["refined_theta"]
            if "refined_theta" in self.particle_stack_large._df.columns
            else self.particle_stack_large["theta"]
        )
        psi = (
            self.particle_stack_large["refined_psi"]
            if "refined_psi" in self.particle_stack_large._df.columns
            else self.particle_stack_large["psi"]
        )

        euler_angles = torch.stack(
            (
                torch.tensor(phi),
                torch.tensor(theta),
                torch.tensor(psi),
            ),
            dim=-1,
        )

        # get z diff for each particle
        if not isinstance(self.centre_vector, torch.Tensor):
            self.centre_vector = torch.tensor(self.centre_vector, dtype=torch.float32)
        rotation_matrices = roma.rotvec_to_rotmat(
            roma.euler_to_rotvec(convention="ZYZ", angles=euler_angles)
        ).to(torch.float32)
        rotated_vectors = rotation_matrices @ self.centre_vector

        # Get z-component for each particle individually
        new_z_diffs = rotated_vectors[
            :, 2
        ]  # This is now a tensor with shape [batch_size]

        # The relative Euler angle offsets to search over
        euler_angle_offsets, _ = self.orientation_refinement_config.euler_angles_offsets
        # euler_angle_offsets = torch.zeros((1, 3))

        # The best defocus values for each particle (+ astigmatism)
        defocus_u = self.particle_stack_large.absolute_defocus_u
        defocus_u = defocus_u - new_z_diffs
        defocus_v = self.particle_stack_large.absolute_defocus_v
        defocus_v = defocus_v - new_z_diffs
        # Store defocus values as instance attributes for later access
        self.zdiffs = new_z_diffs
        defocus_angle = torch.tensor(self.particle_stack_large["astigmatism_angle"])

        # The relative defocus values to search over
        defocus_offsets = self.defocus_refinement_config.defocus_values

        # No pixel size refinement
        pixel_size_offsets = torch.tensor([0.0])

        # Keyword arguments for the CTF filter calculation call
        # NOTE: We currently enforce the parameters (other than the defocus values) are
        # all the same. This could be updated in the future...
        part_stk = self.particle_stack_large
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

        # Ger corr mean and variance
        # I want positions of large but vals from small
        part_stk._df.loc[:, "correlation_average_path"] = self.particle_stack_small[
            "correlation_average_path"
        ][0]
        part_stk._df.loc[:, "correlation_variance_path"] = self.particle_stack_small[
            "correlation_variance_path"
        ][0]
        corr_mean_stack = part_stk.construct_cropped_statistic_stack(
            "correlation_average"
        )
        corr_std_stack = (
            part_stk.construct_cropped_statistic_stack("correlation_variance") ** 0.5
        )  # var to std

        return {
            "particle_stack_dft": particle_images_dft,
            "template_dft": template_dft,
            "euler_angles": euler_angles,
            "euler_angle_offsets": euler_angle_offsets,
            "defocus_u": defocus_u,
            "defocus_v": defocus_v,
            "defocus_angle": defocus_angle,
            "defocus_offsets": defocus_offsets,
            "pixel_size_offsets": pixel_size_offsets,
            "corr_mean": corr_mean_stack,
            "corr_std": corr_std_stack,
            "ctf_kwargs": ctf_kwargs,
            "projective_filters": projective_filters,
            "device": device_list,  # Pass all devices to core_refine_template
        }

    def run_constrained_search(
        self, output_dataframe_path: str, orientation_batch_size: int = 64
    ) -> None:
        """Run the constrained search program and saves the resultant DataFrame to csv.

        Parameters
        ----------
        output_dataframe_path : str
            Path to save the constrained search results.
        orientation_batch_size : int
            Number of orientations to process at once. Defaults to 64.
        """
        backend_kwargs = self.make_backend_core_function_kwargs()

        result = self.get_refine_result(backend_kwargs, orientation_batch_size)

        self.refine_result_to_dataframe(
            output_dataframe_path=output_dataframe_path, result=result
        )

    def get_refine_result(
        self, backend_kwargs: dict, orientation_batch_size: int = 64
    ) -> dict[str, np.ndarray]:
        """Get refine template result.

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
        # Adjust batch size if orientation search is disabled
        if not self.orientation_refinement_config.enabled:
            orientation_batch_size = 1
        elif (
            self.orientation_refinement_config.euler_angles_offsets[0].shape[0]
            < orientation_batch_size
        ):
            orientation_batch_size = (
                self.orientation_refinement_config.euler_angles_offsets[0].shape[0]
            )

        result: dict[str, np.ndarray] = {}
        result = core_refine_template(
            batch_size=orientation_batch_size, **backend_kwargs
        )
        result = {k: v.cpu().numpy() for k, v in result.items()}
        return result

    def refine_result_to_dataframe(
        self, output_dataframe_path: str, result: dict[str, np.ndarray]
    ) -> None:
        """Convert refine template result to dataframe.

        Parameters
        ----------
        output_dataframe_path : str
            Path to save the refined particle data.
        result : dict[str, np.ndarray]
            The result of the refine template program.
        """
        df_refined = self.particle_stack_large._df.copy()

        # x and y positions
        pos_offset_y = result["refined_pos_y"]
        pos_offset_x = result["refined_pos_x"]
        pos_offset_y_ang = pos_offset_y * df_refined["pixel_size"]
        pos_offset_x_ang = pos_offset_x * df_refined["pixel_size"]

        df_refined["refined_pos_y"] = pos_offset_y + df_refined["pos_y"]
        df_refined["refined_pos_x"] = pos_offset_x + df_refined["pos_x"]
        df_refined["refined_pos_y_img"] = pos_offset_y + df_refined["pos_y_img"]
        df_refined["refined_pos_x_img"] = pos_offset_x + df_refined["pos_x_img"]
        df_refined["refined_pos_y_img_angstrom"] = (
            pos_offset_y_ang + df_refined["pos_y_img_angstrom"]
        )
        df_refined["refined_pos_x_img_angstrom"] = (
            pos_offset_x_ang + df_refined["pos_x_img_angstrom"]
        )

        # Euler angles
        angle_idx = result["angle_idx"]
        df_refined["refined_psi"] = result["refined_euler_angles"][:, 2]
        df_refined["refined_theta"] = result["refined_euler_angles"][:, 1]
        df_refined["refined_phi"] = result["refined_euler_angles"][:, 0]

        _, euler_angle_offsets = self.orientation_refinement_config.euler_angles_offsets
        euler_angle_offsets_np = euler_angle_offsets.cpu().numpy()
        # Store the matched original offsets in the dataframe
        df_refined["original_offset_phi"] = euler_angle_offsets_np[angle_idx, 0]
        df_refined["original_offset_theta"] = euler_angle_offsets_np[angle_idx, 1]
        df_refined["original_offset_psi"] = euler_angle_offsets_np[angle_idx, 2]

        # Defocus
        df_refined["refined_relative_defocus"] = (
            result["refined_defocus_offset"]
            + df_refined["refined_relative_defocus"]
            - self.zdiffs.cpu().numpy()
        )

        # Pixel size
        df_refined["refined_pixel_size"] = (
            result["refined_pixel_size_offset"] + df_refined["pixel_size"]
        )

        # Cross-correlation statistics
        refined_mip = result["refined_cross_correlation"]
        refined_scaled_mip = result["refined_z_score"]
        df_refined["refined_mip"] = refined_mip
        df_refined["refined_scaled_mip"] = refined_scaled_mip

        # Reorder the columns
        df_refined = df_refined.reindex(columns=CONSTRAINED_DF_COLUMN_ORDER)

        # Save the refined DataFrame to disk
        df_refined.to_csv(output_dataframe_path)

        # Save a second dataframe
        # I also want the original user input offsets back somewhere
        # This one will have only those above threshold
        num_projections = (
            self.defocus_refinement_config.defocus_values.shape[0]
            * self.orientation_refinement_config.euler_angles_offsets[0].shape[0]
        )
        num_px = (
            self.particle_stack_large.extracted_box_size[0]
            * self.particle_stack_large.extracted_box_size[1]
        )
        num_correlations = num_projections * num_px
        threshold = gaussian_noise_zscore_cutoff(num_correlations, self.false_positives)
        print(
            f"Threshold: {threshold} which gives {self.false_positives} "
            "false positives per particle"
        )
        df_refined_above_threshold = df_refined[
            df_refined["refined_scaled_mip"] > threshold
        ]
        # Also remove if refined_scaled_mip is inf or nan
        df_refined_above_threshold = df_refined_above_threshold[
            df_refined_above_threshold["refined_scaled_mip"] != np.inf
        ]
        df_refined_above_threshold = df_refined_above_threshold[
            df_refined_above_threshold["refined_scaled_mip"] != np.nan
        ]
        # Save the above threshold dataframe
        df_refined_above_threshold.to_csv(
            output_dataframe_path.replace(".csv", "_above_threshold.csv")
        )

    # @classmethod
    # def from_dataframe(cls, dataframe: pd.DataFrame) -> "RefineTemplateManager":
    #     """Tabular data (from DataFrame) and create a RefineTemplateManager object."""
    #     raise NotImplementedError("Method not implemented yet.")

    # @classmethod
    # def from_match_template_manager(
    #     cls, mt_manager: MatchTemplateManager
    # ) -> "RefineTemplateManager":
    #     """Creates a RefineTemplateManager object from MatchTemplateManager object."""
    #     raise NotImplementedError("Method not implemented yet.")

    # def particles_to_dataframe(self) -> pd.DataFrame:
    #     """Export refined particles as dataframe."""
    #     raise NotImplementedError("Method not implemented yet.")

    # def results_to_dataframe(self) -> pd.DataFrame:
    #     """Export full results as dataframe."""
    #     raise NotImplementedError("Method not implemented yet.")

    # def particle_stack_to_dataframe(self) -> pd.DataFrame:
    #     """Export particle stack information as dataframe."""
    #     raise NotImplementedError("Method not implemented yet.")

    # def save_particle_stack(self, save_dir: str) -> None:
    #     """Save the particle stack to disk."""
    #     raise NotImplementedError("Method not implemented yet.")

    # @classmethod
    # def from_results_csv(cls, results_csv_path: str) -> "RefineTemplateManager":
    #     """Take tabular data (from csv) and create a RefineTemplateManager object.

    #     Parameters
    #     ----------
    #     results_csv_path : str
    #         Path to the CSV file containing the per-particle data.

    #     Returns
    #     -------
    #     RefineTemplateManager
    #     """
    #     df = pd.read_csv(results_csv_path)

    #     return cls.from_dataframe(df)
