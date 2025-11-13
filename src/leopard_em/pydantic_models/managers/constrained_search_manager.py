# pylint: disable=duplicate-code
"""Pydantic model for running the constrained search program."""

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import roma
import torch
from pydantic import ConfigDict, Field

from leopard_em.analysis.zscore_metric import gaussian_noise_zscore_cutoff
from leopard_em.backend.core_refine_template import core_refine_template
from leopard_em.pydantic_models.config import (
    ComputationalConfigRefine,
    ConstrainedOrientationConfig,
    DefocusSearchConfig,
    PreprocessingFilters,
)
from leopard_em.pydantic_models.custom_types import BaseModel2DTM, ExcludedTensor
from leopard_em.pydantic_models.data_structures import ParticleStack
from leopard_em.pydantic_models.formats import CONSTRAINED_DF_COLUMN_ORDER
from leopard_em.pydantic_models.utils import (
    _setup_ctf_kwargs_from_particle_stack,
    setup_images_filters_particle_stack,
)
from leopard_em.utils.data_io import load_mrc_volume, load_template_tensor


class ConstrainedSearchManager(BaseModel2DTM):
    """Model holding parameters necessary for running the constrained search program.

    NOTE: The constrained search program should only be run on data from a single
    reference micrograph. That is, if you have data from two or more micrographs, that
    data from each micrograph needs processed separately. This restriction may be lifted
    in the future.

    Attributes
    ----------
    template_volume_path : str
        Path to the template volume MRC file.
    center_vector : list[float]
        The centre vector of the template volume.
    particle_stack_reference : ParticleStack
        Particle stack object containing particle data reference particles.
    particle_stack_constrained : ParticleStack
        Particle stack object containing particle data constrained particles.
    defocus_refinement_config : DefocusSearchConfig
        Configuration for defocus refinement.
    orientation_refinement_config : RefineOrientationConfig
        Configuration for orientation refinement.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    computational_config : ComputationalConfigRefine
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
    center_vector: list[float] = Field(default=[0.0, 0.0, 0.0])

    particle_stack_reference: ParticleStack
    particle_stack_constrained: ParticleStack
    defocus_refinement_config: DefocusSearchConfig
    orientation_refinement_config: ConstrainedOrientationConfig
    preprocessing_filters: PreprocessingFilters
    computational_config: ComputationalConfigRefine

    # Excluded tensors
    template_volume: ExcludedTensor
    zdiffs: ExcludedTensor = torch.tensor([0.0])

    def __init__(self, skip_mrc_preloads: bool = False, **data: Any):
        super().__init__(**data)

        # Load the data from the MRC files
        if not skip_mrc_preloads:
            self.template_volume = load_mrc_volume(self.template_volume_path)

    # pylint: disable=too-many-locals
    def make_backend_core_function_kwargs(
        self, prefer_refined_angles: bool = True
    ) -> dict[str, Any]:
        """Create the kwargs for the backend constrained_template core function."""
        part_stk = self.particle_stack_reference

        # Checks to make sure manager is properly configured
        # pylint: disable=protected-access
        assert part_stk._df["micrograph_path"].nunique() == 1, (
            "Constrained search can only be run on data from a single micrograph. "
            "Please ensure that the particle stack contains particles from only one "
            "micrograph."
        )

        device_list = self.computational_config.gpu_devices

        template = load_template_tensor(
            template_volume=self.template_volume,
            template_volume_path=self.template_volume_path,
        )

        euler_angles = part_stk.get_euler_angles(prefer_refined_angles)

        # The relative Euler angle offsets to search over
        euler_angle_offsets, _ = self.orientation_refinement_config.euler_angles_offsets

        # No pixel size refinement
        pixel_size_offsets = torch.tensor([0.0])

        # Extract and preprocess images and filters
        (
            particle_images_dft,
            template_dft,
            projective_filters,
        ) = setup_images_filters_particle_stack(
            part_stk, self.preprocessing_filters, template
        )

        # get z diff for each particle
        if not isinstance(self.center_vector, torch.Tensor):
            self.center_vector = torch.tensor(self.center_vector, dtype=torch.float32)
        rotation_matrices = roma.rotvec_to_rotmat(
            roma.euler_to_rotvec(convention="ZYZ", angles=euler_angles)
        ).to(torch.float32)
        rotated_vectors = rotation_matrices @ self.center_vector

        # Get z for each particle -> tensor shape [batch_size]
        new_z_diffs = rotated_vectors[:, 2]

        # The best defocus values for each particle (+ astigmatism)
        defocus_u, defocus_v = part_stk.get_absolute_defocus()
        defocus_u = defocus_u - new_z_diffs
        defocus_v = defocus_v - new_z_diffs
        # Store defocus values as instance attributes for later access
        self.zdiffs = new_z_diffs
        defocus_angle = torch.tensor(part_stk["astigmatism_angle"])

        # The relative defocus values to search over
        defocus_offsets = self.defocus_refinement_config.defocus_values

        ctf_kwargs = _setup_ctf_kwargs_from_particle_stack(
            part_stk, (template.shape[-2], template.shape[-1])
        )

        # Ger corr mean and variance
        # The position of the extracted areas needs to be from the larger particle, but
        # the mean and variance must come from the initial match template on the
        # smaller constrained particle.
        # Currently, we just set the searched file (as in paths below) to the first
        # element in the constrained particle stack.
        # NOTE: This will *not* work if the constrained particle stack contains
        # particles from multiple reference images.
        part_stk.set_column(
            "correlation_average_path",
            self.particle_stack_constrained["correlation_average_path"][0],
        )
        part_stk.set_column(
            "correlation_variance_path",
            self.particle_stack_constrained["correlation_variance_path"][0],
        )
        # Get correlation statistics
        corr_mean_stack = part_stk.construct_cropped_statistic_stack(
            stat="correlation_average",
            handle_bounds="pad",
            padding_mode="constant",
            padding_value=0.0,  # pad with zeros
        )
        corr_std_stack = part_stk.construct_cropped_statistic_stack(
            stat="correlation_variance",
            handle_bounds="pad",
            padding_mode="constant",
            padding_value=1e10,  # large to avoid out of bound pixels having inf z-score
        )
        corr_std_stack = corr_std_stack**0.5  # Convert variance to standard deviation

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
        self,
        output_dataframe_path: str,
        false_positives: float = 0.005,
        orientation_batch_size: int = 64,
    ) -> None:
        """Run the constrained search program and saves the resultant DataFrame to csv.

        Parameters
        ----------
        output_dataframe_path : str
            Path to save the constrained search results.
        false_positives : float
            The number of false positives to allow per particle.
        orientation_batch_size : int
            Number of orientations to process at once. Defaults to 64.
        """
        backend_kwargs = self.make_backend_core_function_kwargs()

        result = self.get_refine_result(backend_kwargs, orientation_batch_size)

        self.refine_result_to_dataframe(
            output_dataframe_path=output_dataframe_path,
            result=result,
            false_positives=false_positives,
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

    # pylint: disable=too-many-locals
    def refine_result_to_dataframe(
        self,
        output_dataframe_path: str,
        result: dict[str, np.ndarray],
        false_positives: float = 0.005,
    ) -> None:
        """Convert refine template result to dataframe.

        Parameters
        ----------
        output_dataframe_path : str
            Path to save the refined particle data.
        result : dict[str, np.ndarray]
            The result of the refine template program.
        false_positives : float
            The number of false positives to allow per particle.
        """
        df_refined = self.particle_stack_reference.get_dataframe_copy()

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
            + self.particle_stack_reference.get_relative_defocus().cpu().numpy()
            - self.zdiffs.cpu().numpy()
        )

        # Pixel size
        df_refined["refined_pixel_size"] = (
            result["refined_pixel_size_offset"]
            + self.particle_stack_reference.get_pixel_size().cpu().numpy()
        )

        # Cross-correlation statistics
        refined_mip = result["refined_cross_correlation"]
        refined_scaled_mip = result["refined_z_score"]
        df_refined["refined_mip"] = refined_mip
        df_refined["refined_scaled_mip"] = refined_scaled_mip

        # Reorder the columns
        df_refined = df_refined.reindex(columns=CONSTRAINED_DF_COLUMN_ORDER).fillna(0)

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
            self.particle_stack_reference.extracted_box_size[0]
            - self.particle_stack_reference.original_template_size[0]
            + 1
        ) * (
            self.particle_stack_reference.extracted_box_size[1]
            - self.particle_stack_reference.original_template_size[1]
            + 1
        )
        num_correlations = num_projections * num_px
        threshold = gaussian_noise_zscore_cutoff(
            num_correlations, float(false_positives)
        )

        # Save all parameters to CSV including false-positives
        params_df = pd.DataFrame(
            {
                "num_projections": [num_projections],
                "num_px": [num_px],
                "num_correlations": [num_correlations],
                "false_positives": [false_positives],
                "threshold": [threshold],
            }
        )
        params_df.to_csv(output_dataframe_path.replace(".csv", "_parameters.csv"))

        print(
            f"Threshold: {threshold} which gives {false_positives} "
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
        print(
            f"Saving above threshold dataframe to "
            f"{output_dataframe_path.replace('.csv', '_above_threshold.csv')}"
        )
        df_refined_above_threshold.to_csv(
            output_dataframe_path.replace(".csv", "_above_threshold.csv")
        )
