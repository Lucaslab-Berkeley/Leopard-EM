"""Pydantic model for running the refine template program."""
# pylint: disable=duplicate-code
from typing import Any, ClassVar, Self

import pandas as pd
import torch
from pydantic import ConfigDict, model_validator

from leopard_em.backend.core_differentiable_refine import core_differentiable_refine
from leopard_em.pydantic_models.config import (
    ComputationalConfigRefine,
    DefocusSearchConfig,
    MovieConfig,
    PixelSizeSearchConfig,
    PreprocessingFilters,
    RefineOrientationConfig,
)
from leopard_em.pydantic_models.custom_types import BaseModel2DTM, ExcludedTensor
from leopard_em.pydantic_models.data_structures import ParticleStack
from leopard_em.pydantic_models.formats import REFINED_DF_COLUMN_ORDER
from leopard_em.pydantic_models.utils import setup_particle_backend_kwargs
from leopard_em.utils.data_io import load_mrc_volume, load_template_tensor


class DifferentiableRefineManager(BaseModel2DTM):
    """Model holding parameters necessary for running the differentiable refine program.

    Attributes
    ----------
    template_volume_path : str
        Path to the template volume MRC file.
    particle_stack : ParticleStack
        Particle stack object containing particle data.
    defocus_refinement_config : DefocusSearchConfig
        Configuration for defocus refinement.
    pixel_size_refinement_config : PixelSizeSearchConfig
        Configuration for pixel size refinement.
    orientation_refinement_config : RefineOrientationConfig
        Configuration for orientation refinement.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    computational_config : ComputationalConfigRefine
        What computational resources to allocate for the program.
    apply_global_filtering : bool
        If True, apply filtering to the full micrograph before particle extraction.
        If False, filter are calculated and applied to the cropped particle images.
        Default is True.
    template_volume : ExcludedTensor
        The template volume tensor (excluded from serialization).
    movie_config : MovieConfig
        Configuration for the movie.

    Methods
    -------
    TODO serialization/import methods
    __init__(self, skip_mrc_preloads: bool = False, **data: Any)
        Initialize the differentiable refine manager.
    make_backend_core_function_kwargs(self) -> dict[str, Any]
        Create the kwargs for the backend refine_template core function.
    run_refine_template(self, correlation_batch_size: int = 32) -> None
        Run the refine template program.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    template_volume_path: str  # In df per-particle, but ensure only one reference
    particle_stack: ParticleStack
    defocus_refinement_config: DefocusSearchConfig
    pixel_size_refinement_config: PixelSizeSearchConfig
    orientation_refinement_config: RefineOrientationConfig
    preprocessing_filters: PreprocessingFilters
    computational_config: ComputationalConfigRefine
    movie_config: MovieConfig
    apply_global_filtering: bool = True

    # Excluded tensors
    template_volume: ExcludedTensor

    def __init__(self, skip_mrc_preloads: bool = False, **data: Any):
        super().__init__(**data)

        # Load the data from the MRC files
        if not skip_mrc_preloads:
            self.template_volume = load_mrc_volume(self.template_volume_path)

    @model_validator(mode="after")  # type: ignore
    def validate_global_filtering_disabled(self) -> Self:
        """Validate that global filtering is not enabled for particle stack refinement.

        Global filtering cannot be applied with particle stack refinement, so we
        raise an error if it is enabled to prevent implicit configuration
        modifications.

        Returns
        -------
        Self
            The validated model instance.

        Raises
        ------
        ValueError
            If global filtering is enabled.
        """
        if self.apply_global_filtering:
            raise ValueError(
                "Global filtering cannot be applied with particle stack refinement. "
                "Please set `apply_global_filtering=False`."
            )
        return self

    def make_backend_core_function_kwargs(
        self,
        image_stack: torch.Tensor,
        mean_stack: torch.Tensor,
        std_stack: torch.Tensor,
        particle_indices: list[pd.Index],
        template_tensor: torch.Tensor | None = None,
        prefer_refined_angles: bool = True,
        images_are_particles: bool = False,
    ) -> dict[str, Any]:
        """Create the kwargs for the backend refine_template core function.

        Parameters
        ----------
        prefer_refined_angles : bool
            Whether to use the refined angles from the particle stack. Defaults to
            False.
        images_are_particles: bool
            Whether the images are particles or not. Defaults to False.
        image_stack : torch.Tensor
            Pre-loaded image stack tensor.
        mean_stack : torch.Tensor
            Pre-loaded mean stack tensor.
        std_stack : torch.Tensor
            Pre-loaded std stack tensor.
        template_tensor : torch.Tensor | None
            Pre-loaded template tensor. If None, will be loaded from the template volume
            path. Defaults to None.
        particle_indices : list[pd.Index]
            The particle indices to process.
        """
        # Determine device from image_stack
        device = image_stack.device
        # Ensure the template is loaded in as a Tensor object
        if template_tensor is None:
            template = load_template_tensor(
                template_volume=self.template_volume,
                template_volume_path=self.template_volume_path,
            ).to(device)
        else:
            template = template_tensor.to(device)

        # The set of "best" euler angles from match template search
        # Check if refined angles exist, otherwise use the original angles
        euler_angles = self.particle_stack.get_euler_angles(prefer_refined_angles)
        euler_angles = euler_angles.to(device)

        # The relative Euler angle offsets to search over
        euler_angle_offsets = self.orientation_refinement_config.euler_angles_offsets
        euler_angle_offsets = euler_angle_offsets.to(device)
        # The relative defocus values to search over
        defocus_offsets = self.defocus_refinement_config.defocus_values
        defocus_offsets = defocus_offsets.to(device)
        # The relative pixel size values to search over
        pixel_size_offsets = self.pixel_size_refinement_config.pixel_size_values
        pixel_size_offsets = pixel_size_offsets.to(device)

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
            apply_global_filtering=self.apply_global_filtering,
            device_list=[device],
            image_stack=image_stack,
            mean_stack=mean_stack,
            std_stack=std_stack,
            particle_indices=particle_indices,
            images_are_particles=images_are_particles,
        )

    def run_refine_template(
        self,
        output_dataframe_path: str,
        image_stack: torch.Tensor,
        mean_stack: torch.Tensor,
        std_stack: torch.Tensor,
        particle_indices: list[pd.Index],
        template_tensor: torch.Tensor | None = None,
        correlation_batch_size: int = 32,
        images_are_particles: bool = False,
    ) -> None:
        """Run the refine template program and saves the resultant DataFrame to csv.

        Parameters
        ----------
        output_dataframe_path : str
            Path to save the refined particle data.
        image_stack : torch.Tensor
            Pre-loaded image stack tensor.
        mean_stack : torch.Tensor
            Pre-loaded mean stack tensor.
        std_stack : torch.Tensor
            Pre-loaded std stack tensor.
        template_tensor : torch.Tensor | None
            Pre-loaded template tensor. If None, will be loaded from the template volume
            path. Defaults to None.
        particle_indices : list[pd.Index]
            The particle indices to process.
        correlation_batch_size : int
            Number of cross-correlations to process in one batch, defaults to 32.
        images_are_particles : bool
            Whether the images are particles or not. Defaults to False.
        """
        backend_kwargs = self.make_backend_core_function_kwargs(
            image_stack=image_stack,
            mean_stack=mean_stack,
            std_stack=std_stack,
            template_tensor=template_tensor,
            particle_indices=particle_indices,
            images_are_particles=images_are_particles,
        )

        result = self.get_refine_result(backend_kwargs, correlation_batch_size)

        self.refine_result_to_dataframe(
            output_dataframe_path=output_dataframe_path, result=result
        )

    def get_refine_result(
        self, backend_kwargs: dict, correlation_batch_size: int = 32
    ) -> dict[str, torch.Tensor]:
        """Get differentiable refine result.

        Parameters
        ----------
        backend_kwargs : dict
            Keyword arguments for the backend processing
        correlation_batch_size : int
            Number of orientations to process at once. Defaults to 32.

        Returns
        -------
        dict[str, torch.Tensor]
            The result of the differentiable refine program.
        """
        # pylint: disable=duplicate-code
        result = core_differentiable_refine(
            batch_size=correlation_batch_size,
            num_cuda_streams=self.computational_config.num_cpus,
            **backend_kwargs,
        )

        return result

    # pylint: disable=too-many-locals
    def refine_result_to_dataframe(
        self,
        output_dataframe_path: str,
        result: dict[str, torch.Tensor],
        prefer_refined_angles: bool = True,
    ) -> None:
        """Convert refine template result to dataframe.

        Parameters
        ----------
        output_dataframe_path : str
            Path to save the refined particle data.
        result : dict[str, torch.Tensor]
            The result of the refine template program.
        prefer_refined_angles : bool
            Whether to use the refined angles or not. Defaults to True.
        """
        # pylint: disable=duplicate-code
        df_refined = self.particle_stack._df.copy()  # pylint: disable=protected-access
        result_dict = {k: v.cpu().detach().numpy() for k, v in result.items()}
        # x and y positions
        pos_offset_y = result_dict["refined_pos_y"]
        pos_offset_x = result_dict["refined_pos_x"]
        pos_offset_y_ang = pos_offset_y * df_refined["pixel_size"]
        pos_offset_x_ang = pos_offset_x * df_refined["pixel_size"]

        # pylint: disable=protected-access
        if (
            prefer_refined_angles
            and self.particle_stack._get_position_reference_columns()
            == ("refined_pos_y", "refined_pos_x")
        ):
            pos_y_col = "refined_pos_y"
            pos_x_col = "refined_pos_x"
            pos_y_col_img = "refined_pos_y_img"
            pos_x_col_img = "refined_pos_x_img"
            pos_y_col_img_angstrom = "refined_pos_y_img_angstrom"
            pos_x_col_img_angstrom = "refined_pos_x_img_angstrom"
        else:
            pos_y_col = "pos_y"
            pos_x_col = "pos_x"
            pos_y_col_img = "pos_y_img"
            pos_x_col_img = "pos_x_img"
            pos_y_col_img_angstrom = "pos_y_img_angstrom"
            pos_x_col_img_angstrom = "pos_x_img_angstrom"

        df_refined["refined_pos_y"] = pos_offset_y + df_refined[pos_y_col]
        df_refined["refined_pos_x"] = pos_offset_x + df_refined[pos_x_col]
        df_refined["refined_pos_y_img"] = pos_offset_y + df_refined[pos_y_col_img]
        df_refined["refined_pos_x_img"] = pos_offset_x + df_refined[pos_x_col_img]
        df_refined["refined_pos_y_img_angstrom"] = (
            pos_offset_y_ang + df_refined[pos_y_col_img_angstrom]
        )
        df_refined["refined_pos_x_img_angstrom"] = (
            pos_offset_x_ang + df_refined[pos_x_col_img_angstrom]
        )

        # Euler angles
        df_refined["refined_psi"] = result_dict["refined_euler_angles"][:, 2]
        df_refined["refined_theta"] = result_dict["refined_euler_angles"][:, 1]
        df_refined["refined_phi"] = result_dict["refined_euler_angles"][:, 0]

        # Defocus
        df_refined["refined_relative_defocus"] = (
            result_dict["refined_defocus_offset"]
            + self.particle_stack.get_relative_defocus().cpu().numpy()
        )

        # Pixel size
        df_refined["refined_pixel_size"] = (
            result_dict["refined_pixel_size_offset"]
            + self.particle_stack.get_pixel_size().cpu().numpy()
        )

        refined_mip = result_dict["refined_cross_correlation"]
        refined_scaled_mip = result_dict["refined_z_score"]
        df_refined["refined_mip"] = refined_mip
        df_refined["refined_scaled_mip"] = refined_scaled_mip

        # Reorder the columns
        df_refined = df_refined.reindex(columns=REFINED_DF_COLUMN_ORDER)

        # Save the refined DataFrame to disk
        df_refined.to_csv(output_dataframe_path)
