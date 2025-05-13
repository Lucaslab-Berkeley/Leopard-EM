"""Pydantic model for running the correlate frames program."""

import os
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch
from pydantic import ConfigDict
from ttsim3d.models import Simulator

from leopard_em.backend.core_refine_template import core_refine_template
from leopard_em.pydantic_models.config import (
    ComputationalConfig,
    PreprocessingFilters,
)
from leopard_em.pydantic_models.custom_types import BaseModel2DTM, ExcludedTensor
from leopard_em.pydantic_models.data_structures import ParticleStack
from leopard_em.pydantic_models.utils import (
    _setup_ctf_kwargs_from_particle_stack,
    setup_frame_filters_particle_stack,
    setup_image_normalization_factor,
)
from leopard_em.utils.data_io import (
    load_mrc_image,
    load_mrc_volume,
)


class CorrelateFramesManager(BaseModel2DTM):
    """Model holding parameters necessary for running the correlate frames program.

    Attributes
    ----------
    aligned_frames_path : str
        Path to the aligned frames mrc movie.
    template_pdb_path : str
        Path to the template pdb file.
    particle_stack : ParticleStack
        Particle stack object containing particle data.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    computational_config : ComputationalConfig
        What computational resources to allocate for the program.
    simulator : Simulator
        The simulator object.
    fluence_per_frame : Annotated[float, Field(ge=0.0)]
        The fluence per frame.


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

    aligned_frames_path: str
    sum_mrc_path: str
    particle_stack: ParticleStack
    preprocessing_filters: PreprocessingFilters
    computational_config: ComputationalConfig
    simulator: Simulator

    # Excluded tensors
    aligned_frames: ExcludedTensor
    sum_mrc: ExcludedTensor

    def _setup_frame_independent_kwargs(self) -> dict[str, Any]:
        """Setup backend kwargs that are independent of frames.

        Returns
        -------
        dict[str, Any]
            Dictionary containing frame-independent backend kwargs
        """
        defocus_u, defocus_v = self.particle_stack.get_absolute_defocus()
        defocus_angle = torch.tensor(self.particle_stack["astigmatism_angle"])

        ctf_kwargs = _setup_ctf_kwargs_from_particle_stack(
            self.particle_stack,
            (self.simulator.volume_shape[-2], self.simulator.volume_shape[-1]),
        )
        euler_angles = self.particle_stack.get_euler_angles(prefer_refined_angles=True)

        euler_angle_offsets = torch.zeros((1, 3))
        defocus_offsets = torch.tensor([0.0])
        pixel_size_offsets = torch.tensor([0.0])

        # Get correlation statistics
        corr_mean_stack = self.particle_stack.construct_cropped_statistic_stack(
            "correlation_average"
        )
        corr_std_stack = (
            self.particle_stack.construct_cropped_statistic_stack(
                "correlation_variance"
            )
            ** 0.5
        )  # var to std

        return {
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
            "device": self.computational_config.gpu_devices,
        }

    def _load_and_setup(
        self,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Load movie and sum MRC files and setup normalization.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]
            Tuple containing:
            - movie_mrc: The loaded movie volume
            - sum_mrc: The loaded sum MRC image
            - normalization_factor: The calculated normalization factor
            - refined_mips: Initialized array for refined MIPs
            - z_scores: Initialized array for z-scores
        """
        movie_mrc = load_mrc_volume(self.aligned_frames_path)
        sum_mrc = load_mrc_image(self.sum_mrc_path)

        num_particles = self.particle_stack.num_particles
        num_frames = len(movie_mrc)
        refined_mips = np.zeros((num_particles, num_frames))
        z_scores = np.zeros((num_particles, num_frames))

        normalization_factor = setup_image_normalization_factor(
            particle_stack=self.particle_stack,
            preprocessing_filters=self.preprocessing_filters,
            mrc_image=sum_mrc,
        )

        return movie_mrc, normalization_factor, refined_mips, z_scores

    def _simulate_template_for_frame(
        self, frame_idx: int, num_frames: int
    ) -> torch.Tensor:
        """Simulate template for a specific frame with appropriate fluence.

        Parameters
        ----------
        frame_idx : int
            Index of the current frame
        num_frames : int
            Total number of frames

        Returns
        -------
        torch.Tensor
            The simulated template for this frame
        """
        fluence_per_frame = (
            self.simulator.simulator_config.dose_end
            - self.simulator.simulator_config.dose_start
        ) / num_frames

        frame_fluence_start = (
            self.simulator.simulator_config.dose_start + fluence_per_frame * frame_idx
        )
        frame_fluence_end = (
            self.simulator.simulator_config.dose_start
            + fluence_per_frame * (frame_idx + 1)
        )

        self.simulator.simulator_config.dose_start = frame_fluence_start
        self.simulator.simulator_config.dose_end = frame_fluence_end

        return self.simulator.run()

    def _setup_frame_kwargs(
        self, frame: np.ndarray, template: torch.Tensor, normalization_factor: float
    ) -> dict[str, Any]:
        """Setup backend kwargs for a specific frame.

        Parameters
        ----------
        frame : np.ndarray
            The current frame to process
        template : torch.Tensor
            The simulated template for this frame
        normalization_factor : float
            Normalization factor for the images

        Returns
        -------
        dict[str, Any]
            Dictionary containing frame-specific backend kwargs
        """
        # Move template to GPU
        template = template.to(self.computational_config.gpu_devices[0])

        # pre-process template and frame
        particle_images_dft, template_dft, projective_filters = (
            setup_frame_filters_particle_stack(
                particle_stack=self.particle_stack,
                preprocessing_filters=self.preprocessing_filters,
                template=template,
                mrc_image=frame,
                normalization_factor=normalization_factor,
            )
        )

        return {
            "particle_stack_dft": particle_images_dft,
            "template_dft": template_dft,
            "projective_filters": projective_filters,
        }

    def _process_all_results(
        self, refined_mips: np.ndarray, z_scores: np.ndarray, output_dataframe_path: str
    ) -> None:
        """Process and save all results from all frames.

        Parameters
        ----------
        refined_mips : np.ndarray
            Array containing refined MIPs for all frames
        z_scores : np.ndarray
            Array containing z-scores for all frames
        output_dataframe_path : str
            Path to save the output dataframes
        """
        # Create dataframes for MIPs and z-scores with each frame in a column
        df_refined = self.particle_stack._df.copy()
        frames_df_mip = pd.DataFrame(
            {
                "particle_index": df_refined["particle_index"],
                "aligned_frames_path": self.aligned_frames_path,
            }
        )
        frames_df_zscore = pd.DataFrame(
            {
                "particle_index": df_refined["particle_index"],
                "aligned_frames_path": self.aligned_frames_path,
            }
        )

        # Add each frame's MIP and z-score as a column
        for i in range(refined_mips.shape[1]):
            frames_df_mip[f"frame_{i}_mip"] = refined_mips[:, i]
            frames_df_zscore[f"frame_{i}_zscore"] = z_scores[:, i]

        # Get base path without extension
        base_path = os.path.splitext(output_dataframe_path)[0]

        # Save frames dataframes
        frames_df_mip.to_csv(f"{base_path}_frames_mip.csv", index=False)
        frames_df_zscore.to_csv(f"{base_path}_frames_zscore.csv", index=False)

        # Sum results to standard dataframe
        sum_refined_mip = np.sum(refined_mips, axis=1)
        sum_z_score = np.sum(z_scores, axis=1)

        # Save the summed results
        df_refined = self.particle_stack._df.copy()
        df_refined["refined_mip"] = sum_refined_mip
        df_refined["refined_scaled_mip"] = sum_z_score
        df_refined.to_csv(output_dataframe_path, index=False)

    def run_correlate_frames(
        self, output_dataframe_path: str, orientation_batch_size: int = 64
    ) -> None:
        """Run the correlate frames program.

        Parameters
        ----------
        output_dataframe_path : str
            Path to save the output dataframes
        orientation_batch_size : int
            Number of orientations to process at once. Defaults to 64.

        Returns
        -------
        None
        """
        # 1) Setup frame-independent backend kwargs
        frame_independent_kwargs = self._setup_frame_independent_kwargs()

        # 2) Load movie and setup arrays
        movie_mrc, normalization_factor, refined_mips, z_scores = self._load_and_setup()

        # Process each frame
        for i, frame in enumerate(movie_mrc):
            # Simulate template for this frame
            template = self._simulate_template_for_frame(i, len(movie_mrc))

            # Setup frame-specific kwargs
            frame_kwargs = self._setup_frame_kwargs(
                frame, template, normalization_factor
            )

            # Combine all kwargs
            backend_kwargs = {**frame_independent_kwargs, **frame_kwargs}

            print(f"running for frame {i+1} of {len(movie_mrc)}")
            result = self.get_refine_result(backend_kwargs, orientation_batch_size)

            # Store results directly
            refined_mips[:, i] = result["refined_cross_correlation"]
            z_scores[:, i] = result["refined_z_score"]

        # Process and save all results
        self._process_all_results(refined_mips, z_scores, output_dataframe_path)

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
        orientation_batch_size = 1
        # pylint: disable=duplicate-code
        result: dict[str, np.ndarray] = {}
        result = core_refine_template(
            batch_size=orientation_batch_size, **backend_kwargs
        )
        result = {k: v.cpu().numpy() for k, v in result.items()}

        return result
