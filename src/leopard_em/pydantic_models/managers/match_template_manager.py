"""Root-level model for serialization and validation of 2DTM parameters."""

import os
from typing import Any, ClassVar, Literal, Optional

import mrcfile
import pandas as pd
import torch
from pydantic import ConfigDict, field_validator

from leopard_em.backend.core_match_template import core_match_template
from leopard_em.backend.core_match_template_distributed import (
    core_match_template_distributed,
)
from leopard_em.pydantic_models.config import (
    ComputationalConfigMatch,
    DefocusSearchConfig,
    MultipleOrientationConfig,
    OrientationSearchConfig,
    PreprocessingFilters,
)
from leopard_em.pydantic_models.custom_types import BaseModel2DTM, ExcludedTensor
from leopard_em.pydantic_models.data_structures import OpticsGroup
from leopard_em.pydantic_models.formats import MATCH_TEMPLATE_DF_COLUMN_ORDER
from leopard_em.pydantic_models.results import MatchTemplateResult
from leopard_em.pydantic_models.utils import (
    calculate_ctf_filter_stack,
    preprocess_image,
    volume_to_rfft_fourier_slice,
)
from leopard_em.utils.data_io import load_mrc_image, load_mrc_volume


# pylint: disable=no-self-argument
class MatchTemplateManager(BaseModel2DTM):
    """Model holding parameters necessary for running full orientation 2DTM.

    Attributes
    ----------
    micrograph_path : str
        Path to the micrograph .mrc file.
    template_volume_path : str
        Path to the template volume .mrc file.
    micrograph : ExcludedTensor
        Image to run template matching on. Not serialized.
    template_volume : ExcludedTensor
        Template volume to match against. Not serialized.
    optics_group : OpticsGroup
        Optics group parameters for the imaging system on the microscope.
    defocus_search_config : DefocusSearchConfig
        Parameters for searching over defocus values.
    orientation_search_config : OrientationSearchConfig
        Parameters for searching over orientation angles.
    preprocessing_filters : PreprocessingFilters
        Configurations for the preprocessing filters to apply during
        correlation.
    match_template_result : MatchTemplateResult
        Result of the match template program stored as an instance of the
        `MatchTemplateResult` class.
    computational_config : ComputationalConfigMatch
        Parameters for controlling computational resources.

    Methods
    -------
    validate_micrograph_path(v: str) -> str
        Ensure the micrograph file exists.
    validate_template_volume_path(v: str) -> str
        Ensure the template volume file exists.
    __init__(preload_mrc_files: bool = False , **data: Any)
        Constructor which also loads the micrograph and template volume from disk.
        The 'preload_mrc_files' parameter controls whether to read the MRC files
        immediately upon initialization.
    make_backend_core_function_kwargs() -> dict[str, Any]
        Generates the keyword arguments for backend 'core_match_template' call from
        held parameters. Does the necessary pre-processing steps to filter the image
        and template.
    run_match_template(orientation_batch_size: int = 1, do_result_export: bool = True)
        Runs the base match template program in PyTorch.
    results_to_dataframe(
        half_template_width_pos_shift: bool = True,
        exclude_columns: Optional[list] = None,
        locate_peaks_kwargs: Optional[dict] = None,
    ) -> pd.DataFrame
        Converts the basic extracted peak info DataFrame (from the result object) to a
        DataFrame with additional information about reference files, microscope
        parameters, etc.
    save_config(path: str, mode: Literal["yaml", "json"] = "yaml") -> None
        Save this Pydantic model config to disk.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    # Serialized attributes
    micrograph_path: str
    template_volume_path: str
    optics_group: OpticsGroup
    defocus_search_config: DefocusSearchConfig
    orientation_search_config: OrientationSearchConfig | MultipleOrientationConfig
    preprocessing_filters: PreprocessingFilters
    match_template_result: MatchTemplateResult
    computational_config: ComputationalConfigMatch

    # Non-serialized large array-like attributes
    micrograph: ExcludedTensor
    template_volume: ExcludedTensor

    ###########################
    ### Pydantic Validators ###
    ###########################

    @field_validator("micrograph_path")  # type: ignore
    def validate_micrograph_path(cls, v) -> str:
        """Ensure the micrograph file exists."""
        if not os.path.exists(v):
            raise ValueError(f"File '{v}' for micrograph does not exist.")

        return str(v)

    @field_validator("template_volume_path")  # type: ignore
    def validate_template_volume_path(cls, v) -> str:
        """Ensure the template volume file exists."""
        if not os.path.exists(v):
            raise ValueError(f"File '{v}' for template volume does not exist.")

        return str(v)

    def __init__(self, preload_mrc_files: bool = False, **data: Any):
        super().__init__(**data)

        if preload_mrc_files:
            # Load the data from the MRC files
            self.micrograph = load_mrc_image(self.micrograph_path)
            self.template_volume = load_mrc_volume(self.template_volume_path)

    ############################################
    ### Functional (data processing) methods ###
    ############################################

    def make_backend_core_function_kwargs(self) -> dict[str, Any]:
        """Generates the keyword arguments for backend call from held parameters."""
        # Ensure the micrograph and template are loaded and in the correct format
        if self.micrograph is None:
            self.micrograph = load_mrc_image(self.micrograph_path)
        if self.template_volume is None:
            self.template_volume = load_mrc_volume(self.template_volume_path)

        # Ensure the micrograph and template are both Tensors before proceeding
        if not isinstance(self.micrograph, torch.Tensor):
            image = torch.from_numpy(self.micrograph)
        else:
            image = self.micrograph

        if not isinstance(self.template_volume, torch.Tensor):
            template = torch.from_numpy(self.template_volume)
        else:
            template = self.template_volume

        # Fourier transform the image (RFFT, unshifted)
        image_dft = torch.fft.rfftn(image)  # pylint: disable=E1102
        image_dft[0, 0] = 0 + 0j  # zero out the constant term

        # Get the bandpass filter individually
        bp_config = self.preprocessing_filters.bandpass_filter
        bandpass_filter = bp_config.calculate_bandpass_filter(image_dft.shape)

        # Calculate the cumulative filters for both the image and the template.
        cumulative_filter_image = self.preprocessing_filters.get_combined_filter(
            ref_img_rfft=image_dft,
            output_shape=image_dft.shape,
        )
        # NOTE: Here, manually accounting for the RFFT in output shape since we have not
        # RFFT'd the template volume yet. Also, this is 2-dimensional, not 3-dimensional
        cumulative_filter_template = self.preprocessing_filters.get_combined_filter(
            ref_img_rfft=image_dft,
            output_shape=(template.shape[-2], template.shape[-1] // 2 + 1),
        )

        # Apply the pre-processing and normalization
        image_preprocessed_dft = preprocess_image(
            image_rfft=image_dft,
            cumulative_fourier_filters=cumulative_filter_image,
            bandpass_filter=bandpass_filter,
        )

        # Calculate the CTF filters at each defocus value
        defocus_values = self.defocus_search_config.defocus_values

        # set pixel search to 0.0 for match template
        pixel_size_offsets = torch.tensor([0.0], dtype=torch.float32)

        ctf_filters = calculate_ctf_filter_stack(
            template_shape=(template.shape[0], template.shape[0]),
            optics_group=self.optics_group,
            defocus_offsets=defocus_values,
            pixel_size_offsets=pixel_size_offsets,
        )

        # Grab the Euler angles from the orientation search configuration
        # (phi, theta, psi) for ZYZ convention
        euler_angles = self.orientation_search_config.euler_angles
        euler_angles = euler_angles.to(torch.float32)

        template_dft = volume_to_rfft_fourier_slice(template)

        return {
            "image_dft": image_preprocessed_dft,
            "template_dft": template_dft,
            "ctf_filters": ctf_filters,
            "whitening_filter_template": cumulative_filter_template,
            "euler_angles": euler_angles,
            "defocus_values": defocus_values,
            "pixel_values": pixel_size_offsets,
            "device": self.computational_config.gpu_devices,
        }

    def run_match_template(
        self,
        orientation_batch_size: int = 16,
        do_result_export: bool = True,
        do_valid_cropping: bool = True,
    ) -> None:
        """Runs the base match template in pytorch.

        Parameters
        ----------
        orientation_batch_size : int
            The number of projections to process in a single batch. Default is 1.
        do_result_export : bool
            If True, call the `MatchTemplateResult.export_results` method to save the
            results to disk directly after running the match template. Default is True.
        do_valid_cropping : bool
            If True, apply the valid cropping mode to the results. Default is True.

        Returns
        -------
        None
        """
        core_kwargs = self.make_backend_core_function_kwargs()
        results = core_match_template(
            **core_kwargs,
            orientation_batch_size=orientation_batch_size,
            num_cuda_streams=self.computational_config.num_cpus,
            backend=self.computational_config.backend,
        )

        # Populate the MatchTemplateResult via a private helper
        self._populate_match_template_result(
            results,
            do_result_export=do_result_export,
            do_valid_cropping=do_valid_cropping,
        )

    def run_match_template_distributed(
        self,
        world_size: int,
        rank: int,
        local_rank: int,
        orientation_batch_size: int = 16,
        do_result_export: bool = True,
        do_valid_cropping: bool = True,
    ) -> None:
        """Runs the base match template in a distributed, multi-node environment.

        Parameters
        ----------
        world_size : int
            The total number of processes in the distributed job.
        rank : int
            The global rank of this process.
        local_rank : int
            The local rank of this process (used to assign GPU).
        orientation_batch_size : int
            The number of projections to process in a single batch. Default is 1.
        do_result_export : bool
            If True, call the `MatchTemplateResult.export_results` method to save the
            results to disk directly after running the match template. Default is True.
        do_valid_cropping : bool
            If True, apply the valid cropping mode to the results. Default is True.

        Raises
        ------
        RuntimeError
            If the distributed process group has not been initialized.

        Returns
        -------
        None
        """
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "Distributed process group has not been initialized! "
                "Cannot run distributed match template."
            )

        device = torch.device(f"cuda:{local_rank}")

        if rank == 0:
            core_kwargs = self.make_backend_core_function_kwargs()
        else:
            core_kwargs = {}

        _ = core_kwargs.pop("device", None)

        results = core_match_template_distributed(
            world_size,
            rank,
            local_rank,
            device,
            orientation_batch_size,
            self.computational_config.num_cpus,
            self.computational_config.backend,
            **core_kwargs,
        )

        # Only populate the results on the first rank
        if torch.distributed.get_rank() == 0:
            self._populate_match_template_result(
                results,
                do_result_export=do_result_export,
                do_valid_cropping=do_valid_cropping,
            )

    def _populate_match_template_result(
        self,
        results: dict[str, Any],
        do_result_export: bool = True,
        do_valid_cropping: bool = True,
    ) -> None:
        """Helper function to populate the MatchTemplateResult object post-core call."""
        # Place results into the `MatchTemplateResult` object
        self.match_template_result.mip = results["mip"]
        self.match_template_result.scaled_mip = results["scaled_mip"]

        self.match_template_result.correlation_average = results["correlation_mean"]
        self.match_template_result.correlation_variance = results[
            "correlation_variance"
        ]
        self.match_template_result.orientation_psi = results["best_psi"]
        self.match_template_result.orientation_theta = results["best_theta"]
        self.match_template_result.orientation_phi = results["best_phi"]
        self.match_template_result.relative_defocus = results["best_defocus"]

        self.match_template_result.total_projections = results["total_projections"]
        self.match_template_result.total_orientations = results["total_orientations"]
        self.match_template_result.total_defocus = results["total_defocus"]

        # Apply the valid cropping mode to the results
        if do_valid_cropping:
            nx = self.template_volume.shape[-1]
            self.match_template_result.apply_valid_cropping((nx, nx))

        # Export the results to disk, if requested
        if do_result_export:
            self.match_template_result.export_results()

    def results_to_dataframe(
        self,
        half_template_width_pos_shift: bool = True,
        exclude_columns: Optional[list] = None,
        locate_peaks_kwargs: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Converts the match template results to a DataFrame with additional info.

        Data included in this dataframe should be sufficient to do cross-correlation on
        the extracted peaks, that is, all the microscope parameters, defocus parameters,
        etc. are included in the dataframe. Run-specific filter information is *not*
        included in this dataframe; use the YAML configuration file to replicate a
        match_template run.

        Parameters
        ----------
        half_template_width_pos_shift : bool, optional
            If True, columns for the image peak position are shifted by half a template
            width to correspond to the center of the particle. This should be done when
            the position of a peak corresponds to the top-left corner of the template
            rather than the center. Default is True. This should generally be left as
            True unless you know what you are doing.
        exclude_columns : list, optional
            List of columns to exclude from the DataFrame. Default is None and no
            columns are excluded.
        locate_peaks_kwargs : dict, optional
            Keyword arguments to pass to the 'MatchTemplateResult.locate_peaks' method.
            Default is None and no additional keyword arguments are passed.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the match template results.
        """
        # Short circuit if no kwargs and peaks have already been located
        if locate_peaks_kwargs is None:
            if self.match_template_result.match_template_peaks is None:
                self.match_template_result.locate_peaks()
        else:
            self.match_template_result.locate_peaks(**locate_peaks_kwargs)

        # DataFrame comes with the following columns :
        # ['mip', 'scaled_mip', 'correlation_mean', 'correlation_variance',
        # 'total_correlations'. 'pos_y', 'pos_x', 'psi', 'theta', 'phi',
        # 'relative_defocus', ]
        df = self.match_template_result.peaks_to_dataframe()

        # DataFrame currently contains pixel coordinates for results. Coordinates in
        # image correspond with upper left corner of the template. Need to translate
        # coordinates by half template width to get to particle center in image.
        # NOTE: We are assuming the template is cubic
        nx = mrcfile.open(self.template_volume_path).header.nx
        if half_template_width_pos_shift:
            df["pos_y_img"] = df["pos_y"] + nx // 2
            df["pos_x_img"] = df["pos_x"] + nx // 2
        else:
            df["pos_y_img"] = df["pos_y"]
            df["pos_x_img"] = df["pos_x"]

        # Also, the positions are in terms of pixels. Also add columns for particle
        # positions in terms of Angstroms.
        pixel_size = self.optics_group.pixel_size
        df["pos_y_img_angstrom"] = df["pos_y_img"] * pixel_size
        df["pos_x_img_angstrom"] = df["pos_x_img"] * pixel_size

        # Add microscope (CTF) parameters
        df["defocus_u"] = self.optics_group.defocus_u
        df["defocus_v"] = self.optics_group.defocus_v
        df["astigmatism_angle"] = self.optics_group.astigmatism_angle
        df["pixel_size"] = pixel_size
        df["refined_pixel_size"] = pixel_size
        df["voltage"] = self.optics_group.voltage
        df["spherical_aberration"] = self.optics_group.spherical_aberration
        df["amplitude_contrast_ratio"] = self.optics_group.amplitude_contrast_ratio
        df["phase_shift"] = self.optics_group.phase_shift
        df["ctf_B_factor"] = self.optics_group.ctf_B_factor

        # Add paths to the micrograph and reference template
        df["micrograph_path"] = self.micrograph_path
        df["template_path"] = self.template_volume_path

        # Add paths to the output statistic files
        df["mip_path"] = self.match_template_result.mip_path
        df["scaled_mip_path"] = self.match_template_result.scaled_mip_path
        df["psi_path"] = self.match_template_result.orientation_psi_path
        df["theta_path"] = self.match_template_result.orientation_theta_path
        df["phi_path"] = self.match_template_result.orientation_phi_path
        df["defocus_path"] = self.match_template_result.relative_defocus_path
        df["correlation_average_path"] = (
            self.match_template_result.correlation_average_path
        )
        df["correlation_variance_path"] = (
            self.match_template_result.correlation_variance_path
        )

        # Add particle index
        df["particle_index"] = df.index

        # Reorder columns
        df = df.reindex(columns=MATCH_TEMPLATE_DF_COLUMN_ORDER)

        # Drop columns if requested
        if exclude_columns is not None:
            df = df.drop(columns=exclude_columns)

        return df

    def save_config(self, path: str, mode: Literal["yaml", "json"] = "yaml") -> None:
        """Save this Pydandic model to disk. Wrapper around the serialization methods.

        Parameters
        ----------
        path : str
            Path to save the configuration file.
        mode : Literal["yaml", "json"], optional
            Serialization format to use. Default is 'yaml'.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If an invalid serialization mode is provided.
        """
        if mode == "yaml":
            self.to_yaml(path)
        elif mode == "json":
            self.to_json(path)
        else:
            raise ValueError(f"Invalid serialization mode '{mode}'.")
