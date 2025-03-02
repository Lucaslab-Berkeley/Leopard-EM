"""Root-level model for serialization and validation of 2DTM parameters."""

import os
from typing import Any, ClassVar, Literal, Optional

import mrcfile
import pandas as pd
import torch
from pydantic import ConfigDict, field_validator

from leopard_em.backend.core_match_template import core_match_template
from leopard_em.pydantic_models.computational_config import ComputationalConfig
from leopard_em.pydantic_models.correlation_filters import PreprocessingFilters
from leopard_em.pydantic_models.defocus_search import DefocusSearchConfig
from leopard_em.pydantic_models.formats import MATCH_TEMPLATE_DF_COLUMN_ORDER
from leopard_em.pydantic_models.match_template_result import MatchTemplateResult
from leopard_em.pydantic_models.optics_group import OpticsGroup
from leopard_em.pydantic_models.orientation_search import OrientationSearchConfig
from leopard_em.pydantic_models.types import BaseModel2DTM, ExcludedTensor
from leopard_em.utils.data_io import load_mrc_image, load_mrc_volume
from leopard_em.utils.pre_processing import calculate_ctf_filter_stack


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
    computational_config : ComputationalConfig
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
    orientation_search_config: OrientationSearchConfig
    preprocessing_filters: PreprocessingFilters
    match_template_result: MatchTemplateResult
    computational_config: ComputationalConfig

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

        if not isinstance(self.micrograph, torch.Tensor):
            image = torch.from_numpy(self.micrograph)
        else:
            image = self.micrograph

        if not isinstance(self.template_volume, torch.Tensor):
            template = torch.from_numpy(self.template_volume)
        else:
            template = self.template_volume

        template_shape = template.shape[-2:]

        # Fourier transform the image (RFFT, unshifted)
        image_dft = torch.fft.rfftn(image)
        image_dft[0, 0] = 0 + 0j  # zero out the constant term

        # Get the combined filter from the pre-processing filter attribute
        cumulative_filter = self.preprocessing_filters.get_combined_filter(
            ref_img_rfft=image_dft,
            output_shape=image_dft.shape,
        )
        image_preprocessed_dft = image_dft * cumulative_filter

        # Normalize the image after filtering
        squared_image_dft = torch.abs(image_preprocessed_dft) ** 2
        squared_sum = squared_image_dft.sum() + squared_image_dft[:, 1:-1].sum()
        image_preprocessed_dft /= torch.sqrt(squared_sum)

        # NOTE: For two Gaussian random variables in d-dimensional space --  A and B --
        # each with mean 0 and variance 1 their correlation will have on average a
        # variance of d.
        # NOTE: Since we have the variance of the image and template projections each at
        # 1, we need to multiply the image by the square root of the number of pixels
        # so the cross-correlograms have a variance of 1 and not d.
        # NOTE: When applying the Fourier filters to the image and template, any
        # elements that get set to zero effectively reduce the dimensionality of our
        # cross-correlation. Therefore, instead of multiplying by the number of pixels,
        # we need to multiply tby the effective number of pixels that are non-zero.
        # Below, we calculate the dimensionality of our cross-correlation and divide
        # by the square root of that number to normalize the image.
        bp_config = self.preprocessing_filters.bandpass_filter
        bp_filter_image = bp_config.calculate_bandpass_filter(image_dft.shape)
        dimensionality = bp_filter_image.sum() + bp_filter_image[:, 1:-1].sum()
        image_preprocessed_dft *= dimensionality**0.5

        # Next calculate the filters in terms of the template
        # NOTE: Here, manually accounting for the RFFT in output shape since we have not
        # RFFT'd the template volume yet. Also, this is 2-dimensional, not 3-dimensional
        cumulative_filter_template = self.preprocessing_filters.get_combined_filter(
            ref_img_rfft=image_dft,
            output_shape=(template_shape[-2], template_shape[-1] // 2 + 1),
        )

        # Calculate the CTF filters at each defocus value
        defocus_values = self.defocus_search_config.defocus_values
        # set pixel search to 0.0 for match template
        pixel_size_offsets = torch.tensor([0.0], dtype=torch.float32)
        ctf_filters = calculate_ctf_filter_stack(
            pixel_size=self.optics_group.pixel_size,
            template_shape=(template_shape[0], template_shape[0]),
            defocus_u=self.optics_group.defocus_u * 1e-4,  # A to um
            defocus_v=self.optics_group.defocus_v * 1e-4,  # A to um
            defocus_offsets=defocus_values * 1e-4,  # A to um
            pixel_size_offsets=pixel_size_offsets,
            astigmatism_angle=self.optics_group.astigmatism_angle,
            amplitude_contrast_ratio=self.optics_group.amplitude_contrast_ratio,
            spherical_aberration=self.optics_group.spherical_aberration,
            phase_shift=self.optics_group.phase_shift,
            voltage=self.optics_group.voltage,
            ctf_B_factor=self.optics_group.ctf_B_factor,
        )

        # Grab the Euler angles from the orientation search configuration
        # (phi, theta, psi) for ZYZ convention
        euler_angles = self.orientation_search_config.euler_angles
        euler_angles = euler_angles.to(torch.float32)

        device_list = self.computational_config.gpu_devices

        # Calculate the DFT of the template to take Fourier slices from
        # NOTE: There is an extra FFTshift step before the RFFT since, for some reason,
        # omitting this step will cause a 180 degree phase shift on odd (i, j, k)
        # structure factors in the Fourier domain. This just requires an extra
        # IFFTshift after converting a slice back to real-space (handled already).
        template_dft = torch.fft.fftshift(template, dim=(-3, -2, -1))
        template_dft = torch.fft.rfftn(template_dft, dim=(-3, -2, -1))
        template_dft = torch.fft.fftshift(template_dft, dim=(-3, -2))  # skip rfft dim

        return {
            "image_dft": image_preprocessed_dft,
            "template_dft": template_dft,
            "ctf_filters": ctf_filters,
            "whitening_filter_template": cumulative_filter_template,
            "euler_angles": euler_angles,
            "defocus_values": defocus_values,
            "pixel_values": pixel_size_offsets,
            "device": device_list,
        }

    def run_match_template(
        self,
        orientation_batch_size: int = 1,
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
            **core_kwargs, orientation_batch_size=orientation_batch_size
        )

        # Place results into the `MatchTemplateResult` object and save it.
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
