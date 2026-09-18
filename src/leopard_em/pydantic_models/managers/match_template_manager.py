"""Root-level model for serialization and validation of 2DTM parameters."""

import json
import os
from typing import Any, ClassVar, Literal

import mrcfile
import pandas as pd
import torch
from pydantic import ConfigDict, PrivateAttr, field_validator

from leopard_em.backend.core_match_template import core_match_template
from leopard_em.backend.core_match_template_distributed import (
    core_match_template_distributed,
)
from leopard_em.pydantic_models.config import (
    ComputationalConfigMatch,
    DefocusSearchConfig,
    FastFFTPaddingConfig,
    MultipleOrientationConfig,
    OrientationSearchConfig,
    PreprocessingFilters,
)
from leopard_em.pydantic_models.custom_types import BaseModel2DTM, ExcludedTensor
from leopard_em.pydantic_models.data_structures import OpticsGroup
from leopard_em.pydantic_models.formats import MATCH_TEMPLATE_DF_COLUMN_ORDER
from leopard_em.pydantic_models.results import (
    MatchTemplateResultHDF5,
    MatchTemplateResultMRC,
)
from leopard_em.pydantic_models.results.correlation_table import CorrelationTable
from leopard_em.utils.ctf_utils import calculate_ctf_filter_stack
from leopard_em.utils.data_io import load_mrc_image, load_mrc_volume
from leopard_em.utils.fft_padding import FFTPaddingPlan, filter_correlation_table
from leopard_em.utils.fourier_slice import volume_to_rfft_fourier_slice
from leopard_em.utils.image_processing import (
    get_image_normalization_factor,
    preprocess_image,
)


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
    match_template_result : MatchTemplateResultMRC | MatchTemplateResultHDF5
        Result of the match template program.  Use ``MatchTemplateResultMRC``
        to write individual MRC files or ``MatchTemplateResultHDF5`` to bundle
        all tensors into a single HDF5 file.
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
    fast_fft_padding: FastFFTPaddingConfig = FastFFTPaddingConfig()
    match_template_result: MatchTemplateResultMRC | MatchTemplateResultHDF5
    computational_config: ComputationalConfigMatch

    # Non-serialized large array-like attributes
    micrograph: ExcludedTensor
    template_volume: ExcludedTensor

    # Padding plan for the most recent 'make_backend_core_function_kwargs' call. Used
    # to tell the backend which valid region to accumulate, and to verify the shapes
    # it returns. Reset on every call so it can never go stale.
    _fft_padding_plan: FFTPaddingPlan | None = PrivateAttr(default=None)

    # Keys of the backend result dict holding per-pixel statistics maps.
    _RESULT_MAP_KEYS: ClassVar[tuple[str, ...]] = (
        "mip",
        "scaled_mip",
        "best_phi",
        "best_theta",
        "best_psi",
        "best_defocus",
        "correlation_mean",
        "correlation_variance",
    )

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

    def _ensure_inputs_loaded(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Load the micrograph and template volume as tensors, if not already held.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The micrograph and the template volume.
        """
        if self.micrograph is None:
            self.micrograph = load_mrc_image(self.micrograph_path)
        if self.template_volume is None:
            self.template_volume = load_mrc_volume(self.template_volume_path)

        image = self.micrograph
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)

        template = self.template_volume
        if not isinstance(template, torch.Tensor):
            template = torch.from_numpy(template)

        return image, template

    def _preprocess_padded_image(
        self,
        image_dft: torch.Tensor,
        image_dft_original: torch.Tensor,
        cumulative_filter_image: torch.Tensor,
        bandpass_filter: torch.Tensor,
        padding_plan: FFTPaddingPlan,
    ) -> torch.Tensor:
        """Filter and normalize the image, correcting the scale when padded.

        Parameters
        ----------
        image_dft : torch.Tensor
            RFFT of the padded image.
        image_dft_original : torch.Tensor
            RFFT of the original, unpadded image.
        cumulative_filter_image : torch.Tensor
            Combined Fourier filter evaluated on the padded grid.
        bandpass_filter : torch.Tensor
            Bandpass filter evaluated on the padded grid.
        padding_plan : FFTPaddingPlan
            Plan describing the padding that was applied.

        Returns
        -------
        torch.Tensor
            Filtered and normalized image in Fourier space.
        """
        if not padding_plan.is_padded:
            return preprocess_image(
                image_rfft=image_dft,
                cumulative_fourier_filters=cumulative_filter_image,
                bandpass_filter=bandpass_filter,
                full_image_shape=padding_plan.original_shape,
                extracted_box_shape=padding_plan.original_shape,
            )

        bp_config = self.preprocessing_filters.bandpass_filter
        bandpass_original = bp_config.calculate_bandpass_filter(
            image_dft_original.shape
        )
        cumulative_original = self.preprocessing_filters.get_combined_filter(
            ref_img_rfft=image_dft_original,
            output_shape=image_dft_original.shape,
            apply_random_dropout=False,
        )
        normalization_factor = get_image_normalization_factor(
            image_rfft=image_dft_original,
            cumulative_fourier_filters=cumulative_original,
            bandpass_filter=bandpass_original,
            full_image_shape=padding_plan.original_shape,
            extracted_box_shape=padding_plan.original_shape,
        )

        # The factor scales as 1 / sqrt(number of pixels), so moving from the original
        # grid to the padded grid multiplies it by sqrt(N_original / N_padded).
        original_area = padding_plan.original_shape[0] * padding_plan.original_shape[1]
        padded_area = padding_plan.padded_shape[0] * padding_plan.padded_shape[1]
        normalization_factor = (
            normalization_factor * (original_area / padded_area) ** 0.5
        )

        return image_dft * cumulative_filter_image * normalization_factor

    def make_backend_core_function_kwargs(self) -> dict[str, Any]:
        """Generates the keyword arguments for backend call from held parameters."""
        image, template = self._ensure_inputs_loaded()

        # Reset before recomputing so a failure here cannot leave a stale plan behind.
        self._fft_padding_plan = None
        padding_plan = self.fast_fft_padding.make_plan(
            image_shape=(int(image.shape[-2]), int(image.shape[-1])),
            template_shape=(int(template.shape[-2]), int(template.shape[-1])),
            backend=self.computational_config.backend,
        )
        self._fft_padding_plan = padding_plan

        # Fourier transform the image (RFFT, unshifted). The *unpadded* transform is
        # the reference for every Fourier filter and for the normalization scalar, so
        # that synthetic padding noise can never bias the measured power spectrum.
        # The padded transform is what the backend actually correlates against.
        image_dft_original = torch.fft.rfftn(image)  # pylint: disable=E1102
        image_dft_original[0, 0] = 0 + 0j  # zero out the constant term

        if padding_plan.is_padded:
            image_dft = torch.fft.rfftn(  # pylint: disable=E1102
                padding_plan.pad(image)
            )
            image_dft[0, 0] = 0 + 0j
        else:
            image_dft = image_dft_original

        # Get the bandpass filter individually
        bp_config = self.preprocessing_filters.bandpass_filter
        bandpass_filter = bp_config.calculate_bandpass_filter(image_dft.shape)

        # Calculate the cumulative filters for both the image and the template.
        # NOTE: We don't want to do random fourier masking on the image, so skip the
        # dropout mask for the image-side filter (without mutating the config).
        # NOTE: Filters are computed on the unpadded image but applied to the padded
        # Fourier transformed image.
        cumulative_filter_image = self.preprocessing_filters.get_combined_filter(
            ref_img_rfft=image_dft_original,
            output_shape=image_dft.shape,
            apply_random_dropout=False,
        )

        # NOTE: Here, manually accounting for the RFFT in output shape since we have not
        # RFFT'd the template volume yet. Also, this is 2-dimensional, not 3-dimensional
        cumulative_filter_template = self.preprocessing_filters.get_combined_filter(
            ref_img_rfft=image_dft_original,
            output_shape=(template.shape[-2], template.shape[-1] // 2 + 1),
            apply_random_dropout=True,
        )

        # Apply the pre-processing and normalization
        image_preprocessed_dft = self._preprocess_padded_image(
            image_dft=image_dft,
            image_dft_original=image_dft_original,
            cumulative_filter_image=cumulative_filter_image,
            bandpass_filter=bandpass_filter,
            padding_plan=padding_plan,
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
            "mag_matrix": self.optics_group.mag_matrix_tensor,
        }

    def run_match_template(
        self,
        orientation_batch_size: int = 16,
        do_result_export: bool = True,
        compute_correlation_table: bool = False,
    ) -> None:
        """Runs the base match template in pytorch.

        Parameters
        ----------
        orientation_batch_size : int
            The number of projections to process in a single batch. Default is 1.
        do_result_export : bool
            If True, call the `MatchTemplateResult.export_results` method to save the
            results to disk directly after running the match template. Default is True.
        compute_correlation_table : bool
            If True, track cross-correlation values which surpass the correlation
            table threshold during the search. If False, the `CorrelationTable` will be
            empty. Incurs a small runtime overhead when enabled. Default is False.

        Returns
        -------
        None
        """
        core_kwargs = self.make_backend_core_function_kwargs()
        results = core_match_template(
            **core_kwargs,
            orientation_batch_size=orientation_batch_size,
            num_cuda_streams=self.computational_config.num_cpus,
            backend=self._resolved_backend(),
            compute_correlation_table=compute_correlation_table,
            unpadded_valid_shape=self._unpadded_valid_shape(),
        )

        # Populate the MatchTemplateResult via a private helper
        self._populate_match_template_result(
            results,
            defocus_values=core_kwargs["defocus_values"],
            euler_angles=core_kwargs["euler_angles"],
            do_result_export=do_result_export,
        )

    def run_match_template_distributed(
        self,
        world_size: int,
        rank: int,
        local_rank: int,
        orientation_batch_size: int = 16,
        do_result_export: bool = True,
        compute_correlation_table: bool = False,
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
        compute_correlation_table : bool
            If True, track cross-correlation values which surpass the correlation
            table threshold during the search. If False, the `CorrelationTable` will be
            empty. Incurs a small runtime overhead when enabled. Default is False.

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
            self._resolved_backend(),
            compute_correlation_table=compute_correlation_table,
            unpadded_valid_shape=self._unpadded_valid_shape(),
            **core_kwargs,
        )

        # Only populate the results on the first rank
        if torch.distributed.get_rank() == 0:
            self._populate_match_template_result(
                results,
                defocus_values=core_kwargs["defocus_values"],
                euler_angles=core_kwargs["euler_angles"],
                do_result_export=do_result_export,
            )

    def _resolved_backend(self) -> str:
        """Backend to actually run, honoring any zipFFT fallback from padding.

        Returns
        -------
        str
            ``self.computational_config.backend``, unless the padding plan fell back
            away from ``"zipfft"`` because no compiled shape fit the image.
        """
        plan = self._fft_padding_plan
        if plan is None:
            return self.computational_config.backend
        return plan.effective_backend

    def _unpadded_valid_shape(self) -> tuple[int, int] | None:
        """Valid correlation shape to request from the backend, if padding is active.

        Returns
        -------
        Optional[tuple[int, int]]
            The unpadded valid shape, or None when no padding was applied (or when
            the backend kwargs were built on a different rank).
        """
        plan = self._fft_padding_plan
        if plan is None or not plan.is_padded:
            return None
        return plan.unpadded_valid_shape

    def _unpad_backend_results(self, results: dict[str, Any]) -> dict[str, Any]:
        """Crop padded result maps and drop correlation hits in the padded region.

        Parameters
        ----------
        results : dict[str, Any]
            Result dictionary returned by the backend.

        Returns
        -------
        dict[str, Any]
            Results cropped to the unpadded valid region.
        """
        plan = self._fft_padding_plan
        if plan is None or not plan.is_padded:
            return results

        expected_shape = plan.unpadded_valid_shape
        if tuple(results["mip"].shape) == expected_shape:
            return results

        # NOTE: correlation-table rows filtered before the statistics maps cropped
        results = dict(results)
        results["correlation_table"] = filter_correlation_table(
            results["correlation_table"], expected_shape
        )
        for key in self._RESULT_MAP_KEYS:
            results[key] = plan.unpad(results[key])

        return results

    def _populate_match_template_result(
        self,
        results: dict[str, Any],
        defocus_values: torch.Tensor,
        euler_angles: torch.Tensor,
        do_result_export: bool = True,
    ) -> None:
        """Helper function to populate the MatchTemplateResult object post-core call."""
        results = self._unpad_backend_results(results)

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

        # Build a typed CorrelationTable from the processed backend output, looking up
        # per-detection mean/variance from the statistics tensors independently.
        self.match_template_result.correlation_table = (
            CorrelationTable.from_match_template_results(
                processed_correlation_table=results["correlation_table"],
                defocus_values=defocus_values,
                euler_angles=euler_angles,
                correlation_average=results["correlation_mean"],
                correlation_variance_map=results["correlation_variance"],
            )
        )

        # Export the results to disk, if requested
        if do_result_export:
            self.match_template_result.export_results()

    def results_to_dataframe(
        self,
        half_template_width_pos_shift: bool = True,
        exclude_columns: list | None = None,
        locate_peaks_kwargs: dict | None = None,
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
        # Convert dict columns to JSON strings for CSV serialization
        even_zernikes_value = (
            json.dumps(self.optics_group.even_zernikes)
            if self.optics_group.even_zernikes is not None
            else None
        )
        odd_zernikes_value = (
            json.dumps(self.optics_group.odd_zernikes)
            if self.optics_group.odd_zernikes is not None
            else None
        )
        df["even_zernikes"] = [even_zernikes_value] * len(df)
        df["odd_zernikes"] = [odd_zernikes_value] * len(df)
        # Repeat mag_matrix list for each row in the DataFrame
        df["mag_matrix"] = [self.optics_group.mag_matrix] * len(df)

        # Add paths to the micrograph and reference template
        df["micrograph_path"] = self.micrograph_path
        df["template_path"] = self.template_volume_path

        # Add paths to the output statistic files, branching on storage back-end
        if isinstance(self.match_template_result, MatchTemplateResultMRC):
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
        else:
            # HDF5: all tensors are in one file; individual MRC paths are not applicable
            df["mip_path"] = self.match_template_result.hdf5_path
            df["scaled_mip_path"] = self.match_template_result.hdf5_path
            df["psi_path"] = self.match_template_result.hdf5_path
            df["theta_path"] = self.match_template_result.hdf5_path
            df["phi_path"] = self.match_template_result.hdf5_path
            df["defocus_path"] = self.match_template_result.hdf5_path
            df["correlation_average_path"] = self.match_template_result.hdf5_path
            df["correlation_variance_path"] = self.match_template_result.hdf5_path

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
