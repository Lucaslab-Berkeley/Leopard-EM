"""Reading, storing, and exporting results from the match_template program."""

# NOTE: Disabling pylint for too-many-instance-attributes since this class holds a
# number of result attributes that are independent and should not be grouped further.
# pylint: disable=too-many-instance-attributes

import os
from typing import ClassVar

import pandas as pd
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from leopard_em.analysis import (
    MatchTemplatePeaks,
    extract_peaks_and_statistics_zscore,
    match_template_peaks_to_dataframe,
    match_template_peaks_to_dict,
)
from leopard_em.pydantic_models.custom_types import BaseModel2DTM, ExcludedTensor
from leopard_em.utils.data_io import load_mrc_image, write_mrc_from_tensor


def check_file_path_and_permissions(path: str, allow_overwrite: bool) -> None:
    """Ensures path is writable and it does not exist, if `allow_overwrite` is False."""
    # 1. Create path to file, if it does not exist
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # 2. Check write permissions
    if directory and not os.access(directory, os.W_OK):
        raise ValueError(
            f"Directory '{directory}' does not permit writing."
            f"Will be unable to write results to '{path}'."
        )

    # 3. Check if file exists
    if not allow_overwrite and os.path.exists(path):
        raise ValueError(
            f"File '{path}' already exists, but 'allow_file_overwrite' "
            "is False. Set 'allow_file_overwrite' to True to permit. "
            "overwriting.\n"
            "WARNING: Overwriting will delete the existing file(s)!"
        )


class MatchTemplateResult(BaseModel2DTM):
    """Class to hold and export results from the match_template program.

    TODO: Implement tracking of how far along the template matching is
    (e.g. orientations up to what index have been searched).
    TODO: Implement method for exporting intermediary results in case of error
    or program interruption.
    TODO: Implement functionality for restarting template matching from a
    saved state (e.g. after a program interruption).

    Attributes
    ----------
    allow_file_overwrite : bool = False
        Weather to allow overwriting of existing files. Default is False.
        WARNING: Setting to True can overwrite existing files!
    mip_path : str
        Path to the output maximum intensity projection (MIP) file.
    scaled_mip_path : str
        Path to the output scaled MIP file.
    correlation_average_path : str
        Path to the output correlation average file.
    correlation_variance_path : str
        Path to the output correlation variance file.
    orientation_psi_path : str
        Path to the output orientation psi file.
    orientation_theta_path : str
        Path to the output orientation theta file.
    orientation_phi_path : str
        Path to the output orientation phi file.
    relative_defocus_path : str
        Path to the output relative defocus file.
    mip : ExcludedTensor
        Maximum intensity projection (MIP).
    scaled_mip : ExcludedTensor
        Scaled MIP.
    correlation_average : ExcludedTensor
        Correlation average.
    correlation_variance : ExcludedTensor
        Correlation variance.
    orientation_psi : ExcludedTensor
        Best orientation angle psi.
    orientation_theta : ExcludedTensor
        Best orientation angle theta.
    orientation_phi : ExcludedTensor
        Best orientation angle phi.
    relative_defocus : ExcludedTensor
        Best relative defocus.
    total_projections : int, optional
        Total number of cross-correlograms of projections computed. Should be
        'total_orientations x total_defocus' Default is 0, and this field is updated
        automatically after a match_template run.
    total_orientations : int, optional
        Total number of orientations searched. Default is 0, and this field is updated
        automatically after a match_template run.
    total_defocus : int, optional
        Total number of defocus values searched. Default is 0, and this field is updated
        automatically after a match_template run.
    match_template_peaks : MatchTemplatePeaks
        Named tuple object containing the peak locations, heights, and pose statistics.
        See the 'analysis.pick_match_template_peaks' module for more information.

    Methods
    -------
    validate_paths()
        Validates the output paths for write permissions and overwriting.

    load_tensors_from_paths()
        Load tensors from the specified (held) paths into memory.

    locate_peaks(**kwargs)
        Updates the 'match_template_peaks' attribute with info from held tensors.
        Additional keyword arguments can be passed to the 'extract_peaks_and_statistics'
        function.

    peaks_to_dict()
        Convert the 'match_template_peaks' attribute to a dictionary.

    peaks_to_dataframe()
        Convert the 'match_template_peaks' attribute to a pandas DataFrame.

    export_results()
        Export the torch.Tensor results to the specified mrc files.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    # Serialized attributes
    # NOTE: This overwrite attribute is a bit overbearing currently. I predict
    # it will lead to headaches when attempting to load a result, this is set
    # to True, and the result files already exist.
    allow_file_overwrite: bool = False
    mip_path: str
    scaled_mip_path: str
    correlation_average_path: str
    correlation_variance_path: str
    orientation_psi_path: str
    orientation_theta_path: str
    orientation_phi_path: str
    relative_defocus_path: str

    # Scalar (non-tensor) attributes
    total_projections: int = 0
    total_orientations: int = 0
    total_defocus: int = 0

    match_template_peaks: MatchTemplatePeaks = Field(default=None, exclude=True)

    # Large array-like attributes saved to individual files (not in JSON)
    mip: ExcludedTensor
    scaled_mip: ExcludedTensor
    correlation_average: ExcludedTensor
    correlation_variance: ExcludedTensor
    orientation_psi: ExcludedTensor
    orientation_theta: ExcludedTensor
    orientation_phi: ExcludedTensor
    relative_defocus: ExcludedTensor

    ###########################
    ### Pydantic Validators ###
    ###########################

    @model_validator(mode="after")  # type: ignore
    def validate_paths(self) -> Self:
        """Validate output paths for write permissions and overwriting.

        Note: This method runs after instantiation, so attributes are already
        set. We can safely access them with `self`.

        Returns
        -------
        Self
            The validated instance.

        Raises
        ------
        ValueError
            If the output paths are not writable or do not permit overwriting.
        """
        # 1. Check write permissions and overwriting for each path
        paths = [
            self.mip_path,
            self.scaled_mip_path,
            self.correlation_average_path,
            self.correlation_variance_path,
            self.orientation_psi_path,
            self.orientation_theta_path,
            self.orientation_phi_path,
            self.relative_defocus_path,
        ]
        for path in paths:
            check_file_path_and_permissions(path, self.allow_file_overwrite)

        return self

    ############################################
    ### Functional (data processing) methods ###
    ############################################

    def apply_valid_cropping(self, template_shape: tuple[int, int]) -> None:
        """Applies valid mode cropping to the stored tensors in-place.

        Valid mode cropping ensures that positions correspond to where no overlapping
        occurs between the template and edges of the image (i.e. the template fully
        tiles the image in the cross-correlograms). For an image of shape (H, W) and
        template shape of (h, w), this corresponds to cropping out the region
        (H - h + 1, W - w + 1).

        Parameters
        ----------
        template_shape : tuple[int, int]
            Shape of the template used in the match_template run.

        Returns
        -------
        None
        """
        # Assuming all statistic files have the same shape (which should be true!)
        img_h, img_w = self.mip.shape
        h, w = template_shape
        slice_obj = (slice(img_h - h + 1), slice(img_w - w + 1))

        self.mip = self.mip[slice_obj]
        self.scaled_mip = self.scaled_mip[slice_obj]
        self.correlation_average = self.correlation_average[slice_obj]
        self.correlation_variance = self.correlation_variance[slice_obj]
        self.orientation_psi = self.orientation_psi[slice_obj]
        self.orientation_theta = self.orientation_theta[slice_obj]
        self.orientation_phi = self.orientation_phi[slice_obj]
        self.relative_defocus = self.relative_defocus[slice_obj]

    def load_tensors_from_paths(self) -> None:
        """Use the held paths to load tensors into memory.

        NOTE: Currently only supports .mrc files.
        """
        self.mip = load_mrc_image(self.mip_path)
        self.scaled_mip = load_mrc_image(self.scaled_mip_path)
        self.correlation_average = load_mrc_image(self.correlation_average_path)
        self.correlation_variance = load_mrc_image(self.correlation_variance_path)
        self.orientation_psi = load_mrc_image(self.orientation_psi_path)
        self.orientation_theta = load_mrc_image(self.orientation_theta_path)
        self.orientation_phi = load_mrc_image(self.orientation_phi_path)
        self.relative_defocus = load_mrc_image(self.relative_defocus_path)

    def locate_peaks(self, **kwargs) -> MatchTemplatePeaks:  # type: ignore
        """Updates the 'match_template_peaks' attribute with info from held tensors.

        This method calls the `extract_peaks_and_statistics` function to first locate
        particles based on the z-scores of the correlation results, then finds the
        best orientations and defocus values at those locations. Returned named tuple
        object is stored in the 'match_template_peaks' attribute.

        NOTE: Method intended to be called after running match_template or loading
        the tensors from disk.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the 'extract_peaks_and_statistics'
            function.

        Returns
        -------
        MatchTemplatePeaks
            Named tuple object containing the peak locations, heights, and pose
            statistics.
        """
        self.match_template_peaks = extract_peaks_and_statistics_zscore(
            mip=self.mip,
            scaled_mip=self.scaled_mip,
            best_psi=self.orientation_psi,
            best_theta=self.orientation_theta,
            best_phi=self.orientation_phi,
            best_defocus=self.relative_defocus,
            correlation_average=self.correlation_average,
            correlation_variance=self.correlation_variance,
            total_correlation_positions=self.total_projections,
            **kwargs,
        )

        return self.match_template_peaks

    def peaks_to_dict(self) -> dict:
        """Convert the 'match_template_peaks' attribute to a dictionary."""
        if self.match_template_peaks is None:
            self.locate_peaks()

        return match_template_peaks_to_dict(self.match_template_peaks)

    def peaks_to_dataframe(self) -> pd.DataFrame:
        """Convert the 'match_template_peaks' attribute to a pandas DataFrame."""
        if self.match_template_peaks is None:
            self.locate_peaks()

        return match_template_peaks_to_dataframe(self.match_template_peaks)

    ######################
    ### Export methods ###
    ######################

    def export_results(self) -> None:
        """Export the torch.Tensor results to the specified mrc files."""
        paths = [
            self.mip_path,
            self.scaled_mip_path,
            self.correlation_average_path,
            self.correlation_variance_path,
            self.orientation_psi_path,
            self.orientation_theta_path,
            self.orientation_phi_path,
            self.relative_defocus_path,
        ]
        tensors = [
            self.mip,
            self.scaled_mip,
            self.correlation_average,
            self.correlation_variance,
            self.orientation_psi,
            self.orientation_theta,
            self.orientation_phi,
            self.relative_defocus,
        ]

        for path, tensor in zip(paths, tensors):
            write_mrc_from_tensor(
                data=tensor,
                mrc_path=path,
                mrc_header=None,
                overwrite=self.allow_file_overwrite,
            )
