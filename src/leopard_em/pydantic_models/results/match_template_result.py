"""Reading, storing, and exporting results from the match_template program.

Two public classes are provided for different storage back-ends:

* ``MatchTemplateResultMRC`` - stores each result tensor in a separate MRC
  file (the original behavior).  ``MatchTemplateResult`` is an alias for
  this class for backward compatibility.
* ``MatchTemplateResultHDF5`` - bundles every tensor and all scalar metadata
  into a single HDF5 file.

The base class ``_MatchTemplateResultBase`` holds the tensors, scalar
metadata, and all analysis methods and is not intended to be used directly.
"""

# NOTE: Disabling pylint for too-many-instance-attributes since this class holds a
# number of result attributes that are independent and should not be grouped further.
# pylint: disable=too-many-instance-attributes

import os
from importlib.metadata import PackageNotFoundError, version
from typing import ClassVar, Self

import h5py
import pandas as pd
import torch
from pydantic import ConfigDict, Field, model_validator

from leopard_em.analysis import (
    MatchTemplatePeaks,
    extract_peaks_and_statistics_zscore,
    match_template_peaks_to_dataframe,
    match_template_peaks_to_dict,
)
from leopard_em.pydantic_models.custom_types import BaseModel2DTM, ExcludedTensor
from leopard_em.pydantic_models.results.correlation_table import CorrelationTable
from leopard_em.utils.data_io import load_mrc_image, write_mrc_from_tensor


def _leopard_em_version() -> str:
    try:
        return version("leopard_em")
    except PackageNotFoundError:
        return "uninstalled"


_TENSOR_NAMES = (
    "mip",
    "scaled_mip",
    "correlation_average",
    "correlation_variance",
    "orientation_psi",
    "orientation_theta",
    "orientation_phi",
    "relative_defocus",
)

_HDF5_TENSORS_GROUP = "tensors"


def check_file_path_and_permissions(path: str, allow_overwrite: bool) -> None:
    """Ensures path is writable and it does not exist, if `allow_overwrite` is False."""
    # 1. Create path to file, if it does not exist
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # 2. Check write permissions
    if directory and not os.access(directory, os.W_OK):
        raise ValueError(
            f"Directory '{directory}' does not permit writing. "
            f"Will be unable to write results to '{path}'."
        )

    # 3. Check if file exists
    if not allow_overwrite and os.path.exists(path):
        raise ValueError(
            f"File '{path}' already exists, but 'allow_file_overwrite' "
            "is False. Set 'allow_file_overwrite' to True to permit "
            "overwriting.\n"
            "WARNING: Overwriting will delete the existing file(s)!"
        )


class _MatchTemplateResultBase(BaseModel2DTM):
    """Base class holding result tensors, scalar metadata, and analysis methods.

    Not intended to be instantiated directly — use ``MatchTemplateResultMRC``
    or ``MatchTemplateResultHDF5`` depending on the desired storage back-end.

    Attributes
    ----------
    leopard_em_version : str
        Version of Leopard-EM that produced this result.  Auto-populated from
        the installed package metadata on construction; preserved as-recorded
        when loading from a file.
    total_projections : int
        Total cross-correlations computed (orientations x defocus steps).
    total_orientations : int
        Total orientations searched.
    total_defocus : int
        Total defocus values searched.
    mip : ExcludedTensor
        Maximum intensity projection.
    scaled_mip : ExcludedTensor
        Scaled MIP (z-score normalized).
    correlation_average : ExcludedTensor
        Running mean of the correlation over the search space, per pixel.
    correlation_variance : ExcludedTensor
        Running variance of the correlation over the search space, per pixel.
    orientation_psi : ExcludedTensor
        Per-pixel best psi angle (degrees, ZYZ convention).
    orientation_theta : ExcludedTensor
        Per-pixel best theta angle (degrees, ZYZ convention).
    orientation_phi : ExcludedTensor
        Per-pixel best phi angle (degrees, ZYZ convention).
    relative_defocus : ExcludedTensor
        Per-pixel best relative defocus offset (Angstroms).
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    # Serialized attributes
    # NOTE: This overwrite attribute is a bit overbearing currently. I predict
    # it will lead to headaches when attempting to load a result, this is set
    # to True, and the result files already exist.
    allow_file_overwrite: bool = False
    correlation_table_path: str | None = Field(default=None)

    # Scalar (non-tensor) attributes
    leopard_em_version: str = Field(default_factory=_leopard_em_version)
    total_projections: int = 0
    total_orientations: int = 0
    total_defocus: int = 0

    match_template_peaks: MatchTemplatePeaks = Field(default=None, exclude=True)
    correlation_table: CorrelationTable | None = Field(default=None, exclude=True)

    mip: ExcludedTensor
    scaled_mip: ExcludedTensor
    correlation_average: ExcludedTensor
    correlation_variance: ExcludedTensor
    orientation_psi: ExcludedTensor
    orientation_theta: ExcludedTensor
    orientation_phi: ExcludedTensor
    relative_defocus: ExcludedTensor

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
        # NOTE: Assuming all statistic files have the same shape (which should be true).
        # NOTE: Current match_template runs already return valid-cropped tensors from
        #       the backend regardless of correlation backend, so this method is only
        #       needed for manually cropping legacy (pre-v1.3) uncropped result files.
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

    def export_correlation_table(self) -> None:
        """Write the held CorrelationTable to ``self.correlation_table_path``.

        No-op if ``correlation_table_path`` is not set.
        """
        if self.correlation_table is None:
            raise ValueError("No correlation_table to export.")
        if self.correlation_table_path is None:
            return
        self.correlation_table.to_hdf5(self.correlation_table_path)

    def load_correlation_table_from_path(self) -> None:
        """Load CorrelationTable from HDF5 file at ``self.correlation_table_path``."""
        if self.correlation_table_path is None:
            raise ValueError("No correlation_table_path specified to load from.")
        self.correlation_table = CorrelationTable.from_hdf5(self.correlation_table_path)

    def locate_peaks(self, **kwargs) -> MatchTemplatePeaks:  # type: ignore
        """Locate peaks and store results in ``match_template_peaks``.

        Parameters
        ----------
        **kwargs
            Forwarded to ``extract_peaks_and_statistics_zscore``.

        Returns
        -------
        MatchTemplatePeaks
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
        """Convert ``match_template_peaks`` to a dictionary."""
        if self.match_template_peaks is None:
            self.locate_peaks()
        return match_template_peaks_to_dict(self.match_template_peaks)

    def peaks_to_dataframe(self) -> pd.DataFrame:
        """Convert ``match_template_peaks`` to a pandas DataFrame."""
        if self.match_template_peaks is None:
            self.locate_peaks()
        return match_template_peaks_to_dataframe(self.match_template_peaks)


class MatchTemplateResultMRC(_MatchTemplateResultBase):
    """Stores each result tensor in a separate MRC file.

    Attributes
    ----------
    allow_file_overwrite : bool
        Whether to allow overwriting of existing files.  Default is False.
    mip_path : str
        Output path for the maximum intensity projection MRC file.
    scaled_mip_path : str
        Output path for the scaled MIP MRC file.
    correlation_average_path : str
        Output path for the correlation average MRC file.
    correlation_variance_path : str
        Output path for the correlation variance MRC file.
    orientation_psi_path : str
        Output path for the orientation psi MRC file.
    orientation_theta_path : str
        Output path for the orientation theta MRC file.
    orientation_phi_path : str
        Output path for the orientation phi MRC file.
    relative_defocus_path : str
        Output path for the relative defocus MRC file.

    Methods
    -------
    validate_paths()
        Validates write permissions and overwrite policy for all eight paths.
    load_tensors_from_paths()
        Reads MRC files from the held paths into memory.
    export_results()
        Writes the held tensors to their respective MRC paths.
    """

    allow_file_overwrite: bool = False
    mip_path: str
    scaled_mip_path: str
    correlation_average_path: str
    correlation_variance_path: str
    orientation_psi_path: str
    orientation_theta_path: str
    orientation_phi_path: str
    relative_defocus_path: str

    ###########################
    ### Pydantic Validators ###
    ###########################

    @model_validator(mode="after")  # type: ignore
    def validate_paths(self) -> Self:
        """Validate output paths for write permissions and overwriting.

        Returns
        -------
        Self

        Raises
        ------
        ValueError
            If any path is not writable or already exists and overwriting is
            disabled.
        """
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

    ######################
    ### I/O methods    ###
    ######################

    def load_tensors_from_paths(self) -> None:
        """Read MRC files from the held paths into the tensor attributes."""
        self.mip = load_mrc_image(self.mip_path)
        self.scaled_mip = load_mrc_image(self.scaled_mip_path)
        self.correlation_average = load_mrc_image(self.correlation_average_path)
        self.correlation_variance = load_mrc_image(self.correlation_variance_path)
        self.orientation_psi = load_mrc_image(self.orientation_psi_path)
        self.orientation_theta = load_mrc_image(self.orientation_theta_path)
        self.orientation_phi = load_mrc_image(self.orientation_phi_path)
        self.relative_defocus = load_mrc_image(self.relative_defocus_path)

    def export_results(self) -> None:
        """Write the held tensors to their respective MRC paths."""
        pairs = [
            (self.mip_path, self.mip),
            (self.scaled_mip_path, self.scaled_mip),
            (self.correlation_average_path, self.correlation_average),
            (self.correlation_variance_path, self.correlation_variance),
            (self.orientation_psi_path, self.orientation_psi),
            (self.orientation_theta_path, self.orientation_theta),
            (self.orientation_phi_path, self.orientation_phi),
            (self.relative_defocus_path, self.relative_defocus),
        ]
        for path, tensor in pairs:
            write_mrc_from_tensor(
                data=tensor,
                mrc_path=path,
                mrc_header=None,
                overwrite=self.allow_file_overwrite,
            )

        if self.correlation_table_path is not None:
            self.export_correlation_table()


class MatchTemplateResultHDF5(_MatchTemplateResultBase):
    """Bundles all result tensors and metadata into a single HDF5 file.

    HDF5 file layout
    ----------------
    All eight 2-D result tensors are stored as float32 datasets inside a
    ``/tensors`` group.  When ``compress`` is ``True`` (the default) each
    dataset is compressed with gzip at level 4.  Scalar metadata
    (``total_projections``, ``total_orientations``, ``total_defocus``) are
    stored as attributes on the HDF5 root group.  No MRC paths are written to
    the file; the path to the HDF5 file itself is the only path required at
    load time.

        /  (root)
        │  attrs: leopard_em_version, total_projections,
        │         total_orientations, total_defocus
        └─ tensors/
               mip                  float32, shape (H, W), gzip-4 (if compress=True)
               scaled_mip           float32, shape (H, W), gzip-4
               correlation_average  float32, shape (H, W), gzip-4
               correlation_variance float32, shape (H, W), gzip-4
               orientation_psi      float32, shape (H, W), gzip-4
               orientation_theta    float32, shape (H, W), gzip-4
               orientation_phi      float32, shape (H, W), gzip-4
               relative_defocus     float32, shape (H, W), gzip-4

    Attributes
    ----------
    hdf5_path : str
        Path to the HDF5 output file.
    allow_file_overwrite : bool
        Whether to allow overwriting an existing file.  Default is False.
    compress : bool
        Whether to apply gzip-4 compression to tensor datasets.  Default is
        True.  Disable for faster writes at the cost of larger files.

    Methods
    -------
    validate_hdf5_path()
        Validates write permissions and overwrite policy for ``hdf5_path``.
    to_hdf5()
        Writes tensors and metadata to ``hdf5_path``.
    from_hdf5(path, allow_file_overwrite)
        Class method that loads an instance from an existing HDF5 file.
    """

    hdf5_path: str
    allow_file_overwrite: bool = False
    compress: bool = True

    ###########################
    ### Pydantic Validators ###
    ###########################

    @model_validator(mode="after")  # type: ignore
    def validate_hdf5_path(self) -> Self:
        """Validate ``hdf5_path`` for write permissions and overwriting.

        Returns
        -------
        Self

        Raises
        ------
        ValueError
            If the path is not writable or the file already exists and
            overwriting is disabled.
        """
        check_file_path_and_permissions(self.hdf5_path, self.allow_file_overwrite)
        return self

    ######################
    ### I/O methods    ###
    ######################

    def export_results(self) -> None:
        """Write tensors and metadata to ``hdf5_path``, plus the correlation table."""
        self.to_hdf5()
        if self.correlation_table_path is not None:
            self.export_correlation_table()

    def to_hdf5(self) -> None:
        """Write tensors and scalar metadata to ``hdf5_path``.

        Tensors are cast to float32 before writing.  When ``self.compress`` is
        ``True``, each dataset is compressed with gzip at level 4.
        """
        compression_kwargs: dict = (
            {"compression": "gzip", "compression_opts": 4} if self.compress else {}
        )

        with h5py.File(self.hdf5_path, "w") as f:
            f.attrs["leopard_em_version"] = self.leopard_em_version
            f.attrs["total_projections"] = self.total_projections
            f.attrs["total_orientations"] = self.total_orientations
            f.attrs["total_defocus"] = self.total_defocus

            tensors_group = f.create_group(_HDF5_TENSORS_GROUP)
            for name in _TENSOR_NAMES:
                tensor: torch.Tensor | None = getattr(self, name)
                if tensor is not None:
                    tensors_group.create_dataset(
                        name,
                        data=tensor.cpu().to(torch.float32).numpy(),
                        **compression_kwargs,
                    )

    @classmethod
    def from_hdf5(
        cls,
        path: str | os.PathLike,
        allow_file_overwrite: bool = True,
        compress: bool = True,
    ) -> "MatchTemplateResultHDF5":
        """Load a ``MatchTemplateResultHDF5`` from an existing HDF5 file.

        Parameters
        ----------
        path : str | os.PathLike
            Path to the HDF5 file written by ``to_hdf5``.
        allow_file_overwrite : bool
            Passed to the constructor.  Defaults to ``True`` so that the
            model validator does not reject the path of the file being loaded.
        compress : bool
            Passed to the constructor.  Controls compression on any subsequent
            ``to_hdf5`` call made on the returned instance.  Default is ``True``.

        Returns
        -------
        MatchTemplateResultHDF5
        """
        tensors: dict[str, torch.Tensor] = {}

        with h5py.File(path, "r") as f:
            leopard_em_version = str(f.attrs.get("leopard_em_version", "unknown"))
            total_projections = int(f.attrs["total_projections"])
            total_orientations = int(f.attrs["total_orientations"])
            total_defocus = int(f.attrs["total_defocus"])

            if _HDF5_TENSORS_GROUP in f:
                grp = f[_HDF5_TENSORS_GROUP]
                for name in _TENSOR_NAMES:
                    if name in grp:
                        tensors[name] = torch.from_numpy(grp[name][:])

        return cls(
            hdf5_path=str(path),
            allow_file_overwrite=allow_file_overwrite,
            compress=compress,
            leopard_em_version=leopard_em_version,
            total_projections=total_projections,
            total_orientations=total_orientations,
            total_defocus=total_defocus,
            **tensors,
        )


# Backward-compatibility alias — existing code importing MatchTemplateResult
# continues to receive MatchTemplateResultMRC unchanged.
MatchTemplateResult = MatchTemplateResultMRC
