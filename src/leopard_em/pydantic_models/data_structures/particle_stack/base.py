"""Shared base class of the particle stacks: in-memory state and computation.

``_ParticleStackBase`` holds the particle table, the tensor fields and every method
that does not depend on the storage back-end. It hands the actual work to the helper
modules of this package (geometry, table, extraction, movie). It is not intended to be
used directly; see ``ParticleStackCSV`` and ``ParticleStackHDF5``.
"""

import os
import warnings
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import pydantic
import torch
from pydantic import ConfigDict, Field
from torch_motion_correction.deformation_field import DeformationField

import leopard_em
from leopard_em.pydantic_models.config import PreprocessingFilters
from leopard_em.pydantic_models.custom_types import (
    BaseModel2DTM,
    ExcludedTensor,
    ExcludedTensorDict,
)
from leopard_em.pydantic_models.data_structures.particle_stack._extraction import (
    crop_by_group,
    load_grouped_images,
    scatter_by_group,
)
from leopard_em.pydantic_models.data_structures.particle_stack._movie import (
    check_motion_source,
    dose_weight_particle_movies,
    frame_particle_shifts_from_deformation,
    particle_movie_rfft_stack,
)
from leopard_em.pydantic_models.data_structures.particle_stack._table import (
    POSITION_COLUMNS,
    as_row_positions,
    changed_columns,
    choose_refined_columns,
    column_tensor,
    group_rows,
    normalize_particle_dataframe,
    pixel_positions,
    position_columns,
)
from leopard_em.pydantic_models.data_structures.particle_stack.geometry import (
    ParticleBoxGeometry,
)
from leopard_em.pydantic_models.formats import (
    DEFAULT_LOCAL_STAT_COLUMNS,
    PARTICLE_ID_COLUMN,
    local_stat_padding,
)

# Frames skipped when attributing deprecation warnings, so they point at user code
# (e.g. a script or the YAML-loading call site) rather than library internals.
_DEPRECATION_SKIP_PREFIXES = (
    os.path.dirname(os.path.abspath(pydantic.__file__)),
    os.path.dirname(os.path.abspath(leopard_em.__file__)),
)


def _warn_allow_file_overwrite_deprecated() -> None:
    """Emit the DeprecationWarning for the ignored ``allow_file_overwrite`` option."""
    warnings.warn(
        "'allow_file_overwrite' is deprecated and ignored: Leopard-EM does not "
        "guarantee read-only behavior, and loaded files can be overwritten if a 'save' "
        "method is called. 'allow_file_overwrite' will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
        skip_file_prefixes=_DEPRECATION_SKIP_PREFIXES,
    )


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
class _ParticleStackBase(BaseModel2DTM):
    """Base class holding particle stack data, preprocessing state, and compute methods.

    Not intended to be instantiated directly — use ``ParticleStackCSV`` or
    ``ParticleStackHDF5`` depending on the desired storage back-end.

    Attributes
    ----------
    leopard_em_version : str
        Version of Leopard-EM that created this particle stack.  Auto-populated
        from installed package metadata; preserved as-recorded when loading from
        a file.
    extracted_box_size : tuple[int, int]
        Size of extracted particle boxes in pixels (height, width).
    original_template_size : tuple[int, int]
        Size of the template used during template matching (height, width).
        Must be smaller than or equal to ``extracted_box_size``.
    global_whitening_applied : bool
        True if whitening was computed from and applied to the full micrograph
        before particle extraction.
    local_whitening_applied : bool
        True if whitening was computed from and applied to each individual
        extracted particle box.
    global_normalization_applied : bool
        True if normalization was computed from the full micrograph before
        extraction.
    local_normalization_applied : bool
        True if normalization was computed from and applied to each extracted
        particle box.
    image_stack : ExcludedTensor
        Stack of extracted particle images, shape ``(N, box_h, box_w)``.
        Not serialized to YAML/JSON.
    local_stats : ExcludedTensorDict
        Per-particle local statistic maps, keyed by the ``*_path`` DataFrame column they
        were derived from (e.g. ``"mip_path"``, ``"correlation_average_path"``). Each
        value has shape ``(N, valid_h, valid_w)`` where
        ``valid_h = extracted_box_size[0] - original_template_size[0] + 1`` and
        ``valid_w = extracted_box_size[1] - original_template_size[1] + 1``.
        Populated on demand via :meth:`get_local_stat_maps` -- assign its return value
        (or a subset of it) here to make those maps available for
        :meth:`_stored_local_stat_map` lookups and, for ``ParticleStackHDF5``, for
        ``to_hdf5(include_local_stats=True)``.
        Not serialized to YAML/JSON.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    leopard_em_version: str = Field(default_factory=lambda: leopard_em.__version__)
    extracted_box_size: tuple[int, int]
    original_template_size: tuple[int, int]

    # Pre-processing state flags
    global_whitening_applied: bool = False
    local_whitening_applied: bool = False
    global_normalization_applied: bool = False
    local_normalization_applied: bool = False

    # Private: tabular data (not part of Pydantic schema). Always has a 0..N-1
    # RangeIndex so row labels are row positions; assign via ``_set_dataframe``.
    # TODO: Move away from having a df-backed implementation in favor of either
    #       getter/setter methods OR private fields for the relevant data.
    _df: pd.DataFrame

    # Private: particle image stack as read from (or last written to) the backing
    # file. Unlike the public ``image_stack`` scratch field, library methods never
    # overwrite it; see ``get_stored_image_stack``.
    _stored_image_stack: torch.Tensor | None = None

    # Image and statistics tensors (excluded from YAML/JSON serialization)
    image_stack: ExcludedTensor
    local_stats: ExcludedTensorDict

    def __init__(self, skip_df_load: bool = False, **data: Any):
        """Initialize the particle stack.

        Parameters
        ----------
        skip_df_load : bool, optional
            When True the subclass ``load_df`` is not called automatically.
            Use this when constructing an empty instance before populating
            ``_df`` manually (e.g., during ``from_hdf5``).
        data : dict[str, Any]
            Fields forwarded to the Pydantic constructor.
        """
        super().__init__(**data)
        if not skip_df_load:
            self.load_df()

    def load_df(self) -> None:
        """Load the particle DataFrame from the backing store.

        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement load_df()")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def df_columns(self) -> list[str]:
        """Column names of the underlying DataFrame."""
        return list(self._df.columns.tolist())

    @property
    def num_particles(self) -> int:
        """Number of particles in the stack."""
        return len(self._df)

    @property
    def position_columns(self) -> tuple[str, str]:
        """The ``(y, x)`` position columns in use (``refined_pos_*`` when usable)."""
        return position_columns(self._df)

    @property
    def geometry(self) -> ParticleBoxGeometry:
        """Crop geometry implied by the extracted box and original template sizes."""
        return ParticleBoxGeometry(
            extracted_box_size=self.extracted_box_size,
            original_template_size=self.original_template_size,
        )

    # ------------------------------------------------------------------
    # DataFrame accessor / mutator helpers
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        """Get a column from the underlying DataFrame."""
        try:
            return self._df[key]
        except KeyError as err:
            raise KeyError(f"Key '{key}' not found in underlying DataFrame.") from err

    def set_column(self, column_name: str, value: Any) -> None:
        """Set a column in the underlying DataFrame.

        Parameters
        ----------
        column_name : str
            The column name to set.
        value : Any
            The value(s) to assign. A scalar is broadcast to every particle. A
            ``pd.Series`` is assigned by position (its index is ignored).
        """
        if isinstance(value, pd.Series):
            value = value.to_numpy()
        self._df[column_name] = value
        self._invalidate_derived_tensors(column_name)

    def _invalidate_derived_tensors(self, column_name: str) -> None:
        """Drop stored per-particle tensors that depended on ``column_name``."""
        if column_name in POSITION_COLUMNS:
            self._stored_image_stack = None
            self.image_stack = None
            self.local_stats = {}
        elif column_name == "micrograph_path":
            self._stored_image_stack = None
            self.image_stack = None
        else:
            self.local_stats.pop(column_name, None)

    def get_stored_image_stack(self) -> torch.Tensor | None:
        """Return the particle image stack stored in the backing file, if usable.

        Returns
        -------
        torch.Tensor | None
            The ``(N, box_h, box_w)`` stored image stack, or None if there is none or
            it no longer matches the particle table / ``extracted_box_size``.

        Raises
        ------
        NotImplementedError
            If the stored images were pre-processed (any ``*_whitening_applied`` or
            ``*_normalization_applied`` flag set), which programs do not support yet.
        """
        stored = self._stored_image_stack
        if stored is None:
            return None

        expected_shape = (self.num_particles, *self.extracted_box_size)
        if tuple(stored.shape) != expected_shape:
            warnings.warn(
                f"Ignoring the stored image stack: its shape {tuple(stored.shape)} "
                f"does not match (num_particles, *extracted_box_size) = "
                f"{expected_shape}.",
                UserWarning,
                stacklevel=2,
            )
            return None

        if (
            self.global_whitening_applied
            or self.local_whitening_applied
            or self.global_normalization_applied
            or self.local_normalization_applied
        ):
            raise NotImplementedError(
                "Using a stored image stack whose images were already whitened or "
                "normalized is not supported."
            )
        return stored

    def _set_dataframe(
        self,
        df: pd.DataFrame,
        on_invalid_ids: Literal["raise", "regenerate"] = "raise",
    ) -> None:
        """Replace the underlying DataFrame, normalizing it to the canonical layout.

        Notes
        -----
        The particle images and other tensors stored are coupled to the original rows
        they correspond to. For example, the positions of the particles within the
        image. When the DataFrame is replaced, any stored tensors that depend on changed
        rows or columns are dropped to ensure present data remains valid.
        """
        new_df = normalize_particle_dataframe(df, on_invalid_ids=on_invalid_ids)
        old_df = getattr(self, "_df", None)
        self._df = new_df
        if old_df is None:
            return

        dependencies = [
            PARTICLE_ID_COLUMN,
            *sorted(POSITION_COLUMNS),
            "micrograph_path",
            *self.local_stats,
        ]
        for column in changed_columns(old_df, new_df, dependencies):
            # A reordered or resized table invalidates tensors like moved positions
            self._invalidate_derived_tensors(
                "pos_y" if column == PARTICLE_ID_COLUMN else column
            )

    def _as_positions(self, indexes: Any) -> np.ndarray:
        """Validate and convert particle indexes to 0-based row positions.

        See :func:`_table.as_row_positions`.
        """
        return as_row_positions(indexes, self.num_particles)

    def get_dataframe_copy(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        return self._df.copy()

    # ------------------------------------------------------------------
    # CTF / orientation accessors
    # ------------------------------------------------------------------

    def get_relative_defocus(
        self,
        prefer_refined_defocus: bool = True,
    ) -> torch.Tensor:
        """Get the relative defocus values for each particle.

        Parameters
        ----------
        prefer_refined_defocus : bool, optional
            Whether to use the refined defocus values, by default True.

        Returns
        -------
        torch.Tensor
        """
        (column,) = choose_refined_columns(
            self._df,
            ["relative_defocus"],
            prefer_refined_defocus,
            refined_name="defocus values",
            original_name="defocus values",
        )
        return column_tensor(self._df, column)

    def get_absolute_defocus(
        self, prefer_refined_defocus: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the absolute defocus (u, v) values for each particle.

        Parameters
        ----------
        prefer_refined_defocus : bool, optional
            Whether to use refined defocus, by default True.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(defocus_u, defocus_v)`` tensors in Angstroms.
        """
        particle_defocus = self.get_relative_defocus(prefer_refined_defocus)
        defocus_u = column_tensor(self._df, "defocus_u")
        defocus_v = column_tensor(self._df, "defocus_v")
        defocus_u = defocus_u + particle_defocus
        defocus_v = defocus_v + particle_defocus
        return defocus_u, defocus_v

    def get_pixel_size(
        self,
        prefer_refined_pixel_size: bool = True,
    ) -> torch.Tensor:
        """Get the pixel size for each particle.

        Parameters
        ----------
        prefer_refined_pixel_size : bool, optional
            Whether to use the refined pixel size, by default True.

        Returns
        -------
        torch.Tensor
        """
        (column,) = choose_refined_columns(
            self._df,
            ["pixel_size"],
            prefer_refined_pixel_size,
            refined_name="pixel size",
            original_name="pixel size values",
        )
        return column_tensor(self._df, column)

    def get_euler_angles(self, prefer_refined_angles: bool = True) -> torch.Tensor:
        """Return the Euler angles (phi, theta, psi) of all particles as a tensor.

        Parameters
        ----------
        prefer_refined_angles : bool, optional
            When true, refined angles are used if all three are present and finite,
            by default True.

        Returns
        -------
        torch.Tensor
            Shape ``(N, 3)`` — columns correspond to (phi, theta, psi) in ZYZ.
        """
        columns = choose_refined_columns(
            self._df,
            ["phi", "theta", "psi"],
            prefer_refined_angles,
            refined_name="angles",
            original_name="angles",
        )
        phi, theta, psi = (column_tensor(self._df, column) for column in columns)

        return torch.stack((phi, theta, psi), dim=-1)

    # ------------------------------------------------------------------
    # Image-stack construction
    # ------------------------------------------------------------------

    def load_images_grouped_by_column(
        self, column_name: str
    ) -> tuple[torch.Tensor, list[pd.Index]]:
        """Load images grouped by a column and return images as a tensor with indexes.

        Notes
        -----
        Statistic-map columns (e.g. ``mip_path``, ``psi_path``) may reference standalone
        MRC files or a single HDF5 file bundling several maps as named datasets (the
        ``MatchTemplateResultHDF5`` particle stack file format).

        Parameters
        ----------
        column_name : str
            The column name to group by (e.g., "micrograph_path" or "mip_path").

        Returns
        -------
        tuple[torch.Tensor, list[pd.Index]]
            A tuple containing:
            - A tensor of loaded images with shape (N, H, W) where N is the number of
              unique images and (H, W) is the image size
            - A list of pandas Index objects containing the 0-based row positions of
              the particles from each corresponding image

        Raises
        ------
        ValueError
            If the column is missing or is empty for any particle.
        """
        return load_grouped_images(group_rows(self._df, column_name), column_name)

    def get_local_stat_maps(
        self,
        columns: list[str] | None = None,
        device: torch.device | str = "cpu",
        valid_size: tuple[int, int] | None = None,
        padding_value: float | None = None,
        use_stored: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Extract per-particle local sub-images for arbitrary result-map columns.

        Generalizes the correlation-statistics extraction to any full-micrograph result
        map exposed as a ``*_path`` column. Returned regions are cropped to the valid
        cross-correlation region based on extracted box size and original template size,
        unless ``valid_size`` is explicitly provided.

        Parameters
        ----------
        columns : list[str] | None
            Path columns to extract. Defaults to the six standard 2DTM statistic
            maps (``mip``, ``scaled_mip``, ``psi``, ``theta``, ``phi``, ``defocus``).
        device : torch.device | str
            Target device for the returned tensors. Defaults to ``"cpu"``.
        valid_size : tuple[int, int] | None
            Extraction size ``(height, width)``. Defaults to the valid
            cross-correlation region
            ``extracted_box_size - original_template_size + 1``.
        padding_value : float | None
            Constant pad value for out-of-bounds regions. Defaults to None, which pads
            ``correlation_variance_path`` (a standard deviation map) with a large value
            and every other column with ``0.0``.
        use_stored : bool
            Use maps already held in :attr:`local_stats` (e.g. loaded from an HDF5
            stack) when their shape matches. Defaults to True.

        Returns
        -------
        dict[str, torch.Tensor]
            Maps each requested column name to its
            ``(num_particles, valid_h, valid_w)`` sub-image stack on ``device``.
        """
        if columns is None:
            columns = list(DEFAULT_LOCAL_STAT_COLUMNS)

        if valid_size is None:
            valid_size = self.geometry.valid_size
        expected_shape = (self.num_particles, *valid_size)

        device = torch.device(device)

        stat_maps: dict[str, torch.Tensor] = {}
        for column in columns:
            stored = self.local_stats.get(column) if use_stored else None
            if stored is not None and tuple(stored.shape) == expected_shape:
                stat_maps[column] = stored.to(device)
                continue

            images, indices = self.load_images_grouped_by_column(column)
            stat_maps[column] = self.crop_particle_regions(
                images=images,
                indices=indices,
                extraction_size=valid_size,
                pos_reference="top-left",
                handle_bounds="pad",
                padding_mode="constant",
                padding_value=local_stat_padding(column, padding_value),
            ).to(device)

        return stat_maps

    def crop_particle_regions(
        self,
        images: torch.Tensor,
        indices: list[pd.Index],
        extraction_size: tuple[int, int],
        pos_reference: Literal["center", "top-left"] = "top-left",
        handle_bounds: Literal["pad", "error"] = "pad",
        padding_mode: Literal["constant", "reflect", "replicate"] = "constant",
        padding_value: float = 0.0,
    ) -> torch.Tensor:
        """Crop a per-particle sub-image from each source image.

        Same as :meth:`construct_image_stack` but without storing the result in
        :attr:`image_stack`, e.g. for cropping statistic maps.

        This method preferentially selects refined position columns by default
        (refined_pos_x, refined_pos_y) if they are present in the DataFrame, falling
        back to unrefined positions (pos_x, pos_y) otherwise.

        Each crop starts at the origin given by ``ParticleBoxGeometry.box_origins``:
        the template footprint located by the position columns, shifted up/left by
        ``(extracted_box_size - original_template_size) // 2``. Cropping
        ``extracted_box_size`` from there gives the particle image, and cropping
        ``valid_size`` gives the matching valid cross-correlation region of a
        statistic map (see :mod:`.geometry`).

        Parameters
        ----------
        images : torch.Tensor
            A tensor of loaded images with shape (N, H, W).
        indices : list[pd.Index]
            Row indexes for particles from each corresponding image.
        extraction_size : tuple[int, int]
            Size of the extracted boxes in pixels (height, width).
        pos_reference : Literal["center", "top-left"], optional
            Reference point for the positions, by default "top-left".
        handle_bounds : Literal["pad", "error"], optional
            How to handle out-of-bounds regions, by default "pad".
        padding_mode : Literal["constant", "reflect", "replicate"], optional
            Padding mode when ``handle_bounds="pad"``, by default "constant".
        padding_value : float, optional
            Constant padding value, by default 0.0.

        Returns
        -------
        torch.Tensor
            Stack of extracted images ``(N, extraction_h, extraction_w)``.
        """
        origin_y, origin_x = self.geometry.box_origins(
            *pixel_positions(self._df), pos_reference
        )
        return crop_by_group(
            images,
            indices,
            origin_y,
            origin_x,
            extraction_size,
            handle_bounds=handle_bounds,
            padding_mode=padding_mode,
            padding_value=padding_value,
        )

    def construct_image_stack(
        self,
        images: torch.Tensor,
        indices: list[pd.Index],
        extraction_size: tuple[int, int],
        pos_reference: Literal["center", "top-left"] = "top-left",
        handle_bounds: Literal["pad", "error"] = "pad",
        padding_mode: Literal["constant", "reflect", "replicate"] = "constant",
        padding_value: float = 0.0,
    ) -> torch.Tensor:
        """Construct stack of particle images from the DataFrame.

        Parameters
        ----------
        images : torch.Tensor
            A tensor of loaded images with shape (N, H, W).
        indices : list[pd.Index]
            Row indexes for particles from each corresponding image.
        extraction_size : tuple[int, int]
            Size of the extracted boxes in pixels (height, width).
        pos_reference : Literal["center", "top-left"], optional
            Reference point for the positions, by default "top-left".
        handle_bounds : Literal["pad", "error"], optional
            How to handle out-of-bounds regions, by default "pad".
        padding_mode : Literal["constant", "reflect", "replicate"], optional
            Padding mode when ``handle_bounds="pad"``, by default "constant".
        padding_value : float, optional
            Constant padding value, by default 0.0.

        Returns
        -------
        torch.Tensor
            Stack of extracted images ``(N, extraction_h, extraction_w)``.
        """
        image_stack = self.crop_particle_regions(
            images=images,
            indices=indices,
            extraction_size=extraction_size,
            pos_reference=pos_reference,
            handle_bounds=handle_bounds,
            padding_mode=padding_mode,
            padding_value=padding_value,
        )
        self.image_stack = image_stack

        return image_stack

    def construct_image_filters(
        self,
        preprocess_filters: PreprocessingFilters,
        output_shape: tuple[int, int],
        images_dft: torch.Tensor,
    ) -> torch.Tensor:
        """Get stack of Fourier filters from filter config and reference images.

        Note that here the filters are assumed to be applied globally (i.e. no local
        whitening, etc. is being done). Whitening filters are calculated with reference
        to each image (micrograph or particle).

        Parameters
        ----------
        preprocess_filters : PreprocessingFilters
            Configuration object of filters to apply.
        output_shape : tuple[int, int]
            What shape along the last two dimensions the filters should be.
        images_dft : torch.Tensor
            A tensor of images with shape (N, H, W) where N is the number of images
            (micrographs or particles) and (H, W) is the image size. in Fourier space.

        Returns
        -------
        torch.Tensor
            The stack of filters with shape (N, h, w) where N is the number of images
            and (h, w) is the output shape.
        """
        return preprocess_filters.get_combined_filter_stack(images_dft, output_shape)

    def construct_projective_filters(
        self,
        preprocess_filters: PreprocessingFilters,
        output_shape: tuple[int, int],
        images_dft: torch.Tensor,
        indices: list[pd.Index],
    ) -> torch.Tensor:
        """Get stack of Fourier filters from filter config and reference micrographs.

        Note that here the filters are assumed to be applied globally (i.e. no local
        whitening, etc. is being done). Whitening filters are calculated with reference
        to each original micrograph in the DataFrame.

        Parameters
        ----------
        preprocess_filters : PreprocessingFilters
            Configuration object of filters to apply.
        output_shape : tuple[int, int]
            What shape along the last two dimensions the filters should be.
        images_dft : torch.Tensor
            A tensor of micrograph images in Fourier space with shape (N, H, W).
        indices : list[pd.Index]
            Row positions of the particles from each corresponding micrograph.

        Returns
        -------
        torch.Tensor
            Filter stack of shape ``(M, h, w)`` where M is the number of particles.
        """
        image_filters = preprocess_filters.get_combined_filter_stack(
            images_dft, output_shape
        )
        return scatter_by_group(image_filters, indices, self.num_particles)

    def compute_frame_particle_shifts_from_deformation(
        self,
        movie_frame: torch.Tensor,
        deformation_field: DeformationField,
        normalized_t_value: torch.Tensor,
        pixel_grid: torch.Tensor,
        pixel_spacing: float,
        pos_y_center: torch.Tensor,
        pos_x_center: torch.Tensor,
        gh: int,
        gw: int,
    ) -> torch.Tensor:
        """Compute per-particle shifts for a single frame from a deformation field.

        Parameters
        ----------
        movie_frame : torch.Tensor
            Single movie frame (H, W).
        deformation_field : CubicCatmullRomGrid3d
            The deformation field grid.
        normalized_t_value : torch.Tensor
            Normalized time value for the frame.
        pixel_grid : torch.Tensor
            The pixel grid tensor.
        pixel_spacing : float
            The pixel spacing.
        pos_y_center : torch.Tensor
            Center y positions.
        pos_x_center : torch.Tensor
            Center x positions.
        gh : int
            Height of the deformation field grid.
        gw : int
            Width of the deformation field grid.

        Returns
        -------
        torch.Tensor
            Shifts with shape ``(N, 2)`` as (dy, dx).
        """
        return frame_particle_shifts_from_deformation(
            movie_frame=movie_frame,
            deformation_field=deformation_field,
            normalized_t_value=normalized_t_value,
            pixel_grid=pixel_grid,
            pixel_spacing=pixel_spacing,
            pos_y_center=pos_y_center,
            pos_x_center=pos_x_center,
            gh=gh,
            gw=gw,
        )

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    # pylint: disable=too-many-locals
    def _construct_particle_movie_rfft_stack(
        self,
        movie: torch.Tensor,
        deformation_field: DeformationField | None = None,
        particle_shifts: torch.Tensor | None = None,
        pos_reference: Literal["center", "top-left"] = "top-left",
        handle_bounds: Literal["pad", "error"] = "pad",
        padding_mode: Literal["constant", "reflect", "replicate"] = "constant",
        padding_value: float = 0.0,
        use_gradient_checkpointing: bool = True,
        particle_indices: list[int] | None = None,
        require_motion_source: bool = True,
        normalized_t_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[int]]:
        """Construct per-particle movie frame DFTs after optional motion shifts.

        Parameters
        ----------
        movie : torch.Tensor
            The movie tensor.
        deformation_field : DeformationField | None, optional
            The deformation field grid.
        particle_shifts : torch.Tensor | None, optional
            Per-particle shifts, shape ``(T, N, 2)``.  Exactly one of
            ``deformation_field`` and ``particle_shifts`` must be provided.
        pos_reference : Literal["center", "top-left"], optional
            Position reference for extraction, by default "top-left".
        handle_bounds : Literal["pad", "error"], optional
            How to handle out-of-bounds regions, by default "pad".
        padding_mode : Literal["constant", "reflect", "replicate"], optional
            Padding mode, by default "constant".
        padding_value : float, optional
            Constant padding value, by default 0.0.
        pre_exposure : float, optional
            Pre-exposure in electrons per pixel, by default 0.0.
        fluence_per_frame : float, optional
            Dose per frame in electrons per pixel, by default 0.0.
        use_gradient_checkpointing : bool, optional
            Trade compute for memory during frame processing, by default True.
        particle_indices : list[int] | None, optional
            Subset of particles to process.  If None, all particles are used.
        require_motion_source : bool, optional
            If True, raises an error if neither ``deformation_field`` nor
            ``particle_shifts`` is provided.  If False, assumes the movie is already
            aligned and extracts each frame without shifts.
        normalized_t_values : torch.Tensor | None, optional
            Normalized time values for each frame, shape ``(t,)``.  If None, a linear
            ramp from 0 to 1 is constructed and used.

        Returns
        -------
        tuple[torch.Tensor, list[int]]
            Per-particle movie frame DFTs of shape ``(N, t, box_h, box_w // 2 + 1)``
            and the 0-based row position of each processed particle.
        """
        check_motion_source(deformation_field, particle_shifts, require_motion_source)
        pixel_sizes = self.get_pixel_size()
        if particle_indices is not None:
            # Use provided subset of particles (0-based row positions)
            positions = self._as_positions(particle_indices)
            if particle_shifts is not None:
                particle_shifts = particle_shifts[:, torch.as_tensor(positions)]
        else:
            # Use all particles
            positions = np.arange(self.num_particles, dtype=np.int64)

        all_pos_y, all_pos_x = pixel_positions(self._df)
        pos_y, pos_x = all_pos_y[positions], all_pos_x[positions]
        # Motion is sampled at the template center; crops start at the box origin
        geometry = self.geometry
        center_y, center_x = geometry.template_centers(pos_y, pos_x, pos_reference)
        origin_y, origin_x = geometry.box_origins(pos_y, pos_x, pos_reference)

        aligned_particle_movies_rfft = particle_movie_rfft_stack(
            movie,
            origins=(torch.tensor(origin_y), torch.tensor(origin_x)),
            centers=(torch.tensor(center_y), torch.tensor(center_x)),
            extracted_box_size=self.extracted_box_size,
            pixel_sizes=pixel_sizes,
            deformation_field=deformation_field,
            particle_shifts=particle_shifts,
            normalized_t_values=normalized_t_values,
            handle_bounds=handle_bounds,
            padding_mode=padding_mode,
            padding_value=padding_value,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        return aligned_particle_movies_rfft, positions.tolist()

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    def construct_particle_movie_stack(
        self,
        movie: torch.Tensor,
        deformation_field: DeformationField | None = None,
        particle_shifts: torch.Tensor | None = None,
        pos_reference: Literal["center", "top-left"] = "top-left",
        handle_bounds: Literal["pad", "error"] = "pad",
        padding_mode: Literal["constant", "reflect", "replicate"] = "constant",
        padding_value: float = 0.0,
        use_gradient_checkpointing: bool = True,
        particle_indices: list[int] | None = None,
        normalized_t_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Construct per-frame particle images from a movie without dose summing.

        If neither ``deformation_field`` nor ``particle_shifts`` is provided, the
        movie is assumed to already be aligned and frames are extracted directly.

        Returns
        -------
        torch.Tensor
            Real-space particle movie stack with shape ``(T, N, H, W)``.
        """
        particle_movie_rfft, _ = self._construct_particle_movie_rfft_stack(
            movie=movie,
            deformation_field=deformation_field,
            particle_shifts=particle_shifts,
            pos_reference=pos_reference,
            handle_bounds=handle_bounds,
            padding_mode=padding_mode,
            padding_value=padding_value,
            use_gradient_checkpointing=use_gradient_checkpointing,
            particle_indices=particle_indices,
            require_motion_source=False,
            normalized_t_values=normalized_t_values,
        )
        particle_movie = torch.fft.irfftn(  # pylint: disable=not-callable
            particle_movie_rfft,
            s=self.extracted_box_size,
            dim=(-2, -1),
        )
        return particle_movie.permute(1, 0, 2, 3).contiguous()

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    def construct_image_stack_from_movie(
        self,
        movie: torch.Tensor,
        deformation_field: DeformationField | None = None,
        particle_shifts: torch.Tensor | None = None,
        pos_reference: Literal["center", "top-left"] = "top-left",
        handle_bounds: Literal["pad", "error"] = "pad",
        padding_mode: Literal["constant", "reflect", "replicate"] = "constant",
        padding_value: float = 0.0,
        pre_exposure: float = 0.0,
        fluence_per_frame: float = 0.0,
        use_gradient_checkpointing: bool = True,
        particle_indices: list[int] | None = None,
    ) -> torch.Tensor:
        """Construct a dose-weighted particle image stack from a movie file.

        Returns
        -------
        torch.Tensor
            The stack of images with shape (N, H, W) where N is the number of
            particles and (H, W) is the extracted box size.
        """
        pixel_sizes = self.get_pixel_size()
        aligned_particle_movies_rfft, particle_positions = (
            self._construct_particle_movie_rfft_stack(
                movie=movie,
                deformation_field=deformation_field,
                particle_shifts=particle_shifts,
                pos_reference=pos_reference,
                handle_bounds=handle_bounds,
                padding_mode=padding_mode,
                padding_value=padding_value,
                use_gradient_checkpointing=use_gradient_checkpointing,
                particle_indices=particle_indices,
                require_motion_source=True,
            )
        )
        aligned_particle_images = dose_weight_particle_movies(
            aligned_particle_movies_rfft,
            extracted_box_size=self.extracted_box_size,
            rows=particle_positions,
            pixel_sizes=pixel_sizes,
            voltages=self._df["voltage"].to_numpy(),
            pre_exposure=pre_exposure,
            fluence_per_frame=fluence_per_frame,
        )

        if particle_indices is None:
            self.image_stack = aligned_particle_images
        return aligned_particle_images
