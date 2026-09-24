"""Particle stack Pydantic model for dealing with extracted particle data.

Two public classes are provided for different storage back-ends:

* ``ParticleStackCSV`` - the original behavior, loading particle data from a
  CSV file and micrograph images from referenced paths on disk.
  ``ParticleStack`` is an alias for this class for backward compatibility.
* ``ParticleStackHDF5`` - stores the particle table, optional image stack, and
  optional per-particle local correlation statistics in a single HDF5 file.

The base class ``_ParticleStackBase`` holds all shared computation methods and
tensor fields.  It is not intended to be used directly.
"""

# TODO: Move these into two separate files (long file)

# pylint: disable=too-many-lines

import hashlib
import os
import warnings
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd
import pydantic
import torch
from pydantic import ConfigDict, Discriminator, Field, Tag
from pydantic.json_schema import SkipJsonSchema
from torch.utils.checkpoint import checkpoint
from torch_fourier_shift import fourier_shift_dft_2d
from torch_grid_utils import coordinate_grid
from torch_motion_correction.correct_motion import get_pixel_shifts
from torch_motion_correction.deformation_field import DeformationField

import leopard_em
from leopard_em.pydantic_models.config import PreprocessingFilters
from leopard_em.pydantic_models.custom_types import (
    BaseModel2DTM,
    ExcludedTensor,
    ExcludedTensorDict,
)
from leopard_em.pydantic_models.data_structures._particle_stack_hdf5_io import (
    BOX_SIZE_ATTRS,
    PREPROCESSING_FLAG_ATTRS,
    ParticleStackFileContents,
    StoredTensor,
    read_particle_stack_hdf5,
    write_particle_stack_hdf5,
)
from leopard_em.pydantic_models.formats import (
    MATCH_TEMPLATE_DF_COLUMN_ORDER,
    PARTICLE_ID_COLUMN,
    STATISTIC_MAP_PATH_COLUMNS,
    STATISTIC_MAP_PATH_TO_HDF5_DATASET,
)
from leopard_em.utils.data_io import atomic_write_path, load_result_map_image
from leopard_em.utils.image_processing import dose_weight_movie_to_micrograph

TORCH_TO_NUMPY_PADDING_MODE = {
    "constant": "constant",
    "reflect": "reflect",
    "replicate": "edge",
}

# Full-micrograph 2DTM result maps extracted per-particle by
# ``get_local_stat_maps`` when no explicit columns are requested.
_DEFAULT_LOCAL_STAT_COLUMNS = tuple(STATISTIC_MAP_PATH_COLUMNS)

# Constant used to pad out-of-bounds regions of per-particle local statistic maps.
# The correlation standard deviation map padded with large values so out-of-bounds
# regions produce ~0 z-scores as opposed to NaN or inf values.
_LARGE_STD_PADDING = 1e10
_DEFAULT_LOCAL_STAT_PADDING = {"correlation_variance_path": _LARGE_STD_PADDING}

# Columns holding particle positions which if changed invalidate per-particle tensors
_POSITION_COLUMNS = frozenset({"pos_x", "pos_y", "refined_pos_x", "refined_pos_y"})

# Frames skipped when attributing deprecation warnings, so they point at user code
# (e.g. a script or the YAML-loading call site) rather than library internals.
_DEPRECATION_SKIP_PREFIXES = (
    os.path.dirname(os.path.abspath(pydantic.__file__)),
    os.path.dirname(os.path.abspath(leopard_em.__file__)),
)


# TODO: Make this a shared utility function across the package somehow
def _leopard_em_version() -> str:
    try:
        return version("leopard_em")
    except PackageNotFoundError:
        return "uninstalled"


def _any_nan_or_inf(s: pd.Series) -> bool:
    """Helper function to check if any value in the Series is NaN or infinite."""
    return bool(s.isna().any() or s.isin([float("inf"), float("-inf")]).any())


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


def _check_required_columns(df: pd.DataFrame, source: str) -> None:
    """Raise if ``df`` lacks any column required of a particle table."""
    missing_columns = [
        col for col in MATCH_TEMPLATE_DF_COLUMN_ORDER if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing the following columns in DataFrame from {source}: "
            f"{missing_columns}"
        )


def _generate_particle_ids(df: pd.DataFrame) -> list[str]:
    """Generate unique particle IDs of the form ``{mic_stem}_{local_idx:05d}``.

    Raises
    ------
    ValueError
        If ``micrograph_path`` is missing or contains empty values.
    """
    if "micrograph_path" not in df.columns:
        raise ValueError("Cannot generate particle IDs without 'micrograph_path'.")
    if df["micrograph_path"].isna().any():
        raise ValueError(
            "Cannot generate particle IDs: 'micrograph_path' is empty for "
            f"{int(df['micrograph_path'].isna().sum())} particle(s)."
        )

    groups = df.groupby("micrograph_path", sort=False).indices
    stems = {path: Path(str(path)).stem for path in groups}
    stem_counts = Counter(stems.values())

    ids = np.empty(len(df), dtype=object)
    for path, positions in groups.items():
        stem = stems[path]
        if stem_counts[stem] > 1:
            digest = hashlib.blake2b(str(path).encode(), digest_size=4).hexdigest()
            stem = f"{stem}-{digest}"
        for local_idx, position in enumerate(positions):
            ids[position] = f"{stem}_{local_idx:05d}"

    res: list[str] = ids.tolist()
    return res


def _normalize_particle_dataframe(
    df: pd.DataFrame,
    on_invalid_ids: Literal["raise", "regenerate"] = "raise",
) -> pd.DataFrame:
    """Return a copy of ``df`` in the canonical in-memory particle-table layout.

    Parameters
    ----------
    df : pd.DataFrame
        The particle table.
    on_invalid_ids : Literal["raise", "regenerate"]
        What to do when a ``particle_id`` column is present but contains duplicate or
        empty values. "regenerate" replaces it (with a warning) and is used for legacy
        inputs; "raise" is used everywhere else.

    Returns
    -------
    pd.DataFrame
    """
    if df.index.name == PARTICLE_ID_COLUMN and PARTICLE_ID_COLUMN not in df.columns:
        df = df.reset_index()
    else:
        df = df.reset_index(drop=True)

    if PARTICLE_ID_COLUMN not in df.columns:
        return df

    ids = df[PARTICLE_ID_COLUMN]
    num_empty = int(ids.isna().sum() + (ids.astype(str) == "").sum())
    duplicated = ids[ids.duplicated(keep=False)].unique().tolist()
    if not num_empty and not duplicated:
        return df

    problem = (
        f"{num_empty} empty value(s)"
        if not duplicated
        else f"duplicate value(s) (e.g. {duplicated[:5]})"
    )
    if on_invalid_ids == "raise":
        raise ValueError(f"Column '{PARTICLE_ID_COLUMN}' contains {problem}.")

    warnings.warn(
        f"Column '{PARTICLE_ID_COLUMN}' contains {problem}; regenerating particle IDs.",
        UserWarning,
        stacklevel=3,
    )
    df[PARTICLE_ID_COLUMN] = _generate_particle_ids(df)
    return df


def _to_pixel_positions(values: np.ndarray) -> np.ndarray:
    """Convert (possibly float-typed) pixel coordinates to int64."""
    return np.rint(np.asarray(values, dtype=np.float64)).astype(np.int64)


# ---------------------------------------------------------------------------
# Stand-alone image-extraction helpers (unchanged from original module)
# ---------------------------------------------------------------------------


def get_cropped_image_regions(
    image: torch.Tensor | np.ndarray,
    pos_y: torch.Tensor | np.ndarray,
    pos_x: torch.Tensor | np.ndarray,
    box_size: int | tuple[int, int],
    pos_reference: Literal["center", "top-left"] = "top-left",
    handle_bounds: Literal["pad", "error"] = "pad",
    padding_mode: Literal["constant", "reflect", "replicate"] = "constant",
    padding_value: float = 0.0,
) -> torch.Tensor | np.ndarray:
    """Extracts regions from an image into a stack of cropped images.

    The `pos_reference` argument determines how the (y, x) coordinates are interpreted
    when extracting boxes:

    - If ``pos_reference="center"``:
        The (y, x) coordinate refers to the **center** of the box.
        The box extends from (y - height // 2, x - width // 2) to
        (y + height // 2, x + width // 2).

        Example:
            :                +------------------+
            :                |                  |
            :              height      * (y, x) |
            :                |                  |
            :                +------ width -----+

    - If ``pos_reference="top-left"``:
        The (y, x) coordinate refers to the **top-left corner** of the box.
        The box extends from (y, x) to (y + height, x + width).

        Example:
            :         (y, x) *------ width -----+
            :                |                  |
            :                |                height
            :                |                  |
            :                +------------------+

    Parameters
    ----------
    image : torch.Tensor | np.ndarray
        The input image from which to extract the regions.
    pos_y : torch.Tensor | np.ndarray
        The y positions of the regions to extract. Type must mach `image`
    pos_x : torch.Tensor | np.ndarray
        The x positions of the regions to extract. Type must mach `image`
    box_size : int | tuple[int, int]
        The size of the box to extract. If an integer is passed, the box will be square.
    pos_reference : Literal["center", "top-left"], optional
        The reference point for the positions, by default "center". If "center", the
        boxes extracted will be image[y - box_size // 2 : y + box_size // 2, ...]. If
        "top-left", the boxes will be image[y : y + box_size, ...].
    handle_bounds : Literal["pad", "clip", "error"], optional
        How to handle the bounds of the image, by default "pad". If "pad", the image
        will be padded with the padding value based on the padding mode. If "error", an
        error will be raised if any region exceeds the image bounds. Note clipping is
        not supported since returned stack may have inhomogeneous sizes.
    padding_mode : Literal["constant", "reflect", "replicate"], optional
        The padding mode to use when padding the image, by default "constant".
        "constant" pads with the value `padding_value`, "reflect" pads with the
        reflection of the image at the edge, and "replicate" pads with the last pixel
        of the image. These match the modes available in `torch.nn.functional.pad`.
    padding_value : float, optional
        The value to use for padding when `padding_mode` is "constant", by default 0.0.

    Returns
    -------
    torch.Tensor | np.ndarray
        The stack of cropped images extracted from the input image. Type will match the
        input image type.

    Raises
    ------
    ValueError
        If `pos_reference` is not one of "center" or "top-left", or if `image` is not a
        torch.Tensor or np.ndarray.
    """
    if isinstance(box_size, int):
        box_size = (box_size, box_size)

    if pos_reference == "center":
        pos_y = pos_y - box_size[0] // 2
        pos_x = pos_x - box_size[1] // 2
    elif pos_reference == "top-left":
        pass
    else:
        raise ValueError(f"Unknown pos_reference: {pos_reference}")

    if isinstance(image, torch.Tensor):
        return _get_cropped_image_regions_torch(
            image=image,
            pos_y=pos_y,
            pos_x=pos_x,
            box_size=box_size,
            handle_bounds=handle_bounds,
            padding_mode=padding_mode,
            padding_value=padding_value,
        )

    if isinstance(image, np.ndarray):
        padding_mode_np = TORCH_TO_NUMPY_PADDING_MODE[padding_mode]
        return _get_cropped_image_regions_numpy(
            image=image,
            pos_y=pos_y,
            pos_x=pos_x,
            box_size=box_size,
            handle_bounds=handle_bounds,
            padding_mode=padding_mode_np,
            padding_value=padding_value,
        )

    raise ValueError(f"Unknown image type: {type(image)}")


# pylint: disable=too-many-locals
def _get_cropped_image_regions_numpy(
    image: np.ndarray,
    pos_y: np.ndarray,
    pos_x: np.ndarray,
    box_size: tuple[int, int],
    handle_bounds: Literal["pad", "error"],
    padding_mode: str,
    padding_value: float,
) -> np.ndarray:
    """Helper function for extracting regions from a numpy array.

    NOTE: this function assumes that the position reference is the top-left corner.
    Reference value is handled by the user-exposed 'get_cropped_image_regions' function.
    """
    if handle_bounds == "pad":
        bs1 = box_size[1] - 1
        bs0 = box_size[0] - 1
        pad_kwargs = {}
        if padding_mode == "constant":
            pad_kwargs["constant_values"] = padding_value
        image = np.pad(
            image,
            pad_width=((bs0, bs0), (bs1, bs1)),
            mode=padding_mode,
            **pad_kwargs,
        )
        pos_y = pos_y + bs0
        pos_x = pos_x + bs1

    regions = []
    for y, x in zip(pos_y, pos_x, strict=False):
        if (
            y < 0
            or x < 0
            or y + box_size[0] > image.shape[0]
            or x + box_size[1] > image.shape[1]
        ):
            raise IndexError(
                f"Region bounds [{y}:{y + box_size[0]}, {x}:{x + box_size[1]}] exceed "
                f"image dimensions {image.shape}"
            )

        regions.append(image[y : y + box_size[0], x : x + box_size[1]])

    cropped_images = np.stack(regions)

    return cropped_images


# pylint: disable=too-many-locals
def _get_cropped_image_regions_torch(
    image: torch.Tensor,
    pos_y: torch.Tensor,
    pos_x: torch.Tensor,
    box_size: tuple[int, int],
    handle_bounds: Literal["pad", "error"],
    padding_mode: Literal["constant", "reflect", "replicate"],
    padding_value: float = 0.0,
) -> torch.Tensor:
    """Helper function for extracting regions from a torch tensor.

    NOTE: this function assumes that the position reference is the top-left corner.
    Reference value is handled by the user-exposed 'get_cropped_image_regions' function.
    """
    if handle_bounds == "pad":
        bs1 = box_size[1] - 1
        bs0 = box_size[0] - 1
        pad_kwargs = {}
        if padding_mode == "constant":
            pad_kwargs["value"] = padding_value
        # NOTE: Need to do unsqueeze/squeeze workaround to make torch happy with input
        # tensor shapes. Looks like API for padding may change in the future torch...
        image = torch.nn.functional.pad(
            image.unsqueeze(0),
            pad=(bs1, bs1, bs0, bs0),
            mode=padding_mode,
            **pad_kwargs,
        ).squeeze(0)
        pos_y = pos_y + bs0
        pos_x = pos_x + bs1

    regions = []
    for y, x in zip(pos_y, pos_x, strict=False):
        y = int(y.item() if hasattr(y, "item") else y)
        x = int(x.item() if hasattr(x, "item") else x)
        original_y, original_x = y, x

        if (
            y < 0
            or x < 0
            or y + box_size[0] > image.shape[0]
            or x + box_size[1] > image.shape[1]
        ):
            if handle_bounds == "error":
                raise IndexError(
                    f"Region bounds [{original_y}:{original_y + box_size[0]}, "
                    f"{original_x}:{original_x + box_size[1]}] exceed "
                    f"image dimensions {image.shape}"
                )
            warnings.warn(
                f"Region bounds [{original_y}:{original_y + box_size[0]}, "
                f"{original_x}:{original_x + box_size[1]}] exceed "
                f"image dimensions {image.shape}. Clamping to edges.",
                UserWarning,
                stacklevel=2,
            )
            y = max(0, min(y, image.shape[0] - box_size[0]))
            x = max(0, min(x, image.shape[1] - box_size[1]))

        regions.append(image[y : y + box_size[0], x : x + box_size[1]])

    cropped_images = torch.stack(regions)

    return cropped_images


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


# pylint: disable=too-many-instance-attributes
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

    leopard_em_version: str = Field(default_factory=_leopard_em_version)
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

    def _get_position_reference_columns(self) -> tuple[str, str]:
        """Return the y/x position column names to use (refined preferred)."""
        y_col = "refined_pos_y" if "refined_pos_y" in self._df.columns else "pos_y"
        x_col = "refined_pos_x" if "refined_pos_x" in self._df.columns else "pos_x"
        return y_col, x_col

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
        if column_name in _POSITION_COLUMNS:
            self._stored_image_stack = None
            self.local_stats = {}
        elif column_name == "micrograph_path":
            self._stored_image_stack = None
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
        """Replace the underlying DataFrame, normalizing it to the canonical layout."""
        self._df = _normalize_particle_dataframe(df, on_invalid_ids=on_invalid_ids)

    def _as_positions(self, indexes: Any) -> np.ndarray:
        """Validate and convert particle indexes to 0-based row positions.

        Parameters
        ----------
        indexes : Any
            Integer row positions (e.g. a ``pd.Index`` from
            :meth:`load_images_grouped_by_column`, a list or an array).

        Returns
        -------
        np.ndarray
            1-D int64 array of row positions.

        Raises
        ------
        TypeError
            If ``indexes`` are not integers (e.g. ``particle_id`` strings).
        IndexError
            If any position is outside ``[0, num_particles)``.
        """
        positions = np.asarray(indexes).reshape(-1)
        if positions.size == 0:
            return positions.astype(np.int64)
        if positions.dtype.kind not in "iu":
            raise TypeError(
                "Particle indexes must be integer row positions (0..N-1), got dtype "
                f"'{positions.dtype}'. To select by particle_id, look up the positions "
                f"in the '{PARTICLE_ID_COLUMN}' column first."
            )
        positions = positions.astype(np.int64)
        num_particles = self.num_particles
        out_of_range = (positions < 0) | (positions >= num_particles)
        if out_of_range.any():
            examples = positions[out_of_range][:5].tolist()
            raise IndexError(
                f"{int(out_of_range.sum())} particle index(es) out of range for a "
                f"stack of {num_particles} particles (e.g. {examples})."
            )
        return positions

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
        rel_defocus_col = "relative_defocus"
        if prefer_refined_defocus:
            if "refined_relative_defocus" not in self._df.columns:
                warnings.warn(
                    "Refined defocus values not found in DataFrame, using original "
                    "defocus values...",
                    stacklevel=2,
                )
            elif _any_nan_or_inf(self._df["refined_relative_defocus"]):
                warnings.warn(
                    "Refined defocus values contain NaN or inf values, using original "
                    "defocus values...",
                    stacklevel=2,
                )
            else:
                rel_defocus_col = "refined_relative_defocus"

        return torch.tensor(self._df[rel_defocus_col].to_numpy().copy())

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
        defocus_u = torch.tensor(self._df["defocus_u"].to_numpy().copy())
        defocus_v = torch.tensor(self._df["defocus_v"].to_numpy().copy())
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
        pixel_size_col = "pixel_size"
        if prefer_refined_pixel_size:
            if "refined_pixel_size" not in self._df.columns:
                warnings.warn(
                    "Refined pixel size not found in DataFrame, using original"
                    " pixel size values...",
                    stacklevel=2,
                )
            elif _any_nan_or_inf(self._df["refined_pixel_size"]):
                warnings.warn(
                    "Refined pixel size contain NaN or inf values, using original"
                    " pixel size values...",
                    stacklevel=2,
                )
            else:
                pixel_size_col = "refined_pixel_size"

        return torch.tensor(self._df[pixel_size_col].to_numpy().copy())

    def get_euler_angles(self, prefer_refined_angles: bool = True) -> torch.Tensor:
        """Return the Euler angles (phi, theta, psi) of all particles as a tensor.

        Parameters
        ----------
        prefer_refined_angles : bool, optional
            When true, refined angles are used if present, by default True.

        Returns
        -------
        torch.Tensor
            Shape ``(N, 3)`` — columns correspond to (phi, theta, psi) in ZYZ.
        """
        phi_col = "phi"
        theta_col = "theta"
        psi_col = "psi"
        if prefer_refined_angles:
            if not all(
                x in self._df.columns
                for x in ["refined_phi", "refined_theta", "refined_psi"]
            ):
                warnings.warn(
                    "Refined angles not found in DataFrame, using original angles...",
                    stacklevel=2,
                )
            else:
                phi_col = "refined_phi"
                theta_col = "refined_theta"
                psi_col = "refined_psi"

        phi = torch.tensor(self._df[phi_col].to_numpy().copy())
        theta = torch.tensor(self._df[theta_col].to_numpy().copy())
        psi = torch.tensor(self._df[psi_col].to_numpy().copy())

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
        if column_name not in self._df.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

        num_missing = int(self._df[column_name].isna().sum())
        if num_missing:
            raise ValueError(
                f"Column '{column_name}' is empty for {num_missing} particle(s); every "
                "particle must reference an image file."
            )

        dataset_name = STATISTIC_MAP_PATH_TO_HDF5_DATASET.get(column_name)

        # ``.indices`` gives row *positions*, which is what the positional tensors
        # built from these groups need regardless of the DataFrame's index.
        image_index_groups = self._df.groupby(column_name).indices
        images_list = []
        indices = []
        for img_path, positions in image_index_groups.items():
            try:
                img = load_result_map_image(img_path, dataset_name=dataset_name)
            except (FileNotFoundError, ValueError) as err:
                raise type(err)(
                    f"Could not load '{column_name}' for {len(positions)} particle(s): "
                    f"{err}"
                ) from err
            images_list.append(img)
            indices.append(pd.Index(positions))

        images_tensor = torch.stack(images_list, dim=0)
        return images_tensor, indices

    def _stored_local_stat_map(self, column: str) -> torch.Tensor | None:
        """Return an already-computed local stat map for ``column``, if any.

        Parameters
        ----------
        column : str
            Path column name (e.g. ``"correlation_average_path"``).

        Returns
        -------
        torch.Tensor | None
            The stored ``(num_particles, valid_h, valid_w)`` map, or None if it
            must be derived from the referenced result file.
        """
        return self.local_stats.get(column)

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
            columns = list(_DEFAULT_LOCAL_STAT_COLUMNS)

        if valid_size is None:
            box_h, box_w = self.extracted_box_size
            h, w = self.original_template_size
            valid_size = (box_h - h + 1, box_w - w + 1)
        expected_shape = (self.num_particles, *valid_size)

        device = torch.device(device)

        stat_maps: dict[str, torch.Tensor] = {}
        for column in columns:
            stored = self._stored_local_stat_map(column) if use_stored else None
            if stored is not None and tuple(stored.shape) == expected_shape:
                stat_maps[column] = stored.to(device)
                continue

            pad = (
                padding_value
                if padding_value is not None
                else _DEFAULT_LOCAL_STAT_PADDING.get(column, 0.0)
            )
            images, indices = self.load_images_grouped_by_column(column)
            stat_maps[column] = self._crop_particle_regions(
                images=images,
                indices=indices,
                extraction_size=valid_size,
                pos_reference="top-left",
                handle_bounds="pad",
                padding_mode="constant",
                padding_value=pad,
            ).to(device)

        return stat_maps

    def _crop_particle_regions(
        self,
        images: torch.Tensor,
        indices: list[pd.Index],
        extraction_size: tuple[int, int],
        pos_reference: Literal["center", "top-left"] = "top-left",
        handle_bounds: Literal["pad", "error"] = "pad",
        padding_mode: Literal["constant", "reflect", "replicate"] = "constant",
        padding_value: float = 0.0,
    ) -> torch.Tensor:
        """Crop a per-particle sub-image from each source image, read-only.

        This method preferentially selects refined position columns by default
        (refined_pos_x, refined_pos_y) if they are present in the DataFrame, falling
        back to unrefined positions (pos_x, pos_y) otherwise.

        This method uses columns pos_x and pos_y (or refined_pos_x and refined_pos_y if
        available) to extract the boxes from the images. When using top-left reference
        position, the boxes are extracted as follows, where the dots represent the
        actual particle in the image

        Example:
            :                +----------------------------------+
            :                |                                  |
            :                |                                  |
            :                |     (x, y) *=== box_w ===+       |
            :                |            |             |       |
            :                |            |     ....  box_h     |
            :           img_height        |    ......   |       |
            :                |            |     ....    |       |
            :                |            |             |       |
            :                |            +=============+       |
            :                |                                  |
            :                +------------ img_width -----------+

        When center reference is used, then the position columns in the DataFrame are
        interpreted as the center of the particle, and the boxes are extracted around
        this x and y position as follows:

        Example:
            :                +----------------------------------+
            :                |                                  |
            :                |                                  |
            :                |            +=== box_w ===+       |
            :                |            |             |       |
            :                |            |     ....    |       |
            :           img_height        |(x, y).*.. box_h     |
            :                |            |     ....    |       |
            :                |            |             |       |
            :                |            +=============+       |
            :                |                                  |
            :                +------------ img_width -----------+

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
        y_col, x_col = self._get_position_reference_columns()

        h, w = self.original_template_size
        box_h, box_w = self.extracted_box_size
        device = images.device
        region_stack = torch.zeros(
            (self.num_particles, *extraction_size), device=device
        )

        if images.shape[0] != len(indices):
            raise ValueError(
                f"Number of images ({images.shape[0]}) does not match the number of "
                f"indices ({len(indices)})."
            )

        all_pos_y = _to_pixel_positions(self._df[y_col].to_numpy())
        all_pos_x = _to_pixel_positions(self._df[x_col].to_numpy())

        for i, indexes in enumerate(indices):
            img = images[i]
            positions = self._as_positions(indexes)
            pos_y = all_pos_y[positions]
            pos_x = all_pos_x[positions]

            if pos_reference == "center":
                pos_y = pos_y - h // 2
                pos_x = pos_x - w // 2

            pos_y = pos_y - (box_h - h) // 2
            pos_x = pos_x - (box_w - w) // 2

            pos_y = torch.tensor(pos_y, device=img.device)
            pos_x = torch.tensor(pos_x, device=img.device)

            cropped_images = get_cropped_image_regions(
                img,
                pos_y,
                pos_x,
                extraction_size,
                pos_reference="top-left",
                handle_bounds=handle_bounds,
                padding_mode=padding_mode,
                padding_value=padding_value,
            )
            region_stack[torch.as_tensor(positions, device=device)] = cropped_images

        return region_stack

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
        image_stack = self._crop_particle_regions(
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
        device = images_dft.device
        num_images = images_dft.shape[0]
        filter_stack = torch.zeros((num_images, *output_shape), device=device)

        for i in range(num_images):
            img_dft = images_dft[i]
            cumulative_filter = preprocess_filters.get_combined_filter(
                ref_img_rfft=img_dft,
                output_shape=output_shape,
            )

            filter_stack[i] = cumulative_filter

        return filter_stack

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
        device = images_dft.device
        filter_stack = torch.zeros((self.num_particles, *output_shape), device=device)
        if images_dft.shape[0] != len(indices):
            raise ValueError(
                f"Number of images ({images_dft.shape[0]}) does not match "
                f"the number of indices ({len(indices)})."
            )

        for i, indexes in enumerate(indices):
            img_dft = images_dft[i]
            cumulative_filter = preprocess_filters.get_combined_filter(
                ref_img_rfft=img_dft,
                output_shape=output_shape,
            )

            positions = torch.as_tensor(self._as_positions(indexes), device=device)
            filter_stack[positions] = cumulative_filter

        return filter_stack

    @staticmethod
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    def _process_single_frame_with_shifts_checkpoint(
        movie_frame: torch.Tensor,
        shifts: torch.Tensor,  # (N, 2) -> (dy, dx)
        pos_y: torch.Tensor,
        pos_x: torch.Tensor,
        extracted_box_size: tuple[int, int],
        handle_bounds: Literal["pad", "error"],
        padding_mode: Literal["constant", "reflect", "replicate"],
        padding_value: float,
    ) -> torch.Tensor:
        """Process a single frame using precomputed particle shifts.

        Safe for gradient checkpointing; contains no deformation-field evaluation.

        Parameters
        ----------
        movie_frame : torch.Tensor
            Single movie frame (H, W).
        shifts : torch.Tensor
            Per-particle shifts with shape (N, 2) as (dy, dx).
        pos_y, pos_x : torch.Tensor
            Top-left extraction positions.
        extracted_box_size : tuple[int, int]
            ``(box_h, box_w)``.
        handle_bounds, padding_mode, padding_value
            Passed through to cropping.

        Returns
        -------
        torch.Tensor
            Shifted FFTs with shape ``(N, box_h, box_w//2 + 1)``.
        """
        box_h, box_w = extracted_box_size

        cropped_images = get_cropped_image_regions(
            movie_frame,
            pos_y,
            pos_x,
            extracted_box_size,
            pos_reference="top-left",
            handle_bounds=handle_bounds,
            padding_mode=padding_mode,
            padding_value=padding_value,
        )

        cropped_images_dft = torch.fft.rfftn(  # pylint: disable=not-callable
            cropped_images, dim=(-2, -1)
        )

        shifted_fft = fourier_shift_dft_2d(
            dft=cropped_images_dft,
            image_shape=(box_h, box_w),
            shifts=shifts,
            rfft=True,
            fftshifted=False,
        )

        return shifted_fft

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
        frame_deformation_field = deformation_field.evaluate_at_t(
            t=float(normalized_t_value.item()),
            grid_shape=(10 * gh, 10 * gw),
        )

        pixel_shifts = get_pixel_shifts(
            frame=movie_frame,
            pixel_spacing=pixel_spacing,
            frame_deformation_grid=frame_deformation_field,
            pixel_grid=pixel_grid,
        )

        y_shifts = -pixel_shifts[pos_y_center, pos_x_center, 0]
        x_shifts = -pixel_shifts[pos_y_center, pos_x_center, 1]

        return torch.stack((y_shifts, x_shifts), dim=-1)

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-branches
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
        if deformation_field is not None and particle_shifts is not None:
            raise ValueError(
                "Only one of `deformation_field` or `particle_shifts` can be provided."
            )
        if deformation_field is None and particle_shifts is None:
            if require_motion_source:
                raise ValueError(
                    "One of `deformation_field` or `particle_shifts` must be provided."
                )
            warnings.warn(
                "No deformation field or particle shifts were provided. Assuming the "
                "movie is already aligned and extracting each frame without shifts.",
                stacklevel=2,
            )
        pixel_sizes = self.get_pixel_size()
        y_col, x_col = self._get_position_reference_columns()
        h, w = self.original_template_size
        box_h, box_w = self.extracted_box_size
        t, img_h, img_w = movie.shape
        if deformation_field is not None:
            _, _, gh, gw = deformation_field.data.shape
        else:
            gh = gw = 0
        if normalized_t_values is None:
            normalized_t = torch.linspace(0, 1, steps=t, device=movie.device)
        else:
            if normalized_t_values.numel() != t:
                raise ValueError(
                    "normalized_t_values must have one entry per movie frame."
                )
            normalized_t = normalized_t_values.to(device=movie.device).reshape(t)
        pixel_grid = coordinate_grid(
            image_shape=(img_h, img_w),
            device=movie.device,
        )
        if particle_indices is not None:
            # Use provided subset of particles (0-based row positions)
            positions = self._as_positions(particle_indices)
        else:
            # Use all particles
            positions = np.arange(self.num_particles, dtype=np.int64)
        particle_indexes = positions.tolist()
        num_particles_to_process = len(particle_indexes)

        pos_y = _to_pixel_positions(self._df[y_col].to_numpy())[positions]
        pos_x = _to_pixel_positions(self._df[x_col].to_numpy())[positions]
        # If the position reference is "top-left", shift (x, y) by half the original
        # template width/height so reference is now in the center
        if pos_reference == "center":
            pos_y = pos_y - h // 2
            pos_x = pos_x - w // 2

        pos_y_center = pos_y + h // 2
        pos_x_center = pos_x + w // 2
        pos_y -= (box_h - h) // 2
        pos_x -= (box_w - w) // 2
        pos_y = torch.tensor(pos_y)
        pos_x = torch.tensor(pos_x)
        pos_y_center = torch.tensor(pos_y_center)
        pos_x_center = torch.tensor(pos_x_center)

        aligned_particle_movies_rfft = torch.zeros(
            (num_particles_to_process, t, box_h, box_w // 2 + 1),
            dtype=torch.complex64,
            device=movie.device,
        )
        movie = movie - torch.mean(movie, dim=(-2, -1), keepdim=True)

        for frame_index, movie_frame in enumerate(movie):
            if particle_shifts is not None:
                frame_shifts = particle_shifts[frame_index]  # (N, 2)
                if particle_indices is not None:
                    frame_shifts = frame_shifts[torch.as_tensor(positions)]
            elif deformation_field is None:
                frame_shifts = torch.zeros(
                    (num_particles_to_process, 2),
                    dtype=movie.dtype,
                    device=movie.device,
                )
            else:
                frame_shifts = self.compute_frame_particle_shifts_from_deformation(
                    movie_frame=movie_frame,
                    deformation_field=deformation_field,
                    normalized_t_value=normalized_t[frame_index],
                    pixel_grid=pixel_grid,
                    pixel_spacing=pixel_sizes[0].item(),
                    pos_y_center=pos_y_center,
                    pos_x_center=pos_x_center,
                    gh=gh,
                    gw=gw,
                )

            if use_gradient_checkpointing:
                shifted_fft = checkpoint(
                    self._process_single_frame_with_shifts_checkpoint,
                    movie_frame,
                    frame_shifts,
                    pos_y,
                    pos_x,
                    self.extracted_box_size,
                    handle_bounds,
                    padding_mode,
                    padding_value,
                    use_reentrant=False,
                )
            else:
                shifted_fft = self._process_single_frame_with_shifts_checkpoint(
                    movie_frame=movie_frame,
                    shifts=frame_shifts,
                    pos_y=pos_y,
                    pos_x=pos_x,
                    extracted_box_size=self.extracted_box_size,
                    handle_bounds=handle_bounds,
                    padding_mode=padding_mode,
                    padding_value=padding_value,
                )

            aligned_particle_movies_rfft[:, frame_index] = shifted_fft

            if frame_index % 10 == 0 and frame_index > 0:
                torch.cuda.empty_cache()
        return aligned_particle_movies_rfft, particle_indexes

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
        box_h, box_w = self.extracted_box_size
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
        num_particles_to_process = aligned_particle_movies_rfft.shape[0]

        # Dose weight the aligned particle images
        aligned_particle_images = torch.zeros(
            (num_particles_to_process, box_h, box_w),
            device=movie.device,
        )
        voltages = self._df["voltage"].to_numpy()
        for particle_index in range(num_particles_to_process):
            particle_dft = aligned_particle_movies_rfft[particle_index]

            # Row position of this particle in the full stack
            row = particle_positions[particle_index]

            dw_sum = dose_weight_movie_to_micrograph(
                movie_fft=particle_dft,
                pixel_size=float(pixel_sizes[row].item()),
                pre_exposure=pre_exposure,
                fluence_per_frame=fluence_per_frame,
                voltage=voltages[row],
            )
            aligned_particle_images[particle_index] = dw_sum

        if particle_indices is None:
            self.image_stack = aligned_particle_images
        return aligned_particle_images


# ---------------------------------------------------------------------------
# CSV-backed subclass
# ---------------------------------------------------------------------------


class ParticleStackCSV(_ParticleStackBase):
    """Particle stack whose tabular data is loaded from a CSV file.

    Particle images are extracted from the micrograph paths referenced in the
    CSV at run time.  This is the original ``ParticleStack`` behavior.

    Attributes
    ----------
    df_path : str
        Path to the CSV file containing the particle data.
    """

    df_path: str

    def load_df(self) -> None:
        """Load and validate the particle DataFrame from ``df_path``.

        Raises
        ------
        ValueError
            If required columns are missing from the CSV.
        """
        tmp_df = pd.read_csv(self.df_path)
        _check_required_columns(tmp_df, source=f"'{self.df_path}'")
        self._set_dataframe(tmp_df, on_invalid_ids="regenerate")

    # ---------------------------------------------------------------------------
    # I/O methods
    # ---------------------------------------------------------------------------

    def export_results(self, allow_file_overwrite: bool | None = None) -> None:
        """Write the particle table to ``df_path`` as CSV.

        Parameters
        ----------
        allow_file_overwrite : bool | None
            Deprecated and ignored; files are always overwritten.

        Raises
        ------
        PermissionError
            If the parent directory is not writable.
        """
        if allow_file_overwrite is not None:
            _warn_allow_file_overwrite_deprecated()
        with atomic_write_path(self.df_path) as tmp_path:
            self._df.to_csv(tmp_path)

    def to_hdf5(
        self,
        hdf5_path: str,
        allow_file_overwrite: bool | None = None,
        include_image_stack: bool = False,
        include_local_stats: bool = False,
    ) -> "ParticleStackHDF5":
        """Convert this CSV-backed stack to an HDF5-backed stack and write to disk.

        Parameters
        ----------
        hdf5_path : str
            Destination path for the HDF5 file. An existing file is replaced atomically.
        allow_file_overwrite : bool | None
            Deprecated and ignored; files are always overwritten.
        include_image_stack : bool, optional
            Write ``image_stack`` to the HDF5 file, by default False.
            Raises ``ValueError`` if the image stack has not been loaded.
        include_local_stats : bool, optional
            Write every entry currently in :attr:`local_stats` to the HDF5 file, by
            default False. Raises ``ValueError`` if :attr:`local_stats` is empty.

        Returns
        -------
        ParticleStackHDF5
            The new HDF5-backed stack instance pointing at ``hdf5_path``.
        """
        if allow_file_overwrite is not None:
            _warn_allow_file_overwrite_deprecated()

        hdf5_stack = ParticleStackHDF5(
            hdf5_path=hdf5_path,
            extracted_box_size=self.extracted_box_size,
            original_template_size=self.original_template_size,
            leopard_em_version=self.leopard_em_version,
            global_whitening_applied=self.global_whitening_applied,
            local_whitening_applied=self.local_whitening_applied,
            global_normalization_applied=self.global_normalization_applied,
            local_normalization_applied=self.local_normalization_applied,
            image_stack=self.image_stack if include_image_stack else None,
            local_stats=dict(self.local_stats) if include_local_stats else {},
            skip_df_load=True,
        )
        # Particle IDs are generated on write if the table doesn't have any yet
        hdf5_stack._set_dataframe(self._df)  # pylint: disable=protected-access
        hdf5_stack.to_hdf5(
            include_image_stack=include_image_stack,
            include_local_stats=include_local_stats,
        )
        return hdf5_stack


# ---------------------------------------------------------------------------
# HDF5-backed subclass
# ---------------------------------------------------------------------------


class ParticleStackHDF5(_ParticleStackBase):
    """Particle stack stored entirely within a single HDF5 file.

    The particle table, optional image stack, and optional per-particle local statistic
    maps are all held in one ``.h5`` file. Constructing an instance with ``hdf5_path``
    (e.g. from a YAML config) loads everything from that file: ``extracted_box_size``,
    ``original_template_size``, the pre-processing flags, the particle table and any
    stored tensors. Explicitly given box sizes override the file's (stored tensors that
    no longer fit are then ignored with a warning).

    * **Stored image stack**: when the file has an ``image_stack`` and
      ``use_stored_image_stack`` is True (default), programs (refine, optimize,
      constrained search) use those particle images directly -- the micrographs are
      not needed -- and compute filters per particle, since whole-micrograph
      (global) filtering is impossible without the micrograph. Set
      ``use_stored_image_stack: false`` to re-extract from the micrographs instead.
    * **Stored local stats**: per-particle statistic maps under ``/local_stats``
      (any subset of the ``*_path`` columns) are used in place of re-reading and
      cropping the referenced full-size result maps.

    Populate ``self.local_stats`` (e.g. via
    ``self.local_stats.update(self.get_local_stat_maps())``) before calling
    ``to_hdf5(include_local_stats=True)``, and every entry present at that point is
    written, each to its own dataset under ``/local_stats`` named after its column.

    In memory the particle table has a 0..N-1 ``RangeIndex`` and ``particle_id`` is an
    ordinary column; particle IDs are generated on write when absent.

    HDF5 file layout (format version 2)
    -----------------------------------

    ::

        / (root)
        │  attrs: format_version, writer_version, leopard_em_version,
        │         extracted_box_size, original_template_size,
        │         image_stack_stored, local_stats_stored,
        │         global_whitening_applied, local_whitening_applied,
        │         global_normalization_applied, local_normalization_applied
        ├─ particles/
        │      particle_id            (N,)   variable-length str  "{mic_stem}_{idx:05d}"
        │      <column>               (N,)   native int/float/bool, or variable-length
        │      ...                           str (attr ``encoding`` = numeric | str)
        ├─ image_stack                (N, box_h, box_w)             float32  [optional]
        └─ local_stats/                                                      [optional]
               <column>                (N, valid_h, valid_w)         float32
               ...                                     -- one dataset per entry in
                                                           `local_stats` at write time,
                                                           e.g. `mip_path`,
                                                           `correlation_average_path`

    where ``valid_h = extracted_box_size[0] - original_template_size[0] + 1``
    and   ``valid_w = extracted_box_size[1] - original_template_size[1] + 1``.
    Files written by Leopard-EM v1.3 (no ``format_version``) are read as well.

    Attributes
    ----------
    hdf5_path : str
        Path to the HDF5 file.
    use_stored_image_stack : bool
        Load and use the file's stored image stack, if any. Default True.
    allow_file_overwrite : bool | None
        Deprecated and ignored; files are always overwritten atomically.
    image_stack_stored : bool
        True when ``/image_stack`` is present in the HDF5 file. Set from the file.
    local_stats_stored : bool
        True when the ``/local_stats`` group is present in the HDF5 file. Set from the
        file.
    """

    hdf5_path: str
    use_stored_image_stack: bool = True

    # Kept (and excluded from dumps) so configs written by older versions validate.
    allow_file_overwrite: SkipJsonSchema[bool | None] = Field(
        default=None, exclude=True
    )
    image_stack_stored: SkipJsonSchema[bool] = Field(default=False, exclude=True)
    local_stats_stored: SkipJsonSchema[bool] = Field(default=False, exclude=True)

    def __init__(self, skip_df_load: bool = False, **data: Any):
        """Initialize the stack, loading ``hdf5_path`` unless ``skip_df_load``.

        Parameters
        ----------
        skip_df_load : bool, optional
            When True nothing is read from ``hdf5_path`` (use when constructing an
            instance that will be written to a new file).
        data : dict[str, Any]
            Fields forwarded to the Pydantic constructor.
        """
        contents = None
        if not skip_df_load and data.get("hdf5_path") is not None:
            contents = read_particle_stack_hdf5(
                data["hdf5_path"],
                load_image_stack=data.get("use_stored_image_stack") is not False,
            )
            data = self._merge_file_metadata(data, contents)

        super().__init__(skip_df_load=True, **data)

        if data.get("allow_file_overwrite") is not None:
            _warn_allow_file_overwrite_deprecated()
        if contents is not None:
            self._apply_file_contents(contents)

    @staticmethod
    def _merge_file_metadata(
        data: dict[str, Any], contents: ParticleStackFileContents
    ) -> dict[str, Any]:
        """Fill constructor data from file metadata (explicit > file > default)."""
        data = dict(data)
        for key in BOX_SIZE_ATTRS:
            if data.get(key) is None and key in contents.metadata:
                data[key] = contents.metadata[key]
        # Provenance describes the file's contents, so the file always wins
        for key in ("leopard_em_version", *PREPROCESSING_FLAG_ATTRS):
            if key in contents.metadata:
                data[key] = contents.metadata[key]
        data["image_stack_stored"] = contents.image_stack_stored
        data["local_stats_stored"] = contents.local_stats_stored
        return data

    def _apply_file_contents(self, contents: ParticleStackFileContents) -> None:
        """Adopt the particle table and compatible stored tensors read from file."""
        _check_required_columns(contents.df, source=f"'{self.hdf5_path}'")
        self._set_dataframe(
            contents.df,
            on_invalid_ids="raise" if contents.format_version >= 2 else "regenerate",
        )

        position_columns = self._get_position_reference_columns()

        def _compatible(
            name: str, stored: StoredTensor, shape: tuple[int, ...]
        ) -> bool:
            if tuple(stored.data.shape) != shape:
                reason = (
                    f"shape {tuple(stored.data.shape)} does not match the expected "
                    f"{shape}"
                )
            elif stored.position_columns not in (None, position_columns):
                reason = (
                    f"it was extracted at {stored.position_columns}, not "
                    f"{position_columns}"
                )
            else:
                return True
            warnings.warn(
                f"Ignoring stored '{name}' in '{self.hdf5_path}': {reason}. It will be "
                "recomputed from the referenced files if needed.",
                UserWarning,
                stacklevel=4,
            )
            return False

        n = self.num_particles
        if contents.image_stack is not None and _compatible(
            "image_stack", contents.image_stack, (n, *self.extracted_box_size)
        ):
            self._stored_image_stack = contents.image_stack.data
            self.image_stack = contents.image_stack.data

        valid_shape = (n, *self._valid_local_stat_size())
        self.local_stats = {
            column: stored.data
            for column, stored in contents.local_stats.items()
            if _compatible(f"local_stats/{column}", stored, valid_shape)
        }

    def _valid_local_stat_size(self) -> tuple[int, int]:
        box_h, box_w = self.extracted_box_size
        h, w = self.original_template_size
        return box_h - h + 1, box_w - w + 1

    def get_stored_image_stack(self) -> torch.Tensor | None:
        """Return the stored image stack, unless ``use_stored_image_stack`` is False.

        See :meth:`_ParticleStackBase.get_stored_image_stack`.

        Returns
        -------
        torch.Tensor | None
        """
        if not self.use_stored_image_stack:
            return None
        return super().get_stored_image_stack()

    ###########################
    ### Data loading        ###
    ###########################

    def load_df(self) -> None:
        """(Re)load only the particle DataFrame from the HDF5 file at ``hdf5_path``.

        Raises
        ------
        FileNotFoundError
            If ``hdf5_path`` does not exist.
        """
        contents = read_particle_stack_hdf5(
            self.hdf5_path, load_image_stack=False, load_local_stats=False
        )
        _check_required_columns(contents.df, source=f"'{self.hdf5_path}'")
        self._set_dataframe(
            contents.df,
            on_invalid_ids="raise" if contents.format_version >= 2 else "regenerate",
        )

    ###########################
    ### I/O methods         ###
    ###########################

    def export_results(
        self,
        include_image_stack: bool = False,
        include_local_stats: bool = False,
    ) -> None:
        """Write the particle table (and optional tensors) to ``hdf5_path``.

        Alias for ``to_hdf5``, kept for API symmetry with ``ParticleStackCSV``.
        """
        self.to_hdf5(
            include_image_stack=include_image_stack,
            include_local_stats=include_local_stats,
        )

    def to_hdf5(
        self,
        include_image_stack: bool = False,
        include_local_stats: bool = False,
    ) -> None:
        """Write the particle table and optional tensors to ``hdf5_path``.

        Notes
        -----
        An existing file is replaced atomically. Particle IDs are generated first if
        the table does not have a ``particle_id`` column.

        Parameters
        ----------
        include_image_stack : bool, optional
            Write ``image_stack`` to ``/image_stack``, by default False.
            Raises ``ValueError`` if ``image_stack`` is None.
        include_local_stats : bool, optional
            Write every entry currently in :attr:`local_stats` to its own
            dataset under ``/local_stats``, by default False.

        Raises
        ------
        ValueError
            If a requested tensor is missing or its shape does not match the particle
            table, or if the particle IDs are not unique.
        """
        df = self._df
        if PARTICLE_ID_COLUMN not in df.columns:
            df.insert(0, PARTICLE_ID_COLUMN, _generate_particle_ids(df))
        # Validate IDs, and keep `particle_id` as the first column on disk
        df = _normalize_particle_dataframe(df, on_invalid_ids="raise")
        df = df[[PARTICLE_ID_COLUMN, *(c for c in df.columns if c != "particle_id")]]
        self._df = df

        num_particles = self.num_particles
        image_stack = None
        if include_image_stack:
            if self.image_stack is None:
                raise ValueError(
                    "image_stack is None; cannot write to HDF5. "
                    "Call construct_image_stack() first."
                )
            expected = (num_particles, *self.extracted_box_size)
            if tuple(self.image_stack.shape) != expected:
                raise ValueError(
                    f"image_stack has shape {tuple(self.image_stack.shape)}, expected "
                    f"(num_particles, *extracted_box_size) = {expected}."
                )
            image_stack = self.image_stack

        local_stats = None
        if include_local_stats:
            if not self.local_stats:
                raise ValueError(
                    "local_stats is empty; cannot write to HDF5. Populate it "
                    "first, e.g. "
                    "self.local_stats.update(self.get_local_stat_maps())."
                )
            expected = (num_particles, *self._valid_local_stat_size())
            for column, stat_map in self.local_stats.items():
                if tuple(stat_map.shape) != expected:
                    raise ValueError(
                        f"local_stats['{column}'] has shape {tuple(stat_map.shape)}, "
                        f"expected {expected}."
                    )
            local_stats = dict(self.local_stats)

        metadata: dict[str, Any] = {
            "leopard_em_version": self.leopard_em_version,
            "extracted_box_size": self.extracted_box_size,
            "original_template_size": self.original_template_size,
            **{flag: getattr(self, flag) for flag in PREPROCESSING_FLAG_ATTRS},
        }
        write_particle_stack_hdf5(
            self.hdf5_path,
            metadata=metadata,
            df=df,
            image_stack=image_stack,
            local_stats=local_stats,
            position_columns=self._get_position_reference_columns(),
        )

        # Mirror what is now on disk
        self.image_stack_stored = image_stack is not None
        self._stored_image_stack = image_stack
        self.local_stats_stored = bool(local_stats)

    @classmethod
    def from_hdf5(
        cls,
        path: str,
        allow_file_overwrite: bool | None = None,
    ) -> "ParticleStackHDF5":
        """Load a ``ParticleStackHDF5`` from an existing HDF5 file.

        Equivalent to ``ParticleStackHDF5(hdf5_path=path)``.

        Parameters
        ----------
        path : str
            Path to the HDF5 file written by ``to_hdf5``.
        allow_file_overwrite : bool | None
            Deprecated and ignored.

        Returns
        -------
        ParticleStackHDF5
        """
        if allow_file_overwrite is not None:
            _warn_allow_file_overwrite_deprecated()
        return cls(hdf5_path=str(path))


def _particle_stack_tag(value: Any) -> str | None:
    """Discriminate CSV vs HDF5 particle stacks from an instance or input dict."""
    if isinstance(value, ParticleStackHDF5):
        return "hdf5"
    if isinstance(value, ParticleStackCSV):
        return "csv"
    if isinstance(value, dict):
        if "hdf5_path" in value:
            return "hdf5"
        if "df_path" in value:
            return "csv"
    return None


# Field type for a particle stack of either back-end. The back-end is picked from the
# input (``df_path`` -> CSV, ``hdf5_path`` -> HDF5), so only that class is built and
# validation errors name the right model.
AnyParticleStack = Annotated[
    Annotated[ParticleStackCSV, Tag("csv")] | Annotated[ParticleStackHDF5, Tag("hdf5")],
    Discriminator(_particle_stack_tag),
]


# ---------------------------------------------------------------------------
# Shared result-export helper
# ---------------------------------------------------------------------------

_SUFFIX_TO_FORMAT: dict[str, Literal["csv", "hdf5"]] = {
    ".csv": "csv",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".hdf": "hdf5",
    ".he5": "hdf5",
}


def _resolve_output_format(
    output_path: str,
    output_format: str | None,
    source_particle_stack: "_ParticleStackBase | None",
) -> Literal["csv", "hdf5"]:
    """Pick the output back-end: explicit > file extension > source back-end."""
    suffix_format = _SUFFIX_TO_FORMAT.get(Path(output_path).suffix.lower())
    if output_format is not None:
        if output_format not in ("csv", "hdf5"):
            raise ValueError(
                f"Unknown output_format '{output_format}'; expected 'csv' or 'hdf5'."
            )
        if suffix_format is not None and suffix_format != output_format:
            raise ValueError(
                f"output_format '{output_format}' conflicts with the file extension of "
                f"'{output_path}'."
            )
        return output_format  # type: ignore[return-value]
    if suffix_format is not None:
        return suffix_format
    if source_particle_stack is not None:
        return "hdf5" if isinstance(source_particle_stack, ParticleStackHDF5) else "csv"
    raise ValueError(
        "'output_format' must be specified when it cannot be inferred from the "
        f"extension of '{output_path}' or from 'source_particle_stack'."
    )


# pylint: disable=too-many-arguments
def export_particle_stack(
    df: pd.DataFrame,
    output_path: str,
    source_particle_stack: "_ParticleStackBase | None" = None,
    output_format: Literal["csv", "hdf5"] | None = None,
    allow_file_overwrite: bool | None = None,
    *,
    extracted_box_size: tuple[int, int] | None = None,
    original_template_size: tuple[int, int] | None = None,
) -> "ParticleStackCSV | ParticleStackHDF5":
    """Wrap a particle-result DataFrame in a ParticleStack and write it to disk.

    Notes
    -----
    Used by the refine/optimize/constrained-search managers so that their output back-
    end matches the back-end of the input particle stack by default, while still
    allowing an explicit override. The DataFrame is only an intermediate — the returned
    object is the actual particle stack, reusable directly (e.g. fed into the next
    program) without re-reading from disk. Only the particle table is written; stored
    tensors (image stack, local stats) are never carried over since particle positions
    typically change between programs.

    Parameters
    ----------
    df : pd.DataFrame
        The particle table to write (e.g. a match_template or refined result table).
        Must be a superset of the columns a `ParticleStackCSV`/`ParticleStackHDF5`
        expects; extra columns (e.g. `refined_*`) are preserved as-is. It is copied.
    output_path : str
        Destination file path. An existing file is replaced atomically.
    source_particle_stack : _ParticleStackBase | None
        The particle stack `df` was derived from. Supplies the default output format
        (matches its own back-end) and the shared box-size/pre-processing metadata to
        carry over to the new instance. Required unless both `extracted_box_size` and
        `original_template_size` are given.
    output_format : Literal["csv", "hdf5"] | None
        Explicit output back-end. If None (default), inferred from the extension of
        `output_path` (``.csv`` -> "csv"; ``.h5``/``.hdf5``/``.hdf``/``.he5`` ->
        "hdf5"), otherwise from ``type(source_particle_stack)``.
    allow_file_overwrite : bool | None
        Deprecated and ignored; files are always overwritten.
    extracted_box_size : tuple[int, int] | None
        Keyword-only. Extracted particle box size, ``(H, W)``. Overrides the source's
        value; required when `source_particle_stack` is not given.
    original_template_size : tuple[int, int] | None
        Keyword-only. Original template box size used during the search, ``(H, W)``.
        Overrides the source's value; required when `source_particle_stack` is not
        given.

    Returns
    -------
    ParticleStackCSV | ParticleStackHDF5
        The newly constructed particle stack, already written to
        ``output_path``.

    Raises
    ------
    ValueError
        If the box sizes are unavailable, the output format cannot be inferred or
        conflicts with the file extension, or `output_format` is not "csv" or "hdf5".
    """
    if allow_file_overwrite is not None:
        _warn_allow_file_overwrite_deprecated()

    output_format = _resolve_output_format(
        output_path, output_format, source_particle_stack
    )

    shared_kwargs: dict[str, Any] = {"skip_df_load": True}
    if source_particle_stack is not None:
        spc = source_particle_stack
        shared_kwargs.update(
            {
                "extracted_box_size": spc.extracted_box_size,
                "original_template_size": spc.original_template_size,
                "leopard_em_version": spc.leopard_em_version,
                "global_whitening_applied": spc.global_whitening_applied,
                "local_whitening_applied": spc.local_whitening_applied,
                "global_normalization_applied": spc.global_normalization_applied,
                "local_normalization_applied": spc.local_normalization_applied,
            }
        )
    if extracted_box_size is not None:
        shared_kwargs["extracted_box_size"] = extracted_box_size
    if original_template_size is not None:
        shared_kwargs["original_template_size"] = original_template_size
    if "extracted_box_size" not in shared_kwargs or (
        "original_template_size" not in shared_kwargs
    ):
        raise ValueError(
            "Either 'source_particle_stack' or both 'extracted_box_size' and "
            "'original_template_size' must be provided."
        )

    if output_format == "csv":
        csv_stack = ParticleStackCSV(df_path=output_path, **shared_kwargs)
        csv_stack._set_dataframe(df)  # pylint: disable=protected-access
        csv_stack.export_results()
        return csv_stack

    hdf5_stack = ParticleStackHDF5(hdf5_path=output_path, **shared_kwargs)
    hdf5_stack._set_dataframe(df)  # pylint: disable=protected-access
    hdf5_stack.to_hdf5()
    return hdf5_stack


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

# Existing code that imports `ParticleStack` continues to receive
# `ParticleStackCSV` unchanged.
ParticleStack = ParticleStackCSV
