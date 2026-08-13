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

import json
import os
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, ClassVar, Literal

import h5py
import numpy as np
import pandas as pd
import torch
from pydantic import ConfigDict, Field, model_validator
from torch.utils.checkpoint import checkpoint
from torch_fourier_shift import fourier_shift_dft_2d
from torch_grid_utils import coordinate_grid
from torch_motion_correction.correct_motion import get_pixel_shifts
from torch_motion_correction.deformation_field import DeformationField
from typing_extensions import Self

from leopard_em.pydantic_models.config import PreprocessingFilters
from leopard_em.pydantic_models.custom_types import (
    BaseModel2DTM,
    ExcludedTensor,
)
from leopard_em.pydantic_models.formats import MATCH_TEMPLATE_DF_COLUMN_ORDER
from leopard_em.utils.data_io import load_mrc_image
from leopard_em.utils.image_processing import dose_weight_movie_to_micrograph

TORCH_TO_NUMPY_PADDING_MODE = {
    "constant": "constant",
    "reflect": "reflect",
    "replicate": "edge",
}

_HDF5_PARTICLES_GROUP = "particles"
_HDF5_LOCAL_STATS_GROUP = "local_stats"
_HDF5_IMAGE_STACK_DATASET = "image_stack"
_HDF5_STRING_DTYPE = h5py.string_dtype()


# TODO: Make this a shared utility function across the package somehow
def _leopard_em_version() -> str:
    try:
        return version("leopard_em")
    except PackageNotFoundError:
        return "uninstalled"


def _any_nan_or_inf(s: pd.Series) -> bool:
    """Helper function to check if any value in the Series is NaN or infinite."""
    return bool(s.isna().any() or s.isin([float("inf"), float("-inf")]).any())


def _generate_particle_ids(df: pd.DataFrame) -> list[str]:
    """Generate particle IDs of the form ``{mic_stem}_{local_idx:05d}``."""
    ids: pd.Series = pd.Series("", index=df.index, dtype=object)
    for mic_path, group in df.groupby("micrograph_path", sort=False):
        stem = Path(str(mic_path)).stem
        for local_idx, row_label in enumerate(group.index):
            ids.at[row_label] = f"{stem}_{local_idx:05d}"

    res: list[str] = ids.tolist()
    return res


# TODO: Better management of Zernikie coefficient columns/arrays in the HDF5 format...
#       This is a lot of boilerplate code, and probably a better schema would eliminate
#       these parsing needs.
def _value_to_str(v: Any) -> str:
    """Serialize a value to a string for HDF5 storage."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return json.dumps(v)


def _str_to_value(s: str) -> Any:
    """Deserialize a string back to a Python value after HDF5 load."""
    if s == "":
        return None
    try:
        parsed = json.loads(s)
        if isinstance(parsed, (list, dict)):
            return parsed
        # Plain JSON scalars (numbers) that were originally strings stay as strings
        return s
    except (json.JSONDecodeError, ValueError):
        return s


# NOTE: How are the internals of the hdf5 particle stack being handled? Are they just a
#       pass through for the DataFrame type backed class (don't want this). Need to
#       implement things at the base class level somehow.
def _write_df_to_hdf5_group(f: h5py.File, df: pd.DataFrame) -> None:
    """Write a DataFrame's columns to ``f[_HDF5_PARTICLES_GROUP]``.

    Numeric columns are stored as float64 datasets.  String / object columns
    (including path columns, Zernike coefficient arrays, etc.) are serialized
    to variable-length UTF-8 strings via ``_value_to_str``.

    The dataset names match the DataFrame column names.  ``particle_id``
    (which may be the DataFrame index) is always written as an explicit
    dataset and listed first in ``attrs["columns"]``.
    """
    grp = f.create_group(_HDF5_PARTICLES_GROUP)

    # Build the list of columns to write, ensuring particle_id comes first.
    if df.index.name == "particle_id":
        particle_ids = df.index.tolist()
        col_names = ["particle_id", *list(df.columns)]
    else:
        # particle_id may be an ordinary column
        particle_ids = df["particle_id"].tolist() if "particle_id" in df.columns else []
        col_names = list(df.columns)

    grp.attrs["columns"] = col_names

    # Write particle_id dataset
    if particle_ids:
        grp.create_dataset(
            "particle_id",
            data=np.array([str(v) for v in particle_ids], dtype=object),
            dtype=_HDF5_STRING_DTYPE,
        )

    for col in df.columns:
        if col == "particle_id":
            # Already written above (or will be skipped if index)
            if df.index.name != "particle_id":
                continue
        series = df[col]
        if pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series):
            grp.create_dataset(col, data=series.to_numpy(dtype=np.float64))
        else:
            str_data = [_value_to_str(v) for v in series]
            grp.create_dataset(
                col,
                data=np.array(str_data, dtype=object),
                dtype=_HDF5_STRING_DTYPE,
            )


def _read_df_from_hdf5_group(f: h5py.File) -> pd.DataFrame:
    """Reconstruct a DataFrame from ``f[_HDF5_PARTICLES_GROUP]``.

    ``particle_id`` is restored as the pandas ``Index``.
    """
    grp = f[_HDF5_PARTICLES_GROUP]
    columns: list[str] = list(grp.attrs["columns"])

    data: dict[str, Any] = {}
    for col in columns:
        if col not in grp:
            continue
        raw = grp[col][:]
        if raw.dtype.kind in ("O", "S", "U"):
            decoded = [s.decode() if isinstance(s, bytes) else s for s in raw]
            data[col] = [_str_to_value(s) for s in decoded]
        else:
            data[col] = raw

    df = pd.DataFrame(data)

    if "particle_id" in df.columns:
        df = df.set_index("particle_id")
        df.index.name = "particle_id"

    return df


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
    for y, x in zip(pos_y, pos_x):
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
    for y, x in zip(pos_y, pos_x):
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
    local_stats_correlation_average : ExcludedTensor
        Per-particle local mean of the cross-correlation map, extracted from
        the valid cross-correlation region around each particle center.
        Shape ``(N, valid_h, valid_w)`` where
        ``valid_h = extracted_box_size[0] - original_template_size[0] + 1`` and
        ``valid_w = extracted_box_size[1] - original_template_size[1] + 1``.
        Not serialized to YAML/JSON.
    local_stats_correlation_variance : ExcludedTensor
        Per-particle local variance of the cross-correlation map.  Same shape
        as ``local_stats_correlation_average``.  Not serialized to YAML/JSON.
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

    # Private: tabular data (not part of Pydantic schema)
    # TODO: Move away from having a df-backed implementation in favor of either
    #       getter/setter methods OR private fields for the relevant data.
    _df: pd.DataFrame

    # Image and statistics tensors (excluded from YAML/JSON serialization)
    image_stack: ExcludedTensor
    local_stats_correlation_average: ExcludedTensor
    local_stats_correlation_variance: ExcludedTensor

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
            The value(s) to assign.
        """
        self._df.loc[:, column_name] = value

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
            - A list of pandas Index objects containing the row indexes for particles
              from each corresponding image
        """
        if column_name not in self._df.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

        image_index_groups = self._df.groupby(column_name).groups
        images_list = []
        indices = []
        for img_path, indexes in image_index_groups.items():
            img = load_mrc_image(img_path)
            images_list.append(img)
            indices.append(indexes)

        images_tensor = torch.stack(images_list, dim=0)
        return images_tensor, indices

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
        """Construct stack of images from the DataFrame (updates image_stack in-place).

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
        image_stack = torch.zeros((self.num_particles, *extraction_size), device=device)

        if images.shape[0] != len(indices):
            raise ValueError(
                f"Number of images ({images.shape[0]}) does not match the number of "
                f"indices ({len(indices)})."
            )

        for i, indexes in enumerate(indices):
            img = images[i]
            pos_y = self._df.loc[indexes, y_col].to_numpy().copy()
            pos_x = self._df.loc[indexes, x_col].to_numpy().copy()

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
            image_stack[indexes] = cropped_images

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
            Row indexes for particles from each corresponding micrograph.

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

            filter_stack[indexes] = cumulative_filter

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
    ) -> tuple[torch.Tensor, list[Any]]:
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
        torch.Tensor
            Image stack of shape ``(N, box_h, box_w)``.
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
            # Use provided subset of particles
            particle_indexes = [self._df.index[i] for i in particle_indices]
            num_particles_to_process = len(particle_indices)
        else:
            # Use all particles
            particle_indexes = self._df.index.tolist()
            num_particles_to_process = self.num_particles

        pos_y = self._df.loc[particle_indexes, y_col].to_numpy().copy()
        pos_x = self._df.loc[particle_indexes, x_col].to_numpy().copy()
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
                    frame_shifts = frame_shifts[particle_indices]
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
        aligned_particle_movies_rfft, particle_indexes = (
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
        for particle_index in range(num_particles_to_process):
            particle_dft = aligned_particle_movies_rfft[particle_index]

            # Get the actual dataframe index for this particle
            df_idx = particle_indexes[particle_index]
            df_loc = self._df.index.get_loc(df_idx)

            dw_sum = dose_weight_movie_to_micrograph(
                movie_fft=particle_dft,
                pixel_size=float(pixel_sizes[df_loc].item()),
                pre_exposure=pre_exposure,
                fluence_per_frame=fluence_per_frame,
                voltage=self._df["voltage"].to_numpy()[df_loc],
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

        missing_columns = [
            col for col in MATCH_TEMPLATE_DF_COLUMN_ORDER if col not in tmp_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing the following columns in DataFrame: {missing_columns}"
            )

        self._df = tmp_df

    def to_hdf5(
        self,
        hdf5_path: str,
        allow_file_overwrite: bool = False,
        include_image_stack: bool = False,
        include_local_stats: bool = False,
    ) -> "ParticleStackHDF5":
        """Convert this CSV-backed stack to an HDF5-backed stack and write to disk.

        Parameters
        ----------
        hdf5_path : str
            Destination path for the HDF5 file.
        allow_file_overwrite : bool, optional
            Whether to overwrite an existing file, by default False.
        include_image_stack : bool, optional
            Write ``image_stack`` to the HDF5 file, by default False.
            Raises ``ValueError`` if the image stack has not been loaded.
        include_local_stats : bool, optional
            Write per-particle local stats to the HDF5 file, by default False.
            Raises ``ValueError`` if the local stats have not been set.

        Returns
        -------
        ParticleStackHDF5
            The new HDF5-backed stack instance pointing at ``hdf5_path``.
        """
        # Generate particle_id for each row and add to the copied DataFrame
        df = self._df.copy()
        particle_ids = _generate_particle_ids(df)
        df.insert(0, "particle_id", particle_ids)
        df = df.set_index("particle_id")
        df.index.name = "particle_id"

        hdf5_stack = ParticleStackHDF5(
            hdf5_path=hdf5_path,
            allow_file_overwrite=allow_file_overwrite,
            extracted_box_size=self.extracted_box_size,
            original_template_size=self.original_template_size,
            leopard_em_version=self.leopard_em_version,
            global_whitening_applied=self.global_whitening_applied,
            local_whitening_applied=self.local_whitening_applied,
            global_normalization_applied=self.global_normalization_applied,
            local_normalization_applied=self.local_normalization_applied,
            image_stack=self.image_stack if include_image_stack else None,
            local_stats_correlation_average=(
                self.local_stats_correlation_average if include_local_stats else None
            ),
            local_stats_correlation_variance=(
                self.local_stats_correlation_variance if include_local_stats else None
            ),
            skip_df_load=True,
        )
        hdf5_stack._df = df  # pylint: disable=protected-access
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

    The particle table, optional image stack, and optional per-particle local
    correlation statistics are all held in one ``.h5`` file.  Two loading
    modes are supported — choose one; mixing them raises errors:

    * **Load from referenced files**: ``image_stack`` and ``local_stats`` are
      computed from the paths stored in the particle table.  The HDF5 file
      stores only the particle table (``image_stack_stored=False``).
    * **Load from HDF5**: ``image_stack`` and ``local_stats`` are read
      directly from the HDF5 datasets (``image_stack_stored=True`` and/or
      ``local_stats_stored=True``).

    HDF5 file layout
    ----------------

    ::

        / (root)
        │  attrs: leopard_em_version, extracted_box_size, original_template_size,
        │         image_stack_stored, local_stats_stored,
        │         global_whitening_applied, local_whitening_applied,
        │         global_normalization_applied, local_normalization_applied
        ├─ particles/
        │      particle_id            (N,)   variable-length str  "{mic_stem}_{idx:05d}"
        │      <column>               (N,)   float64 or variable-length str
        │      ...
        ├─ image_stack                (N, box_h, box_w)             float32  [optional]
        └─ local_stats/                                                      [optional]
               correlation_average    (N, valid_h, valid_w)         float32
               correlation_variance   (N, valid_h, valid_w)         float32

    where ``valid_h = extracted_box_size[0] - original_template_size[0] + 1``
    and   ``valid_w = extracted_box_size[1] - original_template_size[1] + 1``.

    Attributes
    ----------
    hdf5_path : str
        Path to the HDF5 file.
    allow_file_overwrite : bool
        Whether to permit overwriting an existing file, by default False.
    image_stack_stored : bool
        True when ``/image_stack`` is present in the HDF5 file.
    local_stats_stored : bool
        True when ``/local_stats`` group is present in the HDF5 file.
    """

    hdf5_path: str
    allow_file_overwrite: bool = False
    image_stack_stored: bool = False
    local_stats_stored: bool = False

    ###########################
    ### Pydantic Validators ###
    ###########################

    @model_validator(mode="after")  # type: ignore
    def _validate_hdf5_path(self) -> Self:
        """Validate that the HDF5 path is writable and the overwrite policy is met.

        Returns
        -------
        Self

        Raises
        ------
        ValueError
            If the path is not writable or the file exists and overwrite is
            disabled.
        """
        directory = str(Path(self.hdf5_path).parent)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        if directory and not os.access(directory, os.W_OK):
            raise ValueError(
                f"Directory '{directory}' does not permit writing to "
                f"'{self.hdf5_path}'."
            )
        if not self.allow_file_overwrite and os.path.exists(self.hdf5_path):
            raise ValueError(
                f"File '{self.hdf5_path}' already exists but "
                "'allow_file_overwrite' is False."
            )
        return self

    ###########################
    ### Data loading        ###
    ###########################

    def load_df(self) -> None:
        """Load the particle DataFrame from the HDF5 file at ``hdf5_path``.

        Raises
        ------
        FileNotFoundError
            If ``hdf5_path`` does not exist.
        """
        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(
                f"HDF5 file '{self.hdf5_path}' does not exist. "
                "Pass skip_df_load=True if you intend to write a new file."
            )
        with h5py.File(self.hdf5_path, "r") as f:
            self._df = _read_df_from_hdf5_group(f)

    ###########################
    ### I/O methods         ###
    ###########################

    def to_hdf5(
        self,
        include_image_stack: bool = False,
        include_local_stats: bool = False,
    ) -> None:
        """Write the particle table and optional tensors to ``hdf5_path``.

        Parameters
        ----------
        include_image_stack : bool, optional
            Write ``image_stack`` to ``/image_stack``, by default False.
            Raises ``ValueError`` if ``image_stack`` is None.
        include_local_stats : bool, optional
            Write per-particle correlation stats to ``/local_stats``, by
            default False.  Raises ``ValueError`` if either stats tensor is
            None.
        """
        with h5py.File(self.hdf5_path, "w") as f:
            # Root attributes — metadata
            f.attrs["leopard_em_version"] = self.leopard_em_version
            f.attrs["extracted_box_size"] = list(self.extracted_box_size)
            f.attrs["original_template_size"] = list(self.original_template_size)
            f.attrs["global_whitening_applied"] = self.global_whitening_applied
            f.attrs["local_whitening_applied"] = self.local_whitening_applied
            f.attrs["global_normalization_applied"] = self.global_normalization_applied
            f.attrs["local_normalization_applied"] = self.local_normalization_applied

            # Particle table
            _write_df_to_hdf5_group(f, self._df)

            # Optional image stack
            if include_image_stack:
                if self.image_stack is None:
                    raise ValueError(
                        "image_stack is None; cannot write to HDF5. "
                        "Call construct_image_stack() first."
                    )
                f.create_dataset(
                    _HDF5_IMAGE_STACK_DATASET,
                    data=self.image_stack.cpu().to(torch.float32).numpy(),
                )
                self.image_stack_stored = True

            f.attrs["image_stack_stored"] = self.image_stack_stored

            # Optional per-particle local stats
            if include_local_stats:
                if (
                    self.local_stats_correlation_average is None
                    or self.local_stats_correlation_variance is None
                ):
                    raise ValueError(
                        "local_stats tensors are None; cannot write to HDF5."
                    )
                local_grp = f.create_group(_HDF5_LOCAL_STATS_GROUP)
                local_grp.create_dataset(
                    "correlation_average",
                    data=self.local_stats_correlation_average.cpu()
                    .to(torch.float32)
                    .numpy(),
                )
                local_grp.create_dataset(
                    "correlation_variance",
                    data=self.local_stats_correlation_variance.cpu()
                    .to(torch.float32)
                    .numpy(),
                )
                self.local_stats_stored = True

            f.attrs["local_stats_stored"] = self.local_stats_stored

    @classmethod
    def from_hdf5(
        cls,
        path: str,
        allow_file_overwrite: bool = True,
    ) -> "ParticleStackHDF5":
        """Load a ``ParticleStackHDF5`` from an existing HDF5 file.

        Parameters
        ----------
        path : str
            Path to the HDF5 file written by ``to_hdf5``.
        allow_file_overwrite : bool, optional
            Passed to the constructor so that the model validator does not
            reject the path of the file being loaded, by default True.

        Returns
        -------
        ParticleStackHDF5
        """
        with h5py.File(path, "r") as f:
            leopard_em_version = str(f.attrs.get("leopard_em_version", "unknown"))
            # pylint: disable=not-an-iterable
            extracted_box_size = tuple(int(v) for v in f.attrs["extracted_box_size"])
            original_template_size = tuple(
                int(v) for v in f.attrs["original_template_size"]
            )
            # pylint: enable=not-an-iterable
            global_whitening_applied = bool(
                f.attrs.get("global_whitening_applied", False)
            )
            local_whitening_applied = bool(
                f.attrs.get("local_whitening_applied", False)
            )
            global_normalization_applied = bool(
                f.attrs.get("global_normalization_applied", False)
            )
            local_normalization_applied = bool(
                f.attrs.get("local_normalization_applied", False)
            )
            image_stack_stored = bool(f.attrs.get("image_stack_stored", False))
            local_stats_stored = bool(f.attrs.get("local_stats_stored", False))

            df = _read_df_from_hdf5_group(f)

            image_stack: torch.Tensor | None = None
            if image_stack_stored:
                if _HDF5_IMAGE_STACK_DATASET not in f:
                    raise ValueError(
                        f"'image_stack_stored' is True but dataset "
                        f"'{_HDF5_IMAGE_STACK_DATASET}' is absent in '{path}'."
                    )
                image_stack = torch.from_numpy(f[_HDF5_IMAGE_STACK_DATASET][:])

            local_avg: torch.Tensor | None = None
            local_var: torch.Tensor | None = None
            if local_stats_stored:
                if _HDF5_LOCAL_STATS_GROUP not in f:
                    raise ValueError(
                        f"'local_stats_stored' is True but group "
                        f"'{_HDF5_LOCAL_STATS_GROUP}' is absent in '{path}'."
                    )
                local_grp = f[_HDF5_LOCAL_STATS_GROUP]
                local_avg = torch.from_numpy(local_grp["correlation_average"][:])
                local_var = torch.from_numpy(local_grp["correlation_variance"][:])

        instance = cls(
            hdf5_path=str(path),
            allow_file_overwrite=allow_file_overwrite,
            extracted_box_size=extracted_box_size,
            original_template_size=original_template_size,
            leopard_em_version=leopard_em_version,
            global_whitening_applied=global_whitening_applied,
            local_whitening_applied=local_whitening_applied,
            global_normalization_applied=global_normalization_applied,
            local_normalization_applied=local_normalization_applied,
            image_stack_stored=image_stack_stored,
            local_stats_stored=local_stats_stored,
            image_stack=image_stack,
            local_stats_correlation_average=local_avg,
            local_stats_correlation_variance=local_var,
            skip_df_load=True,
        )
        instance._df = df
        return instance


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

# Existing code that imports `ParticleStack` continues to receive
# `ParticleStackCSV` unchanged.
ParticleStack = ParticleStackCSV
