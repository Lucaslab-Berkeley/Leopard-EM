"""On-disk schema and codec for ``ParticleStackHDF5`` files.

Private module: ``ParticleStackHDF5`` is the public interface to these files.

File format versions
--------------------
* **1** (Leopard-EM v1.3, no ``format_version`` attribute): numeric columns stored as
  float64, everything else as JSON-ish strings.
* **2**: each ``/particles`` dataset carries an ``encoding`` attribute. ``"numeric"``
  datasets keep their native dtype (int, float, bool). ``"str"`` datasets hold
  variable-length UTF-8 strings where missing values are stored as ``""`` and
  list/dict cells as JSON text; they are read back as plain strings (``""`` -> None),
  i.e. the same in-memory representation a CSV round trip gives.

Both versions share the same group/dataset layout, documented on
``ParticleStackHDF5``.
"""

import json
import os
import warnings
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

from leopard_em.utils.data_io import atomic_write_path

FORMAT_VERSION = 2

PARTICLES_GROUP = "particles"
LOCAL_STATS_GROUP = "local_stats"
IMAGE_STACK_DATASET = "image_stack"

_STRING_DTYPE = h5py.string_dtype()
_ENCODING_ATTR = "encoding"
_ENCODING_NUMERIC = "numeric"
_ENCODING_STR = "str"
_POSITION_COLUMNS_ATTR = "position_columns"

BOX_SIZE_ATTRS = ("extracted_box_size", "original_template_size")
PREPROCESSING_FLAG_ATTRS = (
    "global_whitening_applied",
    "local_whitening_applied",
    "global_normalization_applied",
    "local_normalization_applied",
)


def _leopard_em_version() -> str:
    try:
        return version("leopard_em")
    except PackageNotFoundError:
        return "uninstalled"


@dataclass
class StoredTensor:
    """A tensor read from the file, with the position columns it was extracted at."""

    data: torch.Tensor
    position_columns: tuple[str, ...] | None = None


@dataclass
class ParticleStackFileContents:
    """Everything read from a ``ParticleStackHDF5`` file."""

    format_version: int
    metadata: dict[str, Any]
    df: pd.DataFrame
    image_stack_stored: bool
    local_stats_stored: bool
    image_stack: StoredTensor | None = None
    local_stats: dict[str, StoredTensor] = field(default_factory=dict)


###############
### Writing ###
###############


def _encode_str_cell(value: Any) -> str:
    """Encode one non-numeric cell as a string ("" for missing values)."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list | tuple | dict):
        return json.dumps(value)
    if value is None or (np.ndim(value) == 0 and pd.isna(value)):
        return ""
    return str(value)


def _numeric_array(series: pd.Series) -> np.ndarray | None:
    """Return ``series`` as a native numeric/bool array, or None if it isn't one."""
    dtype = series.dtype
    if isinstance(dtype, np.dtype):
        return series.to_numpy() if dtype.kind in "biuf" else None
    # Nullable extension dtypes (Int64, Float64, boolean)
    if pd.api.types.is_bool_dtype(dtype):
        return None if series.isna().any() else series.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(dtype):
        target = np.float64 if series.isna().any() else dtype.numpy_dtype
        values: np.ndarray = series.to_numpy(dtype=target, na_value=np.nan)
        return values
    return None


def _write_particle_table(f: h5py.File, df: pd.DataFrame) -> None:
    """Write the particle table to ``f[PARTICLES_GROUP]`` (format version 2)."""
    grp = f.create_group(PARTICLES_GROUP)
    grp.attrs["columns"] = [str(col) for col in df.columns]
    for col in df.columns:
        series = df[col]
        numeric = _numeric_array(series)
        if numeric is not None:
            dataset = grp.create_dataset(str(col), data=numeric)
            dataset.attrs[_ENCODING_ATTR] = _ENCODING_NUMERIC
        else:
            encoded = np.array([_encode_str_cell(v) for v in series], dtype=object)
            dataset = grp.create_dataset(str(col), data=encoded, dtype=_STRING_DTYPE)
            dataset.attrs[_ENCODING_ATTR] = _ENCODING_STR


def _write_tensor(
    parent: h5py.Group,
    name: str,
    tensor: torch.Tensor,
    position_columns: tuple[str, ...] | None,
) -> None:
    dataset = parent.create_dataset(
        name, data=tensor.detach().cpu().to(torch.float32).numpy()
    )
    if position_columns is not None:
        dataset.attrs[_POSITION_COLUMNS_ATTR] = list(position_columns)


# pylint: disable=too-many-arguments
def write_particle_stack_hdf5(
    path: str | os.PathLike,
    metadata: dict[str, Any],
    df: pd.DataFrame,
    image_stack: torch.Tensor | None = None,
    local_stats: dict[str, torch.Tensor] | None = None,
    position_columns: tuple[str, ...] | None = None,
) -> None:
    """Atomically write a particle stack file (format version 2).

    Parameters
    ----------
    path : str | os.PathLike
        Destination path; an existing file is replaced atomically.
    metadata : dict[str, Any]
        Root attributes (``leopard_em_version``, box sizes, pre-processing flags).
    df : pd.DataFrame
        The particle table. Written column by column in its current order.
    image_stack : torch.Tensor | None
        Optional ``(N, box_h, box_w)`` particle image stack.
    local_stats : dict[str, torch.Tensor] | None
        Optional per-particle local statistic maps keyed by ``*_path`` column.
    position_columns : tuple[str, ...] | None
        The ``(y, x)`` position columns the tensors were extracted at.
    """
    with atomic_write_path(path) as tmp_path, h5py.File(tmp_path, "w") as f:
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["writer_version"] = _leopard_em_version()
        for key, value in metadata.items():
            f.attrs[key] = list(value) if isinstance(value, tuple) else value

        _write_particle_table(f, df)

        if image_stack is not None:
            _write_tensor(f, IMAGE_STACK_DATASET, image_stack, position_columns)
        f.attrs["image_stack_stored"] = image_stack is not None

        if local_stats:
            local_grp = f.create_group(LOCAL_STATS_GROUP)
            for column, stat_map in local_stats.items():
                _write_tensor(local_grp, column, stat_map, position_columns)
        f.attrs["local_stats_stored"] = bool(local_stats)


###############
### Reading ###
###############


def _decode_strings(raw: np.ndarray) -> list[str | None]:
    """Decode an HDF5 string dataset; empty strings become None."""
    decoded = (v.decode() if isinstance(v, bytes) else str(v) for v in raw)
    return [v if v != "" else None for v in decoded]


def _read_particle_table(f: h5py.File) -> pd.DataFrame:
    """Read ``f[PARTICLES_GROUP]`` into a DataFrame (any format version)."""
    grp = f[PARTICLES_GROUP]
    # pylint: disable=not-an-iterable
    columns = [
        c.decode() if isinstance(c, bytes) else str(c) for c in grp.attrs["columns"]
    ]
    data: dict[str, Any] = {}
    for col in columns:
        if col not in grp:
            continue
        raw = grp[col][()]
        if raw.dtype.kind in "OSU":
            # NOTE: version-1 files JSON-encoded list/dict cells too. These are kept
            # as strings (not parsed back) so both versions, and CSV-backed stacks,
            # share one in-memory representation.
            data[col] = _decode_strings(raw)
        else:
            data[col] = raw
    return pd.DataFrame(data, columns=[c for c in columns if c in data])


def _read_tensor(dataset: h5py.Dataset) -> StoredTensor:
    columns = dataset.attrs.get(_POSITION_COLUMNS_ATTR)
    position_columns = (
        tuple(c.decode() if isinstance(c, bytes) else str(c) for c in columns)
        if columns is not None
        else None
    )
    return StoredTensor(torch.from_numpy(dataset[()]), position_columns)


def _read_metadata(attrs: h5py.AttributeManager) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if "leopard_em_version" in attrs:
        metadata["leopard_em_version"] = str(attrs["leopard_em_version"])
    for key in BOX_SIZE_ATTRS:
        if key in attrs:
            metadata[key] = tuple(int(v) for v in attrs[key])
    for key in PREPROCESSING_FLAG_ATTRS:
        if key in attrs:
            metadata[key] = bool(attrs[key])
    return metadata


def _check_stored_flag(
    attrs: h5py.AttributeManager, name: str, present: bool, path: str
) -> None:
    """Warn when a ``*_stored`` attribute disagrees with the file's contents."""
    if name in attrs and bool(attrs[name]) != present:
        warnings.warn(
            f"'{path}' has attribute {name}={bool(attrs[name])}, but the dataset is "
            f"{'present' if present else 'absent'}; trusting the file contents.",
            UserWarning,
            stacklevel=4,
        )


def read_particle_stack_hdf5(
    path: str | os.PathLike,
    load_image_stack: bool = True,
    load_local_stats: bool = True,
) -> ParticleStackFileContents:
    """Read a particle stack file written by any Leopard-EM version.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the HDF5 file.
    load_image_stack : bool
        Read ``/image_stack`` into memory if present. Default True.
    load_local_stats : bool
        Read ``/local_stats`` into memory if present. Default True.

    Returns
    -------
    ParticleStackFileContents

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is not a particle stack file, or was written by a newer,
        unsupported format version.
    """
    path_str = str(path)
    if not Path(path_str).is_file():
        raise FileNotFoundError(
            f"HDF5 file '{path_str}' does not exist. "
            "Pass skip_df_load=True if you intend to write a new file."
        )

    with h5py.File(path_str, "r") as f:
        format_version = int(f.attrs.get("format_version", 1))
        if format_version > FORMAT_VERSION:
            raise ValueError(
                f"'{path_str}' uses particle stack format version {format_version}, "
                f"but this Leopard-EM supports up to version {FORMAT_VERSION}. "
                "Please upgrade Leopard-EM."
            )
        if PARTICLES_GROUP not in f:
            raise ValueError(
                f"'{path_str}' is not a particle stack file: missing the "
                f"'/{PARTICLES_GROUP}' group."
            )

        image_stack_present = IMAGE_STACK_DATASET in f
        local_stats_present = LOCAL_STATS_GROUP in f and len(f[LOCAL_STATS_GROUP]) > 0
        _check_stored_flag(f.attrs, "image_stack_stored", image_stack_present, path_str)
        _check_stored_flag(f.attrs, "local_stats_stored", local_stats_present, path_str)

        contents = ParticleStackFileContents(
            format_version=format_version,
            metadata=_read_metadata(f.attrs),
            df=_read_particle_table(f),
            image_stack_stored=image_stack_present,
            local_stats_stored=local_stats_present,
        )
        if image_stack_present and load_image_stack:
            contents.image_stack = _read_tensor(f[IMAGE_STACK_DATASET])
        if local_stats_present and load_local_stats:
            local_grp = f[LOCAL_STATS_GROUP]
            contents.local_stats = {
                column: _read_tensor(local_grp[column]) for column in local_grp
            }

    return contents
