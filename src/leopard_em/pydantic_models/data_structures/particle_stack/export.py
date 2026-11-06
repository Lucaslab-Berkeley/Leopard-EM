"""Back-end selection: ``AnyParticleStack``, result export and conversion."""

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import Discriminator, Tag

from leopard_em.pydantic_models.data_structures.particle_stack.base import (
    _ParticleStackBase,
    _warn_allow_file_overwrite_deprecated,
)
from leopard_em.pydantic_models.data_structures.particle_stack.csv_stack import (
    ParticleStackCSV,
)
from leopard_em.pydantic_models.data_structures.particle_stack.hdf5_stack import (
    ParticleStackHDF5,
)
from leopard_em.pydantic_models.formats import split_compression_suffix


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


def _shared_stack_kwargs(source: _ParticleStackBase) -> dict[str, Any]:
    """Box-size, version and pre-processing fields to carry over from ``source``."""
    return {
        "extracted_box_size": source.extracted_box_size,
        "original_template_size": source.original_template_size,
        "leopard_em_version": source.leopard_em_version,
        "global_whitening_applied": source.global_whitening_applied,
        "local_whitening_applied": source.local_whitening_applied,
        "global_normalization_applied": source.global_normalization_applied,
        "local_normalization_applied": source.local_normalization_applied,
    }


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
    source_particle_stack: _ParticleStackBase | None,
) -> Literal["csv", "hdf5"]:
    """Pick the output back-end: explicit > file extension > source back-end."""
    uncompressed_path, compression = split_compression_suffix(output_path)
    suffix_format = _SUFFIX_TO_FORMAT.get(Path(uncompressed_path).suffix.lower())
    if compression is not None:
        if "hdf5" in (suffix_format, output_format):
            raise ValueError(
                f"'{output_path}' has a compression suffix, which is only supported "
                "for CSV output."
            )
        suffix_format = "csv"
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


def _backing_path(stack: _ParticleStackBase) -> str | None:
    """The file a particle stack was loaded from / writes to, if any."""
    if isinstance(stack, ParticleStackHDF5):
        return stack.hdf5_path
    if isinstance(stack, ParticleStackCSV):
        return stack.df_path
    return None


def _same_file(path_a: str, path_b: str) -> bool:
    if os.path.realpath(path_a) == os.path.realpath(path_b):
        return True
    try:
        return os.path.samefile(path_a, path_b)
    except OSError:  # either path does not exist (yet)
        return False


def check_output_not_input(
    output_paths: Iterable[str], *input_stacks: _ParticleStackBase | None
) -> None:
    """Refuse to write a program's output over the file of an input particle stack.

    Parameters
    ----------
    output_paths : Iterable[str]
        Paths the program is about to write.
    *input_stacks : _ParticleStackBase | None
        The program's input particle stacks (None entries are skipped).

    Raises
    ------
    ValueError
        If any output path refers to an input stack's file.
    """
    for output_path in output_paths:
        for stack in input_stacks:
            input_path = _backing_path(stack) if stack is not None else None
            if input_path and _same_file(output_path, input_path):
                raise ValueError(
                    f"Output path '{output_path}' is the input particle stack's file "
                    f"'{input_path}'; writing it would replace the input. Choose a "
                    "different output path."
                )


# pylint: disable=too-many-arguments
def export_particle_stack(
    df: pd.DataFrame,
    output_path: str,
    source_particle_stack: _ParticleStackBase | None = None,
    output_format: Literal["csv", "hdf5"] | None = None,
    allow_file_overwrite: bool | None = None,
    *,
    extracted_box_size: tuple[int, int] | None = None,
    original_template_size: tuple[int, int] | None = None,
) -> ParticleStackCSV | ParticleStackHDF5:
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
        `output_path` (``.csv`` or a compression suffix such as ``.csv.gz`` -> "csv";
        ``.h5``/``.hdf5``/``.hdf``/``.he5`` -> "hdf5"), otherwise from
        ``type(source_particle_stack)``.
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
        conflicts with the file extension, `output_format` is not "csv" or "hdf5", or
        `output_path` is the file of `source_particle_stack`.
    """
    if allow_file_overwrite is not None:
        _warn_allow_file_overwrite_deprecated()

    check_output_not_input([output_path], source_particle_stack)
    output_format = _resolve_output_format(
        output_path, output_format, source_particle_stack
    )

    shared_kwargs: dict[str, Any] = {"skip_df_load": True}
    if source_particle_stack is not None:
        shared_kwargs.update(_shared_stack_kwargs(source_particle_stack))
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


def particle_stack_to_hdf5(
    stack: _ParticleStackBase,
    hdf5_path: str,
    *,
    include_image_stack: bool = False,
    include_local_stats: bool = False,
) -> ParticleStackHDF5:
    """Copy a particle stack (table and optional tensors) into a new HDF5 stack.

    Parameters
    ----------
    stack : _ParticleStackBase
        The particle stack to convert, of either back-end.
    hdf5_path : str
        Destination path for the HDF5 file. An existing file is replaced atomically.
    include_image_stack : bool, optional
        Write ``stack.image_stack`` to the HDF5 file, by default False. Raises
        ``ValueError`` if the image stack has not been loaded.
    include_local_stats : bool, optional
        Write every entry currently in ``stack.local_stats`` to the HDF5 file, by
        default False. Raises ``ValueError`` if ``local_stats`` is empty.

    Returns
    -------
    ParticleStackHDF5
        The new HDF5-backed stack instance pointing at ``hdf5_path``.
    """
    hdf5_stack = ParticleStackHDF5(
        hdf5_path=hdf5_path, skip_df_load=True, **_shared_stack_kwargs(stack)
    )
    # Particle IDs are generated on write if the table doesn't have any yet. Tensors
    # are assigned after the table, since replacing the table discards them.
    hdf5_stack._set_dataframe(stack.get_dataframe_copy())  # pylint: disable=protected-access
    if include_image_stack:
        hdf5_stack.image_stack = stack.image_stack
    if include_local_stats:
        hdf5_stack.local_stats = dict(stack.local_stats)
    hdf5_stack.to_hdf5(
        include_image_stack=include_image_stack,
        include_local_stats=include_local_stats,
    )
    return hdf5_stack


# Backs ``ParticleStackCSV.to_hdf5`` (public since v1.3) without csv_stack importing
# the HDF5 back-end.
# pylint: disable-next=protected-access
ParticleStackCSV._hdf5_converter = particle_stack_to_hdf5
