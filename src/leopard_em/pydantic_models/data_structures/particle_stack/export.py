"""Back-end selection: the ``AnyParticleStack`` field type and result export."""

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
    source_particle_stack: _ParticleStackBase | None,
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
