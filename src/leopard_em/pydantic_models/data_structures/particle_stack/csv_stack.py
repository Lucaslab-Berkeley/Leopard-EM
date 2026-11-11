"""CSV-backed particle stack (the original ``ParticleStack`` behavior)."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd

from leopard_em.pydantic_models.data_structures.particle_stack._table import (
    check_required_columns,
)
from leopard_em.pydantic_models.data_structures.particle_stack.base import (
    _ParticleStackBase,
    _warn_allow_file_overwrite_deprecated,
)
from leopard_em.pydantic_models.formats import split_compression_suffix
from leopard_em.utils.data_io import atomic_write_path

if TYPE_CHECKING:
    from leopard_em.pydantic_models.data_structures.particle_stack.hdf5_stack import (
        ParticleStackHDF5,
    )


def _hdf5_converter_unregistered(*_args: Any, **_kwargs: Any) -> "ParticleStackHDF5":
    """Placeholder until ``export`` registers ``particle_stack_to_hdf5``."""
    raise RuntimeError(
        "No CSV-to-HDF5 converter registered; import "
        "leopard_em.pydantic_models.data_structures.particle_stack."
    )


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

    # Back-end conversion lives in ``export`` (which imports this module), so it is
    # registered here by ``export`` rather than imported.
    _hdf5_converter: ClassVar[Callable[..., "ParticleStackHDF5"]] = (
        _hdf5_converter_unregistered
    )

    def load_df(self) -> None:
        """Load and validate the particle DataFrame from ``df_path``.

        Raises
        ------
        ValueError
            If required columns are missing from the CSV.
        """
        tmp_df = pd.read_csv(self.df_path)
        check_required_columns(tmp_df, source=f"'{self.df_path}'")
        self._set_dataframe(tmp_df, on_invalid_ids="regenerate")

    # ---------------------------------------------------------------------------
    # I/O methods
    # ---------------------------------------------------------------------------

    def export_results(self, allow_file_overwrite: bool | None = None) -> None:
        """Write the particle table to ``df_path`` as CSV.

        Notes
        -----
        A compression suffix on ``df_path`` (``.gz``, ``.bz2``, ``.zip``, ``.xz``, or
        ``.zst``) compresses the output accordingly.

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
        # The temporary path hides the suffix from pandas, so pass the compression
        uncompressed_path, method = split_compression_suffix(self.df_path)
        compression: dict[str, str] | None = None
        if method is not None:
            compression = {"method": method}
            if method == "zip":
                compression["archive_name"] = Path(uncompressed_path).name
        with atomic_write_path(self.df_path) as tmp_path:
            self._df.to_csv(tmp_path, index=False, compression=compression)

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

        return ParticleStackCSV._hdf5_converter(
            self,
            hdf5_path,
            include_image_stack=include_image_stack,
            include_local_stats=include_local_stats,
        )


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

# Existing code that imports `ParticleStack` continues to receive
# `ParticleStackCSV` unchanged.
ParticleStack = ParticleStackCSV
