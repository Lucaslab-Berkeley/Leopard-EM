"""CSV-backed particle stack (the original ``ParticleStack`` behavior)."""

import pandas as pd

from leopard_em.pydantic_models.data_structures.particle_stack._table import (
    check_required_columns,
)
from leopard_em.pydantic_models.data_structures.particle_stack.base import (
    _ParticleStackBase,
    _warn_allow_file_overwrite_deprecated,
)
from leopard_em.pydantic_models.data_structures.particle_stack.hdf5_stack import (
    ParticleStackHDF5,
)
from leopard_em.utils.data_io import atomic_write_path


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
        check_required_columns(tmp_df, source=f"'{self.df_path}'")
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
# Backward-compatibility alias
# ---------------------------------------------------------------------------

# Existing code that imports `ParticleStack` continues to receive
# `ParticleStackCSV` unchanged.
ParticleStack = ParticleStackCSV
