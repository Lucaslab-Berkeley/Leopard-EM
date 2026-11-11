"""HDF5-backed particle stack: table, image stack and local stats in one file."""

import warnings
from typing import Any

import torch
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

from leopard_em.pydantic_models.data_structures.particle_stack._hdf5_io import (
    BOX_SIZE_ATTRS,
    PREPROCESSING_FLAG_ATTRS,
    ParticleStackFileContents,
    StoredTensor,
    read_particle_stack_hdf5,
    write_particle_stack_hdf5,
)
from leopard_em.pydantic_models.data_structures.particle_stack._table import (
    check_required_columns,
    generate_particle_ids,
    normalize_particle_dataframe,
)
from leopard_em.pydantic_models.data_structures.particle_stack.base import (
    _ParticleStackBase,
    _warn_allow_file_overwrite_deprecated,
)
from leopard_em.pydantic_models.data_structures.particle_stack.geometry import (
    ParticleBoxGeometry,
)
from leopard_em.pydantic_models.formats import PARTICLE_ID_COLUMN


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
        check_required_columns(contents.df, source=f"'{self.hdf5_path}'")
        self._set_dataframe(
            contents.df,
            on_invalid_ids="raise" if contents.format_version >= 2 else "regenerate",
        )

        stack_position_columns = self.position_columns
        # Crops start `margin` pixels before the template footprint, so tensors cut
        # with the file's box/template sizes only line up if the margin is unchanged
        # (an explicit `original_template_size` override can change it).
        file_margin = None
        if all(key in contents.metadata for key in BOX_SIZE_ATTRS):
            file_margin = ParticleBoxGeometry(
                extracted_box_size=contents.metadata["extracted_box_size"],
                original_template_size=contents.metadata["original_template_size"],
            ).margin
        stack_margin = self.geometry.margin

        def _compatible(
            name: str, stored: StoredTensor, shape: tuple[int, ...]
        ) -> bool:
            if tuple(stored.data.shape) != shape:
                reason = (
                    f"shape {tuple(stored.data.shape)} does not match the expected "
                    f"{shape}"
                )
            elif file_margin is not None and file_margin != stack_margin:
                reason = (
                    f"its crop origin differs (cropped {file_margin} px before the "
                    f"template, not {stack_margin} px)"
                )
            elif stored.position_columns not in (None, stack_position_columns):
                reason = (
                    f"it was extracted at {stored.position_columns}, not "
                    f"{stack_position_columns}"
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

        valid_shape = (n, *self.geometry.valid_size)
        self.local_stats = {
            column: stored.data
            for column, stored in contents.local_stats.items()
            if _compatible(f"local_stats/{column}", stored, valid_shape)
        }

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
        check_required_columns(contents.df, source=f"'{self.hdf5_path}'")
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
            df.insert(0, PARTICLE_ID_COLUMN, generate_particle_ids(df))
        # Validate IDs, and keep `particle_id` as the first column on disk
        df = normalize_particle_dataframe(df, on_invalid_ids="raise")
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
            expected = (num_particles, *self.geometry.valid_size)
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
            position_columns=self.position_columns,
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
