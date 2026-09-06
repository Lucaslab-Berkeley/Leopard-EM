"""Storage of sparse particle detections in a 2DTM search."""

import warnings

import h5py
import numpy as np
import pandas as pd
import torch

from leopard_em.backend.process_results import decode_global_search_index
from leopard_em.pydantic_models.custom_types import BaseModel2DTM

# Bumped whenever the on-disk HDF5 layout changes incompatibly. Version 2 added
# /search_space/euler_angles, which makes decoding independent of grid ordering.
CORRELATION_TABLE_FORMAT_VERSION = 2


def derive_orientation_grid_from_full_angles(
    euler_angles: torch.Tensor,
) -> tuple[list[tuple[float, float]], list[float]]:
    """Extract unique (phi, theta) pairs and psi values from a grid angles tensor.

    ``euler_angles`` is a Cartesian product of out-of-plane (phi, theta) pairs and
    in-plane psi values, but which of the two varies fastest depends on how the grid
    was generated. Both layouts are detected here rather than assumed:

    - *psi-outer* (what ``torch_so3`` produces): all (phi, theta) pairs for the first
      psi, then all pairs for the next psi, and so on.
    - *psi-inner*: all psi values for the first (phi, theta) pair, then the next pair.

    Assuming the wrong one silently returns nonsense -- notably a ``psi_angles`` list
    of identical values.

    Parameters
    ----------
    euler_angles : torch.Tensor
        All Euler angles used in the search, shape (num_orientations, 3), in ZYZ
        convention (degrees).

    Returns
    -------
    tuple[list[tuple[float, float]], list[float]]
        - ``phi_theta_angles``: list of unique (phi, theta) pairs, one per out-of-plane
          orientation, in the order they appear in the search.
        - ``psi_angles``: list of unique psi values used in the search.

    Notes
    -----
    These axes are descriptive metadata. The authoritative way to map a
    ``search_index`` back to angles is to index the full ``euler_angles`` array, which
    does not depend on the layout at all -- see
    :meth:`CorrelationTable.to_detections_dataframe`.
    """
    n_orientations = euler_angles.shape[0]
    n_psi = int(torch.unique(euler_angles[:, 2]).shape[0])
    n_phi_theta = n_orientations // n_psi

    # If psi is constant across the first n_phi_theta rows then (phi, theta) is the
    # fast axis, i.e. the grid is psi-outer.
    psi_is_outer = bool(torch.all(euler_angles[:n_phi_theta, 2] == euler_angles[0, 2]))

    if psi_is_outer:
        phi_theta_source = euler_angles[:n_phi_theta]
        psi_source = euler_angles[::n_phi_theta, 2]
    else:
        phi_theta_source = euler_angles[::n_psi]
        psi_source = euler_angles[:n_psi, 2]

    phi_theta_angles = [
        (float(row[0]), float(row[1])) for row in phi_theta_source[:n_phi_theta]
    ]
    psi_angles = psi_source[:n_psi].tolist()

    return phi_theta_angles, psi_angles


class CorrelationTable(BaseModel2DTM):
    """Correlation table data structure storing possible detections along a 2DTM search.

    Attributes
    ----------
    correlation_threshold : float
        Pre-defined threshold a cross-correlation value must surpass to be included
        in the correlation table.
    num_observations : int
        Total number of detections in the correlation table (number of search indices
        which surpassed the correlation threshold).
    defocus_offsets : list[float]
        List of defocus offsets (in Angstroms) used in the search.
    phi_theta_angles : list[tuple[float, float]]
        List out-of-plane rotation angles (in degrees, Euler angles phi and theta, in
        ZYZ convention) used in the search.
    psi_angles : list[float]
        List of in-plane rotation angles (in degrees, Euler angle psi, in ZYZ
        convention) used in the search.
    euler_angles : list[tuple[float, float, float]] | None
        Every orientation searched, shape (num_orientations, 3), as ZYZ Euler angles in
        degrees and in the exact order the search used them. This is what makes
        `search_index` decodable regardless of how the orientation grid was laid out;
        prefer it over `phi_theta_angles`/`psi_angles`. `None` only for tables loaded
        from files written before format version 2.
    search_index : list[int]
        Global search index identifying the defocus offset and orientation of each
        detection, as `defocus_index * num_orientations + orientation_index`. Length
        will be equal to `num_observations`.

        Decode it by indexing `euler_angles` directly (see
        `to_detections_dataframe`), *not* by combining `phi_theta_angles` and
        `psi_angles` -- the relative order of those two axes depends on how the grid
        was generated and is not recoverable from the axes alone.
    x : list[int]
        List of x-coordinates (in pixels) of the detections in the micrograph.
    y : list[int]
        List of y-coordinates (in pixels) of the detections in the micrograph.
    correlation_value : list[float]
        List of cross-correlation values for each detection.
    correlation_mean : list[float]
        List of mean cross-correlation values for each detection, calculated across all
        search indices for the same x/y coordinates.
    correlation_variance : list[float]
        Spread of the cross-correlation values for each detection, calculated across
        all search indices for the same x/y coordinates.

        Despite the name -- kept for consistency with the `correlation_variance` map
        written by `match_template` -- this holds the **standard deviation**, not the
        variance: `scale_mip` square-roots it in place. A z-score is therefore
        `(correlation_value - correlation_mean) / correlation_variance`, dividing
        directly rather than by a square root.

    Methods
    -------
    to_dataframe() -> pd.DataFrame
    from_dataframe(df: pd.DataFrame) -> CorrelationTable
    to_hdf5(file_path: str)
    from_hdf5(file_path: str) -> CorrelationTable
    from_match_template_results(...) -> CorrelationTable
    """

    correlation_threshold: float
    num_observations: int

    # Defining and indexing search space
    defocus_offsets: list[float]
    phi_theta_angles: list[tuple[float, float]]  # descriptive: out-of-plane rotations
    psi_angles: list[float]  # descriptive: in-plane rotations
    # Authoritative orientation axis; None only for legacy (pre-v2) files.
    euler_angles: list[tuple[float, float, float]] | None = None
    # defocus_index * num_orientations + orientation_index, length == num_observations
    search_index: list[int]

    # Other detection attributes
    x: list[int]
    y: list[int]
    correlation_value: list[float]
    correlation_mean: list[float]
    correlation_variance: list[float]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert per-detection data to a DataFrame.

        Search-space metadata is stored in ``df.attrs`` so that
        ``from_dataframe`` can reconstruct the full object.

        Returns
        -------
        pd.DataFrame
            One row per detection with columns: search_index, x, y,
            correlation_value, correlation_mean, correlation_variance.
        """
        df = pd.DataFrame(
            {
                "search_index": self.search_index,
                "x": self.x,
                "y": self.y,
                "correlation_value": self.correlation_value,
                "correlation_mean": self.correlation_mean,
                "correlation_variance": self.correlation_variance,
            }
        )
        df.attrs["correlation_threshold"] = self.correlation_threshold
        df.attrs["num_observations"] = self.num_observations
        df.attrs["defocus_offsets"] = self.defocus_offsets
        df.attrs["phi_theta_angles"] = self.phi_theta_angles
        df.attrs["psi_angles"] = self.psi_angles
        df.attrs["euler_angles"] = self.euler_angles
        return df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "CorrelationTable":
        """Reconstruct a CorrelationTable from a DataFrame produced by ``to_dataframe``.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with detection columns and search-space metadata in
            ``df.attrs``.

        Returns
        -------
        CorrelationTable
        """
        return cls(
            correlation_threshold=float(df.attrs["correlation_threshold"]),
            num_observations=int(df.attrs["num_observations"]),
            defocus_offsets=list(df.attrs["defocus_offsets"]),
            phi_theta_angles=[tuple(pair) for pair in df.attrs["phi_theta_angles"]],
            psi_angles=list(df.attrs["psi_angles"]),
            euler_angles=(
                [tuple(row) for row in df.attrs["euler_angles"]]
                if df.attrs.get("euler_angles") is not None
                else None
            ),
            search_index=df["search_index"].tolist(),
            x=df["x"].tolist(),
            y=df["y"].tolist(),
            correlation_value=df["correlation_value"].tolist(),
            correlation_mean=df["correlation_mean"].tolist(),
            correlation_variance=df["correlation_variance"].tolist(),
        )

    def to_detections_dataframe(
        self, euler_angles: torch.Tensor | None = None
    ) -> pd.DataFrame:
        """Decode every detection into explicit angles, defocus, and z-score.

        Unlike ``to_dataframe``, which round-trips the raw storage, this resolves each
        ``search_index`` into the orientation and defocus offset it refers to. Decoding
        indexes the full orientation list directly, so it does not care how the grid
        was laid out.

        Parameters
        ----------
        euler_angles : torch.Tensor | None
            All orientations used in the search, shape (num_orientations, 3), ZYZ in
            degrees. Only needed for legacy tables that predate format version 2 and
            therefore did not store the grid; otherwise the stored grid is used.

        Returns
        -------
        pd.DataFrame
            One row per detection with columns ``search_index``, ``x``, ``y``,
            ``phi``, ``theta``, ``psi``, ``relative_defocus``, ``correlation_value``,
            ``correlation_mean``, ``correlation_variance`` and ``z_score``.

        Raises
        ------
        ValueError
            If no orientation grid is available, if it does not match the stored
            search indices, or if the table came from a search over multiple pixel
            sizes (not representable in this format).

        Notes
        -----
        ``correlation_variance`` holds the *standard deviation* of the correlation
        distribution, not the variance, matching the map of the same name written by
        ``match_template``. The z-score therefore divides by it directly.
        """
        if euler_angles is None:
            if self.euler_angles is None:
                raise ValueError(
                    "This correlation table predates format version 2 and does not "
                    "store its orientation grid, so its detections cannot be decoded. "
                    "Pass the search's Euler angles explicitly via 'euler_angles' -- "
                    "regenerating them from the original config's "
                    "OrientationSearchConfig.euler_angles reproduces them exactly."
                )
            euler_angles = torch.tensor(self.euler_angles, dtype=torch.float64)

        euler_angles = torch.as_tensor(euler_angles, dtype=torch.float64)
        if euler_angles.ndim != 2 or euler_angles.shape[1] != 3:
            raise ValueError(
                f"euler_angles must have shape (num_orientations, 3), got "
                f"{tuple(euler_angles.shape)}."
            )

        defocus_values = torch.tensor(self.defocus_offsets, dtype=torch.float64)
        search_index = torch.tensor(self.search_index, dtype=torch.int64)

        num_slots = int(defocus_values.shape[0]) * int(euler_angles.shape[0])
        if search_index.numel() > 0 and int(search_index.max()) >= num_slots:
            raise ValueError(
                f"Largest search index ({int(search_index.max())}) exceeds the "
                f"{num_slots} (defocus x orientation) combinations implied by the "
                "provided grid. Either the grid does not match this table, or the "
                "search used multiple pixel sizes, which this format cannot express."
            )

        phi, theta, psi, defocus, _ = decode_global_search_index(
            search_index,
            torch.zeros(1, dtype=torch.float64),
            defocus_values,
            euler_angles,
        )

        correlation_value = np.asarray(self.correlation_value, dtype=np.float64)
        correlation_mean = np.asarray(self.correlation_mean, dtype=np.float64)
        # Named "variance" for consistency with the match_template output map, but the
        # stored quantity is a standard deviation (see scale_mip in backend).
        correlation_std = np.asarray(self.correlation_variance, dtype=np.float64)

        return pd.DataFrame(
            {
                "search_index": search_index.numpy(),
                "x": np.asarray(self.x, dtype=np.int64),
                "y": np.asarray(self.y, dtype=np.int64),
                "phi": phi.numpy(),
                "theta": theta.numpy(),
                "psi": psi.numpy(),
                "relative_defocus": defocus.numpy(),
                "correlation_value": correlation_value,
                "correlation_mean": correlation_mean,
                "correlation_variance": correlation_std,
                "z_score": np.divide(
                    correlation_value - correlation_mean,
                    correlation_std,
                    out=np.zeros_like(correlation_value),
                    where=correlation_std != 0,
                ),
            }
        )

    def to_hdf5(self, file_path: str) -> None:
        """Write this CorrelationTable to an HDF5 file.

        Layout::

            /metadata              (attrs: correlation_threshold, num_observations,
                                    format_version)
            /search_space/
                defocus_offsets    float32 1-D
                phi_theta_angles   float32 (n, 2)
                psi_angles         float32 1-D
                euler_angles       float32 (num_orientations, 3)  [format_version >= 2]
            /detections/
                search_index       int32 1-D
                x                  int32 1-D
                y                  int32 1-D
                correlation_value  float32 1-D
                correlation_mean   float32 1-D
                correlation_variance float32 1-D

        Parameters
        ----------
        file_path : str
            Destination HDF5 file path.
        """
        with h5py.File(file_path, "w") as f:
            meta = f.create_group("metadata")
            meta.attrs["correlation_threshold"] = self.correlation_threshold
            meta.attrs["num_observations"] = self.num_observations
            meta.attrs["format_version"] = CORRELATION_TABLE_FORMAT_VERSION

            search_space = f.create_group("search_space")
            if self.euler_angles is not None:
                search_space.create_dataset(
                    "euler_angles",
                    data=np.array(self.euler_angles, dtype=np.float32),
                )
            search_space.create_dataset(
                "defocus_offsets",
                data=np.array(self.defocus_offsets, dtype=np.float32),
            )
            search_space.create_dataset(
                "phi_theta_angles",
                data=np.array(self.phi_theta_angles, dtype=np.float32),
            )
            search_space.create_dataset(
                "psi_angles",
                data=np.array(self.psi_angles, dtype=np.float32),
            )

            detections = f.create_group("detections")
            detections.create_dataset(
                "search_index",
                data=np.array(self.search_index, dtype=np.int32),
            )
            detections.create_dataset("x", data=np.array(self.x, dtype=np.int32))
            detections.create_dataset("y", data=np.array(self.y, dtype=np.int32))
            detections.create_dataset(
                "correlation_value",
                data=np.array(self.correlation_value, dtype=np.float32),
            )
            detections.create_dataset(
                "correlation_mean",
                data=np.array(self.correlation_mean, dtype=np.float32),
            )
            detections.create_dataset(
                "correlation_variance",
                data=np.array(self.correlation_variance, dtype=np.float32),
            )

    @classmethod
    def from_hdf5(cls, file_path: str) -> "CorrelationTable":
        """Load a CorrelationTable from an HDF5 file written by ``to_hdf5``.

        Parameters
        ----------
        file_path : str
            Path to the HDF5 file.

        Returns
        -------
        CorrelationTable
        """
        with h5py.File(file_path, "r") as f:
            correlation_threshold = float(f["metadata"].attrs["correlation_threshold"])
            num_observations = int(f["metadata"].attrs["num_observations"])

            defocus_offsets = f["search_space/defocus_offsets"][:].tolist()
            phi_theta_raw = f["search_space/phi_theta_angles"][:]
            phi_theta_angles = [(float(row[0]), float(row[1])) for row in phi_theta_raw]
            psi_angles = f["search_space/psi_angles"][:].tolist()

            euler_angles = None
            if "search_space/euler_angles" in f:
                euler_angles = [
                    (float(row[0]), float(row[1]), float(row[2]))
                    for row in f["search_space/euler_angles"][:]
                ]
            else:
                warnings.warn(
                    f"'{file_path}' predates correlation table format version 2 and "
                    "does not store the full orientation grid, so detections cannot "
                    "be decoded to angles. Its 'psi_angles' and 'phi_theta_angles' "
                    "axes were written by a version that assumed the wrong grid "
                    "ordering and are very likely incorrect (a tell-tale sign is "
                    "every psi value being identical). Pass the search's Euler angles "
                    "to to_detections_dataframe() to decode it correctly.",
                    stacklevel=2,
                )

            search_index = f["detections/search_index"][:].tolist()
            x = f["detections/x"][:].tolist()
            y = f["detections/y"][:].tolist()
            correlation_value = f["detections/correlation_value"][:].tolist()
            correlation_mean = f["detections/correlation_mean"][:].tolist()
            correlation_variance = f["detections/correlation_variance"][:].tolist()

        return cls(
            correlation_threshold=correlation_threshold,
            num_observations=num_observations,
            defocus_offsets=defocus_offsets,
            phi_theta_angles=phi_theta_angles,
            psi_angles=psi_angles,
            euler_angles=euler_angles,
            search_index=search_index,
            x=x,
            y=y,
            correlation_value=correlation_value,
            correlation_mean=correlation_mean,
            correlation_variance=correlation_variance,
        )

    @classmethod
    def from_match_template_results(
        cls,
        processed_correlation_table: dict,
        defocus_values: torch.Tensor,
        euler_angles: torch.Tensor,
        correlation_average: torch.Tensor,
        correlation_variance_map: torch.Tensor,
    ) -> "CorrelationTable":
        """Construct a CorrelationTable from backend outputs.

        Parameters
        ----------
        processed_correlation_table : dict
            Output of ``process_correlation_table`` with an additional ``global_idx``
            key (list[int]). Expected keys: ``threshold``, ``global_idx``, ``x``,
            ``y``, ``correlation``.
        defocus_values : torch.Tensor
            Defocus offsets used in the search. Shape (num_defocus,).
        euler_angles : torch.Tensor
            All Euler angles used in the search, shape (num_orientations, 3), in ZYZ
            convention (degrees). Must be ordered as a grid: all psi values for the
            first (phi, theta) pair, then all psi values for the second pair, etc.
        correlation_average : torch.Tensor
            Per-pixel mean cross-correlation, shape (H, W).
        correlation_variance_map : torch.Tensor
            Per-pixel standard deviation of cross-correlation, shape (H, W).

        Returns
        -------
        CorrelationTable
        """
        threshold = processed_correlation_table["threshold"]
        global_idx = processed_correlation_table["global_idx"]  # list[int]
        pos_x = processed_correlation_table["x"]  # list[int]
        pos_y = processed_correlation_table["y"]  # list[int]
        corr_values = processed_correlation_table["correlation"]  # list[float]

        defocus_offsets = defocus_values.tolist()
        phi_theta_angles, psi_angles = derive_orientation_grid_from_full_angles(
            euler_angles
        )

        search_index = (
            list(global_idx) if isinstance(global_idx, list) else global_idx.tolist()
        )

        # Look up per-detection statistics from the pre-computed statistics tensors
        num_observations = len(pos_x)
        if num_observations > 0:
            x_tensor = torch.tensor(pos_x, dtype=torch.long)
            y_tensor = torch.tensor(pos_y, dtype=torch.long)
            det_mean = correlation_average[y_tensor, x_tensor].tolist()
            det_variance = correlation_variance_map[y_tensor, x_tensor].tolist()
        else:
            det_mean = []
            det_variance = []

        return cls(
            correlation_threshold=float(threshold),
            num_observations=num_observations,
            defocus_offsets=defocus_offsets,
            phi_theta_angles=phi_theta_angles,
            psi_angles=psi_angles,
            euler_angles=[
                (float(row[0]), float(row[1]), float(row[2])) for row in euler_angles
            ],
            search_index=search_index,
            x=list(pos_x),
            y=list(pos_y),
            correlation_value=list(corr_values),
            correlation_mean=det_mean,
            correlation_variance=det_variance,
        )


def detections_from_hdf5(
    file_path: str,
    euler_angles: torch.Tensor | None = None,
    min_z_score: float | None = None,
    chunk_size: int = 20_000_000,
) -> pd.DataFrame:
    """Stream a correlation table off disk straight into a decoded DataFrame.

    :meth:`CorrelationTable.from_hdf5` materialises every column as a Python list,
    which a full-micrograph table does not survive: 250 million detections need roughly
    40 GB that way against 6 GB as arrays. This reads in chunks, applies the score cut
    during the read so only survivors are ever held, and decodes as it goes.

    Parameters
    ----------
    file_path : str
        Correlation table written by ``match_template``.
    euler_angles : torch.Tensor | None
        Orientation grid, shape (num_orientations, 3). Only needed for tables written
        before format version 2, which did not store it.
    min_z_score : float | None
        Discard detections at or below this z-score while reading. Set it well below
        any analysis threshold -- the point of the table is to hold detections that a
        per-pixel maximum would have thrown away.
    chunk_size : int
        Detections per read.

    Returns
    -------
    pd.DataFrame
        Same columns as :meth:`CorrelationTable.to_detections_dataframe`.

    Raises
    ------
    ValueError
        If no orientation grid is available, or it does not match the stored indices.
    """
    with h5py.File(file_path, "r") as handle:
        defocus = torch.tensor(
            handle["search_space/defocus_offsets"][:], dtype=torch.float64
        )
        if "search_space/euler_angles" in handle:
            euler_angles = torch.tensor(
                handle["search_space/euler_angles"][:], dtype=torch.float64
            )
        elif euler_angles is None:
            raise ValueError(
                f"'{file_path}' predates format version 2 and stores no orientation "
                "grid; pass euler_angles explicitly."
            )

        grid = torch.as_tensor(euler_angles, dtype=torch.float64)
        num_slots = int(defocus.shape[0]) * int(grid.shape[0])
        detections = handle["detections"]
        total = detections["search_index"].shape[0]

        frames = []
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            value = detections["correlation_value"][start:stop]
            mean = detections["correlation_mean"][start:stop]
            spread = detections["correlation_variance"][start:stop]
            z_score = np.divide(
                value - mean,
                spread,
                out=np.zeros_like(value),
                where=spread != 0,
            )

            keep = (
                np.ones(len(z_score), dtype=bool)
                if min_z_score is None
                else z_score > min_z_score
            )
            if not keep.any():
                continue

            index = detections["search_index"][start:stop][keep].astype(np.int64)
            if int(index.max()) >= num_slots:
                raise ValueError(
                    f"Largest search index ({int(index.max())}) exceeds the "
                    f"{num_slots} combinations the supplied grid allows; the grid "
                    "does not match this table."
                )
            phi, theta, psi, relative_defocus, _ = decode_global_search_index(
                torch.from_numpy(index),
                torch.zeros(1, dtype=torch.float64),
                defocus,
                grid,
            )
            frames.append(
                pd.DataFrame(
                    {
                        "search_index": index,
                        "x": detections["x"][start:stop][keep].astype(np.int64),
                        "y": detections["y"][start:stop][keep].astype(np.int64),
                        "phi": phi.numpy(),
                        "theta": theta.numpy(),
                        "psi": psi.numpy(),
                        "relative_defocus": relative_defocus.numpy(),
                        "correlation_value": value[keep],
                        "correlation_mean": mean[keep],
                        "correlation_variance": spread[keep],
                        "z_score": z_score[keep],
                    }
                )
            )

    if not frames:
        return pd.DataFrame(
            columns=[
                "search_index",
                "x",
                "y",
                "phi",
                "theta",
                "psi",
                "relative_defocus",
                "correlation_value",
                "correlation_mean",
                "correlation_variance",
                "z_score",
            ]
        )

    return pd.concat(frames, ignore_index=True)
