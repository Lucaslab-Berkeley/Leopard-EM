"""Storage of sparse particle detections in a 2DTM search."""

import h5py
import numpy as np
import pandas as pd
import torch

from leopard_em.pydantic_models.custom_types import BaseModel2DTM


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
    euler_angles : list[tuple[float, float, float]] | None
        Every orientation searched, shape (num_orientations, 3), as ZYZ Euler angles
        in degrees and in the exact order the search used them. These angles describe
        passive rotations.
    search_index : list[int]
        Global search index identifying the defocus offset and orientation of each
        detection, as `defocus_index * num_orientations + orientation_index`. Length
        will be equal to `num_observations`.

        Decode it by indexing `euler_angles` directly -- `search_index %
        num_orientations` gives the row.
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
        List of variance of cross-correlation values for each detection, calculated
        across all search indices for the same x/y coordinates.

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
    # Authoritative orientation axis; None only for tables from older files.
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
            euler_angles=df.attrs.get("euler_angles"),
            search_index=df["search_index"].tolist(),
            x=df["x"].tolist(),
            y=df["y"].tolist(),
            correlation_value=df["correlation_value"].tolist(),
            correlation_mean=df["correlation_mean"].tolist(),
            correlation_variance=df["correlation_variance"].tolist(),
        )

    def to_hdf5(self, file_path: str, compress: bool = True) -> None:
        """Write this CorrelationTable to an HDF5 file.

        Layout::

            /metadata              (attrs: correlation_threshold, num_observations)
            /search_space/
                defocus_offsets    float32 1-D
                euler_angles       float32 (num_orientations, 3), gzip-4 + shuffle
                                   (if compress=True)
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
        compress : bool
            Whether to gzip-4 + byte-shuffle the ``euler_angles`` dataset.
        """
        compression_kwargs: dict = (
            {"compression": "gzip", "compression_opts": 4, "shuffle": True}
            if compress
            else {}
        )

        with h5py.File(file_path, "w") as f:
            meta = f.create_group("metadata")
            meta.attrs["correlation_threshold"] = self.correlation_threshold
            meta.attrs["num_observations"] = self.num_observations

            search_space = f.create_group("search_space")
            search_space.create_dataset(
                "defocus_offsets",
                data=np.array(self.defocus_offsets, dtype=np.float32),
            )
            if self.euler_angles is not None:
                search_space.create_dataset(
                    "euler_angles",
                    data=np.array(self.euler_angles, dtype=np.float32),
                    **compression_kwargs,
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

        Notes
        -----
        Files written with only ``phi_theta_angles`` and ``psi_angles`` (development and
        v1.3 exactly format) automatically get unpacked into full ZYZ Euler angles under
        ``euler_angles``. The order of the angles is preserved.
        """
        with h5py.File(file_path, "r") as f:
            correlation_threshold = float(f["metadata"].attrs["correlation_threshold"])
            num_observations = int(f["metadata"].attrs["num_observations"])

            defocus_offsets = f["search_space/defocus_offsets"][:].tolist()
            search_space = f["search_space"]
            full_angles = None
            if "euler_angles" in search_space:
                full_angles = search_space["euler_angles"][:].tolist()
            elif "phi_theta_angles" in search_space and "psi_angles" in search_space:
                phi_theta_angles = search_space["phi_theta_angles"][:].tolist()
                psi_angles = search_space["psi_angles"][:].tolist()
                full_angles = [
                    (phi, theta, psi)
                    for psi in psi_angles
                    for phi, theta in phi_theta_angles
                ]

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
            euler_angles=full_angles,
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
            convention (degrees), describing passive rotations, in the order the search
            used them. Any order is accepted -- it is stored verbatim, and
            ``search_index`` is resolved against it.
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
            defocus_offsets=defocus_values.tolist(),
            # Recorded verbatim: this is what makes search_index decodable.
            # `.tolist()` converts the whole tensor in one C call; row-by-row
            # `float()` conversion is over an order of magnitude slower on the
            # full search grid (1M+ orientations).
            euler_angles=[tuple(row) for row in euler_angles.tolist()],
            search_index=search_index,
            x=list(pos_x),
            y=list(pos_y),
            correlation_value=list(corr_values),
            correlation_mean=det_mean,
            correlation_variance=det_variance,
        )
