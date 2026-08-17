"""Storage of sparse particle detections in a 2DTM search."""

import h5py
import numpy as np
import pandas as pd
import torch

from leopard_em.pydantic_models.custom_types import BaseModel2DTM


def derive_orientation_grid_from_full_angles(
    euler_angles: torch.Tensor,
) -> tuple[list[tuple[float, float]], list[float]]:
    """Extract unique (phi, theta) pairs and psi values from a grid angles tensor.

    Assumes ``euler_angles`` is ordered as a Cartesian product: all psi values for
    the first (phi, theta) pair, then all psi values for the second pair, etc.

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
        - ``psi_angles``: list of unique psi values, in the order they cycle within each
          (phi, theta) group.
    """
    n_orientations = euler_angles.shape[0]
    n_psi = int(torch.unique(euler_angles[:, 2]).shape[0])
    n_phi_theta = n_orientations // n_psi

    phi_theta_angles = [
        (float(euler_angles[i * n_psi, 0]), float(euler_angles[i * n_psi, 1]))
        for i in range(n_phi_theta)
    ]
    psi_angles = euler_angles[:n_psi, 2].tolist()

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
    search_index : list[int]
        Global search index defining defocus offset, phi/theta angles, and psi angle for
        each detection. Calculated as `i * (n_j * n_k) + j * n_k + k`, where `i` is the
        index of the defocus offset, `j` is the index of the phi/theta angles, and `k`
        is the index of the psi angle. Length will be equal to `num_observations`.
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
    defocus_offsets: list[float]  # index 'i'
    phi_theta_angles: list[tuple[float, float]]  # index 'j', out-of-plane rotations
    psi_angles: list[float]  # index 'k', in-plane rotations
    search_index: list[int]  # i * (n_j * n_k) + j * n_k + k, length == num_observations

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
            search_index=df["search_index"].tolist(),
            x=df["x"].tolist(),
            y=df["y"].tolist(),
            correlation_value=df["correlation_value"].tolist(),
            correlation_mean=df["correlation_mean"].tolist(),
            correlation_variance=df["correlation_variance"].tolist(),
        )

    def to_hdf5(self, file_path: str) -> None:
        """Write this CorrelationTable to an HDF5 file.

        Layout::

            /metadata              (attrs: correlation_threshold, num_observations)
            /search_space/
                defocus_offsets    float32 1-D
                phi_theta_angles   float32 (n, 2)
                psi_angles         float32 1-D
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

            search_space = f.create_group("search_space")
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
            search_index=search_index,
            x=list(pos_x),
            y=list(pos_y),
            correlation_value=list(corr_values),
            correlation_mean=det_mean,
            correlation_variance=det_variance,
        )
