"""Unit tests for CorrelationTable."""

import os
import tempfile

import h5py
import numpy as np
import pytest
import torch

from leopard_em.pydantic_models.results.correlation_table import CorrelationTable

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def grid_euler_angles() -> torch.Tensor:
    """2 (phi, theta) pairs x 3 psi values → 6 orientations."""
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 180.0],
            [45.0, 30.0, 0.0],
            [45.0, 30.0, 90.0],
            [45.0, 30.0, 180.0],
        ]
    )


@pytest.fixture()
def psi_outer_euler_angles() -> torch.Tensor:
    """The same 6 orientations in the order ``torch_so3`` emits: psi varies slowest."""
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [45.0, 30.0, 0.0],
            [0.0, 0.0, 90.0],
            [45.0, 30.0, 90.0],
            [0.0, 0.0, 180.0],
            [45.0, 30.0, 180.0],
        ]
    )


@pytest.fixture()
def minimal_table() -> CorrelationTable:
    """A small, hand-crafted CorrelationTable for roundtrip tests."""
    return CorrelationTable(
        correlation_threshold=5.5,
        num_observations=3,
        defocus_offsets=[-500.0, 0.0, 500.0],
        euler_angles=[(0.0, 0.0, 0.0), (0.0, 0.0, 90.0), (45.0, 30.0, 180.0)],
        search_index=[0, 5, 11],
        x=[10, 20, 30],
        y=[15, 25, 35],
        correlation_value=[6.1, 7.2, 5.8],
        correlation_mean=[0.1, 0.2, 0.3],
        correlation_variance=[0.5, 0.6, 0.7],
    )


@pytest.fixture()
def empty_table() -> CorrelationTable:
    """A CorrelationTable with no detections."""
    return CorrelationTable(
        correlation_threshold=5.5,
        num_observations=0,
        defocus_offsets=[-500.0, 0.0],
        euler_angles=[(0.0, 0.0, 0.0), (0.0, 0.0, 90.0)],
        search_index=[],
        x=[],
        y=[],
        correlation_value=[],
        correlation_mean=[],
        correlation_variance=[],
    )


# ---------------------------------------------------------------------------
# CorrelationTable construction
# ---------------------------------------------------------------------------


class TestCorrelationTableConstruction:
    def test_basic_construction(self, minimal_table):
        assert minimal_table.num_observations == 3
        assert minimal_table.correlation_threshold == 5.5
        assert len(minimal_table.search_index) == 3
        assert len(minimal_table.x) == 3

    def test_empty_construction(self, empty_table):
        assert empty_table.num_observations == 0
        assert empty_table.search_index == []
        assert empty_table.x == []


# ---------------------------------------------------------------------------
# DataFrame roundtrip
# ---------------------------------------------------------------------------


class TestDataFrameRoundtrip:
    def test_columns_present(self, minimal_table):
        df = minimal_table.to_dataframe()
        expected = {
            "search_index",
            "x",
            "y",
            "correlation_value",
            "correlation_mean",
            "correlation_variance",
        }
        assert expected == set(df.columns)

    def test_metadata_in_attrs(self, minimal_table):
        df = minimal_table.to_dataframe()
        assert df.attrs["correlation_threshold"] == minimal_table.correlation_threshold
        assert df.attrs["num_observations"] == minimal_table.num_observations
        assert df.attrs["defocus_offsets"] == minimal_table.defocus_offsets
        assert df.attrs["euler_angles"] == minimal_table.euler_angles

    def test_roundtrip_detection_data(self, minimal_table):
        recovered = CorrelationTable.from_dataframe(minimal_table.to_dataframe())
        assert recovered.search_index == minimal_table.search_index
        assert recovered.x == minimal_table.x
        assert recovered.y == minimal_table.y
        assert recovered.correlation_value == pytest.approx(
            minimal_table.correlation_value
        )
        assert recovered.correlation_mean == pytest.approx(
            minimal_table.correlation_mean
        )

    def test_roundtrip_search_space(self, minimal_table):
        recovered = CorrelationTable.from_dataframe(minimal_table.to_dataframe())
        assert recovered.defocus_offsets == minimal_table.defocus_offsets
        assert recovered.euler_angles == minimal_table.euler_angles

    def test_row_count(self, minimal_table):
        df = minimal_table.to_dataframe()
        assert len(df) == minimal_table.num_observations

    def test_empty_table_roundtrip(self, empty_table):
        recovered = CorrelationTable.from_dataframe(empty_table.to_dataframe())
        assert recovered.num_observations == 0
        assert recovered.x == []


# ---------------------------------------------------------------------------
# HDF5 roundtrip
# ---------------------------------------------------------------------------


class TestHDF5Roundtrip:
    def test_roundtrip_detection_data(self, minimal_table):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            minimal_table.to_hdf5(path)
            recovered = CorrelationTable.from_hdf5(path)
            assert recovered.search_index == minimal_table.search_index
            assert recovered.x == minimal_table.x
            assert recovered.y == minimal_table.y
            assert recovered.correlation_value == pytest.approx(
                minimal_table.correlation_value, abs=1e-5
            )
        finally:
            os.unlink(path)

    def test_roundtrip_search_space(self, minimal_table):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            minimal_table.to_hdf5(path)
            recovered = CorrelationTable.from_hdf5(path)
            assert recovered.defocus_offsets == pytest.approx(
                minimal_table.defocus_offsets, abs=1e-5
            )
            assert recovered.euler_angles == pytest.approx(
                minimal_table.euler_angles, abs=1e-5
            )
        finally:
            os.unlink(path)

    def test_roundtrip_metadata(self, minimal_table):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            minimal_table.to_hdf5(path)
            recovered = CorrelationTable.from_hdf5(path)
            assert (
                recovered.correlation_threshold == minimal_table.correlation_threshold
            )
            assert recovered.num_observations == minimal_table.num_observations
        finally:
            os.unlink(path)

    def test_empty_table_roundtrip(self, empty_table):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            empty_table.to_hdf5(path)
            recovered = CorrelationTable.from_hdf5(path)
            assert recovered.num_observations == 0
            assert recovered.search_index == []
        finally:
            os.unlink(path)

    def test_file_is_created(self, minimal_table):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        os.unlink(path)
        try:
            minimal_table.to_hdf5(path)
            assert os.path.isfile(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestHDF5Compression:
    """euler_angles is the only dataset large enough for compression to matter."""

    @pytest.fixture()
    def large_table(self) -> CorrelationTable:
        """A table with a big, repetitive (and thus compressible) euler_angles grid."""
        n_phi_theta = 2000
        psi_values = [0.0, 90.0, 180.0, 270.0]
        phi_theta = [(float(i % 360), float((i * 7) % 180)) for i in range(n_phi_theta)]
        euler_angles = [
            (phi, theta, psi) for psi in psi_values for phi, theta in phi_theta
        ]
        return CorrelationTable(
            correlation_threshold=5.5,
            num_observations=0,
            defocus_offsets=[0.0],
            euler_angles=euler_angles,
            search_index=[],
            x=[],
            y=[],
            correlation_value=[],
            correlation_mean=[],
            correlation_variance=[],
        )

    def test_compress_defaults_to_true(self, large_table):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            large_table.to_hdf5(path)
            with h5py.File(path, "r") as f:
                ds = f["search_space/euler_angles"]
                assert ds.compression == "gzip"
                assert ds.compression_opts == 4
                assert ds.shuffle is True
        finally:
            os.unlink(path)

    def test_compress_true_is_smaller_than_uncompressed(self, large_table):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path_compressed = f.name
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path_uncompressed = f.name
        try:
            large_table.to_hdf5(path_compressed, compress=True)
            large_table.to_hdf5(path_uncompressed, compress=False)
            assert os.path.getsize(path_compressed) < os.path.getsize(path_uncompressed)
        finally:
            os.unlink(path_compressed)
            os.unlink(path_uncompressed)

    def test_compress_false_disables_filters(self, large_table):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            large_table.to_hdf5(path, compress=False)
            with h5py.File(path, "r") as f:
                ds = f["search_space/euler_angles"]
                assert ds.compression is None
                assert ds.shuffle is False
        finally:
            os.unlink(path)

    def test_roundtrip_preserved_regardless_of_compression(self, large_table):
        for compress in (True, False):
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                path = f.name
            try:
                large_table.to_hdf5(path, compress=compress)
                recovered = CorrelationTable.from_hdf5(path)
                assert recovered.euler_angles == pytest.approx(
                    large_table.euler_angles, abs=1e-4
                )
            finally:
                os.unlink(path)


# ---------------------------------------------------------------------------
# from_match_template_results factory
# ---------------------------------------------------------------------------


@pytest.fixture()
def factory_inputs(grid_euler_angles):
    """Common inputs for from_match_template_results tests."""
    H, W = 64, 80
    defocus_values = torch.tensor([-500.0, 0.0, 500.0])
    corr_avg = torch.rand(H, W)
    corr_var = torch.rand(H, W)
    proc_table = {
        "threshold": 5.5,
        "global_idx": [0, 5, 11],
        "x": [10, 20, 30],
        "y": [15, 25, 35],
        "correlation": [6.1, 7.2, 5.8],
    }
    return {
        "processed_correlation_table": proc_table,
        "defocus_values": defocus_values,
        "euler_angles": grid_euler_angles,
        "correlation_average": corr_avg,
        "correlation_variance_map": corr_var,
    }


class TestFromMatchTemplateResults:
    def test_euler_angles_stored_verbatim(self, factory_inputs, grid_euler_angles):
        ct = CorrelationTable.from_match_template_results(**factory_inputs)
        assert ct.euler_angles == [tuple(row) for row in grid_euler_angles.tolist()]
        assert ct.defocus_offsets == pytest.approx([-500.0, 0.0, 500.0])

    def test_num_observations(self, factory_inputs):
        ct = CorrelationTable.from_match_template_results(**factory_inputs)
        assert ct.num_observations == 3

    def test_search_index_passthrough(self, factory_inputs):
        ct = CorrelationTable.from_match_template_results(**factory_inputs)
        assert ct.search_index == [0, 5, 11]

    def test_xy_positions(self, factory_inputs):
        ct = CorrelationTable.from_match_template_results(**factory_inputs)
        assert ct.x == [10, 20, 30]
        assert ct.y == [15, 25, 35]

    def test_mean_variance_looked_up_from_tensors(self, factory_inputs):
        corr_avg = factory_inputs["correlation_average"]
        corr_var = factory_inputs["correlation_variance_map"]
        ct = CorrelationTable.from_match_template_results(**factory_inputs)

        xs = factory_inputs["processed_correlation_table"]["x"]
        ys = factory_inputs["processed_correlation_table"]["y"]
        expected_mean = [corr_avg[y, x].item() for x, y in zip(xs, ys, strict=False)]
        expected_var = [corr_var[y, x].item() for x, y in zip(xs, ys, strict=False)]

        assert ct.correlation_mean == pytest.approx(expected_mean)
        assert ct.correlation_variance == pytest.approx(expected_var)

    def test_empty_detections(self, factory_inputs, grid_euler_angles):
        empty_proc = {
            "threshold": 5.5,
            "global_idx": [],
            "x": [],
            "y": [],
            "correlation": [],
        }
        ct = CorrelationTable.from_match_template_results(
            processed_correlation_table=empty_proc,
            defocus_values=factory_inputs["defocus_values"],
            euler_angles=grid_euler_angles,
            correlation_average=factory_inputs["correlation_average"],
            correlation_variance_map=factory_inputs["correlation_variance_map"],
        )
        assert ct.num_observations == 0
        assert ct.correlation_mean == []
        assert ct.correlation_variance == []


# ---------------------------------------------------------------------------
# The full angle list is the authoritative record of the search
# ---------------------------------------------------------------------------


class TestEulerAnglesAlwaysStored:
    """Whatever the layout, the orientations are recorded verbatim."""

    def _table(self, factory_inputs, euler_angles):
        return CorrelationTable.from_match_template_results(
            **{**factory_inputs, "euler_angles": euler_angles}
        )

    def test_stored_for_a_psi_outer_grid(self, factory_inputs, psi_outer_euler_angles):
        ct = self._table(factory_inputs, psi_outer_euler_angles)
        assert ct.euler_angles == [
            tuple(row) for row in psi_outer_euler_angles.tolist()
        ]

    def test_stored_for_a_non_factorable_search(self, factory_inputs):
        """A constrained search still stores its angles verbatim."""
        angles = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 90.0], [45.0, 30.0, 0.0]])
        ct = self._table(factory_inputs, angles)
        assert ct.euler_angles == [tuple(row) for row in angles.tolist()]

    def test_hdf5_roundtrip_preserves_order(
        self, factory_inputs, psi_outer_euler_angles
    ):
        ct = self._table(factory_inputs, psi_outer_euler_angles)
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            ct.to_hdf5(path)
            recovered = CorrelationTable.from_hdf5(path)
            assert recovered.euler_angles == pytest.approx(ct.euler_angles, abs=1e-5)
        finally:
            os.unlink(path)

    def test_dataframe_roundtrip_preserves_order(
        self, factory_inputs, psi_outer_euler_angles
    ):
        ct = self._table(factory_inputs, psi_outer_euler_angles)
        recovered = CorrelationTable.from_dataframe(ct.to_dataframe())
        assert recovered.euler_angles == ct.euler_angles


# ---------------------------------------------------------------------------
# Legacy (pre-euler_angles) v1.3 files: reconstruct euler_angles on load
# ---------------------------------------------------------------------------


def _write_legacy_v1_3_hdf5(
    path: str, phi_theta_angles: list, psi_angles: list
) -> None:
    """Write a v1.3-style file with phi_theta_angles/psi_angles but no euler_angles."""
    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["correlation_threshold"] = 5.5
        meta.attrs["num_observations"] = 1

        search_space = f.create_group("search_space")
        search_space.create_dataset(
            "defocus_offsets", data=np.array([0.0], dtype=np.float32)
        )
        search_space.create_dataset(
            "phi_theta_angles",
            data=np.array(phi_theta_angles, dtype=np.float32),
        )
        search_space.create_dataset(
            "psi_angles", data=np.array(psi_angles, dtype=np.float32)
        )

        detections = f.create_group("detections")
        detections.create_dataset("search_index", data=np.array([0], dtype=np.int32))
        detections.create_dataset("x", data=np.array([10], dtype=np.int32))
        detections.create_dataset("y", data=np.array([15], dtype=np.int32))
        detections.create_dataset(
            "correlation_value", data=np.array([6.1], dtype=np.float32)
        )
        detections.create_dataset(
            "correlation_mean", data=np.array([0.1], dtype=np.float32)
        )
        detections.create_dataset(
            "correlation_variance", data=np.array([0.5], dtype=np.float32)
        )


class TestLegacyFileReconstruction:
    """A pre-euler_angles v1.3 file has its full grid rebuilt from the two axes."""

    def test_reconstructs_psi_outer_cartesian_product(self):
        """torch_so3 always emits psi-outer order, so that's the order to rebuild."""
        phi_theta_angles = [(0.0, 0.0), (45.0, 30.0)]
        psi_angles = [0.0, 90.0, 180.0]
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            _write_legacy_v1_3_hdf5(path, phi_theta_angles, psi_angles)
            recovered = CorrelationTable.from_hdf5(path)
            assert recovered.euler_angles == pytest.approx(
                [
                    (0.0, 0.0, 0.0),
                    (45.0, 30.0, 0.0),
                    (0.0, 0.0, 90.0),
                    (45.0, 30.0, 90.0),
                    (0.0, 0.0, 180.0),
                    (45.0, 30.0, 180.0),
                ]
            )
        finally:
            os.unlink(path)

    def test_file_with_neither_axis_still_loads(self, minimal_table):
        """A file with no orientation datasets at all has no grid to reconstruct."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            table = minimal_table.model_copy(update={"euler_angles": None})
            table.to_hdf5(path)
            recovered = CorrelationTable.from_hdf5(path)
            assert recovered.euler_angles is None
        finally:
            os.unlink(path)
