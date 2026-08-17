"""Unit tests for CorrelationTable and derive_orientation_grid_from_full_angles."""

import os
import tempfile

import pytest
import torch

from leopard_em.pydantic_models.results.correlation_table import (
    CorrelationTable,
    derive_orientation_grid_from_full_angles,
)

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
def minimal_table() -> CorrelationTable:
    """A small, hand-crafted CorrelationTable for roundtrip tests."""
    return CorrelationTable(
        correlation_threshold=5.5,
        num_observations=3,
        defocus_offsets=[-500.0, 0.0, 500.0],
        phi_theta_angles=[(0.0, 0.0), (45.0, 30.0)],
        psi_angles=[0.0, 90.0, 180.0],
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
        phi_theta_angles=[(0.0, 0.0)],
        psi_angles=[0.0, 90.0],
        search_index=[],
        x=[],
        y=[],
        correlation_value=[],
        correlation_mean=[],
        correlation_variance=[],
    )


# ---------------------------------------------------------------------------
# derive_orientation_grid_from_full_angles
# ---------------------------------------------------------------------------


class TestDeriveOrientationGrid:
    def test_basic_grid(self, grid_euler_angles):
        phi_theta, psi = derive_orientation_grid_from_full_angles(grid_euler_angles)
        assert phi_theta == [(0.0, 0.0), (45.0, 30.0)]
        assert psi == [0.0, 90.0, 180.0]

    def test_single_phi_theta(self):
        angles = torch.tensor([[10.0, 20.0, 0.0], [10.0, 20.0, 45.0]])
        phi_theta, psi = derive_orientation_grid_from_full_angles(angles)
        assert phi_theta == [(10.0, 20.0)]
        assert psi == [0.0, 45.0]

    def test_single_psi(self):
        angles = torch.tensor([[0.0, 0.0, 0.0], [45.0, 30.0, 0.0]])
        phi_theta, psi = derive_orientation_grid_from_full_angles(angles)
        assert phi_theta == [(0.0, 0.0), (45.0, 30.0)]
        assert psi == [0.0]

    def test_return_lengths_match_grid(self, grid_euler_angles):
        phi_theta, psi = derive_orientation_grid_from_full_angles(grid_euler_angles)
        assert len(phi_theta) * len(psi) == grid_euler_angles.shape[0]


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
        assert df.attrs["psi_angles"] == minimal_table.psi_angles

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
        assert recovered.phi_theta_angles == minimal_table.phi_theta_angles
        assert recovered.psi_angles == minimal_table.psi_angles

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
            assert recovered.phi_theta_angles == pytest.approx(
                minimal_table.phi_theta_angles, abs=1e-5
            )
            assert recovered.psi_angles == pytest.approx(
                minimal_table.psi_angles, abs=1e-5
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
    def test_search_space_derivation(self, factory_inputs):
        ct = CorrelationTable.from_match_template_results(**factory_inputs)
        assert ct.phi_theta_angles == [(0.0, 0.0), (45.0, 30.0)]
        assert ct.psi_angles == [0.0, 90.0, 180.0]
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
        expected_mean = [corr_avg[y, x].item() for x, y in zip(xs, ys)]
        expected_var = [corr_var[y, x].item() for x, y in zip(xs, ys)]

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
