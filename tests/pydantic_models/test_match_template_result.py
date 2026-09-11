"""Unit tests for MatchTemplateResultMRC and MatchTemplateResultHDF5."""

import importlib.metadata

import h5py
import pytest
import torch

from leopard_em.pydantic_models.results import (
    MatchTemplateResult,
    MatchTemplateResultHDF5,
    MatchTemplateResultMRC,
)
from leopard_em.pydantic_models.results.match_template_result import _TENSOR_NAMES

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

_TENSOR_SHAPE = (16, 16)


def _make_tensors(shape=_TENSOR_SHAPE) -> dict[str, torch.Tensor]:
    """Return a dict of distinct float32 tensors for all eight result fields."""
    return {
        name: torch.full(shape, float(i), dtype=torch.float32)
        for i, name in enumerate(_TENSOR_NAMES)
    }


def _mrc_paths(tmp_path) -> dict[str, str]:
    """Return a dict of non-existent MRC output paths under *tmp_path*."""
    names = [
        "mip_path",
        "scaled_mip_path",
        "correlation_average_path",
        "correlation_variance_path",
        "orientation_psi_path",
        "orientation_theta_path",
        "orientation_phi_path",
        "relative_defocus_path",
    ]
    return {name: str(tmp_path / f"{name}.mrc") for name in names}


@pytest.fixture
def tensors():
    return _make_tensors()


@pytest.fixture
def mrc_result(tmp_path, tensors):
    """Construct a fully-populated MatchTemplateResultMRC."""
    return MatchTemplateResultMRC(
        **_mrc_paths(tmp_path),
        total_projections=100,
        total_orientations=50,
        total_defocus=2,
        **tensors,
    )


@pytest.fixture
def hdf5_result(tmp_path, tensors):
    """Construct a fully-populated MatchTemplateResultHDF5."""
    return MatchTemplateResultHDF5(
        hdf5_path=str(tmp_path / "result.h5"),
        allow_file_overwrite=True,
        total_projections=100,
        total_orientations=50,
        total_defocus=2,
        **tensors,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def test_public_imports():
    """All expected public names are importable from the results package."""
    from leopard_em.pydantic_models.results import (  # noqa: F401
        MatchTemplateResult,
        MatchTemplateResultHDF5,
        MatchTemplateResultMRC,
    )


def test_base_class_not_public():
    """The private base class is not exported from the public API."""
    with pytest.raises(ImportError):
        from leopard_em.pydantic_models.results import (
            _MatchTemplateResultBase,  # noqa: F401
        )


def test_backward_compat_alias():
    """MatchTemplateResult is an alias for MatchTemplateResultMRC."""
    assert MatchTemplateResult is MatchTemplateResultMRC


# ---------------------------------------------------------------------------
# MatchTemplateResultMRC — construction and validation
# ---------------------------------------------------------------------------


def test_mrc_construction(tmp_path):
    """MatchTemplateResultMRC is instantiated with the required MRC paths."""
    result = MatchTemplateResultMRC(**_mrc_paths(tmp_path))
    assert result.allow_file_overwrite is False
    assert result.total_projections == 0
    assert result.total_orientations == 0
    assert result.total_defocus == 0


def test_mrc_rejects_existing_file(tmp_path):
    """validate_paths raises when a path already exists and overwrite is off."""
    paths = _mrc_paths(tmp_path)
    # Pre-create one of the output files
    existing = paths["mip_path"]
    open(existing, "w").close()

    with pytest.raises(ValueError, match="already exists"):
        MatchTemplateResultMRC(**paths, allow_file_overwrite=False)


def test_mrc_allows_existing_file_with_overwrite_flag(tmp_path):
    """validate_paths passes when allow_file_overwrite=True, even if file exists."""
    paths = _mrc_paths(tmp_path)
    open(paths["mip_path"], "w").close()
    # Should not raise
    MatchTemplateResultMRC(**paths, allow_file_overwrite=True)


def test_mrc_version_auto_populated(tmp_path):
    """leopard_em_version is set to the installed package version on construction."""
    result = MatchTemplateResultMRC(**_mrc_paths(tmp_path))
    expected = importlib.metadata.version("leopard_em")
    assert result.leopard_em_version == expected


# ---------------------------------------------------------------------------
# MatchTemplateResultHDF5 — construction and validation
# ---------------------------------------------------------------------------


def test_hdf5_construction(tmp_path):
    """MatchTemplateResultHDF5 is instantiated with only hdf5_path."""
    result = MatchTemplateResultHDF5(hdf5_path=str(tmp_path / "out.h5"))
    assert result.compress is True
    assert result.allow_file_overwrite is False
    assert result.total_projections == 0


def test_hdf5_rejects_existing_file(tmp_path):
    """validate_hdf5_path raises when the file exists and overwrite is off."""
    path = tmp_path / "out.h5"
    path.touch()
    with pytest.raises(ValueError, match="already exists"):
        MatchTemplateResultHDF5(hdf5_path=str(path), allow_file_overwrite=False)


def test_hdf5_allows_existing_file_with_overwrite_flag(tmp_path):
    """validate_hdf5_path passes when allow_file_overwrite=True."""
    path = tmp_path / "out.h5"
    path.touch()
    MatchTemplateResultHDF5(hdf5_path=str(path), allow_file_overwrite=True)


def test_hdf5_version_auto_populated(tmp_path):
    """leopard_em_version is set to the installed package version on construction."""
    result = MatchTemplateResultHDF5(hdf5_path=str(tmp_path / "out.h5"))
    expected = importlib.metadata.version("leopard_em")
    assert result.leopard_em_version == expected


# ---------------------------------------------------------------------------
# MatchTemplateResultHDF5 — to_hdf5
# ---------------------------------------------------------------------------


def test_to_hdf5_creates_file(hdf5_result):
    """to_hdf5 creates the file at hdf5_path."""
    hdf5_result.to_hdf5()
    import os

    assert os.path.exists(hdf5_result.hdf5_path)


def test_to_hdf5_root_attributes(hdf5_result):
    """to_hdf5 writes scalar metadata as HDF5 root attributes."""
    hdf5_result.to_hdf5()
    with h5py.File(hdf5_result.hdf5_path, "r") as f:
        assert f.attrs["total_projections"] == 100
        assert f.attrs["total_orientations"] == 50
        assert f.attrs["total_defocus"] == 2
        assert f.attrs["leopard_em_version"] == hdf5_result.leopard_em_version


def test_to_hdf5_tensor_group_exists(hdf5_result):
    """to_hdf5 creates a 'tensors' group containing all eight datasets."""
    hdf5_result.to_hdf5()
    with h5py.File(hdf5_result.hdf5_path, "r") as f:
        assert "tensors" in f
        for name in _TENSOR_NAMES:
            assert name in f["tensors"], f"missing dataset: {name}"


def test_to_hdf5_tensor_shape_and_dtype(hdf5_result):
    """Each tensor dataset has the expected shape and float32 dtype."""
    hdf5_result.to_hdf5()
    with h5py.File(hdf5_result.hdf5_path, "r") as f:
        for name in _TENSOR_NAMES:
            ds = f["tensors"][name]
            assert ds.shape == _TENSOR_SHAPE, f"{name}: shape mismatch"
            assert ds.dtype == "float32", f"{name}: dtype mismatch"


def test_to_hdf5_tensor_values(hdf5_result):
    """Tensor values written to HDF5 match those held in the object."""
    hdf5_result.to_hdf5()
    with h5py.File(hdf5_result.hdf5_path, "r") as f:
        for name in _TENSOR_NAMES:
            stored = torch.from_numpy(f["tensors"][name][:])
            original = getattr(hdf5_result, name).cpu().to(torch.float32)
            assert torch.allclose(stored, original), f"{name}: value mismatch"


def test_to_hdf5_compression_enabled(hdf5_result):
    """When compress=True (default), datasets are gzip-compressed."""
    hdf5_result.to_hdf5()
    with h5py.File(hdf5_result.hdf5_path, "r") as f:
        for name in _TENSOR_NAMES:
            assert f["tensors"][name].compression == "gzip", f"{name}: expected gzip"


def test_to_hdf5_compression_disabled(tmp_path, tensors):
    """When compress=False, datasets are written without compression."""
    result = MatchTemplateResultHDF5(
        hdf5_path=str(tmp_path / "result_uncompressed.h5"),
        allow_file_overwrite=True,
        compress=False,
        **tensors,
    )
    result.to_hdf5()
    with h5py.File(result.hdf5_path, "r") as f:
        for name in _TENSOR_NAMES:
            assert (
                f["tensors"][name].compression is None
            ), f"{name}: expected no compression"


# ---------------------------------------------------------------------------
# MatchTemplateResultHDF5 — from_hdf5 round-trip
# ---------------------------------------------------------------------------


def test_from_hdf5_roundtrip_metadata(hdf5_result):
    """Scalar metadata survives a to_hdf5 / from_hdf5 round-trip."""
    hdf5_result.to_hdf5()
    loaded = MatchTemplateResultHDF5.from_hdf5(hdf5_result.hdf5_path)

    assert loaded.total_projections == hdf5_result.total_projections
    assert loaded.total_orientations == hdf5_result.total_orientations
    assert loaded.total_defocus == hdf5_result.total_defocus
    assert loaded.leopard_em_version == hdf5_result.leopard_em_version


def test_from_hdf5_roundtrip_tensors(hdf5_result):
    """All eight tensors survive a to_hdf5 / from_hdf5 round-trip."""
    hdf5_result.to_hdf5()
    loaded = MatchTemplateResultHDF5.from_hdf5(hdf5_result.hdf5_path)

    for name in _TENSOR_NAMES:
        original = getattr(hdf5_result, name).cpu().to(torch.float32)
        restored = getattr(loaded, name)
        assert restored is not None, f"{name} is None after loading"
        assert torch.allclose(
            original, restored
        ), f"{name}: value mismatch after round-trip"


def test_from_hdf5_preserves_hdf5_path(hdf5_result):
    """The loaded instance's hdf5_path matches the path it was loaded from."""
    hdf5_result.to_hdf5()
    loaded = MatchTemplateResultHDF5.from_hdf5(hdf5_result.hdf5_path)
    assert loaded.hdf5_path == hdf5_result.hdf5_path


def test_from_hdf5_missing_version_attribute(tmp_path):
    """Files without leopard_em_version degrade gracefully to 'unknown'."""
    path = tmp_path / "old_result.h5"
    with h5py.File(str(path), "w") as f:
        f.attrs["total_projections"] = 0
        f.attrs["total_orientations"] = 0
        f.attrs["total_defocus"] = 0

    loaded = MatchTemplateResultHDF5.from_hdf5(str(path))
    assert loaded.leopard_em_version == "unknown"


def test_from_hdf5_partial_tensors(tmp_path):
    """from_hdf5 succeeds when only a subset of tensors are stored."""
    path = tmp_path / "partial.h5"
    with h5py.File(str(path), "w") as f:
        f.attrs["leopard_em_version"] = "test"
        f.attrs["total_projections"] = 0
        f.attrs["total_orientations"] = 0
        f.attrs["total_defocus"] = 0
        grp = f.create_group("tensors")
        grp.create_dataset("mip", data=torch.zeros(_TENSOR_SHAPE).numpy())

    loaded = MatchTemplateResultHDF5.from_hdf5(str(path))
    assert loaded.mip is not None
    assert loaded.scaled_mip is None


# ---------------------------------------------------------------------------
# export_results alias
# ---------------------------------------------------------------------------


def test_export_results_alias(hdf5_result):
    """export_results() on MatchTemplateResultHDF5 is equivalent to to_hdf5()."""
    import os

    hdf5_result.export_results()
    assert os.path.exists(hdf5_result.hdf5_path)
    with h5py.File(hdf5_result.hdf5_path, "r") as f:
        assert "tensors" in f


# ---------------------------------------------------------------------------
# apply_valid_cropping (inherited from base)
# ---------------------------------------------------------------------------


def test_apply_valid_cropping_mrc(mrc_result):
    """apply_valid_cropping reduces tensor dimensions correctly for MRC result."""
    template_shape = (4, 4)
    expected_h = _TENSOR_SHAPE[0] - template_shape[0] + 1  # 13
    expected_w = _TENSOR_SHAPE[1] - template_shape[1] + 1  # 13

    mrc_result.apply_valid_cropping(template_shape)

    for name in _TENSOR_NAMES:
        t = getattr(mrc_result, name)
        assert t.shape == (
            expected_h,
            expected_w,
        ), f"{name}: wrong shape after cropping"


def test_apply_valid_cropping_hdf5(hdf5_result):
    """apply_valid_cropping reduces tensor dimensions correctly for HDF5 result."""
    template_shape = (4, 4)
    expected_h = _TENSOR_SHAPE[0] - template_shape[0] + 1
    expected_w = _TENSOR_SHAPE[1] - template_shape[1] + 1

    hdf5_result.apply_valid_cropping(template_shape)

    for name in _TENSOR_NAMES:
        t = getattr(hdf5_result, name)
        assert t.shape == (
            expected_h,
            expected_w,
        ), f"{name}: wrong shape after cropping"
