"""Tests that MatchTemplateManager wires fast-FFT padding into both run paths."""

import pytest
import torch

from leopard_em.pydantic_models.config import (
    ComputationalConfigMatch,
    DefocusSearchConfig,
    FastFFTPaddingConfig,
    OrientationSearchConfig,
    PreprocessingFilters,
)
from leopard_em.pydantic_models.managers import match_template_manager as mtm_module
from leopard_em.pydantic_models.managers.match_template_manager import (
    MatchTemplateManager,
)
from leopard_em.utils.fft_padding import FFTPaddingPlan

IMAGE_SHAPE = (118, 122)
PADDED_SHAPE = (128, 128)
TEMPLATE_SHAPE = (16, 16)
VALID_SHAPE = (
    IMAGE_SHAPE[0] - TEMPLATE_SHAPE[0] + 1,
    IMAGE_SHAPE[1] - TEMPLATE_SHAPE[1] + 1,
)
PADDED_VALID_SHAPE = (
    PADDED_SHAPE[0] - TEMPLATE_SHAPE[0] + 1,
    PADDED_SHAPE[1] - TEMPLATE_SHAPE[1] + 1,
)

MAP_KEYS = (
    "mip",
    "scaled_mip",
    "best_phi",
    "best_theta",
    "best_psi",
    "best_defocus",
    "correlation_mean",
    "correlation_variance",
)


def _ramp(shape):
    """Map whose value encodes its own position, so an off-by-one is visible."""
    rows = torch.arange(shape[0], dtype=torch.float32)[:, None] * 1000.0
    cols = torch.arange(shape[1], dtype=torch.float32)[None, :]
    return rows + cols


def _fake_results(shape):
    results = {key: _ramp(shape) for key in MAP_KEYS}
    results["correlation_table"] = {
        "threshold": 5.5,
        "global_idx": [0, 1, 2],
        "pixel_size": [0.0, 0.0, 0.0],
        "defocus": [0.0, 0.0, 0.0],
        "phi": [0.0, 0.0, 0.0],
        "theta": [0.0, 0.0, 0.0],
        "psi": [0.0, 0.0, 0.0],
        # One row inside the unpadded valid region, two inside the padded region only.
        "x": [3, VALID_SHAPE[1] + 2, 5],
        "y": [4, 6, VALID_SHAPE[0] + 1],
        "correlation": [9.0, 9.1, 9.2],
    }
    results["total_projections"] = 1
    results["total_orientations"] = 1
    results["total_defocus"] = 1
    return results


def _manager(enabled=True):
    manager = MatchTemplateManager.model_construct(
        preprocessing_filters=PreprocessingFilters(),
        computational_config=ComputationalConfigMatch(),
        defocus_search_config=DefocusSearchConfig(enabled=False),
        orientation_search_config=OrientationSearchConfig(),
    )
    manager.fast_fft_padding = FastFFTPaddingConfig(enabled=enabled)
    manager._fft_padding_plan = FFTPaddingPlan(
        original_shape=IMAGE_SHAPE,
        padded_shape=PADDED_SHAPE if enabled else IMAGE_SHAPE,
        template_shape=TEMPLATE_SHAPE,
    )
    return manager


# --- the shape requested from the backend -------------------------------------------


def test_unpadded_valid_shape_is_requested_when_padded():
    assert _manager(enabled=True)._unpadded_valid_shape() == VALID_SHAPE


def test_no_shape_requested_when_not_padded():
    assert _manager(enabled=False)._unpadded_valid_shape() is None


def test_no_shape_requested_before_kwargs_are_built():
    manager = _manager(enabled=True)
    manager._fft_padding_plan = None
    assert manager._unpadded_valid_shape() is None


# --- the defensive un-pad path ------------------------------------------------------


def test_defensive_unpad_is_a_noop_when_backend_honoured_the_shape():
    manager = _manager(enabled=True)
    results = _fake_results(VALID_SHAPE)
    assert manager._unpad_backend_results(results) is results


def test_defensive_unpad_crops_maps_to_the_unpadded_region():
    manager = _manager(enabled=True)
    unpadded = manager._unpad_backend_results(_fake_results(PADDED_VALID_SHAPE))
    for key in MAP_KEYS:
        assert unpadded[key].shape == VALID_SHAPE, key
        # The ramp makes an off-by-one or a wrong-corner crop immediately visible.
        assert torch.equal(
            unpadded[key], _ramp(PADDED_VALID_SHAPE)[: VALID_SHAPE[0], : VALID_SHAPE[1]]
        )


def test_defensive_unpad_drops_correlation_rows_outside_the_valid_region():
    manager = _manager(enabled=True)
    unpadded = manager._unpad_backend_results(_fake_results(PADDED_VALID_SHAPE))
    table = unpadded["correlation_table"]
    assert table["x"] == [3]
    assert table["y"] == [4]
    assert table["correlation"] == [9.0]
    assert table["threshold"] == 5.5
    # Every remaining row must index safely into the cropped statistics maps.
    for x, y in zip(table["x"], table["y"], strict=True):
        assert unpadded["correlation_mean"][y, x] is not None


# --- both run paths must request the shape ------------------------------------------


def _patch_backend(monkeypatch, name):
    captured = {}

    def fake_backend(*args, **kwargs):
        captured.update(kwargs)
        return _fake_results(VALID_SHAPE)

    monkeypatch.setattr(mtm_module, name, fake_backend)
    monkeypatch.setattr(
        MatchTemplateManager, "_populate_match_template_result", lambda *a, **k: None
    )
    monkeypatch.setattr(
        MatchTemplateManager,
        "make_backend_core_function_kwargs",
        lambda self: {
            "defocus_values": torch.tensor([0.0]),
            "euler_angles": torch.zeros(1, 3),
        },
    )
    return captured


def test_run_match_template_requests_the_unpadded_valid_shape(monkeypatch):
    captured = _patch_backend(monkeypatch, "core_match_template")
    manager = _manager(enabled=True)
    manager.run_match_template(do_result_export=False)
    assert captured["unpadded_valid_shape"] == VALID_SHAPE


def test_run_match_template_distributed_requests_the_unpadded_valid_shape(monkeypatch):
    captured = _patch_backend(monkeypatch, "core_match_template_distributed")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    manager = _manager(enabled=True)
    manager.run_match_template_distributed(
        world_size=1, rank=0, local_rank=0, do_result_export=False
    )
    assert captured["unpadded_valid_shape"] == VALID_SHAPE


@pytest.mark.parametrize(
    "backend_name, runner",
    [
        ("core_match_template", "single"),
        ("core_match_template_distributed", "distributed"),
    ],
)
def test_no_shape_requested_when_padding_disabled(monkeypatch, backend_name, runner):
    captured = _patch_backend(monkeypatch, backend_name)
    manager = _manager(enabled=False)
    if runner == "single":
        manager.run_match_template(do_result_export=False)
    else:
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        manager.run_match_template_distributed(
            world_size=1, rank=0, local_rank=0, do_result_export=False
        )
    assert captured["unpadded_valid_shape"] is None


# --- backward compatibility of existing configuration files --------------------------


def _write_manager_config(tmp_path, include_padding_block):
    import yaml

    from leopard_em.utils.data_io import write_mrc_from_tensor

    write_mrc_from_tensor(torch.randn(64, 64), tmp_path / "mic.mrc", overwrite=True)
    write_mrc_from_tensor(torch.randn(16, 16, 16), tmp_path / "vol.mrc", overwrite=True)

    result_keys = [
        "mip",
        "scaled_mip",
        "orientation_psi",
        "orientation_theta",
        "orientation_phi",
        "relative_defocus",
        "correlation_average",
        "correlation_variance",
    ]
    config = {
        "micrograph_path": str(tmp_path / "mic.mrc"),
        "template_volume_path": str(tmp_path / "vol.mrc"),
        "optics_group": {
            "label": "test",
            "voltage": 300.0,
            "pixel_size": 1.06,
            "defocus_u": 5000.0,
            "defocus_v": 5000.0,
            "astigmatism_angle": 0.0,
            "spherical_aberration": 2.7,
            "amplitude_contrast_ratio": 0.07,
            "phase_shift": 0.0,
            "ctf_B_factor": 60.0,
        },
        "defocus_search_config": {"enabled": False},
        "orientation_search_config": {"psi_step": 5.0, "theta_step": 5.0},
        "preprocessing_filters": {"whitening_filter": {"enabled": True}},
        "match_template_result": {
            "allow_file_overwrite": True,
            **{f"{key}_path": str(tmp_path / f"{key}.mrc") for key in result_keys},
        },
        "computational_config": {"gpu_ids": [0], "num_cpus": 1},
    }
    if include_padding_block:
        config["fast_fft_padding"] = {
            "enabled": False,
            "allowed_factors": [2],
            "target_shape": None,
            "noise_seed": 3,
        }

    path = tmp_path / "config.yaml"
    with open(path, "w", encoding="utf-8") as handle:
        yaml.dump(config, handle)
    return path


def test_config_without_padding_block_still_loads(tmp_path):
    """Configs written before this feature existed must keep working.

    BaseModel2DTM forbids extra keys, so the new field has to carry a default
    instance rather than being required.
    """
    path = _write_manager_config(tmp_path, include_padding_block=False)
    manager = MatchTemplateManager.from_yaml(path)
    assert manager.fast_fft_padding == FastFFTPaddingConfig()


def test_config_with_padding_block_loads(tmp_path):
    path = _write_manager_config(tmp_path, include_padding_block=True)
    manager = MatchTemplateManager.from_yaml(path)
    assert manager.fast_fft_padding.enabled is False
    assert manager.fast_fft_padding.noise_seed == 3


def test_manager_config_round_trips_through_yaml(tmp_path):
    path = _write_manager_config(tmp_path, include_padding_block=True)
    manager = MatchTemplateManager.from_yaml(path)
    round_trip_path = tmp_path / "round_trip.yaml"
    manager.to_yaml(round_trip_path)
    assert (
        MatchTemplateManager.from_yaml(round_trip_path).fast_fft_padding
        == manager.fast_fft_padding
    )
