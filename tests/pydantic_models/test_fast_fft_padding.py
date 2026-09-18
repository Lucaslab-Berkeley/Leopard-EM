"""Tests for the automatic fast-FFT padding configuration."""

import pytest

from leopard_em.pydantic_models.config import FastFFTPaddingConfig
from leopard_em.utils.fft_padding import FFTPaddingWarning

TEMPLATE_SHAPE = (512, 512)


def test_defaults_are_enabled_with_two_three_factors():
    config = FastFFTPaddingConfig()
    assert config.enabled is True
    assert config.allowed_factors == [2, 3]
    assert config.target_shape is None
    assert config.noise_seed == 0


def test_make_plan_pads_to_next_even_fast_size():
    config = FastFFTPaddingConfig()
    plan = config.make_plan((4000, 4000), TEMPLATE_SHAPE, "streamed")
    assert plan.padded_shape == (4096, 4096)
    assert plan.original_shape == (4000, 4000)
    assert plan.is_padded


def test_make_plan_is_a_noop_when_disabled():
    plan = FastFFTPaddingConfig(enabled=False).make_plan(
        (4000, 4000), TEMPLATE_SHAPE, "streamed"
    )
    assert not plan.is_padded
    assert plan.padded_shape == (4000, 4000)


def test_make_plan_is_a_noop_for_already_fast_sizes():
    plan = FastFFTPaddingConfig().make_plan((4096, 4096), TEMPLATE_SHAPE, "streamed")
    assert not plan.is_padded


def test_make_plan_is_pure_and_repeatable():
    config = FastFFTPaddingConfig()
    first = config.make_plan((4096, 4096), TEMPLATE_SHAPE, "streamed")
    second = config.make_plan((4096, 4096), TEMPLATE_SHAPE, "streamed")
    assert first == second


def test_allowed_factors_must_contain_two():
    with pytest.raises(ValueError, match="must contain 2"):
        FastFFTPaddingConfig(allowed_factors=[3, 5])


def test_allowed_factors_are_respected():
    config = FastFFTPaddingConfig(allowed_factors=[2, 3, 5, 7])
    # 4000 = 2^5 * 5^3 is already 7-smooth, so nothing to do.
    assert not config.make_plan((4000, 4000), TEMPLATE_SHAPE, "streamed").is_padded


def test_target_shape_overrides_automatic_selection():
    config = FastFFTPaddingConfig(target_shape=[5000, 5000])
    with pytest.warns(FFTPaddingWarning):
        plan = config.make_plan((4000, 4000), TEMPLATE_SHAPE, "streamed")
    assert plan.padded_shape == (5000, 5000)


def test_target_shape_smaller_than_image_is_rejected():
    config = FastFFTPaddingConfig(target_shape=[100, 100])
    with pytest.raises(ValueError, match="smaller than the image"):
        config.make_plan((4000, 4000), TEMPLATE_SHAPE, "streamed")


def test_odd_target_shape_is_rejected():
    config = FastFFTPaddingConfig(target_shape=[5000, 5001])
    with pytest.raises(ValueError, match="must be even"):
        config.make_plan((4000, 4000), TEMPLATE_SHAPE, "streamed")


def test_target_shape_must_have_two_entries():
    with pytest.raises(ValueError):
        FastFFTPaddingConfig(target_shape=[4096])


def test_large_padding_emits_warning():
    """649 -> 768 grows the pixel count by ~40%, well past LARGE_AREA_GROWTH."""
    config = FastFFTPaddingConfig()
    with pytest.warns(FFTPaddingWarning, match="orientation_batch_size"):
        config.make_plan((649, 649), (64, 64), "streamed")


@pytest.mark.parametrize("image_shape", [(4092, 4092), (4000, 4000)])
def test_modest_padding_does_not_warn(recwarn, image_shape):
    """Common cases must stay quiet."""
    FastFFTPaddingConfig().make_plan(image_shape, TEMPLATE_SHAPE, "streamed")
    assert [w for w in recwarn if issubclass(w.category, FFTPaddingWarning)] == []


def test_zipfft_backend_falls_back_when_library_absent():
    """Without zipfft compiled configs, padding falls back to the next fast size."""
    from leopard_em.utils.zipfft_support import ZIPFFT_AVAILABLE

    if ZIPFFT_AVAILABLE:
        pytest.skip("zipfft is installed; fallback path not exercised")

    config = FastFFTPaddingConfig()
    with pytest.warns(FFTPaddingWarning):
        plan = config.make_plan((4000, 4000), TEMPLATE_SHAPE, "zipfft")
    assert plan.padded_shape == (4096, 4096)
    assert plan.effective_backend == "batched"


def test_yaml_round_trip(tmp_path):
    """Guards against list-vs-tuple serialization breaking from_yaml."""
    config = FastFFTPaddingConfig(allowed_factors=[2, 3, 5], noise_seed=7)
    path = tmp_path / "padding.yaml"
    config.to_yaml(path)
    assert FastFFTPaddingConfig.from_yaml(path) == config
