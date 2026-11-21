"""Tests that ``computational_config.template_pad_factor`` reaches template setup."""

import pytest
from pydantic import ValidationError

from leopard_em.pydantic_models.config import (
    ComputationalConfigMatch,
    ComputationalConfigRefine,
    DefocusSearchConfig,
    MovieConfig,
    OrientationSearchConfig,
    PixelSizeSearchConfig,
    PreprocessingFilters,
    RefineOrientationConfig,
)
from leopard_em.pydantic_models.data_structures import (
    OpticsGroup,
    export_particle_stack,
)
from leopard_em.pydantic_models.managers import (
    MatchTemplateManager,
    RefineTemplateManager,
)
from leopard_em.pydantic_models.results import MatchTemplateResult

TEMPLATE_SIZE = 16  # matches tests/conftest.py


@pytest.mark.parametrize(
    "config_cls", [ComputationalConfigMatch, ComputationalConfigRefine]
)
def test_template_pad_factor_default_and_validation(config_cls):
    assert config_cls().template_pad_factor == 2.0
    assert config_cls(template_pad_factor=1.0).template_pad_factor == 1.0
    with pytest.raises(ValidationError):
        config_cls(template_pad_factor=0.5)


def test_template_pad_factor_round_trips():
    config = ComputationalConfigRefine(gpu_ids=0, template_pad_factor=1.5)
    assert ComputationalConfigRefine.model_validate(config.model_dump()) == config
    # Configs written before the field existed still load, with the default
    assert (
        ComputationalConfigRefine.model_validate({"gpu_ids": 0}).template_pad_factor
        == 2.0
    )


@pytest.mark.parametrize("pad_factor", [1.0, 2.0])
def test_refine_manager_pads_template(workspace, pad_factor):
    workspace.df["refined_relative_defocus"] = 0.0
    stack = export_particle_stack(
        workspace.df,
        str(workspace.root / "stack.h5"),
        extracted_box_size=workspace.extracted_box_size,
        original_template_size=workspace.original_template_size,
    )
    manager = RefineTemplateManager(
        template_volume_path=str(workspace.template_path),
        particle_stack=stack,
        defocus_refinement_config=DefocusSearchConfig(enabled=False),
        pixel_size_refinement_config=PixelSizeSearchConfig(enabled=False),
        orientation_refinement_config=RefineOrientationConfig(enabled=False),
        preprocessing_filters=PreprocessingFilters(),
        computational_config=ComputationalConfigRefine(
            gpu_ids="cpu", template_pad_factor=pad_factor
        ),
        movie_config=MovieConfig(enabled=False),
    )
    kwargs = manager.make_backend_core_function_kwargs(prefer_refined_angles=False)

    padded = int(TEMPLATE_SIZE * pad_factor)
    assert tuple(kwargs["template_dft"].shape) == (padded, padded, padded // 2 + 1)
    # Filters (and so the projection size) stay at the unpadded template size
    assert tuple(kwargs["projective_filters"].shape[-2:]) == (
        TEMPLATE_SIZE,
        TEMPLATE_SIZE // 2 + 1,
    )


@pytest.mark.parametrize("pad_factor", [1.0, 2.0])
def test_match_template_manager_pads_template(workspace, pad_factor):
    result_dir = workspace.root / "mt_results"
    result_dir.mkdir()
    manager = MatchTemplateManager(
        micrograph_path=str(workspace.micrograph_paths[0]),
        template_volume_path=str(workspace.template_path),
        optics_group=OpticsGroup(
            label="test",
            pixel_size=1.0,
            voltage=300.0,
            defocus_u=12000.0,
            defocus_v=11800.0,
            astigmatism_angle=0.0,
        ),
        defocus_search_config=DefocusSearchConfig(enabled=False),
        orientation_search_config=OrientationSearchConfig(),
        preprocessing_filters=PreprocessingFilters(),
        match_template_result=MatchTemplateResult(
            allow_file_overwrite=True,
            **{
                f"{key}_path": str(result_dir / f"{key}.mrc")
                for key in (
                    "mip",
                    "scaled_mip",
                    "orientation_psi",
                    "orientation_theta",
                    "orientation_phi",
                    "relative_defocus",
                    "correlation_average",
                    "correlation_variance",
                )
            },
        ),
        computational_config=ComputationalConfigMatch(
            gpu_ids="cpu", template_pad_factor=pad_factor
        ),
    )
    kwargs = manager.make_backend_core_function_kwargs()

    padded = int(TEMPLATE_SIZE * pad_factor)
    assert tuple(kwargs["template_dft"].shape) == (padded, padded, padded // 2 + 1)
