"""Unit tests for the per-frame peak inspection manager."""

import pandas as pd
import pytest
import torch

from leopard_em.pydantic_models.config import (
    ComputationalConfigRefine,
    DefocusSearchConfig,
    MovieConfig,
    PixelSizeSearchConfig,
    PreprocessingFilters,
    RefineOrientationConfig,
)
from leopard_em.pydantic_models.data_structures.particle_stack import ParticleStack
from leopard_em.pydantic_models.managers.frame_inspection_manager import (
    FrameInspectionManager,
)


def make_frame_inspection_manager() -> FrameInspectionManager:
    """Construct a minimal frame inspection manager for method-level tests."""
    particle_stack = ParticleStack(
        df_path="",
        extracted_box_size=(2, 2),
        original_template_size=(2, 2),
        skip_df_load=True,
    )
    return FrameInspectionManager.model_construct(
        template_volume_path="",
        particle_stack=particle_stack,
        defocus_refinement_config=DefocusSearchConfig(enabled=False),
        pixel_size_refinement_config=PixelSizeSearchConfig(
            enabled=False,
            pixel_size_step=1.0,
        ),
        orientation_refinement_config=RefineOrientationConfig(enabled=False),
        preprocessing_filters=PreprocessingFilters(),
        computational_config=ComputationalConfigRefine(gpu_ids="cpu"),
        movie_config=MovieConfig(enabled=False),
        apply_global_filtering=True,
        template_volume=None,
    )


# ---------------------------------------------------------------------------
# _stack_frame_results: pure static stacking of per-frame backend outputs.
# ---------------------------------------------------------------------------


def test_stack_frame_results_cross_correlation():
    """Cross-correlation results stack with frame as the leading axis, in order."""
    frame_results = [
        torch.full((2, 1, 1, 1, 2, 2), float(i), dtype=torch.float32) for i in range(3)
    ]
    stacked = FrameInspectionManager._stack_frame_results(
        frame_results, output_mode="cross_correlation"
    )
    assert isinstance(stacked, torch.Tensor)
    assert stacked.shape == (3, 2, 1, 1, 1, 2, 2)
    assert torch.all(stacked[0] == 0)
    assert torch.all(stacked[2] == 2)


def test_stack_frame_results_frc_returns_tensor_and_shared_bins():
    """FRC results stack the spectra and carry the frequency bins through once."""
    freq_bins = torch.linspace(0.0, 0.5, 4)
    frame_results = [
        (torch.full((2, 1, 1, 1, 4), float(i)), freq_bins) for i in range(3)
    ]
    stacked, bins = FrameInspectionManager._stack_frame_results(
        frame_results, output_mode="frc"
    )
    assert stacked.shape == (3, 2, 1, 1, 1, 4)
    assert torch.equal(bins, freq_bins)


def test_stack_frame_results_empty_raises():
    """An empty result list is an error in either output mode."""
    with pytest.raises(ValueError, match="No frame results"):
        FrameInspectionManager._stack_frame_results([], output_mode="cross_correlation")


# ---------------------------------------------------------------------------
# _frame_dose_template: per-frame template selection / dose weighting.
# ---------------------------------------------------------------------------


def test_frame_dose_template_disabled_returns_same_template():
    """With dose weighting off, the shared template is returned unchanged."""
    manager = make_frame_inspection_manager()
    template = torch.randn((4, 4, 4))
    result = manager._frame_dose_template(
        template, frame_idx=2, apply_template_dose_weighting=False
    )
    assert result is template


def test_frame_dose_template_applies_distinct_dose_per_frame():
    """Each frame's template is dose-weighted over its own exposure interval."""
    manager = make_frame_inspection_manager()
    manager.particle_stack._df = pd.DataFrame(
        {
            "particle_index": [0, 1],
            "pixel_size": [1.0, 1.0],
            "refined_pixel_size": [1.0, 1.0],
        }
    )
    manager.movie_config = MovieConfig(
        enabled=True,
        movie_path="movie.mrc",
        pre_exposure=0.0,
        fluence_per_frame=10.0,
    )

    torch.manual_seed(0)
    template = torch.randn((6, 6, 6))

    frame0 = manager._frame_dose_template(
        template, frame_idx=0, apply_template_dose_weighting=True
    )
    frame1 = manager._frame_dose_template(
        template, frame_idx=1, apply_template_dose_weighting=True
    )

    # Dose weighting actually attenuates the template ...
    assert not torch.allclose(frame0, template)
    # ... and later frames (higher cumulative exposure) differ from earlier ones.
    assert not torch.allclose(frame0, frame1)
    assert frame0.shape == template.shape


# ---------------------------------------------------------------------------
# run_peak_inspection_per_frame: real input-validation guard (no backend).
# ---------------------------------------------------------------------------


def test_run_peak_inspection_per_frame_requires_movie():
    """Per-frame inspection fails fast when the movie config is disabled."""
    manager = make_frame_inspection_manager()
    with pytest.raises(ValueError, match="requires movie_config.enabled"):
        manager.run_peak_inspection_per_frame(template_tensor=torch.zeros((2, 2, 2)))
