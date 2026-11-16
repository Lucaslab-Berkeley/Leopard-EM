"""Tests for the particle backend setup helpers, incl. stored HDF5 image stacks."""

import shutil
from types import SimpleNamespace

import pytest
import torch

from leopard_em.pydantic_models.config import PreprocessingFilters
from leopard_em.pydantic_models.data_structures import (
    ParticleStackHDF5,
    export_particle_stack,
)
from leopard_em.pydantic_models.managers import (
    ConstrainedSearchManager,
    constrained_search_manager,
)
from leopard_em.utils.backend_setup import (
    _setup_correlation_stacks_from_micrographs,
    setup_particle_backend_kwargs,
)
from leopard_em.utils.data_io import load_mrc_volume

COMPARED_KEYS = (
    "particle_stack_dft",
    "projective_filters",
    "corr_mean",
    "corr_std",
    "defocus_u",
    "defocus_v",
    "defocus_angle",
    "euler_angles",
)


def _backend_kwargs(workspace, particle_stack, apply_global_filtering):
    template = load_mrc_volume(workspace.template_path)
    return setup_particle_backend_kwargs(
        particle_stack=particle_stack,
        template=template,
        preprocessing_filters=PreprocessingFilters(),
        euler_angles=particle_stack.get_euler_angles(prefer_refined_angles=False),
        euler_angle_offsets=torch.zeros(1, 3),
        defocus_offsets=torch.tensor([0.0]),
        pixel_size_offsets=torch.tensor([0.0]),
        apply_global_filtering=apply_global_filtering,
        device_list=[torch.device("cpu")],
    )


def _store_images(workspace, df):
    """Export ``df`` to an HDF5 stack storing its particle images and local stats."""
    df["refined_relative_defocus"] = 0.0
    stack = export_particle_stack(
        df,
        str(workspace.root / "stack.h5"),
        extracted_box_size=workspace.extracted_box_size,
        original_template_size=workspace.original_template_size,
    )
    images, indices = stack.load_images_grouped_by_column("micrograph_path")
    # Same padding as the micrograph extraction path in backend_setup
    stack.construct_image_stack(
        images, indices, stack.extracted_box_size, padding_mode="reflect"
    )
    stack.local_stats.update(
        stack.get_local_stat_maps(
            columns=["correlation_average_path", "correlation_variance_path"]
        )
    )
    stack.to_hdf5(include_image_stack=True, include_local_stats=True)
    return stack


@pytest.fixture
def self_contained_stack(workspace):
    """An HDF5 stack storing its particle images and local stats."""
    return _store_images(workspace, workspace.df)


def test_stored_image_stack_matches_micrograph_path(workspace, self_contained_stack):
    """Stored images + local stats give the same backend inputs as re-extraction."""
    path = self_contained_stack.hdf5_path
    from_micrographs = _backend_kwargs(
        workspace,
        ParticleStackHDF5(hdf5_path=path, use_stored_image_stack=False),
        apply_global_filtering=False,
    )

    # Remove every referenced file: the stack must be self-contained now
    for directory in {p.parent for p in workspace.micrograph_paths}:
        shutil.rmtree(directory)

    from_stored = _backend_kwargs(
        workspace,
        ParticleStackHDF5(hdf5_path=path),
        apply_global_filtering=False,
    )

    for key in COMPARED_KEYS:
        torch.testing.assert_close(from_stored[key], from_micrographs[key], msg=key)


def test_stored_image_stack_rejects_global_filtering(workspace, self_contained_stack):
    """Global filtering of stored particle images is an error, not a silent switch."""
    stack = ParticleStackHDF5(hdf5_path=self_contained_stack.hdf5_path)
    with pytest.raises(ValueError, match="use_stored_image_stack: false"):
        _backend_kwargs(workspace, stack, apply_global_filtering=True)


class _StopSetup(Exception):
    """Raised by a stub to stop constrained search setup once its inputs are known."""


@pytest.mark.parametrize("apply_global_filtering", [True, False])
def test_constrained_search_passes_global_filtering_through(
    workspace, monkeypatch, apply_global_filtering
):
    """Constrained search uses its own ``apply_global_filtering`` for stored stacks."""
    first = workspace.df["micrograph_path"].iloc[0]
    one_micrograph = workspace.df[workspace.df["micrograph_path"] == first]
    stack = _store_images(workspace, one_micrograph.reset_index(drop=True))

    captured = {}

    def _capture(*_args, **kwargs):
        captured.update(kwargs)
        raise _StopSetup

    monkeypatch.setattr(
        constrained_search_manager, "setup_images_filters_particle_stack", _capture
    )
    manager = ConstrainedSearchManager.model_construct(
        template_volume=load_mrc_volume(workspace.template_path),
        template_volume_path=str(workspace.template_path),
        particle_stack_reference=ParticleStackHDF5(hdf5_path=stack.hdf5_path),
        orientation_refinement_config=SimpleNamespace(
            euler_angles_offsets=(torch.zeros(1, 3), None)
        ),
        computational_config=SimpleNamespace(gpu_devices=[torch.device("cpu")]),
        preprocessing_filters=PreprocessingFilters(),
        apply_global_filtering=apply_global_filtering,
    )

    if apply_global_filtering:
        with pytest.raises(ValueError, match="use_stored_image_stack: false"):
            manager.make_backend_core_function_kwargs(prefer_refined_angles=False)
        return

    with pytest.raises(_StopSetup):
        manager.make_backend_core_function_kwargs(prefer_refined_angles=False)
    assert captured["apply_global_filtering"] is False
    assert captured["images_are_particles"] is True


def _explicit_particle_kwargs(workspace, stack, apply_global_filtering):
    """Pass particle images explicitly, as ``run_differentiable_refine`` does."""
    return setup_particle_backend_kwargs(
        particle_stack=stack,
        template=load_mrc_volume(workspace.template_path),
        preprocessing_filters=PreprocessingFilters(),
        euler_angles=stack.get_euler_angles(prefer_refined_angles=False),
        euler_angle_offsets=torch.zeros(1, 3),
        defocus_offsets=torch.tensor([0.0]),
        pixel_size_offsets=torch.tensor([0.0]),
        apply_global_filtering=apply_global_filtering,
        device_list=[torch.device("cpu")],
        image_stack=stack.image_stack,
        mean_stack=stack.local_stats["correlation_average_path"],
        std_stack=stack.local_stats["correlation_variance_path"],
        images_are_particles=True,
    )


def test_explicit_particle_images_are_filtered_per_particle(
    workspace, self_contained_stack
):
    """Particle images passed by the caller are whitened per particle.

    ``run_differentiable_refine(images_are_particles=True)`` passes particle images
    directly, bypassing ``resolve_particle_image_source``.
    """
    expected = _backend_kwargs(
        workspace,
        ParticleStackHDF5(
            hdf5_path=self_contained_stack.hdf5_path, use_stored_image_stack=False
        ),
        apply_global_filtering=False,
    )

    kwargs = _explicit_particle_kwargs(
        workspace, self_contained_stack, apply_global_filtering=False
    )

    assert not torch.all(kwargs["projective_filters"] == 1.0)
    for key in COMPARED_KEYS:
        torch.testing.assert_close(kwargs[key], expected[key], msg=key)


def test_explicit_particle_images_reject_global_filtering(
    workspace, self_contained_stack
):
    """Global filtering of caller-provided particle images is an error.

    They used to go through the global branch and come back unfiltered and
    unnormalized, with all-ones projective filters.
    """
    with pytest.raises(ValueError, match="images_are_particles=True"):
        _explicit_particle_kwargs(
            workspace, self_contained_stack, apply_global_filtering=True
        )


def test_caller_micrograph_stats_bypass_stored_image_stack(
    workspace, self_contained_stack
):
    """Full-micrograph mean/std from the caller mean the micrograph path is in use."""
    path = self_contained_stack.hdf5_path
    expected = _backend_kwargs(
        workspace,
        ParticleStackHDF5(hdf5_path=path, use_stored_image_stack=False),
        apply_global_filtering=True,
    )

    stack = ParticleStackHDF5(hdf5_path=path)
    assert stack.get_stored_image_stack() is not None
    mean_maps, indices = stack.load_images_grouped_by_column("correlation_average_path")
    std_maps, _ = stack.load_images_grouped_by_column("correlation_variance_path")
    kwargs = setup_particle_backend_kwargs(
        particle_stack=stack,
        template=load_mrc_volume(workspace.template_path),
        preprocessing_filters=PreprocessingFilters(),
        euler_angles=stack.get_euler_angles(prefer_refined_angles=False),
        euler_angle_offsets=torch.zeros(1, 3),
        defocus_offsets=torch.tensor([0.0]),
        pixel_size_offsets=torch.tensor([0.0]),
        apply_global_filtering=True,
        device_list=[torch.device("cpu")],
        mean_stack=mean_maps,
        std_stack=std_maps,
        particle_indices=indices,
    )
    for key in COMPARED_KEYS:
        torch.testing.assert_close(kwargs[key], expected[key], msg=key)


def test_opting_out_of_stored_image_stack_needs_micrographs(
    workspace, self_contained_stack
):
    for directory in {p.parent for p in workspace.micrograph_paths}:
        shutil.rmtree(directory)
    stack = ParticleStackHDF5(
        hdf5_path=self_contained_stack.hdf5_path, use_stored_image_stack=False
    )
    with pytest.raises(FileNotFoundError, match="micrograph_path"):
        _backend_kwargs(workspace, stack, apply_global_filtering=True)


def test_correlation_setup_does_not_touch_image_stack(workspace):
    """Correlation crops used to overwrite ``image_stack`` as a side effect."""
    stack = export_particle_stack(
        workspace.df,
        str(workspace.root / "stack.h5"),
        extracted_box_size=workspace.extracted_box_size,
        original_template_size=workspace.original_template_size,
    )
    sentinel = torch.zeros(len(workspace.df), 20, 20)
    stack.image_stack = sentinel

    corr_mean, corr_std = _setup_correlation_stacks_from_micrographs(
        particle_stack=stack,
        mean_stack=None,
        std_stack=None,
        particle_indices=None,
        extracted_box_size=(5, 5),
        device=torch.device("cpu"),
    )
    assert stack.image_stack is sentinel
    assert corr_mean.shape == corr_std.shape == (len(workspace.df), 5, 5)


@pytest.mark.parametrize("apply_global_filtering", [True, False])
def test_image_setup_does_not_store_crops_in_image_stack(
    workspace, apply_global_filtering
):
    """Globally filtered crops must not land in ``image_stack`` (or to_hdf5)."""
    workspace.df["refined_relative_defocus"] = 0.0
    stack = export_particle_stack(
        workspace.df,
        str(workspace.root / "stack.h5"),
        extracted_box_size=workspace.extracted_box_size,
        original_template_size=workspace.original_template_size,
    )
    assert stack.image_stack is None

    _backend_kwargs(workspace, stack, apply_global_filtering=apply_global_filtering)

    assert stack.image_stack is None
    with pytest.raises(ValueError, match="image_stack is None"):
        stack.to_hdf5(include_image_stack=True)


def test_zero_padded_std_maps_give_finite_zscores(workspace):
    """v1.3 stored variance crops were zero-padded; they must not yield inf z-scores."""
    stack = export_particle_stack(
        workspace.df,
        str(workspace.root / "stack.h5"),
        extracted_box_size=workspace.extracted_box_size,
        original_template_size=workspace.original_template_size,
    )
    num = len(workspace.df)
    stack.local_stats = {
        "correlation_average_path": torch.zeros(num, 5, 5),
        "correlation_variance_path": torch.zeros(num, 5, 5),
    }
    _mean, corr_std = _setup_correlation_stacks_from_micrographs(
        particle_stack=stack,
        mean_stack=None,
        std_stack=None,
        particle_indices=None,
        extracted_box_size=(5, 5),
        device=torch.device("cpu"),
    )
    assert torch.isfinite(1.0 / corr_std).all()
    assert (corr_std > 0).all()
