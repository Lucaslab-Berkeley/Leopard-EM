"""Tests for the particle backend setup helpers, incl. stored HDF5 image stacks."""

import shutil

import pytest
import torch

from leopard_em.pydantic_models.config import PreprocessingFilters
from leopard_em.pydantic_models.data_structures import (
    ParticleStackHDF5,
    export_particle_stack,
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


@pytest.fixture
def self_contained_stack(workspace):
    """An HDF5 stack storing its particle images and local stats."""
    workspace.df["refined_relative_defocus"] = 0.0
    stack = export_particle_stack(
        workspace.df,
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

    with pytest.warns(UserWarning, match="per particle"):
        from_stored = _backend_kwargs(
            workspace,
            ParticleStackHDF5(hdf5_path=path),
            apply_global_filtering=True,
        )

    for key in COMPARED_KEYS:
        torch.testing.assert_close(from_stored[key], from_micrographs[key], msg=key)


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
