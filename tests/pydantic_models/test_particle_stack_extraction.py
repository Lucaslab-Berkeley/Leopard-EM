"""Tests for the grouped-extraction and movie helpers behind the particle stacks."""

import numpy as np
import pandas as pd
import pytest
import torch

from leopard_em.pydantic_models.data_structures.particle_stack import ParticleStack
from leopard_em.pydantic_models.data_structures.particle_stack import _movie as movie
from leopard_em.pydantic_models.data_structures.particle_stack._extraction import (
    LARGE_STD_PADDING,
    combined_filter_stack,
    crop_by_group,
    local_stat_padding,
    scatter_by_group,
)


def test_local_stat_padding():
    assert local_stat_padding("correlation_variance_path") == LARGE_STD_PADDING
    assert local_stat_padding("mip_path") == 0.0
    assert local_stat_padding("correlation_variance_path", padding_value=2.0) == 2.0


def test_crop_by_group_places_crops_at_group_rows():
    images = torch.stack([torch.arange(100.0).reshape(10, 10), -torch.ones(10, 10)])
    origin_y = np.array([0, 2, 5, 0])
    origin_x = np.array([1, 3, 6, 0])
    groups = [pd.Index([2, 0]), pd.Index([1])]  # row 3 belongs to no image

    crops = crop_by_group(images, groups, origin_y, origin_x, (2, 3))

    assert crops.shape == (4, 2, 3)
    torch.testing.assert_close(crops[0], images[0, 0:2, 1:4])
    torch.testing.assert_close(crops[2], images[0, 5:7, 6:9])
    torch.testing.assert_close(crops[1], -torch.ones(2, 3))
    assert torch.count_nonzero(crops[3]) == 0


def test_crop_by_group_pads_out_of_bounds():
    images = torch.ones(1, 4, 4)
    crops = crop_by_group(
        images, [pd.Index([0])], np.array([-1]), np.array([3]), (2, 2), padding_value=7
    )
    torch.testing.assert_close(crops[0], torch.tensor([[7.0, 7.0], [1.0, 7.0]]))


def test_crop_by_group_validates_groups():
    images = torch.zeros(1, 4, 4)
    origins = np.zeros(2, dtype=np.int64)
    with pytest.raises(ValueError, match="Number of images"):
        crop_by_group(images, [], origins, origins, (2, 2))
    with pytest.raises(IndexError, match="out of range"):
        crop_by_group(images, [pd.Index([2])], origins, origins, (2, 2))


def test_scatter_by_group():
    values = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    out = scatter_by_group(values, [pd.Index([0, 3]), pd.Index([1])], num_particles=4)
    assert out[:, 0].tolist() == [1.0, 2.0, 0.0, 1.0]
    with pytest.raises(TypeError, match="integer row positions"):
        scatter_by_group(values[:1], [pd.Index(["p_00000"])], num_particles=4)


def test_combined_filter_stack_uses_each_image_as_reference():
    class _MeanFilter:
        def get_combined_filter(self, ref_img_rfft, output_shape):
            return torch.full(output_shape, float(ref_img_rfft.real.mean()))

    images_dft = torch.stack([torch.full((4, 3), 1.0), torch.full((4, 3), 3.0)])
    filters = combined_filter_stack(_MeanFilter(), images_dft, (2, 2))
    assert filters.shape == (2, 2, 2)
    assert filters[:, 0, 0].tolist() == [1.0, 3.0]


def test_crop_particle_regions_leaves_image_stack_alone():
    stack = ParticleStack(
        df_path="",
        extracted_box_size=(4, 4),
        original_template_size=(2, 2),
        skip_df_load=True,
    )
    stack._df = pd.DataFrame({"pos_y": [3], "pos_x": [4]})
    images = torch.arange(100.0).reshape(1, 10, 10)

    crops = stack.crop_particle_regions(images, [pd.Index([0])], (4, 4))
    assert stack.image_stack is None
    torch.testing.assert_close(crops[0], images[0, 2:6, 3:7])

    constructed = stack.construct_image_stack(images, [pd.Index([0])], (4, 4))
    assert stack.image_stack is constructed
    torch.testing.assert_close(constructed, crops)


def test_check_motion_source():
    shifts = torch.zeros(1, 1, 2)
    with pytest.raises(ValueError, match="Only one of"):
        movie.check_motion_source(object(), shifts, require_motion_source=False)
    with pytest.raises(ValueError, match="must be provided"):
        movie.check_motion_source(None, None, require_motion_source=True)
    with pytest.warns(UserWarning, match="already aligned"):
        movie.check_motion_source(None, None, require_motion_source=False)
    movie.check_motion_source(None, shifts, require_motion_source=True)


def test_dose_weight_particle_movies_uses_stack_rows(monkeypatch):
    calls = []

    def _fake_dose_weight(
        movie_fft, pixel_size, pre_exposure, fluence_per_frame, voltage
    ):
        calls.append((pixel_size, voltage, pre_exposure, fluence_per_frame))
        return torch.full((4, 6), pixel_size)

    monkeypatch.setattr(movie, "dose_weight_movie_to_micrograph", _fake_dose_weight)
    rfft = torch.zeros(2, 3, 4, 4, dtype=torch.complex64)
    images = movie.dose_weight_particle_movies(
        rfft,
        extracted_box_size=(4, 6),
        rows=[2, 0],
        pixel_sizes=torch.tensor([1.0, 1.5, 2.0]),
        voltages=np.array([100.0, 200.0, 300.0]),
        pre_exposure=0.5,
        fluence_per_frame=1.0,
    )

    assert images.shape == (2, 4, 6)
    assert calls == [(2.0, 300.0, 0.5, 1.0), (1.0, 100.0, 0.5, 1.0)]
    assert images[:, 0, 0].tolist() == [2.0, 1.0]
