"""Test the backed utility functions for box extraction from images."""

import numpy as np
import pytest
import torch

from leopard_em.pydantic_models.data_structures.particle_stack import (
    _get_cropped_image_regions_numpy,
    _get_cropped_image_regions_torch,
    get_cropped_image_regions,
)


def get_test_patch(box_size: tuple[int, int]) -> np.ndarray:
    """Create example data of ones with a two in the top-left corner of box size.

    Parameters
    ----------
    box_size : tuple[int, int]
        Size of the box to create, e.g., (32, 32).

    Returns
    -------
    np.ndarray
        A 2D array of ones with a two in the top-left corner.
    """
    patch = np.ones(box_size, dtype=np.float32)
    patch[0, 0] = 2.0
    return patch


def test_get_cropped_image_regions_numpy_fixed_positions():
    """Fixed positions test for _get_cropped_image_regions_numpy function."""
    box_size = (5, 5)
    pos_x = np.array([0, 64, 128])
    pos_y = np.array([0, 100, 15])

    # Create example image data for testing purposes
    test_patch = get_test_patch(box_size)
    image = np.zeros((256, 256), dtype=np.float32)
    for x, y in zip(pos_x, pos_y):
        image[y : y + box_size[0], x : x + box_size[1]] = test_patch

    # Get cropped regions using the numpy implementation
    cropped_regions = _get_cropped_image_regions_numpy(
        image=image,
        pos_y=pos_y,
        pos_x=pos_x,
        box_size=box_size,
        handle_bounds="error",
        padding_mode="constant",
        padding_value=0.0,
    )

    ground_truth = np.stack([test_patch] * 3, axis=0)

    assert cropped_regions.shape == (3, box_size[0], box_size[1])
    assert np.allclose(cropped_regions, ground_truth)


def test_get_cropped_image_regions_numpy_random_nonoverlapping():
    """Random non-overlapping pos for _get_cropped_image_regions_numpy function."""
    box_size = (5, 5)
    num_patches = 32
    image_size = (256, 256)
    test_patch = get_test_patch(box_size)
    image = np.zeros(image_size, dtype=np.float32)

    # Generate non-overlapping positions
    positions = []
    for _ in range(num_patches):
        total_failures = 0
        while total_failures < 100:
            y = np.random.randint(0, image_size[0] - box_size[0] + 1)
            x = np.random.randint(0, image_size[1] - box_size[1] + 1)
            # Check if new position overlaps with any existing position
            overlap = any(
                abs(y - py) < box_size[0] and abs(x - px) < box_size[1]
                for py, px in positions
            )

            if not overlap:
                positions.append((y, x))
                break
            total_failures += 1

    pos_y, pos_x = zip(*positions)

    for x, y in zip(pos_x, pos_y):
        image[y : y + box_size[0], x : x + box_size[1]] = test_patch

    cropped_regions = _get_cropped_image_regions_numpy(
        image=image,
        pos_y=pos_y,
        pos_x=pos_x,
        box_size=box_size,
        handle_bounds="error",
        padding_mode="constant",
        padding_value=0.0,
    )

    ground_truth = np.stack([test_patch] * num_patches, axis=0)

    assert cropped_regions.shape == (num_patches, box_size[0], box_size[1])
    assert np.allclose(cropped_regions, ground_truth)


def test_get_cropped_image_regions_torch_fixed_positions():
    """Fixed positions test for _get_cropped_image_regions_torch function."""
    box_size = (5, 5)
    pos_x = np.array([0, 64, 128])
    pos_y = np.array([0, 100, 15])

    test_patch = get_test_patch(box_size)
    image = np.zeros((256, 256), dtype=np.float32)
    for x, y in zip(pos_x, pos_y):
        image[y : y + box_size[0], x : x + box_size[1]] = test_patch

    image_tensor = torch.from_numpy(image)
    pos_y_tensor = torch.from_numpy(pos_y)
    pos_x_tensor = torch.from_numpy(pos_x)

    cropped_regions = _get_cropped_image_regions_torch(
        image=image_tensor,
        pos_y=pos_y_tensor,
        pos_x=pos_x_tensor,
        box_size=box_size,
        handle_bounds="error",
        padding_mode="constant",
        padding_value=0.0,
    )

    ground_truth = np.stack([test_patch] * 3, axis=0)
    assert cropped_regions.shape == (3, box_size[0], box_size[1])
    assert np.allclose(cropped_regions.numpy(), ground_truth)


def test_get_cropped_image_regions_torch_random_nonoverlapping():
    """Random non-overlapping pos  for _get_cropped_image_regions_torch function."""
    box_size = (5, 5)
    num_patches = 4
    image_size = (256, 256)
    test_patch = get_test_patch(box_size)
    image = np.zeros(image_size, dtype=np.float32)

    # Generate non-overlapping positions
    positions = []
    for _ in range(num_patches):
        while True:
            y = np.random.randint(0, image_size[0] - box_size[0] + 1)
            x = np.random.randint(0, image_size[1] - box_size[1] + 1)
            if all(
                not (y <= py < y + box_size[0] and x <= px < x + box_size[1])
                for py, px in positions
            ):
                positions.append((y, x))
                break

    pos_y, pos_x = zip(*positions)

    for x, y in zip(pos_x, pos_y):
        image[y : y + box_size[0], x : x + box_size[1]] = test_patch

    image_tensor = torch.from_numpy(image)
    pos_y_tensor = torch.tensor(pos_y)
    pos_x_tensor = torch.tensor(pos_x)

    cropped_regions = _get_cropped_image_regions_torch(
        image=image_tensor,
        pos_y=pos_y_tensor,
        pos_x=pos_x_tensor,
        box_size=box_size,
        handle_bounds="error",
        padding_mode="constant",
        padding_value=0.0,
    )

    ground_truth = np.stack([test_patch] * num_patches, axis=0)
    assert cropped_regions.shape == (num_patches, box_size[0], box_size[1])
    assert np.allclose(cropped_regions.numpy(), ground_truth)


def test_get_cropped_image_regions_defaults():
    """Test for default parameters for the get_cropped_image_regions function."""
    box_size = (5, 5)
    pos_x = np.array([0, 64, 128])
    pos_y = np.array([0, 100, 15])

    test_patch = get_test_patch(box_size)
    image = np.zeros((256, 256), dtype=np.float32)
    for x, y in zip(pos_x, pos_y):
        image[y : y + box_size[0], x : x + box_size[1]] = test_patch

    cropped_regions = get_cropped_image_regions(
        image=image,
        pos_y=pos_y,
        pos_x=pos_x,
        box_size=box_size,
    )

    ground_truth = np.stack([test_patch] * 3, axis=0)

    assert isinstance(cropped_regions, np.ndarray)
    assert cropped_regions.shape == (3, box_size[0], box_size[1])
    assert np.allclose(cropped_regions, ground_truth)


def test_get_cropped_image_regions_center_top_left_consistency():
    """Test that the center and top-left positions yield the same results."""
    box_size = (5, 5)
    pos_x = np.array([0, 64, 128])
    pos_y = np.array([0, 100, 15])

    test_patch = get_test_patch(box_size)
    image = np.zeros((256, 256), dtype=np.float32)
    for x, y in zip(pos_x, pos_y):
        image[y : y + box_size[0], x : x + box_size[1]] = test_patch

    cropped_regions_center = get_cropped_image_regions(
        image=image,
        pos_y=pos_y + box_size[0] // 2,
        pos_x=pos_x + box_size[1] // 2,
        box_size=box_size,
        pos_reference="center",
    )

    cropped_regions_top_left = get_cropped_image_regions(
        image=image,
        pos_y=pos_y,
        pos_x=pos_x,
        box_size=box_size,
        pos_reference="top-left",
    )

    assert np.allclose(cropped_regions_center, cropped_regions_top_left)


def test_get_cropped_image_regions_edge_of_image_numpy():
    """Test the edge cases (literally) for the get_cropped_image_regions function."""
    # When handle_bounds is 'error', underlying function should raise an error
    box_size = (5, 5)
    pos_x = np.array([-1, 255])
    pos_y = np.array([-1, 255])
    image = np.ones((256, 256), dtype=np.float32)

    # Test with handle_bounds='error' which should throw error
    with pytest.raises(IndexError):
        get_cropped_image_regions(
            image=image,
            pos_y=pos_y,
            pos_x=pos_x,
            box_size=box_size,
            handle_bounds="error",
            pos_reference="top-left",
        )

    # Test with padding of 0.0 which should
    # 1. Not throw an error
    # 2. Return with zeros in the padded areas
    cropped_regions = get_cropped_image_regions(
        image=image,
        pos_y=pos_y,
        pos_x=pos_x,
        box_size=box_size,
        handle_bounds="pad",
        padding_mode="constant",
        padding_value=0.0,
        pos_reference="top-left",
    )

    ground_truth = np.ones((2, box_size[0], box_size[1]), dtype=np.float32)
    # First case
    ground_truth[0, :, 0] = 0  # Left edge padded with zeros
    ground_truth[0, 0, :] = 0  # Top edge padded with zeros
    # Second case
    ground_truth[1, :, 1:] = 0  # Right edge padded with zeros
    ground_truth[1, 1:, :] = 0  # Bottom edge padded with zeros

    assert cropped_regions.shape == (2, box_size[0], box_size[1])
    assert np.allclose(cropped_regions, ground_truth)

    # Test with replicate padding
    cropped_regions_replicate = get_cropped_image_regions(
        image=image,
        pos_y=pos_y,
        pos_x=pos_x,
        box_size=box_size,
        handle_bounds="pad",
        padding_mode="replicate",
        padding_value=0.0,
        pos_reference="top-left",
    )
    ground_truth_replicate = np.ones((2, box_size[0], box_size[1]), dtype=np.float32)

    assert cropped_regions_replicate.shape == (2, box_size[0], box_size[1])
    assert np.allclose(cropped_regions_replicate, ground_truth_replicate)


def test_get_cropped_image_regions_edge_of_image_torch():
    """Test the edge cases for the torch version of get_cropped_image_regions."""
    box_size = (5, 5)
    pos_x = np.array([-1, 255])
    pos_y = np.array([-1, 255])
    image = np.ones((256, 256), dtype=np.float32)

    image_tensor = torch.from_numpy(image)
    pos_x_tensor = torch.from_numpy(pos_x)
    pos_y_tensor = torch.from_numpy(pos_y)

    # Test with handle_bounds='error' which should throw error
    with pytest.raises(IndexError):
        get_cropped_image_regions(
            image=image_tensor,
            pos_y=pos_y_tensor,
            pos_x=pos_x_tensor,
            box_size=box_size,
            handle_bounds="error",
            pos_reference="top-left",
        )

    # Test with padding of 0.0
    cropped_regions = get_cropped_image_regions(
        image=image_tensor,
        pos_y=pos_y_tensor,
        pos_x=pos_x_tensor,
        box_size=box_size,
        handle_bounds="pad",
        padding_mode="constant",
        padding_value=0.0,
        pos_reference="top-left",
    )

    ground_truth = np.ones((2, box_size[0], box_size[1]), dtype=np.float32)
    ground_truth[0, :, 0] = 0  # Left edge padded with zeros
    ground_truth[0, 0, :] = 0  # Top edge padded with zeros
    ground_truth[1, :, 1:] = 0  # Right edge padded with zeros
    ground_truth[1, 1:, :] = 0  # Bottom edge padded with zeros

    assert cropped_regions.shape == (2, box_size[0], box_size[1])
    assert np.allclose(cropped_regions.numpy(), ground_truth)

    # Test with replicate padding
    cropped_regions_replicate = get_cropped_image_regions(
        image=image_tensor,
        pos_y=pos_y_tensor,
        pos_x=pos_x_tensor,
        box_size=box_size,
        handle_bounds="pad",
        padding_mode="replicate",
        padding_value=0.0,
        pos_reference="top-left",
    )
    ground_truth_replicate = np.ones((2, box_size[0], box_size[1]), dtype=np.float32)

    assert cropped_regions_replicate.shape == (2, box_size[0], box_size[1])
    assert np.allclose(cropped_regions_replicate.numpy(), ground_truth_replicate)
