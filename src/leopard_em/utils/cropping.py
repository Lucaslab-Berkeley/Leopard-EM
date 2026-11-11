"""Extract fixed-size rectangular regions from 2D images (numpy or torch)."""

import warnings
from typing import Literal

import numpy as np
import torch

TORCH_TO_NUMPY_PADDING_MODE = {
    "constant": "constant",
    "reflect": "reflect",
    "replicate": "edge",
}


def get_cropped_image_regions(
    image: torch.Tensor | np.ndarray,
    pos_y: torch.Tensor | np.ndarray,
    pos_x: torch.Tensor | np.ndarray,
    box_size: int | tuple[int, int],
    pos_reference: Literal["center", "top-left"] = "top-left",
    handle_bounds: Literal["pad", "error"] = "pad",
    padding_mode: Literal["constant", "reflect", "replicate"] = "constant",
    padding_value: float = 0.0,
) -> torch.Tensor | np.ndarray:
    """Extracts regions from an image into a stack of cropped images.

    The `pos_reference` argument determines how the (y, x) coordinates are interpreted
    when extracting boxes:

    - If ``pos_reference="center"``:
        The (y, x) coordinate refers to the **center** of the box.
        The box extends from (y - height // 2, x - width // 2) to
        (y + height // 2, x + width // 2).

        Example:
            :                +------------------+
            :                |                  |
            :              height      * (y, x) |
            :                |                  |
            :                +------ width -----+

    - If ``pos_reference="top-left"``:
        The (y, x) coordinate refers to the **top-left corner** of the box.
        The box extends from (y, x) to (y + height, x + width).

        Example:
            :         (y, x) *------ width -----+
            :                |                  |
            :                |                height
            :                |                  |
            :                +------------------+

    Parameters
    ----------
    image : torch.Tensor | np.ndarray
        The input image from which to extract the regions.
    pos_y : torch.Tensor | np.ndarray
        The y positions of the regions to extract. Type must mach `image`
    pos_x : torch.Tensor | np.ndarray
        The x positions of the regions to extract. Type must mach `image`
    box_size : int | tuple[int, int]
        The size of the box to extract. If an integer is passed, the box will be square.
    pos_reference : Literal["center", "top-left"], optional
        The reference point for the positions, by default "center". If "center", the
        boxes extracted will be image[y - box_size // 2 : y + box_size // 2, ...]. If
        "top-left", the boxes will be image[y : y + box_size, ...].
    handle_bounds : Literal["pad", "clip", "error"], optional
        How to handle the bounds of the image, by default "pad". If "pad", the image
        will be padded with the padding value based on the padding mode. If "error", an
        error will be raised if any region exceeds the image bounds. Note clipping is
        not supported since returned stack may have inhomogeneous sizes.
    padding_mode : Literal["constant", "reflect", "replicate"], optional
        The padding mode to use when padding the image, by default "constant".
        "constant" pads with the value `padding_value`, "reflect" pads with the
        reflection of the image at the edge, and "replicate" pads with the last pixel
        of the image. These match the modes available in `torch.nn.functional.pad`.
    padding_value : float, optional
        The value to use for padding when `padding_mode` is "constant", by default 0.0.

    Returns
    -------
    torch.Tensor | np.ndarray
        The stack of cropped images extracted from the input image. Type will match the
        input image type.

    Raises
    ------
    ValueError
        If `pos_reference` is not one of "center" or "top-left", or if `image` is not a
        torch.Tensor or np.ndarray.
    """
    if isinstance(box_size, int):
        box_size = (box_size, box_size)

    if pos_reference == "center":
        pos_y = pos_y - box_size[0] // 2
        pos_x = pos_x - box_size[1] // 2
    elif pos_reference == "top-left":
        pass
    else:
        raise ValueError(f"Unknown pos_reference: {pos_reference}")

    if isinstance(image, torch.Tensor):
        return _get_cropped_image_regions_torch(
            image=image,
            pos_y=pos_y,
            pos_x=pos_x,
            box_size=box_size,
            handle_bounds=handle_bounds,
            padding_mode=padding_mode,
            padding_value=padding_value,
        )

    if isinstance(image, np.ndarray):
        padding_mode_np = TORCH_TO_NUMPY_PADDING_MODE[padding_mode]
        return _get_cropped_image_regions_numpy(
            image=image,
            pos_y=pos_y,
            pos_x=pos_x,
            box_size=box_size,
            handle_bounds=handle_bounds,
            padding_mode=padding_mode_np,
            padding_value=padding_value,
        )

    raise ValueError(f"Unknown image type: {type(image)}")


# pylint: disable=too-many-locals
def _get_cropped_image_regions_numpy(
    image: np.ndarray,
    pos_y: np.ndarray,
    pos_x: np.ndarray,
    box_size: tuple[int, int],
    handle_bounds: Literal["pad", "error"],
    padding_mode: str,
    padding_value: float,
) -> np.ndarray:
    """Helper function for extracting regions from a numpy array.

    NOTE: this function assumes that the position reference is the top-left corner.
    Reference value is handled by the user-exposed 'get_cropped_image_regions' function.
    """
    if handle_bounds == "pad":
        bs1 = box_size[1] - 1
        bs0 = box_size[0] - 1
        pad_kwargs = {}
        if padding_mode == "constant":
            pad_kwargs["constant_values"] = padding_value
        image = np.pad(
            image,
            pad_width=((bs0, bs0), (bs1, bs1)),
            mode=padding_mode,
            **pad_kwargs,
        )
        pos_y = pos_y + bs0
        pos_x = pos_x + bs1

    regions = []
    for y, x in zip(pos_y, pos_x, strict=False):
        if (
            y < 0
            or x < 0
            or y + box_size[0] > image.shape[0]
            or x + box_size[1] > image.shape[1]
        ):
            raise IndexError(
                f"Region bounds [{y}:{y + box_size[0]}, {x}:{x + box_size[1]}] exceed "
                f"image dimensions {image.shape}"
            )

        regions.append(image[y : y + box_size[0], x : x + box_size[1]])

    cropped_images = np.stack(regions)

    return cropped_images


# pylint: disable=too-many-locals
def _get_cropped_image_regions_torch(
    image: torch.Tensor,
    pos_y: torch.Tensor,
    pos_x: torch.Tensor,
    box_size: tuple[int, int],
    handle_bounds: Literal["pad", "error"],
    padding_mode: Literal["constant", "reflect", "replicate"],
    padding_value: float = 0.0,
) -> torch.Tensor:
    """Helper function for extracting regions from a torch tensor.

    NOTE: this function assumes that the position reference is the top-left corner.
    Reference value is handled by the user-exposed 'get_cropped_image_regions' function.
    """
    if handle_bounds == "pad":
        bs1 = box_size[1] - 1
        bs0 = box_size[0] - 1
        pad_kwargs = {}
        if padding_mode == "constant":
            pad_kwargs["value"] = padding_value
        # NOTE: Need to do unsqueeze/squeeze workaround to make torch happy with input
        # tensor shapes. Looks like API for padding may change in the future torch...
        image = torch.nn.functional.pad(
            image.unsqueeze(0),
            pad=(bs1, bs1, bs0, bs0),
            mode=padding_mode,
            **pad_kwargs,
        ).squeeze(0)
        pos_y = pos_y + bs0
        pos_x = pos_x + bs1

    regions = []
    for y, x in zip(pos_y, pos_x, strict=False):
        y = int(y.item() if hasattr(y, "item") else y)
        x = int(x.item() if hasattr(x, "item") else x)
        original_y, original_x = y, x

        if (
            y < 0
            or x < 0
            or y + box_size[0] > image.shape[0]
            or x + box_size[1] > image.shape[1]
        ):
            if handle_bounds == "error":
                raise IndexError(
                    f"Region bounds [{original_y}:{original_y + box_size[0]}, "
                    f"{original_x}:{original_x + box_size[1]}] exceed "
                    f"image dimensions {image.shape}"
                )
            warnings.warn(
                f"Region bounds [{original_y}:{original_y + box_size[0]}, "
                f"{original_x}:{original_x + box_size[1]}] exceed "
                f"image dimensions {image.shape}. Clamping to edges.",
                UserWarning,
                stacklevel=2,
            )
            y = max(0, min(y, image.shape[0] - box_size[0]))
            x = max(0, min(x, image.shape[1] - box_size[1]))

        regions.append(image[y : y + box_size[0], x : x + box_size[1]])

    cropped_images = torch.stack(regions)

    return cropped_images
