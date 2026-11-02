"""Functionality for working with box geometry in real and statistic-map space.

For top-left reference
----------------------
Note that the particle is still centered within the box, and that the (x, y) coordinate
refers to the top-left corner of the template projection, for example a shape (512, 512)
projection would place the center of the particle at/around (x+256, y+256). Additional
padding is added around the tight projection box by `extracted_box_size`.

         +----------------------------------+
         |                                  |
         |                                  |
         |     (x, y) *=== box_w ===+       |
         |            |             |       |
         |            |     ....  box_h     |
    img_height        |    ......   |       |
         |            |     ....    |       |
         |            |             |       |
         |            +=============+       |
         |                                  |
         +------------ img_width -----------+

For center reference
--------------------
When center reference is used, then the position is interpreted as the center of the
particle, and the boxes are extracted around this x and y position as follows:

         +----------------------------------+
         |                                  |
         |                                  |
         |            +=== box_w ===+       |
         |            |             |       |
         |            |     ....    |       |
    img_height        |(x, y).*.. box_h     |
         |            |     ....    |       |
         |            |             |       |
         |            +=============+       |
         |                                  |
         +------------ img_width -----------+

"""

from dataclasses import dataclass
from typing import Literal, TypeVar

import numpy as np
import torch

# Positions are returned with the same array type they were given in
PositionArray = TypeVar("PositionArray", np.ndarray, torch.Tensor)


@dataclass(frozen=True)
class ParticleBoxGeometry:
    """Box and template sizes with cropping coordinates.

    Attributes
    ----------
    extracted_box_size : tuple[int, int]
        Size of the extracted particle boxes ``(box_h, box_w)`` in pixels.
    original_template_size : tuple[int, int]
        Size of the template used during template matching ``(h, w)`` in pixels.
    """

    extracted_box_size: tuple[int, int]
    original_template_size: tuple[int, int]

    @property
    def valid_size(self) -> tuple[int, int]:
        """Size of the valid cross-correlation region, ``box - tmpl + 1``."""
        box_h, box_w = self.extracted_box_size
        h, w = self.original_template_size
        return box_h - h + 1, box_w - w + 1

    @property
    def margin(self) -> tuple[int, int]:
        """Offset of the template footprint within the box, ``(box - tmpl) // 2``."""
        box_h, box_w = self.extracted_box_size
        h, w = self.original_template_size
        return (box_h - h) // 2, (box_w - w) // 2

    def template_top_left(
        self,
        pos_y: PositionArray,
        pos_x: PositionArray,
        pos_reference: Literal["center", "top-left"] = "top-left",
    ) -> tuple[PositionArray, PositionArray]:
        """Top-left corner of the template footprint for each position.

        Parameters
        ----------
        pos_y : np.ndarray | torch.Tensor
            Particle y positions in pixels.
        pos_x : np.ndarray | torch.Tensor
            Particle x positions in pixels.
        pos_reference : Literal["center", "top-left"], optional
            Point of the template footprint coordinates, by default "top-left".

        Returns
        -------
        tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]
            The ``(y, x)`` template top-left positions.
        """
        if pos_reference == "center":
            h, w = self.original_template_size
            return pos_y - h // 2, pos_x - w // 2
        return pos_y, pos_x

    def box_origins(
        self,
        pos_y: PositionArray,
        pos_x: PositionArray,
        pos_reference: Literal["center", "top-left"] = "top-left",
    ) -> tuple[PositionArray, PositionArray]:
        """Top-left corner ``O`` shared by the particle box and its valid region.

        Parameters
        ----------
        pos_y : np.ndarray | torch.Tensor
            Particle y positions in pixels.
        pos_x : np.ndarray | torch.Tensor
            Particle x positions in pixels.
        pos_reference : Literal["center", "top-left"], optional
            Point of the template footprint coordinates, by default "top-left".

        Returns
        -------
        tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]
            The ``(y, x)`` crop origins. Crop ``extracted_box_size`` from here for the
            particle image, or ``valid_size`` for the statistic-map region.
        """
        top_y, top_x = self.template_top_left(pos_y, pos_x, pos_reference)
        margin_y, margin_x = self.margin
        return top_y - margin_y, top_x - margin_x

    def template_centers(
        self,
        pos_y: PositionArray,
        pos_x: PositionArray,
        pos_reference: Literal["center", "top-left"] = "top-left",
    ) -> tuple[PositionArray, PositionArray]:
        """Center of the template footprint for each position.

        Parameters
        ----------
        pos_y : np.ndarray | torch.Tensor
            Particle y positions in pixels.
        pos_x : np.ndarray | torch.Tensor
            Particle x positions in pixels.
        pos_reference : Literal["center", "top-left"], optional
            Point of the template footprint coordinates, by default "top-left".

        Returns
        -------
        tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]
            The ``(y, x)`` template centers, e.g. where per-particle motion is sampled.
        """
        top_y, top_x = self.template_top_left(pos_y, pos_x, pos_reference)
        h, w = self.original_template_size
        return top_y + h // 2, top_x + w // 2
