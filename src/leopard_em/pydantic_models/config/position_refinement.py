"""Serialization and validation of particle position refinement parameters."""

from typing import Annotated

from pydantic import Field

from leopard_em.pydantic_models.custom_types import BaseModel2DTM


class PositionRefinementConfig(BaseModel2DTM):
    """How refine template searches particle positions (GEMM backend only).

    Notes
    -----
    With ``computational_config.backend == "gemm"``, positions are searched in a
    ``(2r + 1) x (2r + 1)`` pixel window instead of the whole valid correlation crop.
    The window starts at the peak of one FFT correlation at each particle's prior
    orientation and defocus, and is re-centered if the best position lands on its
    edge. The ``"fft"`` backend always searches the whole valid crop and ignores this
    configuration.

    Attributes
    ----------
    shift_search_radius : int | None
        Window half-width ``r`` in pixels, defaults to 2. None searches the whole
        valid crop (the same positions as the ``"fft"`` backend).
    fft_centering : bool
        If True (default), the window starts at the peak of one FFT correlation at the
        prior orientation, defocus, and pixel size. If False, it starts at the center
        of the valid crop (the particle position from the input).
    """

    shift_search_radius: Annotated[int, Field(ge=0)] | None = 2
    fft_centering: bool = True
