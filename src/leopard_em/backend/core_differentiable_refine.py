"""Deprecated alias of :func:`leopard_em.backend.core_refine_template`.

``core_refine_template`` is differentiable: when grad mode is enabled and any input
requires grad, its refined cross-correlation and z-score carry gradients.
"""

import warnings
from typing import Any

import torch

from leopard_em.backend.core_refine_template import core_refine_template


def core_differentiable_refine(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
    """Deprecated: call ``core_refine_template`` inside ``torch.enable_grad()``.

    Parameters
    ----------
    *args : Any
        Positional arguments of :func:`core_refine_template`.
    **kwargs : Any
        Keyword arguments of :func:`core_refine_template`.

    Returns
    -------
    dict[str, torch.Tensor]
        The output of :func:`core_refine_template`.
    """
    warnings.warn(
        "core_differentiable_refine is deprecated; core_refine_template is "
        "differentiable. Call it inside 'with torch.enable_grad():'.",
        DeprecationWarning,
        stacklevel=2,
    )
    with torch.enable_grad():
        return core_refine_template(*args, **kwargs)
