"""Utility and helper functions associated with the backend of Leopard-EM."""

import os
import re
import warnings
from typing import Any, Callable, TypeVar

import torch

# Suppress the specific deprecation warnings from PyTorch internals
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=re.escape("Logical operators 'and' and 'or' are deprecated for non-scalar"),
)

F = TypeVar("F", bound=Callable[..., Any])


def attempt_torch_compilation(
    target_func: F, backend: str = "inductor", mode: str = "default"
) -> F:
    """Compile a function using Torch's compilation utilities.

    NOTE: This function will fall back onto the original function if compilation fails
    or is not supported. Under these circumstances, a warning is issued to inform the
    user of the failure, but the program will continue to run with the original
    function.

    Parameters
    ----------
    target_func : Callable
        The function to compile.
    backend : str, optional
        The backend to use for compilation (default is "inductor").
    mode : str, optional
        The mode for compilation (default is "default")

    Returns
    -------
    Callable
        The potentially compiled function.

    Warning
    -------
    If compilation fails, the original function is returned without modification which
    is useful for program consistency. If compilation is not supported, then a
    warning is generated, and the original function is returned.
    """
    # Check if compilation is disabled via environment variable
    disable_compilation = os.environ.get("LEOPARDEM_DISABLE_TORCH_COMPILATION", "0")
    if disable_compilation != "0":
        return target_func

    try:
        compiled_func = torch.compile(target_func, backend=backend, mode=mode)
        return compiled_func  # type: ignore[no-any-return]
    except (RuntimeError, NotImplementedError) as e:
        warnings.warn(
            f"Failed to compile function {target_func.__name__} with"
            f"backend {backend}: {e}. "
            "Returning the original function instead and continuing...",
            UserWarning,
            stacklevel=2,
        )
        return target_func


def normalize_template_projection(
    projections: torch.Tensor,  # shape (batch, h, w)
    small_shape: tuple[int, int],  # (h, w)
    large_shape: tuple[int, int],  # (H, W)
) -> torch.Tensor:
    r"""Subtract mean of edge values and set variance to 1 (in large shape).

    This function uses the fact that variance of a sequence, Var(X), is scaled by the
    relative size of the small (unpadded) and large (padded with zeros) space. Some
    negligible error is introduced into the variance (~1e-4) due to this routine.

    Let $X$ be the large, zero-padded projection and $x$ the small projection each
    with sizes $(H, W)$ and $(h, w)$, respectively. The mean of the zero-padded
    projection in terms of the small projection is:
    .. math::
        \begin{align}
            \mu(X) &= \frac{1}{H \cdot W} \sum_{i=1}^{H} \sum_{j=1}^{W} X_{ij} \\
            \mu(X) &= \frac{1}{H \cdot W} \sum_{i=1}^{h} \sum_{j=1}^{w} X_{ij} + 0 \\
            \mu(X) &= \frac{h \cdot w}{H \cdot W} \mu(x)
        \end{align}
    The variance of the zero-padded projection in terms of the small projection can be
    obtained by:
    .. math::
        \begin{align}
            Var(X) &= \frac{1}{H \cdot W} \sum_{i=1}^{H} \sum_{j=1}^{W} (X_{ij} -
                \mu(X))^2 \\
            Var(X) &= \frac{1}{H \cdot W} \left(\sum_{i=1}^{h}
                \sum_{j=1}^{w} (X_{ij} - \mu(X))^2 +
                \sum_{i=h+1}^{H}\sum_{i=w+1}^{W} \mu(X)^2 \right) \\
            Var(X) &= \frac{1}{H \cdot W} \sum_{i=1}^{h} \sum_{j=1}^{w} (X_{ij} -
                \mu(X))^2 + (H-h)(W-w)\mu(X)^2
        \end{align}

    Parameters
    ----------
    projections : torch.Tensor
        Real-space projections of the template (in small space).
    small_shape : tuple[int, int]
        Shape of the template.
    large_shape : tuple[int, int]
        Shape of the image (in large space).

    Returns
    -------
    torch.Tensor
        Edge-mean subtracted projections, still in small space, but normalized
        so variance of zero-padded projection is 1.
    """
    # Extract edges while preserving batch dimensions
    top_edge = projections[..., 0, :]  # shape: (..., w)
    bottom_edge = projections[..., -1, :]  # shape: (..., w)
    left_edge = projections[..., 1:-1, 0]  # shape: (..., h-2)
    right_edge = projections[..., 1:-1, -1]  # shape: (..., h-2)
    edge_pixels = torch.concatenate(
        [top_edge, bottom_edge, left_edge, right_edge], dim=-1
    )

    # Subtract the edge pixel mean and calculate variance of small, unpadded projection
    projections = projections - edge_pixels.mean(dim=-1)[..., None, None]

    # # Calculate variance like cisTEM (does not match desired results...)
    # variance = (projections**2).sum(dim=(-1, -2), keepdim=True) * relative_size - (
    #     projections.mean(dim=(-1, -2), keepdim=True) * relative_size
    # ) ** 2

    # Fast calculation of mean/var using Torch + appropriate scaling.
    large_size_sqrt = (large_shape[0] * large_shape[1]) ** 0.5
    relative_size = (small_shape[0] * small_shape[1]) / (
        large_shape[0] * large_shape[1]
    )

    mean = torch.mean(projections, dim=(-2, -1), keepdim=True) * relative_size
    mean = mean * relative_size

    # First term of the variance calculation
    variance = torch.sum((projections - mean) ** 2, dim=(-2, -1), keepdim=True)
    # Add the second term of the variance calculation
    variance = variance + (
        (large_shape[0] - small_shape[0]) * (large_shape[1] - small_shape[1]) * mean**2
    )

    projections = (projections * large_size_sqrt) / torch.sqrt(variance.clamp_min(1e-8))
    return projections


def do_iteration_statistics_updates(
    cross_correlation: torch.Tensor,
    current_indexes: torch.Tensor,
    mip: torch.Tensor,
    best_global_index: torch.Tensor,
    correlation_sum: torch.Tensor,
    correlation_squared_sum: torch.Tensor,
    img_h: int,
    img_w: int,
) -> None:
    """Helper function for updating maxima and tracked statistics.

    NOTE: The batch dimensions are effectively unraveled since taking the
    maximum over a single batch dimensions is much faster than
    multi-dimensional maxima.

    NOTE: Updating the maxima was found to be fastest and least memory
    impactful when using torch.where directly. Other methods tested were
    boolean masking and torch.where with tuples of tensor indexes.

    Parameters
    ----------
    cross_correlation : torch.Tensor
        Cross-correlation values for the current iteration. Has shape
        (num_cs, num_defocus, num_orientations, H, W) where 'num_cs' are the number of
        different pixel sizes (controlled by spherical aberration Cs) in the
        cross-correlation batch, 'num_defocus' are the number of different defocus
        values in the cross-correlation batch, and 'num_orientations' are the number of
        different orientations in the cross-correlation batch.
    current_indexes : torch.Tensor
        The global search indexes for the *current* batch of pixel sizes, defocus
        values, and orientations. Has shape `num_cs * num_defocus * num_orientations`
        to uniquely identify the set of pixel sizes, defocus values, and orientations
        associated with the batch from the global search space.
    mip : torch.Tensor
        Maximum intensity projection of the cross-correlation values.
    best_global_index : torch.Tensor
        Previous best global search indexes. Has shape (H, W) and is int32 type.
    correlation_sum : torch.Tensor
        Sum of cross-correlation values for each pixel.
    correlation_squared_sum : torch.Tensor
        Sum of squared cross-correlation values for each pixel.
    img_h : int
        Height of the cross-correlation values.
    img_w : int
        Width of the cross-correlation values.
    """
    cc_reshaped = cross_correlation.view(-1, img_h, img_w)

    # Need two passes for maxima operator for memory efficiency
    # and to distinguish between batch position which would both update
    max_values, max_indices = torch.max(cc_reshaped, dim=0)

    # Do masked updates with torch.where directly (in-place)
    update_mask = max_values > mip
    torch.where(update_mask, max_values, mip, out=mip)
    torch.where(
        update_mask,
        current_indexes[max_indices],
        best_global_index,
        out=best_global_index,
    )

    correlation_sum += cc_reshaped.sum(dim=0)
    correlation_squared_sum += (cc_reshaped**2).sum(dim=0)


# These are compiled normalization and stat update functions
normalize_template_projection_compiled = attempt_torch_compilation(
    normalize_template_projection, backend="inductor", mode="default"
)
do_iteration_statistics_updates_compiled = attempt_torch_compilation(
    do_iteration_statistics_updates,
    backend="inductor",
    mode="max-autotune-no-cudagraphs",
)
