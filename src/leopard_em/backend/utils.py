"""Utility and helper functions associated with the backend of Leopard-EM."""

import os
import re
import warnings
from typing import Any, Callable, TypeVar

import roma
import tensordict
import torch

# Suppress the specific deprecation warnings from PyTorch internals
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=re.escape("Logical operators 'and' and 'or' are deprecated for non-scalar"),
)

F = TypeVar("F", bound=Callable[..., Any])

# This is assuming the Euler angles are in the ZYZ intrinsic format
# AND that the angles are ordered in (phi, theta, psi)
EULER_ANGLE_FMT = "ZYZ"


def combine_euler_angles(angle_a: torch.Tensor, angle_b: torch.Tensor) -> torch.Tensor:
    """Helper function for composing rotations defined by two sets of Euler angles.

    Parameters
    ----------
    angle_a : torch.Tensor
        First set of Euler angles in ZYZ convention.
    angle_b : torch.Tensor
        Second set of Euler angles in ZYZ convention.

    Returns
    -------
    torch.Tensor
        Composed Euler angles representing the combined rotation.
    """
    # Ensure both input angles have the same dtype
    common_dtype = angle_a.dtype
    if angle_b.dtype != common_dtype:
        angle_b = angle_b.to(common_dtype)

    rotmat_a = roma.euler_to_rotmat(
        EULER_ANGLE_FMT, angle_a, degrees=True, device=angle_a.device
    )
    rotmat_b = roma.euler_to_rotmat(
        EULER_ANGLE_FMT, angle_b, degrees=True, device=angle_b.device
    )
    rotmat_c = roma.rotmat_composition((rotmat_a, rotmat_b))
    euler_angles_c = roma.rotmat_to_euler(EULER_ANGLE_FMT, rotmat_c, degrees=True)

    return euler_angles_c


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


@torch.compile  # type: ignore[misc]
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def _stats_and_table_core(
    cross_correlation: torch.Tensor,
    current_indexes: torch.Tensor,
    mip: torch.Tensor,
    best_global_index: torch.Tensor,
    threshold: float,
    valid_shape_h: int,
    valid_shape_w: int,
    needs_valid_cropping: bool = True,
    compute_correlation_table: bool = True,
    apply_eligibility: bool = False,
    pixel_ok: torch.Tensor | None = None,
    orient_ok: torch.Tensor | None = None,
    defocus_ok: torch.Tensor | None = None,
    stats_from_valid_orientations_defocus: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Compiled function to find new maxima and do correlation table updates.

    Parameters
    ----------
    cross_correlation : torch.Tensor
        Cross-correlation values for the current iteration. Has shape
        (num_cs, num_defocus, num_orientations, H, W) where 'num_cs' are the number of
        different pixel sizes (controlled by spherical aberration Cs) in the
        cross-correlation batch, 'num_defocus' are the number of different defocus
        values in the cross-correlation batch, and 'num_orientations' are the number of
        different orientations in the cross-correlation batch. H and W can either be the
        full image heigh/width or the valid cropped height/width.
    current_indexes : torch.Tensor
        The global search indexes for the *current* batch of pixel sizes, defocus
        values, and orientations. Has shape `num_cs * num_defocus * num_orientations`
        to uniquely identify the set of pixel sizes, defocus values, and orientations
        associated with the batch from the global search space.
    mip : torch.Tensor
        Maximum intensity projection of the cross-correlation values.
    best_global_index : torch.Tensor
        Previous best global search indexes. Has shape (H, W) and is int32 type.
    threshold : float
        The threshold value for adding entries to the correlation table.
    valid_shape_h : int
        Height of the valid region of the cross-correlation values.
    valid_shape_w : int
        Width of the valid region of the cross-correlation values.
    needs_valid_cropping : bool, optional
        Whether the cross-correlation tensor should be cropped (via a view operation).
        If False, the cross-correlation tensor is assumed to already be in the valid
        shape.
    compute_correlation_table : bool, optional
        Whether to find and return threshold exceedances for the correlation table.
    apply_eligibility : bool, optional
        If True, mask illegal (pixel, orientation, defocus) tuples before
        taking the MIP max.
    pixel_ok : torch.Tensor, optional
        Boolean mask of shape (H, W). True where the pixel is in play.
    orient_ok : torch.Tensor, optional
        Boolean mask of shape (B,) for the current batch of search indexes.
    defocus_ok : torch.Tensor, optional
        Boolean mask of shape (B, H, W). True where this batch index's defocus
        is allowed at that pixel.
    stats_from_valid_orientations_defocus : bool, optional
        If True, running sums use only eligible (pixel, orientation, defocus)
        tuples. Default False uses all searched angles and defocus values.
    """
    # create cropped view as in existing functions
    if needs_valid_cropping:
        cc_reshaped = cross_correlation.view(
            -1, cross_correlation.shape[-2], cross_correlation.shape[-1]
        )
        cc_reshaped = cc_reshaped.as_strided(
            size=(cc_reshaped.shape[0], valid_shape_h, valid_shape_w),
            stride=(
                cc_reshaped.stride(0),
                cc_reshaped.stride(1),
                cc_reshaped.stride(2),
            ),
        )
    else:
        cc_reshaped = cross_correlation.view(
            -1, cross_correlation.shape[-2], cross_correlation.shape[-1]
        )

    if apply_eligibility:
        assert pixel_ok is not None
        assert orient_ok is not None
        eligible = orient_ok[:, None, None] & pixel_ok[None, :, :]
        if defocus_ok is not None:
            eligible = eligible & defocus_ok
        neg_inf = torch.finfo(cc_reshaped.dtype).min
        cc_for_mip = torch.where(eligible, cc_reshaped, neg_inf)
        max_values, max_indices = torch.max(cc_for_mip, dim=0)
        any_allowed = eligible.any(dim=0)
        update_mask = (max_values > mip) & any_allowed
        if stats_from_valid_orientations_defocus:
            cc_for_stats = torch.where(eligible, cc_reshaped, 0)
            corr_count = eligible.to(cc_reshaped.dtype).sum(dim=0)
        else:
            cc_for_stats = cc_reshaped
            corr_count = torch.zeros(
                (valid_shape_h, valid_shape_w),
                dtype=cc_reshaped.dtype,
                device=cc_reshaped.device,
            )
    else:
        max_values, max_indices = torch.max(cc_reshaped, dim=0)
        update_mask = max_values > mip
        cc_for_stats = cc_reshaped
        corr_count = torch.zeros(
            (valid_shape_h, valid_shape_w),
            dtype=cc_reshaped.dtype,
            device=cc_reshaped.device,
        )
        eligible = None

    new_mip = torch.where(update_mask, max_values, mip)
    new_best_global_index = torch.where(
        update_mask, current_indexes[max_indices], best_global_index
    )

    corr_sum = cc_for_stats.sum(dim=0)
    corr_sq_sum = (cc_for_stats * cc_for_stats).sum(dim=0)

    # find threshold exceedances (for correlation table)
    if compute_correlation_table:
        table_mask = cc_reshaped > threshold
        if apply_eligibility:
            table_mask = table_mask & eligible
        batch_idxs, y_idxs, x_idxs = torch.where(table_mask)
        values = cc_reshaped[batch_idxs, y_idxs, x_idxs]
        global_idxs = current_indexes[batch_idxs]
    else:
        empty_int = torch.empty(0, dtype=torch.int64, device=cc_reshaped.device)
        y_idxs = x_idxs = empty_int
        values = torch.empty(0, dtype=cc_reshaped.dtype, device=cc_reshaped.device)
        global_idxs = current_indexes[empty_int]

    return (
        new_mip,
        new_best_global_index,
        corr_sum,
        corr_sq_sum,
        corr_count,
        global_idxs,
        y_idxs,
        x_idxs,
        values,
    )


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
def do_iteration_and_correlation_table_updates(
    cross_correlation: torch.Tensor,
    current_indexes: torch.Tensor,
    correlation_table: tensordict.TensorDict,
    mip: torch.Tensor,
    best_global_index: torch.Tensor,
    correlation_sum: torch.Tensor,
    correlation_squared_sum: torch.Tensor,
    threshold: float,
    valid_shape_h: int,
    valid_shape_w: int,
    needs_valid_cropping: bool = True,
    compute_correlation_table: bool = True,
    eligible_pixels: torch.Tensor | None = None,
    allowed_search_mask: torch.Tensor | None = None,
    defocus_eligible: torch.Tensor | None = None,
    n_orientations: int = 1,
    stats_from_valid_orientations_defocus: bool = False,
    correlation_count: torch.Tensor | None = None,
) -> None:
    """Helper function for updating maxima, tracked statistics, and correlation table.

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
    correlation_table : tensordict.TensorDict
        The correlation table to update. Has keys
        ["threshold", "pos_x", "pos_y", "corr_value"] each of which are tensors.
    mip : torch.Tensor
        Maximum intensity projection of the cross-correlation values.
    best_global_index : torch.Tensor
        Previous best global search indexes. Has shape (H, W) and is int32 type.
    correlation_sum : torch.Tensor
        Sum of cross-correlation values for each pixel.
    correlation_squared_sum : torch.Tensor
        Sum of squared cross-correlation values for each pixel.
    threshold : float
        The threshold value for adding entries to the correlation table.
    valid_shape_h : int
        Height of the valid region of the cross-correlation values.
    valid_shape_w : int
        Width of the valid region of the cross-correlation values.
    needs_valid_cropping : bool, optional
        Whether the cross-correlation tensor should be cropped (via a view operation)
        to the valid dimensions (defined by `img_h` and `img_w`). If False, the
        cross-correlation tensor is assumed to already be in the valid shape.
    compute_correlation_table : bool, optional
        Whether to threshold the cross-correlation values and add exceedances to the
        correlation table.
    eligible_pixels : torch.Tensor, optional
        Boolean (H, W) mask of pixels in play. If None, all pixels are eligible.
    allowed_search_mask : torch.Tensor, optional
        Boolean mask indexed by global search index. If None, every searched
        index in this run is allowed.
    defocus_eligible : torch.Tensor, optional
        Boolean ``(n_defocus, H, W)`` mask. If None, every defocus in this run
        is allowed at every pixel.
    n_orientations : int, optional
        Number of orientations in the global search. Used to decode the defocus
        index from ``current_indexes``. Default 1.
    stats_from_valid_orientations_defocus : bool, optional
        If True, accumulate sums and a per-pixel count over eligible
        (pixel, orientation, defocus) tuples only.
    correlation_count : torch.Tensor, optional
        Running eligible-tuple count per pixel. Updated when
        ``stats_from_valid_orientations_defocus`` is True.
    """
    apply_eligibility = (
        eligible_pixels is not None
        or allowed_search_mask is not None
        or defocus_eligible is not None
    )
    device = cross_correlation.device
    pixel_ok = eligible_pixels
    if apply_eligibility and pixel_ok is None:
        pixel_ok = torch.ones(
            (valid_shape_h, valid_shape_w),
            dtype=torch.bool,
            device=device,
        )
    if apply_eligibility:
        if allowed_search_mask is None:
            orient_ok = torch.ones(
                current_indexes.numel(),
                dtype=torch.bool,
                device=device,
            )
        else:
            orient_ok = allowed_search_mask[current_indexes]
    else:
        orient_ok = None

    defocus_ok = None
    if defocus_eligible is not None:
        n_defocus = int(defocus_eligible.shape[0])
        n_orient = max(int(n_orientations), 1)
        defocus_idx = (current_indexes % (n_defocus * n_orient)) // n_orient
        defocus_ok = defocus_eligible[defocus_idx]

    (
        new_mip,
        new_best_global_index,
        corr_sum,
        corr_sq_sum,
        corr_count,
        global_idxs,
        y_idxs,
        x_idxs,
        values,
    ) = _stats_and_table_core(
        cross_correlation,
        current_indexes,
        mip,
        best_global_index,
        threshold,
        valid_shape_h,
        valid_shape_w,
        needs_valid_cropping=needs_valid_cropping,
        compute_correlation_table=compute_correlation_table,
        apply_eligibility=apply_eligibility,
        pixel_ok=pixel_ok,
        orient_ok=orient_ok,
        defocus_ok=defocus_ok,
        stats_from_valid_orientations_defocus=stats_from_valid_orientations_defocus,
    )

    # update inplace the statistics tensors
    mip.copy_(new_mip)
    best_global_index.copy_(new_best_global_index)

    correlation_sum += corr_sum
    correlation_squared_sum += corr_sq_sum
    if correlation_count is not None and stats_from_valid_orientations_defocus:
        correlation_count += corr_count

    # update correlation_table (tensordict operations not compiled)
    if global_idxs.numel() > 0:
        correlation_table["global_idx"] = torch.cat(
            [correlation_table["global_idx"], global_idxs]
        )
        correlation_table["pos_x"] = torch.cat([correlation_table["pos_x"], x_idxs])
        correlation_table["pos_y"] = torch.cat([correlation_table["pos_y"], y_idxs])
        correlation_table["corr_value"] = torch.cat(
            [correlation_table["corr_value"], values]
        )


# These are compiled normalization and stat update functions
normalize_template_projection_compiled = attempt_torch_compilation(
    normalize_template_projection, backend="inductor", mode="default"
)
