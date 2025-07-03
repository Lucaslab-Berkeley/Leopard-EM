"""Utility and helper functions associated with the backend of Leopard-EM."""

import warnings
from multiprocessing import Manager, Process
from typing import Any, Callable, Optional, TypeVar

import torch

F = TypeVar("F", bound=Callable[..., Any])


def attempt_torch_compilation(target_func: F, backend: str = "inductor") -> F:
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
    try:
        compiled_func = torch.compile(target_func, backend=backend)
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
    projections -= edge_pixels.mean(dim=-1)[..., None, None]

    # # Calculate variance like cisTEM (does not match desired results...)
    # variance = (projections**2).sum(dim=(-1, -2), keepdim=True) * relative_size - (
    #     projections.mean(dim=(-1, -2), keepdim=True) * relative_size
    # ) ** 2

    # Fast calculation of mean/var using Torch + appropriate scaling.
    relative_size = (small_shape[0] * small_shape[1]) / (
        large_shape[0] * large_shape[1]
    )
    mean = torch.mean(projections, dim=(-2, -1), keepdim=True) * relative_size
    mean *= relative_size

    # First term of the variance calculation
    variance = torch.sum((projections - mean) ** 2, dim=(-2, -1), keepdim=True)
    # Add the second term of the variance calculation
    variance += (
        (large_shape[0] - small_shape[0]) * (large_shape[1] - small_shape[1]) * mean**2
    )
    variance /= large_shape[0] * large_shape[1]

    return projections / torch.sqrt(variance)


# NOTE: Disabling pylint for number of argument since these all need updated in-place
# and is more efficient than packing into some other type of object.
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def do_iteration_statistics_updates(
    cross_correlation: torch.Tensor,
    euler_angles: torch.Tensor,
    defocus_values: torch.Tensor,
    pixel_values: torch.Tensor,
    mip: torch.Tensor,
    best_phi: torch.Tensor,
    best_theta: torch.Tensor,
    best_psi: torch.Tensor,
    best_defocus: torch.Tensor,
    best_pixel_size: torch.Tensor,
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
        Cross-correlation values for the current iteration. Has either shape
        (batch, H, W) or (defocus, orientations, H, W).
    euler_angles : torch.Tensor
        Euler angles for the current iteration. Has shape (orientations, 3).
    defocus_values : torch.Tensor
        Defocus values for the current iteration. Has shape (defocus,).
    pixel_values : torch.Tensor
        Pixel size values for the current iteration. Has shape (pixel_size_batch,).
    mip : torch.Tensor
        Maximum intensity projection of the cross-correlation values.
    best_phi : torch.Tensor
        Best phi angle for each pixel.
    best_theta : torch.Tensor
        Best theta angle for each pixel.
    best_psi : torch.Tensor
        Best psi angle for each pixel.
    best_defocus : torch.Tensor
        Best defocus value for each pixel.
    best_pixel_size : torch.Tensor
        Best pixel size value for each pixel.
    correlation_sum : torch.Tensor
        Sum of cross-correlation values for each pixel.
    correlation_squared_sum : torch.Tensor
        Sum of squared cross-correlation values for each pixel.
    img_h : int
        Height of the cross-correlation values.
    img_w : int
        Width of the cross-correlation values.
    """
    num_cs, num_defocs, num_orientations = cross_correlation.shape[0:3]
    max_values, max_indices = torch.max(cross_correlation.view(-1, img_h, img_w), dim=0)
    max_cs_idx = (max_indices // (num_defocs * num_orientations)) % num_cs
    max_defocus_idx = (max_indices // num_orientations) % num_defocs
    max_orientation_idx = max_indices % num_orientations

    # using torch.where directly
    update_mask = max_values > mip

    torch.where(update_mask, max_values, mip, out=mip)
    torch.where(
        update_mask, euler_angles[max_orientation_idx, 0], best_phi, out=best_phi
    )
    torch.where(
        update_mask, euler_angles[max_orientation_idx, 1], best_theta, out=best_theta
    )
    torch.where(
        update_mask, euler_angles[max_orientation_idx, 2], best_psi, out=best_psi
    )
    torch.where(
        update_mask, defocus_values[max_defocus_idx], best_defocus, out=best_defocus
    )
    torch.where(
        update_mask, pixel_values[max_cs_idx], best_pixel_size, out=best_pixel_size
    )

    correlation_sum += cross_correlation.view(-1, img_h, img_w).sum(dim=0)
    correlation_squared_sum += (cross_correlation.view(-1, img_h, img_w) ** 2).sum(
        dim=0
    )


def run_multiprocess_jobs(
    target: Callable,
    kwargs_list: list[dict[str, Any]],
    extra_args: tuple[Any, ...] = (),
    extra_kwargs: Optional[dict[str, Any]] = None,
) -> dict[Any, Any]:
    """Helper function for running multiple processes on the same target function.

    Spawns multiple processes to run the same target function with different keyword
    arguments, aggregates results in a shared dictionary, and returns them.

    Parameters
    ----------
    target : Callable
        The function that each process will execute. It must accept at least two
        positional arguments: a shared dict and a unique index.
    kwargs_list : list[dict[str, Any]]
        A list of dictionaries containing keyword arguments for each process.
    extra_args : tuple[Any, ...], optional
        Additional positional arguments to pass to the target (prepending the shared
        parameters).
    extra_kwargs : Optional[dict[str, Any]], optional
        Additional common keyword arguments for all processes.

    Returns
    -------
    dict[Any, Any]
        Aggregated results stored in the shared dictionary.

    Example
    -------
    ```
    def worker_fn(result_dict, idx, param1, param2):
        result_dict[idx] = param1 + param2


    kwargs_per_process = [
        {"param1": 1, "param2": 2},
        {"param1": 3, "param2": 4},
    ]
    results = run_multiprocess_jobs(worker_fn, kwargs_per_process)
    print(results)
    # {0: 3, 1: 7}
    ```
    """
    if extra_kwargs is None:
        extra_kwargs = {}

    # Manager object for shared result data as a dictionary
    manager = Manager()
    result_dict = manager.dict()
    processes: list[Process] = []

    for i, kwargs in enumerate(kwargs_list):
        args = (*extra_args, result_dict, i)

        # Merge per-process kwargs with common kwargs.
        proc_kwargs = {**extra_kwargs, **kwargs}
        p = Process(target=target, args=args, kwargs=proc_kwargs)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    return dict(result_dict)


# These are compiled normalization and stat update functions
normalize_template_projection_compiled = attempt_torch_compilation(
    normalize_template_projection, backend="inductor"
)
do_iteration_statistics_updates_compiled = attempt_torch_compilation(
    do_iteration_statistics_updates, backend="inductor"
)
