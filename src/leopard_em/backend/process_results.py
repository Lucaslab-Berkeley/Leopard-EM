"""Functions related to result processing after backend functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import tensordict
import torch

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class MatchTemplateRunResult(TypedDict):
    """One ``core_match_template`` (or distributed) result bundle."""

    mip: torch.Tensor
    best_global_index: torch.Tensor
    correlation_sum: torch.Tensor
    correlation_squared_sum: torch.Tensor


def _to_cpu_float32(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.float32)
    return torch.as_tensor(x, dtype=torch.float32).cpu()


def _to_cpu_int32(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.int32)
    return torch.as_tensor(x, dtype=torch.int32).cpu()


def aggregate_distributed_results(
    results: list[dict[str, torch.Tensor | np.ndarray]],
) -> dict[str, torch.Tensor]:
    """Combine the 2DTM results from multiple devices.

    NOTE: This assumes that all tensors have been passed back to the CPU and are in
    the form of numpy arrays.

    Parameters
    ----------
    results : list[dict[str, np.ndarray]]
        List of dictionaries containing the results from each device. Each dictionary
        contains the following keys:
            - "mip": Maximum intensity projection of the cross-correlation values.
            - "best_global_index": Best global search index
            - "correlation_sum": Sum of cross-correlation values for each pixel.
            - "correlation_squared_sum": Sum of squared cross-correlation values for
              each pixel.
    """
    # Ensure all the tensors are passed back to CPU as numpy arrays
    # Not sure why cannot sync across devices, but this is a workaround
    results = [
        {
            key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
            for key, value in result.items()
        }
        for result in results
    ]

    # Stack results from all devices into a single array. Dim 0 is device index
    mips = np.stack([result["mip"] for result in results], axis=0)
    best_index = np.stack([result["best_global_index"] for result in results], axis=0)

    # Find the maximum MIP across all devices, then decode the best index
    mip_max = mips.max(axis=0)
    mip_argmax = mips.argmax(axis=0)
    best_index = np.take_along_axis(best_index, mip_argmax[None, ...], axis=0)[0]

    # Sum the sums and squared sums of the cross-correlation values
    correlation_sum = np.stack(
        [result["correlation_sum"] for result in results], axis=0
    ).sum(axis=0)
    correlation_squared_sum = np.stack(
        [result["correlation_squared_sum"] for result in results], axis=0
    ).sum(axis=0)

    # Cast back to torch tensors on the CPU
    mip_max = torch.from_numpy(mip_max)
    best_index = torch.from_numpy(best_index)
    correlation_sum = torch.from_numpy(correlation_sum)
    correlation_squared_sum = torch.from_numpy(correlation_squared_sum)

    # Concatenate the per-device/per-rank correlation table entries
    per_key_values: dict[str, list[torch.Tensor]] = {}
    threshold = None
    for result in results:
        correlation_table = result["correlation_table"]
        correlation_table = (
            correlation_table.cpu().to_dict()
            if isinstance(correlation_table, tensordict.TensorDict)
            else dict(correlation_table)
        )
        threshold = correlation_table.pop("threshold")
        for key, value in correlation_table.items():
            per_key_values.setdefault(key, []).append(torch.as_tensor(value))

    full_correlation_table = {
        key: torch.cat(values) for key, values in per_key_values.items()
    }
    full_correlation_table["threshold"] = threshold

    return {
        "mip": mip_max,
        "best_global_index": best_index,
        "correlation_sum": correlation_sum,
        "correlation_squared_sum": correlation_squared_sum,
        "correlation_table": full_correlation_table,
    }


# pylint: disable=too-many-locals
def decode_global_search_index(
    global_indices: torch.Tensor,  # integer tensor
    pixel_values: torch.Tensor,  # (num_cs,)
    defocus_values: torch.Tensor,  # (num_defocus,)
    euler_angles: torch.Tensor,  # (num_orientations, 3)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode flattened global indices back into (cs, defocus, orientation)."""
    _ = pixel_values  # Unused, but possible to add in future

    # num_cs = pixel_values.shape[0]
    num_defocus = defocus_values.shape[0]
    num_orientations = euler_angles.shape[0]

    stride_cs = num_defocus * num_orientations
    stride_defocus = num_orientations

    # Calculate the indexes for each "best" array
    pixel_idx = global_indices // stride_cs
    rem = global_indices % stride_cs
    defocus_idx = rem // stride_defocus
    orientations_idx = rem % stride_defocus

    phi = euler_angles[orientations_idx, 0]
    theta = euler_angles[orientations_idx, 1]
    psi = euler_angles[orientations_idx, 2]
    defocus = defocus_values[defocus_idx]
    pixels = pixel_values[pixel_idx]

    return phi, theta, psi, defocus, pixels


# pylint: disable=too-many-locals
def process_correlation_table(
    correlation_table: dict[int | str, Any],
    pixel_values: torch.Tensor,  # (num_cs,)
    defocus_values: torch.Tensor,  # (num_defocus,)
    euler_angles: torch.Tensor,  # (num_orientations, 3)
) -> dict[str, list[float | int]]:
    """Process the correlation table by applying a threshold.

    Parameters
    ----------
    correlation_table : dict[int, torch.Tensor]
        Dictionary containing the correlation table. Keys are global search indices,
        values are tensors of shape (num_hits, 3) containing (x, y, cc) values.
    pixel_values : torch.Tensor
        Tensor containing the pixel values used in the search. Shape is (num_cs,).
    defocus_values : torch.Tensor
        Tensor containing the defocus values used in the search. Shape is
        (num_defocus,).
    euler_angles : torch.Tensor
        Tensor containing the Euler angles used in the search. Shape is
        (num_orientations, 3).

    Returns
    -------
    dict[str, list[float | int]]
        Processed correlation with keys for the unique point in search space and image
        position for all cross-correlations which surpassed the threshold.
    """
    threshold = correlation_table.pop("threshold")
    threshold = threshold.item() if isinstance(threshold, torch.Tensor) else threshold

    # Convert string keys to integer tensor for decoding
    global_indices = correlation_table["global_idx"]
    phi, theta, psi, defocus, pixel_values = decode_global_search_index(
        global_indices, pixel_values, defocus_values, euler_angles
    )

    processed_table = {
        "threshold": threshold,
        "global_idx": global_indices.numpy().tolist(),
        "pixel_size": pixel_values.numpy().tolist(),
        "defocus": defocus.numpy().tolist(),
        "phi": phi.numpy().tolist(),
        "theta": theta.numpy().tolist(),
        "psi": psi.numpy().tolist(),
        "x": correlation_table["pos_x"].numpy().tolist(),
        "y": correlation_table["pos_y"].numpy().tolist(),
        "correlation": correlation_table["corr_value"].numpy().tolist(),
    }

    return processed_table


def correlation_sum_and_squared_sum_to_mean_and_variance(
    correlation_sum: torch.Tensor,
    correlation_squared_sum: torch.Tensor,
    total_correlation_positions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert the sum and squared sum of the correlation values to mean and variance.

    Parameters
    ----------
    correlation_sum : torch.Tensor
        Sum of the correlation values.
    correlation_squared_sum : torch.Tensor
        Sum of the squared correlation values.
    total_correlation_positions : int
        Total number cross-correlograms calculated.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple containing the mean and variance of the correlation values.
    """
    correlation_mean = correlation_sum / total_correlation_positions
    correlation_variance = correlation_squared_sum / total_correlation_positions
    correlation_variance -= correlation_mean**2
    correlation_variance = torch.sqrt(torch.clamp(correlation_variance, min=0))
    return correlation_mean, correlation_variance


def scale_mip(
    mip: torch.Tensor,
    mip_scaled: torch.Tensor,
    correlation_sum: torch.Tensor,
    correlation_squared_sum: torch.Tensor,
    total_correlation_positions: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale the MIP to Z-score map by the mean and variance of the correlation values.

    Z-score is accounting for the variation in image intensity and spurious correlations
    by subtracting the mean and dividing by the standard deviation pixel-wise. Since
    cross-correlation values are roughly normally distributed for pure noise, Z-score
    effectively becomes a measure of how unexpected (highly correlated to the reference
    template) a region is in the image. Note that we are looking at maxima of millions
    of Gaussian distributions, so Z-score has to be compared with a generalized extreme
    value distribution (GEV) to determine significance (done elsewhere).

    NOTE: This method also updates the correlation_sum and correlation_squared_sum
    tensors in-place into the mean and variance, respectively. Likely should reflect
    conversions in variable names...

    Parameters
    ----------
    mip : torch.Tensor
        MIP of the correlation values.
    mip_scaled : torch.Tensor
        Scaled MIP of the correlation values.
    correlation_sum : torch.Tensor
        Sum of the correlation values. Updated to mean of the correlation values.
    correlation_squared_sum : torch.Tensor
        Sum of the squared correlation values. Updated to variance of the correlation.
    total_correlation_positions : int
        Total number cross-correlograms calculated.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple containing, in order, the MIP, scaled MIP, correlation mean, and
        correlation variance.
    """
    corr_mean, corr_variance = correlation_sum_and_squared_sum_to_mean_and_variance(
        correlation_sum, correlation_squared_sum, total_correlation_positions
    )

    # Calculate normalized MIP
    mip_scaled = mip - corr_mean
    torch.where(
        corr_variance != 0,  # preventing zero division error, albeit unlikely
        mip_scaled / corr_variance,
        torch.zeros_like(mip_scaled),
        out=mip_scaled,
    )

    # # Update correlation_sum and correlation_squared_sum to mean and variance
    # correlation_sum.copy_(corr_mean)
    # correlation_squared_sum.copy_(corr_variance)

    return mip, mip_scaled, corr_mean, corr_variance


def merge_runs_pooled_zscore(  # pylint: disable=too-many-locals
    runs: Sequence[Mapping[str, Any]],
    *,
    pooled_total_correlation_positions: int,
    run_tag_values: torch.Tensor | Sequence[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Merge several match-template runs using one pooled noise model, then z-score.

    For each run (e.g. one defocus plane), ``correlation_*`` are summed over the
    search space of that run. Across runs, this function takes the elementwise maximum
    of ``mip``, keeps ``best_global_index`` (and optional per-run tags) from the
    winning run at each pixel, **sums** ``correlation_sum`` and
    ``correlation_squared_sum`` across runs, then applies :func:`scale_mip` once with
    ``pooled_total_correlation_positions`` (typically ``n_runs * n_orientations`` when
    each run sums over the same orientation count).

    Parameters
    ----------
    runs
        Each mapping must provide ``mip``, ``best_global_index``,
        ``correlation_sum``, and ``correlation_squared_sum`` (``torch.Tensor`` or
        array-like).
    pooled_total_correlation_positions
        Denominator for the pooled mean/std used in the final z-score map.
    run_tag_values
        If set, length ``len(runs)`` (e.g. defocus offset per plane). Per-pixel
        ``best_run_tag`` in the output copies the tag for the run that won the MIP
        at that pixel.

    Returns
    -------
    dict
        ``mip`` (max raw MIP), ``scaled_mip``, ``correlation_mean``,
        ``correlation_variance`` (from pooled stats), ``best_global_index``, and
        optionally ``best_run_tag``.
    """
    if len(runs) == 0:
        raise ValueError("runs must be non-empty")

    tags: torch.Tensor | None = None
    if run_tag_values is not None:
        tags = _to_cpu_float32(run_tag_values).reshape(-1)
        if tags.numel() != len(runs):
            raise ValueError(
                "run_tag_values must have length len(runs); "
                f"got {tags.numel()} and {len(runs)} runs."
            )

    mip_acc: torch.Tensor | None = None
    best_gi: torch.Tensor | None = None
    best_tag: torch.Tensor | None = None
    corr_sum_acc: torch.Tensor | None = None
    corr_sq_acc: torch.Tensor | None = None

    for i, run in enumerate(runs):
        mip_k = _to_cpu_float32(run["mip"])
        bgi = _to_cpu_int32(run["best_global_index"])
        csum = _to_cpu_float32(run["correlation_sum"])
        csq = _to_cpu_float32(run["correlation_squared_sum"])

        if mip_acc is None:
            mip_acc = mip_k.clone()
            best_gi = bgi.clone()
            corr_sum_acc = csum.clone()
            corr_sq_acc = csq.clone()
            if tags is not None:
                best_tag = torch.full_like(mip_acc, tags[i].item())
        else:
            improved = mip_k > mip_acc
            mip_acc = torch.maximum(mip_acc, mip_k)
            best_gi = torch.where(improved, bgi, best_gi)
            corr_sum_acc = corr_sum_acc + csum
            corr_sq_acc = corr_sq_acc + csq
            if best_tag is not None and tags is not None:
                best_tag = torch.where(
                    improved,
                    torch.full_like(mip_acc, tags[i].item()),
                    best_tag,
                )

    assert mip_acc is not None and best_gi is not None
    assert corr_sum_acc is not None and corr_sq_acc is not None

    mip_scaled = torch.empty_like(mip_acc)
    mip_acc, mip_scaled, corr_mean, corr_var = scale_mip(
        mip=mip_acc,
        mip_scaled=mip_scaled,
        correlation_sum=corr_sum_acc,
        correlation_squared_sum=corr_sq_acc,
        total_correlation_positions=pooled_total_correlation_positions,
    )

    out: dict[str, torch.Tensor] = {
        "mip": mip_acc,
        "scaled_mip": mip_scaled,
        "correlation_mean": corr_mean,
        "correlation_variance": corr_var,
        "best_global_index": best_gi,
    }
    if best_tag is not None:
        out["best_run_tag"] = best_tag
    return out


def merge_runs_independent_zscore(  # pylint: disable=too-many-locals
    runs: Sequence[Mapping[str, Any]],
    *,
    total_correlation_positions_per_run: int | Sequence[int],
) -> dict[str, torch.Tensor | int]:
    """Merge runs by per-run z-score, then take the best z at each pixel.

    For each run, mean and standard deviation are computed from that run's
    ``correlation_sum`` / ``correlation_squared_sum`` and its own
    ``total_correlation_positions`` (e.g. orientations in that sector). The z-score
    map is ``(mip - mean) / std`` per run (same convention as :func:`scale_mip`).
    The output ``scaled_mip`` is the **maximum** z across runs at each pixel; other
    fields are **gathered** from the winning run.

    **Total correlation count:** the returned ``total_correlation_positions`` is the
    **sum** of the per-run denominators. Use that (not a single sector's count) when
    applying multiplicity correction or peak-significance thresholds that assume one
    global search over **all** correlations (e.g. GEV), provided the runs partition
    the search without double-counting the same orientation.

    **Index semantics:** ``best_global_index`` is taken from the winning run. If each
    run used a **subset** of orientations (e.g. a sector), indices are usually **local**
    to that run's search — decode with the euler table for
    ``winner_run_index[h, w]``, or remap indices before calling
    :func:`decode_global_search_index`.

    Parameters
    ----------
    runs
        Same keys as :func:`merge_runs_pooled_zscore`.
    total_correlation_positions_per_run
        One int (shared by all runs) or a sequence of length ``len(runs)``.

    Returns
    -------
    dict
        ``mip`` (raw MIP from the winning run at each pixel), ``scaled_mip`` (best z),
        ``correlation_mean``, ``correlation_variance`` (from the winning run),
        ``best_global_index`` (from the winning run), ``winner_run_index`` (``int32``,
        which run index won each pixel), and ``total_correlation_positions`` (``int``,
        sum of per-run correlation counts — use for full-search peak thresholds).
    """
    if len(runs) == 0:
        raise ValueError("runs must be non-empty")

    if isinstance(total_correlation_positions_per_run, int):
        totals = [total_correlation_positions_per_run] * len(runs)
    else:
        totals = list(total_correlation_positions_per_run)
        if len(totals) != len(runs):
            raise ValueError(
                "total_correlation_positions_per_run must have length len(runs); "
                f"got {len(totals)} and {len(runs)} runs."
            )

    z_list: list[torch.Tensor] = []
    mean_list: list[torch.Tensor] = []
    std_list: list[torch.Tensor] = []
    mip_list: list[torch.Tensor] = []
    bgi_list: list[torch.Tensor] = []

    for run, n_pos in zip(runs, totals, strict=True):
        mip_k = _to_cpu_float32(run["mip"])
        csum = _to_cpu_float32(run["correlation_sum"])
        csq = _to_cpu_float32(run["correlation_squared_sum"])
        bgi_list.append(_to_cpu_int32(run["best_global_index"]))
        mip_list.append(mip_k)

        corr_mean, corr_std = correlation_sum_and_squared_sum_to_mean_and_variance(
            csum, csq, int(n_pos)
        )
        z = mip_k - corr_mean
        z = torch.where(
            corr_std != 0,
            z / corr_std,
            torch.zeros_like(z),
        )
        z_list.append(z)
        mean_list.append(corr_mean)
        std_list.append(corr_std)

    z_stack = torch.stack(z_list, dim=0)
    winner = z_stack.argmax(dim=0).to(torch.int64)

    mip_stack = torch.stack(mip_list, dim=0)
    mean_stack = torch.stack(mean_list, dim=0)
    std_stack = torch.stack(std_list, dim=0)
    bgi_stack = torch.stack(bgi_list, dim=0)

    w = winner.unsqueeze(0)
    mip_out = torch.gather(mip_stack, 0, w).squeeze(0)
    mip_scaled = torch.gather(z_stack, 0, w).squeeze(0)
    corr_mean_out = torch.gather(mean_stack, 0, w).squeeze(0)
    corr_var_out = torch.gather(std_stack, 0, w).squeeze(0)
    best_gi = torch.gather(bgi_stack, 0, w).squeeze(0)

    total_correlation_positions = sum(int(t) for t in totals)

    return {
        "mip": mip_out,
        "scaled_mip": mip_scaled,
        "correlation_mean": corr_mean_out,
        "correlation_variance": corr_var_out,
        "best_global_index": best_gi,
        "winner_run_index": winner.to(torch.int32),
        "total_correlation_positions": total_correlation_positions,
    }
