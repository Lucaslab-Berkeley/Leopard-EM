"""Characterization tests for the refine-template per-particle reducer.

These lock the behavior of ``_reduce_refine_best_zscore`` on deterministic synthetic
correlation batches so that backend refactors (e.g. moving z-score computation
out of the batch generator and into the reducer) provably do not change the
refined statistics. Runs on CPU; no GPU or downloaded data required.
"""

import torch

from leopard_em.backend.core_refine_template import _reduce_refine_best_zscore

# Shapes for the synthetic search space.
N_PX, N_DEF, CROP_H, CROP_W = 2, 3, 4, 5
BATCH_SIZES = (3, 2)  # two orientation batches, 5 offsets total


def _build_synthetic_batches() -> tuple[list[tuple], torch.Tensor, torch.Tensor]:
    """Build deterministic correlation batches in the post-refactor tuple format.

    Returns the batch list ``(start_idx, angle_offsets, cross_correlation,
    crop_h, crop_w)`` (no pre-computed z-score) plus the ``corr_mean`` and
    ``corr_std`` maps. The RNG draw order is fixed so the result is reproducible.
    """
    torch.manual_seed(0)
    corr_mean = torch.randn(CROP_H, CROP_W)
    corr_std = torch.rand(CROP_H, CROP_W) + 0.5  # strictly positive
    euler_offsets = torch.randn(sum(BATCH_SIZES), 3)

    batches = []
    start = 0
    for batch_size in BATCH_SIZES:
        cross_correlation = torch.randn(N_PX, N_DEF, batch_size, CROP_H, CROP_W)
        batches.append(
            (
                start,
                euler_offsets[start : start + batch_size],
                cross_correlation,
                CROP_H,
                CROP_W,
            )
        )
        start += batch_size

    return batches, corr_mean, corr_std


def test_reduce_refine_best_zscore_matches_snapshot():
    """The reducer reproduces the pre-refactor refined statistics exactly.

    Expected values were captured from the original implementation (which
    pre-computed z-score inside the batch generator) on the same synthetic data.
    """
    batches, corr_mean, corr_std = _build_synthetic_batches()
    defocus_offsets = torch.tensor([-10.0, 0.0, 10.0])  # len N_DEF
    pixel_size_offsets = torch.tensor([-0.01, 0.01])  # len N_PX

    result = _reduce_refine_best_zscore(
        iter(batches),
        corr_mean=corr_mean,
        corr_std=corr_std,
        defocus_offsets=defocus_offsets,
        pixel_size_offsets=pixel_size_offsets,
    )

    def _val(key: str) -> float:
        value = result[key]
        return value.item() if torch.is_tensor(value) else value

    assert _val("max_cc") == 3.4028263092041016
    assert _val("max_z_score") == 5.6347527503967285
    assert _val("refined_phi_offset") == -2.2187793254852295
    assert _val("refined_theta_offset") == 0.2589845359325409
    assert _val("refined_psi_offset") == -1.0297021865844727
    assert _val("refined_defocus_offset") == 0.0
    assert _val("refined_pixel_size_offset") == 0.009999999776482582
    assert _val("refined_pos_y") == 1
    assert _val("refined_pos_x") == 0
    assert _val("angle_idx") == 2


if __name__ == "__main__":
    test_reduce_refine_best_zscore_matches_snapshot()
