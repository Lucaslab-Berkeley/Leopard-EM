"""Characterization tests for the refine-template per-particle reducer.

These lock the behavior of ``_reduce_refine_best_zscore`` on deterministic synthetic
correlation batches so that backend refactors (e.g. moving z-score computation
out of the batch generator and into the reducer) provably do not change the
refined statistics. Runs on CPU; no GPU or downloaded data required.
"""

from unittest.mock import patch

import torch

from leopard_em.backend.core_refine_template import (
    _reduce_refine_best_zscore,
    core_refine_template,
)

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


def _minimal_refine_kwargs(num_particles: int = 2) -> dict:
    """Build minimal CPU tensors accepted by ``core_refine_template``."""
    img_h, img_w = 8, 5  # RFFT width
    tmpl_h, tmpl_w = 4, 3
    crop_h = img_h - tmpl_h + 1
    crop_w = (2 * (img_w - 1)) - (2 * (tmpl_w - 1)) + 1
    return {
        "particle_stack_dft": torch.randn(num_particles, img_h, img_w),
        "template_dft": torch.randn(4, tmpl_h, tmpl_w),
        "euler_angles": torch.zeros(num_particles, 3),
        "euler_angle_offsets": torch.zeros(1, 3),
        "defocus_offsets": torch.zeros(1),
        "defocus_u": torch.zeros(num_particles),
        "defocus_v": torch.zeros(num_particles),
        "defocus_angle": torch.zeros(num_particles),
        "pixel_size_offsets": torch.zeros(1),
        "corr_mean": torch.zeros(num_particles, crop_h, crop_w),
        "corr_std": torch.ones(num_particles, crop_h, crop_w),
        "ctf_kwargs": {},
        "projective_filters": torch.ones(num_particles, tmpl_h, tmpl_w),
        "batch_size": 1,
    }


def _fake_single_gpu_result(num_particles: int = 2) -> dict:
    """Numpy payload matching ``_core_refine_template_single_gpu`` output."""
    return {
        "refined_cross_correlation": torch.zeros(num_particles).numpy(),
        "refined_z_score": torch.ones(num_particles).numpy(),
        "refined_euler_angles": torch.zeros(num_particles, 3).numpy(),
        "refined_defocus_offset": torch.zeros(num_particles).numpy(),
        "refined_pixel_size_offset": torch.zeros(num_particles).numpy(),
        "refined_pos_y": torch.zeros(num_particles).numpy(),
        "refined_pos_x": torch.zeros(num_particles).numpy(),
        "particle_indices": torch.arange(num_particles).numpy(),
        "angle_idx": torch.zeros(num_particles, dtype=torch.long).numpy(),
    }


def test_core_refine_template_single_device_runs_in_process():
    """Single-GPU refine must not spawn a child process (avoids CUDA IPC errors)."""
    num_particles = 2
    kwargs = _minimal_refine_kwargs(num_particles)
    fake = _fake_single_gpu_result(num_particles)

    def _in_process(result_dict, device_id, **_kwargs):
        result_dict[device_id] = fake

    with (
        patch(
            "leopard_em.backend.core_refine_template.run_multiprocess_jobs"
        ) as mock_mp,
        patch(
            "leopard_em.backend.core_refine_template._core_refine_template_single_gpu",
            side_effect=_in_process,
        ) as mock_single,
    ):
        result = core_refine_template(device=torch.device("cpu"), **kwargs)

    mock_mp.assert_not_called()
    mock_single.assert_called_once()
    assert result["refined_z_score"].shape == (num_particles,)


def test_core_refine_template_multi_device_uses_multiprocess():
    """Multi-GPU refine still dispatches through ``run_multiprocess_jobs``."""
    num_particles = 2
    kwargs = _minimal_refine_kwargs(num_particles)
    fake = _fake_single_gpu_result(num_particles)

    with (
        patch(
            "leopard_em.backend.core_refine_template.run_multiprocess_jobs",
            return_value={0: fake},
        ) as mock_mp,
        patch(
            "leopard_em.backend.core_refine_template._core_refine_template_single_gpu"
        ) as mock_single,
    ):
        result = core_refine_template(
            device=[torch.device("cpu"), torch.device("cpu")],
            **kwargs,
        )

    mock_mp.assert_called_once()
    mock_single.assert_not_called()
    assert result["refined_z_score"].shape == (num_particles,)


if __name__ == "__main__":
    test_reduce_refine_best_zscore_matches_snapshot()
    test_core_refine_template_single_device_runs_in_process()
    test_core_refine_template_multi_device_uses_multiprocess()
