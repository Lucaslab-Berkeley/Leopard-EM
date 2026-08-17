"""Round-trip tests for saving/loading self-describing inspection ``.npz`` files."""

import numpy as np
import torch

from leopard_em.analysis.inspect_peaks_result import (
    CROSS_CORRELATION_AXES,
    CROSS_CORRELATION_FRAME_AXES,
    FRC_AXES,
    FRC_FRAME_AXES,
    load_inspection_result,
    save_inspection_result,
)


def _common_kwargs(n_particles: int = 2, n_orient: int = 3):
    """Build the shared base orientation/defocus/offset arrays for a tiny run."""
    return {
        "euler_angle_offsets": torch.zeros((n_orient, 3)),
        "defocus_offsets": torch.tensor([-100.0, 0.0, 100.0]),
        "pixel_size_offsets": torch.tensor([0.0]),
        "base_euler_angles": torch.arange(n_particles * 3, dtype=torch.float32).reshape(
            n_particles, 3
        ),
        "base_defocus": torch.tensor([[1.0, 2.0, 0.5], [3.0, 4.0, 1.5]]),
    }


def test_save_load_cross_correlation_roundtrip(tmp_path):
    """Spatial CC result round-trips with base orientation + astigmatic defocus."""
    # (N=2, n_px=1, n_def=3, n_orient=3, H=2, W=2)
    scores = torch.randn((2, 1, 3, 3, 2, 2))
    kwargs = _common_kwargs()

    path = save_inspection_result(
        tmp_path / "spatial",
        result=scores,
        output_mode="cross_correlation",
        particle_index=torch.tensor([10, 11]),
        **kwargs,
    )
    assert path.suffix == ".npz"

    result = load_inspection_result(path)
    assert result.output_mode == "cross_correlation"
    assert result.axes == CROSS_CORRELATION_AXES
    assert result.metadata["per_frame"] is False
    assert result.scores.shape == (2, 1, 3, 3, 2, 2)
    np.testing.assert_allclose(result.scores, scores.numpy())
    assert result.base_defocus.shape == (2, 3)
    np.testing.assert_allclose(result.base_defocus, kwargs["base_defocus"].numpy())
    np.testing.assert_allclose(
        result.base_euler_angles, kwargs["base_euler_angles"].numpy()
    )
    np.testing.assert_array_equal(result.particle_index, np.array([10, 11]))
    assert result.frequency_bins is None
    assert result.frame_index is None


def test_save_load_per_frame_cross_correlation_inserts_frame_axis(tmp_path):
    """Per-frame CC result carries a ``frame`` axis and a stored frame index."""
    # (N=2, T=4, n_px=1, n_def=3, n_orient=3, H=2, W=2)
    scores = torch.randn((2, 4, 1, 3, 3, 2, 2))
    kwargs = _common_kwargs()

    path = save_inspection_result(
        tmp_path / "per_frame",
        result=scores,
        output_mode="cross_correlation",
        per_frame=True,
        frame_index=torch.arange(4),
        **kwargs,
    )

    result = load_inspection_result(path)
    assert result.axes == CROSS_CORRELATION_FRAME_AXES
    assert result.axes[:2] == ("particle", "frame")
    assert result.metadata["per_frame"] is True
    assert result.scores.shape == (2, 4, 1, 3, 3, 2, 2)
    np.testing.assert_array_equal(result.frame_index, np.arange(4))


def test_save_load_frc_modes(tmp_path):
    """FRC results round-trip with frequency bins in spatial and per-frame layouts."""
    freq_bins = torch.linspace(0.0, 0.5, 5)
    kwargs = _common_kwargs()

    # Spatial FRC: (N, n_px, n_def, n_orient, n_freq)
    spatial = torch.randn((2, 1, 3, 3, 5))
    spatial_path = save_inspection_result(
        tmp_path / "frc_spatial",
        result=(spatial, freq_bins),
        output_mode="frc",
        **kwargs,
    )
    spatial_result = load_inspection_result(spatial_path)
    assert spatial_result.axes == FRC_AXES
    np.testing.assert_allclose(spatial_result.frequency_bins, freq_bins.numpy())

    # Per-frame FRC: (N, T, n_px, n_def, n_orient, n_freq)
    per_frame = torch.randn((2, 4, 1, 3, 3, 5))
    per_frame_path = save_inspection_result(
        tmp_path / "frc_per_frame",
        result=(per_frame, freq_bins),
        output_mode="frc",
        per_frame=True,
        frame_index=torch.arange(4),
        **kwargs,
    )
    per_frame_result = load_inspection_result(per_frame_path)
    assert per_frame_result.axes == FRC_FRAME_AXES
    assert per_frame_result.scores.shape == (2, 4, 1, 3, 3, 5)
    np.testing.assert_allclose(per_frame_result.frequency_bins, freq_bins.numpy())
