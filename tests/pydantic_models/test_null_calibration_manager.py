"""Tests for NullCalibrationManager (CPU; the GPU search is replaced by a stand-in)."""

import json

import mrcfile
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from scipy.ndimage import gaussian_filter

from leopard_em.analysis import NullPeaks
from leopard_em.pydantic_models.managers import (
    MatchTemplateManager,
    NullCalibrationManager,
)
from leopard_em.utils.phase_randomization import phase_randomize_image

IMAGE = 96
TEMPLATE = 16
VALID = IMAGE - TEMPLATE + 1


def _write_mrc(path, data):
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(data, dtype=np.float32))


@pytest.fixture(name="config_path")
def fixture_config_path(tmp_path):
    rng = np.random.default_rng(0)
    _write_mrc(tmp_path / "micrograph.mrc", rng.normal(size=(IMAGE, IMAGE)))
    _write_mrc(
        tmp_path / "template.mrc",
        gaussian_filter(rng.normal(size=(TEMPLATE,) * 3), sigma=1.0),
    )
    config = {
        "template_volume_path": str(tmp_path / "template.mrc"),
        "micrograph_path": str(tmp_path / "micrograph.mrc"),
        "match_template_result": {
            "allow_file_overwrite": True,
            **{
                f"{name}_path": str(tmp_path / f"real_{name}.mrc")
                for name in (
                    "mip",
                    "scaled_mip",
                    "orientation_psi",
                    "orientation_theta",
                    "orientation_phi",
                    "relative_defocus",
                    "correlation_average",
                    "correlation_variance",
                )
            },
        },
        "optics_group": {
            "label": "test",
            "voltage": 300.0,
            "pixel_size": 1.5,
            "defocus_u": 8000.0,
            "defocus_v": 8000.0,
            "astigmatism_angle": 0.0,
            "spherical_aberration": 2.7,
            "amplitude_contrast_ratio": 0.07,
        },
        "defocus_search_config": {"enabled": False},
        "orientation_search_config": {"psi_step": 30.0, "theta_step": 30.0},
        "preprocessing_filters": {"whitening_filter": {"enabled": True}},
        "fast_fft_padding": {"enabled": False},
        "computational_config": {"gpu_ids": [0], "num_cpus": 1},
    }
    path = tmp_path / "match_template.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _fake_run(self, orientation_batch_size=16, do_result_export=True, **_):
    """Stand-in search: a smoothed, standardised copy of the injected micrograph."""
    del orientation_batch_size, do_result_export
    image = self.micrograph.numpy()
    z = gaussian_filter(image, 1.5)[:VALID, :VALID]
    self.match_template_result.scaled_mip = torch.from_numpy(
        ((z - z.mean()) / z.std()).astype(np.float32)
    )


def test_phase_randomized_null_uses_each_seed(monkeypatch, tmp_path, config_path):
    injected = []

    def record(self, *args, **kwargs):
        injected.append(self.micrograph.clone())
        _fake_run(self, *args, **kwargs)

    monkeypatch.setattr(MatchTemplateManager, "run_match_template", record)
    manager = NullCalibrationManager(
        match_template_config_path=str(config_path),
        output_directory=str(tmp_path / "null"),
        seeds=[3, 4, 5],
        peak_floor=1.5,
        mask_radius=4.0,
        border=2,
    )
    null_peaks = manager.run_phase_randomized_null()
    image = torch.from_numpy(
        mrcfile.read(tmp_path / "micrograph.mrc").astype(np.float32)
    )
    for seed, copy in zip([3, 4, 5], injected, strict=False):
        assert torch.equal(copy, phase_randomize_image(image, seed))
    assert isinstance(null_peaks, NullPeaks)
    assert null_peaks.seeds == [3, 4, 5]
    assert all(z.min() >= 1.5 for z in null_peaks.zscores)
    assert null_peaks.valid_areas == [(VALID - 4) ** 2] * 3
    assert manager.valid_area() == (VALID - 4) ** 2


def test_run_writes_summary(monkeypatch, tmp_path, config_path):
    monkeypatch.setattr(MatchTemplateManager, "run_match_template", _fake_run)
    manager = NullCalibrationManager(
        match_template_config_path=str(config_path),
        output_directory=str(tmp_path / "null"),
        seeds=[1, 2],
        false_positives=[1.0, 5.0],
        peak_floor=1.0,
        mask_radius=4.0,
        analytic={"n_mean_samples": 40, "n_kappa_pairs": 4, "n_lattice_samples": 8},
    )
    manager.run()
    summary = json.loads((tmp_path / "null" / "null_summary.json").read_text())
    assert summary["valid_area"] == VALID**2
    rows = {row["false_positives"]: row for row in summary["thresholds"]}
    assert set(rows) == {1.0, 5.0}
    for row in rows.values():
        assert np.isfinite(row["nominal"])
        assert np.isfinite(row["phase_randomized"])
        assert np.isfinite(row["analytic"])
    assert rows[5.0]["phase_randomized"] < rows[1.0]["phase_randomized"]
    assert rows[5.0]["nominal"] < rows[1.0]["nominal"]
    again = NullPeaks.from_dict(summary["null_peaks"])
    assert again.zscore_cutoff(1.0) == rows[1.0]["phase_randomized"]
    assert summary["analytic"]["details"]["planes"] == 1


def test_run_writes_refined_thresholds(monkeypatch, tmp_path, config_path):
    monkeypatch.setattr(MatchTemplateManager, "run_match_template", _fake_run)
    refine_path = tmp_path / "refine.yaml"
    refine_path.write_text(
        yaml.safe_dump(
            {
                "orientation_refinement_config": {
                    "enabled": True,
                    "psi_step_coarse": 30.0,
                    "psi_step_fine": 15.0,
                    "theta_step_coarse": 30.0,
                    "theta_step_fine": 15.0,
                },
                "defocus_refinement_config": {"enabled": False},
            }
        )
    )
    with pytest.raises(ValueError, match="go together"):
        NullCalibrationManager(
            match_template_config_path=str(config_path),
            output_directory=str(tmp_path / "null"),
            refine_template_config_path=str(refine_path),
        )
    manager = NullCalibrationManager(
        match_template_config_path=str(config_path),
        output_directory=str(tmp_path / "null"),
        seeds=[],
        mask_radius=4.0,
        analytic={
            "n_mean_samples": 40,
            "n_kappa_pairs": 4,
            "n_lattice_samples": 8,
            "refine": {"n_candidates": 2, "n_draws": 2000},
        },
        refine_template_config_path=str(refine_path),
        refine_candidate_threshold=3.0,
    )
    manager.run()
    summary = json.loads((tmp_path / "null" / "null_summary.json").read_text())
    row = summary["thresholds"][0]
    assert row["analytic_refined"] > row["analytic"]
    refined = summary["analytic_refined"]
    assert refined["candidate_threshold"] == 3.0
    assert refined["gain"]["n_windows"] == 2
    assert refined["details"]["defocus_offsets"] == [0.0]


def test_floor_too_high_reports_none(monkeypatch, tmp_path, config_path):
    monkeypatch.setattr(MatchTemplateManager, "run_match_template", _fake_run)
    manager = NullCalibrationManager(
        match_template_config_path=str(config_path),
        output_directory=str(tmp_path / "null"),
        seeds=[1],
        false_positives=[1000.0],
        peak_floor=3.5,
        analytic={"enabled": False},
    )
    summary = manager.run()
    assert summary["thresholds"][0]["phase_randomized"] is None
    assert "analytic" not in summary["thresholds"][0]


def test_null_particles_are_the_counted_peaks(monkeypatch, tmp_path, config_path):
    def run_with_maps(self, *args, **kwargs):
        _fake_run(self, *args, **kwargs)
        result = self.match_template_result
        rng = np.random.default_rng(1)
        for name in (
            "mip",
            "orientation_psi",
            "orientation_theta",
            "orientation_phi",
            "relative_defocus",
            "correlation_average",
            "correlation_variance",
        ):
            values = rng.normal(size=(VALID, VALID)).astype(np.float32)
            setattr(result, name, torch.from_numpy(values))
        result.total_projections = 7

    monkeypatch.setattr(MatchTemplateManager, "run_match_template", run_with_maps)
    manager = NullCalibrationManager(
        match_template_config_path=str(config_path),
        output_directory=str(tmp_path / "null"),
        seeds=[3],
        peak_floor=1.5,
        mask_radius=4.0,
        border=2,
        write_null_particles=True,
    )
    null_peaks = manager.run_phase_randomized_null()
    df = pd.read_csv(tmp_path / "null" / "null_seed3_particles.csv")
    assert len(df) == len(null_peaks.zscores[0])
    np.testing.assert_allclose(df["scaled_mip"], null_peaks.zscores[0], rtol=1e-6)
    assert (df["total_correlations"] == 7).all()
    assert (df["pos_y_img"] - df["pos_y"] == TEMPLATE // 2).all()
