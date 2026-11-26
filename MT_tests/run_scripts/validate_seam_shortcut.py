"""Does the Stage 7 shortcut reproduce a seam that is known to be there?

Stage 7 places a PATCH at the sites a RING search already found, instead of running a
full match_template with the patch. That shortcut has never been checked. On the GDP
frame it returned a null, and a null from an unvalidated method says nothing.

The curved synthetic settles it. A full patch search on that frame reads the register at
2.19x chance over 42 sites, with the dimer control at 0.60x -- a real, known signal. The
same frame also has a ring search. So: take the ring's sites, run them through exactly
the Stage 7 shortcut, and see whether 2.19x comes back.

  reproduces it  -> the shortcut works, and the GDP null is a property of that specimen
  does not       -> the shortcut is broken, and the GDP null means nothing

Usage:
    python run_scripts/validate_seam_shortcut.py
"""

import pathlib
import sys
import time

import numpy as np
import pandas as pd
import yaml

from leopard_em.analysis.filament_lattice import (
    TemplateLatticeGeometry,
    extract_lattice_sites,
    image_offset_from_template_point,
)
from leopard_em.pydantic_models.managers import RefineTemplateManager
from leopard_em.pydantic_models.results.correlation_table import detections_from_hdf5

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from frame_geometry import (  # noqa: E402
    LEGACY_PIXEL_SIZE,
    legacy_frame_geometry,
    legacy_map,
)
from seam_sites_curved import alternation  # noqa: E402

ROLL_STEP_DEG = 4.0
RESULTS = "results_curved"
RING_TAG = "clean_truth_13pf_6dpv"
RING_STEM = "6dpu_atoms_6dpv_lattice_13pf_2rings_flatB"
PATCH_STEM = "6dpu_atoms_6dpv_lattice_13pf_patch5_2rings_flatB"
MICROGRAPH = "Frames/synthetic_curved_13pf_6dpv_sag100_clean.mrc"
KNOWN_RATIO = 2.19          # full patch search, 42 sites
KNOWN_CONTROL = 0.60

OPTICS = {
    "defocus_u": 8390.503906, "defocus_v": 8035.022461,
    "astigmatism_angle": -0.864719, "pixel_size": LEGACY_PIXEL_SIZE,
    "refined_pixel_size": LEGACY_PIXEL_SIZE, "voltage": 300.0,
    "spherical_aberration": 2.7, "amplitude_contrast_ratio": 0.07,
    "phase_shift": 0.0, "ctf_B_factor": 60.0,
}


def build_stack(clean, geometry, seeds, frame):
    """Stage 7's stack builder, verbatim in behaviour: offset applied per roll seed."""
    axis_point = np.array([*geometry.axis_offset_angstrom, 0.0], dtype=np.float64)
    rows = []
    for _, det in clean.iterrows():
        for seed in seeds:
            phi = (float(det["phi"]) + seed) % 360.0
            x, y = float(det["x"]), float(det["y"])
            if np.linalg.norm(axis_point) > 1e-9:
                offset = image_offset_from_template_point(
                    axis_point, np.array([phi]), np.array([float(det["theta"])]),
                    np.array([float(det["psi"])]),
                )[0]
                x -= offset[0] / frame.pixel_size
                y -= offset[1] / frame.pixel_size
            rows.append({
                "mip": det["correlation_value"], "scaled_mip": det["z_score"],
                "correlation_mean": det["correlation_mean"],
                "correlation_variance": det["correlation_variance"],
                "total_correlations": frame.correlations_per_pixel,
                "pos_x": int(round(x)), "pos_y": int(round(y)),
                "phi": phi, "theta": det["theta"], "psi": det["psi"],
                "relative_defocus": det["relative_defocus"],
            })
    stack = pd.DataFrame(rows)
    stack["particle_index"] = np.arange(len(stack))
    stack["pos_x_img"] = stack["pos_x"] + frame.half_width
    stack["pos_y_img"] = stack["pos_y"] + frame.half_width
    stack["pos_x_img_angstrom"] = stack["pos_x_img"] * frame.pixel_size
    stack["pos_y_img_angstrom"] = stack["pos_y_img"] * frame.pixel_size
    for key, value in OPTICS.items():
        stack[key] = value
    for key in ("even_zernikes", "odd_zernikes", "mag_matrix"):
        stack[key] = np.nan
    stack["micrograph_path"] = MICROGRAPH
    for key, name in (
        ("mip_path", "output_mip"), ("scaled_mip_path", "output_scaled_mip"),
        ("psi_path", "output_orientation_psi"),
        ("theta_path", "output_orientation_theta"),
        ("phi_path", "output_orientation_phi"),
        ("defocus_path", "output_relative_defocus"),
        ("correlation_average_path", "output_correlation_average"),
        ("correlation_variance_path", "output_correlation_variance"),
    ):
        stack[key] = f"{RESULTS}/{name}_{RING_TAG}.mrc"
    return stack


def run(stem, stack, tag, gpus=(0, 1, 2, 3)):
    """Refine one template over the stack and return its per-row score."""
    stack_path = MT_ROOT / RESULTS / f"validate_seam_stack_{tag}.csv"
    template = str(legacy_map(stem).relative_to(MT_ROOT))
    stack["template_path"] = template
    stack.to_csv(stack_path, index=True)
    config = {
        "template_volume_path": template,
        "particle_stack": {"df_path": str(stack_path.relative_to(MT_ROOT)),
                           "extracted_box_size": [640, 640],
                           "original_template_size": [600, 600]},
        "defocus_refinement_config": {"enabled": True, "defocus_min": -100.0,
                                      "defocus_max": 100.0, "defocus_step": 50.0},
        "pixel_size_refinement_config": {"enabled": False, "pixel_size_min": -0.01,
                                         "pixel_size_max": 0.01,
                                         "pixel_size_step": 0.01},
        "orientation_refinement_config": {
            "enabled": True, "phi_step_coarse": 2.5, "phi_step_fine": 0.4,
            "theta_step_coarse": 3.5, "theta_step_fine": 0.4,
            "psi_step_coarse": 2.5, "psi_step_fine": 0.4,
            "base_grid_method": "uniform"},
        "preprocessing_filters": {
            "whitening_filter": {"enabled": True, "do_power_spectrum": True,
                                 "max_freq": 1.0},
            "bandpass_filter": {"enabled": False}},
        "computational_config": {"gpu_ids": list(gpus), "num_cpus": 8},
        "movie_config": {"enabled": False, "movie_path": "",
                         "deformation_field_path": "", "pre_exposure": 0.0,
                         "fluence_per_frame": 1.0},
    }
    config_path = MT_ROOT / "configs" / f"validate_seam_{tag}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    out = MT_ROOT / RESULTS / f"validate_seam_refined_{tag}.csv"
    manager = RefineTemplateManager.from_yaml(str(config_path))
    start = time.time()
    manager.run_refine_template(str(out), 8)
    print(f"  {tag}: {len(stack)} rows in {time.time() - start:.0f}s")
    return pd.read_csv(out, index_col=0)["refined_scaled_mip"].to_numpy()


def main() -> None:
    """Run the Stage 7 shortcut on a frame whose seam is known, and compare."""
    ring_geometry = TemplateLatticeGeometry.from_pdb(
        str(MT_ROOT / "models" / f"{RING_STEM}.pdb"))
    patch_geometry = TemplateLatticeGeometry.from_pdb(
        str(MT_ROOT / "models" / f"{PATCH_STEM}.pdb"))

    table = MT_ROOT / RESULTS / f"output_correlation_table_{RING_TAG}.h5"
    detections = detections_from_hdf5(str(table), min_z_score=7.0).reset_index(drop=True)
    frame = legacy_frame_geometry(table, RING_STEM)
    print(frame.describe())
    sites = extract_lattice_sites(
        detections, ring_geometry, frame.pixel_size, bootstrap_score_threshold=8.0
    )
    clean = detections.iloc[np.asarray(sites.detection_index)].reset_index(drop=True)
    index = np.asarray(sites.axial_index).astype(int)
    print(f"ring sites: {len(clean)} ({sites.occupancy:.0%} of {sites.n_predicted}), "
          f"rise {sites.rise_angstrom:.3f} A")
    print(f"the full PATCH search on this frame gave {KNOWN_RATIO}x over 42 sites "
          f"(control {KNOWN_CONTROL}x)\n")

    half = 180.0 / ring_geometry.n_protofilaments
    seeds = np.linspace(-half, half,
                        max(int(np.ceil(2 * half / ROLL_STEP_DEG)), 1) + 1)[:-1]

    rng = np.random.default_rng(0)
    print(f"{'template':<8} {'n':>4} {'amp':>7} {'chance':>8} {'ratio':>7} "
          f"{'dimer ctrl':>11} {'mean z':>8}")
    results = {}
    for key, stem, geometry in (("patch", PATCH_STEM, patch_geometry),
                                ("ring", RING_STEM, ring_geometry)):
        stack = build_stack(clean, geometry, seeds, frame)
        z = run(stem, stack, key).reshape(len(clean), len(seeds))
        score = z.max(axis=1)
        amplitude, null = alternation(index, score, rng)
        ctrl_amp, ctrl_null = alternation(index // 2, score, rng)
        results[key] = amplitude / null
        print(f"{key:<8} {len(index):4d} {amplitude:7.3f} {null:8.3f} "
              f"{amplitude / null:6.2f}x {ctrl_amp / ctrl_null:10.2f}x "
              f"{score.mean():8.3f}")

    print()
    if results["patch"] >= 0.7 * KNOWN_RATIO:
        print(f"PASS: the shortcut recovers {results['patch']:.2f}x against the full "
              f"search's {KNOWN_RATIO}x.\n  The GDP null is therefore a property of that "
              f"specimen, not of the method.")
    else:
        print(f"FAIL: the shortcut gives {results['patch']:.2f}x where a full patch "
              f"search gives {KNOWN_RATIO}x on the SAME frame.\n  The shortcut loses the "
              f"seam, so the GDP null says nothing about the GDP specimen.\n  The seam "
              f"needs its own match_template run.")


if __name__ == "__main__":
    main()
