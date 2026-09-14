"""Refine the orientation of the clean lattice sites, to get phi off the search grid.

The supertwist measurement is limited by angular sampling, not by counts: the search grid
quantises phi at 3.495 deg, and within one protofilament the 82 clean sites take only one
or two distinct values, so a straight-line fit reads out where a grid step happens to
fall rather than a drift. refine_template searches phi off-grid at 0.4 deg, which is the
one thing that would make the drift measurable.

Writes a particle stack for the clean sites, then refines it.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from leopard_em.analysis.filament_lattice import (
    TemplateLatticeGeometry,
    extract_lattice_sites,
)
from leopard_em.pydantic_models.managers import RefineTemplateManager
from leopard_em.pydantic_models.results.correlation_table import detections_from_hdf5

HERE = Path(__file__).resolve().parent.parent
PIXEL_SIZE = 0.9194
TEMPLATE_HALF_WIDTH = 300  # maps are 600^3, and pos_*_img is the map frame plus this

TABLE = "results_full/output_correlation_table_patch4_full.h5"
GEOMETRY = "models/6dpu_4_patches_aligned_zero.pdb"
TEMPLATE = "maps/GMPCPP_4patches_0.9194_bscale0.5.mrc"
MICROGRAPH = "Frames/2025-05-23_14.39.26_25May23_GMPCPP1_26-94_0005_X-1Y+1-0_sum_DW.mrc"

SEARCH_OUTPUTS = {
    "mip_path": "results_full/output_mip_patch4_full.mrc",
    "scaled_mip_path": "results_full/output_scaled_mip_patch4_full.mrc",
    "psi_path": "results_full/output_orientation_psi_patch4_full.mrc",
    "theta_path": "results_full/output_orientation_theta_patch4_full.mrc",
    "phi_path": "results_full/output_orientation_phi_patch4_full.mrc",
    "defocus_path": "results_full/output_relative_defocus_patch4_full.mrc",
    "correlation_average_path": "results_full/output_correlation_average_patch4_full.mrc",
    "correlation_variance_path": (
        "results_full/output_correlation_variance_patch4_full.mrc"
    ),
}

OPTICS = {
    "defocus_u": 8390.503906,
    "defocus_v": 8035.022461,
    "astigmatism_angle": -0.864719,
    "pixel_size": PIXEL_SIZE,
    "refined_pixel_size": PIXEL_SIZE,
    "voltage": 300.0,
    "spherical_aberration": 2.7,
    "amplitude_contrast_ratio": 0.07,
    "phase_shift": 0.0,
    "ctf_B_factor": 60.0,
}


def clean_sites(min_z: float, bootstrap: float) -> pd.DataFrame:
    """Detections occupying the sites of the fitted lattice."""
    geometry = TemplateLatticeGeometry.from_pdb(str(HERE / GEOMETRY))
    detections = detections_from_hdf5(
        str(HERE / TABLE), min_z_score=min_z
    ).reset_index(drop=True)
    sites = extract_lattice_sites(
        detections, geometry, PIXEL_SIZE, bootstrap_score_threshold=bootstrap
    )
    clean = detections.iloc[np.asarray(sites.detection_index)].reset_index(drop=True)
    print(
        f"{len(detections):,} detections at z > {min_z} -> {len(clean)} clean sites "
        f"({sites.occupancy:.0%} of {sites.n_predicted}), rise {sites.rise_angstrom:.3f} A"
    )
    return clean


def particle_stack(clean: pd.DataFrame) -> pd.DataFrame:
    """Particle stack in the layout RefineTemplateManager expects."""
    stack = pd.DataFrame(
        {
            "particle_index": np.arange(len(clean)),
            "mip": clean["correlation_value"].to_numpy(),
            "scaled_mip": clean["z_score"].to_numpy(),
            "correlation_mean": clean["correlation_mean"].to_numpy(),
            "correlation_variance": clean["correlation_variance"].to_numpy(),
            "total_correlations": 0,
            "pos_x": clean["x"].astype(int).to_numpy(),
            "pos_y": clean["y"].astype(int).to_numpy(),
            "phi": clean["phi"].to_numpy(),
            "theta": clean["theta"].to_numpy(),
            "psi": clean["psi"].to_numpy(),
            "relative_defocus": clean["relative_defocus"].to_numpy(),
        }
    )
    stack["pos_x_img"] = stack["pos_x"] + TEMPLATE_HALF_WIDTH
    stack["pos_y_img"] = stack["pos_y"] + TEMPLATE_HALF_WIDTH
    stack["pos_x_img_angstrom"] = stack["pos_x_img"] * PIXEL_SIZE
    stack["pos_y_img_angstrom"] = stack["pos_y_img"] * PIXEL_SIZE
    for key, value in OPTICS.items():
        stack[key] = value
    for key in ("even_zernikes", "odd_zernikes", "mag_matrix"):
        stack[key] = np.nan
    stack["micrograph_path"] = MICROGRAPH
    stack["template_path"] = TEMPLATE
    # The backend re-reads the correlation mean and variance maps to rescale each
    # extracted box, so these must point at the search's own outputs, not be blank.
    for key, path in SEARCH_OUTPUTS.items():
        stack[key] = path
    return stack


def write_config(stack_path: Path, config_path: Path, gpus: list[int]) -> None:
    """Refinement config: orientation off-grid, defocus and pixel size held fixed."""
    config = {
        "template_volume_path": TEMPLATE,
        "particle_stack": {
            "df_path": str(stack_path.relative_to(HERE)),
            "extracted_box_size": [640, 640],
            "original_template_size": [600, 600],
        },
        "defocus_refinement_config": {
            "enabled": True,
            "defocus_min": -200.0,
            "defocus_max": 200.0,
            "defocus_step": 50.0,
        },
        "pixel_size_refinement_config": {
            "enabled": False,
            "pixel_size_min": -0.01,
            "pixel_size_max": 0.01,
            "pixel_size_step": 0.01,
        },
        "orientation_refinement_config": {
            "enabled": True,
            "phi_step_coarse": 2.0,
            "phi_step_fine": 0.4,
            "theta_step_coarse": 2.0,
            "theta_step_fine": 0.4,
            "psi_step_coarse": 2.0,
            "psi_step_fine": 0.4,
            "base_grid_method": "uniform",
        },
        "preprocessing_filters": {
            "whitening_filter": {
                "enabled": True,
                "do_power_spectrum": True,
                "max_freq": 1.0,
            },
            "bandpass_filter": {"enabled": False},
        },
        "computational_config": {"gpu_ids": gpus, "num_cpus": 8},
        "movie_config": {
            "enabled": False,
            "movie_path": "",
            "deformation_field_path": "",
            "pre_exposure": 0.0,
            "fluence_per_frame": 1.0,
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))


def main() -> None:
    """Build the clean-site stack and refine it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-z", type=float, default=7.0)
    parser.add_argument("--bootstrap", type=float, default=9.0)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tag", default="clean_sites_patch4")
    args = parser.parse_args()

    stack_path = HERE / "results_full" / f"stack_{args.tag}.csv"
    config_path = HERE / "configs" / f"refine_{args.tag}.yaml"
    output_path = HERE / "results_full" / f"refined_{args.tag}.csv"

    stack = particle_stack(clean_sites(args.min_z, args.bootstrap))
    stack.to_csv(stack_path, index=False)
    print(f"wrote {stack_path} ({len(stack)} particles)")

    write_config(stack_path, config_path, args.gpus)
    print(f"wrote {config_path}")

    manager = RefineTemplateManager.from_yaml(str(config_path))
    offsets = manager.orientation_refinement_config.euler_angles_offsets
    defocus = manager.defocus_refinement_config.defocus_values
    print(f"orientation offsets {tuple(offsets.shape)}, defocus {tuple(defocus.shape)}")
    manager.run_refine_template(str(output_path), args.batch_size)
    print(f"DONE -> {output_path}")


if __name__ == "__main__":
    main()
