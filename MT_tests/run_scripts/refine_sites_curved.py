"""Stage 5 of PIPELINE.md: refine the EXTRACTED LATTICE SITES, not the peak list.

The sites are what every later measurement is about. They come from the top-down pass
over the correlation table, so they include weak detections that the thresholded peak
list discarded -- on this frame, 27,544 table detections against 288 peaks. Refining the
peak list instead would sharpen the wrong objects.

Refinement is what gets phi off the search grid, and phi is the azimuth about the tube:
the protofilament harmonic, the helical unwrap and the supertwist test all read it
directly. Quantised phi previously faked a supertwist slope from 19 distinct values
across 82 sites.

Sites are extracted per segment, exactly as the spacing analysis does it, because a bent
tube mis-indexes sites when measured over its whole length -- 110 A of axial distortion
on region 1, which is 2.7 monomers. The winners are then pooled into one stack and
refined together.

Usage:
    python run_scripts/refine_sites_curved.py --tag gdp_curved_13pf
"""

import argparse
import pathlib
import sys
import time

import h5py
import numpy as np
import pandas as pd
import yaml

from leopard_em.analysis.filament_lattice import (
    TemplateLatticeGeometry,
    axis_points_from_peaks,
    extract_lattice_sites,
)
from leopard_em.pydantic_models.managers import RefineTemplateManager
from leopard_em.pydantic_models.results.correlation_table import detections_from_hdf5

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyse_gdp_curved as A  # noqa: E402
from frame_geometry import (  # noqa: E402
    legacy_frame_geometry,
    legacy_map,
    template_half_width,
)

EXTRACTION_MARGIN_PX = 40  # extracted box = template box + this (640 for a 600^3 map)


def collect_sites(detections, geometry, drawn, region_map, frame, segments=None):
    """Winning detections at every predicted lattice site, segment by segment."""
    half = frame.half_width
    y = np.clip(detections["y"].to_numpy() + half, 0, region_map.shape[0] - 1)
    x = np.clip(detections["x"].to_numpy() + half, 0, region_map.shape[1] - 1)
    owner = region_map[y, x]

    keep_index, provenance = [], []
    for region in sorted(drawn):
        mask = owner == region
        if mask.sum() < 100:
            continue
        subset = detections[mask]
        original = np.flatnonzero(mask)
        curve = A.DrawnCurve(drawn[region], pixel_size=frame.pixel_size)
        k = segments or A.segment_count(curve)
        points = axis_points_from_peaks(
            subset.reset_index(drop=True), geometry, frame.pixel_size
        ) + half
        arc, _ = curve.project(points)
        edges = np.linspace(0.0, curve.length_px, k + 1)

        for part in range(k):
            inside = (arc >= edges[part]) & (arc <= edges[part + 1])
            if inside.sum() < 200:
                continue
            piece = subset[inside].reset_index(drop=True)
            boot = A.pick_bootstrap(piece["z_score"].to_numpy())
            if boot is None:
                continue
            try:
                sites = extract_lattice_sites(
                    piece, geometry, frame.pixel_size, bootstrap_score_threshold=boot
                )
            except ValueError:
                continue
            chosen = original[inside][np.asarray(sites.detection_index)]
            keep_index.extend(chosen.tolist())
            provenance.extend([(region, part + 1, k, boot)] * len(chosen))
            print(f"  region {region} segment {part + 1}/{k}: "
                  f"{len(chosen):3d} sites, {sites.occupancy:4.0%} occupied, "
                  f"rise {sites.rise_angstrom:7.3f} A, bootstrap z {boot:.1f}")

    order = np.argsort(keep_index)
    keep_index = np.asarray(keep_index)[order]
    provenance = [provenance[i] for i in order]
    clean = detections.iloc[keep_index].reset_index(drop=True)
    clean["region"] = [p[0] for p in provenance]
    clean["segment"] = [p[1] for p in provenance]
    return clean


def particle_stack(clean, optics, micrograph, template, results, tag, frame):
    """Particle stack in the layout RefineTemplateManager expects.

    ``frame`` (``frame_geometry`` of the table the sites came from) supplies the offset
    to micrograph coordinates, the pixel size, and ``total_correlations``.
    """
    stack = pd.DataFrame({
        "particle_index": np.arange(len(clean)),
        "mip": clean["correlation_value"].to_numpy(),
        "scaled_mip": clean["z_score"].to_numpy(),
        "correlation_mean": clean["correlation_mean"].to_numpy(),
        "correlation_variance": clean["correlation_variance"].to_numpy(),
        "total_correlations": frame.correlations_per_pixel,
        "pos_x": clean["x"].astype(int).to_numpy(),
        "pos_y": clean["y"].astype(int).to_numpy(),
        "phi": clean["phi"].to_numpy(),
        "theta": clean["theta"].to_numpy(),
        "psi": clean["psi"].to_numpy(),
        "relative_defocus": clean["relative_defocus"].to_numpy(),
    })
    stack["pos_x_img"] = stack["pos_x"] + frame.half_width
    stack["pos_y_img"] = stack["pos_y"] + frame.half_width
    stack["pos_x_img_angstrom"] = stack["pos_x_img"] * frame.pixel_size
    stack["pos_y_img_angstrom"] = stack["pos_y_img"] * frame.pixel_size
    for key, value in optics.items():
        stack[key] = value
    stack["refined_pixel_size"] = optics["pixel_size"]
    for key in ("even_zernikes", "odd_zernikes", "mag_matrix"):
        stack[key] = np.nan
    stack["micrograph_path"] = micrograph
    stack["template_path"] = template
    # The backend re-reads the correlation mean and variance maps to rescale each
    # extracted box, so these must point at the search's own outputs.
    for key, name in (
        ("mip_path", "output_mip"), ("scaled_mip_path", "output_scaled_mip"),
        ("psi_path", "output_orientation_psi"),
        ("theta_path", "output_orientation_theta"),
        ("phi_path", "output_orientation_phi"),
        ("defocus_path", "output_relative_defocus"),
        ("correlation_average_path", "output_correlation_average"),
        ("correlation_variance_path", "output_correlation_variance"),
    ):
        stack[key] = f"{results}/{name}_{tag}.mrc"
    stack["region"] = clean["region"].to_numpy()
    stack["segment"] = clean["segment"].to_numpy()
    return stack


def write_config(stack_path, config_path, template, search_config, gpus):
    """Refine config whose coarse steps are the steps the search actually used."""
    with open(search_config, encoding="utf-8") as handle:
        search = yaml.safe_load(handle)
    psi_step = float(search["orientation_search_config"]["psi_step"])
    theta_step = float(search["orientation_search_config"]["theta_step"])
    defocus_step = float(search["defocus_search_config"]["defocus_step"])
    print(f"coarse steps from the search: psi {psi_step}, theta {theta_step}, "
          f"defocus {defocus_step}")
    # Box sizes from the template itself, not constants: 600 -> 640 in vitro, 528 -> 568
    # for the in situ map.
    box = 2 * template_half_width(MT_ROOT / template)
    extracted = box + EXTRACTION_MARGIN_PX

    config = {
        "template_volume_path": template,
        "particle_stack": {
            "df_path": str(stack_path.relative_to(MT_ROOT)),
            "extracted_box_size": [extracted, extracted],
            "original_template_size": [box, box],
        },
        "defocus_refinement_config": {
            "enabled": True,
            "defocus_min": -defocus_step / 2.0,
            "defocus_max": defocus_step / 2.0,
            "defocus_step": defocus_step / 8.0,
        },
        "pixel_size_refinement_config": {
            "enabled": False, "pixel_size_min": -0.01,
            "pixel_size_max": 0.01, "pixel_size_step": 0.01,
        },
        "orientation_refinement_config": {
            "enabled": True,
            "phi_step_coarse": psi_step,   # inert in the uniform branch; see PIPELINE.md
            "phi_step_fine": 0.4,
            "theta_step_coarse": theta_step,
            "theta_step_fine": 0.4,
            "psi_step_coarse": psi_step,
            "psi_step_fine": 0.4,
            "base_grid_method": "uniform",
        },
        "preprocessing_filters": {
            "whitening_filter": {"enabled": True, "do_power_spectrum": True,
                                 "max_freq": 1.0},
            "bandpass_filter": {"enabled": False},
        },
        "computational_config": {"gpu_ids": gpus, "num_cpus": 8},
        "movie_config": {"enabled": False, "movie_path": "",
                         "deformation_field_path": "", "pre_exposure": 0.0,
                         "fluence_per_frame": 1.0},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))


def main() -> None:
    """Extract the sites, build a stack from them, refine, and report the sharpening."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="gdp_curved_13pf")
    parser.add_argument("--results", default="results_gdp_curved")
    parser.add_argument("--constraint", type=pathlib.Path,
                        default=MT_ROOT / "configs"
                        / "filament_constraint_gdp_curved.h5")
    parser.add_argument("--optics", type=pathlib.Path,
                        default=MT_ROOT / "results_ctf"
                        / "GDP_curved_mgraph_optics.yaml")
    parser.add_argument("--min-z", type=float, default=6.0)
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    stem = None
    for short, template_stem, _ in A.SEARCHES:
        if args.tag.endswith(short):
            stem = template_stem
            break
    if stem is None:
        raise SystemExit(f"cannot infer a template for {args.tag}")
    geometry = TemplateLatticeGeometry.from_pdb(str(MT_ROOT / "models" / f"{stem}.pdb"))
    template = str(legacy_map(stem).relative_to(MT_ROOT))

    with open(args.optics, encoding="utf-8") as handle:
        optics = yaml.safe_load(handle)["optics_group"]
    optics.pop("label", None)
    optics["ctf_B_factor"] = 60.0

    table = MT_ROOT / args.results / f"output_correlation_table_{args.tag}.h5"
    detections = detections_from_hdf5(
        str(table), min_z_score=args.min_z
    ).reset_index(drop=True)
    frame = legacy_frame_geometry(table, stem)
    with h5py.File(args.constraint, "r") as handle:
        region_map = handle["maps/region_id"][:]
        drawn = {int(name): handle[f"regions/{name}/path"][:]
                 for name in handle["regions"]}
    print(frame.describe())
    print(f"{len(detections):,} table detections above z {args.min_z}; "
          f"N {geometry.n_protofilaments}, repeat {geometry.rise_angstrom:.3f} A, "
          f"axis offset {geometry.axis_offset_magnitude_angstrom:.1f} A\n")

    clean = collect_sites(detections, geometry, drawn, region_map,
                          frame, args.segments)
    if len(clean) < 10:
        raise SystemExit(f"only {len(clean)} sites collected; nothing to refine")
    print(f"\n{len(clean)} clean sites pooled, z "
          f"{clean['z_score'].min():.2f}-{clean['z_score'].max():.2f}")

    micrograph = f"Frames/GDP_curved_mgraph.mrc"
    stack = particle_stack(clean, optics, micrograph, template, args.results, args.tag,
                           frame)
    stack_path = MT_ROOT / args.results / f"stack_sites_{args.tag}.csv"
    stack.to_csv(stack_path, index=True)

    config_path = MT_ROOT / "configs" / f"refine_sites_{args.tag}.yaml"
    write_config(stack_path, config_path,
                 template, MT_ROOT / "configs" / f"match_tm_{args.tag}.yaml", args.gpus)

    manager = RefineTemplateManager.from_yaml(str(config_path))
    offsets = manager.orientation_refinement_config.euler_angles_offsets
    print(f"{offsets.shape[0]} orientation offsets x "
          f"{len(manager.defocus_refinement_config.defocus_values)} defocus")

    out = MT_ROOT / args.results / f"refined_sites_{args.tag}.csv"
    start = time.time()
    manager.run_refine_template(str(out), args.batch_size)
    print(f"wall time {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")

    after = pd.read_csv(out, index_col=0)
    print(f"\n{'angle':<20} {'distinct before':>16} {'distinct after':>15} "
          f"{'median |move|':>14}")
    for name, coarse, refined in (("phi (roll)", "phi", "refined_phi"),
                                  ("theta (tilt)", "theta", "refined_theta"),
                                  ("psi (in-plane)", "psi", "refined_psi")):
        if refined not in after.columns:
            continue
        old, new = after[coarse].to_numpy(), after[refined].to_numpy()
        move = abs(((new - old + 180.0) % 360.0) - 180.0)
        print(f"  {name:<18} {len(set(old.round(2))):16d} "
              f"{len(set(new.round(2))):15d} {move.mean():13.2f}°")
    gain = after["refined_scaled_mip"] - after["scaled_mip"]
    print(f"\n  z {after['scaled_mip'].mean():.3f} -> "
          f"{after['refined_scaled_mip'].mean():.3f} "
          f"(mean {gain.mean():+.3f}, {(gain > 0).mean():.0%} improved), "
          f"max {after['scaled_mip'].max():.2f} -> "
          f"{after['refined_scaled_mip'].max():.2f}")
    print(f"wrote {out}")


# NOTE: invoking under __main__ is required for the multiprocessing/GPU distribution.
if __name__ == "__main__":
    main()
