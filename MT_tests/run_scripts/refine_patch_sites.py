"""Refine the PATCH's own continuously-indexed sites, then re-test the register.

Stage 5 applied to the patch. The sites come from the patch's own search and its own
continuous arc-length index, so score and index share a template -- the thing the earlier
shortcut got wrong. Refinement then takes phi off the search grid, which matters here
more than it did for the rise: the register is read from the SCORE at each site, and a
site scored at a quantised phi is scored slightly off its own optimum.

Whether that helps is an open question rather than an assumption. Refinement lifted mean
z by +0.586 on the ring, but the register is a ~10% difference between alternate sites,
and refinement could as easily lift both parities equally and change nothing. Reported
before and after so the answer is visible.

Usage:
    python run_scripts/refine_patch_sites.py
"""

import argparse
import pathlib
import sys
import time

import h5py
import numpy as np
import pandas as pd
import yaml

from leopard_em.analysis.filament_lattice import TemplateLatticeGeometry
from leopard_em.pydantic_models.managers import RefineTemplateManager
from leopard_em.pydantic_models.results.correlation_table import detections_from_hdf5

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyse_gdp_curved as A  # noqa: E402
from continuous_unwrap_curved import continuous_sites  # noqa: E402
from frame_geometry import legacy_frame_geometry, legacy_map  # noqa: E402
from refine_sites_curved import particle_stack  # noqa: E402
from seam_from_patch_search import TEMPLATES, alternation  # noqa: E402



def main() -> None:
    """Extract continuously, refine, and compare the register before and after."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default="patch", choices=sorted(TEMPLATES))
    parser.add_argument("--results", default="results_gdp_curved")
    parser.add_argument("--constraint", type=pathlib.Path,
                        default=MT_ROOT / "configs"
                        / "filament_constraint_gdp_curved.h5")
    parser.add_argument("--optics", type=pathlib.Path,
                        default=MT_ROOT / "results_ctf"
                        / "GDP_curved_mgraph_optics.yaml")
    parser.add_argument("--min-z", type=float, default=6.0)
    parser.add_argument("--resolve-protofilaments", action="store_true",
                        help="one site per (axial index, protofilament); patch only -- a "
                             "ring's phi does not identify a protofilament")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    tag, stem = TEMPLATES[args.template]
    geometry = TemplateLatticeGeometry.from_pdb(str(MT_ROOT / "models" / f"{stem}.pdb"))
    template = str(legacy_map(stem).relative_to(MT_ROOT))
    with open(args.optics, encoding="utf-8") as handle:
        optics = yaml.safe_load(handle)["optics_group"]
    optics.pop("label", None)
    optics["ctf_B_factor"] = 60.0

    table = MT_ROOT / args.results / f"output_correlation_table_{tag}.h5"
    detections = detections_from_hdf5(
        str(table), min_z_score=args.min_z).reset_index(drop=True)
    frame = legacy_frame_geometry(table, stem)
    print(frame.describe())
    with h5py.File(args.constraint, "r") as handle:
        region_map = handle["maps/region_id"][:]
        drawn_ids = sorted(int(n) for n in handle["regions"])
    y = np.clip(detections["y"].to_numpy() + frame.half_width, 0,
                region_map.shape[0] - 1)
    x = np.clip(detections["x"].to_numpy() + frame.half_width, 0,
                region_map.shape[1] - 1)
    owner = region_map[y, x]

    rng = np.random.default_rng(0)
    rows, labels = [], []
    for region in drawn_ids:
        mask = owner == region
        if mask.sum() < 200:
            continue
        subset = detections[mask].reset_index(drop=True)
        boot = A.pick_bootstrap(subset["z_score"].to_numpy(), want=20)
        if boot is None:
            continue
        sites = continuous_sites(
            subset, geometry, boot, pixel_size=frame.pixel_size,
            resolve_protofilaments=args.resolve_protofilaments)
        clean = subset.iloc[sites["detection_index"]].reset_index(drop=True)
        clean["region"] = region
        clean["segment"] = 1
        clean["axial_index"] = sites["axial_index"]
        clean["protofilament"] = (sites["protofilament"]
                                  if sites["protofilament"] is not None else -1)
        rows.append(clean)
        labels.append((region, sites))
    if not rows:
        raise SystemExit("no sites")
    clean = pd.concat(rows, ignore_index=True)
    print(f"{len(clean)} continuously-indexed {args.template} sites "
          f"over {len(rows)} region(s)\n")

    stack = particle_stack(clean, optics, "Frames/GDP_curved_mgraph.mrc",
                           template, args.results, tag, frame)
    stack["axial_index"] = clean["axial_index"].to_numpy()
    stack["protofilament"] = clean["protofilament"].to_numpy()
    stack_path = MT_ROOT / args.results / f"stack_continuous_{tag}.csv"
    stack.to_csv(stack_path, index=True)

    config = {
        "template_volume_path": template,
        "particle_stack": {"df_path": str(stack_path.relative_to(MT_ROOT)),
                           "extracted_box_size": [640, 640],
                           "original_template_size": [600, 600]},
        "defocus_refinement_config": {"enabled": True, "defocus_min": -100.0,
                                      "defocus_max": 100.0, "defocus_step": 25.0},
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
        "computational_config": {"gpu_ids": args.gpus, "num_cpus": 8},
        "movie_config": {"enabled": False, "movie_path": "",
                         "deformation_field_path": "", "pre_exposure": 0.0,
                         "fluence_per_frame": 1.0},
    }
    config_path = MT_ROOT / "configs" / f"refine_continuous_{tag}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    out = MT_ROOT / args.results / f"refined_continuous_{tag}.csv"
    start = time.time()
    manager = RefineTemplateManager.from_yaml(str(config_path))
    manager.run_refine_template(str(out), args.batch_size, allow_file_overwrite=True)
    after = pd.read_csv(out, index_col=0)
    print(f"refined {len(after)} sites in {time.time() - start:.0f}s")

    move = abs(((after["refined_phi"].to_numpy() - after["phi"].to_numpy() + 180)
                % 360) - 180)
    gain = after["refined_scaled_mip"] - after["scaled_mip"]
    print(f"  phi {len(set(after['phi'].round(2))):3d} -> "
          f"{len(set(after['refined_phi'].round(2))):3d} distinct, median move "
          f"{np.median(move):.2f}°")
    print(f"  z {after['scaled_mip'].mean():.3f} -> "
          f"{after['refined_scaled_mip'].mean():.3f} ({gain.mean():+.3f}, "
          f"{(gain > 0).mean():.0%} improved)\n")

    index_all = clean["axial_index"].to_numpy().astype(int)
    region_all = clean["region"].to_numpy()
    print(f"  {'region':<8} {'n':>4} {'score':<10} {'amp':>7} {'chance':>8} "
          f"{'ratio':>7} {'ctrl':>7}")
    for region in sorted(set(region_all.tolist())):
        keep = region_all == region
        index = index_all[keep]
        for label, score in (("before", after["scaled_mip"].to_numpy()[keep]),
                             ("refined", after["refined_scaled_mip"].to_numpy()[keep])):
            amplitude, null = alternation(index, score, rng)
            c_amp, c_null = alternation(index // 2, score, rng)
            print(f"  {region:<8} {len(index):4d} {label:<10} {amplitude:7.3f} "
                  f"{null:8.3f} {amplitude / null:6.2f}x {c_amp / c_null:6.2f}x")
    print("\n  Reference: 2.19x on the curved synthetic where the register IS there.")


# NOTE: invoking under __main__ is required for the multiprocessing/GPU distribution.
if __name__ == "__main__":
    main()
