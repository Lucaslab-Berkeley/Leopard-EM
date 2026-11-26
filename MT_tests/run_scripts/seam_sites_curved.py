"""Stage 7 of PIPELINE.md: the monomer register / seam, with the PATCH template.

The ring cannot do this test. A 3-start helix spreads 13 protofilaments' alpha/beta
phases over 1.5 dimer periods, so only 21.7% of the dimer order survives projection for a
ring against 54.8% for a 5-protofilament patch -- and on the curved synthetic with known
truth the patch read the register at 2.19x chance where the ring gave 0.69x, i.e. BELOW
its own chance level. The ring is carried here only as the negative control.

Two things this stage needs that the earlier ones did not:

* **The axial index.** The alternation is score against axial index parity, so sites are
  re-extracted here with their indices kept.
* **The patch's axis offset.** A ring is axis-centred, so its box centre is the axis
  point. The 5-PF patch's box centre sits 85.8 A off the axis and that offset ROTATES
  with phi, so the patch's expected position has to be recomputed for every roll seed:
  position = axis_point - R^T(offset) / pixel_size.

Controls, all reported:
  * the DIMER-rate index, which puts every site in the same register -- any alternation
    there is an artefact of the method and must come out near 1;
  * a score-permutation null, the 95th percentile over 2000 shuffles;
  * the ring, which should sit at or below chance.

Writes every intermediate to disk, so it can be re-read without re-running.

Usage:
    python run_scripts/seam_sites_curved.py --tag gdp_curved_13pf
"""

import argparse
import json
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
    image_offset_from_template_point,
)
from leopard_em.pydantic_models.managers import RefineTemplateManager
from leopard_em.pydantic_models.results.correlation_table import detections_from_hdf5

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyse_gdp_curved as A  # noqa: E402
from compete_sites_curved import write_config  # noqa: E402
from frame_geometry import legacy_frame_geometry, legacy_map  # noqa: E402

ROLL_STEP_DEG = 4.0

RING_STEM = "6dpu_atoms_6dpv_lattice_13pf_2rings_flatB"
PATCH_STEM = "6dpu_atoms_6dpv_lattice_13pf_patch5_2rings_flatB"


def sites_with_index(detections, geometry, drawn, region_map, frame):
    """Clean lattice sites, keeping the axial index the register test needs."""
    half = frame.half_width
    y = np.clip(detections["y"].to_numpy() + half, 0, region_map.shape[0] - 1)
    x = np.clip(detections["x"].to_numpy() + half, 0, region_map.shape[1] - 1)
    owner = region_map[y, x]

    rows = []
    for region in sorted(drawn):
        mask = owner == region
        if mask.sum() < 100:
            continue
        subset = detections[mask]
        original = np.flatnonzero(mask)
        curve = A.DrawnCurve(drawn[region], pixel_size=frame.pixel_size)
        k = A.segment_count(curve)
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
            for row, index in zip(chosen, np.asarray(sites.axial_index)):
                rows.append((int(row), int(region), part + 1, int(index)))
            print(f"  region {region} segment {part + 1}/{k}: {len(chosen):3d} sites, "
                  f"{sites.occupancy:4.0%} occupied, rise {sites.rise_angstrom:7.3f} A")

    frame = detections.iloc[[r[0] for r in rows]].reset_index(drop=True)
    frame["region"] = [r[1] for r in rows]
    frame["segment"] = [r[2] for r in rows]
    frame["axial_index"] = [r[3] for r in rows]
    return frame


def build_stack(sites, geometry, optics, results, tag, seeds, ring_stack, frame):
    """One row per (site, roll seed), with the patch's own axis offset applied."""
    axis_point = np.array([*geometry.axis_offset_angstrom, 0.0], dtype=np.float64)
    blocks = []
    for position in range(len(sites)):
        row = ring_stack.iloc[position]
        for seed in seeds:
            copy = row.copy()
            phi = (float(row["phi"]) + seed) % 360.0
            copy["phi"] = phi
            if np.linalg.norm(axis_point) > 1e-9:
                # Where this template's BOX CENTRE must sit for its axis to fall on the
                # site. A ring has a zero offset and skips this entirely.
                offset = image_offset_from_template_point(
                    axis_point, np.array([phi]), np.array([float(row["theta"])]),
                    np.array([float(row["psi"])]),
                )[0]
                copy["pos_x"] = int(round(float(row["pos_x"])
                                          - offset[0] / frame.pixel_size))
                copy["pos_y"] = int(round(float(row["pos_y"])
                                          - offset[1] / frame.pixel_size))
                copy["pos_x_img"] = copy["pos_x"] + frame.half_width
                copy["pos_y_img"] = copy["pos_y"] + frame.half_width
            blocks.append(copy)
    stack = pd.DataFrame(blocks).reset_index(drop=True)
    stack["particle_index"] = np.arange(len(stack))
    return stack


def alternation(index, score, rng, n_null=2000):
    """Period-2 amplitude of score against axial index, and its permutation null."""
    weight = score - score.mean()
    phase = np.exp(1j * np.pi * index)
    amplitude = float(np.abs((weight * phase).sum()) / len(index))
    null = float(np.percentile(
        [float(np.abs((rng.permutation(weight) * phase).sum()) / len(index))
         for _ in range(n_null)], 95))
    return amplitude, null


def main() -> None:
    """Score the patch at the ring's sites and test the register."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="gdp_curved_13pf")
    parser.add_argument("--results", default="results_gdp_curved")
    parser.add_argument("--constraint", type=pathlib.Path,
                        default=MT_ROOT / "configs"
                        / "filament_constraint_gdp_curved.h5")
    parser.add_argument("--min-z", type=float, default=6.0)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    ring_geometry = TemplateLatticeGeometry.from_pdb(
        str(MT_ROOT / "models" / f"{RING_STEM}.pdb"))
    patch_geometry = TemplateLatticeGeometry.from_pdb(
        str(MT_ROOT / "models" / f"{PATCH_STEM}.pdb"))
    print(f"ring  N {ring_geometry.n_protofilaments}, axis offset "
          f"{ring_geometry.axis_offset_magnitude_angstrom:.1f} A")
    table = MT_ROOT / args.results / f"output_correlation_table_{args.tag}.h5"
    frame = legacy_frame_geometry(table, RING_STEM)
    print(f"patch N {patch_geometry.n_protofilaments}, axis offset "
          f"{patch_geometry.axis_offset_magnitude_angstrom:.1f} A "
          f"({patch_geometry.axis_offset_magnitude_angstrom / frame.pixel_size:.0f} px)")
    print(f"{frame.describe()}\n")

    sites_path = MT_ROOT / args.results / f"sites_indexed_{args.tag}.csv"
    if sites_path.is_file() and args.report_only:
        sites = pd.read_csv(sites_path, index_col=0)
    else:
        detections = detections_from_hdf5(
            str(table), min_z_score=args.min_z,
        ).reset_index(drop=True)
        with h5py.File(args.constraint, "r") as handle:
            region_map = handle["maps/region_id"][:]
            drawn = {int(name): handle[f"regions/{name}/path"][:]
                     for name in handle["regions"]}
        sites = sites_with_index(detections, ring_geometry, drawn, region_map, frame)
        sites.to_csv(sites_path, index=True)
    print(f"\n{len(sites)} indexed sites\n")

    # The ring stack already exists from Stage 5; reuse its optics and paths verbatim.
    ring_stack = pd.read_csv(
        MT_ROOT / args.results / f"stack_sites_{args.tag}.csv", index_col=0)
    refined = pd.read_csv(
        MT_ROOT / args.results / f"refined_sites_{args.tag}.csv", index_col=0)
    for column in ("phi", "theta", "psi", "pos_x", "pos_y", "relative_defocus"):
        ring_stack[column] = refined[f"refined_{column}"].to_numpy()
    ring_stack["pos_x"] = ring_stack["pos_x"].round().astype(int)
    ring_stack["pos_y"] = ring_stack["pos_y"].round().astype(int)

    if len(ring_stack) != len(sites):
        print(f"WARNING: {len(sites)} indexed sites but {len(ring_stack)} in the Stage 5 "
              f"stack; using the first {min(len(sites), len(ring_stack))}")
        n = min(len(sites), len(ring_stack))
        sites, ring_stack = sites.iloc[:n], ring_stack.iloc[:n]

    half = 180.0 / ring_geometry.n_protofilaments
    seeds = np.linspace(-half, half, max(int(np.ceil(2 * half / ROLL_STEP_DEG)), 1) + 1)[:-1]
    print(f"{len(seeds)} roll seeds spanning {seeds.min():+.1f}..{seeds.max():+.1f}°\n")

    scores = {}
    for key, stem, geometry in (("patch", PATCH_STEM, patch_geometry),
                                ("ring", RING_STEM, ring_geometry)):
        out = MT_ROOT / args.results / f"seam_refined_{key}.csv"
        if not (args.report_only and out.is_file()):
            stack = build_stack(sites, geometry, None, args.results, args.tag,
                                seeds, ring_stack, frame)
            stack_path = MT_ROOT / args.results / f"seam_stack_{key}.csv"
            stack.to_csv(stack_path, index=True)
            config_path = MT_ROOT / "configs" / f"seam_{key}.yaml"
            write_config(stack_path, config_path,
                         str(legacy_map(stem).relative_to(MT_ROOT)),
                         MT_ROOT / "configs" / f"match_tm_{args.tag}.yaml", args.gpus)
            start = time.time()
            manager = RefineTemplateManager.from_yaml(str(config_path))
            manager.run_refine_template(str(out), args.batch_size)
            print(f"  {key}: {len(stack)} rows refined in {time.time() - start:.0f}s")
        result = pd.read_csv(out, index_col=0)
        z = result["refined_scaled_mip"].to_numpy().reshape(len(sites), len(seeds))
        scores[key] = z.max(axis=1)

    sites["patch_z"] = scores["patch"]
    sites["ring_z"] = scores["ring"]
    sites.to_csv(MT_ROOT / args.results / f"seam_scores_{args.tag}.csv", index=True)

    rng = np.random.default_rng(0)
    summary = {}
    print(f"\n{'=' * 78}\nMONOMER REGISTER -- period-2 alternation of score with axial "
          f"index\n{'=' * 78}")
    # PER SEGMENT ONLY. extract_lattice_sites phases the axial index on each segment's
    # own bootstrap, so index 0 means a different physical subunit in every segment --
    # each one here runs from about -13 to +20 about its own origin. Parity is therefore
    # not comparable between segments, and pooling them does not average the alternation,
    # it SCRAMBLES it: pooling these six gave amplitude 0.005 against 0.086-0.108 per
    # region, which is the mixing, not the lattice. Combining segments would need their
    # phases tied together first, which needs one axis spanning the whole tube -- the
    # curved-axis work that does not exist yet.
    print(f"  {'segment':<12} {'n':>4} {'template':<7} {'amp':>7} {'chance':>8} "
          f"{'ratio':>7} {'dimer ctrl':>11}")
    ratios = {"patch": [], "ring": []}
    for (region_id, segment), block in sites.groupby(["region", "segment"]):
        index = block["axial_index"].to_numpy().astype(int)
        if len(index) < 8:
            print(f"  {f'R{region_id} seg{segment}':<12} {len(index):4d}  too few sites")
            continue
        for key in ("patch", "ring"):
            score = block[f"{key}_z"].to_numpy()
            amplitude, null = alternation(index, score, rng)
            # Dimer control: index at half the rate, so every site shares a register.
            ctrl_amp, ctrl_null = alternation(index // 2, score, rng)
            ratio = amplitude / null if null else float("nan")
            ratios[key].append(ratio)
            summary[f"region{region_id}/segment{segment}/{key}"] = {
                "n": int(len(index)), "amplitude": amplitude, "chance": null,
                "ratio": ratio,
                "dimer_control_ratio": ctrl_amp / ctrl_null if ctrl_null else float("nan"),
            }
            print(f"  {f'R{region_id} seg{segment}':<12} {len(index):4d} {key:<7} "
                  f"{amplitude:7.3f} {null:8.3f} {ratio:6.2f}x "
                  f"{ctrl_amp / ctrl_null:10.2f}x")

    print()
    for key in ("patch", "ring"):
        if ratios[key]:
            values = np.asarray(ratios[key])
            print(f"  {key:<6} median {np.median(values):.2f}x, max {values.max():.2f}x "
                  f"over {len(values)} segments")
            summary[f"{key}_median_ratio"] = float(np.median(values))
            summary[f"{key}_max_ratio"] = float(values.max())

    (MT_ROOT / args.results / f"seam_summary_{args.tag}.json").write_text(
        json.dumps(summary, indent=2))
    print("\n  A ratio above 1 on the PATCH with the dimer control near 1 is a register.")
    print("  The RING is the negative control and is expected at or below 1 -- on the")
    print("  curved synthetic with known truth it gave 0.69x against the patch's 2.19x.")
    print(f"\nwrote seam_scores_{args.tag}.csv and seam_summary_{args.tag}.json")


# NOTE: invoking under __main__ is required for the multiprocessing/GPU distribution.
if __name__ == "__main__":
    main()
