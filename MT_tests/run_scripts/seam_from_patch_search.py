"""The seam, from the PATCH's own match_template table. The validated path.

This replaces the Stage 7 shortcut, which placed a patch at the ring's sites and does not
work: on the curved synthetic a full patch search reads the register at 2.19x and the
shortcut returns 0.06x. The fault was never the placement (the placed patch scored mean z
11.94 against the full search's 11.73) but the INDEX -- the full patch extraction indexes
each detection by its own phi through ``s_corr = s + (lateral_offset/angular_spacing)*phi``,
so score and index must come from the same template. See validate_seam_shortcut.py.

So here everything comes from the patch: its own correlation table, its own geometry, its
own extracted sites, its own axial indices, its own scores. The ring is read the same way
from the ring table as the negative control -- on the synthetic it gives 0.69x, below its
own chance level, because a 3-start helix cancels the dimer order for a full ring.

Segmented exactly as the other curved readouts are, and reported PER SEGMENT only:
extract_lattice_sites phases the axial index on each segment's own bootstrap, so index 0 is
a different physical subunit in every segment and pooling scrambles the parity rather than
averaging it.

Usage:
    python run_scripts/seam_from_patch_search.py
"""

import argparse
import json
import pathlib
import sys

import h5py
import numpy as np

from leopard_em.analysis.filament_lattice import (
    TemplateLatticeGeometry,
    axis_points_from_peaks,
    extract_lattice_sites,
)
from leopard_em.pydantic_models.results.correlation_table import detections_from_hdf5

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyse_gdp_curved as A  # noqa: E402
from frame_geometry import legacy_frame_geometry  # noqa: E402

TEMPLATES = {
    "patch": ("gdp_curved_13pf_patch",
              "6dpu_atoms_6dpv_lattice_13pf_patch5_2rings_flatB"),
    "ring": ("gdp_curved_13pf", "6dpu_atoms_6dpv_lattice_13pf_2rings_flatB"),
}


def alternation(index, score, rng, n_null=4000):
    """Period-2 amplitude of score against axial index, and its permutation null."""
    weight = score - score.mean()
    phase = np.exp(1j * np.pi * index)
    amplitude = float(np.abs((weight * phase).sum()) / len(index))
    null = float(np.percentile(
        [float(np.abs((rng.permutation(weight) * phase).sum()) / len(index))
         for _ in range(n_null)], 95))
    return amplitude, null


def segments_for(detections, geometry, drawn, region_map, frame):
    """Yield (region, segment, detections) for each stretch of drawn tube.

    ``frame`` (``frame_geometry``) carries the table's pixel size and its offset to the
    micrograph frame that ``region_map`` and the drawn paths live in.
    """
    half = frame.half_width
    y = np.clip(detections["y"].to_numpy() + half, 0, region_map.shape[0] - 1)
    x = np.clip(detections["x"].to_numpy() + half, 0, region_map.shape[1] - 1)
    owner = region_map[y, x]
    for region in sorted(drawn):
        mask = owner == region
        if mask.sum() < 100:
            continue
        subset = detections[mask].reset_index(drop=True)
        curve = A.DrawnCurve(drawn[region], pixel_size=frame.pixel_size)
        k = A.segment_count(curve)
        points = axis_points_from_peaks(subset, geometry, frame.pixel_size) + half
        arc, _ = curve.project(points)
        edges = np.linspace(0.0, curve.length_px, k + 1)
        for part in range(k):
            inside = (arc >= edges[part]) & (arc <= edges[part + 1])
            if inside.sum() < 200:
                continue
            yield region, part + 1, k, subset[inside].reset_index(drop=True)


def main() -> None:
    """Register test from each template's own search, per segment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results_gdp_curved")
    parser.add_argument("--constraint", type=pathlib.Path,
                        default=MT_ROOT / "configs"
                        / "filament_constraint_gdp_curved.h5")
    parser.add_argument("--min-z", type=float, default=6.0)
    args = parser.parse_args()

    with h5py.File(args.constraint, "r") as handle:
        region_map = handle["maps/region_id"][:]
        drawn = {int(name): handle[f"regions/{name}/path"][:]
                 for name in handle["regions"]}

    rng = np.random.default_rng(0)
    summary, ratios = {}, {"patch": [], "ring": []}
    print(f"{'template':<7} {'segment':<10} {'n':>4} {'rise Å':>8} {'amp':>7} "
          f"{'chance':>8} {'ratio':>7} {'dimer ctrl':>11}")
    for key, (tag, stem) in TEMPLATES.items():
        table = MT_ROOT / args.results / f"output_correlation_table_{tag}.h5"
        if not table.is_file():
            print(f"{key:<7} no table at {table.name} -- has the search finished?")
            continue
        geometry = TemplateLatticeGeometry.from_pdb(str(MT_ROOT / "models" / f"{stem}.pdb"))
        detections = detections_from_hdf5(
            str(table), min_z_score=args.min_z
        ).reset_index(drop=True)
        frame = legacy_frame_geometry(table, stem)
        print(f"# {key}: {frame.describe()}")
        print(f"# {key}: {len(detections):,} detections, max z "
              f"{detections['z_score'].max():.2f}, axis offset "
              f"{geometry.axis_offset_magnitude_angstrom:.1f} Å")

        for region, part, k, piece in segments_for(
            detections, geometry, drawn, region_map, frame
        ):
            boot = A.pick_bootstrap(piece["z_score"].to_numpy())
            if boot is None:
                continue
            try:
                sites = extract_lattice_sites(
                    piece, geometry, frame.pixel_size, bootstrap_score_threshold=boot
                )
                control = extract_lattice_sites(
                    piece, geometry, frame.pixel_size, bootstrap_score_threshold=boot,
                    initial_rise_angstrom=2.0 * geometry.rise_angstrom,
                )
            except ValueError:
                continue
            index = np.asarray(sites.axial_index).astype(int)
            if len(index) < 8:
                continue
            amplitude, null = alternation(index, np.asarray(sites.score), rng)
            ctrl_index = np.asarray(control.axial_index).astype(int)
            ctrl = (alternation(ctrl_index, np.asarray(control.score), rng)
                    if len(ctrl_index) >= 8 else (float("nan"), float("nan")))
            ratio = amplitude / null if null else float("nan")
            ratios[key].append(ratio)
            summary[f"{key}/region{region}/segment{part}"] = {
                "n": int(len(index)), "rise": float(sites.rise_angstrom),
                "amplitude": amplitude, "chance": null, "ratio": ratio,
                "dimer_control_ratio": float(ctrl[0] / ctrl[1]) if ctrl[1] else None,
            }
            print(f"{key:<7} {f'R{region} seg{part}/{k}':<10} {len(index):4d} "
                  f"{sites.rise_angstrom:8.3f} {amplitude:7.3f} {null:8.3f} "
                  f"{ratio:6.2f}x {ctrl[0] / ctrl[1]:10.2f}x")

    print()
    for key in ("patch", "ring"):
        if ratios[key]:
            values = np.asarray(ratios[key])
            print(f"  {key:<6} median {np.median(values):.2f}x, max {values.max():.2f}x "
                  f"over {len(values)} segments")
            summary[f"{key}_median_ratio"] = float(np.median(values))
            summary[f"{key}_max_ratio"] = float(values.max())
    print("\n  Reference points: on the curved synthetic with KNOWN truth a full patch")
    print("  search gives 2.19x with the dimer control at 0.60x, and the ring 0.69x.")
    print("  A patch ratio above 1 with its control near 1 is a register.")

    (MT_ROOT / args.results / "seam_from_patch_search.json").write_text(
        json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
