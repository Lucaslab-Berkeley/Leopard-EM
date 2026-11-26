"""Re-measure the lattice spacing from the REFINED sites (Stage 5 output).

Three things change against the Stage 4 numbers, and each is reported separately so the
effect of each is visible rather than bundled:

1. **Refined angles.** phi enters the helical unwrap directly
   (``s_corr = s + (lateral_offset/angular_spacing)*phi``), so quantised phi propagates
   straight into the rise. Refinement took phi from 59 to 101 distinct values here.
2. **Refined positions.** refine_template searches position inside the extracted box,
   so the site coordinates move too.
3. **The sin(theta) correction.** Out-of-plane tilt shortens the projected repeat by
   sin(theta). ``extract_lattice_sites`` does NOT apply it -- its internal call to
   ``estimate_lattice_rise`` passes no ``theta_deg`` -- so every Stage 4 rise is a
   projected repeat, not an axial one. Passing per-peak theta divides it out.

The chord-vs-arc shortening is still measured off the drawn curve per segment and
divided out, exactly as in Stage 4.

Usage:
    python run_scripts/remeasure_spacing_curved.py --tag gdp_curved_13pf
"""

import argparse
import math
import pathlib
import sys

import h5py
import numpy as np
import pandas as pd

from leopard_em.analysis.filament_lattice import (
    TemplateLatticeGeometry,
    axis_points_from_peaks,
    estimate_lattice_rise,
    filament_coordinates,
    filament_direction_from_angles,
    fit_filament_axis,
    unwrap_helical_axial_coordinate,
)

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyse_gdp_curved as A  # noqa: E402


def frame_for(sites: pd.DataFrame, refined: bool) -> pd.DataFrame:
    """A detections-shaped frame using either the coarse or the refined columns."""
    prefix = "refined_" if refined else ""
    x = sites[f"{prefix}pos_x"] if refined else sites["pos_x"]
    y = sites[f"{prefix}pos_y"] if refined else sites["pos_y"]
    return pd.DataFrame({
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y, dtype=float),
        "phi": sites[f"{prefix}phi"].to_numpy(),
        "theta": sites[f"{prefix}theta"].to_numpy(),
        "psi": sites[f"{prefix}psi"].to_numpy(),
        "z_score": sites[f"{prefix}scaled_mip"].to_numpy()
        if refined else sites["scaled_mip"].to_numpy(),
    })


def rise_for(frame: pd.DataFrame, geometry, use_theta: bool, pixel_size: float) -> tuple:
    """Fitted rise for one segment, by the same path extract_lattice_sites takes."""
    points = axis_points_from_peaks(frame, geometry, pixel_size, prefer_refined=False)
    try:
        reference = filament_direction_from_angles(
            frame["phi"].to_numpy(), frame["theta"].to_numpy(),
            frame["psi"].to_numpy(),
        )
    except ValueError:
        reference = None
    axis = fit_filament_axis(points, frame["z_score"].to_numpy(),
                             reference_direction=reference)
    along, _ = filament_coordinates(points, axis)
    unwrapped = unwrap_helical_axial_coordinate(
        along * pixel_size, frame["phi"].to_numpy(), geometry
    )
    result = estimate_lattice_rise(
        unwrapped / pixel_size,
        pixel_size,
        geometry.rise_angstrom,
        frame["z_score"].to_numpy(),
        theta_deg=frame["theta"].to_numpy() if use_theta else None,
    )
    return float(result.rise_angstrom), axis


def main() -> None:
    """Report the rise per segment under coarse and refined angles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="gdp_curved_13pf")
    parser.add_argument("--results", default="results_gdp_curved")
    parser.add_argument("--constraint", type=pathlib.Path,
                        default=MT_ROOT / "configs"
                        / "filament_constraint_gdp_curved.h5")
    args = parser.parse_args()

    stem = None
    for short, template_stem, _ in A.SEARCHES:
        if args.tag.endswith(short):
            stem = template_stem
            break
    geometry = TemplateLatticeGeometry.from_pdb(str(MT_ROOT / "models" / f"{stem}.pdb"))

    sites = pd.read_csv(
        MT_ROOT / args.results / f"refined_sites_{args.tag}.csv", index_col=0
    )
    # run_refine_template writes its own fixed column set, so the region and segment
    # labels do not survive it. Bring them back from the stack it was given, matched on
    # particle_index rather than on row order.
    if "region" not in sites.columns:
        stack = pd.read_csv(
            MT_ROOT / args.results / f"stack_sites_{args.tag}.csv", index_col=0
        )
        key = "particle_index" if "particle_index" in sites.columns else None
        if key is None:
            sites["region"] = stack["region"].to_numpy()[: len(sites)]
            sites["segment"] = stack["segment"].to_numpy()[: len(sites)]
        else:
            labels = stack.set_index("particle_index")[["region", "segment"]]
            sites = sites.join(labels, on=key)
    with h5py.File(args.constraint, "r") as handle:
        drawn = {int(name): handle[f"regions/{name}/path"][:]
                 for name in handle["regions"]}
    # The refined sites carry the search's optics, so the pixel size they were measured
    # at comes with them rather than from a constant.
    pixel_size = float(sites["pixel_size"].iloc[0])

    print(f"pixel size {pixel_size:.4f} A (from the refined sites)")
    print(f"{len(sites)} refined sites; template repeat "
          f"{geometry.rise_angstrom:.3f} Å (N {geometry.n_protofilaments})")
    theta = np.radians(sites["refined_theta"].to_numpy())
    # The correction applied is the per-peak mean of 1/sin(theta), NOT 1/sin(mean theta).
    # With theta spread over ~8 degrees those differ by an order of magnitude, because
    # 1/sin is convex -- the same reason STATUS warns that mean(sin t) != sin(mean t).
    print(f"theta {sites['refined_theta'].min():.1f}-{sites['refined_theta'].max():.1f}°, "
          f"mean {sites['refined_theta'].mean():.2f}°")
    print(f"  1/sin(mean theta) - 1 = "
          f"{100 * (1 / math.sin(theta.mean()) - 1):+.3f}%  <- the naive number")
    print(f"  mean(1/sin theta) - 1 = "
          f"{100 * (np.mean(1 / np.sin(theta)) - 1):+.3f}%  <- what is actually applied\n")

    print(f"  {'seg':<8} {'n':>4} {'coarse':>9} {'refined':>9} {'+sin θ':>9} "
          f"{'bias':>8} {'FINAL':>9} {'vs template':>12}")
    rows = []
    for region in sorted(sites["region"].unique()):
        curve = A.DrawnCurve(drawn[int(region)], pixel_size=pixel_size)
        block = sites[sites["region"] == region]
        k = int(block["segment"].max())
        # Segment count was chosen from the curve in Stage 4; recover its edges.
        k = A.segment_count(curve)
        edges = np.linspace(0.0, curve.length_px, k + 1)
        for segment in sorted(block["segment"].unique()):
            piece = block[block["segment"] == segment]
            if len(piece) < 8:
                print(f"  {f'{region}.{segment}':<8} {len(piece):4d}  too few sites")
                continue
            coarse, _ = rise_for(frame_for(piece, False), geometry, False, pixel_size)
            refined, _ = rise_for(frame_for(piece, True), geometry, False, pixel_size)
            tilted, _ = rise_for(frame_for(piece, True), geometry, True, pixel_size)
            bias = curve.shortening(edges[segment - 1], edges[segment])
            final = tilted / (1.0 + bias)
            rows.append((int(region), len(piece), final))
            print(f"  {f'{region}.{segment}':<8} {len(piece):4d} {coarse:9.3f} "
                  f"{refined:9.3f} {tilted:9.3f} {100 * bias:+7.3f}% {final:9.3f} "
                  f"{100 * (final - geometry.rise_angstrom) / geometry.rise_angstrom:+11.2f}%")

    if len(rows) < 2:
        return
    # Report per region. The two regions are different microtubules, so pooling them
    # into one mean would average two specimens that need not agree.
    print()
    for region in sorted({r[0] for r in rows}):
        block = [r for r in rows if r[0] == region]
        if len(block) < 2:
            print(f"  REGION {region}: only {len(block)} segment, no scatter estimate")
            continue
        counts = np.array([r[1] for r in block], dtype=float)
        values = np.array([r[2] for r in block])
        mean = float(np.average(values, weights=counts))
        scatter = float(np.std(values, ddof=1))
        percent = 100 * (mean - geometry.rise_angstrom) / geometry.rise_angstrom
        print(f"  REGION {region}: {mean:.3f} Å over {int(counts.sum())} sites in "
              f"{len(block)} segments, scatter {scatter:.3f} Å "
              f"(sem {scatter / math.sqrt(len(values)):.3f})  ->  {percent:+.2f}% "
              f"from the template repeat")
    print("\n  Columns: 'coarse' is the Stage 4 path (search-grid angles, no tilt "
          "correction);\n  'refined' swaps in the Stage 5 angles and positions; "
          "'+sin θ' divides out the\n  out-of-plane tilt that extract_lattice_sites "
          "never applied; 'bias' is the\n  chord-vs-arc shortening measured off the "
          "drawn curve.")


if __name__ == "__main__":
    main()
