"""Stage 3: read the ring's correlation table, per filament -- membership from the data.

Two readouts come straight off the table with no axis fit and no refinement:

* **polarity** -- ``estimate_polarity`` on psi alone, so bending cannot bias it;
* **protofilament number, first line** -- ``estimate_protofilament_number``, a pure
  azimuth harmonic on phi, independent of the Stage 6 competition.

What changed from the in vitro runs is WHICH detections go into each filament. That was
the drawn region at each detection's box centre, which takes membership from the
drawing and breaks wherever bands overlap. Here ``assign_detections_to_filaments`` does
it from the detections themselves -- each one's own predicted axis point and in-plane
direction, against curves fitted to the data -- using the drawn paths only to seed who
starts where. Detections no filament accepts (clutter between tubes, off-axis false
positives) are reported and left out.

Geometry is read from the table. Tables from before geometry was recorded need
``--in-vitro-legacy`` (0.9194 A/px, the template's own map for the box).

Writes ``stage3_membership_<tag>.csv`` (one row per detection, with its filament) for
Stage 4, and ``stage3_membership_<tag>.png``.

Usage:
    python run_scripts/stage3_read_table.py \\
        --table results_in_situ_823/output_correlation_table_in_situ_823_13pf.h5 \\
        --constraint configs/filament_constraint_in_situ_823.h5 \\
        --micrograph results_in_situ/set1_823_Aug09_10.05.20_DW.mrc
"""

import argparse
import pathlib
import sys

import h5py
import mrcfile
import numpy as np
from scipy.spatial import cKDTree

from leopard_em.analysis.filament_lattice import (
    TemplateLatticeGeometry,
    assign_detections_to_filaments,
    estimate_polarity,
    estimate_protofilament_number,
)
from leopard_em.pydantic_models.results.correlation_table import detections_from_hdf5

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from frame_geometry import frame_geometry, legacy_frame_geometry  # noqa: E402

RING_STEM = "6dpu_atoms_6dpv_lattice_13pf_2rings_flatB"
STRETCH_NM = 100.0  # polarity per ~100 nm of tube, as the in vitro runs did per segment


def drawn_seeds(constraint: pathlib.Path) -> dict[int, np.ndarray]:
    """The drawn paths as (x, y) micrograph-frame polylines, keyed by region id."""
    with h5py.File(constraint, "r") as handle:
        return {int(name): np.asarray(handle[f"regions/{name}/path"][:])[:, ::-1]
                for name in handle["regions"]}


def harmonic_margin(power: np.ndarray, best: int) -> tuple[float, int]:
    """Power at the winning order over the runner-up's, and which order that was."""
    orders = np.arange(2, 2 + len(power))
    rest = np.where(orders == best, -np.inf, power)
    runner = int(orders[int(np.argmax(rest))])
    return float(power[orders == best][0] / power[orders == runner][0]), runner


def polarity_by_stretch(subset, axis_xy, curve, pixel_size):
    """Polarity per ~STRETCH_NM along the fitted curve, against its local tangent.

    A bent tube pooled whole gives a chord average of its direction; polarity is read
    per stretch instead (PIPELINE Stage 3), and each stretch is checked against the
    DATA-fitted curve's tangent there, not the drawing's. Returns rows of
    (stretch, n, psi, ratio, |psi - tangent| folded to the pole the stretch chose).
    """
    arc_step = np.hypot(*np.diff(curve, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(arc_step)])
    _, nearest = cKDTree(curve).query(axis_xy)
    position = arc[nearest]
    n_stretch = max(int(round(arc[-1] * pixel_size / 10 / STRETCH_NM)), 1)
    edges = np.linspace(0.0, arc[-1], n_stretch + 1)
    tangent = np.gradient(curve, axis=0)
    rows = []
    for k in range(n_stretch):
        inside = (position >= edges[k]) & (position <= edges[k + 1])
        if inside.sum() < 10:
            rows.append((k + 1, int(inside.sum()), None, None, None))
            continue
        estimate = estimate_polarity(subset[inside])
        middle = int(np.clip(np.searchsorted(arc, 0.5 * (edges[k] + edges[k + 1])),
                             0, len(curve) - 1))
        # The detections' axis direction is at 180 - psi in image axes.
        curve_dir = np.degrees(np.arctan2(tangent[middle, 1], tangent[middle, 0]))
        axis_dir = 180.0 - estimate.psi_deg
        off = abs(((axis_dir - curve_dir + 90.0) % 180.0) - 90.0)
        rows.append((k + 1, int(inside.sum()), estimate.psi_deg,
                     estimate.score_ratio, off))
    return rows


def old_rule_labels(detections, constraint, half) -> np.ndarray:
    """What the drawn-region-at-box-centre rule would have said, for comparison."""
    with h5py.File(constraint, "r") as handle:
        region = handle["maps/region_id"][:]
    y = np.clip(detections["y"].to_numpy() + half, 0, region.shape[0] - 1)
    x = np.clip(detections["x"].to_numpy() + half, 0, region.shape[1] - 1)
    return region[y.astype(int), x.astype(int)]


def plot(result, seeds, micrograph, out_path, frame):
    """Detections coloured by filament over the micrograph, with fitted and drawn curves."""
    import matplotlib  # pylint: disable=import-outside-toplevel
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    from scipy.ndimage import gaussian_filter  # pylint: disable=import-outside-toplevel

    fig, ax = plt.subplots(figsize=(8, 10))
    if micrograph is not None:
        image = mrcfile.read(micrograph).astype(np.float32)
        binning = 4
        h, w = image.shape[0] // binning * binning, image.shape[1] // binning * binning
        image = image[:h, :w].reshape(h // binning, binning, w // binning, binning).mean((1, 3))
        image = gaussian_filter(image, 1.5) - gaussian_filter(image, 60)
        lo, hi = np.percentile(image, (1, 99))
        ax.imshow(image, cmap="gray", vmin=lo, vmax=hi,
                  extent=(0, w, h, 0))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
               "#e377c2", "#17becf"]
    axis = result.axis_points_px
    unassigned = result.labels == 0
    ax.scatter(axis[unassigned, 0], axis[unassigned, 1], s=1, c="#aaaaaa", alpha=0.3,
               label=f"unassigned ({unassigned.sum():,})")
    for i, label in enumerate(sorted(seeds)):
        colour = palette[i % len(palette)]
        mine = result.labels == label
        ax.scatter(axis[mine, 0], axis[mine, 1], s=1, c=colour, alpha=0.4,
                   label=f"filament {label} ({mine.sum():,})")
        ax.plot(seeds[label][:, 0], seeds[label][:, 1], "--", color=colour, lw=0.8)
        if label in result.curves_px:
            curve = result.curves_px[label]
            ax.plot(curve[:, 0], curve[:, 1], "-", color="white", lw=0.7)
    ax.set_title(f"Stage 3 membership: dashed = drawn seed, white = fitted to data "
                 f"({frame.pixel_size:.3f} A/px)", fontsize=9)
    ax.legend(loc="lower left", markerscale=8, fontsize=7)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


# pylint: disable=too-many-locals,too-many-statements
def main() -> None:
    """Membership, then polarity and protofilament number, per filament."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=pathlib.Path, required=True)
    parser.add_argument("--constraint", type=pathlib.Path, required=True,
                        help="the HDF5 whose drawn paths seed membership")
    parser.add_argument("--stem", default=RING_STEM,
                        help="template model (for its axis offset); the RING by default")
    parser.add_argument("--micrograph", type=pathlib.Path, default=None,
                        help="only for the figure")
    parser.add_argument("--min-z", type=float, default=6.0)
    parser.add_argument("--in-vitro-legacy", action="store_true",
                        help="table predates recorded geometry: 0.9194 A/px, and the "
                             "box of maps/<stem>_0.9194_bscale0.5.mrc")
    parser.add_argument("--tag", default=None, help="output name; default from table")
    args = parser.parse_args()

    table = args.table if args.table.is_absolute() else MT_ROOT / args.table
    constraint = (args.constraint if args.constraint.is_absolute()
                  else MT_ROOT / args.constraint)
    frame = (legacy_frame_geometry(table, args.stem) if args.in_vitro_legacy
             else frame_geometry(table))
    geometry = TemplateLatticeGeometry.from_pdb(str(MT_ROOT / "models" / f"{args.stem}.pdb"))
    tag = args.tag or table.stem.removeprefix("output_correlation_table_")

    detections = detections_from_hdf5(str(table), min_z_score=args.min_z)
    detections = detections.reset_index(drop=True)
    detections["x_img"] = detections["x"] + frame.half_width
    detections["y_img"] = detections["y"] + frame.half_width
    seeds = drawn_seeds(constraint)

    print(frame.describe())
    print(f"template {args.stem}: N {geometry.n_protofilaments}, axis offset "
          f"{geometry.axis_offset_magnitude_angstrom:.1f} A")
    print(f"{len(detections):,} detections at z > {args.min_z} "
          f"(max {detections['z_score'].max():.2f}); {len(seeds)} drawn filament(s)\n")

    result = assign_detections_to_filaments(
        detections, geometry, frame.pixel_size, seed_curves_px=seeds,
        position_columns=("x_img", "y_img"))
    old = old_rule_labels(detections, constraint, frame.half_width)
    counts = result.counts()
    print(f"MEMBERSHIP (from the data; converged {result.converged} in "
          f"{result.n_iterations} rounds)")
    print(f"  unassigned: {counts.get(0, 0):,} of {len(detections):,} "
          f"({counts.get(0, 0) / max(len(detections), 1):.1%})")
    moved = (old != result.labels) & (result.labels > 0)
    print(f"  detections the drawn-region rule would have put in a DIFFERENT filament: "
          f"{moved.sum():,} ({moved.sum() / max((result.labels > 0).sum(), 1):.2%})\n")

    print(f"{'fil':<4} {'n':>7} {'length':>8} {'transv':>8} {'tangent':>8} "
          f"{'drawn off':>10}   {'polarity':<22} {'PF':>3} {'margin':>8}")
    for label in sorted(seeds):
        mine = result.labels == label
        if label not in result.curves_px or mine.sum() < 20:
            print(f"{label:<4} {mine.sum():>7}  too few detections for a readout")
            continue
        subset = detections[mine]
        curve = result.curves_px[label]
        length_nm = float(np.hypot(*np.diff(curve, axis=0).T).sum()) * frame.pixel_size / 10
        transverse = float(np.sqrt(np.nanmean(result.transverse_angstrom[mine] ** 2)))
        tangent = float(np.nanmedian(result.tangent_mismatch_deg[mine]))
        drawn_dense = np.vstack([np.linspace(a, b, 20, endpoint=False)
                                 for a, b in zip(seeds[label][:-1], seeds[label][1:])])
        # Only where the drawing runs alongside the fitted curve: the drawn path goes
        # on past the searched area, and those overhangs would be measured to the
        # curve's endpoint and swamp the number.
        distance, nearest = cKDTree(curve).query(drawn_dense)
        alongside = (nearest > 2) & (nearest < len(curve) - 3)
        drawn_off = (float(np.sqrt(np.mean(distance[alongside] ** 2)) * frame.pixel_size)
                     if alongside.any() else float("nan"))
        polarity = estimate_polarity(subset)
        ratio = ("inf" if not np.isfinite(polarity.score_ratio)
                 else f"{polarity.score_ratio:.1f}")
        best, power = estimate_protofilament_number(subset)
        margin, runner = harmonic_margin(power, best)
        print(f"{label:<4} {mine.sum():>7,} {length_nm:6.0f}nm {transverse:6.1f} A "
              f"{tangent:6.1f}° {drawn_off:8.1f} A   psi {polarity.psi_deg:5.1f}° "
              f"{ratio:>5}:1  {best:>3} {margin:6.2f}x (vs {runner})")
    print("\nPOLARITY PER STRETCH (~100 nm along each fitted curve)")
    for label in sorted(seeds):
        mine = result.labels == label
        if label not in result.curves_px or mine.sum() < 20:
            continue
        rows = polarity_by_stretch(detections[mine], result.axis_points_px[mine],
                                   result.curves_px[label], frame.pixel_size)
        parts = []
        for k, n, psi, ratio, off in rows:
            if psi is None:
                parts.append(f"{k}: n={n}, too few")
                continue
            shown = "inf" if not np.isfinite(ratio) else f"{ratio:.0f}"
            parts.append(f"{k}: psi {psi:5.1f} {shown}:1 ({off:.1f}° off curve, n={n})")
        print(f"  filament {label}: " + " | ".join(parts))

    print("\n  transv  = rms distance of its detections' axis points from its fitted curve")
    print("  tangent = median angle between a detection's direction and the curve")
    print("  drawn off = rms distance of the DRAWN path from the fitted curve, where they "
          "run alongside")
    print("  margin  = harmonic power at the winning PF number over the runner-up")

    out_csv = table.parent / f"stage3_membership_{tag}.csv"
    columns = ["x", "y", "x_img", "y_img", "phi", "theta", "psi", "relative_defocus",
               "correlation_value", "correlation_mean", "correlation_variance",
               "z_score", "search_index"]
    frame_out = detections[columns].copy()
    frame_out["filament"] = result.labels
    frame_out["axis_x_img"] = result.axis_points_px[:, 0]
    frame_out["axis_y_img"] = result.axis_points_px[:, 1]
    frame_out["transverse_angstrom"] = result.transverse_angstrom
    frame_out.to_csv(out_csv, index=False)
    out_png = table.parent / f"stage3_membership_{tag}.png"
    micrograph = (None if args.micrograph is None else
                  args.micrograph if args.micrograph.is_absolute()
                  else MT_ROOT / args.micrograph)
    plot(result, seeds, micrograph, out_png, frame)
    print(f"\nwrote {out_csv.relative_to(MT_ROOT)} and {out_png.relative_to(MT_ROOT)}")


if __name__ == "__main__":
    main()
