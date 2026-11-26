"""Does the overlap architecture work? Scored on a synthetic whose answer is known.

``simulate_microtubule_bundle.py`` builds 823's layout in Angstroms -- three parallel
tubes 411 A and 463 A apart, a fourth crossing all three at ~29 degrees -- with full
ground truth and hand-drawing-like paths. Three constrained searches ran on it:

    ring_single    the OLD constraint: one psi per pixel, nearest drawn path wins
    ring_layered   the NEW constraint: every covering path kept, gate ORs over them
    patch_layered  the new constraint with the patch, whose box centre is off-axis

Checks, in order of how much they would hurt if wrong:

A1  kernel correctness. Statistics come from the full grid, so the two ring searches
    must agree exactly everywhere except the gate: identical mean/variance maps, the
    single-layer table an exact SUBSET of the layered one, and every extra row on a
    pixel that more than one path covers. Anything else is a bug, not a result.
A2  what the layered gate buys: detections and axis coverage per tube, overall and in
    the crossing zones, single vs layered.
B   membership from the data. ``assign_detections_to_filaments`` seeded from the drawn
    paths, scored against the truth; the old rule (drawn region at the box centre)
    scored the same way; and the fitted curves against both the truth and the drawing.

Truth labels use the TRUE curves with a tight gate (axis point within 30 A, direction
within 10 deg of the true tangent). Detections 30-90 A from every tube are a grey zone
and are left out of the accuracy; those more than 90 A from every tube are clutter.

Usage:
    python run_scripts/validate_overlap_architecture.py
"""

import argparse
import json
import pathlib
import sys

import h5py
import mrcfile
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from leopard_em.analysis.filament_lattice import (
    TemplateLatticeGeometry,
    assign_detections_to_filaments,
    axis_points_from_peaks,
    image_offset_from_template_point,
)
from leopard_em.pydantic_models.results.correlation_table import detections_from_hdf5

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from frame_geometry import legacy_frame_geometry  # noqa: E402

STEM = "synthetic_bundle_823like"
RESULTS = MT_ROOT / "results_bundle"
MODELS = {
    "ring": "6dpu_atoms_6dpv_lattice_13pf_2rings_flatB",
    "patch": "6dpu_atoms_6dpv_lattice_13pf_patch5_2rings_flatB",
}
# All three searches used 600^3 maps at the synthetic's pixel size; read it from the
# table (and its template, since these tables predate recorded geometry), then check
# it against what the simulation says it built.
_FRAME = legacy_frame_geometry(
    RESULTS / "output_correlation_table_bundle_ring_layered.h5", MODELS["ring"])
HALF, PIXEL = _FRAME.half_width, _FRAME.pixel_size
_TRUTH_PIXEL = json.loads(
    (MT_ROOT / "Frames" / f"{STEM}.truth.json").read_text())["pixel_size_angstrom"]
assert abs(_TRUTH_PIXEL - PIXEL) < 1e-6, (_TRUTH_PIXEL, PIXEL)
FRAME_SIZE = int(mrcfile.open(MT_ROOT / "Frames" / f"{STEM}.mrc",
                              header_only=True).header.nx)
TRUTH_GATE_A, TRUTH_TANGENT_DEG, CLUTTER_A = 30.0, 10.0, 90.0
CROSSING_HALF_LENGTH_A = 300.0


def load_truth():
    """True tubes as dense (x, y) micrograph-frame curves, plus the drawn paths."""
    truth = json.loads((MT_ROOT / "Frames" / f"{STEM}.truth.json").read_text())
    tubes = {}
    for tube in truth["tubes"]:
        yx = np.asarray(tube["curve_yx"], dtype=np.float64)
        xy = yx[:, ::-1]
        steps = np.hypot(*np.diff(xy, axis=0).T)
        arc = np.concatenate([[0.0], np.cumsum(steps)])
        fine = np.arange(0.0, arc[-1], 1.0)
        tubes[int(tube["label"])] = np.column_stack(
            [np.interp(fine, arc, xy[:, 0]), np.interp(fine, arc, xy[:, 1])]
        )
    paths = json.loads((MT_ROOT / "Frames" / f"{STEM}.paths.json").read_text())["paths"]
    drawn = {i + 1: np.asarray(p["points"], dtype=np.float64)[:, ::-1]
             for i, p in enumerate(paths)}
    return truth, tubes, drawn


def tangents(curve):
    """Unit tangent along a dense curve."""
    t = np.gradient(curve, axis=0)
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def truth_labels(axis_img, direction_deg, tubes):
    """Label from the TRUE curves: tube id, 0 for clutter, -1 for the grey zone."""
    labels = np.zeros(len(axis_img), dtype=int)
    nearest_any = np.full(len(axis_img), np.inf)
    best = np.full(len(axis_img), np.inf)
    for label, curve in tubes.items():
        distance, index = cKDTree(curve).query(axis_img)
        nearest_any = np.minimum(nearest_any, distance)
        tan = tangents(curve)[index]
        tan_deg = np.degrees(np.arctan2(tan[:, 1], tan[:, 0]))
        mismatch = np.abs(((direction_deg - tan_deg + 90.0) % 180.0) - 90.0)
        ok = (distance * PIXEL <= TRUTH_GATE_A) & (mismatch <= TRUTH_TANGENT_DEG)
        better = ok & (distance < best)
        labels[better] = label
        best[better] = distance[better]
    grey = (labels == 0) & (nearest_any * PIXEL <= CLUTTER_A)
    labels[grey] = -1
    return labels


def crossing_points(tubes):
    """Where tube 4 crosses each of the bundle tubes, micrograph (x, y)."""
    points = {}
    tree = cKDTree(tubes[4])
    for label in (1, 2, 3):
        distance, index = tree.query(tubes[label])
        k = int(np.argmin(distance))
        if distance[k] < 2.0:
            points[label] = tubes[label][k]
    return points


def load_run(tag, min_z):
    """Detections of one search, with micrograph-frame positions added."""
    table = RESULTS / f"output_correlation_table_bundle_{tag}.h5"
    det = detections_from_hdf5(str(table), min_z_score=min_z).reset_index(drop=True)
    det["x_img"] = det["x"] + HALF
    det["y_img"] = det["y"] + HALF
    return det


def directions_deg(det):
    """Each detection's own in-plane axis direction, degrees."""
    d = image_offset_from_template_point(
        np.array([0.0, 0.0, 1.0]), det["phi"].to_numpy(), det["theta"].to_numpy(),
        det["psi"].to_numpy())
    return np.degrees(np.arctan2(d[:, 1], d[:, 0]))


def axis_coverage(axis_img, labels, tubes, zone=None):
    """Fraction of each tube's true axis (41 A steps) with a detection within 41 A."""
    out = {}
    for label, curve in tubes.items():
        samples = curve[:: max(int(41.0 / PIXEL), 1)]
        inside = ((samples[:, 0] >= HALF) & (samples[:, 0] < FRAME_SIZE - HALF)
                  & (samples[:, 1] >= HALF) & (samples[:, 1] < FRAME_SIZE - HALF))
        if zone is not None:
            inside &= zone(samples)
        samples = samples[inside]
        mine = axis_img[labels == label]
        if len(samples) == 0:
            continue
        if len(mine) == 0:
            out[label] = (0.0, len(samples))
            continue
        distance, _ = cKDTree(mine).query(samples)
        out[label] = (float(np.mean(distance * PIXEL <= 41.0)), len(samples))
    return out


def check_kernel(min_z):
    """A1: the two ring searches may differ ONLY by the gate."""
    print("A1  KERNEL: single-layer and layered ring searches must differ only by the gate")
    ok = True
    # Four GPUs pull batches from one queue, so summation order differs between runs
    # and the maps differ in the last bits. Measured in the unit that matters -- the
    # per-pixel SD that turns a correlation into a z-score -- a gate leak into the
    # statistics would be orders of magnitude above that noise.
    sd = mrcfile.read(RESULTS / "output_correlation_variance_bundle_ring_single.mrc")
    for name in ("correlation_average", "correlation_variance"):
        a = mrcfile.read(RESULTS / f"output_{name}_bundle_ring_single.mrc")
        b = mrcfile.read(RESULTS / f"output_{name}_bundle_ring_layered.mrc")
        worst = float(np.max(np.abs(a - b) / sd))
        same = worst < 1e-4
        ok &= same
        print(f"    {name:<22} equal to float noise: {same}  "
              f"(max difference {worst:.1e} SD)")

    keys = {}
    for tag in ("ring_single", "ring_layered"):
        det = detections_from_hdf5(
            str(RESULTS / f"output_correlation_table_bundle_{tag}.h5"), min_z_score=None)
        keys[tag] = set(zip(det["search_index"].to_numpy().tolist(),
                            det["x"].to_numpy().tolist(), det["y"].to_numpy().tolist()))
        print(f"    {tag:<22} {len(det):>10,} rows")
    single, layered = keys["ring_single"], keys["ring_layered"]
    subset = single <= layered
    ok &= subset
    extra = layered - single
    print(f"    single is a subset of layered: {subset}   "
          f"({len(extra):,} rows only in layered)")

    with h5py.File(MT_ROOT / "configs" / "filament_constraint_bundle_layered.h5") as h:
        region_extra = h["maps/region_id_extra"][:]
    covered_twice = (region_extra > 0).any(axis=0)
    if extra:
        ex = np.asarray(list(extra))
        on_overlap = covered_twice[ex[:, 2] + HALF, ex[:, 1] + HALF]
        frac = float(on_overlap.mean())
        ok &= frac == 1.0
        print(f"    extra rows on multiply-covered pixels: {100 * frac:.2f}%")
    print(f"    => {'PASS' if ok else 'FAIL'}\n")
    return ok


def score_gate(tubes, min_z):
    """A2: what the layered gate recovers, per tube and at the crossings."""
    print(f"A2  GATE: ring detections at z >= {min_z}, single vs layered")
    runs = {}
    for tag in ("ring_single", "ring_layered"):
        det = load_run(tag, min_z)
        geometry = TemplateLatticeGeometry.from_pdb(
            str(MT_ROOT / "models" / f"{MODELS['ring']}.pdb"))
        axis = axis_points_from_peaks(det, geometry, PIXEL,
                                      position_columns=("x_img", "y_img"))
        runs[tag] = (det, axis, truth_labels(axis, directions_deg(det), tubes))
    crossings = crossing_points(tubes)
    radius = CROSSING_HALF_LENGTH_A / PIXEL

    def near_crossings(points):
        mask = np.zeros(len(points), dtype=bool)
        for c in crossings.values():
            mask |= np.hypot(*(points - c).T) <= radius
        return mask

    print(f"    crossings inside the search area: tube 4 x tubes "
          f"{sorted(crossings)}; zone = +/-{CROSSING_HALF_LENGTH_A:.0f} A")
    print(f"    {'tube':<6}{'axis covered':>26}{'  in crossing zones':>28}")
    print(f"    {'':<6}{'single':>13}{'layered':>13}{'single':>14}{'layered':>14}")
    cover = {tag: axis_coverage(axis, lab, tubes) for tag, (_, axis, lab) in runs.items()}
    zone = {tag: axis_coverage(axis, lab, tubes, near_crossings)
            for tag, (_, axis, lab) in runs.items()}
    for label in tubes:
        s, l_ = cover["ring_single"].get(label), cover["ring_layered"].get(label)
        zs, zl = zone["ring_single"].get(label), zone["ring_layered"].get(label)
        print(f"    {label:<6}{s[0]:12.1%} {l_[0]:12.1%} "
              f"{(zs[0] if zs else float('nan')):13.1%} {(zl[0] if zl else float('nan')):13.1%}")
    for tag, (det, _, lab) in runs.items():
        print(f"    {tag:<14} rows on a tube {np.sum(lab > 0):>9,}, "
              f"clutter {np.sum(lab == 0):>8,}, grey {np.sum(lab == -1):>8,}")
    print()
    return runs


def score_membership(tag, truth_tubes, drawn, min_z):
    """B: data-driven membership against the truth, and the old box-centre rule."""
    kind = tag.split("_")[0]
    geometry = TemplateLatticeGeometry.from_pdb(
        str(MT_ROOT / "models" / f"{MODELS[kind]}.pdb"))
    det = load_run(tag, min_z)
    seeds = {label: curve for label, curve in drawn.items()}
    result = assign_detections_to_filaments(
        det, geometry, PIXEL, seed_curves_px=seeds,
        position_columns=("x_img", "y_img"))
    labels_true = truth_labels(result.axis_points_px, directions_deg(det), truth_tubes)

    with h5py.File(MT_ROOT / "configs" / "filament_constraint_bundle_layered.h5") as h:
        region = h["maps/region_id"][:]
    bx = np.clip(det["x_img"].to_numpy().astype(int), 0, region.shape[1] - 1)
    by = np.clip(det["y_img"].to_numpy().astype(int), 0, region.shape[0] - 1)
    old = region[by, bx]

    on_tube = labels_true > 0
    clutter = labels_true == 0
    new_acc = float(np.mean(result.labels[on_tube] == labels_true[on_tube]))
    old_acc = float(np.mean(old[on_tube] == labels_true[on_tube]))
    print(f"B   MEMBERSHIP [{tag}]  axis offset "
          f"{np.hypot(*geometry.axis_offset_angstrom):.1f} A, "
          f"{len(det):,} detections at z >= {min_z}, converged {result.converged} "
          f"in {result.n_iterations} rounds")
    print(f"    on-tube detections: {on_tube.sum():,}   clutter: {clutter.sum():,}")
    print(f"    correct tube, data-driven (axis point + tangent):   {new_acc:8.3%}")
    print(f"    correct tube, old rule (drawn region at box centre): {old_acc:8.3%}")
    if clutter.any():
        print(f"    clutter wrongly given a tube: data-driven "
              f"{np.mean(result.labels[clutter] > 0):.2%}, old rule "
              f"{np.mean(old[clutter] > 0):.2%} (old rule labels anything in a band)")
    print(f"    {'tube':<6}{'n':>8}{'data-driven':>14}{'old rule':>11}")
    for label in sorted(truth_tubes):
        mine = labels_true == label
        if mine.any():
            print(f"    {label:<6}{mine.sum():>8,}{np.mean(result.labels[mine] == label):>13.2%}"
                  f"{np.mean(old[mine] == label):>11.2%}")
    confusion = pd.crosstab(pd.Series(labels_true[on_tube], name="truth"),
                            pd.Series(result.labels[on_tube], name="assigned"))
    print("    confusion (on-tube detections):")
    print("      " + confusion.to_string().replace("\n", "\n      "))

    print(f"    {'tube':<6}{'fitted vs truth rms':>21}{'drawn vs truth rms':>21}")
    for label, truth_curve in truth_tubes.items():
        tree = cKDTree(truth_curve)
        fitted = result.curves_px.get(label)
        f_rms = (np.sqrt(np.mean(tree.query(fitted)[0] ** 2)) * PIXEL
                 if fitted is not None else float("nan"))
        drawn_dense = np.vstack([np.linspace(a, b, 20, endpoint=False)
                                 for a, b in zip(drawn[label][:-1], drawn[label][1:])])
        d_rms = np.sqrt(np.mean(tree.query(drawn_dense)[0] ** 2)) * PIXEL
        print(f"    {label:<6}{f_rms:19.1f} A{d_rms:19.1f} A")
    print()
    return {"tag": tag, "new_accuracy": new_acc, "old_accuracy": old_acc,
            "converged": bool(result.converged)}, result, det, labels_true


def plot(result, det, labels_true, drawn, out_path):
    """Detections coloured by assigned tube over the micrograph; misassignments ringed."""
    import matplotlib  # pylint: disable=import-outside-toplevel
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    from scipy.ndimage import gaussian_filter  # pylint: disable=import-outside-toplevel

    image = mrcfile.read(MT_ROOT / "Frames" / f"{STEM}.mrc").astype(np.float32)
    image = gaussian_filter(image, 6) - gaussian_filter(image, 150)
    fig, ax = plt.subplots(figsize=(9, 9))
    lo, hi = np.percentile(image, (1, 99))
    ax.imshow(image, cmap="gray", vmin=lo, vmax=hi)
    colours = {0: "#888888", 1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}
    axis = result.axis_points_px
    for label, colour in colours.items():
        keep = result.labels == label
        ax.scatter(axis[keep, 0], axis[keep, 1], s=1, c=colour, alpha=0.4,
                   label=f"tube {label}" if label else "unassigned")
    wrong = (labels_true > 0) & (result.labels != labels_true)
    ax.scatter(axis[wrong, 0], axis[wrong, 1], s=30, facecolors="none",
               edgecolors="yellow", linewidths=1.0, label=f"wrong tube ({wrong.sum()})")
    for label, curve in drawn.items():
        ax.plot(curve[:, 0], curve[:, 1], "--", color=colours[label], lw=0.8)
    for label, curve in result.curves_px.items():
        ax.plot(curve[:, 0], curve[:, 1], "-", color="white", lw=0.6)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.legend(loc="lower left", markerscale=6, fontsize=8)
    ax.set_title("Membership from the data: dashed = drawn seed, white = fitted curve",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"wrote {out_path}")


def main() -> None:
    """Run all three checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    # 6, not 7: tube 1 (roll 0) scores ~1 z above the others -- a property of the
    # synthetic, not the search grid (every true roll is within 0.1 deg of a grid
    # value) -- so at 7 the sample is almost all tube 1. At 6 every tube has
    # hundreds of detections, and accuracy is reported per tube.
    parser.add_argument("--min-z", type=float, default=6.0)
    args = parser.parse_args()

    _, tubes, drawn = load_truth()
    kernel_ok = check_kernel(args.min_z)
    score_gate(tubes, args.min_z)
    summaries = []
    for tag in ("ring_layered", "patch_layered"):
        summary, result, det, labels_true = score_membership(tag, tubes, drawn, args.min_z)
        summaries.append(summary)
        plot(result, det, labels_true, drawn, RESULTS / f"membership_{tag}.png")
    (RESULTS / "validation_summary.json").write_text(
        json.dumps({"kernel_ok": kernel_ok, "membership": summaries}, indent=2))


if __name__ == "__main__":
    main()
