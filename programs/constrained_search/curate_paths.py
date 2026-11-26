"""Curate automatic filament tracks: keep, join, extend -- the semi-manual step.

``auto_trace_filaments.py`` proposes tracks; a person looks at its overlay and says
which are microtubules. This turns that verdict into the final paths without redrawing:

* ``--keep 11 7 15 13``      keep these tracks (numbers as on the overlay), drop the rest;
* ``--join 9+12``            one microtubule traced in pieces: merge them into one path
                             (vertices ordered along the common direction, re-smoothed);
* ``--part 3:0:0.5``         keep only a stretch of a track (fractions of its length from
                             its left end) -- one that runs onto another filament;
* ``--extend-to-edge``       carry each kept path straight on to the micrograph's edge
                             (tubes that cross the whole frame), along a line fitted to
                             the last 150 nm of each end.

Writes ``<output>.paths.json`` (the drawing tool's format: reopen it with
``napari_choose_filament_path.py --paths`` for any further correction), an overlay PNG,
and ``<output>.curation.json`` recording the source and every choice -- so each curated
micrograph is a fully annotated training example, with its provenance.

Usage:
    python programs/constrained_search/curate_paths.py \\
        MT_tests/results_autotrace/kinesin_2_0031.paths.json \\
        --micrograph MT_tests/results_kinesin/20260916_lysates_2_0031_DW.mrc \\
        --keep 11 7 13 15 --join 9+12 --extend-to-edge \\
        -o MT_tests/configs/filament_constraint_kinesin_2_0031
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import mrcfile
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from napari_choose_filament_path import (  # noqa: E402
    load_paths_json,
    save_paths_json,
    spline_through_points,
)


def resample(curve: np.ndarray, spacing: float) -> np.ndarray:
    """Points along ``curve`` every ``spacing`` px of arc, both ends included."""
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(curve, axis=0).T))])
    targets = np.append(np.arange(0.0, arc[-1], spacing), arc[-1])
    return np.column_stack([np.interp(targets, arc, curve[:, k]) for k in (0, 1)])


def keep_part(points: np.ndarray, start: float, end: float) -> np.ndarray:
    """The stretch of a track from fraction ``start`` to ``end`` of its length.

    Fractions run from the track's LEFT end (smaller x), whichever way it was traced --
    for a track that runs onto another filament at one end.
    """
    if points[0, 1] > points[-1, 1]:
        points = points[::-1]
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    targets = np.linspace(start * arc[-1], end * arc[-1], max(2, len(points)))
    return np.column_stack([np.interp(targets, arc, points[:, k]) for k in (0, 1)])


def join_fragments(fragments: list[np.ndarray]) -> np.ndarray:
    """One path through several fragments of one tube, ordered along it."""
    points = np.vstack(fragments)
    centre = points.mean(axis=0)
    direction = np.linalg.svd(points - centre)[2][0]
    return points[np.argsort((points - centre) @ direction)]


def distance_to_edge(start: np.ndarray, direction: np.ndarray, shape) -> float:
    """How far a ray from ``start`` along ``direction`` runs before leaving the frame."""
    limits = []
    for axis in (0, 1):
        if direction[axis] > 1e-12:
            limits.append((shape[axis] - 1 - start[axis]) / direction[axis])
        elif direction[axis] < -1e-12:
            limits.append(-start[axis] / direction[axis])
    return max(0.0, min(limits)) if limits else 0.0


def end_direction(curve: np.ndarray, at_start: bool, span: float) -> np.ndarray:
    """Outward direction at one end: a line fitted to the last ``span`` px of the curve.

    A spline's own end tangent is the noisiest point on it, and an extension multiplies
    any error by its length (a short track carried ~4 um across the frame), so the
    direction comes from the whole stretch near the end -- all of it for a short track.
    """
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(curve, axis=0).T))])
    near = arc <= span if at_start else arc >= arc[-1] - span
    points = curve[near] if near.sum() >= 3 else curve
    centre = points.mean(axis=0)
    direction = np.linalg.svd(points - centre)[2][0]
    outward = (curve[0] - curve[-1]) if at_start else (curve[-1] - curve[0])
    return direction * np.sign(direction @ outward or 1.0)


def extend_to_edge(curve: np.ndarray, shape, span: float) -> np.ndarray:
    """Carry a smooth curve straight on from both ends to the frame edge."""
    pieces = []
    for end, at_start in ((0, True), (-1, False)):
        outward = end_direction(curve, at_start, span)
        reach = distance_to_edge(curve[end], outward, shape)
        steps = np.linspace(0.0, reach, max(2, int(reach // 20) + 2))[1:]
        pieces.append(curve[end] + np.outer(steps, outward))
    return np.vstack([pieces[0][::-1], curve, pieces[1]])


def overlay(micrograph: np.ndarray, paths: list[dict], labels: list[str], out_path, title):
    """The curated paths, numbered, on a binned and flattened micrograph."""
    import matplotlib  # pylint: disable=import-outside-toplevel
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    from scipy.ndimage import gaussian_filter  # pylint: disable=import-outside-toplevel

    binning = 8
    h, w = (micrograph.shape[0] // binning * binning, micrograph.shape[1] // binning * binning)
    small = micrograph[:h, :w].reshape(h // binning, binning, w // binning, binning).mean((1, 3))
    small = gaussian_filter(small, 1.0) - gaussian_filter(small, 40)
    low, high = np.percentile(small, [1, 99])
    fig, ax = plt.subplots(figsize=(10, 10 * small.shape[0] / small.shape[1] + 0.4))
    ax.imshow(small, cmap="gray", vmin=low, vmax=high)
    colours = plt.get_cmap("tab10")
    for k, (path, label) in enumerate(zip(paths, labels)):
        points = np.asarray(path["points"]) / binning
        ax.plot(points[:, 1], points[:, 0], color=colours(k % 10), lw=2.2)
        mid = points[len(points) // 2]
        ax.text(mid[1], mid[0], f"{k + 1}", color="white", fontsize=10, weight="bold",
                ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.2", "fc": colours(k % 10), "ec": "none"})
        ax.text(points[0][1], points[0][0], f" {label}", color=colours(k % 10),
                fontsize=8, va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)


def main() -> None:
    """Keep, join and extend tracks; write paths.json, overlay and record."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths_json", type=pathlib.Path,
                        help="the automatic tracks (auto_trace_filaments.py output)")
    parser.add_argument("--micrograph", type=pathlib.Path, required=True)
    parser.add_argument("--keep", type=int, nargs="*", default=[],
                        help="track numbers to keep, as on the overlay")
    parser.add_argument("--join", nargs="*", default=[],
                        help="fragments of one tube, e.g. 9+12: merged into one path "
                             "(implies keeping them)")
    parser.add_argument("--part", nargs="*", default=[], metavar="TRACK:FROM:TO",
                        help="keep only this stretch of a kept track, as fractions of its "
                             "length from its left end, e.g. 3:0:0.5 (the left half)")
    parser.add_argument("--extend-to-edge", action="store_true",
                        help="carry each path straight on to the frame edge")
    parser.add_argument("--vertex-spacing-nm", type=float, default=50.0)
    parser.add_argument("--pixel-size", type=float, required=True,
                        help="A/px of the micrograph (for the vertex spacing)")
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True,
                        help="output stem: writes <stem>.paths.json, <stem>_curated.png "
                             "and <stem>.curation.json")
    args = parser.parse_args()

    _, tracks = load_paths_json(str(args.paths_json))
    with mrcfile.open(str(args.micrograph), permissive=True) as handle:
        micrograph = np.asarray(handle.data, dtype=np.float32)
    shape = micrograph.shape
    spacing = args.vertex_spacing_nm * 10.0 / args.pixel_size

    parts = {int(t): (float(a), float(b))
             for t, a, b in (item.split(":") for item in args.part)}
    groups = [[int(n) for n in group.split("+")] for group in args.join]
    joined = {n for group in groups for n in group}
    singles = [[n] for n in args.keep if n not in joined]
    paths, labels = [], []
    for group in groups + singles:
        for n in group:
            if not 1 <= n <= len(tracks):
                raise SystemExit(f"track {n} does not exist (1-{len(tracks)})")
        fragments = [keep_part(np.asarray(tracks[n - 1]["points"], dtype=float), *parts[n])
                     if n in parts else np.asarray(tracks[n - 1]["points"], dtype=float)
                     for n in group]
        points = join_fragments(fragments) if len(group) > 1 else fragments[0]
        curve, _ = spline_through_points(points, smooth_px=10.0)
        if args.extend_to_edge:
            curve = extend_to_edge(curve, shape, 1500.0 / args.pixel_size)
        paths.append({"points": resample(curve, spacing),
                      "width_px": float(tracks[group[0] - 1]["width_px"]),
                      "polarity": "both"})
        labels.append("+".join(str(n) for n in group))

    out_json = pathlib.Path(f"{args.output}.paths.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    save_paths_json(str(out_json), str(args.micrograph.resolve()), paths)
    title = (f"{args.micrograph.name}: {len(paths)} curated path(s) from "
             f"{args.paths_json.name}" + (" , extended to the edge" if args.extend_to_edge
                                           else ""))
    overlay(micrograph, paths, labels, pathlib.Path(f"{args.output}_curated.png"), title)
    record = {
        "source_tracks": str(args.paths_json.resolve()),
        "micrograph": str(args.micrograph.resolve()),
        "kept": args.keep, "joined": args.join, "parts": args.part,
        "extended_to_edge": args.extend_to_edge,
        "dropped": [n for n in range(1, len(tracks) + 1)
                    if n not in joined and n not in args.keep],
        "paths": [{"path": k + 1, "from_tracks": label,
                   "length_nm": float(np.sum(np.hypot(*np.diff(p["points"], axis=0).T))
                                      * args.pixel_size / 10)}
                  for k, (p, label) in enumerate(zip(paths, labels))],
    }
    pathlib.Path(f"{args.output}.curation.json").write_text(json.dumps(record, indent=2))
    for item in record["paths"]:
        print(f"path {item['path']}: from track(s) {item['from_tracks']}, "
              f"{item['length_nm']:.0f} nm")
    print(f"dropped: {record['dropped']}")
    print(f"wrote {out_json}, {args.output}_curated.png, {args.output}.curation.json")


if __name__ == "__main__":
    main()
