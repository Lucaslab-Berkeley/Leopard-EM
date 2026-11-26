"""Score automatic filament tracks against reference paths, as numbers.

usage: python run_scripts/score_autotrace.py --auto A.paths.json [B.paths.json ...]
           --reference R.paths.json | --truth Frames/<synthetic>.truth.json
           --pixel 0.9194 --shape H W [--edge-margin-nm 37] [--match-angstrom 150]

Same matching rule as ``auto_trace_filaments.compare``: a reference point is covered when
an automatic track passes within ``--match-angstrom`` of it, within 15 deg of its
direction. References are clipped to the region a template search can score (more than
``--edge-margin-nm`` from every edge), so tracers with different boxes are compared on the
same ground.

Real micrographs are NOT fully annotated: tracks matching no reference are reported as
"extra" (length and count), not as false positives. Only a synthetic's --truth lists
every tube, so only there is extra = false.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "programs/constrained_search"))
from napari_choose_filament_path import load_paths_json, spline_through_points  # noqa: E402


def clip(points: np.ndarray, shape, margin: float) -> list[np.ndarray]:
    """Runs of a dense curve inside the margin box."""
    inside = ((points[:, 0] >= margin) & (points[:, 0] <= shape[0] - 1 - margin)
              & (points[:, 1] >= margin) & (points[:, 1] <= shape[1] - 1 - margin))
    runs, start = [], None
    for i, ok in enumerate(np.append(inside, False)):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= 2:
                runs.append(points[start:i])
            start = None
    return runs


def dense(points, step=4.0) -> np.ndarray:
    pts = np.asarray(points, float)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, s[-1], step)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], 1)


def score(auto_paths, references, pixel, shape, margin_px, match_a, tol_deg=15.0):
    curves = [spline_through_points(p["points"], smooth_px=5.0) for p in auto_paths]
    samples = np.vstack([c[0] for c in curves]) if curves else np.empty((0, 2))
    tangents = np.vstack([c[1] for c in curves]) if curves else np.empty((0, 2))
    ids = (np.concatenate([np.full(len(c[0]), k) for k, c in enumerate(curves)])
           if curves else np.empty(0, int))
    tree = cKDTree(samples) if len(samples) else None
    limit, cos_tol = match_a / pixel, np.cos(np.radians(tol_deg))
    per_ref, matched = [], set()
    for k, ref in enumerate(references, start=1):
        for run in clip(dense(ref["points"]), shape, margin_px):
            if len(run) < 4:
                continue
            tan = np.gradient(run, axis=0)
            tan /= np.linalg.norm(tan, axis=1, keepdims=True)
            length_nm = float(np.sum(np.hypot(*np.diff(run, axis=0).T))) * pixel / 10
            if tree is None:
                per_ref.append({"ref": k, "length_nm": length_nm, "covered": 0.0,
                                "median_off_A": np.nan, "tracks": []})
                continue
            dist, near = tree.query(run)
            cos = np.abs(np.sum(tangents[near] * tan, axis=1))
            cov = (dist <= limit) & (cos >= cos_tol)
            who = sorted({int(ids[i]) + 1 for i in near[cov]})
            matched.update(w - 1 for w in who)
            per_ref.append({"ref": k, "length_nm": length_nm, "covered": float(cov.mean()),
                            "median_off_A": float(np.median(dist[cov]) * pixel) if cov.any() else np.nan,
                            "tracks": who})
    lengths = [float(np.sum(np.hypot(*np.diff(np.asarray(p["points"], float), axis=0).T))) * pixel / 10
               for p in auto_paths]
    extra = [k for k in range(len(auto_paths)) if k not in matched]
    total_ref = sum(r["length_nm"] for r in per_ref)
    return {
        "coverage": sum(r["covered"] * r["length_nm"] for r in per_ref) / total_ref if total_ref else np.nan,
        "ref_length_nm": total_ref, "n_tracks": len(auto_paths), "n_matched": len(matched),
        "n_extra": len(extra), "extra_length_nm": float(sum(lengths[k] for k in extra)),
        "matched_length_nm": float(sum(lengths[k] for k in matched)), "per_ref": per_ref,
    }


def truth_paths(truth_json: pathlib.Path) -> list[dict]:
    t = json.loads(truth_json.read_text())
    if "tubes" in t:
        return [{"points": tb["curve_yx"]} for tb in t["tubes"]]
    raise ValueError(f"{truth_json}: no 'tubes' list")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", type=pathlib.Path, nargs="+", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--reference", type=pathlib.Path)
    g.add_argument("--truth", type=pathlib.Path)
    ap.add_argument("--pixel", type=float, required=True)
    ap.add_argument("--shape", type=int, nargs=2, required=True, metavar=("H", "W"))
    ap.add_argument("--edge-margin-nm", type=float, default=37.0)
    ap.add_argument("--match-angstrom", type=float, default=150.0)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()
    refs = truth_paths(a.truth) if a.truth else load_paths_json(str(a.reference))[1]
    margin = a.edge_margin_nm * 10 / a.pixel
    kind = "false (full truth)" if a.truth else "extra (not annotated -- not necessarily wrong)"
    out = {}
    for path in a.auto:
        _, auto = load_paths_json(str(path))
        s = score(auto, refs, a.pixel, a.shape, margin, a.match_angstrom)
        out[str(path)] = s
        print(f"{path}: coverage {s['coverage']:.0%} of {s['ref_length_nm']:.0f} nm reference; "
              f"{s['n_matched']}/{s['n_tracks']} tracks on a reference; {s['n_extra']} {kind}, "
              f"{s['extra_length_nm']:.0f} nm")
        for r in s["per_ref"]:
            print(f"    ref {r['ref']}: {r['length_nm']:5.0f} nm, covered {r['covered']:4.0%}, "
                  f"median off {r['median_off_A']:5.0f} A, tracks {r['tracks']}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
