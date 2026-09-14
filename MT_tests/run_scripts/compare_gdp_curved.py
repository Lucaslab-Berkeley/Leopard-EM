"""Compare the rival templates on the real curved micrograph, paired site by site.

Peak counts and maximum z-scores are poor comparators -- they depend on how many peaks
happened to clear a threshold and they throw away the pairing. Every run here searched
the same micrograph under the same constraint, so the same physical site can be read
from all of them, which removes the site-to-site variation (defocus, ice, near versus
far wall, and here also position along a bend) that otherwise swamps the differences
being asked about.

Three questions, each with its own confound and its own control:

* **protofilament number** -- 12 against 13 against 14 against 15. The confound is
  template MASS, since a 15-PF tube is simply bigger. It is self-diagnosing: a
  monotonic trend with N is mass, a middle-peaked one is structure.
* **lattice spacing** -- expanded against compacted at the winning N. No confound to
  remove: both templates are built from the SAME atoms (6DPU) and differ only in the
  lattice they are tiled on, by construction.
* **ring against patch** -- not a competition but a calibration, since they are
  different sizes and cannot be compared on z directly. What matters is that the patch
  can read a register the ring cannot.

Everything is also reported per drawn region, because the two tubes on this frame are
separate specimens and need not agree -- and if they disagree, that is a result rather
than a problem.

Usage:
    python run_scripts/compare_gdp_curved.py
    python run_scripts/compare_gdp_curved.py --radius 3 --group pf
"""

import argparse
import pathlib
import sys

import h5py
import mrcfile
import numpy as np
import pandas as pd

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from setup_gdp_curved import SEARCHES  # noqa: E402

CLUSTER_RADIUS_PX = 12.0

GROUPS = {
    "pf": ("PROTOFILAMENT NUMBER (same atoms, same lattice, N varies)",
           ["12pf", "13pf", "14pf", "15pf"]),
    "lattice13": ("LATTICE at N = 13 (same 6DPU atoms, lattice varies)",
                  ["13pf", "13pf_expanded"]),
    "lattice14": ("LATTICE at N = 14 (same 6DPU atoms, lattice varies)",
                  ["14pf", "14pf_expanded"]),
    "patch": ("RING against PATCH (a calibration, not a competition)",
              ["13pf", "13pf_patch", "14pf", "14pf_patch"]),
}


def load_run(tag: str, results: pathlib.Path) -> dict | None:
    """Peak table and z-map for one search, or None if it has not been run."""
    csv_path = results / f"results_{tag}.csv"
    mrc_path = results / f"output_scaled_mip_{tag}.mrc"
    if not csv_path.is_file() or not mrc_path.is_file():
        return None
    with mrcfile.open(mrc_path, permissive=True) as handle:
        zmap = np.asarray(handle.data, dtype=np.float32)
    return {"peaks": pd.read_csv(csv_path, index_col=0), "zmap": zmap, "tag": tag}


def union_sites(runs: dict) -> np.ndarray:
    """Cluster every run's peaks into one shared site list, strongest first.

    No single run defines the site set, so no template is handed the advantage of
    choosing where it is judged.
    """
    stacked = np.concatenate(
        [run["peaks"][["pos_y", "pos_x", "scaled_mip"]].to_numpy()
         for run in runs.values()]
    )
    stacked = stacked[np.argsort(-stacked[:, 2])]
    kept: list[np.ndarray] = []
    for row in stacked:
        if all(np.hypot(*(row[:2] - k[:2])) > CLUSTER_RADIUS_PX for k in kept):
            kept.append(row)
    return np.array([k[:2] for k in kept])


def sample(zmap: np.ndarray, sites: np.ndarray, radius: int) -> np.ndarray:
    """Local maximum of a z-map near each site, so a template is never punished for
    landing a pixel away."""
    out = np.empty(len(sites))
    for i, (y, x) in enumerate(sites.astype(int)):
        y0, y1 = max(y - radius, 0), min(y + radius + 1, zmap.shape[0])
        x0, x1 = max(x - radius, 0), min(x + radius + 1, zmap.shape[1])
        out[i] = zmap[y0:y1, x0:x1].max()
    return out


def paired(values: np.ndarray, label: str) -> None:
    """Report a per-site contrast as mean, standard error and sign test."""
    n = len(values)
    if n < 2:
        print(f"  {label:34s} only {n} paired sites")
        return
    mean = values.mean()
    sem = values.std(ddof=1) / np.sqrt(n)
    n_pos = int((values > 0).sum())
    t_stat = mean / sem if sem > 0 else np.inf
    print(f"  {label:34s} {mean:+7.3f} +/- {sem:5.3f} z   t = {t_stat:+7.2f}   "
          f"{n_pos}/{n} sites positive")


def compare(title: str, tags: list[str], runs: dict, sites: np.ndarray,
            radius: int, note: str = "") -> None:
    """One group of templates, ranked and then paired against the leader."""
    available = [t for t in tags if t in runs]
    print(f"\n{'-' * 78}\n{title}\n{'-' * 78}")
    missing = [t for t in tags if t not in runs]
    if missing:
        print(f"  not yet run: {', '.join(missing)}")
    if len(available) < 2:
        print("  need at least two runs to compare")
        return

    z = {tag: sample(runs[tag]["zmap"], sites, radius) for tag in available}
    best = max(z, key=lambda t: z[t].mean())
    print(f"  {'template':<26} {'peaks':>7} {'max z':>8} {'mean z at sites':>16}")
    for tag in available:
        print(f"  {tag:<26} {len(runs[tag]['peaks']):7d} "
              f"{runs[tag]['peaks']['scaled_mip'].max():8.3f} {z[tag].mean():16.3f}")

    print(f"\n  paired against the strongest ({best})")
    for tag in available:
        if tag != best:
            paired(z[best] - z[tag], f"{best} - {tag}")
    ordered = sorted(available, key=lambda t: z[t].mean(), reverse=True)
    print(f"\n  ranking at matched sites: {' > '.join(ordered)}")
    if note:
        print(f"  {note}")


def main() -> None:
    """Rank the templates over the whole frame and within each drawn region."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="gdp_curved")
    parser.add_argument("--results", default="results_gdp_curved")
    parser.add_argument("--constraint", type=pathlib.Path,
                        default=MT_ROOT / "configs"
                        / "filament_constraint_gdp_curved.h5")
    parser.add_argument("--radius", type=int, default=3,
                        help="site search half-width, px")
    parser.add_argument("--group", choices=sorted(GROUPS), default=None,
                        help="only this comparison")
    parser.add_argument("--pooled-only", action="store_true",
                        help="skip the per-region breakdown")
    args = parser.parse_args()

    results = MT_ROOT / args.results
    runs = {}
    for short, _stem, _purpose in SEARCHES:
        run = load_run(f"{args.prefix}_{short}", results)
        if run is not None:
            runs[short] = run
    if not runs:
        raise SystemExit(f"no completed searches under {results}")

    print(f"{len(runs)} of {len(SEARCHES)} searches complete: "
          f"{', '.join(sorted(runs))}")
    sites = union_sites(runs)
    print(f"{len(sites)} paired sites, read at the local maximum within "
          f"{args.radius} px")

    with h5py.File(args.constraint, "r") as handle:
        region_map = handle["maps/region_id"][:]
    y = np.clip(sites[:, 0].astype(int), 0, region_map.shape[0] - 1)
    x = np.clip(sites[:, 1].astype(int), 0, region_map.shape[1] - 1)
    owner = region_map[y, x]
    counts = {int(r): int((owner == r).sum()) for r in np.unique(owner)}
    print(f"sites per region: {counts}")

    wanted = [args.group] if args.group else list(GROUPS)
    notes = {
        "pf": "monotonic in N would be template mass; middle-peaked is structure",
        "lattice13": "no model-quality confound: identical atoms, different lattice",
        "lattice14": "no model-quality confound: identical atoms, different lattice",
        "patch": "different sizes, so read the register test, not this ranking",
    }

    print(f"\n{'=' * 78}\nPOOLED OVER THE WHOLE FRAME\n{'=' * 78}")
    for key in wanted:
        title, tags = GROUPS[key]
        compare(title, tags, runs, sites, args.radius, notes.get(key, ""))

    if args.pooled_only:
        return
    for region in sorted(r for r in counts if r > 0):
        keep = owner == region
        if keep.sum() < 20:
            continue
        print(f"\n{'=' * 78}\nREGION {region} ONLY ({int(keep.sum())} sites)\n"
              f"{'=' * 78}")
        for key in wanted:
            title, tags = GROUPS[key]
            compare(title, tags, runs, sites[keep], args.radius)

    print("\nThe two regions are separate tubes. Agreement between them is evidence;")
    print("disagreement is a result about this specimen, not a failure of the method.")


if __name__ == "__main__":
    main()
