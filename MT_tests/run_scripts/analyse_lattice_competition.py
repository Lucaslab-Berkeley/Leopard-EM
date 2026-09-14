"""Compare the 2x2 lattice-competition runs, paired site by site.

Raw peak counts and maximum z-scores are poor comparators: they depend on how many
peaks happened to clear a threshold, and they throw away the pairing. Every run here
searched the same micrograph, so the same physical site can be compared across all
four templates, which removes site-to-site variation (defocus, ice thickness, near
versus far wall) that otherwise swamps a sub-percent lattice difference.

Sites are the union of the four peak lists, so no single run defines the site set, and
each map is read at its own local maximum near the site so a template is never punished
for landing a pixel away.

The 2x2 then separates into two orthogonal contrasts per site:

    lattice effect  = mean over atom source of (expanded - compacted)   <- the question
    quality effect  = mean over lattice of (6dpu atoms - 6dpv atoms)    <- the nuisance
    interaction     = whether those two are separable at all

Usage:  python run_scripts/analyse_lattice_competition.py [--native-b] [--radius 3]
"""

import argparse
import itertools
import pathlib

import mrcfile
import numpy as np
import pandas as pd

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULT_DIR = MT_ROOT / "results_cropped"

CELLS = list(itertools.product(("6dpu", "6dpv"), repeat=2))  # (atoms, lattice)
CLUSTER_RADIUS_PX = 12.0


def cell_tag(atoms: str, lattice: str, suffix: str) -> str:
    return f"{atoms}_atoms_{lattice}_lattice_2rings{suffix}_crop"


def load_runs(suffix: str) -> dict[tuple[str, str], dict]:
    """Peak table and z-score map for each cell of the 2x2."""
    runs = {}
    for atoms, lattice in CELLS:
        tag = cell_tag(atoms, lattice, suffix)
        csv_path = RESULT_DIR / f"results_{tag}.csv"
        mrc_path = RESULT_DIR / f"output_scaled_mip_{tag}.mrc"
        if not csv_path.is_file() or not mrc_path.is_file():
            raise FileNotFoundError(f"missing results for {tag}")
        with mrcfile.open(mrc_path, permissive=True) as handle:
            zmap = np.asarray(handle.data, dtype=np.float32)
        runs[(atoms, lattice)] = {
            "peaks": pd.read_csv(csv_path, index_col=0),
            "zmap": zmap,
            "tag": tag,
        }
    return runs


def union_sites(runs: dict) -> np.ndarray:
    """Cluster every run's peaks into one shared site list, strongest first."""
    stacked = np.concatenate(
        [
            run["peaks"][["pos_y", "pos_x", "scaled_mip"]].to_numpy()
            for run in runs.values()
        ]
    )
    stacked = stacked[np.argsort(-stacked[:, 2])]

    kept: list[np.ndarray] = []
    for row in stacked:
        if all(np.hypot(*(row[:2] - k[:2])) > CLUSTER_RADIUS_PX for k in kept):
            kept.append(row)
    return np.array([k[:2] for k in kept])


def sample(zmap: np.ndarray, sites: np.ndarray, radius: int) -> np.ndarray:
    """Local maximum of a z-map near each site."""
    out = np.empty(len(sites))
    for i, (y, x) in enumerate(sites.astype(int)):
        y0, y1 = max(y - radius, 0), min(y + radius + 1, zmap.shape[0])
        x0, x1 = max(x - radius, 0), min(x + radius + 1, zmap.shape[1])
        out[i] = zmap[y0:y1, x0:x1].max()
    return out


def paired(values: np.ndarray, label: str) -> None:
    """Report a per-site contrast as mean, standard error and sign test."""
    n = len(values)
    mean = values.mean()
    sem = values.std(ddof=1) / np.sqrt(n)
    n_pos = int((values > 0).sum())
    t_stat = mean / sem if sem > 0 else np.inf
    print(
        f"  {label:34s} {mean:+7.3f} +/- {sem:5.3f} z   "
        f"t = {t_stat:+7.2f}   {n_pos}/{n} sites positive"
    )


def load_stems(stems: list[str]) -> dict[str, dict]:
    """Peak table and z-map for an arbitrary set of templates."""
    runs = {}
    for stem in stems:
        tag = f"{stem}_crop"
        with mrcfile.open(RESULT_DIR / f"output_scaled_mip_{tag}.mrc", permissive=True) as h:
            zmap = np.asarray(h.data, dtype=np.float32)
        runs[stem] = {
            "peaks": pd.read_csv(RESULT_DIR / f"results_{tag}.csv", index_col=0),
            "zmap": zmap,
            "tag": tag,
        }
    return runs


def compare_stems(stems: list[str], radius: int) -> None:
    """Paired per-site comparison across templates, all against the best one.

    No 2x2 decomposition here: templates that differ only in protofilament number are
    built from the same atoms with the same B-factors, so there is no model-quality
    confound to separate out. The one residual is template mass, and it is
    self-diagnosing -- a monotonic trend with N means mass, a middle-peaked one means
    structure.
    """
    runs = load_stems(stems)
    sites = union_sites(runs)
    z = {stem: sample(run["zmap"], sites, radius) for stem, run in runs.items()}
    best = max(z, key=lambda s: z[s].mean())

    print(f"paired sites: {len(sites)}\n")
    print(f"  {'template':34s} {'peaks':>6s} {'max z':>7s} {'mean z':>8s} {'atoms':>9s}")
    for stem in stems:
        print(f"  {stem:34s} {len(runs[stem]['peaks']):6d} "
              f"{runs[stem]['peaks']['scaled_mip'].max():7.3f} {z[stem].mean():8.3f}")

    print(f"\nPaired against the strongest ({best})")
    for stem in stems:
        if stem == best:
            continue
        paired(z[best] - z[stem], f"{best.split('_')[1]} - {stem.split('_')[1]}")

    ordered = sorted(stems, key=lambda s: z[s].mean(), reverse=True)
    print(f"\nRanking by mean z at matched sites: "
          f"{' > '.join(s.split('_')[1] for s in ordered)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-b", action="store_true")
    parser.add_argument("--radius", type=int, default=3, help="site search half-width px")
    parser.add_argument(
        "--stems",
        nargs="+",
        metavar="STEM",
        help="compare these templates pairwise instead of the expanded/compacted 2x2",
    )
    args = parser.parse_args()
    suffix = "" if args.native_b else "_flatB"

    if args.stems:
        compare_stems(args.stems, args.radius)
        return

    runs = load_runs(suffix)
    sites = union_sites(runs)
    z = {cell: sample(run["zmap"], sites, args.radius) for cell, run in runs.items()}

    print(f"B-factors: {'native' if args.native_b else 'equalised'}   "
          f"paired sites: {len(sites)}\n")

    print("Per-run summary (not the comparator -- see the paired contrasts below)")
    print(f"  {'cell':34s} {'peaks':>6s} {'max z':>7s} {'mean z at sites':>16s}")
    for cell in CELLS:
        name = f"{cell[0]} atoms @ {cell[1]} lattice"
        print(
            f"  {name:34s} {len(runs[cell]['peaks']):6d} "
            f"{runs[cell]['peaks']['scaled_mip'].max():7.3f} {z[cell].mean():16.3f}"
        )

    exp, comp = "6dpu", "6dpv"  # expanded / compacted lattice
    lattice = 0.5 * ((z[("6dpu", exp)] - z[("6dpu", comp)])
                     + (z[("6dpv", exp)] - z[("6dpv", comp)]))
    quality = 0.5 * ((z[("6dpu", exp)] - z[("6dpv", exp)])
                     + (z[("6dpu", comp)] - z[("6dpv", comp)]))
    interaction = ((z[("6dpu", exp)] - z[("6dpu", comp)])
                   - (z[("6dpv", exp)] - z[("6dpv", comp)]))

    print("\nPaired contrasts (positive = first term wins)")
    paired(lattice, "LATTICE  expanded - compacted")
    paired(quality, "quality  6dpu atoms - 6dpv atoms")
    paired(interaction, "interaction")
    print("\nNative cells only, for reference")
    paired(z[("6dpu", "6dpu")] - z[("6dpv", "6dpv")], "6DPU native - 6DPV native")

    verdict = "EXPANDED (6DPU)" if lattice.mean() > 0 else "COMPACTED (6DPV)"
    ratio = abs(lattice.mean()) / (abs(quality.mean()) + 1e-9)
    trustworthy = ratio >= 2.0
    print(f"\nVerdict: {verdict}, by {ratio:.1f}x the model-quality effect.")
    print(
        "  The lattice effect dominates, so the result is NOT an artefact of one\n"
        "  model simply being better."
        if trustworthy
        else "  Model quality is comparable to the lattice effect, so this result is\n"
        "  NOT trustworthy whichever way it points -- treat it as inconclusive."
    )


if __name__ == "__main__":
    main()
