"""Stage 6's protofilament-number competition on the SVD scorer instead of refine_template.

``compete_sites_curved.py`` refines every rival ring at every Stage 5 site from 6-8 roll
seeds (a rival of another N has its own roll): ~35 refines per site, ~40 min on 4 GPUs
for in situ 823. The SVD scorer (``programs/constrained_search/svd_score_sites.py``,
Panther-EM environment) evaluates every roll at once, so each rival costs one
decomposition and seconds of scoring. Per site and rival: the maximum z over the rival's
own roll period (phi within +/-180/N of the site's refined phi), +/-``--angle-window`` in
theta and psi, +/-``--radius-px`` in position and the defocus planes within +/-200 A --
the same window for every rival but its roll, which is what N sets.

For the PROTOFILAMENT NUMBER only. Lattice spacing (13pf vs 13pf_exp, or 14pf vs 14pf_cmp
for GMPCPP; Stages 6/6b) and the register (Stage 7) stay on refine_template, where the
full sensitivity is needed. The other-lattice ring is scored here only with
``--with-lattice``, as a cross-check against refine, not to decide.

Reports exactly as Stage 6 does (``compete_sites_curved.report``: ranking, paired
contrasts against the winner, mean z against N) per region and pooled, and -- when the
refine Stage 6 summary of the same tag exists -- the two side by side.

Usage:
    python run_scripts/stage6_svd.py --table results_svd/output_correlation_table_<tag>.h5
"""

import argparse
import os
import pathlib
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from leopard_em.analysis.filament_lattice import TemplateLatticeGeometry

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from compete_sites_curved import DEFAULT_KEYS, LATTICE_PAIRS, RIVAL_SETS, report  # noqa: E402
from frame_geometry import map_for  # noqa: E402

PANTHER_ENV = os.environ.get("PANTHER_ENV",
                             str(pathlib.Path(__file__).resolve().parent / "panther_env.sh"))
SCORER = MT_ROOT.parent / "programs" / "constrained_search" / "svd_score_sites.py"


def tube_directions(stack: pd.DataFrame, sites: pd.DataFrame, half: int = 3):
    """Unit tube direction (x, y) at every site: the principal axis of its own filament's
    sites within +/-``half`` places along the tube (ordered by axial index)."""
    ux, uy = np.zeros(len(sites)), np.zeros(len(sites))
    xy = sites[["pos_x", "pos_y"]].to_numpy(float)
    label = stack["filament"].to_numpy() if "filament" in stack else stack["region"].to_numpy()
    order_key = (stack["axial_index"].to_numpy() if "axial_index" in stack else None)
    for value in np.unique(label):
        idx = np.flatnonzero(label == value)
        if order_key is not None:
            idx = idx[np.argsort(order_key[idx], kind="stable")]
        else:  # along the filament's own principal axis
            p = xy[idx] - xy[idx].mean(axis=0)
            idx = idx[np.argsort(p @ np.linalg.svd(p, full_matrices=False)[2][0])]
        for j, i in enumerate(idx):
            near = xy[idx[max(0, j - half): j + half + 1]]
            u = np.linalg.svd(near - near.mean(axis=0), full_matrices=False)[2][0]
            ux[i], uy[i] = u
    return ux, uy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=pathlib.Path, required=True)
    parser.add_argument("--rivals", choices=sorted(RIVAL_SETS), default="compacted13")
    parser.add_argument("--search-config", type=pathlib.Path, default=None,
                        help="default configs/match_tm_<tag>.yaml")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0],
                        help="the rivals are shared out over these, one scorer per GPU")
    parser.add_argument("--with-lattice", action="store_true",
                        help="also score the other-lattice ring (13pf_exp, or 14pf_cmp for "
                             "expanded14) -- a cross-check only: the lattice is decided by "
                             "refine (Stage 6b)")
    parser.add_argument("--along-px", type=float, default=8.0,
                        help="position window along the tube (refine's winning rival "
                             "moved |along| <= 6 px at the 90th percentile on 823)")
    parser.add_argument("--across-px", type=float, default=20.0,
                        help="and across it: a ring of another N lines up with one wall "
                             "of the tube, 6-11 px off the axis (refine allows +/-20)")
    parser.add_argument("--angle-window", type=float, default=3.0,
                        help="+/- degrees in theta and psi (the roll gets half a PF period)")
    parser.add_argument("--rescore", action="store_true",
                        help="score again even if the scores file exists")
    args = parser.parse_args()

    table = args.table if args.table.is_absolute() else MT_ROOT / args.table
    tag = table.stem.removeprefix("output_correlation_table_")
    results = table.parent
    config = args.search_config or MT_ROOT / "configs" / f"match_tm_{tag}.yaml"
    stack = pd.read_csv(results / f"stack_sites_{tag}.csv", index_col=0)
    refined = pd.read_csv(results / f"refined_sites_{tag}.csv", index_col=0)
    pixel_size = float(stack["pixel_size"].iloc[0])

    # The sites at their REFINED poses, as compete_sites_curved starts every rival.
    sites = pd.DataFrame({
        "pos_x": np.rint(refined["refined_pos_x"].to_numpy()).astype(int),
        "pos_y": np.rint(refined["refined_pos_y"].to_numpy()).astype(int),
        "phi": refined["refined_phi"].to_numpy(),
        "theta": refined["refined_theta"].to_numpy(),
        "psi": refined["refined_psi"].to_numpy(),
        "relative_defocus": refined["refined_relative_defocus"].to_numpy(),
        "region": stack["region"].to_numpy(), "segment": stack["segment"].to_numpy(),
    }, index=stack.index)
    sites["axis_x"], sites["axis_y"] = tube_directions(stack, sites)
    sites_path = results / f"compete_svd_sites_{tag}.csv"
    scores_path = results / f"compete_svd_scores_{tag}.csv"
    sites.to_csv(sites_path)

    # the lattice pair's member that is not a plain N-PF ring (13pf_exp; 14pf_cmp) -- NOT
    # simply the pair's second entry, which for expanded14 is the 14-PF reference itself
    lattice_rival = next(k for k in LATTICE_PAIRS[args.rivals][:2]
                         if not re.fullmatch(r"\d+pf", k))
    keys, specs = [], {}
    for key, stem, _ in RIVAL_SETS[args.rivals]:
        if key not in DEFAULT_KEYS[args.rivals]:
            continue
        if key == lattice_rival and not args.with_lattice:
            continue
        template = map_for(stem, pixel_size)
        if not template.is_file():
            print(f"  {key}: no map at {template.name}, skipped")
            continue
        n = TemplateLatticeGeometry.from_pdb(str(MT_ROOT / "models" / f"{stem}.pdb")
                                             ).n_protofilaments
        keys.append(key)
        specs[key] = f"--template {key}={template} --phi-window {key}={180.0 / n}"
    print(f"{len(sites)} sites from {tag} ({pixel_size} A/px); rivals {', '.join(keys)}; "
          f"window +/-{args.along_px} px along x +/-{args.across_px} px across the tube, "
          f"+/-{args.angle_window} deg theta/psi, roll +/-180/N")
    start = time.time()
    if args.rescore or not scores_path.is_file():
        # one scorer per GPU, the rivals dealt out round-robin, run side by side
        groups = [keys[i::len(args.gpus)] for i in range(len(args.gpus))]
        jobs = []
        for gpu, group in zip(args.gpus, groups):
            if not group:
                continue
            part = scores_path.with_name(f"{scores_path.stem}_gpu{gpu}.csv")
            shell = (f"source {PANTHER_ENV} && CUDA_VISIBLE_DEVICES={gpu} python {SCORER} "
                     f"--config {config} --sites {sites_path} --out {part} "
                     f"--along-px {args.along_px} --across-px {args.across_px} "
                     f"--angle-window {args.angle_window} "
                     + " ".join(specs[k] for k in group))
            log = MT_ROOT / "logs" / f"stage6_svd_{tag}_scorer_gpu{gpu}.log"
            handle = open(log, "w", encoding="utf-8")  # pylint: disable=consider-using-with
            jobs.append((subprocess.Popen(["bash", "-c", shell], cwd=MT_ROOT, stdout=handle,
                                          stderr=subprocess.STDOUT), handle, part, log))
        parts = []
        for process, handle, part, log in jobs:
            if process.wait() != 0:
                raise SystemExit(f"scorer failed: see {log.relative_to(MT_ROOT)}")
            handle.close()
            parts.append(pd.read_csv(part, index_col=0))
        merged = parts[0]
        for part in parts[1:]:
            merged = merged.join(part[[c for c in part.columns if c not in merged.columns]])
        merged.to_csv(scores_path)
        print(f"scored in {time.time() - start:.0f} s on GPU(s) "
              f"{' '.join(str(g) for g in args.gpus)} (logs logs/stage6_svd_{tag}_scorer_gpu*.log)")
    scored = pd.read_csv(scores_path, index_col=0)
    keys = [k for k in keys if f"{k}_z_max" in scored.columns]
    scores = {k: scored[f"{k}_z_max"].to_numpy() for k in keys}
    for key in keys:
        dphi = (scored[f"{key}_max_phi"] - scored["phi"] + 180.0) % 360.0 - 180.0
        print(f"  {key:<9} mean z {scores[key].mean():6.3f}   median |roll moved| "
              f"{np.median(np.abs(dphi)):4.1f} deg   hypotheses/site "
              f"{int(np.median(scored[f'{key}_n_window'])):,}")

    region = sites["region"].to_numpy()
    for value in sorted(set(region.tolist())):
        mask = region == value
        if mask.sum() >= 8:
            report({k: v[mask] for k, v in scores.items()}, f"SVD: REGION {value}",
                   int(mask.sum()), LATTICE_PAIRS[args.rivals])
    winner = report(scores, "SVD: ALL POOLED", len(sites), LATTICE_PAIRS[args.rivals])
    frame = pd.DataFrame(scores, index=sites.index)
    frame["region"], frame["segment"] = region, sites["segment"].to_numpy()
    frame.to_csv(results / f"compete_svd_summary_{tag}.csv")

    refine_path = results / f"compete_summary_{tag}.csv"
    if refine_path.is_file():
        old = pd.read_csv(refine_path, index_col=0)
        common = [k for k in keys if k in old.columns]
        print(f"\n== refine_template Stage 6 ({refine_path.name}) vs SVD, same sites")
        print(f"  {'rival':<9}{'refine z':>10}{'SVD z':>9}{'per-site r':>12}")
        for key in common:
            r = np.corrcoef(old[key].to_numpy(), scores[key])[0, 1]
            print(f"  {key:<9}{old[key].mean():>10.3f}{scores[key].mean():>9.3f}{r:>12.2f}")
        rank_old = sorted(common, key=lambda k: old[k].mean(), reverse=True)
        rank_new = sorted(common, key=lambda k: scores[k].mean(), reverse=True)
        print("  refine ranking: " + " > ".join(rank_old))
        print("  SVD ranking:    " + " > ".join(rank_new)
              + ("   (same)" if rank_old == rank_new else "   (DIFFERS)"))
        best = rank_old[0]
        print(f"  t of {best} over each rival (refine | SVD):")
        for key in common:
            if key == best:
                continue
            values = []
            for source in (old[best].to_numpy() - old[key].to_numpy(),
                           scores[best] - scores[key]):
                values.append(source.mean() / (source.std(ddof=1) / np.sqrt(len(source))))
            print(f"    {best} - {key:<9} {values[0]:+7.1f} | {values[1]:+7.1f}")
    print(f"\nwinner (SVD): {winner}; wrote compete_svd_summary_{tag}.csv")


if __name__ == "__main__":
    main()
