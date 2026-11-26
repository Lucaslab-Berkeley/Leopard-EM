"""The register as a difference of MEANS, with a z-cutoff sweep.

Every site on the tube is assigned to one of two groups by the parity of its axial index:
one group has the template in register, the other has it one monomer out (beta where alpha
should be). The statistic is simply

    delta = mean z of one group  -  mean z of the other

which is the quantity the GMPCPP result was quoted in (1.35 z on a mean of 13.3, about
10%). It is the same information as the period-2 harmonic amplitude -- for period 2 that
amplitude is |delta|/2 when the parities are balanced -- but in a form that carries a unit,
a standard error and a t-statistic, where "3x a 95th-percentile chance level" carries none
of those and is hard to read.

Three things are reported against it:

* a **permutation null** on the parity labels, which is exact: it keeps every score and
  every group size and destroys only the pairing;
* the **dimer control**, sites re-indexed at twice the rise so every site shares a
  register -- any difference there is method artefact;
* a **z-cutoff sweep**, since a register should be easiest to see among strong sites.

⚠️ A cutoff is CONSERVATIVE for this statistic, not permissive. Truncating both groups at
the same z pulls both means up toward the cutoff and shrinks their difference, so a delta
that survives a rising cutoff is real; one that grows with it is more likely selection.

Usage:
    python run_scripts/register_mean_difference.py
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from seam_from_patch_search import TEMPLATES  # noqa: E402

CUTOFFS = (0.0, 7.0, 7.5, 8.0, 8.5, 9.0)


def difference(index, score, rng, n_null=20000):
    """Mean-z difference between parities, its error, and a permutation null."""
    even, odd = index % 2 == 0, index % 2 == 1
    if even.sum() < 4 or odd.sum() < 4:
        return None
    a, b = score[even], score[odd]
    delta = float(a.mean() - b.mean())
    sem = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    # Permuting the parity LABELS keeps both score sets and both group sizes intact and
    # destroys only which site is which register -- the only thing delta depends on.
    labels = np.concatenate([np.ones(len(a), bool), np.zeros(len(b), bool)])
    pool = np.concatenate([a, b])
    null = np.empty(n_null)
    for i in range(n_null):
        perm = rng.permutation(labels)
        null[i] = pool[perm].mean() - pool[~perm].mean()
    p = float((np.abs(null) >= abs(delta)).mean())
    return {"delta": delta, "sem": sem, "t": delta / sem if sem else np.nan,
            "p": p, "n_even": int(even.sum()), "n_odd": int(odd.sum()),
            "mean": float(score.mean())}


def main() -> None:
    """Report the difference of means for each template and region."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results_gdp_curved")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    for key in ("patch", "ring"):
        tag, _stem = TEMPLATES[key]
        refined = MT_ROOT / args.results / f"refined_continuous_{tag}.csv"
        stack = MT_ROOT / args.results / f"stack_continuous_{tag}.csv"
        if not refined.is_file():
            print(f"{key}: no refined_continuous_{tag}.csv")
            continue
        after = pd.read_csv(refined, index_col=0)
        labels = pd.read_csv(stack, index_col=0)
        region_all = labels["region"].to_numpy()
        index_all = labels["axial_index"].to_numpy().astype(int)
        score_all = after["refined_scaled_mip"].to_numpy()

        print(f"\n{'=' * 82}\n{key.upper()}  ({tag})\n{'=' * 82}")
        for region in sorted(set(region_all.tolist())):
            keep = region_all == region
            print(f"\n  region {region}: {int(keep.sum())} sites")
            print(f"    {'z cut':>6} {'n even':>7} {'n odd':>6} {'mean z':>8} "
                  f"{'delta':>8} {'sem':>7} {'t':>7} {'% of mean':>10} {'p':>8}")
            for cut in CUTOFFS:
                mask = keep & (score_all > cut)
                if mask.sum() < 10:
                    continue
                result = difference(index_all[mask], score_all[mask], rng)
                if result is None:
                    continue
                print(f"    {cut:6.1f} {result['n_even']:7d} {result['n_odd']:6d} "
                      f"{result['mean']:8.3f} {result['delta']:+8.3f} "
                      f"{result['sem']:7.3f} {result['t']:+7.2f} "
                      f"{100 * abs(result['delta']) / result['mean']:9.1f}% "
                      f"{result['p']:8.4f}")

            # THE null: same sites, but grouped at the DIMER rate, where both groups
            # contain both registers.
            mask = keep
            control = difference(index_all[mask] // 2, score_all[mask], rng)
            if control:
                print(f"    {'dimer':>6} {control['n_even']:7d} {control['n_odd']:6d} "
                      f"{control['mean']:8.3f} {control['delta']:+8.3f} "
                      f"{control['sem']:7.3f} {control['t']:+7.2f} "
                      f"{100 * abs(control['delta']) / control['mean']:9.1f}% "
                      f"{control['p']:8.4f}   <- THE null")

    print("\n  Reference: the GMPCPP frame gave delta 1.35 z on a mean of 13.3 (10.2%).")
    print("  A cutoff SHRINKS a real delta, so one that survives a rising cut is real and")
    print("  one that appears only at a high cut is selection.")


if __name__ == "__main__":
    main()
