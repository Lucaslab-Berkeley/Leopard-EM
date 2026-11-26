"""Is the register hiding in the strong sites, or in a good-looking window?

Two questions, kept apart because only one of them is answerable honestly.

(A) STRATIFY BY SCORE -- legitimate. If the register is real it should be easier to see
    where the signal is strong, and the split is decided by z-score, not by eye. Sites
    are halved at their own segment's median z and each half tested separately.

(B) THE BEST WINDOW -- a trap, handled with a matched null. Scanning a track for the
    stretch that looks most alternating and then testing that stretch is circular: over
    7 segments and every window position something always looks periodic. The honest
    version asks whether the BEST window found in the real data beats the BEST window
    found in permuted data, scanning identically. That is a max-statistic test and it
    costs nothing extra to do properly.

Usage:
    python run_scripts/seam_stratify_curved.py
"""

import argparse
import pathlib
import sys

import numpy as np

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plot_seam_curved import collect  # noqa: E402

MIN_WINDOW = 10


def amplitude_of(index, score):
    """Period-2 amplitude of score against axial index."""
    weight = score - score.mean()
    return float(np.abs((weight * np.exp(1j * np.pi * index)).sum()) / len(index))


def best_window(index, score, min_window=MIN_WINDOW):
    """Largest period-2 amplitude over every contiguous window, and where."""
    best, where = -1.0, None
    n = len(index)
    for start in range(n - min_window + 1):
        for stop in range(start + min_window, n + 1):
            value = amplitude_of(index[start:stop], score[start:stop])
            if value > best:
                best, where = value, (index[start], index[stop - 1], stop - start)
    return best, where


def main() -> None:
    """Run both tests on every patch segment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results_gdp_curved")
    parser.add_argument("--constraint", type=pathlib.Path,
                        default=MT_ROOT / "configs"
                        / "filament_constraint_gdp_curved.h5")
    parser.add_argument("--min-z", type=float, default=6.0)
    parser.add_argument("--n-null", type=int, default=400)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    segments = collect("patch", args.results, args.constraint, args.min_z, rng)
    if not segments:
        raise SystemExit("no patch segments")

    print("(A) SPLIT BY SCORE -- decided by z, not by eye")
    print(f"  {'segment':<12} {'half':<8} {'n':>4} {'mean z':>8} {'amp':>7} "
          f"{'chance':>8} {'ratio':>7}")
    for block in segments:
        index, score = block["index"], block["score"]
        cut = np.median(score)
        for name, mask in (("strong", score >= cut), ("weak", score < cut)):
            if mask.sum() < 8:
                continue
            i, s = index[mask], score[mask]
            amplitude = amplitude_of(i, s)
            null = float(np.percentile(
                [amplitude_of(i, rng.permutation(s)) for _ in range(2000)], 95))
            print(f"  {block['label']:<12} {name:<8} {int(mask.sum()):4d} "
                  f"{s.mean():8.3f} {amplitude:7.3f} {null:8.3f} "
                  f"{amplitude / null:6.2f}x")

    print(f"\n(B) BEST WINDOW, against a null scanned the SAME way "
          f"({args.n_null} permutations)")
    print(f"  {'segment':<12} {'window':>14} {'n':>4} {'best amp':>9} "
          f"{'null best':>10} {'ratio':>7} {'p':>7}")
    for block in segments:
        index, score = block["index"], block["score"]
        if len(index) < MIN_WINDOW + 2:
            continue
        observed, where = best_window(index, score)
        null = np.array([best_window(index, rng.permutation(score))[0]
                         for _ in range(args.n_null)])
        p = float((null >= observed).mean())
        print(f"  {block['label']:<12} {f'{where[0]}..{where[1]}':>14} {where[2]:4d} "
              f"{observed:9.3f} {np.percentile(null, 95):10.3f} "
              f"{observed / np.percentile(null, 95):6.2f}x {p:7.3f}")

    print("\n  In (B) the null is the best window found in SHUFFLED data, scanned over")
    print("  exactly the same windows -- so it already contains the advantage that")
    print("  cherry-picking gives. A p below 0.05 there would be a real local register;")
    print("  anything above says the window only looks good because we went looking.")


if __name__ == "__main__":
    main()
