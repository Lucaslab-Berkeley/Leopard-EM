"""Calibrate both register statistics on a frame where the register is KNOWN.

Two statistics are in play and they behave differently under a detection threshold:

* **difference of means** -- mean z of one register minus the other. A threshold
  truncates the weaker register from below, pulling its surviving mean UP, so this
  statistic is destroyed by a rising cut even when the effect is real.
* **occupancy by register** -- what fraction of PREDICTED sites are detected in each.
  A threshold is exactly what makes this visible: the weaker register loses sites
  outright rather than having its mean shifted.

On the GDP frame the mean difference decayed to zero and flipped sign under a cut, which
I read as noise, while the occupancy ratio came out 1.21 and 1.27 (p 0.07 each, Fisher
0.034). Those two readings cannot both be right, and neither can be judged without
knowing what a REAL register looks like through the same pipeline.

The curved synthetic is that calibration: its register is known and the harmonic test
reads it at 2.19x. So run both statistics on it and see what magnitudes a true register
produces.

Usage:
    python run_scripts/calibrate_register_statistics.py
"""

import argparse
import pathlib
import sys

import numpy as np
from scipy import stats

from leopard_em.analysis.filament_lattice import TemplateLatticeGeometry
from leopard_em.pydantic_models.results.correlation_table import detections_from_hdf5

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyse_gdp_curved as A  # noqa: E402
from continuous_unwrap_curved import continuous_sites  # noqa: E402
from frame_geometry import legacy_frame_geometry  # noqa: E402
from register_mean_difference import difference  # noqa: E402

CUTOFFS = (0.0, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0)
# What the GDP frame gave, for side-by-side reading.
GDP = {"patch region 1": (1.27, 0.0711), "patch region 2": (1.21, 0.0782)}


def occupancy_by_register(index, n_protofilaments):
    """Detected fraction of predicted sites in each register, and a binomial test."""
    span = np.arange(index.min(), index.max() + 1)
    predicted_even = int((span % 2 == 0).sum()) * n_protofilaments
    predicted_odd = int((span % 2 == 1).sum()) * n_protofilaments
    n_even = int((index % 2 == 0).sum())
    n_odd = int((index % 2 == 1).sum())
    share = predicted_even / (predicted_even + predicted_odd)
    p = stats.binomtest(n_even, n_even + n_odd, share).pvalue
    return {"n_even": n_even, "n_odd": n_odd,
            "pred_even": predicted_even, "pred_odd": predicted_odd,
            "occ_even": n_even / predicted_even, "occ_odd": n_odd / predicted_odd,
            "ratio": (n_even / predicted_even) / (n_odd / predicted_odd), "p": p}


def main() -> None:
    """Both statistics on the synthetic, against what the GDP frame gave."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="results_curved/"
                        "output_correlation_table_clean_patch13.h5")
    parser.add_argument("--stem",
                        default="6dpu_atoms_6dpv_lattice_13pf_patch5_2rings_flatB")
    parser.add_argument("--min-z", type=float, default=6.0)
    parser.add_argument("--resolve-protofilaments", action="store_true", default=True)
    args = parser.parse_args()

    geometry = TemplateLatticeGeometry.from_pdb(str(MT_ROOT / "models" / f"{args.stem}.pdb"))
    detections = detections_from_hdf5(
        str(MT_ROOT / args.table), min_z_score=args.min_z).reset_index(drop=True)
    # --stem names the template that made --table, which is what an older table needs.
    frame = legacy_frame_geometry(MT_ROOT / args.table, args.stem)
    print(frame.describe())
    boot = A.pick_bootstrap(detections["z_score"].to_numpy(), want=20)
    sites = continuous_sites(detections, geometry, boot, pixel_size=frame.pixel_size,
                             resolve_protofilaments=args.resolve_protofilaments)
    index = sites["axial_index"].astype(int)
    score = sites["score"]
    print(f"curved synthetic, patch, protofilament-resolved: {len(index)} sites, "
          f"rise {sites['rise']:.3f} Å, deviation "
          f"{np.sqrt((sites['deviation'] ** 2).mean()):.2f} Å")
    print("the harmonic test reads this frame's register at 2.19x chance\n")

    occ = occupancy_by_register(
        index, geometry.n_protofilaments if args.resolve_protofilaments else 1)
    print("OCCUPANCY BY REGISTER  (a threshold makes this visible)")
    print(f"  even {occ['n_even']:5d}/{occ['pred_even']:<5d} = {occ['occ_even']:6.1%}   "
          f"odd {occ['n_odd']:5d}/{occ['pred_odd']:<5d} = {occ['occ_odd']:6.1%}")
    print(f"  ratio {occ['ratio']:.2f}   p = {occ['p']:.2e}")
    print("  GDP frame gave: " + ", ".join(
        f"{k} {v[0]:.2f} (p {v[1]:.3f})" for k, v in GDP.items()))

    rng = np.random.default_rng(0)
    print(f"\nDIFFERENCE OF MEANS under a rising cut  (a threshold DESTROYS this)")
    print(f"  {'z cut':>6} {'n even':>7} {'n odd':>6} {'mean z':>8} {'delta':>8} "
          f"{'t':>7} {'% of mean':>10} {'p':>9}")
    for cut in CUTOFFS:
        keep = score > cut
        if keep.sum() < 10:
            continue
        result = difference(index[keep], score[keep], rng, n_null=5000)
        if result is None:
            continue
        print(f"  {cut:6.1f} {result['n_even']:7d} {result['n_odd']:6d} "
              f"{result['mean']:8.3f} {result['delta']:+8.3f} {result['t']:+7.2f} "
              f"{100 * abs(result['delta']) / result['mean']:9.1f}% {result['p']:9.4f}")
    control = difference(index // 2, score, rng, n_null=5000)
    if control:
        print(f"  {'dimer':>6} {control['n_even']:7d} {control['n_odd']:6d} "
              f"{control['mean']:8.3f} {control['delta']:+8.3f} {control['t']:+7.2f} "
              f"{100 * abs(control['delta']) / control['mean']:9.1f}% "
              f"{control['p']:9.4f}   <- THE null")

    print("\n  Read it this way. If the KNOWN register gives an occupancy ratio far above")
    print("  the GDP frame's 1.21-1.27, that asymmetry is too small to be the same thing.")
    print("  And if delta ALSO decays under a rising cut here, then decay is what a real")
    print("  register does through a threshold and my earlier reading of it as noise was")
    print("  wrong.")


if __name__ == "__main__":
    main()
