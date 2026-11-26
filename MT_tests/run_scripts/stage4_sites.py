"""Stage 4: lattice sites per filament, indexed continuously along a data-fitted curve.

Reads Stage 3's membership (``stage3_membership_<tag>.csv``) so each filament's
detections are the ones its own axis points and directions assigned it -- not the drawn
region they happened to fall in. Each filament then goes through ``continuous_sites``:
a curve fitted to the detections' axis points, arc-length indexing with the helical
unwrap, one winning detection per lattice site, and the rise refitted with the
per-detection sin(theta) correction.

For a RING template (the default) this is the lattice-spacing readout and the site list
Stage 5 refines. ``--resolve-protofilaments`` is for a patch only: a ring's phi does not
identify a protofilament.

Writes ``stage4_sites_<tag>.csv``: the winning detection at every site, with its
filament and axial index.

Usage:
    python run_scripts/stage4_sites.py \\
        --table results_in_situ_823/output_correlation_table_in_situ_823_13pf.h5
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

from leopard_em.analysis.correlation_peaks import find_peaks_orientation_aware
from leopard_em.analysis.filament_lattice import TemplateLatticeGeometry

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from continuous_unwrap_curved import continuous_sites  # noqa: E402
from frame_geometry import frame_geometry, legacy_frame_geometry  # noqa: E402
from stage3_read_table import RING_STEM  # noqa: E402


BOOTSTRAP_PEAKS = 20  # what continuous_sites requires
# A filament's rise is read only with enough sites landing well inside the +/-rise/2
# window. Empirical, from the bundle synthetic and GDP: filaments that measured right had
# 23-65 sites at 3.2-6.6 A rms deviation; the two that did not had 7.9-10.1 A (one with
# 16 sites was 3.4 sigma from the truth; GDP region 1 was already "unresolved"). Large
# deviation means sites are being assigned to the wrong subunit.
MIN_SITES, MAX_DEVIATION_ANGSTROM = 20, 7.0


BLOCK_SITES = 25  # at least this many sites per block for its own rise


def block_rises(frame, axial_index, geometry, pixel_size):
    """Rise on contiguous blocks along the tube, and the error their scatter implies.

    The fit's own error is statistical only and blind to the rise varying ALONG a tube.
    Measured, it does: halves of one filament differ by 0.96% on GDP and 0.4-1.9% on
    in situ 823, against 0.09% on a synthetic whose lattice is uniform by construction --
    so the variation is in the data, and an honest error must include it. Blocks are
    contiguous runs of axial index with at least BLOCK_SITES sites (2-4 of them), each
    re-indexed on its own curve. Returns (block rises, standard error of their mean),
    or ([], nan) when the tube is too short to split.
    """
    order = np.argsort(np.asarray(axial_index))
    n_blocks = min(4, len(order) // BLOCK_SITES)
    if n_blocks < 2:
        return [], float("nan")
    rises = []
    for part in np.array_split(order, n_blocks):
        piece = frame.iloc[part].reset_index(drop=True)
        try:
            sites = continuous_sites(piece, geometry, float(piece["z_score"].min()) - 0.01,
                                     pixel_size=pixel_size)
        except ValueError:
            continue
        rises.append(float(sites["rise"]))
    if len(rises) < 2:
        return rises, float("nan")
    return rises, float(np.std(rises, ddof=1) / np.sqrt(len(rises)))


def reliability(n_sites: int, deviation_rms: float) -> str:
    """'ok', or why the rise should not be read."""
    reasons = []
    if n_sites < MIN_SITES:
        reasons.append(f"<{MIN_SITES} sites")
    if deviation_rms > MAX_DEVIATION_ANGSTROM:
        reasons.append(f"dev >{MAX_DEVIATION_ANGSTROM:.0f} A")
    return "ok" if not reasons else "UNRELIABLE: " + ", ".join(reasons)


def bootstrap_threshold(subset, floor, xy_radius_px=12.0, angular_radius_deg=15.0):
    """Highest z cut leaving BOOTSTRAP_PEAKS distinct peaks -- as continuous_sites counts.

    ``pick_bootstrap`` counted table ROWS above a few fixed cuts, but continuous_sites
    needs distinct PEAKS after orientation-aware suppression (12 px AND 15 deg), and on a
    weak tube the two differ by an order of magnitude: the bundle synthetic's strongest
    tube had 20,057 rows yet only 17 peaks above the cut rows chose. So count peaks,
    stepping down from 8.0 in 0.25 until enough survive or ``floor`` is reached.
    """
    # at least the 8.0 cut, even when every site is stronger: on a strong tube (GMPCPP,
    # all refined z > 8.4) arange(8.0, floor) was empty and the lattice 'did not fit'
    for cut in np.arange(8.0, min(floor, 8.0 - 1e-6), -0.25):
        peaks = find_peaks_orientation_aware(
            subset, xy_radius_px=xy_radius_px, angular_radius_deg=angular_radius_deg,
            score_column="z_score", score_threshold=float(cut))
        if len(peaks) >= BOOTSTRAP_PEAKS:
            return float(cut)
    return None


def main() -> None:
    """Continuous sites and lattice rise for every filament Stage 3 found."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=pathlib.Path, required=True,
                        help="the correlation table Stage 3 read")
    parser.add_argument("--stem", default=RING_STEM)
    parser.add_argument("--min-z", type=float, default=6.0,
                        help="detections considered; Stage 3's CSV may hold lower ones")
    parser.add_argument("--resolve-protofilaments", action="store_true")
    parser.add_argument("--in-vitro-legacy", action="store_true")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    table = args.table if args.table.is_absolute() else MT_ROOT / args.table
    tag = args.tag or table.stem.removeprefix("output_correlation_table_")
    frame = (legacy_frame_geometry(table, args.stem) if args.in_vitro_legacy
             else frame_geometry(table))
    geometry = TemplateLatticeGeometry.from_pdb(str(MT_ROOT / "models" / f"{args.stem}.pdb"))
    membership = pd.read_csv(table.parent / f"stage3_membership_{tag}.csv")
    membership = membership[membership["z_score"] > args.min_z].reset_index(drop=True)

    print(frame.describe())
    print(f"template {args.stem}: repeat {geometry.rise_angstrom:.3f} A, "
          f"N {geometry.n_protofilaments}")
    print(f"{len(membership):,} Stage 3 detections at z > {args.min_z}\n")
    print(f"{'filament':<9} {'dets':>7} {'boot z':>7} {'sites':>6} {'occ':>6} "
          f"{'rise A':>16} {'block SE':>9} {'vs template':>12} {'dev rms':>8} "
          f"{'knots':>6} {'arc nm':>7}")

    rows, block_rows = [], []
    for label in sorted(int(v) for v in membership["filament"].unique() if v > 0):
        subset = membership[membership["filament"] == label].reset_index(drop=True)
        boot = bootstrap_threshold(subset, floor=args.min_z)
        if boot is None:
            print(f"{label:<9} {len(subset):>7}  fewer than {BOOTSTRAP_PEAKS} distinct "
                  f"peaks even at z > {args.min_z}")
            continue
        try:
            sites = continuous_sites(subset, geometry, boot, pixel_size=frame.pixel_size,
                                     resolve_protofilaments=args.resolve_protofilaments)
        except ValueError as error:
            print(f"{label:<9} {len(subset):>7}  {error}")
            continue
        index = sites["axial_index"]
        occupancy = len(index) / max(sites["n_predicted"], 1)
        rise = sites["rise"]
        rms = float(np.sqrt(np.mean(sites["deviation"] ** 2)))
        error = sites["rise_standard_error"]
        offset = 100 * (rise / geometry.rise_angstrom - 1)
        blocks, block_error = block_rises(
            subset.iloc[sites["detection_index"]].reset_index(drop=True), index,
            geometry, frame.pixel_size)
        block_rows.append((label, blocks))
        print(f"{label:<9} {len(subset):>7,} {boot:>7.2f} {len(index):>6} "
              f"{occupancy:>5.0%} {rise:>8.3f} ± {error:5.3f} {block_error:>9.3f} "
              f"{offset:>+11.2f}% "
              f"{rms:>6.2f} A {sites['n_knots']:>6} "
              f"{sites['arc_span_px'] * frame.pixel_size / 10:>7.0f}  "
              f"{reliability(len(index), rms)}")
        chosen = subset.iloc[sites["detection_index"]].copy()
        chosen["axial_index"] = index
        chosen["protofilament"] = (sites["protofilament"]
                                   if sites["protofilament"] is not None else -1)
        chosen["filament_rise_angstrom"] = rise
        rows.append(chosen)

    print("\n  block rises (contiguous runs along each tube):")
    for label, blocks in block_rows:
        print(f"    filament {label}: " + (", ".join(f"{r:.3f}" for r in blocks)
                                          if blocks else "too short to split"))
    print("  block SE: error from the block scatter -- USE THIS, not the fit's +/-, where "
          "it exists")
    print("\n  rise: per-detection sin(theta) applied, fitted along ARC length; the error")
    print("  is statistical only. On the bundle synthetic it was honest (0.1-1.3 sigma from")
    print("  truth) for tubes with 23+ sites, but 3.4 sigma off for one with 16 sites and an")
    print(f"  8.9 A deviation. Flagged UNRELIABLE below {MIN_SITES} sites or above "
          f"{MAX_DEVIATION_ANGSTROM:.0f} A rms deviation.")
    print("  dev rms: where sites actually landed within the +/-rise/2 window -- small")
    print("  against the window means the data, not the model, set the positions")
    if not rows:
        raise SystemExit("no filament produced sites")
    out = table.parent / f"stage4_sites_{tag}.csv"
    pd.concat(rows, ignore_index=True).to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(MT_ROOT)}")


if __name__ == "__main__":
    main()
