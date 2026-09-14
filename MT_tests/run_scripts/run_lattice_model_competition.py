"""Compete lattice models against a fixed set of detections.

Complementary to competing whole templates. That varies the template, and so carries
template-quality and template-size confounds; this holds the detections completely fixed
and varies only the *model* -- predicting where every subunit should appear in (x, y) and
azimuth, then asking whether a detection is actually there above a threshold.

Two properties make it worth running:

* Sites a model predicts but that hold nothing are counted, so occupancy is a real
  statistic rather than the near tautology it becomes when the sites are defined by the
  detections themselves.
* The grid spans both protofilament number and lattice repeat, and **the repeat answer is
  already known independently** (extended, t = 10). That makes the repeat axis a built-in
  positive control: if this test does not return extended, its protofilament verdict is
  worthless and should not be reported.

What it cannot see: tube radius. The template's axis offset belongs to the template that
produced the detections, not to the model under test, so it is held fixed. Radius is what
competing whole templates measures; the two are complementary.

Usage:  python run_scripts/run_lattice_model_competition.py [--min-z 8.0]
"""

import argparse
import pathlib

import numpy as np

from leopard_em.analysis.filament_lattice import (
    TemplateLatticeGeometry,
    extract_lattice_sites,
    score_lattice_model,
)
from leopard_em.pydantic_models.results.correlation_table import detections_from_hdf5

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PIXEL_SIZE = 0.9194

# The patch table, not the ring: phi is well determined here, and different
# protofilaments land at different (x, y). A ring template is N-fold pseudo-symmetric,
# so its phi carries no protofilament identity and the angular gate would be meaningless.
TABLE = MT_ROOT / "results_full" / "output_correlation_table_patch4_full.h5"
TEMPLATE_PDB = MT_ROOT / "models" / "6dpu_4_patches_aligned_zero.pdb"
AXIS_PDB = MT_ROOT / "models" / "6dpu_2rings_aligned_zero.pdb"
RING_TABLE = MT_ROOT / "results_full" / "output_correlation_table_2rings_full.h5"

PROTOFILAMENTS = (12, 13, 14, 15)
REPEATS = {"extended": 83.958, "compacted": 81.671}

# The correlation table was written at a raw cross-correlation threshold of 6.5, so no
# detection below it exists to be found however permissive the statistics allow.
TABLE_FLOOR = 6.5


def main() -> None:
    """Score every {protofilament number} x {repeat} model on the same detections."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-z", type=float, default=8.0)
    parser.add_argument("--xy-radius-px", type=float, default=12.0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--null-draws", type=int, default=12)
    # 0.5 A resolved nothing the matching window could see; 2 A is 4x faster.
    parser.add_argument("--phase-step", type=float, default=2.0)
    args = parser.parse_args()

    print(f"streaming {TABLE.name} at z > {args.min_z}")
    detections = detections_from_hdf5(str(TABLE), min_z_score=args.min_z)
    print(f"  {len(detections):,} detections")

    # The axis comes from the ring run, which has the better positional accuracy, but it
    # must be expressed in the patch detections' own frame -- so fit it from the patch
    # table using the patch template's calibration.
    geometry = TemplateLatticeGeometry.from_pdb(str(TEMPLATE_PDB))
    axis = extract_lattice_sites(
        detections, geometry, PIXEL_SIZE, bootstrap_score_threshold=10.0
    ).axis
    print(f"  axis residual {axis.residual_rms_px:.2f} px, "
          f"in-plane angle {axis.angle_deg:.2f} deg")

    threshold = args.threshold if args.threshold is not None else TABLE_FLOOR
    print(f"  site threshold z > {threshold} "
          f"({'given' if args.threshold else 'the table write floor'})\n")

    rng = np.random.default_rng(0)
    # Eligibility and the spatial index depend on the detections and the anchor, not on
    # the model, so build them once and share across all models and all null draws.
    cache: dict = {}
    rows = []
    header = (f"{'model':>22} {'sites':>7} {'found':>7} {'occup':>7} {'null':>14} "
              f"{'excess':>8} {'mean z':>8} {'dev rms':>8}")
    print(header)
    print("-" * len(header))
    for name, rise in REPEATS.items():
        for n_pf in PROTOFILAMENTS:
            common = dict(
                n_protofilaments=n_pf,
                rise_angstrom=rise,
                score_threshold=threshold,
                xy_radius_px=args.xy_radius_px,
                phase_step_angstrom=args.phase_step,
                detection_index_cache=cache,
            )
            score = score_lattice_model(
                detections, geometry, axis, PIXEL_SIZE, **common
            )
            # Occupancy alone is not comparable between models: a coarser azimuth grid
            # finds something in window more often whatever the truth, so every model
            # has its own floor. Rotating the predicted grid to a random azimuth
            # measures that floor while holding N, repeat, windows and threshold fixed.
            null = np.array(
                [
                    score_lattice_model(
                        detections,
                        geometry,
                        axis,
                        PIXEL_SIZE,
                        azimuth_offset_deg=float(rng.uniform(0, 360)),
                        **common,
                    ).occupancy
                    for _ in range(args.null_draws)
                ]
            )
            excess = (
                (score.occupancy - null.mean()) / null.std(ddof=1)
                if null.std(ddof=1) > 0
                else float("nan")
            )
            dev = (
                float(np.sqrt((score.axial_deviation_angstrom**2).mean()))
                if score.n_found
                else float("nan")
            )
            rows.append((name, n_pf, score, dev, excess))
            print(f"{name + ' N=' + str(n_pf):>22} {score.n_predicted:7d} "
                  f"{score.n_found:7d} {score.occupancy:6.1%} "
                  f"{null.mean():7.1%} +/-{null.std(ddof=1):5.1%} {excess:+8.1f} "
                  f"{score.mean_score:8.3f} {dev:7.2f}A")

    print()
    best = max(rows, key=lambda r: r[4] if np.isfinite(r[4]) else -np.inf)
    print(f"highest EXCESS over null: {best[0]} N={best[1]} ({best[4]:+.1f} sigma)")
    raw = max(rows, key=lambda r: r[2].occupancy)
    print(f"highest raw occupancy:    {raw[0]} N={raw[1]} ({raw[2].occupancy:.1%})")

    # The repeat axis is the control: its answer is already known independently.
    for n_pf in PROTOFILAMENTS:
        ext = next(r[2] for r in rows if r[0] == "extended" and r[1] == n_pf)
        com = next(r[2] for r in rows if r[0] == "compacted" and r[1] == n_pf)
        flag = "extended" if ext.occupancy > com.occupancy else "COMPACTED"
        print(f"  N={n_pf}: extended {ext.occupancy:.1%} vs compacted "
              f"{com.occupancy:.1%}  -> {flag}")
    print("\nCONTROL: the repeat axis must say extended. If it does not, this test is "
          "not working and its protofilament answer should not be reported.")


if __name__ == "__main__":
    main()
