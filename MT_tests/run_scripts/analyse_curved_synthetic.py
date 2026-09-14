"""Score the four readouts on the curved synthetic, against its known truth.

Everything here has a right answer, which is the point: the tube's protofilament number,
lattice, monomer rise, polarity and curve were all inputs to
``simulate_curved_microtubule.py`` and are recorded in its ``.truth.json``.

This runs the EXISTING straight-filament analysis first, deliberately. It is the baseline
that says how badly a straight axis fails on a bent tube and where -- the axis residual
against the known sagitta, the bow in the site deviations, the bias in the rise. Those
numbers are what a curved axis has to beat, and without them "the curved version is
better" is an assertion rather than a measurement.

Usage:
    python run_scripts/analyse_curved_synthetic.py
    python run_scripts/analyse_curved_synthetic.py --tag truth_13pf_6dpv --min-z 7
"""

import argparse
import json
import math
import pathlib

import numpy as np

from leopard_em.analysis.filament_lattice import (
    TemplateLatticeGeometry,
    axis_points_from_peaks,
    estimate_polarity,
    estimate_protofilament_number,
    extract_lattice_sites,
    filament_coordinates,
    filament_direction_from_angles,
    fit_filament_axis,
)
from leopard_em.pydantic_models.results.correlation_table import detections_from_hdf5

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PIXEL_SIZE = 0.9194

# The 2-ring geometry that matches the simulated tube. 13-PF models are built by
# models/build_mt_templates.py --protofilaments 13.
GEOMETRY_PDB = {
    13: MT_ROOT / "models" / "6dpu_13pf_2rings_flatB.pdb",
    14: MT_ROOT / "models" / "6dpu_2rings_aligned_zero.pdb",
}


def header(text: str) -> None:
    """A labelled section, so the report reads top to bottom."""
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def verdict(name: str, measured: float, truth: float, tolerance: float,
            unit: str = "") -> bool:
    """Print one scored line and return whether it passed."""
    ok = abs(measured - truth) <= tolerance
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name:<34} measured {measured:9.4f}{unit}  "
          f"truth {truth:9.4f}{unit}  (tol {tolerance:g})")
    return ok


def report_polarity(detections, truth) -> None:
    """Which way the tube points. Uses psi only, so no axis is involved."""
    header("1. POLARITY")
    estimate = estimate_polarity(detections, score_threshold=7.0)
    # The tube sweeps, so the truth is the mid-arc psi, not a single value.
    mid = 0.5 * (truth["psi_deg_start"] + truth["psi_deg_end"])
    print(f"  dominant psi {estimate.psi_deg:.2f}°, "
          f"score ratio {estimate.score_ratio:.1f}:1")
    print(f"  truth: psi sweeps {truth['psi_deg_start']:.2f}° → "
          f"{truth['psi_deg_end']:.2f}°, mid {mid:.2f}°")
    # estimate_polarity reports the axis modulo 180, so compare on that.
    difference = abs(((estimate.psi_deg - mid + 90.0) % 180.0) - 90.0)
    verdict("psi vs mid-arc truth (mod 180)", difference, 0.0, 3.0, "°")
    print("  NOTE the spread is the BEND, not error: a swept psi dilutes the ratio,")
    print("       and psi_deg is a chord average. Report it per segment on a curve.")


def report_protofilaments(detections, truth) -> None:
    """Protofilament number from the azimuth harmonic. No axis involved either."""
    header("2. PROTOFILAMENT NUMBER (azimuth harmonic)")
    strong = detections[detections["z_score"] > 7.0]
    best, power = estimate_protofilament_number(strong, max_count=20)
    # power covers orders 2..max_count; show the range that matters.
    orders = np.arange(2, 2 + len(power))
    window = (orders >= 11) & (orders <= 16)
    print("  order:  " + "  ".join(f"{o:6d}" for o in orders[window]))
    print("  power:  " + "  ".join(f"{p:6.3f}" for p in power[window]))
    others = power[window][orders[window] != best]
    margin = power[orders == best][0] / others.max() if len(others) else float("inf")
    print(f"  best overall {best}, margin over the runner-up in 11-16 {margin:.2f}x")
    verdict("protofilament number", float(best),
            float(truth["n_protofilaments"]), 0.0)


def report_axis(detections, truth, geometry) -> np.ndarray:
    """Fit the STRAIGHT axis and measure how far it is from the known curve."""
    header("3. AXIS -- the straight fit against a known bend")
    strong = detections[detections["z_score"] > 8.0]
    points = axis_points_from_peaks(strong, geometry, PIXEL_SIZE)
    reference = filament_direction_from_angles(
        strong["phi"].to_numpy(), strong["theta"].to_numpy(),
        strong["psi"].to_numpy(),
    )
    axis = fit_filament_axis(
        points, strong["z_score"].to_numpy(), reference_direction=reference
    )
    along, across = filament_coordinates(points, axis)

    sagitta = truth["sagitta_px"]
    radius = truth["radius_of_curvature_px"]
    # Curvature only shows over LENGTH. Judge the fit against the span actually
    # detected, not the whole simulated arc -- a short stretch of a gentle bend is
    # genuinely almost straight, and calling that a failure would be wrong.
    span = float(np.ptp(along))
    span_sagitta = span**2 / (8.0 * radius) if radius else 0.0
    expected = span_sagitta / math.sqrt(5.0)
    print(f"  {len(strong):,} detections above z 8, spanning {span:.0f} px "
          f"of a {truth['arc_length_px']:.0f} px arc")
    print(f"  axis angle {axis.angle_deg:.2f}°, residual {axis.residual_rms_px:.2f} px")
    print(f"  truth: sagitta {sagitta:.0f} px over the full arc, radius "
          f"{radius:.0f} px")
    print(f"  over the DETECTED span the sagitta is {span_sagitta:.1f} px, so a "
          f"straight fit should leave ~{expected:.1f} px")
    if span < 0.3 * truth["arc_length_px"]:
        print("  [SKIP] too little of the tube detected to test the axis; "
              "the bend needs length")
    else:
        verdict("residual tracks the sagitta", axis.residual_rms_px, expected,
                max(0.4 * expected, 2.0), " px")

    # The transverse coordinate is the bend, laid bare. extract_lattice_sites throws
    # this away, which is why curvature currently degrades it silently.
    order = np.argsort(along)
    bins = np.array_split(order, 9)
    print("\n  transverse offset along the track (this is the arc, discarded today):")
    print("    along (px): " + "  ".join(
        f"{along[b].mean():8.0f}" for b in bins))
    print("    across(px): " + "  ".join(
        f"{across[b].mean():8.1f}" for b in bins))
    print(f"    peak-to-peak {np.ptp([across[b].mean() for b in bins]):.1f} px "
          f"against a {sagitta:.0f} px sagitta")
    return axis


def report_spacing(detections, truth, geometry) -> None:
    """Lattice spacing from the straight top-down extraction."""
    header("4. LATTICE SPACING (straight extraction)")
    sites = extract_lattice_sites(
        detections, geometry, PIXEL_SIZE, bootstrap_score_threshold=8.0
    )
    rise = float(sites.rise_angstrom)
    deviation = np.asarray(sites.axial_deviation_angstrom)
    index = np.asarray(sites.axial_index, dtype=float)
    truth_rise = truth["monomer_rise_angstrom"]

    print(f"  {len(sites.detection_index)} of {sites.n_predicted} predicted sites "
          f"occupied ({sites.occupancy:.0%})")
    print(f"  deviation {np.sqrt((deviation**2).mean()):.2f} Å rms "
          f"against a ±{rise / 2:.1f} Å window")
    verdict("monomer rise", rise, truth_rise, 0.25, " Å")
    print(f"  that is {100 * (rise - truth_rise) / truth_rise:+.2f}% from truth; "
          f"expanded vs compacted differ by 2.5%")

    # Chord-vs-arc error is systematic in position, so the deviations BOW rather than
    # scatter. This is the cleanest curvature alarm and nothing looks at it today.
    if len(index) > 12:
        fit = np.polyfit(index, deviation, 2)
        curvature_term = fit[0] * np.ptp(index) ** 2
        print(f"  deviation vs index: quadratic term spans {curvature_term:+.2f} Å "
              f"across the track")
        print("  (a bow here means the axis is wrong, not the lattice; a straight "
              "tube gives none)")


def report_register(detections, truth, geometry) -> None:
    """The monomer register, as the seam test reads it."""
    header("5. MONOMER REGISTER (period-2 alternation)")
    sites = extract_lattice_sites(
        detections, geometry, PIXEL_SIZE, bootstrap_score_threshold=8.0
    )
    index = np.asarray(sites.axial_index).astype(int)
    score = np.asarray(sites.score)
    order = np.argsort(index)
    index, score = index[order], score[order]

    weight = score - score.mean()
    phase = np.exp(1j * np.pi * index)
    amplitude = float(np.abs((weight * phase).sum()) / len(index))
    rng = np.random.default_rng(0)
    null = float(np.percentile(
        [float(np.abs((rng.permutation(weight) * phase).sum()) / len(index))
         for _ in range(2000)], 95))
    print(f"  {len(index)} sites, period-2 amplitude {amplitude:.3f}, "
          f"chance {null:.3f}  ({amplitude / null:.2f}x)")
    print("  truth: the template has a definite register and was tiled at its own")
    print("         repeat, so an alternation SHOULD be present.")


def main() -> None:
    """Run every readout and score it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="truth_13pf_6dpv")
    parser.add_argument("--truth", type=pathlib.Path, default=None)
    parser.add_argument("--min-z", type=float, default=7.0)
    args = parser.parse_args()

    truth_path = args.truth or (
        MT_ROOT / "Frames" / "synthetic_curved_13pf_6dpv_sag100.truth.json"
    )
    with open(truth_path, encoding="utf-8") as handle:
        truth = json.load(handle)

    table = MT_ROOT / "results_curved" / f"output_correlation_table_{args.tag}.h5"
    detections = detections_from_hdf5(str(table), min_z_score=args.min_z)
    detections = detections.reset_index(drop=True)

    geometry = TemplateLatticeGeometry.from_pdb(
        str(GEOMETRY_PDB[truth["n_protofilaments"]])
    )

    header("GROUND TRUTH")
    for key in ("n_protofilaments", "lattice", "monomer_rise_angstrom",
                "sagitta_px", "radius_of_curvature_px", "total_turn_deg",
                "psi_deg_start", "psi_deg_end", "bend_strain_present"):
        print(f"  {key:<28} {truth[key]}")
    print(f"\n  {len(detections):,} detections above z {args.min_z}, "
          f"max z {detections['z_score'].max():.2f}")

    report_polarity(detections, truth)
    report_protofilaments(detections, truth)
    report_axis(detections, truth, geometry)
    report_spacing(detections, truth, geometry)
    report_register(detections, truth, geometry)


if __name__ == "__main__":
    main()
