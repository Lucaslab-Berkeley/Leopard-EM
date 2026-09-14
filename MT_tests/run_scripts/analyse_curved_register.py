"""Can a patch template read the monomer register on a BENT tube?

Two things could stop it, and they need separating:

* the **template** -- a ring is N-fold pseudo-symmetric and averages the whole
  circumference, so it cannot do this test at all (on real data it gives ~3% and sits
  below its own chance level, against a patch's 10%);
* the **bend** -- a straight axis mis-assigns sites by a bow that grows along the track,
  which would smear an alternation that is really there.

Splitting the track separates them. Curvature error is quadratic in span, so a short
segment of a bent tube is effectively straight: if the alternation appears within
segments but not across the whole track, the axis is the culprit rather than the
template, and a curved axis would recover it. If it is absent at every scale, the
template is.

The dimer-rate extraction is the control throughout -- indexing at the dimer puts every
site in the SAME register, so any alternation there is an artefact of the method.

Usage:
    python run_scripts/analyse_curved_register.py --tag clean_patch13
"""

import argparse
import json
import pathlib

import numpy as np

from leopard_em.analysis.filament_lattice import (
    TemplateLatticeGeometry,
    axis_points_from_peaks,
    extract_lattice_sites,
    filament_coordinates,
    filament_direction_from_angles,
    fit_filament_axis,
)
from leopard_em.pydantic_models.results.correlation_table import detections_from_hdf5

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PIXEL_SIZE = 0.9194


def alternation(index: np.ndarray, score: np.ndarray, rng, n_null: int = 2000):
    """Period-2 amplitude of score against axial index, and its permutation null.

    Shuffling the scores among the sites keeps every score and every index but destroys
    the pairing, which is the only thing the statistic depends on. The line is the 95th
    percentile of the shuffled distribution.
    """
    weight = score - score.mean()
    phase = np.exp(1j * np.pi * index)
    amplitude = float(np.abs((weight * phase).sum()) / len(index))
    null = float(np.percentile(
        [float(np.abs((rng.permutation(weight) * phase).sum()) / len(index))
         for _ in range(n_null)], 95))
    return amplitude, null


def sites_for(detections, geometry, rise=None, bootstrap=8.0):
    """Occupied lattice sites, indexed at the monomer rise unless told otherwise."""
    return extract_lattice_sites(
        detections, geometry, PIXEL_SIZE,
        bootstrap_score_threshold=bootstrap, initial_rise_angstrom=rise,
    )


def report(label, detections, geometry, rng, rise=None) -> None:
    """One alternation measurement, printed as a row."""
    try:
        sites = sites_for(detections, geometry, rise)
    except ValueError as error:
        print(f"  {label:<30} {error}")
        return
    index = np.asarray(sites.axial_index).astype(int)
    score = np.asarray(sites.score)
    if len(index) < 8:
        print(f"  {label:<30} only {len(index)} sites, skipped")
        return
    order = np.argsort(index)
    index, score = index[order], score[order]
    amplitude, null = alternation(index, score, rng)

    even, odd = score[index % 2 == 0], score[index % 2 == 1]
    difference = (
        abs(even.mean() - odd.mean()) if len(even) > 1 and len(odd) > 1 else float("nan")
    )
    print(f"  {label:<30} {len(index):4d} sites  rise {sites.rise_angstrom:7.3f}  "
          f"amp {amplitude:6.3f}  chance {null:6.3f}  {amplitude / null:5.2f}x  "
          f"|diff| {difference:5.2f} z ({100 * difference / score.mean():4.1f}%)")


def main() -> None:
    """Global and per-segment alternation, plus the dimer control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="clean_patch13")
    parser.add_argument("--geometry", default="6dpu_atoms_6dpv_lattice_13pf_patch5_"
                                              "2rings_flatB.pdb")
    parser.add_argument("--truth", type=pathlib.Path, default=None)
    parser.add_argument("--min-z", type=float, default=7.0)
    args = parser.parse_args()

    truth_path = args.truth or (
        MT_ROOT / "Frames" / "synthetic_curved_13pf_6dpv_sag100_clean.truth.json"
    )
    with open(truth_path, encoding="utf-8") as handle:
        truth = json.load(handle)

    geometry = TemplateLatticeGeometry.from_pdb(str(MT_ROOT / "models" / args.geometry))
    table = MT_ROOT / "results_curved" / f"output_correlation_table_{args.tag}.h5"
    detections = detections_from_hdf5(
        str(table), min_z_score=args.min_z
    ).reset_index(drop=True)

    print(f"template {args.geometry}")
    print(f"  N {geometry.n_protofilaments}, axis offset "
          f"{geometry.axis_offset_magnitude_angstrom:.1f} Å "
          f"({'PATCH' if geometry.axis_offset_magnitude_angstrom > 20 else 'RING'})")
    print(f"  {len(detections):,} detections above z {args.min_z}, "
          f"max z {detections['z_score'].max():.2f}")
    print(f"  tube: {truth['n_protofilaments']} PF, {truth['lattice']}, "
          f"rise {truth['monomer_rise_angstrom']:.3f} Å, "
          f"sagitta {truth['sagitta_px']:.0f} px, R {truth['radius_of_curvature_px']:.0f} px")

    rng = np.random.default_rng(0)
    print("\nWHOLE TRACK")
    report("monomer spacing", detections, geometry, rng)
    report("dimer spacing (control)", detections, geometry, rng,
           rise=2.0 * truth["monomer_rise_angstrom"])

    # Split on the along-axis coordinate of one global straight fit, so the segments are
    # contiguous stretches of tube rather than arbitrary subsets.
    strong = detections[detections["z_score"] > 8.0]
    points = axis_points_from_peaks(strong, geometry, PIXEL_SIZE)
    axis = fit_filament_axis(
        points, strong["z_score"].to_numpy(),
        reference_direction=filament_direction_from_angles(
            strong["phi"].to_numpy(), strong["theta"].to_numpy(),
            strong["psi"].to_numpy(),
        ),
    )
    along, _ = filament_coordinates(
        axis_points_from_peaks(detections, geometry, PIXEL_SIZE), axis
    )
    radius = truth["radius_of_curvature_px"]

    for n_parts in (2, 3, 4):
        span = np.ptp(along) / n_parts
        sagitta = span**2 / (8.0 * radius) if radius else 0.0
        print(f"\nSPLIT INTO {n_parts}  (each {span:.0f} px, sagitta {sagitta:.1f} px "
              f"against {truth['sagitta_px']:.0f} px for the whole arc)")
        edges = np.linspace(along.min(), along.max(), n_parts + 1)
        for part in range(n_parts):
            keep = (along >= edges[part]) & (along <= edges[part + 1])
            report(f"segment {part + 1}/{n_parts}",
                   detections[keep].reset_index(drop=True), geometry, rng)

    print("\nRead it this way: alternation appearing WITHIN segments but not across the")
    print("whole track means the straight axis is smearing it, and a curved axis would")
    print("recover it. Absent at every scale means the template cannot do this test.")


if __name__ == "__main__":
    main()
