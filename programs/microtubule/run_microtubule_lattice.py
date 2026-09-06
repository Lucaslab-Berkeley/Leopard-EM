"""Measure microtubule lattice parameters from a constrained 2DTM search.

Takes the sparse ``CorrelationTable`` written by a (constrained) ``match_template``
run, picks peaks in position *and* orientation so that overlapping walls of the tube
survive, and reads the filament geometry off the resulting peak set.

What this reports, and how far to trust each number:

* **Polarity** -- solid. The two in-plane poles differ by orders of magnitude when the
  template resolves the filament's direction.
* **Lattice spacing** -- solid, and quoted both absolutely and as a ratio against the
  template's own repeat. Prefer the ratio: it is the form in which a later pixel-size
  calibration is a single multiplication.
* **Protofilament number** -- indicative only. Check the margin over the runner-up; a
  narrow margin means inconclusive, and template competition across protofilament
  numbers is the more reliable route.

It deliberately does *not* attempt the seam. Locating a seam requires telling alpha
from beta, which needs either sub-3-Angstrom resolution or sub-Angstrom axial
precision; neither is reachable here. ``subunit_contrast`` below reports how far away
the template is, as a noise-free upper bound.

Positions and orientations are read from ``refined_*`` columns when present, so run
``refine_template`` on the peak table first for the best numbers.
"""

import mrcfile
import numpy as np

from leopard_em.analysis import (
    TemplateLatticeGeometry,
    axis_points_from_peaks,
    estimate_lattice_rise,
    estimate_polarity,
    estimate_protofilament_number,
    filament_coordinates,
    find_peaks_orientation_aware,
    fit_filament_axis,
)
from leopard_em.analysis.filament_lattice import (
    filament_direction_from_angles,
    rise_from_template_autocorrelation,
    subunit_contrast,
    unwrap_helical_axial_coordinate,
)
from leopard_em.pydantic_models.config.orientation_search import OrientationSearchConfig
from leopard_em.pydantic_models.results.correlation_table import CorrelationTable

CORRELATION_TABLE_PATH = (
    "results_cropped/output_correlation_table_2rings_constrained.h5"
)
TEMPLATE_MODEL_PATH = "models/6dpu_2rings_aligned_zero.pdb"
TEMPLATE_VOLUME_PATH = "maps/GMPCPP_2rings_0.9194_bscale0.5.mrc"
PIXEL_SIZE_ANGSTROM = 0.9194

# Only needed for tables written before correlation-table format version 2, which did
# not store the orientation grid. Regenerating it from the search config is exact.
PSI_STEP, THETA_STEP = 2.5, 3.5

SCORE_THRESHOLD = 8.0


def main() -> None:
    """Report the lattice parameters of one constrained filament search."""
    geometry = TemplateLatticeGeometry.from_pdb(TEMPLATE_MODEL_PATH)
    volume = mrcfile.read(TEMPLATE_VOLUME_PATH).astype(np.float32)
    reference_px = rise_from_template_autocorrelation(
        volume, geometry.rise_angstrom / PIXEL_SIZE_ANGSTROM
    )

    print(f"template  {TEMPLATE_MODEL_PATH}")
    print(
        f"  {geometry.n_protofilaments} protofilaments, radius "
        f"{geometry.subunit_radius_angstrom:.1f} A, axis offset "
        f"{geometry.axis_offset_magnitude_angstrom:.1f} A"
    )
    turns = geometry.turns_are_closed_in_repeats
    closure = (
        "closes on a whole repeat, so the lattice has no discontinuity at this level"
        if abs(turns - round(turns)) < 0.1
        else "cannot close on a whole repeat, which forces a seam"
    )
    print(
        f"  repeat {reference_px:.3f} px by autocorrelation; "
        f"{geometry.axial_shift_per_turn_angstrom:.0f} A per turn = "
        f"{turns:+.2f} repeats -- {closure}"
    )
    contrast = subunit_contrast(volume, 2 * reference_px)
    verdict = (
        "subunits indistinguishable, no seam recoverable"
        if contrast > 0.9
        else "subunits differ, a seam may be recoverable"
    )
    print(f"  subunit contrast {contrast:.3f} -- {verdict}")

    table = CorrelationTable.from_hdf5(CORRELATION_TABLE_PATH)
    euler_angles = (
        None
        if table.euler_angles is not None
        else OrientationSearchConfig(
            base_grid_method="uniform", psi_step=PSI_STEP, theta_step=THETA_STEP
        ).euler_angles
    )
    detections = table.to_detections_dataframe(euler_angles=euler_angles)
    peaks = find_peaks_orientation_aware(detections, score_threshold=SCORE_THRESHOLD)
    print(f"\n{len(detections):,} detections -> {len(peaks)} orientation-aware peaks")

    weights = peaks["z_score"].to_numpy()
    phi = peaks["phi"].to_numpy()
    direction = filament_direction_from_angles(
        phi, peaks["theta"].to_numpy(), peaks["psi"].to_numpy()
    )
    axis_points = axis_points_from_peaks(peaks, geometry, PIXEL_SIZE_ANGSTROM)
    axis = fit_filament_axis(axis_points, weights, direction)
    along, _ = filament_coordinates(axis_points, axis)
    print(
        f"  axis residual {axis.residual_rms_px * PIXEL_SIZE_ANGSTROM:.2f} A, "
        f"direction {axis.angle_deg:.1f} deg, span "
        f"{np.ptp(along) * PIXEL_SIZE_ANGSTROM:.0f} A"
    )

    polarity = estimate_polarity(detections, score_threshold=SCORE_THRESHOLD)
    print(
        f"\nPOLARITY  psi {polarity.psi_deg:.1f} deg, {polarity.score_ratio:.0f}:1 "
        f"({polarity.n_winning} vs {polarity.n_opposite} detections)"
    )

    count, power = estimate_protofilament_number(peaks)
    order = np.argsort(-power)
    margin = power[order[0]] / power[order[1]]
    print(
        f"PROTOFILAMENTS  {count} (margin {margin:.2f}x) -- "
        f"{'indicative only' if margin < 2 else 'reasonably clear'}"
    )

    unwrapped = unwrap_helical_axial_coordinate(
        along * PIXEL_SIZE_ANGSTROM, phi, geometry
    )
    rise = estimate_lattice_rise(
        unwrapped / PIXEL_SIZE_ANGSTROM,
        PIXEL_SIZE_ANGSTROM,
        geometry.rise_angstrom,
        weights,
        reference_rise_angstrom=reference_px * PIXEL_SIZE_ANGSTROM,
        theta_deg=peaks["theta"].to_numpy(),
    )
    sharpness = float(
        np.abs(
            (weights * np.exp(2j * np.pi * unwrapped / geometry.rise_angstrom)).sum()
        )
        / weights.sum()
    )
    print(f"LATTICE   {rise.summary()}")
    print(
        f"          pixel-size-free ratio {rise.rise_pixels / reference_px:.4f}; "
        f"sharpness |R| = {sharpness:.3f}"
    )


# NOTE: Invoking program under `if __name__ == "__main__"` necessary for multiprocesing
if __name__ == "__main__":
    main()
