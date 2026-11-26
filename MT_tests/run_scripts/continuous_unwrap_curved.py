"""Index a bent filament continuously, in ARC length, with no segments.

``extract_lattice_sites`` fits a straight ``FilamentAxis`` and projects onto it, so its
axial coordinate is CHORD length. On a bent tube chord is not arc, and the non-linear
part of the difference mis-indexes sites -- 110 A over region 1 of the GDP frame, which
is 2.7 monomers. Cutting the tube into near-straight segments works around that, but it
costs the thing the register test is starved of: each segment gets its own phase, so the
parity cannot be compared between them and 130 sites become 7 groups of 14-34.

Measuring along a CURVE removes the problem instead of working around it. The whole tube
then carries one lattice and one index, and every site enters one test.

Which curve matters. The hand-drawn path is not accurate enough -- its own axial
distortion is 110 A (region 1) and 31 A (region 2), both larger than the half-monomer
that parity assignment needs. So the curve is fitted to the DETECTIONS' axis points: each
detection independently predicts a point on the axis, so a smooth fit through thousands
of them is far better than a drawing through 53 clicks.

Flexibility is the risk and is capped deliberately. A curve free to wiggle at the lattice
period would absorb the very signal being measured, so the knot spacing is held at or
above ``MIN_KNOT_REPEATS`` repeats -- the axis carries no power near the 42 A frequency.

Usage:
    python run_scripts/continuous_unwrap_curved.py --template patch
"""

import argparse
import pathlib
import sys

import h5py
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from scipy.spatial import cKDTree

from leopard_em.analysis.filament_lattice import (
    TemplateLatticeGeometry,
    axis_points_from_peaks,
    estimate_lattice_rise,
    filament_coordinates,
    filament_direction_from_angles,
    fit_filament_axis,
    unwrap_helical_axial_coordinate,
)
from leopard_em.pydantic_models.results.correlation_table import detections_from_hdf5

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyse_gdp_curved as A  # noqa: E402
from frame_geometry import legacy_frame_geometry  # noqa: E402
from seam_from_patch_search import TEMPLATES, alternation  # noqa: E402

MIN_KNOT_REPEATS = 10.0   # >= 10 x 41 A, so the axis has no power at the lattice period


def fit_axis_curve(points_xy, order_hint, weights, rise_angstrom, pixel_size,
                   samples_per_px=1.0):
    """Smooth curve through the detections' axis points, arc-length parameterised.

    x and y are each fitted as a spline against ORDER_HINT -- the drawn path's arc
    coordinate -- rather than with splprep's own parameterisation. That matters for a
    reason that cost a 10% error before it was found: splprep with a large smoothing
    budget pulls the endpoints inward, and the curve then spans LESS than the data.
    On the synthetic the fitted curve covered 1628 px where the detections spanned 1777
    (8.4% short), the arc coordinate was compressed to match, and the recovered rise came
    out at 36.60 A against a truth of 40.94 -- a 10.6% error that destroyed the register.
    Fitting against a monotone coordinate makes the curve span the data by construction.

    Flexibility is capped with EXPLICIT knots at MIN_KNOT_REPEATS repeats, so the axis
    carries no power near the lattice period and cannot absorb the signal being measured.
    Points are binned first: thousands of detections share axis pixels, and a
    score-weighted mean per bin estimates the axis far better than any single detection.
    """
    knot_spacing_px = MIN_KNOT_REPEATS * rise_angstrom / pixel_size
    bin_width = knot_spacing_px / 6.0
    lo, hi = float(order_hint.min()), float(order_hint.max())
    edges = np.arange(lo, hi + bin_width, bin_width)
    which = np.clip(np.digitize(order_hint, edges) - 1, 0, len(edges) - 2)

    def binned(min_points):
        centres, at = [], []
        for b in range(len(edges) - 1):
            inside = which == b
            if inside.sum() < min_points:
                continue
            w = weights[inside]
            centres.append((points_xy[inside] * w[:, None]).sum(axis=0) / w.sum())
            at.append(float((order_hint[inside] * w).sum() / w.sum()))
        return np.asarray(centres), np.asarray(at)

    try:
        # The original fit: bins of 2+ detections, knots every knot_spacing_px.
        centres, at = binned(2)
        if len(centres) < 5:
            raise ValueError(f"only {len(centres)} occupied bins; cannot fit a curve")
        interior = np.arange(at[0] + knot_spacing_px, at[-1] - knot_spacing_px * 0.5,
                             knot_spacing_px)
        interior = interior[(interior > at[1]) & (interior < at[-2])]
        fit_x = LSQUnivariateSpline(at, centres[:, 0], interior, k=3)
        fit_y = LSQUnivariateSpline(at, centres[:, 1], interior, k=3)
    except ValueError:
        # Only when that fails -- a sparse or patchy tube, e.g. clean refined sites at
        # ~40% occupancy: single-site bins count, and a knot is kept only with at least
        # two bins of data on each side of it, since a knot over an empty stretch breaks
        # the Schoenberg-Whitney condition. Results that already fitted are unchanged.
        centres, at = binned(1)
        if len(centres) < 5:
            raise ValueError(
                f"only {len(centres)} occupied bins; cannot fit a curve") from None
        interior, previous = [], at[0]
        for knot in np.arange(at[0] + knot_spacing_px, at[-1] - knot_spacing_px * 0.5,
                              knot_spacing_px):
            if ((at > previous) & (at <= knot)).sum() >= 2 and (at > knot).sum() >= 2:
                interior.append(float(knot))
                previous = knot
        while True:
            try:
                fit_x = LSQUnivariateSpline(at, centres[:, 0], interior, k=3)
                fit_y = LSQUnivariateSpline(at, centres[:, 1], interior, k=3)
                break
            except ValueError:
                if not interior:
                    raise
                interior = interior[::2] if len(interior) > 1 else []
        interior = np.asarray(interior)

    # Evaluate across the FULL span the data occupies, ends included.
    dense = np.linspace(lo, hi, max(int((hi - lo) * samples_per_px), 400))
    curve, arc = _trace(fit_x, fit_y, dense)
    every, every_at = binned(1)
    if not _overshoots(arc[-1], every, every_at, lo, hi, bin_width):
        return curve, arc, len(interior) + 1

    # Guard. On sparse points the spline can leave the data: past the last occupied
    # bin it is EXTRAPOLATED as a cubic (bins of one point are dropped above, so that
    # can be hundreds of px), and across a knot interval with no data it is unpinned.
    # It then loops away and the arc coordinate jumps -- Stage 6b refitting ~37 refined
    # sites at z >= 8 traced 87,663 px of curve along a 3,400 px tube, and indexed one
    # stretch of sites 600+ lattice indices away from its neighbours. The data's own
    # polyline bounds the arc, so a fit that overshoots it is redone stiffly: every
    # point counts, knots only where data lie on both sides, straight lines past the
    # data's ends, and fewer knots until it stays with the data. Fits that pass the
    # check above are returned exactly as before.
    if len(every) < 5:
        raise ValueError(f"only {len(every)} occupied bins; cannot fit a curve")
    knots, previous = [], every_at[0]
    for knot in np.arange(every_at[0] + knot_spacing_px,
                          every_at[-1] - knot_spacing_px * 0.5, knot_spacing_px):
        if ((every_at > previous) & (every_at <= knot)).sum() >= 2 and \
                (every_at > knot).sum() >= 2:
            knots.append(float(knot))
            previous = knot
    while True:
        try:
            fit_x = LSQUnivariateSpline(every_at, every[:, 0], knots, k=3)
            fit_y = LSQUnivariateSpline(every_at, every[:, 1], knots, k=3)
        except ValueError:
            if not knots:
                raise
            knots = knots[::2] if len(knots) > 1 else []
            continue
        curve, arc = _trace(fit_x, fit_y, dense, (every_at[0], every_at[-1]))
        if not knots or not _overshoots(arc[-1], every, every_at, lo, hi, bin_width):
            return curve, arc, len(knots) + 1
        knots = knots[::2] if len(knots) > 1 else []


def _trace(fit_x, fit_y, dense, data_span=None):
    """Curve samples and their cumulative arc. With ``data_span`` the splines are used
    only inside it and continue straight along their end tangents outside it."""
    curve = np.column_stack([fit_x(dense), fit_y(dense)])
    if data_span is not None:
        for end, beyond in ((data_span[0], dense < data_span[0]),
                            (data_span[1], dense > data_span[1])):
            if beyond.any():
                at_end = np.array([fit_x(end), fit_y(end)], dtype=np.float64).ravel()
                slope = np.array([fit_x.derivative()(end), fit_y.derivative()(end)],
                                 dtype=np.float64).ravel()
                curve[beyond] = at_end + np.outer(dense[beyond] - end, slope)
    step = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    return curve, np.concatenate([[0.0], np.cumsum(step)])


OVERSHOOT_RATIO = 1.05   # a curve through the data is no longer than its polyline


def _overshoots(arc_length, centres, at, lo, hi, bin_width):
    """True if the curve is longer than the data can explain: the polyline through the
    single-point bins (which a smooth curve through them can only undercut), plus the
    stretches of ordering coordinate beyond the first and last of them."""
    polyline = float(np.linalg.norm(np.diff(centres, axis=0), axis=1).sum())
    polyline += (at[0] - lo) + (hi - at[-1])
    return arc_length > OVERSHOOT_RATIO * polyline + 2.0 * bin_width


def project(curve, arc, points_xy):
    """Arc coordinate of each point, by nearest curve sample."""
    _, index = cKDTree(curve).query(points_xy)
    return arc[index]


def order_from_data(points, boot_points, boot_scores, boot):
    """An ordering coordinate along the tube, derived ONLY from the detections.

    The drawn path must not be used for this. It was, and it cost a silent sign error:
    the arc then ran whichever way the path happened to be drawn, the helical unwrap
    term fought the lattice instead of removing it, and the register vanished while the
    fitted period still looked right. ``fit_filament_axis`` pins the direction against
    the orientations the detections themselves report, so taking the ordering from it
    makes the sign a property of the data and not of the annotation.
    """
    reference = filament_direction_from_angles(
        boot["phi"].to_numpy(), boot["theta"].to_numpy(), boot["psi"].to_numpy())
    axis = fit_filament_axis(boot_points, boot_scores, reference_direction=reference)
    return (filament_coordinates(points, axis)[0],
            filament_coordinates(boot_points, axis)[0])


def point_on_curve(arc_px, arc_table, curve):
    """Points at arc lengths ``arc_px`` along ``curve``, straight on past its ends.

    ``np.interp`` alone clamps: a site a little beyond the last fitted point -- the end
    sites of a tube, whose axis points fall outside the bootstrap's span -- would be
    put AT the end point, up to hundreds of Angstroms from where it is.
    """
    arc_px = np.asarray(arc_px, dtype=np.float64)
    xy = np.column_stack([np.interp(arc_px, arc_table, curve[:, 0]),
                          np.interp(arc_px, arc_table, curve[:, 1])])
    for end, beyond, (a, b) in ((0, arc_px < arc_table[0], (0, 1)),
                                (-1, arc_px > arc_table[-1], (-2, -1))):
        if beyond.any():
            tangent = curve[b] - curve[a]  # forward, at either end
            tangent = tangent / np.linalg.norm(tangent)
            xy[beyond] = curve[end] + np.outer(arc_px[beyond] - arc_table[end], tangent)
    return xy


def continuous_sites(detections, geometry, bootstrap, *, pixel_size, n_iterations=3,
                     xy_radius_px=12.0, angular_radius_deg=15.0, n_curve_passes=2,
                     resolve_protofilaments=False):
    """Lattice sites indexed continuously along a curve fitted to the DETECTIONS.

    ``pixel_size`` is the searched micrograph's, from the table (``frame_geometry``):
    it sets the knot spacing and turns every position into Angstroms.

    Nothing here comes from the hand-drawn path -- not the shape, not the arc length and
    above all not the direction. See ``order_from_data``.

    Otherwise this mirrors ``extract_lattice_sites`` step for step, so the only
    difference is arc length against chord length.

    ``resolve_protofilaments`` keeps one site per (axial index, protofilament) pair
    instead of collapsing all N protofilaments onto one axial lattice. The helical
    unwrap folds every protofilament onto a common 1-D lattice, so the default keeps
    only the BEST detection at each axial index and throws away the other twelve --
    a 13-fold loss of independent sites, which is ruinous for a test as short of them
    as the register. Each (axial, protofilament) pair is a different physical subunit,
    and the register is still read from the parity of the axial index.

    ⚠️ Only valid for a template whose azimuth identifies a protofilament. A complete
    ring is N-fold pseudo-symmetric, so its phi does not, and the breakdown is
    meaningless; the 5-PF patch sits 85.8 A off the axis and its phi does.
    """
    from leopard_em.analysis.correlation_peaks import find_peaks_orientation_aware

    points = axis_points_from_peaks(detections, geometry, pixel_size)
    scores = detections["z_score"].to_numpy()
    phi = detections["phi"].to_numpy()
    theta = detections["theta"].to_numpy()

    boot = find_peaks_orientation_aware(
        detections, xy_radius_px=xy_radius_px,
        angular_radius_deg=angular_radius_deg, score_column="z_score",
        score_threshold=bootstrap)
    if len(boot) < 20:
        raise ValueError(f"only {len(boot)} bootstrap peaks above {bootstrap}")
    boot_points = axis_points_from_peaks(boot, geometry, pixel_size)
    boot_scores = boot["z_score"].to_numpy()
    boot_phi = boot["phi"].to_numpy()

    # Ordering from the data. A straight fit is only used to SORT points along the tube
    # and to fix which way is forward; it never measures anything.
    hint, boot_hint = order_from_data(points, boot_points, boot_scores, boot)
    straight = hint.copy()

    # Principal-curve passes: fit a curve against the current ordering, re-project to get
    # a better one. A straight projection is not monotone along a tube that turns far, so
    # one pass is not enough once the bend is large.
    for _ in range(max(n_curve_passes, 1)):
        curve, arc_table, n_knots = fit_axis_curve(
            boot_points, boot_hint, boot_scores, geometry.rise_angstrom, pixel_size)
        hint = project(curve, arc_table, points)
        boot_hint = project(curve, arc_table, boot_points)

    along = hint * pixel_size
    boot_along = boot_hint * pixel_size

    # The curve's own parameterisation may still run either way; pin it against the
    # data-derived straight coordinate, which is itself pinned by the detections' angles.
    flipped, span = False, 0.0
    if np.corrcoef(boot_along, filament_coordinates(
            boot_points, fit_filament_axis(
                boot_points, boot_scores,
                reference_direction=filament_direction_from_angles(
                    boot_phi, boot["theta"].to_numpy(), boot["psi"].to_numpy()),
            ))[0])[0, 1] < 0:
        span = max(along.max(), boot_along.max())
        along, boot_along = span - along, span - boot_along
        flipped = True
    assert np.corrcoef(boot_along, filament_coordinates(
        boot_points, fit_filament_axis(
            boot_points, boot_scores,
            reference_direction=filament_direction_from_angles(
                boot_phi, boot["theta"].to_numpy(), boot["psi"].to_numpy()),
        ))[0])[0, 1] > 0, "arc direction is not pinned to the data"

    rise = float(geometry.rise_angstrom)
    rise_error = float("nan")
    selected = np.empty(0, dtype=np.int64)
    axial = np.empty(0, dtype=np.int64)
    deviation = np.empty(0)
    for _ in range(max(n_iterations, 1)):
        unwrapped = unwrap_helical_axial_coordinate(along, phi, geometry)
        boot_unwrapped = unwrap_helical_axial_coordinate(boot_along, boot_phi, geometry)
        phase = (np.angle((boot_scores
                           * np.exp(2j * np.pi * boot_unwrapped / rise)).sum())
                 / (2 * np.pi) * rise)
        fractional = (unwrapped - phase) / rise
        used_phase, used_rise = phase, rise
        index_all = np.round(fractional).astype(np.int64)
        if resolve_protofilaments:
            spacing = geometry.protofilament_angular_spacing_deg
            pf_all = (np.round(phi / spacing).astype(np.int64)
                      % geometry.n_protofilaments)
            key = ((index_all - index_all.min()) * geometry.n_protofilaments) + pf_all
        else:
            pf_all = None
            key = index_all - index_all.min()
        order = np.lexsort((-scores, key))
        first = np.concatenate(([True], key[order][1:] != key[order][:-1]))
        selected = order[first]
        axial = index_all[selected]
        deviation = (fractional[selected] - axial) * rise
        fitted = estimate_lattice_rise(
            unwrapped[selected] / pixel_size, pixel_size, rise,
            scores[selected], theta_deg=theta[selected],
        )
        rise, rise_error = fitted.rise_angstrom, fitted.standard_error_angstrom
    # Where each selected site SHOULD be on the fitted lattice -- set by the curve, the
    # phase and the rise, which thousands of detections determine, not by the one
    # detection that won the site. A neutral starting point for scoring rival templates.
    def predict(axial_index, phi_deg):
        """Image position of lattice site ``axial_index`` seen at roll ``phi_deg``."""
        axial_index = np.asarray(axial_index, dtype=np.float64)
        helical = unwrap_helical_axial_coordinate(
            np.zeros(len(axial_index)), np.asarray(phi_deg, dtype=np.float64), geometry)
        along_pred = used_phase + axial_index * used_rise - helical
        arc_px = ((span - along_pred) if flipped else along_pred) / pixel_size
        return point_on_curve(arc_px, arc_table, curve)

    predicted_xy = predict(axial, phi[selected])
    n_axial = int(axial.max() - axial.min() + 1)
    n_predicted = (n_axial * geometry.n_protofilaments
                   if resolve_protofilaments else n_axial)
    return {
        "detection_index": selected, "axial_index": axial,
        "score": scores[selected], "deviation": deviation,
        "rise": rise, "rise_standard_error": rise_error,
        # ``deviation`` is PROJECTED Angstroms against a lattice of this spacing; a
        # local projected spacing is indexing_rise + d(deviation)/d(index), and only
        # after dividing by sin(theta) is it a rise.
        "indexing_rise": used_rise,
        "n_predicted": n_predicted, "n_knots": n_knots,
        "protofilament": None if pf_all is None else pf_all[selected],
        "arc_span_px": float(np.ptp(along[selected]) / pixel_size),
        "predicted_xy": predicted_xy, "predict": predict,
        "straight_span_px": float(np.ptp(straight)),
    }


def main() -> None:
    """Index each region continuously and run the register test on all its sites."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default="patch", choices=sorted(TEMPLATES))
    parser.add_argument("--results", default="results_gdp_curved")
    parser.add_argument("--constraint", type=pathlib.Path,
                        default=MT_ROOT / "configs"
                        / "filament_constraint_gdp_curved.h5")
    parser.add_argument("--min-z", type=float, default=6.0)
    parser.add_argument("--resolve-protofilaments", action="store_true",
                        help="one site per (axial index, protofilament); patch only")
    args = parser.parse_args()

    tag, stem = TEMPLATES[args.template]
    geometry = TemplateLatticeGeometry.from_pdb(str(MT_ROOT / "models" / f"{stem}.pdb"))
    table = MT_ROOT / args.results / f"output_correlation_table_{tag}.h5"
    detections = detections_from_hdf5(
        str(table), min_z_score=args.min_z).reset_index(drop=True)
    frame = legacy_frame_geometry(table, stem)
    with h5py.File(args.constraint, "r") as handle:
        region_map = handle["maps/region_id"][:]
        drawn = {int(n): handle[f"regions/{n}/path"][:] for n in handle["regions"]}

    y = np.clip(detections["y"].to_numpy() + frame.half_width, 0,
                region_map.shape[0] - 1)
    x = np.clip(detections["x"].to_numpy() + frame.half_width, 0,
                region_map.shape[1] - 1)
    owner = region_map[y, x]

    rng = np.random.default_rng(0)
    print(f"template {args.template} ({stem})")
    print(frame.describe())
    print(f"{'region':<8} {'sites':>6} {'occ':>6} {'rise Å':>9} {'dev rms':>9} "
          f"{'knots':>6} {'amp':>7} {'chance':>8} {'ratio':>7} {'ctrl':>7}")
    for region in sorted(drawn):
        mask = owner == region
        if mask.sum() < 200:
            continue
        subset = detections[mask].reset_index(drop=True)
        boot = A.pick_bootstrap(subset["z_score"].to_numpy(), want=20)
        if boot is None:
            continue
        try:
            sites = continuous_sites(subset, geometry, boot,
                                     pixel_size=frame.pixel_size,
                                     resolve_protofilaments=args.resolve_protofilaments)
        except ValueError as error:
            print(f"{region:<8} {error}")
            continue
        index = sites["axial_index"]
        score = sites["score"]
        amplitude, null = alternation(index, score, rng)
        ctrl_amp, ctrl_null = alternation(index // 2, score, rng)
        occupancy = len(index) / max(sites["n_predicted"], 1)
        print(f"{region:<8} {len(index):6d} {occupancy:5.0%} {sites['rise']:9.3f} "
              f"{np.sqrt((sites['deviation'] ** 2).mean()):8.2f}Å {sites['n_knots']:6d} "
              f"{amplitude:7.3f} {null:8.3f} {amplitude / null:6.2f}x "
              f"{ctrl_amp / ctrl_null:6.2f}x")
    print("\n  Segmented, the same data gave 14-34 sites per test and a patch median of")
    print("  0.17x. If the continuous index is sound the site count per test should rise")
    print("  several-fold with the deviation staying near the segmented 2-5 Å; a large")
    print("  deviation means the fitted curve's arc length is drifting and the index is")
    print("  not to be trusted.")


if __name__ == "__main__":
    main()
