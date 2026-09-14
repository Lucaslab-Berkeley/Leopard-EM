"""The four readouts on the REAL curved GDP micrograph, per drawn region.

Nothing here has a right answer, which is the whole difference from
``analyse_curved_synthetic.py``. Every number therefore has to carry its own control,
and the drawn path itself supplies the strongest one available: it is an independent
measurement of where the tube is and which way it points, made by eye before any search
ran. Polarity is scored against the path's own tangent; the straight-axis residual is
scored against the path's own departure from a line.

The second difference is that these tubes are bent far harder than the synthetic. The
chord-vs-arc rise bias goes as

    delta u / u  ~  -L^2 / (40 R^2)

so it is quadratic in the length measured over, and on a path that turns 90 degrees it
reaches several percent -- against the 2.5 percent that separates an expanded lattice
from a compacted one. A single straight axis over the whole track cannot answer the
lattice question at all. It can be made to, though, by measuring over SEGMENTS short
enough that their own bias is small, because dividing the span by k divides the bias by
k^2. That is what this does: it picks the segment count from the drawn curvature and a
bias tolerance, reports the rise per segment, and takes the scatter between segments as
the error bar -- which is an honest one, since the segments are independent stretches of
tube.

Segments are cut on arc length along the DRAWN curve, not on a straight fit. On a tube
that turns 90 degrees a straight fit's "along" coordinate is not monotonic in position,
so cutting on it would not give contiguous pieces of tube.

Usage:
    python run_scripts/analyse_gdp_curved.py --tag gdp_curved_13pf
    python run_scripts/analyse_gdp_curved.py --tag gdp_curved_13pf_patch --region 1
"""

import argparse
import importlib.util
import math
import pathlib
import sys

import h5py
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
REPO_ROOT = MT_ROOT.parent
PIXEL_SIZE = 0.9194

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from setup_gdp_curved import SEARCHES  # noqa: E402

# Same smoothing the constraint was rasterised with, so the curve analysed here is the
# curve the search was actually constrained to.
SMOOTH_PX = 5.0
# How far the chord coordinate may drift from the arc coordinate, NON-LINEARLY, inside
# one segment. Sites are assigned by rounding the axial coordinate to the nearest
# multiple of the repeat, so a drift approaching half a repeat mis-indexes them; a
# LINEAR drift is not the problem, since it is absorbed into the fitted rise and
# divided out. Calibrated on the curved synthetic, whose residual works out at about
# 2.3 A and which recovered its rise to 0.006 A -- 5 A is a quarter of the +/-20.5 A
# rounding window and twice what is known to work.
MAX_AXIAL_DISTORTION_ANGSTROM = 5.0
MONOMER_RISE_ANGSTROM = 41.0  # only for reporting a segment length in repeats
MAX_SEGMENTS = 40


def _load_path_tool():
    """Import the standalone napari tool for its spline, without importing napari."""
    path = REPO_ROOT / "programs" / "constrained_search" / "napari_choose_filament_path.py"
    spec = importlib.util.spec_from_file_location("_filament_path_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATH_TOOL = _load_path_tool()


def header(text: str) -> None:
    """A labelled section, so the report reads top to bottom."""
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


class DrawnCurve:
    """The hand-drawn centreline, resampled and differentiated.

    This is the semi-ground-truth of the whole exercise. It is not precise -- it is a
    hand drawing -- but it is independent of the template search, so anything it agrees
    with was not produced by the search agreeing with itself.
    """

    def __init__(self, points_yx: np.ndarray) -> None:
        self.curve, self.tangent = PATH_TOOL.spline_through_points(
            points_yx, smooth_px=SMOOTH_PX
        )
        step = np.linalg.norm(np.diff(self.curve, axis=0), axis=1)
        self.arc = np.concatenate([[0.0], np.cumsum(step)])
        psi = PATH_TOOL.psi_from_tangent(self.tangent)
        self.psi = np.degrees(np.unwrap(np.radians(psi)))
        # kappa = d(psi)/ds. Sign carries which way it bends; magnitude is 1/R.
        self.kappa = np.gradient(np.radians(self.psi), self.arc)

    @property
    def length_px(self) -> float:
        """Arc length of the drawn path."""
        return float(self.arc[-1])

    @property
    def chord_px(self) -> float:
        """Straight-line distance between the two ends."""
        return float(np.linalg.norm(self.curve[-1] - self.curve[0]))

    @property
    def turn_deg(self) -> float:
        """End-to-end change in the tangent angle."""
        return float(self.psi[-1] - self.psi[0])

    def radius_px(self, lo: float = 0.0, hi: float | None = None) -> float:
        """Radius of curvature over an arc window, from the median |kappa|.

        Only a summary for reporting. It is not used to predict the bias, because a
        single kappa describes a circular arc and these paths are not circular: region
        2 turns 60 degrees end to end yet its arc exceeds its chord by under 2 percent,
        which only an S-bend does. Averaging |kappa| over a sign change overstates how
        far the tube actually departs from a line.
        """
        hi = self.length_px if hi is None else hi
        window = (self.arc >= lo) & (self.arc <= hi)
        kappa = np.median(np.abs(self.kappa[window])) if window.any() else 0.0
        return float("inf") if kappa <= 0 else float(1.0 / kappa)

    def _window(self, lo: float, hi: float | None) -> tuple[np.ndarray, np.ndarray]:
        """Curve samples and their arc coordinates inside an arc window."""
        hi = self.length_px if hi is None else hi
        mask = (self.arc >= lo) & (self.arc <= hi)
        if mask.sum() < 3:
            raise ValueError(f"arc window [{lo:.0f}, {hi:.0f}] holds too few samples")
        return self.curve[mask], self.arc[mask]

    def shortening(self, lo: float = 0.0, hi: float | None = None) -> float:
        """Fractional rise bias a straight axis carries over this arc window, EXACTLY.

        A straight-axis analysis measures every position as its projection onto one
        least-squares line. Sites uniformly spaced along the arc project onto that line
        with a mean spacing of (projected span) / (arc length) times the true one, so
        this ratio *is* the bias on the fitted repeat -- no circular-arc assumption, no
        small-angle expansion, and correct for an S-bend where the -L^2/(40 R^2) law is
        not.
        """
        points, arc = self._window(lo, hi)
        centroid = points.mean(axis=0)
        _, _, basis = np.linalg.svd(points - centroid, full_matrices=False)
        along = (points - centroid) @ basis[0]
        return float(np.ptp(along) / (arc[-1] - arc[0]) - 1.0)

    def straight_residual_px(self, lo: float = 0.0,
                             hi: float | None = None) -> tuple[float, float]:
        """Rms and peak departure of the drawn curve from its own best-fit line."""
        points, _ = self._window(lo, hi)
        centroid = points.mean(axis=0)
        _, _, basis = np.linalg.svd(points - centroid, full_matrices=False)
        across = (points - centroid) @ basis[1]
        return float(np.sqrt((across**2).mean())), float(np.abs(across).max())

    def axial_distortion_angstrom(self, lo: float = 0.0,
                                  hi: float | None = None) -> tuple[float, float]:
        """Axial error a straight axis cannot remove by rescaling, rms and peak.

        ``extract_lattice_sites`` measures every site's axial position by projecting
        onto one line, and the lattice is uniform in ARC length, so what it sees is the
        chord coordinate t as a function of the arc coordinate s. The best straight-line
        relation between them is exactly the shortening, and that is absorbed into the
        fitted repeat. Only the RESIDUAL mis-assigns sites -- it is what makes the
        deviation-against-index plot bow -- so it, not the sideways bow, is the quantity
        that has to stay small.
        """
        points, arc = self._window(lo, hi)
        centroid = points.mean(axis=0)
        _, _, basis = np.linalg.svd(points - centroid, full_matrices=False)
        along = (points - centroid) @ basis[0]
        residual = along - np.polyval(np.polyfit(arc, along, 1), arc)
        return (float(np.sqrt((residual**2).mean()) * PIXEL_SIZE),
                float(np.abs(residual).max() * PIXEL_SIZE))

    def psi_at(self, arc_value: float) -> float:
        """Tangent angle at an arc position, in degrees modulo 360."""
        return float(np.interp(arc_value, self.arc, self.psi) % 360.0)

    def project(self, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Arc coordinate and signed transverse offset of each point, in pixels.

        The curve is sampled at about one point per pixel, so a nearest-sample search
        is already accurate to half a pixel -- far below anything measured here, and
        cheaper and more robust than a Newton refinement on a hand-drawn spline.
        """
        from scipy.spatial import cKDTree

        points_yx = points_xy[:, ::-1]
        _, index = cKDTree(self.curve).query(points_yx)
        delta = points_yx - self.curve[index]
        tangent = self.tangent[index]
        # 2-D cross product: positive is to the left of the direction of travel.
        across = tangent[:, 0] * delta[:, 1] - tangent[:, 1] * delta[:, 0]
        return self.arc[index], across


def bias_fraction(length_px: float, radius_px: float) -> float:
    """The -L^2/(40 R^2) law, kept only as a cross-check on the exact measurement.

    Negative: the chord is shorter than the arc, so the repeat reads low. The 1/40
    rather than the 1/24 of a raw end-to-end chord is because a least-squares line is a
    better approximation than the chord (confirmed against the synthetic). It assumes a
    circular arc, which these hand-drawn paths are not.
    """
    if not np.isfinite(radius_px) or radius_px <= 0:
        return 0.0
    return -(length_px**2) / (40.0 * radius_px**2)


def segment_table(curve: DrawnCurve, k: int) -> tuple[float, float, float, float]:
    """Worst bias, worst bow, worst axial distortion and repeats, for k segments."""
    edges = np.linspace(0.0, curve.length_px, k + 1)
    bias, bow, distortion = [], [], []
    for i in range(k):
        bias.append(abs(curve.shortening(edges[i], edges[i + 1])))
        bow.append(curve.straight_residual_px(edges[i], edges[i + 1])[1])
        distortion.append(curve.axial_distortion_angstrom(edges[i], edges[i + 1])[1])
    repeats = (curve.length_px / k) * PIXEL_SIZE / MONOMER_RISE_ANGSTROM
    return max(bias), max(bow), max(distortion), repeats


def segment_count(curve: DrawnCurve) -> int:
    """Fewest equal segments that keep site indexing intact.

    Neither the rise bias nor the sideways bow sets this. The bias is measured per
    segment and divided out; the bow costs nothing at all, because
    ``extract_lattice_sites`` discards the transverse coordinate entirely (``along, _ =
    filament_coordinates(...)``) and applies no transverse gate. What breaks is site
    ASSIGNMENT, which rounds the axial coordinate to the nearest repeat.
    """
    for k in range(1, MAX_SEGMENTS + 1):
        if segment_table(curve, k)[2] <= MAX_AXIAL_DISTORTION_ANGSTROM:
            return k
    return MAX_SEGMENTS


def report_geometry(index: int, curve: DrawnCurve, n_detections: int) -> int:
    """What the drawing says about this tube, before any measurement."""
    header(f"REGION {index} -- the drawn path, which nothing below depends on")
    radius = curve.radius_px()
    excess = curve.length_px / curve.chord_px - 1.0
    residual_rms, residual_max = curve.straight_residual_px()
    print(f"  {n_detections:,} detections inside this region")
    print(f"  arc {curve.length_px:.0f} px ({curve.length_px * PIXEL_SIZE / 10:.0f} nm), "
          f"chord {curve.chord_px:.0f} px -- arc exceeds chord by {100 * excess:.2f}%")
    print(f"  psi {curve.psi_at(0):.1f}° → {curve.psi_at(curve.length_px):.1f}°, "
          f"end-to-end turn {curve.turn_deg:+.1f}°")
    print(f"  median radius of curvature {radius:.0f} px "
          f"({radius * PIXEL_SIZE / 1e4:.2f} µm)")
    print(f"  departure from its own best-fit line: {residual_rms:.0f} px rms, "
          f"{residual_max:.0f} px peak")

    whole = curve.shortening()
    print(f"\n  straight-axis rise bias over the WHOLE arc: {100 * whole:+.2f}% "
          f"(measured from the drawing)")
    print(f"    the -L²/(40R²) law would say "
          f"{100 * bias_fraction(curve.length_px, radius):+.2f}%; they agree only "
          f"when the tube is a\n    circular arc, and the measured one is what is "
          f"used below")
    if abs(whole) > 0.025:
        print("    that exceeds the 2.5% separating expanded from compacted, so a")
        print("    single straight axis cannot answer the lattice question here")

    # Two things pull against each other: short segments are straight enough to
    # measure, long ones have the repeats to measure precisely. Print the trade-off
    # rather than hiding the choice.
    k = segment_count(curve)
    print(f"\n  {'k':>3} {'segment px':>11} {'repeats':>9} {'bow px':>9} "
          f"{'bias':>9} {'axial Å':>9}")
    for trial in sorted({1, 2, 4, 6, 8, 10, 12, 16, 20, k}):
        if trial > MAX_SEGMENTS:
            continue
        bias, bow, distortion, repeats = segment_table(curve, trial)
        mark = "  <- chosen" if trial == k else ""
        print(f"  {trial:3d} {curve.length_px / trial:11.0f} {repeats:9.1f} "
              f"{bow:9.1f} {100 * bias:8.3f}% {distortion:9.2f}{mark}")
    print(f"  k is set by the last column against {MAX_AXIAL_DISTORTION_ANGSTROM:.0f} Å."
          f" The bow costs nothing (no transverse\n  gate exists) and the bias is "
          f"divided out per segment, so neither sets k.")
    return k


def report_polarity(detections, curve: DrawnCurve, arc: np.ndarray, k: int) -> None:
    """Which way the tube points -- scored against the drawing, not against nothing.

    ``estimate_polarity`` uses psi alone and takes no axis, so bending does not break
    it. What bending does is spread psi across the track, which dilutes a single
    whole-track answer. Per segment the spread is small and each answer has the drawn
    tangent to be checked against.
    """
    header("1. POLARITY (psi only -- no axis, so curvature cannot bias it)")
    whole = estimate_polarity(detections, score_threshold=7.0)
    print(f"  whole track: psi {whole.psi_deg:.2f}°, ratio {whole.score_ratio:.2f}:1 "
          f"({whole.n_winning} vs {whole.n_opposite})")
    print("  the tube turns, so this is a chord average -- read the segments instead\n")

    print(f"  {'segment':<12} {'measured psi':>13} {'drawn psi':>11} {'|diff|':>8} "
          f"{'ratio':>7} {'n':>6}")
    edges = np.linspace(0.0, curve.length_px, k + 1)
    for part in range(k):
        keep = (arc >= edges[part]) & (arc <= edges[part + 1])
        if keep.sum() < 20:
            print(f"  {part + 1}/{k:<10} only {int(keep.sum())} detections, skipped")
            continue
        estimate = estimate_polarity(
            detections[keep].reset_index(drop=True), score_threshold=7.0
        )
        drawn = curve.psi_at(0.5 * (edges[part] + edges[part + 1]))
        # estimate_polarity reports the pole; the drawing has no polarity, only an
        # axis, so the comparison is modulo 180.
        difference = abs(((estimate.psi_deg - drawn + 90.0) % 180.0) - 90.0)
        print(f"  {f'{part + 1}/{k}':<12} {estimate.psi_deg:12.2f}° "
              f"{drawn:10.2f}° {difference:7.2f}° {estimate.score_ratio:6.2f}:1 "
              f"{int(keep.sum()):6d}")
    print("\n  A ratio near 1 means the template cannot tell the two ends apart, "
          "whatever\n  the psi says. Agreement with the drawn tangent says the "
          "detections sit on\n  the tube that was drawn, which is the only external "
          "check available here.")


def report_protofilaments(detections) -> None:
    """Protofilament number from the azimuth harmonic. Takes no axis either."""
    header("2. PROTOFILAMENT NUMBER (azimuth harmonic on phi)")
    strong = detections[detections["z_score"] > 7.0]
    if len(strong) < 50:
        print(f"  only {len(strong)} detections above z 7, not enough")
        return
    best, power = estimate_protofilament_number(strong, max_count=20)
    orders = np.arange(2, 2 + len(power))
    window = (orders >= 11) & (orders <= 16)
    print(f"  {len(strong):,} detections above z 7")
    print("  order:  " + "  ".join(f"{o:6d}" for o in orders[window]))
    print("  power:  " + "  ".join(f"{p:6.3f}" for p in power[window]))
    others = power[window][orders[window] != best]
    margin = power[orders == best][0] / others.max() if len(others) else float("inf")
    print(f"  best overall {best}, margin over the runner-up in 11-16 {margin:.2f}x")
    print("  This is independent of the template competition, which is the point --")
    print("  agreement between a harmonic and a set of rival templates is evidence;")
    print("  either alone is one method's opinion.")


def report_axis(detections, curve: DrawnCurve, geometry, arc: np.ndarray,
                across: np.ndarray) -> None:
    """How badly a straight axis misses, against what the drawing says it should."""
    header("3. AXIS -- a straight fit against a path that is visibly bent")
    strong = detections[detections["z_score"] > 8.0]
    if len(strong) < 30:
        print(f"  only {len(strong)} detections above z 8, skipped")
        return
    points = axis_points_from_peaks(strong, geometry, PIXEL_SIZE)
    try:
        reference = filament_direction_from_angles(
            strong["phi"].to_numpy(), strong["theta"].to_numpy(),
            strong["psi"].to_numpy(),
        )
    except ValueError as error:
        # Expected on a strongly bent tube: the per-peak directions fan out and the
        # global mean cancels. That is itself the measurement.
        print(f"  filament_direction_from_angles: {error}")
        print("  (the per-peak directions fan out -- exactly what a bent tube does to "
              "a global mean)")
        reference = None
    axis = fit_filament_axis(
        points, strong["z_score"].to_numpy(), reference_direction=reference
    )
    along, straight_across = filament_coordinates(points, axis)

    span = float(np.ptp(along))
    # Judge the fit over the stretch of tube the detections actually cover, not the
    # whole drawn path: a short piece of a gentle bend really is nearly straight, and
    # calling that a failure would be wrong.
    expected, expected_max = curve.straight_residual_px(
        float(arc.min()), float(arc.max())
    )
    print(f"  {len(strong):,} detections above z 8, spanning {span:.0f} px "
          f"(arc {arc.min():.0f}-{arc.max():.0f} px of the drawing)")
    print(f"  straight fit: angle {axis.angle_deg:.2f}°, residual "
          f"{axis.residual_rms_px:.2f} px")
    print(f"  the drawn curve departs from ITS own best-fit line over that same "
          f"stretch by\n  {expected:.1f} px rms ({expected_max:.0f} px peak) -- that "
          f"is what a straight fit has to absorb")
    print(f"  scatter about the DRAWN curve instead: "
          f"{np.sqrt((across**2).mean()):.2f} px rms, "
          f"|max| {np.abs(across).max():.0f} px")
    ratio = axis.residual_rms_px / max(np.sqrt((across**2).mean()), 1e-9)
    print(f"  the curve is tighter by {ratio:.1f}x -- that ratio IS the curvature, "
          "measured\n  two independent ways (template positions and a hand drawing)")
    print(f"\n  transverse offset against the straight fit, in 9 bins along it:")
    order = np.argsort(along)
    bins = np.array_split(order, 9)
    print("    along (px): " + "  ".join(f"{along[b].mean():8.0f}" for b in bins))
    print("    across(px): " + "  ".join(f"{straight_across[b].mean():8.1f}" for b in bins))


def report_spacing(detections, curve: DrawnCurve, geometry, arc: np.ndarray,
                   k: int) -> None:
    """Lattice spacing, measured over segments short enough to be nearly straight."""
    header("4. LATTICE SPACING (per segment, so the chord bias stays small)")
    template_rise = geometry.rise_angstrom
    edges = np.linspace(0.0, curve.length_px, k + 1)
    span = curve.length_px / k

    print(f"  template repeat {template_rise:.3f} Å; {k} segments of {span:.0f} px\n")
    print(f"  {'segment':<10} {'sites':>6} {'occ':>6} {'rise Å':>9} {'bias':>8} "
          f"{'corrected':>10} {'vs template':>12} {'dev rms Å':>10}")

    rises, weights = [], []
    for part in range(k):
        keep = (arc >= edges[part]) & (arc <= edges[part + 1])
        subset = detections[keep].reset_index(drop=True)
        if len(subset) < 200:
            print(f"  {f'{part + 1}/{k}':<10} only {len(subset)} detections, skipped")
            continue
        try:
            sites = extract_lattice_sites(
                subset, geometry, PIXEL_SIZE, bootstrap_score_threshold=8.0
            )
        except ValueError as error:
            print(f"  {f'{part + 1}/{k}':<10} {error}")
            continue
        rise = float(sites.rise_angstrom)
        # Each segment has its own shortening -- the bend is not uniform along a
        # hand-drawn tube, so one number for all of them would be wrong.
        bias = curve.shortening(edges[part], edges[part + 1])
        corrected = rise / (1.0 + bias)
        deviation = np.asarray(sites.axial_deviation_angstrom)
        n_sites = len(sites.detection_index)
        print(f"  {f'{part + 1}/{k}':<10} {n_sites:6d} {sites.occupancy:5.0%} "
              f"{rise:9.3f} {100 * bias:+7.3f}% {corrected:10.3f} "
              f"{100 * (corrected - template_rise) / template_rise:+11.2f}% "
              f"{np.sqrt((deviation**2).mean()):10.2f}")
        rises.append(corrected)
        weights.append(n_sites)

    if len(rises) < 2:
        print("\n  too few segments measured to combine")
        return
    rises, weights = np.asarray(rises), np.asarray(weights, dtype=float)
    mean = float(np.average(rises, weights=weights))
    # Scatter BETWEEN segments, not within one. Segments are separate stretches of
    # tube, so this includes everything that varies along the filament -- which is
    # what an error bar on a real measurement has to include.
    scatter = float(np.std(rises, ddof=1))
    print(f"\n  weighted mean of the corrected rises {mean:.3f} Å, "
          f"between-segment scatter {scatter:.3f} Å "
          f"(sem {scatter / math.sqrt(len(rises)):.3f})")
    print(f"  that is {100 * (mean - template_rise) / template_rise:+.2f}% "
          f"from this template's own repeat; expanded and compacted differ by 2.5%")
    print(f"  spread across segments {100 * np.ptp(rises) / mean:.2f}% -- if that is "
          f"real rather than\n  noise, different parts of this tube have different "
          f"spacings, which is the\n  question the segment-wise measurement exists to "
          f"ask")


def alternation(index: np.ndarray, score: np.ndarray, rng, n_null: int = 2000):
    """Period-2 amplitude of score against axial index, and its permutation null."""
    weight = score - score.mean()
    phase = np.exp(1j * np.pi * index)
    amplitude = float(np.abs((weight * phase).sum()) / len(index))
    null = float(np.percentile(
        [float(np.abs((rng.permutation(weight) * phase).sum()) / len(index))
         for _ in range(n_null)], 95))
    return amplitude, null


def report_register(detections, curve: DrawnCurve, geometry, arc: np.ndarray,
                    k: int) -> None:
    """The monomer register, per segment, with the dimer-rate control."""
    header("5. MONOMER REGISTER / SEAM (period-2 alternation)")
    patch = geometry.axis_offset_magnitude_angstrom > 20
    print(f"  template is a {'PATCH' if patch else 'RING'} "
          f"(axis offset {geometry.axis_offset_magnitude_angstrom:.1f} Å)")
    if not patch:
        print("  a ring averages the whole circumference and sits below its own "
              "chance level\n  on real data -- run this on the patch, the ring row "
              "is here only as the\n  negative control")

    rng = np.random.default_rng(0)
    edges = np.linspace(0.0, curve.length_px, k + 1)
    print(f"\n  {'segment':<12} {'sites':>6} {'rise Å':>9} {'amp':>7} {'chance':>8} "
          f"{'ratio':>7} {'control':>8}")
    for part in range(k):
        keep = (arc >= edges[part]) & (arc <= edges[part + 1])
        subset = detections[keep].reset_index(drop=True)
        if len(subset) < 200:
            continue
        row = [f"{part + 1}/{k}"]
        values = []
        for rise in (None, 2.0 * geometry.rise_angstrom):
            try:
                sites = extract_lattice_sites(
                    subset, geometry, PIXEL_SIZE, bootstrap_score_threshold=8.0,
                    initial_rise_angstrom=rise,
                )
            except ValueError:
                values.append(None)
                continue
            index = np.asarray(sites.axial_index).astype(int)
            score = np.asarray(sites.score)
            if len(index) < 8:
                values.append(None)
                continue
            order = np.argsort(index)
            amplitude, null = alternation(index[order], score[order], rng)
            values.append((len(index), float(sites.rise_angstrom), amplitude, null))
        if values[0] is None:
            continue
        n_sites, rise, amplitude, null = values[0]
        control = f"{values[1][2] / values[1][3]:.2f}x" if values[1] else "--"
        print(f"  {row[0]:<12} {n_sites:6d} {rise:9.3f} {amplitude:7.3f} "
              f"{null:8.3f} {amplitude / null:6.2f}x {control:>8}")
    print("\n  The control indexes at the DIMER rate, which puts every site in the "
          "same\n  register, so any alternation there is an artefact of the method "
          "rather than\n  a property of the lattice. It has to come out near 1.")


def main() -> None:
    """Every readout, for every drawn region."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="gdp_curved_13pf")
    parser.add_argument("--constraint", type=pathlib.Path,
                        default=MT_ROOT / "configs"
                        / "filament_constraint_gdp_curved.h5")
    parser.add_argument("--results", default="results_gdp_curved")
    parser.add_argument("--geometry", default=None,
                        help="override the template PDB used for calibration")
    parser.add_argument("--region", type=int, default=None,
                        help="analyse only this region id")
    parser.add_argument("--min-z", type=float, default=7.0)
    parser.add_argument("--segments", type=int, default=None,
                        help="override the segment count chosen from the curvature")
    args = parser.parse_args()

    stem = args.geometry
    if stem is None:
        for short, template_stem, _ in SEARCHES:
            if args.tag.endswith(short):
                stem = f"{template_stem}.pdb"
                break
    if stem is None:
        raise SystemExit(f"cannot infer a template for tag {args.tag}; pass --geometry")
    geometry = TemplateLatticeGeometry.from_pdb(str(MT_ROOT / "models" / stem))

    table = MT_ROOT / args.results / f"output_correlation_table_{args.tag}.h5"
    if not table.is_file():
        raise SystemExit(f"no correlation table at {table} -- run the search first")
    detections = detections_from_hdf5(
        str(table), min_z_score=args.min_z
    ).reset_index(drop=True)

    with h5py.File(args.constraint, "r") as handle:
        region_map = handle["maps/region_id"][:]
        drawn = {int(name): handle[f"regions/{name}/path"][:]
                 for name in handle["regions"]}

    header("RUN")
    print(f"  tag        {args.tag}")
    print(f"  template   {stem}")
    print(f"  N {geometry.n_protofilaments}, repeat {geometry.rise_angstrom:.3f} Å, "
          f"axis offset {geometry.axis_offset_magnitude_angstrom:.1f} Å "
          f"({'PATCH' if geometry.axis_offset_magnitude_angstrom > 20 else 'RING'})")
    print(f"  {len(detections):,} detections above z {args.min_z}, "
          f"max z {detections['z_score'].max():.2f}")
    print(f"  {len(drawn)} drawn region(s): {sorted(drawn)}")

    # Region ownership is decided at the detection's own pixel, which is where the
    # constraint gated it, so a detection can never be analysed against a curve that
    # did not permit it.
    y = np.clip(detections["y"].to_numpy(), 0, region_map.shape[0] - 1)
    x = np.clip(detections["x"].to_numpy(), 0, region_map.shape[1] - 1)
    owner = region_map[y, x]

    for index in sorted(drawn):
        if args.region is not None and index != args.region:
            continue
        subset = detections[owner == index].reset_index(drop=True)
        if len(subset) < 100:
            print(f"\nREGION {index}: only {len(subset)} detections, skipped")
            continue
        curve = DrawnCurve(drawn[index])
        k = args.segments or report_geometry(index, curve, len(subset))

        points = axis_points_from_peaks(subset, geometry, PIXEL_SIZE)
        arc, across = curve.project(points)

        report_polarity(subset, curve, arc, k)
        report_protofilaments(subset)
        report_axis(subset, curve, geometry, arc, across)
        report_spacing(subset, curve, geometry, arc, k)
        report_register(subset, curve, geometry, arc, k)


if __name__ == "__main__":
    main()
