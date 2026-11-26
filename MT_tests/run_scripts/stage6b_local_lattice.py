"""Stage 6b: compacted vs expanded lattice PER SEGMENT along each filament.

Stage 6 answers "compacted or expanded" once per tube. The lattice can change along a
tube, so this reads it in contiguous segments of sites -- as short as still decides.

Two independent readouts per segment, both signed so that positive = compacted:

* **local rise** (the headline) -- the slope of each refined site's deviation from the
  filament's continuous lattice against its index, added to the spacing the deviations
  were measured against, divided by the segment's own sin(theta). Its target is known
  before looking: the two lattices' repeats (the maps', 40.943 / 41.833 A) sit
  HALF_GAP = 0.445 A either side of the midpoint, so how many sites a segment needs
  follows from the tube's own site scatter.
* **competition** -- dz = z(compacted) - z(expanded) at each site, averaged over the
  segment, both templates scored at a NEUTRAL pose (``neutral_stack``): Stage 6's scores
  from the found poses hand the template that found the sites +0.17 / -0.12 z, the size
  of the signal itself.

**Neighbouring sites are not independent** -- a 2-ring template spans four monomers,
and neighbouring refined positions wander together (on 823, lag-1 correlation of the
deviations 0.25-0.57). Both errors are widened by the variance inflation of their own
residuals, estimated per filament after removing a slow trend so a real lattice change
is not mistaken for correlated noise. The rise error is estimated from n - 2 residuals,
so its t is converted to a normal-equivalent z through Student's t before any call.

**How short.** Per filament, the smallest segment whose EXPECTED rise z clears the call
threshold (|z| > 3) with 90% probability -- from that filament's residual scatter,
inflation and sin(theta). ``--size`` fixes one length instead. With ``--truth`` (a
synthetic with known domains, ``simulate_microtubule_bundle.py --domains lattice``)
every length in ``--sizes`` is also scored against the truth, with the rise pulls, so
the error model can be checked rather than trusted.

Usage:
    python run_scripts/stage6b_local_lattice.py --table <ring table>
    python run_scripts/stage6b_local_lattice.py --table <ring table> --truth <truth.json>
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

from leopard_em.analysis.filament_lattice import TemplateLatticeGeometry

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from continuous_unwrap_curved import continuous_sites  # noqa: E402
from frame_geometry import frame_geometry, legacy_frame_geometry, map_for  # noqa: E402
from compete_sites_curved import (  # noqa: E402
    LATTICE_PAIRS,
    PATCH_PAIRS,
    RIVAL_SETS,
    write_config,
)
from stage4_sites import MAX_DEVIATION_ANGSTROM  # noqa: E402
from stage5_refine import refined_frame  # noqa: E402

# Which templates compete is set by the search's reference (compete_sites_curved's rival
# sets): ring keys LATTICE_PAIRS[rivals] (compacted, expanded), patch keys PATCH_PAIRS.


def reference_stems(rivals: str) -> tuple[str, str]:
    """(ring, patch) template stems of a rival set: its reference and its 'patch'."""
    entries = RIVAL_SETS[rivals]
    ring = next(stem for _, stem, purpose in entries if "(reference)" in purpose)
    patch = next(stem for key, stem, _ in entries if key == "patch")
    return ring, patch
# The rendered maps' own repeats -- what the templates, and the synthetics built from
# them, actually carry (PDB: 40.835 / 41.979, same midpoint to 0.02 A).
C_REPEAT, E_REPEAT = 40.943, 41.833
RISE_MIDPOINT = (C_REPEAT + E_REPEAT) / 2
HALF_GAP = (E_REPEAT - C_REPEAT) / 2
MAX_LAG = 6          # autocorrelation is ~0 by lag 6 (2-ring template, 4 monomers)
TREND_SITES = 24     # slow trend removed before estimating noise and its correlation
DECIDE_Z = 3.0       # normal-equivalent |z| a segment needs for a call
POWER = 0.90         # a segment length must call a pure segment right this often
MIN_SIZE = 6
RIGHT_AT_LEAST, WRONG_AT_MOST = 0.90, 0.02
NEIGHBOURS = 4       # each side, for a site's neutral pose (its own pose excluded)
ROLL_GRID_DEG = np.arange(-12.0, 12.01, 1.0)   # --refit-roll: offsets scored per site
ROLL_BLOCK = 8       # --refit-roll bootstrap: lattice indices resampled together
PURE = 0.75          # a segment is scored against the truth only if this much is one lattice


def detrended(values: np.ndarray) -> np.ndarray:
    """``values`` (in order along the tube) minus a TREND_SITES running mean."""
    n = len(values)
    kernel = np.ones(TREND_SITES) / TREND_SITES
    trend = np.convolve(np.pad(values, TREND_SITES // 2, mode="edge"), kernel, "same")
    return values - trend[TREND_SITES // 2: TREND_SITES // 2 + n]


def variance_inflation(values: np.ndarray) -> float:
    """1 + 2 sum rho_k over lags 1..MAX_LAG, from ``values`` with the slow trend removed."""
    n = len(values)
    if n < 3 * MAX_LAG:
        return 1.0 + 2.0 * MAX_LAG * 0.3   # too short to estimate: a conservative guess
    residual = detrended(values)
    rho = [np.corrcoef(residual[:-k], residual[k:])[0, 1] for k in range(1, MAX_LAG + 1)]
    return float(1.0 + 2.0 * sum(max(r, 0.0) * (1 - k / n)
                                 for k, r in enumerate(rho, start=1)))


def normal_z(t: float, dof: float) -> float:
    """The normal z with the same tail probability as Student's t on ``dof``."""
    if not np.isfinite(t) or dof <= 0:
        return float("nan")
    return float(np.sign(t) * stats.norm.isf(stats.t.sf(abs(t), dof)))


def call(z):
    """Compacted, expanded, or undecided."""
    if not np.isfinite(z):
        return "?"
    return "C" if z > DECIDE_Z else ("E" if z < -DECIDE_Z else "?")


def local_rise(block, geometry, pixel_size, resolve=False):
    """Per-site lattice bookkeeping on the filament's own continuous lattice.

    Returns (axial index, projected deviation, sin(theta), z) per row of ``block``
    (NaN where the fit dropped a site) and the spacing the deviations are measured
    against -- or None if no continuous lattice fits. ``resolve`` keeps one site per
    (axial index, protofilament): the patch's sites.
    """
    frame = refined_frame(block).reset_index(drop=True)
    floor = float(frame["z_score"].min()) - 0.01
    # at least the 8.0 cut, even when every site is stronger: on a strong tube (GMPCPP,
    # all refined z > 8.4) arange(8.0, floor) was empty and the lattice 'did not fit'
    for cut in np.arange(8.0, min(floor, 8.0 - 1e-6), -0.25):
        try:
            sites = continuous_sites(frame, geometry, float(cut), pixel_size=pixel_size,
                                     resolve_protofilaments=resolve)
            break
        except ValueError:
            continue
    else:
        return None
    chosen = sites["detection_index"]
    per_site = np.full((4, len(frame)), np.nan)
    per_site[0, chosen] = sites["axial_index"]
    per_site[1, chosen] = sites["deviation"]
    per_site[2, chosen] = np.sin(np.radians(frame["theta"].to_numpy()[chosen]))
    per_site[3, chosen] = frame["z_score"].to_numpy()[chosen]
    return per_site, float(sites["indexing_rise"])


def per_index(index, values, weights=None):
    """(unique lattice indices, weighted mean of ``values`` at each), NaNs dropped.

    A ring gives one site per index; a protofilament-resolved patch gives up to N, and
    those share an axial position, so both readouts are averaged per index before any
    slope, noise or correlation is estimated along the tube.
    """
    index = np.asarray(index, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    weights = np.ones_like(values) if weights is None else np.asarray(weights, float)
    ok = ~(np.isnan(index) | np.isnan(values))
    unique, inverse = np.unique(index[ok], return_inverse=True)
    total = np.bincount(inverse, weights=weights[ok])
    return unique, np.bincount(inverse, weights=weights[ok] * values[ok]) / total


class RiseModel:
    """One filament's local-rise bookkeeping: per-site data and its noise model.

    Everything is measured per LATTICE INDEX (``per_index``): sigma is the scatter of
    the per-index mean deviation, and n in ``expected_error`` counts indices.
    """

    def __init__(self, per_site, indexing_rise):
        self.per_site = per_site
        self.indexing_rise = indexing_rise
        index, deviation, sines, weights = per_site
        ok = ~np.isnan(index)
        _, ordered = per_index(index, deviation, weights)
        self.inflation = variance_inflation(ordered)
        self.sigma = float(detrended(ordered).std(ddof=1))
        self.sine = float(np.sum(weights[ok] * sines[ok]) / np.sum(weights[ok]))
        self.sites_per_index = float(ok.sum() / len(ordered))
        # Stage 4's rule: sites landing this far inside the +/- rise/2 window are being
        # assigned to the wrong subunit, and the rise is not read (a whole-tube fit of
        # 823 filament 4 at 8.1 A rms read 40.19; its patch reads 40.83).
        self.deviation_rms = float(np.sqrt(np.nanmean(deviation[ok] ** 2)))
        self.reliable = self.deviation_rms <= MAX_DEVIATION_ANGSTROM

    def segment(self, part, per_site=None):
        """(rise, error, dof) of the sites ``part``, sin(theta)-corrected; or None.

        ``part`` indexes ``per_site`` (default this model's own, over its found sites).

        The slope of the projected deviation against index, plus the spacing the
        deviations were measured against, is the segment's PROJECTED spacing; dividing
        by its own score-weighted mean sin(theta) -- as ``estimate_lattice_rise`` does
        for a whole filament -- makes it the 3-D rise.
        """
        per_site = self.per_site if per_site is None else per_site
        index, deviation, sines, weights = per_site[:, part]
        ok = ~np.isnan(index)
        x, y = per_index(index, deviation, weights)
        if len(x) < 4:
            return None
        spread = float(np.sum((x - x.mean()) ** 2))
        if spread <= 0:
            return None
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (slope * x + intercept)
        dof = len(x) - 2
        slope_se = float(np.sqrt(np.sum(residual ** 2) / max(dof, 1) / spread))
        foreshortening = float(np.sum(weights[ok] * sines[ok]) / np.sum(weights[ok]))
        return ((self.indexing_rise + float(slope)) / foreshortening,
                slope_se * np.sqrt(self.inflation) / foreshortening, dof)

    def expected_error(self, n):
        """The rise error a contiguous run of ``n`` lattice indices should have."""
        return (self.sigma * np.sqrt(12.0 / (n * (n * n - 1.0)))
                * np.sqrt(self.inflation) / self.sine)

    def resolving_size(self, n_max):
        """Smallest n whose expected |z| clears DECIDE_Z with probability POWER."""
        for n in range(MIN_SIZE, n_max + 1):
            needed = (stats.t.isf(stats.norm.sf(DECIDE_Z), n - 2)
                      + stats.norm.isf(1 - POWER))
            if HALF_GAP / self.expected_error(n) >= needed:
                return n
        return None


def segments(order_index, size, step=None):
    """Sites in contiguous runs of about ``size`` occupied lattice indices along the tube.

    With ``step`` (calibration), windows of exactly ``size`` indices every ``step``;
    otherwise the whole tube split into floor(n / size) runs, the remainder shared out.
    A ring has one site per index, so this is runs of sites; a patch has several.
    """
    order_index = np.asarray(order_index)
    unique = np.unique(order_index)
    if step is not None:
        groups = [unique[i:i + size] for i in range(0, len(unique) - size + 1, step)]
    else:
        count = len(unique) // size
        groups = list(np.array_split(unique, count)) if count else []
    return [np.flatnonzero(np.isin(order_index, group)) for group in groups]


def circular_mean(values, period):
    """Mean of ``values`` on a circle of ``period`` (degrees), in [0, period)."""
    angle = 2 * np.pi * np.asarray(values, dtype=np.float64) / period
    return float(np.angle(np.exp(1j * angle).mean()) / (2 * np.pi) * period) % period


def wrap(values, period):
    """Into [-period/2, period/2)."""
    return (np.asarray(values, dtype=np.float64) + period / 2) % period - period / 2


def neutral_stack(stack, refined, geometry, pixel_size, half_width, all_sites=False,
                  resolve=False, predicts=None):
    """Each site's pose as its LATTICE and its NEIGHBOURS say, not as it was found.

    Stage 6 scores both templates from the refined pose of the detection that won the
    site -- chosen by one template at its own noise peak -- and lets each template
    climb from there. On the known-domain synthetic that hands the template that found
    the sites +0.17 z (compacted-found) or -0.12 z (expanded-found), as large as the
    lattice signal. Any search at scoring time can climb back to that peak: a start
    from the lattice with a roll search still left +0.06 / -0.10. So here nothing is
    searched, and no site sets its own pose:

    * roll: the neighbours' roll PHASE (phi modulo one protofilament step, which is
      smooth along a tube), on the protofilament labelling of the nearest neighbour;
    * position: where the filament's continuous lattice (fitted to the refined sites)
      puts that site at that roll, plus the median offset of the neighbours from their
      own predicted positions -- which follows a real local lattice change or bend,
      not this site's noise;
    * theta, defocus: the neighbours' median; psi: the neighbours' AXIS (their psi
      modulo 180), on the tube's polarity -- the side most of its found sites take.
      Neighbours found in the other polarity give nothing (their roll is a different
      roll); a site found that way is posed like any other, by its neighbours. (psi
      once came from a median of differences wrapped at 360: where found sites mix
      polarity it landed on either, or at +/-90 -- the template across the tube.)

    Neighbours are the detected sites within NEIGHBOURS lattice indices, the site
    itself excluded -- and, with ``resolve`` (a patch, whose roll picks out ONE
    protofilament), on the same protofilament: the site is that subunit, so its pose
    comes from that protofilament's neighbours. With ``all_sites`` every
    lattice index in the filament's span is scored, found or not (with ``resolve``, every
    protofilament at every index), so which sites are scored no longer depends on either
    template's noise at all.

    Returns the stack with ``filament``, ``axial_index`` and ``detected``, plus what
    ``refit_roll`` needs to move a site along its lattice: ``roll_abs`` (the roll the
    position was predicted at -- the lattice prediction is linear in roll, so modulo 360
    is not enough) and ``lattice_offset_x/y`` (the neighbours' median offset). Detected
    sites keep their original index labels (so ``refined.loc`` finds them); the others
    are labelled from 1,000,000. ``predicts``, if given, collects each filament's
    lattice predictor.
    """
    period = float(geometry.protofilament_angular_spacing_deg)
    parts, next_label = [], 1_000_000
    for label in sorted(stack["filament"].unique()):
        mask = (stack["filament"] == label).to_numpy()
        frame = refined_frame(refined[mask]).reset_index(drop=True)
        floor = float(frame["z_score"].min()) - 0.01
        fitted = None
        # at least the 8.0 cut, even when every site is stronger: on a strong tube (GMPCPP,
        # all refined z > 8.4) arange(8.0, floor) was empty and the lattice 'did not fit'
        for cut in np.arange(8.0, min(floor, 8.0 - 1e-6), -0.25):
            try:
                fitted = continuous_sites(frame, geometry, float(cut),
                                          pixel_size=pixel_size,
                                          resolve_protofilaments=resolve)
                break
            except ValueError:
                continue
        if fitted is None:
            print(f"  filament {label}: no continuous lattice, left out")
            continue
        order = np.argsort(fitted["axial_index"], kind="stable")
        rows = stack[mask].iloc[fitted["detection_index"][order]]
        own = refined[mask].iloc[fitted["detection_index"][order]]
        k_found = fitted["axial_index"][order]
        pf_found = (fitted["protofilament"][order] if resolve
                    else np.zeros(len(k_found), dtype=int))
        phi_found = own["refined_phi"].to_numpy(dtype=float)
        xy_found = own[["refined_pos_x", "refined_pos_y"]].to_numpy(dtype=float)
        offset = xy_found - fitted["predicted_xy"][order]
        theta = own["refined_theta"].to_numpy(dtype=float)
        psi = own["refined_psi"].to_numpy(dtype=float)
        defocus = own["refined_relative_defocus"].to_numpy(dtype=float)
        axis = circular_mean(psi, 180.0)
        forward = np.abs(wrap(psi - axis, 360.0)) < 90.0
        polarity = axis if forward.sum() >= (~forward).sum() else axis + 180.0
        same_polarity = np.abs(wrap(psi - polarity, 360.0)) < 90.0
        if predicts is not None:
            predicts[label] = fitted["predict"]

        if all_sites:
            # every lattice index in the span -- and with ``resolve`` every protofilament
            # at each (lattice_fill.py, and the 6b CLI's --patch --start lattice)
            pfs = range(geometry.n_protofilaments) if resolve else (0,)
            targets = [(k, pf) for k in range(k_found.min(), k_found.max() + 1)
                       for pf in pfs]
        else:
            targets = list(zip(k_found, pf_found))
        out, own_rows = [], []
        for k, pf in targets:
            same = (pf_found == pf) & same_polarity
            distance = np.where(same, np.abs(k_found - k), np.iinfo(np.int64).max)
            near = same & (distance > 0) & (distance <= NEIGHBOURS)
            if near.sum() < 2:   # sparse stretch: the nearest few, still not itself
                candidates = np.flatnonzero(same & (distance > 0))
                if candidates.size == 0:
                    continue
                near = np.zeros(len(k_found), dtype=bool)
                near[candidates[np.argsort(distance[candidates])[:2 * NEIGHBOURS]]] = True
            nearest = np.flatnonzero(near)[np.argmin(distance[near])]
            phase = circular_mean(phi_found[near], period)
            phi = phi_found[nearest] + wrap(phase - phi_found[nearest], period)
            lattice_offset = np.median(offset[near], axis=0)
            xy = fitted["predict"](np.array([k]), np.array([phi]))[0] + lattice_offset
            # same protofilament, same index -- whichever polarity it was found in
            itself = np.flatnonzero((pf_found == pf) & (k_found == k))
            detected = itself.size > 0
            row = rows.iloc[itself[0] if detected else nearest].copy()
            if detected:
                own_rows.append(itself[0])
            else:
                row.name = next_label
                next_label += 1
            x, y = (int(v) for v in np.rint(xy))
            row["pos_x"], row["pos_y"] = x, y
            row["pos_x_img"], row["pos_y_img"] = x + half_width, y + half_width
            row["pos_x_img_angstrom"] = (x + half_width) * pixel_size
            row["pos_y_img_angstrom"] = (y + half_width) * pixel_size
            row["phi"] = phi % 360.0
            row["roll_abs"] = phi
            row["lattice_offset_x"], row["lattice_offset_y"] = lattice_offset
            row["theta"] = float(np.median(theta[near]))
            local_axis = circular_mean(psi[near], 180.0)
            flip = 0.0 if abs(wrap(local_axis - polarity, 360.0)) < 90.0 else 180.0
            row["psi"] = (local_axis + flip) % 360.0
            row["relative_defocus"] = float(np.median(defocus[near]))
            row["axial_index"] = int(k)
            row["protofilament"] = int(pf) if resolve else -1
            row["filament"] = label
            row["region"], row["segment"] = label, 1   # what compete reports by
            row["detected"] = detected
            out.append(row)
        base = pd.DataFrame(out)
        found = base["detected"].to_numpy()
        moved = np.hypot(*(base.loc[found, ["pos_x", "pos_y"]].to_numpy(dtype=float)
                           - xy_found[own_rows]).T) * pixel_size
        print(f"  filament {label}: {len(base)} sites ({found.sum()} found); neutral "
              f"position {np.median(moved):.1f} A from the found one (median; 90th pct "
              f"{np.percentile(moved, 90):.1f} A); {(~same_polarity).sum()} of "
              f"{len(psi)} found sites in the other polarity, not used")
        parts.append(base)
    return pd.concat(parts)


def roll_curve(index, base_phase, scores, degree, start=None):
    """Polynomial roll phase along a tube that maximises the summed reference z.

    ``scores[i]`` is site i's z on ``ROLL_GRID_DEG`` about its own ``base_phase[i]``;
    each is a cubic spline, penalised linearly past the grid's ends. ``u`` runs -1..1
    over the tube's lattice indices, so coefficients are comparable between resamples.
    """
    # pylint: disable=import-outside-toplevel
    from scipy.interpolate import CubicSpline
    from scipy.optimize import minimize

    splines = [CubicSpline(ROLL_GRID_DEG, row) for row in scores]
    low, high = ROLL_GRID_DEG[0], ROLL_GRID_DEG[-1]

    def total(coeffs):
        offset = np.polyval(coeffs, index) - base_phase
        clipped = np.clip(offset, low, high)
        value = sum(float(f(o)) for f, o in zip(splines, clipped))
        return -(value - 2.0 * np.abs(offset - clipped).sum())

    first = np.polyfit(index, base_phase, degree) if start is None else np.array(start)
    best = None
    for shift in ((-4.0, -2.0, 0.0, 2.0, 4.0) if start is None else (0.0,)):
        trial = first.copy()
        trial[-1] += shift
        result = minimize(total, trial, method="Nelder-Mead",
                          options={"xatol": 0.01, "fatol": 1e-4, "maxiter": 4000})
        if best is None or result.fun < best.fun:
            best = result
    return best.x


def refit_roll(stack, predicts, period, half_width, pixel_size, tag, results, template,
               gpus, rng):
    """One smooth roll curve per tube, fitted by the reference ring, replacing the
    neutral rolls (each a mean of a few found rolls, with errors of degrees).

    Roll is where a decorated template is fragile -- a motor sits ~50% further out than
    the wall, and on the kinesin synthetic a roll error that keeps 44% of the bare
    ring's z (5 deg) removes all the motor contrast -- while a tube's roll changes
    smoothly (the supertwist). So every site is scored with the reference ring at
    ``ROLL_GRID_DEG`` about its neutral roll, the position following the roll along the
    site's lattice (the prediction is linear in roll), and one polynomial in lattice
    index (straight for spans under 50 indices, quadratic beyond) is fitted per tube to
    the summed z, with a block-bootstrap error. Three parameters per tube from tens of
    sites: no site sets its own roll, so no finder bias returns.
    """
    # pylint: disable=import-outside-toplevel
    from scipy.interpolate import CubicSpline

    from leopard_em.pydantic_models.managers import RefineTemplateManager

    def place(row, roll):
        xy = (predicts[row["filament"]](np.array([row["axial_index"]]), np.array([roll]))[0]
              + np.array([row["lattice_offset_x"], row["lattice_offset_y"]]))
        x, y = (int(v) for v in np.rint(xy))
        return {"pos_x": x, "pos_y": y, "pos_x_img": x + half_width,
                "pos_y_img": y + half_width,
                "pos_x_img_angstrom": (x + half_width) * pixel_size,
                "pos_y_img_angstrom": (y + half_width) * pixel_size,
                "phi": roll % 360.0, "roll_abs": roll}

    rows = []
    for _, row in stack.iterrows():
        for offset in ROLL_GRID_DEG:
            moved = row.copy()
            for key, value in place(row, float(row["roll_abs"]) + offset).items():
                moved[key] = value
            rows.append(moved)
    grid = pd.DataFrame(rows).reset_index(drop=True)
    grid_path = results / f"stack_sites_{tag}_rollgrid.csv"
    grid.to_csv(grid_path, index=True)
    config = MT_ROOT / "configs" / f"rollgrid_{tag}.yaml"
    write_config(grid_path, config, str(template.relative_to(MT_ROOT)),
                 MT_ROOT / "configs" / f"match_tm_{tag}.yaml", gpus, neutral_pose=True)
    out = results / f"rollgrid_refined_{tag}.csv"
    print(f"  roll re-fit: {len(stack)} sites x {len(ROLL_GRID_DEG)} rolls "
          f"({ROLL_GRID_DEG[0]:+.0f}..{ROLL_GRID_DEG[-1]:+.0f} deg) with {template.name}")
    RefineTemplateManager.from_yaml(str(config)).run_refine_template(str(out), 80)
    z = (pd.read_csv(out, index_col=0)["refined_scaled_mip"].to_numpy()
         .reshape(len(stack), len(ROLL_GRID_DEG)))

    stack = stack.copy()
    zero = int(np.argmin(np.abs(ROLL_GRID_DEG)))
    print(f"  {'filament':<9}{'sites':>6}{'degree':>7}{'moved rms':>10}{'max':>7}"
          f"{'curve sd':>9}{'z before':>9}{'z after':>9}")
    for label in sorted(stack["filament"].unique()):
        sites = np.flatnonzero((stack["filament"] == label).to_numpy())
        order = sites[np.argsort(stack["axial_index"].to_numpy()[sites], kind="stable")]
        index = stack["axial_index"].to_numpy(dtype=float)[order]
        roll = stack["roll_abs"].to_numpy(dtype=float)[order]
        phase = np.unwrap(roll * 2 * np.pi / period) * period / (2 * np.pi)
        degree = 2 if np.ptp(index) >= 50 else 1
        coeffs = roll_curve(index, phase, z[order], degree)
        fitted = np.polyval(coeffs, index)
        # block bootstrap: resample runs of ROLL_BLOCK lattice indices, refit
        blocks = {}
        for i, k in enumerate(index):
            blocks.setdefault(int(k // ROLL_BLOCK), []).append(i)
        members = list(blocks.values())
        curves = []
        for _ in range(100):
            pick = np.concatenate([members[j] for j in rng.integers(0, len(members),
                                                                    len(members))])
            boot = roll_curve(index[pick], phase[pick], z[order][pick], degree, coeffs)
            curves.append(np.polyval(boot, index))
        spread = float(np.median(np.std(curves, axis=0)))
        moved = fitted - phase
        for site, value in zip(order, roll + moved):
            for key, item in place(stack.iloc[site], value).items():
                stack.iat[site, stack.columns.get_loc(key)] = item
        clipped = np.clip(moved, ROLL_GRID_DEG[0], ROLL_GRID_DEG[-1])
        after = np.mean([float(CubicSpline(ROLL_GRID_DEG, z[s])(m))
                         for s, m in zip(order, clipped)])
        print(f"  {label:<9}{len(order):>6}{degree:>7}{np.sqrt(np.mean(moved ** 2)):>9.1f}°"
              f"{np.abs(moved).max():>6.1f}°{spread:>8.2f}°{z[order, zero].mean():>9.2f}"
              f"{after:>9.2f}")
    return stack


def run_neutral_competition(tag, results, stack, gpus, suffix, keys,
                            map_pixel_size=None, rivals="compacted13"):
    """Score compacted and expanded at the neutral poses; their per-site scores."""
    neutral = f"{tag}_{suffix}"
    stack.to_csv(results / f"stack_sites_{neutral}.csv", index=True)
    shutil.copy(MT_ROOT / "configs" / f"match_tm_{tag}.yaml",
                MT_ROOT / "configs" / f"match_tm_{neutral}.yaml")
    log = MT_ROOT / "logs" / f"stage6b_{neutral}.log"
    command = [sys.executable, str(MT_ROOT / "run_scripts" / "compete_sites_curved.py"),
               "--tag", neutral, "--results", str(results.relative_to(MT_ROOT)),
               "--rivals", rivals, "--only", *keys, "--neutral-pose",
               "--gpus", *[str(g) for g in gpus]]
    if map_pixel_size:
        command += ["--map-pixel-size", str(map_pixel_size)]
    print(f"  scoring both templates at the neutral poses (log: "
          f"{log.relative_to(MT_ROOT)})")
    with open(log, "w", encoding="utf-8") as handle:
        subprocess.run(command, check=True, cwd=MT_ROOT, stdout=handle, stderr=handle)
    return pd.read_csv(results / f"compete_summary_{neutral}.csv", index_col=0)


def analyse(frame, sizes, geometry, refined_models, step=None):
    """One row per segment for every filament: local rise, competition, combined.

    ``sizes`` is one length for every filament, or a dict of lengths per filament.
    """
    rows = []
    for label in sorted(frame["filament"].unique()):
        size = sizes.get(label) if isinstance(sizes, dict) else sizes
        if not size:
            continue
        block = frame[frame["filament"] == label]
        dz = block["dz"].to_numpy()
        axial = block["axial_index"].to_numpy()
        _, ordered = per_index(axial, dz)
        inflation = variance_inflation(ordered)
        noise = (float(detrended(ordered).std(ddof=1)) if len(ordered) > TREND_SITES
                 else float(ordered.std(ddof=1)))
        model = refined_models.get(label)
        per_site_rows = None
        if model is not None:
            # The model's per-site arrays are over the FOUND sites; align them onto this
            # block's rows (lattice-only sites have no refined position).
            per_site_rows = np.full((4, len(block)), np.nan)
            position = {name: i for i, name in enumerate(model.labels)}
            for j, name in enumerate(block.index):
                if name in position:
                    per_site_rows[:, j] = model.per_site[:, position[name]]
        for k, part in enumerate(segments(axial, size, step)):
            _, dz_index = per_index(axial[part], dz[part])
            mean = float(dz_index.mean())
            se = noise * np.sqrt(inflation / len(dz_index))
            z_comp = mean / se
            span = int(axial[part].max() - axial[part].min() + 1)
            row = {"filament": int(label), "segment": k + 1,
                   "first_index": int(axial[part].min()),
                   "last_index": int(axial[part].max()), "n": len(dz_index),
                   "n_sites": len(part),
                   "span_nm": span * geometry.rise_angstrom / 10.0,
                   "dz": mean, "se": se, "z_comp": z_comp, "comp_call": call(z_comp),
                   "comp_inflation": inflation}
            measured = (model.segment(part, per_site_rows)
                        if per_site_rows is not None and model.reliable else None)
            if measured is not None:
                rise, rise_se, dof = measured
                z_rise = normal_z((RISE_MIDPOINT - rise) / rise_se, dof)
                # Which repeats the rise is CONSISTENT with (|z| < DECIDE_Z): a
                # segment between the two is undecided because it is measured to lie
                # between them, not because it is noisy.
                fits = [name for name, repeat in (("C", C_REPEAT), ("E", E_REPEAT))
                        if abs(normal_z((rise - repeat) / rise_se, dof)) < DECIDE_Z]
                row.update(local_rise=rise, rise_se=rise_se, rise_dof=dof,
                           z_rise=z_rise, rise_call=call(z_rise),
                           rise_fits={2: "both", 1: fits[0] if fits else "",
                                      0: "neither"}[len(fits)],
                           rise_inflation=model.inflation,
                           z_combined=(z_rise + z_comp) / np.sqrt(2.0))
                row["combined_call"] = call(row["z_combined"])
            row["call"] = row.get("rise_call", row["comp_call"])
            row["agree"] = (row["comp_call"] == row["call"]
                            or "?" in (row["comp_call"], row["call"]))
            if "truth" in block.columns:
                truth = block["truth"].to_numpy()[part]
                values, counts = np.unique(truth, return_counts=True)
                row["truth"] = str(values[np.argmax(counts)])
                row["purity"] = float(counts.max() / len(part))
                row["true_rise"] = float(np.mean(np.where(truth == "C", C_REPEAT,
                                                          E_REPEAT)))
            rows.append(row)
    return pd.DataFrame(rows)


def rise_models(stack, refined, geometry, pixel_size, resolve=False):
    """A RiseModel per filament, from its refined (found) sites."""
    models = {}
    for label in sorted(stack["filament"].unique()):
        names = stack.index[(stack["filament"] == label).to_numpy()]
        names = names[names.isin(refined.index)]
        if len(names) < 20:
            continue
        found = local_rise(refined.loc[names], geometry, pixel_size, resolve)
        if found is None:
            continue
        model = RiseModel(*found)
        model.labels = list(names)
        models[label] = model
    return models


def truth_labels(stack, truth_path):
    """Known lattice at each site: that of the nearest simulated block."""
    from scipy.spatial import cKDTree  # pylint: disable=import-outside-toplevel

    truth = json.loads(pathlib.Path(truth_path).read_text())
    xy, lattice = [], []
    for tube in truth["tubes"]:
        for block in tube["blocks"]:
            xy.append((block["x"], block["y"]))
            lattice.append(block["lattice"])
    _, nearest = cKDTree(np.asarray(xy)).query(
        stack[["pos_x_img", "pos_y_img"]].to_numpy(dtype=float))
    return np.asarray(lattice)[nearest]


READOUTS = (("rise", "rise_call"), ("competition", "comp_call"),
            ("combined", "combined_call"))


def calibrate(frame, sizes, geometry, models, adaptive):
    """Score each segment length against the truth, for every readout.

    Windows slide by half a length, so a short synthetic yields enough segments to
    count; neighbouring windows share sites, so treat the rates as indicative. The
    rise pull (measured - true) / error should have unit spread if the error is honest.
    """
    print(f"\n{'':>19}" + "".join(f"{'-- ' + key + ' --':>21}" for key, _ in READOUTS)
          + f"{'rise pull':>13}")
    print(f"{'sites':>6} {'nm':>6} {'segs':>5}" + "  right wrong undec" * len(READOUTS)
          + f"{'sd':>7}{'mean':>7}   (windows >= {PURE:.0%} one lattice)")
    rows = [(size, analyse(frame, size, geometry, models, step=max(size // 2, 1)))
            for size in sizes]
    rows.append(("adaptive", analyse(frame, adaptive, geometry, models)))
    for size, table in rows:
        if table.empty or "purity" not in table:
            continue
        pure = table[table["purity"] >= PURE]
        if pure.empty:
            continue
        cells = []
        for _, column in READOUTS:
            if column not in pure.columns:
                cells.append(f"{'--':>7}{'--':>6}{'--':>6}")
                continue
            calls = pure[column].fillna("?")
            right = float((calls == pure["truth"]).mean())
            wrong = float(((calls != "?") & (calls != pure["truth"])).mean())
            cells.append(f"{right:>7.0%}{wrong:>6.0%}{(calls == '?').mean():>6.0%}")
        pull = ((pure["local_rise"] - pure["true_rise"]) / pure["rise_se"]).dropna() \
            if "local_rise" in pure else pd.Series(dtype=float)
        nm = (f"{size * geometry.rise_angstrom / 10.0:>6.0f}" if size != "adaptive"
              else f"{pure['span_nm'].median():>6.0f}")
        label = f"{size:>6}" if size != "adaptive" else "adapt."
        print(f"{label} {nm} {len(pure):>5}" + "".join(cells)
              + (f"{pull.std(ddof=1):>7.2f}{pull.mean():>+7.2f}" if len(pull) > 2 else ""))


def plot(table, frame, out_path):
    """Per filament: local rise per segment (top), per-site and segment dz (bottom)."""
    import matplotlib  # pylint: disable=import-outside-toplevel
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    labels = sorted(table["filament"].unique())
    fig, axes = plt.subplots(2 * len(labels), 1, figsize=(10, 3.6 * len(labels)),
                             sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [1.2, 1] * len(labels)})
    colours = {"C": "#1f4e79", "E": "#c62828", "?": "#9e9e9e"}
    for i, label in enumerate(labels):
        top, bottom = axes[2 * i, 0], axes[2 * i + 1, 0]
        part = table[table["filament"] == label]
        for _, seg in part.iterrows():
            mid = (seg["first_index"] + seg["last_index"]) / 2
            half = (seg["last_index"] - seg["first_index"]) / 2
            if np.isfinite(seg.get("local_rise", np.nan)):
                top.errorbar(mid, seg["local_rise"], xerr=half, yerr=2 * seg["rise_se"],
                             fmt="o", color=colours[seg["rise_call"]], ms=4, lw=1.5,
                             capsize=2)
            bottom.hlines(seg["dz"], seg["first_index"], seg["last_index"],
                          color=colours[seg["comp_call"]], lw=2.5)
            bottom.fill_between([seg["first_index"], seg["last_index"]],
                                seg["dz"] - 2 * seg["se"], seg["dz"] + 2 * seg["se"],
                                color=colours[seg["comp_call"]], alpha=0.25, lw=0)
        for value, name in ((C_REPEAT, "compacted"), (E_REPEAT, "expanded")):
            top.axhline(value, color=colours[name[0].upper()], lw=0.8, ls="--")
        top.axhline(RISE_MIDPOINT, color="black", lw=0.5, ls=":")
        top.set_ylabel(f"fil {label}\nlocal rise (A)")
        block = frame[frame["filament"] == label].sort_values("axial_index")
        bottom.plot(block["axial_index"], block["dz"], ".", color="#bbbbbb", ms=3)
        if "truth" in block.columns:
            expanded = block["truth"].to_numpy() == "E"
            top.scatter(block["axial_index"][expanded],
                        np.full(expanded.sum(), E_REPEAT + 0.35), marker="|",
                        color="#c62828", s=25)
            top.scatter(block["axial_index"][~expanded],
                        np.full((~expanded).sum(), E_REPEAT + 0.35), marker="|",
                        color="#1f4e79", s=25)
        bottom.axhline(0, color="black", lw=0.8)
        bottom.set_ylabel("dz (C − E)")
        for ax in (top, bottom):
            ax.grid(alpha=0.25, ls=":")
    axes[-1, 0].set_xlabel("axial index along the tube (monomers)")
    fig.suptitle("lattice per segment -- top: local rise ±2 SE (dashed: compacted "
                 f"{C_REPEAT:.2f}, expanded {E_REPEAT:.2f} A); bottom: competition dz "
                 f"±2 SE\nblue compacted, red expanded, grey undecided "
                 f"(|z| < {DECIDE_Z:.0f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


# pylint: disable=too-many-locals,too-many-statements
def main() -> None:
    """Per-segment lattice state along every filament."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=pathlib.Path, required=True,
                        help="the RING table Stages 3-6 were run on, or with --patch "
                             "the PATCH table Stage 7 was run on")
    parser.add_argument("--patch", action="store_true",
                        help="read the protofilament-resolved patch sites (Stage 4 "
                             "--resolve-protofilaments, refined by Stage 7) and compete "
                             "the compacted and expanded PATCHES")
    parser.add_argument("--rivals", choices=sorted(RIVAL_SETS), default="compacted13",
                        help="the search's reference: compacted13 (GDP, in situ) or "
                             "expanded14 / expanded13 (GMPCPP) -- sets the templates competed")
    parser.add_argument("--stem", default=None,
                        help="template geometry (default: the 13-PF ring, or the "
                             "5-PF patch with --patch)")
    parser.add_argument("--truth", type=pathlib.Path, default=None,
                        help="truth.json of a known-domain synthetic: also calibrate")
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16, 20, 24, 32],
                        help="lengths scored against --truth")
    parser.add_argument("--size", type=int, default=None,
                        help="one segment length for every filament (default: the "
                             "smallest each filament resolves)")
    parser.add_argument("--in-vitro-legacy", action="store_true")
    parser.add_argument("--start", choices=("neutral", "lattice", "found"),
                        default="neutral",
                        help="competition scores: neutral pose at the found sites "
                             "(default); lattice: every lattice index at a neutral "
                             "pose; found: Stage 6's, from the found poses (biased "
                             "towards the template that found them)")
    parser.add_argument("--rescore", action="store_true",
                        help="re-run the neutral scoring even if its output exists")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--refit-roll", action="store_true",
                        help="ring, --start neutral: replace the neutral rolls by one "
                             "smooth roll curve per tube fitted by the reference ring "
                             "(see refit_roll); outputs get _rollfit")
    parser.add_argument("--pixel-size", type=float, default=None,
                        help="a FITTED pixel size (fit_pixel_size.py) instead of the "
                             "table's: the local rise is measured at it, and the "
                             "competition's templates are rendered at it (the CTF "
                             "stays at the search's)")
    args = parser.parse_args()

    table_path = args.table if args.table.is_absolute() else MT_ROOT / args.table
    tag = table_path.stem.removeprefix("output_correlation_table_")
    results = table_path.parent
    ring_stem, patch_stem = reference_stems(args.rivals)
    stem = args.stem or (patch_stem if args.patch else ring_stem)
    keys = PATCH_PAIRS[args.rivals] if args.patch else LATTICE_PAIRS[args.rivals][:2]
    frame_geo = (legacy_frame_geometry(table_path, stem) if args.in_vitro_legacy
                 else frame_geometry(table_path))
    geometry = TemplateLatticeGeometry.from_pdb(str(MT_ROOT / "models" / f"{stem}.pdb"))
    print(frame_geo.describe())
    pixel_size = args.pixel_size or frame_geo.pixel_size
    if args.pixel_size:
        print(f"pixel size FITTED: {pixel_size} A/px ({100 * (pixel_size / frame_geo.pixel_size - 1):+.2f}% "
              f"from the table's {frame_geo.pixel_size})")
    print(f"sites: {'5-PF patch, protofilament-resolved' if args.patch else 'ring'}"
          f" ({stem}); competition {keys[0]} vs {keys[1]}")

    if args.patch:
        # Stage 7 refined Stage 4's protofilament-resolved sites row for row.
        stack = pd.read_csv(results / f"stage7_stack_{tag}_in.csv", index_col=0)
        refined = pd.read_csv(results / f"stage7_refined_{tag}_in.csv", index_col=0)
        labels = pd.read_csv(results / f"stage4_sites_{tag}.csv")
        if len(labels) != len(stack) or not np.array_equal(
                labels["x"].to_numpy(), stack["pos_x"].to_numpy()):
            raise SystemExit("Stage 7's stack does not match Stage 4's sites row for row")
        for column in ("filament", "axial_index", "protofilament"):
            stack[column] = labels[column].to_numpy()
    else:
        stack = pd.read_csv(results / f"stack_sites_{tag}.csv", index_col=0)
        refined = pd.read_csv(results / f"refined_sites_{tag}.csv", index_col=0)
    models = rise_models(stack, refined, geometry, pixel_size, args.patch)
    if args.patch and args.start == "found":
        raise SystemExit("--patch supports --start neutral or lattice only")
    if args.start == "found":
        scores = pd.read_csv(results / f"compete_summary_{tag}.csv", index_col=0)
    else:
        suffix = (args.start + (f"_px{args.pixel_size}" if args.pixel_size else "")
                  + ("_rollfit" if args.refit_roll else ""))
        summary = results / f"compete_summary_{tag}_{suffix}.csv"
        stack_path = results / f"stack_sites_{tag}_{suffix}.csv"
        nominal_stack = results / f"stack_sites_{tag}_{args.start}.csv"
        if summary.is_file() and stack_path.is_file() and not args.rescore:
            print(f"reusing {summary.relative_to(MT_ROOT)} (--rescore to redo)")
            stack = pd.read_csv(stack_path, index_col=0)
            scores = pd.read_csv(summary, index_col=0)
        else:
            if args.pixel_size and nominal_stack.is_file():
                # The neutral poses are pixel positions and angles: the same at any
                # scale, so a fitted-scale run scores exactly the nominal run's poses.
                stack = pd.read_csv(nominal_stack, index_col=0)
            else:
                predicts = {}
                stack = neutral_stack(stack, refined, geometry, frame_geo.pixel_size,
                                      frame_geo.half_width,
                                      all_sites=args.start == "lattice",
                                      resolve=args.patch, predicts=predicts)
                if args.refit_roll:
                    if args.patch or args.start != "neutral":
                        raise SystemExit("--refit-roll is for the ring at --start neutral")
                    stack = refit_roll(stack, predicts,
                                       float(geometry.protofilament_angular_spacing_deg),
                                       frame_geo.half_width, frame_geo.pixel_size, tag,
                                       results, map_for(stem, frame_geo.pixel_size),
                                       args.gpus, np.random.default_rng(0))
            scores = run_neutral_competition(tag, results, stack, args.gpus, suffix,
                                             keys, args.pixel_size, args.rivals)
    frame = pd.DataFrame({
        "filament": stack["filament"].to_numpy(),
        "axial_index": stack["axial_index"].to_numpy(),
        "dz": (scores[keys[0]] - scores[keys[1]]).to_numpy(),
    }, index=stack.index)
    print(f"competition start: {args.start}; {len(frame)} sites over "
          f"{frame['filament'].nunique()} filament(s); dz = z({keys[0]}) - "
          f"z({keys[1]}), positive = compacted")

    print(f"\n{'filament':<9} {'sites':>6} {'/index':>7} {'dev sd':>7} {'infl':>5} "
          f"{'sin':>6} {'resolves':>9} {'nm':>5}   (rise: smallest n lattice indices "
          f"expected to call |z| > {DECIDE_Z:.0f} with {POWER:.0%} power)")
    sizes = {}
    for label in sorted(frame["filament"].unique()):
        count = int((frame["filament"] == label).sum())
        indices = int(frame.loc[frame["filament"] == label, "axial_index"].nunique())
        model = models.get(label)
        if args.size:
            sizes[label] = args.size
        elif model is not None:
            sizes[label] = model.resolving_size(indices)
        if model is None:
            print(f"{label:<9} {count:>6}  no local-rise model (too few refined sites)")
            continue
        if not model.reliable:
            # No rise readout: segment by what the competition would need is unknown,
            # so use the tube's own rise length as if it were reliable, and say so.
            length = sizes[label] or "no"
            print(f"{label:<9} {count:>6}  local rise UNRELIABLE (site deviation "
                  f"{model.deviation_rms:.1f} A rms > {MAX_DEVIATION_ANGSTROM:.0f} A): "
                  f"competition only, segments of {length} indices")
            continue
        n = sizes[label]
        print(f"{label:<9} {count:>6} {model.sites_per_index:>7.1f} {model.sigma:>6.2f}A "
              f"{model.inflation:>5.2f} {model.sine:>6.4f} {n if n else 'never':>9} "
              f"{(n * geometry.rise_angstrom / 10 if n else float('nan')):>5.0f}")

    if args.truth is not None:
        frame["truth"] = truth_labels(stack, args.truth)
        calibrate(frame, args.sizes, geometry, models, sizes)

    table = analyse(frame, sizes, geometry, models)
    if table.empty:
        raise SystemExit("no filament long enough to split")
    print("\nPER SEGMENT (headline call: local rise; competition alongside)")
    for label in sorted(table["filament"].unique()):
        part = table[table["filament"] == label]
        print(f"  filament {label}: {len(part)} segment(s) of {part['n'].min()}-"
              f"{part['n'].max()} lattice indices, {part['n_sites'].sum()} sites "
              f"(~{part['span_nm'].median():.0f} nm); resolves {sizes[label]}")
        print(f"    call         {''.join(part['call'])}")
        print(f"    competition  {''.join(part['comp_call'])}"
              + ("" if part["agree"].all() else "   <- DISAGREES somewhere"))
        if "local_rise" in part and part["local_rise"].notna().any():
            print("    rise A       " + " ".join(f"{r:.2f}" for r in part["local_rise"]))
            print("    rise z       " + " ".join(f"{z:+.1f}" for z in part["z_rise"]))
            print("    consistent   " + " ".join(part["rise_fits"].fillna("--"))
                  + f"   (repeats within |z| < {DECIDE_Z:.0f}: C {C_REPEAT}, "
                  f"E {E_REPEAT})")
        print("    dz           " + " ".join(f"{d:+.2f}" for d in part["dz"]))
        print("    comp z       " + " ".join(f"{z:+.1f}" for z in part["z_comp"]))
        if "truth" in part.columns:
            print("    true         " + "".join(
                t if p >= PURE else t.lower()
                for t, p in zip(part["truth"], part["purity"]))
                  + "   (lower case: mixed segment)")
    name = (f"{tag}_{args.start}" + (f"_px{args.pixel_size}" if args.pixel_size else "")
            + (f"_n{args.size}" if args.size else ""))
    out = results / f"stage6b_segments_{name}.csv"
    table.to_csv(out, index=False)
    # The same readouts in a window of each tube's length centred at every few lattice
    # indices: the resolution is still the window, but a feature no longer depends on
    # where tile boundaries fall (on 823 tube 1, non-overlapping tiles split an
    # intermediate 41.5 A stretch between two segments and averaged it away).
    steps = {label: max(1, size // 8) for label, size in sizes.items() if size}
    running = pd.concat([analyse(frame, {label: sizes[label]}, geometry, models,
                                 step=steps[label]) for label in steps], ignore_index=True)
    running["centre_index"] = (running["first_index"] + running["last_index"]) / 2
    running.to_csv(results / f"stage6b_profile_{name}.csv", index=False)
    plot(table, frame, results / f"stage6b_segments_{name}.png")
    print(f"\n  C compacted, E expanded, ? undecided (|z| < {DECIDE_Z:.0f})")
    print(f"wrote {out.relative_to(MT_ROOT)}, the .png, and the running profile "
          f"stage6b_profile_{name}.csv")


if __name__ == "__main__":
    main()
