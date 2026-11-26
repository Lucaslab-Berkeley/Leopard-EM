"""Is a microtubule decorated? Its 8 nm layer line against its 4 nm one, per tube.

A bare microtubule repeats every 4 nm along each protofilament (a tubulin monomer); the
8 nm (alpha/beta dimer) repeat is there only through the small alpha/beta difference, so
its 8 nm layer line is weak. A motor bound to every beta-tubulin -- kinesin in rigor --
adds a large 8 nm-periodic mass, and the 8 nm layer line becomes strong. That needs no
kinesin model: only the tube's path, the micrograph and its CTF.

Per tube:

1. find its axis in the data: start from the path (``.paths.json``), and in blocks of
   ``--block-nm`` along it move the centre to where the tube's averaged cross-profile is
   most mirror-symmetric; a smooth curve through those centres is the axis (the drawn or
   traced path only says which tube, never where its axis runs);
2. straighten it: sample the micrograph along the axis, a strip ``--width-nm`` across;
3. cut the strip into windows ``--window-nm`` long (overlapping by 3/4), skipping any
   within ``--exclude-nm`` of another tube (a crossing tube's layer lines would land on
   top) or running off the frame;
4. average the windows' power spectra (layer lines sit at a fixed axial frequency, so
   they add up incoherently), summed over transverse frequency |R| <= ``--r-max``. Both
   lines are near-meridional -- 8 nm: the dimer lattice is not helical (the seam), so it
   splits into Bessel orders n = -1/-2 peaking at |R| < 1/250 A; 4 nm: n = -3 at ~1/170 A
   -- so 1/80 A keeps them and rejects a crossing filament's layer lines, which land off
   the meridian (actin's 5.9 nm line crossing at 45 deg sits at 1/83 A along the tube, but
   1/83 A across);
5. for each layer line (dimer ~8.3 nm, monomer ~4.15 nm), integrate the power over a band
   +/-``--search-percent`` around nominal (compacted to expanded, and pixel-size
   uncertainty, all fall inside) above a straight background fitted to the flanks;
6. divide each excess by CTF^2 at its frequency. At the defocus of these micrographs the
   CTF passes 1/(8 nm) several times more weakly than 1/(4 nm), so without this the
   ratio mostly measures the defocus.

The number to compare is ``ratio = (8 nm excess / CTF^2) / (4 nm excess / CTF^2)``, read
against bare microtubules measured the same way; each line's SNR (excess over the
flanks' scatter) says whether it is there at all. Sections of ``--section-nm`` along each
tube give the same numbers locally, for where a tube is decorated.

Usage:
    python run_scripts/layer_line_check.py --label kinesin_2_0031 \\
        --micrograph results_kinesin/20260916_lysates_2_0031_DW.mrc \\
        --paths configs/filament_constraint_kinesin_2_0031.paths.json \\
        --optics results_ctf_kinesin/20260916_lysates_2_0031_DW_optics.yaml
"""

import argparse
import json
import pathlib
import sys

import mrcfile
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d, map_coordinates, median_filter
from scipy.spatial import cKDTree

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MT_ROOT.parent / "programs" / "constrained_search"))
from napari_choose_filament_path import load_paths_json, spline_through_points  # noqa: E402

LINES = {"8 nm": 83.0, "4 nm": 41.5}   # nominal repeats, A (between compacted and expanded)
PAD = 4                                # axial zero-padding of each window's FFT
# off-lattice repeats (A) for each measurement's own null: clear of 83 and 41.5 and of
# their orders (27.7, 20.75), each band +/- search plus one window resolution
NULL_REPEATS = (33.0, 35.0, 37.0, 52.0, 56.0, 60.0, 64.0, 68.0, 104.0, 112.0, 120.0)


def ctf_squared(k, optics: dict):
    """|CTF|^2 at spatial frequency ``k`` (1/A), at the mean defocus."""
    kv = float(optics["voltage"]) * 1e3
    wavelength = 12.2643 / np.sqrt(kv * (1 + 0.97845e-6 * kv))
    defocus = 0.5 * (float(optics["defocus_u"]) + float(optics["defocus_v"]))
    cs = float(optics["spherical_aberration"]) * 1e7
    amplitude = float(optics["amplitude_contrast_ratio"])
    chi = np.pi * wavelength * defocus * k ** 2 - 0.5 * np.pi * cs * wavelength ** 3 * k ** 4
    return np.sin(chi + np.arcsin(amplitude)) ** 2


def straighten(image, curve, tangent, half_width):
    """The image sampled along ``curve``: (arc, across) strip, bilinear, NaN off-frame."""
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    offsets = np.arange(-half_width, half_width + 1, dtype=float)
    rows = curve[:, None, 0] + offsets[None, :] * normal[:, None, 0]
    cols = curve[:, None, 1] + offsets[None, :] * normal[:, None, 1]
    return map_coordinates(image, [rows, cols], order=1, mode="constant", cval=np.nan)


def symmetric_centre(profile, max_shift, half_span):
    """Offset (px, sub-pixel) about which ``profile`` is most mirror-symmetric."""
    centre = len(profile) // 2
    profile = profile - profile.mean()
    shifts = np.arange(-max_shift, max_shift + 1)
    span = np.arange(1, half_span + 1)
    score = np.array([np.sum(profile[centre + s + span] * profile[centre + s - span])
                      for s in shifts])
    best = int(np.argmax(score))
    if 0 < best < len(score) - 1:
        a, b, c = score[best - 1:best + 2]
        denominator = a - 2 * b + c
        return shifts[best] + (0.5 * (a - c) / denominator if denominator < 0 else 0.0)
    return float(shifts[best])


def refine_axis(image, curve, tangent, pixel, args):
    """The tube's axis from the data: centres of mirror symmetry along the path.

    Returns the refined curve and tangent, and the RMS distance (nm) the axis moved from
    the path on the last pass.
    """
    block = int(round(args.block_nm * 10 / pixel))
    max_shift = int(round(args.max_shift_nm * 10 / pixel))
    half_span = int(round(args.width_nm * 10 / pixel / 2))
    moved = np.nan
    for _ in range(2):
        strip = straighten(image, curve, tangent, half_span + max_shift)
        strip = gaussian_filter1d(strip, 10.0 / pixel, axis=1)    # ~1 nm across
        arcs, offsets = [], []
        for start in range(0, len(curve) - block + 1, block // 2):
            piece = strip[start:start + block]
            if np.isnan(piece).any():
                continue
            arcs.append(start + block // 2)
            offsets.append(symmetric_centre(piece.mean(axis=0), max_shift, half_span))
        if len(offsets) < 5:
            break
        offsets = median_filter(np.asarray(offsets), size=5, mode="nearest")
        normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
        arcs = np.asarray(arcs)
        centres = curve[arcs] + offsets[:, None] * normal[arcs]
        moved = float(np.sqrt(np.mean(offsets ** 2))) * pixel / 10
        # beyond the first/last measured block (off-frame) keep the path, shifted by the
        # nearest measured offset so the curve stays continuous
        head = curve[:arcs[0]:block] + offsets[0] * normal[:arcs[0]:block]
        tail = (curve[arcs[-1] + block::block]
                + offsets[-1] * normal[arcs[-1] + block::block])
        curve, tangent = spline_through_points(np.vstack([head, centres, tail]),
                                               smooth_px=args.axis_smooth_px)
    return curve, tangent, moved


def band_excess(profile, axial, repeat, search, resolution):
    """(excess, its sd, peak repeat A) of the layer line near 1/``repeat``.

    The band runs over +/-``search`` of the nominal frequency plus one ``resolution``
    (the window's own frequency step) each side; the background is a straight line fitted
    to four resolution steps of flank on both sides.
    """
    target = 1.0 / repeat
    lo, hi = target * (1 - search) - resolution, target * (1 + search) + resolution
    band = (axial >= lo) & (axial <= hi)
    flank = (((axial >= lo - 4 * resolution) & (axial < lo)) |
             ((axial > hi) & (axial <= hi + 4 * resolution)))
    fit = np.polyfit(axial[flank], profile[flank], 1)
    residual = profile[flank] - np.polyval(fit, axial[flank])
    excess = profile[band] - np.polyval(fit, axial[band])
    # samples are PAD-fold oversampled: the band holds band.sum() / PAD independent ones,
    # each the sum of PAD samples
    noise = residual.std(ddof=2) * np.sqrt(band.sum() / PAD) * PAD
    peak = axial[band][np.argmax(excess)]
    return float(excess.sum()), float(noise), float(1.0 / peak)


def spectrum(pieces, pixel, r_max):
    """Mean power spectrum of tapered pieces, and its |R| <= r_max meridional profile."""
    window, width = pieces[0].shape
    taper = np.outer(np.hanning(window), np.hanning(width))
    power = np.zeros((window * PAD, width))
    for piece in pieces:
        piece = (piece - piece.mean()) * taper
        power += np.abs(np.fft.fft2(piece, s=(window * PAD, width))) ** 2
    power = np.fft.fftshift(power / len(pieces))
    axial = np.fft.fftshift(np.fft.fftfreq(window * PAD, d=pixel))
    across = np.fft.fftshift(np.fft.fftfreq(width, d=pixel))
    profile = power[:, np.abs(across) <= r_max].sum(axis=1)
    return power, axial, profile


def readout(pieces, pixel, args, optics):
    """Both layer lines, CTF-corrected, from a set of straightened windows.

    Each set carries its own null: the same band measurement at off-lattice repeats
    (``NULL_REPEATS``). Where there is no line the SNR reads about −1 (the flank fit),
    so the 8 nm excess is corrected by the null's mean before the ratio is formed, and
    the ratio's error is the 8 nm line's own noise (with the 4 nm line's added): a
    ratio of two noisy excesses has no usable error when the 8 nm one is near zero.
    CTF^2 is taken at the nominal repeats, not where the band's maximum happens to fall.
    """
    power, axial, profile = spectrum(pieces, pixel, args.r_max)
    resolution = 1.0 / (pieces[0].shape[0] * pixel)
    search = args.search_percent / 100
    result = {"windows": len(pieces)}
    for name, repeat in LINES.items():
        excess, noise, peak = band_excess(profile, axial, repeat, search, resolution)
        ctf = float(ctf_squared(1.0 / repeat, optics))
        result[name] = {"excess": excess, "noise": noise, "snr": excess / noise,
                        "repeat_A": peak, "ctf2": ctf, "corrected": excess / ctf,
                        "corrected_sd": noise / ctf}
    null = np.array([(lambda e: e[0] / e[1])(band_excess(profile, axial, r, search, resolution))
                     for r in NULL_REPEATS])
    result["null_mean"], result["null_sd"] = float(null.mean()), float(null.std(ddof=1))
    result["null_max"] = float(null.max())
    a, b = result["8 nm"], result["4 nm"]
    result["ratio"] = a["corrected"] / b["corrected"]
    result["ratio_sd"] = abs(result["ratio"]) * np.hypot(a["corrected_sd"] / a["corrected"],
                                                         b["corrected_sd"] / b["corrected"])
    corrected_8 = (a["excess"] - result["null_mean"] * a["noise"]) / a["ctf2"]
    result["ratio_bc"] = corrected_8 / b["corrected"]
    result["ratio_bc_sd"] = float(np.hypot(a["corrected_sd"] / b["corrected"],
                                           result["ratio_bc"] / b["snr"]))
    result["_power"], result["_axial"], result["_profile"] = power, axial, profile
    return result


def same_tube(axes, k, pixel, args):
    """Per point of track ``k``: which other tracks run along the same tube there.

    Two microtubules' axes cannot come closer than a tube's width, so another track
    within ``--duplicate-nm`` and within ``--parallel-deg`` of parallel is the same tube
    traced twice (a wall side-lobe of the automatic tracer, or an overlapping piece), not
    a neighbour.
    """
    curve, tangent, _ = axes[k]
    duplicate_px = args.duplicate_nm * 10 / pixel
    parallel = np.cos(np.radians(args.parallel_deg))
    shared = {}
    for j, (other, other_tangent, _) in enumerate(axes):
        if j == k:
            continue
        distance, index = cKDTree(other).query(curve)
        aligned = np.abs(np.sum(tangent * other_tangent[index], axis=1)) > parallel
        shared[j] = (distance < duplicate_px) & aligned
    return shared


def blocked_by_others(axes, k, shared, pixel, args):
    """Points of track ``k`` within ``--exclude-nm`` of a different tube.

    Only tracks within ``--exclude-angle-deg`` of parallel block: a filament crossing at
    angle a puts its layer lines at axial frequency cos(a)/d and transverse sin(a)/d, so
    with a narrow transverse band (``--r-max``) a steep crossing lands outside it. In a
    dense field (lysate) this keeps the windows a blanket exclusion would lose.
    """
    curve, tangent, _ = axes[k]
    blocked = np.zeros(len(curve), dtype=bool)
    near_parallel = np.cos(np.radians(args.exclude_angle_deg))
    for j, (other, other_tangent, _) in enumerate(axes):
        if j == k:
            continue
        distance, index = cKDTree(other).query(curve)
        shallow = np.abs(np.sum(tangent * other_tangent[index], axis=1)) >= near_parallel
        blocked |= (distance < args.exclude_nm * 10 / pixel) & ~shared[j] & shallow
    return blocked


def analyse_tube(image, curve, tangent, blocked, args, pixel, optics):
    """Whole-tube and per-section layer lines for one tube."""
    window = int(round(args.window_nm * 10 / pixel))
    half_width = int(round(args.width_nm * 10 / pixel / 2))
    strip = straighten(image, curve, tangent, half_width)
    clear = ~blocked
    starts = [s for s in range(0, len(curve) - window + 1, window // 4)
              if clear[s:s + window].all() and not np.isnan(strip[s:s + window]).any()]
    if len(starts) < 3:
        return None
    pieces = [strip[s:s + window] for s in starts]
    result = readout(pieces, pixel, args, optics)
    result["_pieces"] = pieces
    result["length_nm"] = len(curve) * pixel / 10
    result["clear_nm"] = len({i for s in starts for i in range(s, s + window)}) * pixel / 10
    section = int(round(args.section_nm * 10 / pixel))
    result["sections"] = []
    for first in range(0, len(curve), section):
        chosen = [p for s, p in zip(starts, pieces) if first <= s + window / 2 < first + section]
        if len(chosen) < 4:
            continue
        local = readout(chosen, pixel, args, optics)
        result["sections"].append({
            "from_nm": first * pixel / 10,
            "to_nm": min(first + section, len(curve)) * pixel / 10,
            "windows": len(chosen), "ratio": local["ratio"], "ratio_sd": local["ratio_sd"],
            "ratio_bc": local["ratio_bc"], "ratio_bc_sd": local["ratio_bc_sd"],
            "null_mean": local["null_mean"],
            "snr_8nm": local["8 nm"]["snr"], "snr_4nm": local["4 nm"]["snr"]})
    result["profile"] = running_profile(starts, pieces, window, curve, pixel, args, optics)
    return result


def running_profile(starts, pieces, window, curve, pixel, args, optics):
    """Both lines in every run of ``--profile-windows`` consecutive clear windows.

    A tube in a crowded field is captured well in some stretches and not in others
    (crossings, overlaps, a path drifting off a bending tube): the 4 nm line says where
    the lattice was caught, and only there does the 8 nm line mean anything. Runs step
    by one window (a quarter window length), so neighbouring entries share windows.
    """
    step, n = window // 4, args.profile_windows
    out = []
    for i in range(len(starts) - n + 1):
        if any(starts[j + 1] - starts[j] != step for j in range(i, i + n - 1)):
            continue
        local = readout(pieces[i:i + n], pixel, args, optics)
        centre = int(round(0.5 * (starts[i] + starts[i + n - 1] + window)))
        out.append({"first": i, "arc_nm": centre * pixel / 10,
                    "row": float(curve[centre, 0]), "col": float(curve[centre, 1]),
                    "snr_4nm": local["4 nm"]["snr"], "snr_8nm": local["8 nm"]["snr"],
                    "null_mean": local["null_mean"], "ratio_bc": local["ratio_bc"],
                    "ratio_bc_sd": local["ratio_bc_sd"]})
    return out


def confirmed_windows(result, args):
    """Indices of the windows in any profile run whose 4 nm line clears the threshold."""
    keep = set()
    for item in result["profile"]:
        if item["snr_4nm"] >= args.min_snr_4nm:
            keep.update(range(item["first"], item["first"] + args.profile_windows))
    return sorted(keep)


def main() -> None:
    """Layer lines for every tube of one micrograph."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--micrograph", type=pathlib.Path, required=True)
    parser.add_argument("--paths", type=pathlib.Path, required=True)
    parser.add_argument("--optics", type=pathlib.Path, required=True,
                        help="YAML with an optics_group (search config or PICASSO)")
    parser.add_argument("--label", required=True)
    parser.add_argument("--tubes", type=int, nargs="*", help="only these paths (1-based)")
    parser.add_argument("--width-nm", type=float, default=50.0,
                        help="strip width: the tube plus any bound motors")
    parser.add_argument("--window-nm", type=float, default=58.0,
                        help="~7 dimers; short enough to fit between crossings")
    parser.add_argument("--section-nm", type=float, default=700.0)
    parser.add_argument("--exclude-nm", type=float, default=50.0,
                        help="skip windows this close to a different tube")
    parser.add_argument("--exclude-angle-deg", type=float, default=90.0,
                        help="only tracks crossing at less than this angle block a window "
                             "(90: all); with --r-max 1/120, 40 keeps crossings' lines out")
    parser.add_argument("--duplicate-nm", type=float, default=30.0,
                        help="a parallel track closer than this is the same tube")
    parser.add_argument("--parallel-deg", type=float, default=10.0)
    parser.add_argument("--min-snr-4nm", type=float, default=5.0,
                        help="a track is a microtubule only if its 4 nm line has this SNR")
    parser.add_argument("--block-nm", type=float, default=50.0)
    parser.add_argument("--max-shift-nm", type=float, default=8.0)
    parser.add_argument("--axis-smooth-px", type=float, default=10.0)
    parser.add_argument("--r-max", type=float, default=1 / 80.0,
                        help="transverse frequency summed over, 1/A: the near-meridian, "
                             "where both lines live (seam-split n = -1/-2 at 8 nm, n = -3 "
                             "at 4 nm) and a crossing filament's lines do not")
    parser.add_argument("--search-percent", type=float, default=3.5)
    parser.add_argument("--profile-windows", type=int, default=4,
                        help="consecutive windows per entry of the running profile along "
                             "each tube (4 x 58 nm overlapping 3/4: ~100 nm)")
    parser.add_argument("--outdir", type=pathlib.Path, default=MT_ROOT / "results_layer_lines")
    args = parser.parse_args()

    optics = yaml.safe_load(args.optics.read_text())["optics_group"]
    pixel = float(optics["pixel_size"])
    with mrcfile.open(str(args.micrograph), permissive=True) as handle:
        image = np.asarray(handle.data, dtype=np.float32)
    image = (image - image.mean()) / image.std()
    _, paths = load_paths_json(str(args.paths))
    numbers = args.tubes or list(range(1, len(paths) + 1))
    axes = [refine_axis(image, *spline_through_points(p["points"], smooth_px=5.0), pixel, args)
            for p in paths]
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"{args.label}: {args.micrograph.name}, {pixel} A/px, defocus "
          f"{0.5 * (optics['defocus_u'] + optics['defocus_v']) / 1e4:.2f} um, "
          f"{optics['voltage']:.0f} kV")
    print(f"{'tube':<5}{'axis moved':>11}{'clear':>8}{'8nm A':>7}{'SNR':>6}{'null':>11}"
          f"{'4nm SNR':>8}{'8/4, null-corrected':>22}  same tube as")
    rows = []
    for k in numbers:
        curve, tangent, moved = axes[k - 1]
        shared = same_tube(axes, k - 1, pixel, args)
        blocked = blocked_by_others(axes, k - 1, shared, pixel, args)
        result = analyse_tube(image, curve, tangent, blocked, args, pixel, optics)
        if result is None:
            print(f"{k:<5} fewer than 3 windows clear of the other tubes")
            continue
        result["axis_moved_nm"] = moved
        result["same_tube_as"] = [j + 1 for j, near in shared.items() if near.mean() > 0.5]
        a, b = result["8 nm"], result["4 nm"]
        print(f"{k:<5}{moved:>9.1f}nm{result['clear_nm']:>6.0f}nm{a['repeat_A']:>7.1f}"
              f"{a['snr']:>6.1f} {result['null_mean']:+.1f}..{result['null_max']:+.1f}"
              f"{b['snr']:>8.1f}{result['ratio_bc']:>13.2f} +/- {result['ratio_bc_sd']:.2f}  "
              f"{result['same_tube_as'] or ''}"
              + ("" if b["snr"] >= args.min_snr_4nm else "  (no 4 nm line: not a tube)"))
        for item in result["sections"]:
            print(f"      {item['from_nm']:>5.0f}-{item['to_nm']:<5.0f} nm "
                  f"{item['windows']:>3} windows  8/4 {item['ratio_bc']:6.2f} +/- "
                  f"{item['ratio_bc_sd']:.2f}  (SNR 8 nm {item['snr_8nm']:.1f}, "
                  f"4 nm {item['snr_4nm']:.1f}, null mean {item['null_mean']:+.1f})")
        rows.append((k, result))

    # pooled over the micrograph: every track with a 4 nm line, one track per tube (the
    # one whose 4 nm line is strongest), all their clear windows together
    tubes = sorted([(k, r) for k, r in rows if r["4 nm"]["snr"] >= args.min_snr_4nm],
                   key=lambda item: -item[1]["4 nm"]["snr"])
    kept = []
    for k, r in tubes:
        if not any(j in r["same_tube_as"] or k in other["same_tube_as"] for j, other in kept):
            kept.append((k, r))
    pooled = None
    if kept:
        pooled = readout([p for _, r in kept for p in r["_pieces"]], pixel, args, optics)
        a, b = pooled["8 nm"], pooled["4 nm"]
        print(f"pooled over tracks {[k for k, _ in kept]}: {pooled['windows']} windows, "
              f"8 nm SNR {a['snr']:.1f} ({a['repeat_A']:.1f} A; null {pooled['null_mean']:+.1f}.."
              f"{pooled['null_max']:+.1f}), 4 nm SNR {b['snr']:.1f}, 8/4 null-corrected "
              f"{pooled['ratio_bc']:.2f} +/- {pooled['ratio_bc_sd']:.2f}")

    # confirmed stretches: only windows in runs whose own 4 nm line clears the threshold --
    # where the lattice was caught -- per tube and pooled (a different band decides, so
    # the 8 nm line is not selected on)
    confirmed = {}
    pool = []
    print(f"\nconfirmed stretches (runs of {args.profile_windows} windows with 4 nm SNR >= "
          f"{args.min_snr_4nm:g}):")
    for k, r in rows:
        chosen = confirmed_windows(r, args)
        if len(chosen) < 3:
            print(f"  tube {k}: none")
            continue
        local = readout([r["_pieces"][i] for i in chosen], pixel, args, optics)
        confirmed[k] = local
        if not any(k in other_r["same_tube_as"] for other_k, other_r in rows if other_k in confirmed
                   and other_k != k):
            pool += [r["_pieces"][i] for i in chosen]
        print(f"  tube {k}: {len(chosen)} windows, 8 nm SNR {local['8 nm']['snr']:+.1f} (null "
              f"{local['null_mean']:+.1f}..{local['null_max']:+.1f}), 4 nm SNR "
              f"{local['4 nm']['snr']:.1f}, 8/4 null-corrected {local['ratio_bc']:.2f} +/- "
              f"{local['ratio_bc_sd']:.2f}")
    pooled_confirmed = readout(pool, pixel, args, optics) if len(pool) >= 3 else None
    if pooled_confirmed is not None:
        print(f"  pooled: {len(pool)} windows, 8 nm SNR {pooled_confirmed['8 nm']['snr']:+.1f} "
              f"(null {pooled_confirmed['null_mean']:+.1f}..{pooled_confirmed['null_max']:+.1f}),"
              f" 4 nm SNR {pooled_confirmed['4 nm']['snr']:.1f}, 8/4 null-corrected "
              f"{pooled_confirmed['ratio_bc']:.2f} +/- {pooled_confirmed['ratio_bc_sd']:.2f}")

    def brief(item):
        return None if item is None else {
            "windows": item["windows"], "ratio_bc": item["ratio_bc"],
            "ratio_bc_sd": item["ratio_bc_sd"], "null_mean": item["null_mean"],
            "null_max": item["null_max"], "snr_8nm": item["8 nm"]["snr"],
            "snr_4nm": item["4 nm"]["snr"]}

    summary = {"label": args.label, "micrograph": str(args.micrograph), "pixel": pixel,
               "confirmed": {"pooled": brief(pooled_confirmed),
                             "tubes": {str(k): brief(v) for k, v in confirmed.items()}},
               "defocus_A": 0.5 * (optics["defocus_u"] + optics["defocus_v"]),
               "pooled": None if pooled is None else {
                   "tracks": [k for k, _ in kept], "windows": pooled["windows"],
                   "ratio": pooled["ratio"], "ratio_sd": pooled["ratio_sd"],
                   "ratio_bc": pooled["ratio_bc"], "ratio_bc_sd": pooled["ratio_bc_sd"],
                   "null_mean": pooled["null_mean"], "null_max": pooled["null_max"],
                   **{f"{n}_{key}": pooled[n][key] for n in LINES for key in
                      ("repeat_A", "snr", "ctf2", "excess", "corrected")}},
               "settings": {key: (str(value) if isinstance(value, pathlib.Path) else value)
                            for key, value in vars(args).items()},
               "tubes": [{"tube": k, "windows": r["windows"], "clear_nm": r["clear_nm"],
                          "ratio": r["ratio"], "ratio_sd": r["ratio_sd"],
                          "ratio_bc": r["ratio_bc"], "ratio_bc_sd": r["ratio_bc_sd"],
                          "null_mean": r["null_mean"], "null_max": r["null_max"],
                          "axis_moved_nm": r["axis_moved_nm"],
                          "same_tube_as": r["same_tube_as"],
                          **{f"{n}_{key}": r[n][key] for n in LINES for key in
                             ("repeat_A", "snr", "ctf2", "excess", "corrected")},
                          "sections": r["sections"], "profile": r["profile"]}
                         for k, r in rows]}
    (args.outdir / f"{args.label}.json").write_text(json.dumps(summary, indent=2))
    plot(rows, args, pixel)
    plot_profile(rows, args, image)
    print(f"wrote {args.outdir / (args.label + '.json')} and .png")


def plot_profile(rows, args, image):
    """Where along each tube the lattice was caught (4 nm) and where the 8 nm line is."""
    import matplotlib  # pylint: disable=import-outside-toplevel
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    from scipy.ndimage import gaussian_filter  # pylint: disable=import-outside-toplevel

    rows = [(k, r) for k, r in rows if r["profile"]]
    if not rows:
        return
    fig = plt.figure(figsize=(16, 2.4 * len(rows) + 1))
    grid = fig.add_gridspec(len(rows), 2, width_ratios=[1.1, 1.0])
    ax_map = fig.add_subplot(grid[:, 0])
    b = 8
    h, w = image.shape[0] // b * b, image.shape[1] // b * b
    small = image[:h, :w].reshape(h // b, b, w // b, b).mean((1, 3))
    small = gaussian_filter(small, 1.0) - gaussian_filter(small, 40)
    low, high = np.percentile(small, [1, 99])
    ax_map.imshow(small, cmap="gray", vmin=low, vmax=high)
    cmap = plt.get_cmap("coolwarm")
    for i, (k, r) in enumerate(rows):
        prof = r["profile"]
        excess = np.array([p["snr_8nm"] - p["null_mean"] for p in prof])
        caught = np.array([p["snr_4nm"] >= args.min_snr_4nm for p in prof])
        rows_, cols = np.array([p["row"] for p in prof]) / b, np.array([p["col"] for p in prof]) / b
        ax_map.scatter(cols[~caught], rows_[~caught], s=6, color="#999999")
        scatter = ax_map.scatter(cols[caught], rows_[caught], s=26, c=excess[caught], cmap=cmap,
                                 vmin=-4, vmax=4, edgecolors="none")
        ax_map.text(cols[len(cols) // 2] + 6, rows_[len(rows_) // 2], str(k), color="white",
                    fontsize=11, fontweight="bold")
        ax = fig.add_subplot(grid[i, 1])
        arc = np.array([p["arc_nm"] for p in prof])
        ax.plot(arc, [p["snr_4nm"] for p in prof], color="#1f4e79", lw=1.4, label="4 nm SNR")
        ax.plot(arc, excess, color="#c62828", lw=1.4, label="8 nm SNR − null mean")
        ax.axhline(args.min_snr_4nm, color="#1f4e79", lw=0.7, ls=":")
        ax.axhline(0, color="grey", lw=0.7)
        ax.axhline(3, color="#c62828", lw=0.7, ls=":")
        for x, ok in zip(arc, caught):
            if ok:
                ax.axvspan(x - 7, x + 7, color="#e8eef4", lw=0, zorder=0)
        ax.set_ylabel(f"tube {k}", fontsize=10)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
        if i == len(rows) - 1:
            ax.set_xlabel("position along the path (nm)")
    ax_map.set_xticks([]), ax_map.set_yticks([])
    fig.colorbar(scatter, ax=ax_map, fraction=0.03, label="8 nm SNR − null mean")
    ax_map.set_title(f"{args.label}: running {args.profile_windows}-window layer lines; grey = no "
                     "4 nm line there", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.outdir / f"{args.label}_profile.png", dpi=90)


def plot(rows, args, pixel):
    """Per tube: averaged straightened power spectrum, and its meridional profile."""
    import matplotlib  # pylint: disable=import-outside-toplevel
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    if not rows:
        return
    colours = {"8 nm": "#c62828", "4 nm": "#1f4e79"}
    fig, axes = plt.subplots(2, len(rows), figsize=(3.2 * len(rows), 6.5), squeeze=False)
    for col, (k, result) in enumerate(rows):
        power, axial, profile = result["_power"], result["_axial"], result["_profile"]
        across_max = 1 / (2 * pixel)
        shown = (axial > -1 / 30) & (axial < 1 / 30)
        top = axes[0, col]
        top.imshow(np.log(power[shown] + 1e-12), cmap="gray", aspect="auto",
                   extent=(-across_max, across_max, axial[shown][-1], axial[shown][0]))
        for name in LINES:
            top.axhline(1 / result[name]["repeat_A"], color=colours[name], lw=0.6, ls=":")
        top.set_xlim(-1 / 25, 1 / 25)
        top.set_title(f"tube {k}: 8/4 = {result['ratio']:.2f} +/- {result['ratio_sd']:.2f}",
                      fontsize=9)
        top.set_xlabel("across (1/A)", fontsize=8)
        top.set_ylabel("along (1/A)", fontsize=8)
        bottom = axes[1, col]
        sel = (axial > 1 / 200) & (axial < 1 / 30)
        bottom.plot(axial[sel], profile[sel], color="black", lw=0.8)
        for name in LINES:
            bottom.axvline(1 / result[name]["repeat_A"], color=colours[name], lw=0.8, ls="--",
                           label=f"{name}: SNR {result[name]['snr']:.1f}")
        bottom.set_yscale("log")
        bottom.legend(fontsize=7)
        bottom.set_xlabel("axial frequency (1/A)", fontsize=8)
    fig.suptitle(f"{args.label}: straightened tubes' power spectra (profile: |R| <= "
                 f"1/{1 / args.r_max:.0f} A summed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.outdir / f"{args.label}.png", dpi=110)


if __name__ == "__main__":
    main()
