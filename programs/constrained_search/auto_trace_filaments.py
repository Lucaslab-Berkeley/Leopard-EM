"""Propose filament paths automatically: a coarse template-matching tracer.

Stage 1 of the microtubule pipeline has a person trace each microtubule in napari
(``napari_choose_filament_path.py``). Those traces only seed the analysis -- every later
stage rebuilds each tube's path, and which detections belong to it, from the data -- so
they need to be roughly right: within the +/-20 deg direction cone and a band ~50 nm wide.
This proposes them automatically. The output is the drawing tool's own ``.paths.json``,
so a person reviews and corrects it in the same tool instead of drawing from scratch::

    python napari_choose_filament_path.py --micrograph <DW.mrc> \\
        --paths <prefix>.paths.json --output <sidecar.yaml> ...

Method
------
1. **Bin, and build a coarse template.** The micrograph is Fourier-cropped by an integer
   factor to about ``--target-pixel``. At the coarse resolution a microtubule IS a smooth
   hollow tube, so the template is a ``--cylinder-length-nm`` (64 nm) long cylinder with
   the microtubule's own radial profile -- the rotational average of the dataset's ring
   template, so the radius matches its protofilament number -- low-passed to
   ``--lowpass``. A long cylinder gathers far more signal than the 16 nm ring (z grows
   with the square root of the length; on set1_823 the ring at 15 A barely registered),
   and it has no roll and looks the same turned by 180 deg, so neither is searched.
2. **Resolution from the defocus** (unless set). The coarse search can only use spatial
   frequencies the CTF passes. The first zero sits near sqrt(lambda * defocus): ~13-14 A
   at 0.8-1 um, but ~6 A for a near-focus image (GDP, 0.2 um), where a 15 A search sees
   nothing. So ``--lowpass`` defaults to 1.2x that, kept within 8-15 A, and the binning to
   about a third of the low-pass (never coarser than 4.5 A/px).
3. **Coarse search.** Leopard-EM's own ``match_template`` on the binned pair at one
   defocus plane: psi over half a turn in steps of the template's angular resolution
   (low-pass / length, ~1.3 deg), theta within ``--theta-half-range`` (10 deg) of
   in-plane, one phi. A few hundred orientations: well under a minute on one GPU. It
   gives a per-pixel z (scaled MIP) and psi; the z map peaks along each tube's AXIS.
4. **Ridge points.** Pixels above the background (median + ``--threshold-sigma`` robust
   sd) that beat the z map ``--nms-radius-nm`` (22 nm) either side ACROSS the tube --
   perpendicular to their own psi. The reach matters: a cylinder also scores where its
   wall meets the tube's other wall, a side-lobe ~22 nm off the axis; two touching
   tubes are >= 25 nm apart, so real neighbours survive. Pixels within
   ``--gold-exclusion-nm`` of a very dark blob (gold, ice) are dropped: gold puts a
   bright halo in the z map that otherwise seeds a knot of short tracks.
5. **Tracing.** From the strongest ridge point not yet used, step along its direction
   both ways; at each step take the strongest ridge point near the predicted position
   whose direction agrees within ``--angle-tolerance``, and turn by at most
   ``--max-turn-deg`` per step (a microtubule cannot bend tighter than a few hundred
   nm, so a faster turn is the walker stepping onto a crossing tube). Through gaps --
   crossings, where the per-pixel maximum keeps only one tube's angle, or weak stretches
   -- it continues straight, up to ``--max-gap-nm``. Ridge points along a finished track
   that point its way are claimed, so a tube is traced once while a tube crossing it at
   another angle keeps its points.
6. **Clean-up.** Tracks shorter than ``--min-length-nm`` are dropped; tracks that carry
   on from one another in line -- across a gap, or overlapping side by side -- are
   joined; a track lying mostly within 6 nm of a longer one is dropped.
7. **Output.** Each track is smoothed with the drawing tool's own spline and resampled
   to a vertex every ``--vertex-spacing-nm`` (50 nm: few enough to edit by hand), with a
   band of ``--width-nm`` and polarity ``both``, strongest track first. Written to
   ``<prefix>.paths.json``; ``<prefix>_overlay.png`` shows the tracks on the micrograph
   and on the coarse z map; ``<prefix>_autotrace.json`` records every parameter, each
   track's length and mean z, and with ``--reference`` (hand-drawn ``.paths.json``) how
   much of each hand-drawn path an automatic track covers, how far off, at what angle,
   and which automatic tracks are extra (not necessarily wrong: a person draws the tubes
   they want, not all of them).

What it does not do
-------------------
* **It finds tubes, not microtubules.** At 8-15 A a bundle of thinner filaments ~25 nm
  wide looks like a microtubule to a smooth cylinder, and the coarse z does not separate
  them (on set1_823, an unroofed neurite, the hand-drawn tubes and the extra tracks score
  alike). In vitro this rarely matters; in cells, review the extra tracks. A check with
  the full-resolution ring at a few points along each track would separate them.
* **Polarity** is left ``both``; Stage 3 measures it from the full search.
* **Edges.** Like any template search it only scores positions where the whole template
  box fits, so nothing within about half a box (~45 nm here) of the frame edge is traced.
* **Crowded fields** (dozens of crossing tubes, as on the GDP micrograph) fragment the
  ridges; a lower ``--threshold-sigma`` (3) recovers more of each tube at the cost of
  more extra tracks and more pieces per tube.

Usage
-----
    python programs/constrained_search/auto_trace_filaments.py \\
        --micrograph MT_tests/results_kinesin/20260916_lysates_2_0031_DW.mrc \\
        --template MT_tests/maps/6dpu_14pf_2rings_flatB_1.13_bscale0.5.mrc \\
        --optics MT_tests/results_ctf_kinesin/20260916_lysates_2_0031_DW_optics.yaml \\
        --outdir MT_tests/results_autotrace --prefix kinesin_2_0031 --gpu 3

``--optics`` is any YAML with an ``optics_group`` (a Leopard-EM search config or a PICASSO
``*_optics.yaml``); the micrograph's pixel size is read from it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import mrcfile
import numpy as np
import yaml
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
# The drawing tool owns the path geometry and file format; reuse, do not re-derive.
from napari_choose_filament_path import (  # noqa: E402
    load_paths_json,
    save_paths_json,
    spline_through_points,
)


# --------------------------------------------------------------------------- binning
def fourier_crop(array: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by an integer ``factor`` in every axis by cropping the spectrum.

    Band-limited, so no aliasing; the grid origin is kept, so binned index k sits at
    full-resolution index k * factor.
    """
    shape = tuple(n // factor * factor for n in array.shape)
    array = array[tuple(slice(0, n) for n in shape)]
    spectrum = np.fft.fftshift(np.fft.fftn(array))
    out_shape = tuple(n // factor for n in shape)
    # Zero frequency sits at n // 2 after fftshift and must land at m // 2 of the crop
    # (where ifftshift expects it). (n - m) // 2 is one short when m is odd, which
    # multiplies the image by a cosine -- two flat bands across a 1023-px-wide bin.
    starts = tuple(n // 2 - m // 2 for n, m in zip(shape, out_shape))
    cropped = spectrum[tuple(slice(s, s + m) for s, m in zip(starts, out_shape))]
    return np.real(np.fft.ifftn(np.fft.ifftshift(cropped))).astype(np.float32)


def lowpass(volume: np.ndarray, pixel_size: float, resolution: float) -> np.ndarray:
    """Cosine-edged low-pass to ``resolution`` A (half amplitude there)."""
    freqs = np.meshgrid(*[np.fft.fftfreq(n, d=pixel_size) for n in volume.shape],
                        indexing="ij")
    radius = np.sqrt(sum(f ** 2 for f in freqs))
    centre, half_width = 1.0 / resolution, 0.15 / resolution
    edge = np.clip((radius - (centre - half_width)) / (2 * half_width), 0.0, 1.0)
    weight = 0.5 * (1 + np.cos(np.pi * edge))
    return np.real(np.fft.ifftn(np.fft.fftn(volume) * weight)).astype(np.float32)


def radial_profile(volume: np.ndarray, pixel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotational average of a tube map about its axis (array axis 0): (radius A, density).

    Averaged over every slice that holds density, so it is the wall's mean radial
    profile -- the protofilament number sets the radius, the lattice does not matter.
    """
    n = volume.shape[1]
    yy, xx = np.indices(volume.shape[1:]) - (n // 2)
    radius = np.hypot(yy, xx).astype(np.int64)
    occupied = volume.reshape(volume.shape[0], -1).max(axis=1) > 0.05 * volume.max()
    stack = volume[occupied].mean(axis=0)
    density = np.bincount(radius.ravel(), stack.ravel()) / np.bincount(radius.ravel())
    return np.arange(len(density)) * pixel_size, density


def cylinder_template(radius_a, density, pixel_size, length_nm, resolution):
    """A smooth hollow tube of the given radial profile, ``length_nm`` long, axis on z.

    The coarse step only needs where a microtubule is and which way it runs. At 15-20 A
    a microtubule IS a smooth tube, so a long cylinder of its own radial profile is the
    matched filter: signal grows with length (z with its square root) while the roll
    angle, and the 180 deg turn, drop out of the search entirely.
    """
    length = length_nm * 10.0
    margin = 2.0 * resolution + 40.0
    box = int(np.ceil((max(length, 2 * radius_a[density > 0.05 * density.max()].max())
                       + 2 * margin) / pixel_size))
    box += box % 2
    centre = box // 2
    z, y, x = (np.indices((box, box, box)) - centre) * pixel_size
    radial = np.interp(np.hypot(y, x), radius_a, density, right=0.0)
    # cosine-tapered ends over 5 nm, so the ends are not a sharp feature in the filter
    taper = 50.0
    along = np.clip((np.abs(z) - (length / 2 - taper)) / taper, 0.0, 1.0)
    volume = radial * 0.5 * (1 + np.cos(np.pi * along))
    return lowpass(volume.astype(np.float32), pixel_size, resolution)


def write_mrc(path: pathlib.Path, data: np.ndarray, pixel_size: float) -> None:
    """Write ``data`` with its voxel size."""
    with mrcfile.new(str(path), overwrite=True) as handle:
        handle.set_data(np.ascontiguousarray(data, dtype=np.float32))
        handle.voxel_size = pixel_size


# ----------------------------------------------------------------------- coarse search
def read_optics(path: pathlib.Path) -> dict:
    """The ``optics_group`` of a Leopard-EM search config or a PICASSO optics YAML."""
    payload = yaml.safe_load(pathlib.Path(path).read_text())
    optics = dict(payload["optics_group"])
    optics.setdefault("label", pathlib.Path(path).stem)
    return optics


def coarse_search(micrograph_path, template_path, optics, args, workdir):
    """Leopard-EM match_template on the binned pair; (z map, psi map, half box)."""
    # pylint: disable=import-outside-toplevel
    from leopard_em.pydantic_models.managers import MatchTemplateManager

    out = {name: str(workdir / f"coarse_{name}.mrc") for name in (
        "mip", "scaled_mip", "orientation_psi", "orientation_theta", "orientation_phi",
        "relative_defocus", "correlation_average", "correlation_variance")}
    config = {
        "micrograph_path": str(micrograph_path),
        "template_volume_path": str(template_path),
        "optics_group": optics,
        "defocus_search_config": {"enabled": False, "defocus_min": 0.0,
                                  "defocus_max": 0.0, "defocus_step": 200.0},
        # A cylinder has no roll and looks the same turned by 180 deg, so: one phi,
        # psi over half a turn, and theta near in-plane.
        "orientation_search_config": {
            "base_grid_method": "uniform", "symmetry": None,
            "psi_step": args.psi_step, "theta_step": args.theta_step,
            "psi_min": 0.0, "psi_max": 180.0,
            "theta_min": 90.0 - args.theta_half_range,
            "theta_max": 90.0 + args.theta_half_range,
            "phi_min": 0.0, "phi_max": 0.5,
        },
        "computational_config": {"gpu_ids": [args.gpu], "num_cpus": 4},
        "preprocessing_filters": {
            "whitening_filter": {"enabled": True, "do_power_spectrum": True,
                                 "max_freq": 1.0, "num_freq_bins": None},
            "bandpass_filter": {"enabled": False, "falloff": 0.05,
                                "high_freq_cutoff": 0.5, "low_freq_cutoff": 0.0},
        },
        "match_template_result": {"allow_file_overwrite": True,
                                  **{f"{k}_path": v for k, v in out.items()}},
    }
    config_path = workdir / "coarse_search.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    manager = MatchTemplateManager.from_yaml(str(config_path))
    n_orient = int(manager.orientation_search_config.euler_angles.shape[0])
    print(f"coarse search: {n_orient:,} orientations, gpu {args.gpu}")
    start = time.time()
    manager.run_match_template(orientation_batch_size=16, do_result_export=True)
    print(f"  done in {time.time() - start:.0f}s")
    with mrcfile.open(out["scaled_mip"], permissive=True) as handle:
        z = np.asarray(handle.data, dtype=np.float32)
    with mrcfile.open(out["orientation_psi"], permissive=True) as handle:
        psi = np.asarray(handle.data, dtype=np.float32)
    with mrcfile.open(str(template_path), header_only=True, permissive=True) as handle:
        half = int(handle.header.nx) // 2
    return z, psi, half


# ------------------------------------------------------------------------ the tracer
def tangent_from_psi(psi_deg: np.ndarray) -> np.ndarray:
    """(dy, dx) unit tangent from psi; inverse of the exporter's atan2(-dy, dx)."""
    psi = np.radians(np.asarray(psi_deg, dtype=np.float64))
    return np.stack([-np.sin(psi), np.cos(psi)], axis=-1)


def ridge_points(z: np.ndarray, psi: np.ndarray, threshold: float, radius: float,
                 excluded: np.ndarray | None = None):
    """Pixels above ``threshold`` that are the maximum ACROSS the tube (y, x, z, tangent).

    "Across" reaches ``radius`` (a tube radius) either side, perpendicular to the
    pixel's own psi -- not just the next pixel. A cylinder template also scores where
    one of its walls lines up with the tube's opposite wall, a side-lobe about one
    radius off the axis; requiring the axis to beat everything within a radius removes
    it, while real neighbours (bundled tubes sit >= 2 radii apart) are untouched.
    ``excluded`` (same shape as ``z``) removes masked pixels, e.g. around gold.
    """
    above = z > threshold
    if excluded is not None:
        above &= ~excluded
    ys, xs = np.nonzero(above)
    tangent = tangent_from_psi(psi[ys, xs])
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=-1)
    keep = np.ones(len(ys), dtype=bool)
    for offset in np.arange(1.0, radius + 1.0):
        for sign in (1.0, -1.0):
            neighbour = map_coordinates(
                z, [ys + sign * offset * normal[:, 0], xs + sign * offset * normal[:, 1]],
                order=1, mode="nearest")
            keep &= z[ys, xs] >= neighbour
    return (np.column_stack([ys[keep], xs[keep]]).astype(np.float64), z[ys, xs][keep],
            tangent[keep])


def walk(start, direction, points, values, tangents, tree, shape, params):
    """Follow the ridge from ``start`` along ``direction``; the points found, in order."""
    step, lateral, max_gap, cos_tol = (params["step"], params["lateral"],
                                       params["max_gap"], params["cos_tol"])
    max_turn = params["max_turn"]          # radians per step
    found, history = [], [np.asarray(start, dtype=np.float64)]
    position, heading, gap = history[0].copy(), np.asarray(direction, float), 0.0
    for _ in range(100_000):
        predicted = position + step * heading
        if not (0 <= predicted[0] < shape[0] and 0 <= predicted[1] < shape[1]):
            break
        near = tree.query_ball_point(predicted, r=np.hypot(step / 2, lateral))
        best = None
        if near:
            near = np.asarray(near)
            offset = points[near] - predicted
            normal = np.array([heading[1], -heading[0]])
            ahead = (points[near] - position) @ heading > 0.25 * step
            aligned = np.abs(tangents[near] @ heading) >= cos_tol
            beside = np.abs(offset @ normal) <= lateral
            ok = near[ahead & aligned & beside]
            if ok.size:
                best = ok[np.argmax(values[ok])]
        if best is None:
            gap += step
            if gap > max_gap:
                break
            position = predicted
            continue
        gap = 0.0
        position = points[best].copy()
        found.append(best)
        history.append(position)
        # New heading: the displacement over the last ~20 nm, nudged by the point's
        # own angle -- but turned by at most ``max_turn`` per step. A microtubule
        # cannot bend tighter than a few hundred nm, so a faster turn is the walker
        # stepping onto a crossing tube, not the tube bending.
        back = history[max(0, len(history) - params["history"])]
        travelled = position - back
        local = tangents[best] * np.sign(tangents[best] @ heading or 1.0)
        wanted = (travelled / np.linalg.norm(travelled) + 0.5 * local
                  if np.linalg.norm(travelled) > step else heading + 0.5 * local)
        wanted /= np.linalg.norm(wanted)
        turn = np.arctan2(heading[0] * wanted[1] - heading[1] * wanted[0], heading @ wanted)
        turn = float(np.clip(turn, -max_turn, max_turn))
        c, s_ = np.cos(turn), np.sin(turn)
        heading = np.array([c * heading[0] - s_ * heading[1], s_ * heading[0] + c * heading[1]])
        heading /= np.linalg.norm(heading)
    return found


def trace(points, values, tangents, shape, params):
    """Tracks (lists of ridge-point indices in order), strongest seeds first."""
    tree = cKDTree(points)
    claimed = np.zeros(len(points), dtype=bool)
    tracks = []
    for seed in np.argsort(-values):
        if claimed[seed]:
            continue
        forward = walk(points[seed], tangents[seed], points, values, tangents, tree,
                       shape, params)
        backward = walk(points[seed], -tangents[seed], points, values, tangents, tree,
                        shape, params)
        track = list(reversed(backward)) + [seed] + forward
        claimed[seed] = True
        coords = points[track]
        if len(track) < 2:
            continue
        length = float(np.sum(np.hypot(*np.diff(coords, axis=0).T)))
        if length < params["min_length"]:
            continue
        # Claim the ridge points that lie along this track AND point its way.
        for k, index in enumerate(track):
            a, b = coords[max(k - 2, 0)], coords[min(k + 2, len(coords) - 1)]
            local = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            near = np.asarray(tree.query_ball_point(points[index], r=params["claim"]))
            if near.size:
                aligned = np.abs(tangents[near] @ local) >= params["cos_tol"]
                claimed[near[aligned]] = True
        tracks.append(track)
    return tracks


def end_direction(coords: np.ndarray, at_start: bool, span: int = 6) -> np.ndarray:
    """Outward unit direction at one end of a polyline."""
    if at_start:
        vector = coords[0] - coords[min(span, len(coords) - 1)]
    else:
        vector = coords[-1] - coords[max(len(coords) - 1 - span, 0)]
    return vector / max(np.linalg.norm(vector), 1e-9)


def join(a: np.ndarray, b: np.ndarray, params) -> np.ndarray | None:
    """``a`` continued by ``b`` if b carries on from a's end in line, else None.

    In line: b runs a's way (within the angle tolerance) and b's first point sits
    within ``join_lateral`` of the line a's end is on. b may start up to
    ``merge_gap`` past a's end (a gap) or overlap a's end (fragments of one tube found
    from different seeds); only b's points beyond a's end are kept.
    """
    out_a = end_direction(a, at_start=False)
    in_b = -end_direction(b, at_start=True)
    if out_a @ in_b < params["cos_tol"]:
        return None
    normal = np.array([out_a[1], -out_a[0]])
    offset = b[0] - a[-1]
    along, lateral = float(offset @ out_a), abs(float(offset @ normal))
    if lateral > params["join_lateral"] or along > params["merge_gap"]:
        return None
    if along < 0:   # overlapping: b's start must lie beside a, not off to one side
        distance, _ = cKDTree(a).query(b[0])
        if distance > params["join_lateral"]:
            return None
    beyond = (b - a[-1]) @ out_a > 0
    if not beyond.any():
        return a
    return np.vstack([a, b[np.argmax(beyond):]])


def merge_tracks(tracks: list[np.ndarray], params) -> list[np.ndarray]:
    """Join tracks that continue one another in line; repeat until none do."""
    tracks = [np.asarray(t, dtype=np.float64) for t in tracks]
    merged = True
    while merged:
        merged = False
        for i in range(len(tracks)):
            for j in range(len(tracks)):
                if i == j:
                    continue
                for a in (tracks[i], tracks[i][::-1]):
                    for b in (tracks[j], tracks[j][::-1]):
                        joined = join(a, b, params)
                        if joined is not None:
                            tracks[i] = joined
                            del tracks[j]
                            merged = True
                            break
                    if merged:
                        break
                if merged:
                    break
            if merged:
                break
    return tracks


def drop_duplicates(tracks: list[np.ndarray], params) -> list[np.ndarray]:
    """Drop a track that mostly lies along a longer one."""
    lengths = [float(np.sum(np.hypot(*np.diff(t, axis=0).T))) for t in tracks]
    order = np.argsort(lengths)[::-1]
    kept: list[np.ndarray] = []
    for index in order:
        track = tracks[index]
        duplicate = False
        for other in kept:
            distance, _ = cKDTree(other).query(track)
            if np.mean(distance <= params["join_lateral"]) > 0.5:
                duplicate = True
                break
        if not duplicate:
            kept.append(track)
    return kept


def gold_mask(image: np.ndarray, pixel_size: float, radius_nm: float,
              sigma: float = 8.0) -> np.ndarray:
    """Pixels within ``radius_nm`` of a very dark blob (gold fiducial, ice crystal).

    Gold is far darker than anything biological; its halo in the z map runs out to
    about half the template length, so the exclusion covers both. Found on a lightly
    smoothed copy as pixels more than ``sigma`` robust sd below the median.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter  # pylint: disable=import-outside-toplevel

    smooth = gaussian_filter(image, 1.5)
    median = float(np.median(smooth))
    robust_sd = 1.4826 * float(np.median(np.abs(smooth - median)))
    dark = smooth < median - sigma * robust_sd
    if not dark.any():
        return dark
    radius = int(np.ceil(radius_nm * 10.0 / pixel_size))
    # Dilation by a disk of that radius, as a distance transform: binary_dilation with
    # the (2r+1)^2 disk (r = 176 px at 2.28 A/px) allocated ~96 GB and ran for many
    # minutes whenever any pixel was dark; this is the same set, in seconds.
    return distance_transform_edt(~dark) <= radius


def track_score(track: np.ndarray, z: np.ndarray) -> float:
    """Mean z along a track (coarse-map coordinates)."""
    return float(np.mean(map_coordinates(z, track.T, order=1, mode="nearest")))


# --------------------------------------------------------------------------- output
def to_full_resolution(track: np.ndarray, half: int, factor: int) -> np.ndarray:
    """Coarse-map (y, x) -> full-resolution micrograph (row, column).

    The map pixel is the template box's top-left corner; its centre is ``half`` further
    on, and binned index k sits at full-resolution k * factor.
    """
    return (track + half) * factor


def resample_vertices(curve: np.ndarray, spacing: float) -> np.ndarray:
    """Points along ``curve`` every ``spacing`` px of arc, both ends included."""
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(curve, axis=0).T))])
    targets = np.append(np.arange(0.0, arc[-1], spacing), arc[-1])
    return np.column_stack([np.interp(targets, arc, curve[:, k]) for k in (0, 1)])


def compare(auto_paths, reference_paths, pixel_size, match_angstrom=150.0, tol_deg=15.0):
    """Score hand-drawn paths against the automatic ones; text report lines."""
    lines = []
    auto_curves = [spline_through_points(p["points"], smooth_px=5.0) for p in auto_paths]
    ids = np.concatenate([np.full(len(c[0]), k) for k, c in enumerate(auto_curves)]) \
        if auto_curves else np.empty(0, int)
    samples = np.vstack([c[0] for c in auto_curves]) if auto_curves else np.empty((0, 2))
    tangents = np.vstack([c[1] for c in auto_curves]) if auto_curves else np.empty((0, 2))
    tree = cKDTree(samples) if len(samples) else None
    limit = match_angstrom / pixel_size
    cos_tol = np.cos(np.radians(tol_deg))
    matched_auto = set()
    lines.append(f"{'hand path':<10} {'length nm':>10} {'covered':>8} {'median off A':>13} "
                 f"{'median angle':>13}  auto track(s)")
    for k, ref in enumerate(reference_paths, start=1):
        curve, tangent = spline_through_points(ref["points"], smooth_px=5.0)
        length = float(np.sum(np.hypot(*np.diff(curve, axis=0).T))) * pixel_size / 10
        if tree is None:
            lines.append(f"{k:<10} {length:>10.0f} {0:>7.0%}")
            continue
        distance, nearest = tree.query(curve)
        cosine = np.abs(np.sum(tangents[nearest] * tangent, axis=1))
        covered = (distance <= limit) & (cosine >= cos_tol)
        who = sorted({int(ids[i]) + 1 for i in nearest[covered]})
        matched_auto.update(w - 1 for w in who)
        off = np.median(distance[covered]) * pixel_size if covered.any() else np.nan
        angle = (np.degrees(np.median(np.arccos(np.clip(cosine[covered], -1, 1))))
                 if covered.any() else np.nan)
        lines.append(f"{k:<10} {length:>10.0f} {covered.mean():>7.0%} {off:>13.0f} "
                     f"{angle:>12.1f}°  {', '.join(map(str, who)) or '-'}")
    extra = [k + 1 for k in range(len(auto_paths)) if k not in matched_auto]
    lines.append(f"automatic tracks matching a hand-drawn path: {len(matched_auto)} of "
                 f"{len(auto_paths)}; extra (not drawn by hand -- check the overlay): "
                 f"{', '.join(map(str, extra)) or 'none'}")
    return lines


def overlay(micrograph_binned, z, half, factor, auto_paths, reference_paths, out_path,
            title):
    """Two panels: tracks on the micrograph, and on the coarse z map."""
    import matplotlib  # pylint: disable=import-outside-toplevel
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    from scipy.ndimage import gaussian_filter  # pylint: disable=import-outside-toplevel

    image = gaussian_filter(micrograph_binned, 1.0) - gaussian_filter(micrograph_binned, 40)
    low, high = np.percentile(image, [1, 99])
    aspect = image.shape[1] / image.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(2 * 7 * aspect + 0.6, 7.4))
    zlow, zhigh = np.percentile(z, [50, 99.7])
    axes[0].imshow(image, cmap="gray", vmin=low, vmax=high)
    axes[1].imshow(z, cmap="magma", vmin=zlow, vmax=zhigh,
                   extent=(half - 0.5, half + z.shape[1] - 0.5,
                           half + z.shape[0] - 0.5, half - 0.5))
    axes[1].set_xlim(-0.5, image.shape[1] - 0.5)
    axes[1].set_ylim(image.shape[0] - 0.5, -0.5)
    colours = plt.get_cmap("tab10")
    for ax in axes:
        for ref in reference_paths:
            curve, _ = spline_through_points(ref["points"], smooth_px=5.0)
            ax.plot(curve[:, 1] / factor, curve[:, 0] / factor, color="white", lw=1.0,
                    ls=(0, (4, 3)), alpha=0.9)
        for k, path in enumerate(auto_paths, start=1):
            curve, _ = spline_through_points(path["points"], smooth_px=5.0)
            colour = colours((k - 1) % 10)
            ax.plot(curve[:, 1] / factor, curve[:, 0] / factor, color=colour, lw=1.8)
            mid = curve[len(curve) // 2] / factor
            ax.text(mid[1], mid[0], str(k), color="white", fontsize=9, weight="bold",
                    ha="center", va="center",
                    bbox={"boxstyle": "round,pad=0.15", "fc": colour, "ec": "none"})
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_title("automatic tracks (numbered); dashed white: hand-drawn", fontsize=10)
    axes[1].set_title("coarse z map (scaled MIP) the tracer followed", fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)


# ----------------------------------------------------------------------------- main
# pylint: disable=too-many-locals,too-many-statements
def main() -> None:
    """Coarse search, trace, write paths.json (+ overlay, + comparison)."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--micrograph", type=pathlib.Path, required=True)
    parser.add_argument("--template", type=pathlib.Path, required=True,
                        help="a RING template map (tube axis on z) whose rotational "
                             "average gives the cylinder's radial profile; any pixel "
                             "size, its own is read from the header")
    parser.add_argument("--optics", type=pathlib.Path, required=True,
                        help="YAML with an optics_group: a search config or a PICASSO "
                             "*_optics.yaml")
    parser.add_argument("--outdir", type=pathlib.Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--reference", type=pathlib.Path, default=None,
                        help="hand-drawn .paths.json to score the result against")
    parser.add_argument("--maps", type=pathlib.Path, default=None,
                        help="z/psi maps from svd_trace_maps.py (a Panther-EM SVD search "
                             "of the RING template); replaces the cylinder search, and "
                             "its bin factor replaces --target-pixel")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target-pixel", type=float, default=None,
                        help="A/px to bin to (an integer factor is chosen); default "
                             "about a third of the low-pass, at most 4.5")
    parser.add_argument("--lowpass", type=float, default=None,
                        help="template low-pass, A; default 1.2 x the first CTF zero "
                             "sqrt(lambda * defocus), within 8-15 A")
    parser.add_argument("--cylinder-length-nm", type=float, default=64.0,
                        help="length of the coarse tube template; signal grows with it, "
                             "but a bent tube must stay within ~1/4 of the resolution of "
                             "straight over it (64 nm: sagitta 1.3 nm at R = 0.4 um)")
    parser.add_argument("--psi-step", type=float, default=None,
                        help="default: the template's angular resolution, "
                             "resolution / length (~1.3 deg)")
    parser.add_argument("--theta-step", type=float, default=5.0)
    parser.add_argument("--theta-half-range", type=float, default=10.0)
    parser.add_argument("--threshold-sigma", type=float, default=4.0,
                        help="ridge threshold: median + k robust sd of the z map")
    parser.add_argument("--nms-radius-nm", type=float, default=22.0,
                        help="a ridge point must beat the z map this far either side, "
                             "across the tube. The cylinder also scores where its wall "
                             "meets the tube's OTHER wall, ~22 nm (wall to wall) off "
                             "the axis; two touching tubes are >= 25 nm apart")
    parser.add_argument("--max-turn-deg", type=float, default=1.0,
                        help="most the tracker's heading may turn per step (1 deg per "
                             "2 nm = a 115 nm radius, far tighter than a microtubule "
                             "bends); stops it stepping onto a crossing tube")
    parser.add_argument("--gold-exclusion-nm", type=float, default=40.0,
                        help="ridge points this close to a very dark blob are dropped")
    parser.add_argument("--angle-tolerance", type=float, default=15.0)
    parser.add_argument("--step-nm", type=float, default=2.0)
    parser.add_argument("--lateral-nm", type=float, default=2.0,
                        help="how far beside the predicted point a step may land")
    parser.add_argument("--max-gap-nm", type=float, default=60.0)
    parser.add_argument("--min-length-nm", type=float, default=100.0)
    parser.add_argument("--vertex-spacing-nm", type=float, default=50.0)
    parser.add_argument("--width-nm", type=float, default=52.0,
                        help="band width written for each path")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    workdir = args.outdir / f"{args.prefix}_work"
    workdir.mkdir(exist_ok=True)
    optics = read_optics(args.optics)
    pixel = float(optics["pixel_size"])
    defocus = 0.5 * (float(optics["defocus_u"]) + float(optics["defocus_v"]))
    kv = float(optics["voltage"]) * 1e3
    wavelength = 12.2643 / np.sqrt(kv * (1 + 0.97845e-6 * kv))   # A, relativistic
    first_zero = float(np.sqrt(wavelength * abs(defocus)))
    if args.lowpass is None:
        args.lowpass = float(np.clip(1.2 * first_zero, 8.0, 15.0))
    if args.target_pixel is None:
        args.target_pixel = min(4.5, args.lowpass / 2.8)
    print(f"defocus {defocus / 1e4:.2f} um at {kv / 1e3:.0f} kV: first CTF zero "
          f"~{first_zero:.1f} A -> low-pass {args.lowpass:.1f} A, target "
          f"{args.target_pixel:.2f} A/px")
    factor = max(1, int(round(args.target_pixel / pixel)))
    svd_maps = None
    if args.maps is not None:
        svd_maps = np.load(args.maps)
        factor = int(svd_maps["factor"])
        if abs(float(svd_maps["binned_pixel"]) - pixel * factor) > 1e-4:
            raise ValueError(f"{args.maps} was made at {float(svd_maps['binned_pixel'])} "
                             f"A/px, not {pixel} x {factor}: different optics")
    binned_pixel = pixel * factor
    if args.psi_step is None:
        args.psi_step = float(np.clip(np.degrees(args.lowpass / (args.cylinder_length_nm
                                                                 * 10.0)), 0.5, 3.0))
    print(f"{args.micrograph.name}: {pixel} A/px -> bin {factor} -> {binned_pixel:.3f} A/px; "
          f"{args.cylinder_length_nm:.0f} nm cylinder of {args.template.name}'s radial "
          f"profile, low-passed to {args.lowpass:.0f} A; psi step {args.psi_step:.2f} deg")

    # 1. Bin both.
    with mrcfile.open(str(args.micrograph), permissive=True) as handle:
        micrograph = np.asarray(handle.data, dtype=np.float32)
    binned = fourier_crop(micrograph, factor)
    if svd_maps is not None:
        z = np.asarray(svd_maps["z"], dtype=np.float32)
        psi = np.asarray(svd_maps["psi"], dtype=np.float32)
        half = int(svd_maps["half"])
        expected = (binned.shape[0] - 2 * half + 1, binned.shape[1] - 2 * half + 1)
        if z.shape not in (expected, (expected[0] - 1, expected[1] - 1)):
            raise ValueError(f"{args.maps}: z map {z.shape} is not the valid grid of the "
                             f"binned micrograph {binned.shape} with a {2 * half} box")
        print(f"  SVD ring maps from {args.maps.name}: z {z.shape}, box {2 * half} binned")
    else:
        z, psi, half = None, None, None
    with mrcfile.open(str(args.template), permissive=True) as handle:
        volume = np.asarray(handle.data, dtype=np.float32)
        template_pixel = float(handle.voxel_size.x)
    radius_a, density = radial_profile(volume, template_pixel)
    wall = radius_a[density > 0.5 * density.max()]
    print(f"  {args.template.name}: wall at radius {wall.min() / 10:.1f}-"
          f"{wall.max() / 10:.1f} nm (half maximum of its rotational average)")
    template = cylinder_template(radius_a, density, binned_pixel,
                                 args.cylinder_length_nm, args.lowpass)
    micrograph_path = workdir / "micrograph_binned.mrc"
    template_path = workdir / "template_binned.mrc"
    write_mrc(micrograph_path, binned, binned_pixel)
    write_mrc(template_path, template, binned_pixel)
    print(f"  micrograph {micrograph.shape} -> {binned.shape}, template {volume.shape} "
          f"-> {template.shape}")

    # 2. Coarse search.
    if svd_maps is None:
        optics_binned = dict(optics)
        optics_binned["pixel_size"] = binned_pixel
        z, psi, half = coarse_search(micrograph_path, template_path, optics_binned, args,
                                     workdir)

    # 3. Ridge points, away from gold.
    median = float(np.median(z))
    robust_sd = 1.4826 * float(np.median(np.abs(z - median)))
    threshold = median + args.threshold_sigma * robust_sd
    nm = 10.0 / binned_pixel
    gold = gold_mask(binned, binned_pixel, args.gold_exclusion_nm)
    excluded = gold[half:half + z.shape[0], half:half + z.shape[1]]
    points, values, tangents = ridge_points(z, psi, threshold, args.nms_radius_nm * nm,
                                            excluded)
    print(f"  z map: median {median:.2f}, robust sd {robust_sd:.2f}, threshold "
          f"{threshold:.2f}, max {z.max():.2f}; {excluded.mean():.1%} of it excluded "
          f"around gold; {len(points):,} ridge points")

    # 4-5. Trace and clean up (all lengths in coarse-map pixels).
    params = {"step": args.step_nm * nm, "lateral": args.lateral_nm * nm,
              "max_gap": args.max_gap_nm * nm, "merge_gap": 2 * args.max_gap_nm * nm,
              "min_length": args.min_length_nm * nm, "claim": 3.0 * nm,
              "join_lateral": 6.0 * nm,
              "history": max(2, int(round(20.0 / args.step_nm))),
              "max_turn": float(np.radians(args.max_turn_deg)),
              "cos_tol": float(np.cos(np.radians(args.angle_tolerance)))}
    tracks = trace(points, values, tangents, z.shape, params) if len(points) else []
    tracks = [points[t] for t in tracks]
    n_raw = len(tracks)
    tracks = merge_tracks(tracks, params)
    tracks = drop_duplicates(tracks, params)
    print(f"  {n_raw} track(s) traced -> {len(tracks)} after merging and de-duplicating")

    # 6. Output.
    width_px = args.width_nm * 10.0 / pixel
    auto_paths, scores = [], []
    for track in sorted(tracks, key=lambda t: -track_score(t, z)):
        scores.append(track_score(track, z))
        full = to_full_resolution(track, half, factor)
        curve, _ = spline_through_points(full, smooth_px=2.0 * factor)
        vertices = resample_vertices(curve, args.vertex_spacing_nm * 10.0 / pixel)
        auto_paths.append({"points": vertices, "width_px": width_px, "polarity": "both"})
    paths_out = args.outdir / f"{args.prefix}.paths.json"
    save_paths_json(str(paths_out), str(args.micrograph.resolve()), auto_paths)
    lengths = [float(np.sum(np.hypot(*np.diff(p["points"], axis=0).T)) * pixel / 10)
               for p in auto_paths]
    summary = [f"{k}: {length:.0f} nm, mean z {score:.2f}"
               for k, (length, score) in enumerate(zip(lengths, scores), start=1)]
    print("  tracks: " + "; ".join(summary))

    reference = []
    report = []
    if args.reference is not None:
        _, reference = load_paths_json(str(args.reference))
        report = compare(auto_paths, reference, pixel)
        print("\n".join(["", f"against {args.reference.name}:"] + report))
    overlay(binned, z, half, factor, auto_paths, reference,
            args.outdir / f"{args.prefix}_overlay.png",
            f"{args.prefix}: {len(auto_paths)} automatic track(s), bin {factor} "
            f"({binned_pixel:.2f} A/px), z threshold {threshold:.1f}"
            + (" -- SVD ring maps" if args.maps else " -- cylinder search"))
    meta = {"micrograph": str(args.micrograph.resolve()), "template": str(args.template),
            "optics": str(args.optics), "bin_factor": factor,
            "maps": str(args.maps) if args.maps else None,
            "binned_pixel_size": binned_pixel, "threshold": threshold,
            "z_median": median, "z_robust_sd": robust_sd,
            "parameters": {k: v for k, v in vars(args).items()
                           if not isinstance(v, pathlib.Path)},
            "tracks": [{"track": k, "length_nm": length, "mean_z": score}
                       for k, (length, score) in enumerate(zip(lengths, scores), start=1)],
            "comparison": report}
    (args.outdir / f"{args.prefix}_autotrace.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {paths_out} and {args.prefix}_overlay.png -- open the paths in "
          f"napari_choose_filament_path.py --paths to review and correct")


# NOTE: __main__ guard required for the GPU / multiprocessing setup inside Leopard-EM.
if __name__ == "__main__":
    main()
