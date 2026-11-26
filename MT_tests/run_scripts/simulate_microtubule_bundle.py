"""Simulate a micrograph with several microtubules that OVERLAP, with full ground truth.

Built to test the overlap architecture on the layout that exposed it: in situ
micrograph ``set1_823`` has three parallel tubes 411 A and 463 A apart and a fourth
crossing them at about 29 degrees. Everything is matched in ANGSTROMS, not pixels, so it
runs on the existing 0.9194 A/px templates:

    tube 1   bent arc, psi 42 at mid-arc          (the bundle's reference)
    tube 2   parallel to 1, +411 A along its normal
    tube 3   parallel to 1, -463 A along its normal
    tube 4   crossing arc, psi 13 at mid-arc, through the middle of the bundle

Each tube has its own roll (phi) and polarity, so the search cannot get away with one
orientation per region. Parallel tubes are true CONCENTRIC arcs -- each offset curve is
tiled along its OWN arc length, since stepping an offset curve by the reference tube's
arc length would stretch its lattice by the ratio of radii (about 5% here).

The drawn paths are the true curves plus a smooth wander of ``--draw-error-angstrom``,
with an independent phase per tube: paths drawn one at a time do not err in step, and
lockstep errors would cancel between neighbours and hide the relative error that
mislabels them. The band width is 515 A, the same physical width as 823's 490 px.

Writes ``Frames/synthetic_bundle_<tag>.mrc`` with ``.truth.json`` and ``.paths.json``.

Usage:
    python run_scripts/simulate_microtubule_bundle.py
"""

import argparse
import json
import math
import pathlib
import sys

import mrcfile
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from simulate_curved_microtubule import (  # noqa: E402
    MT_ROOT,
    NOISE_SOURCE,
    PIXEL_SIZE,
    TEMPLATE_STEMS,
    apply_ctf,
    arc_curve,
    orient_curve,
    phase_randomised,
    psi_from_tangent_angle,
)

from leopard_em.analysis.filament_lattice import (  # noqa: E402
    rise_from_template_autocorrelation,
)
from leopard_em.utils.fourier_slice import (  # noqa: E402
    get_real_space_projections_from_volume,
)

BAND_ANGSTROM = 515.0


def offset_along_normal(curve_yx: np.ndarray, tangent_rad: np.ndarray,
                        distance_px: float) -> np.ndarray:
    """Displace a curve along its own normal: a concentric neighbour of an arc."""
    tangent = np.column_stack([np.sin(tangent_rad), np.cos(tangent_rad)])  # (y, x)
    normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    return curve_yx + distance_px * normal


def own_arc_length(curve_yx: np.ndarray) -> np.ndarray:
    """Arc length along this curve, zero at its middle sample."""
    steps = np.hypot(*np.diff(curve_yx, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    return arc - arc[len(arc) // 2]


def tile_tube(volume, shape, curve_yx, tangent_rad, step_px, phi_deg, flip):
    """Paste 2-ring blocks along one tube at its local tangent; returns canvas, blocks.

    ``flip`` reverses the polarity (psi + 180). Blocks step by exactly four monomers
    of the tube's OWN arc length, so the lattice runs continuously along it.
    """
    arc = own_arc_length(curve_yx)
    canvas = np.zeros(shape, dtype=np.float64)
    blocks = []
    n_blocks = int(min(-arc[0], arc[-1]) / step_px)
    for index in range(-n_blocks, n_blocks + 1):
        s = index * step_px
        y = float(np.interp(s, arc, curve_yx[:, 0]))
        x = float(np.interp(s, arc, curve_yx[:, 1]))
        angle = float(np.interp(s, arc, tangent_rad)) + (math.pi if flip else 0.0)
        psi = float(psi_from_tangent_angle(angle))
        projection = get_real_space_projections_from_volume(
            volume, torch.tensor(phi_deg), torch.tensor(90.0), torch.tensor(psi)
        ).numpy()
        height, width = projection.shape
        y0, x0 = int(round(y - height / 2)), int(round(x - width / 2))
        ys0, xs0 = max(y0, 0), max(x0, 0)
        ys1, xs1 = min(y0 + height, shape[0]), min(x0 + width, shape[1])
        if ys1 <= ys0 or xs1 <= xs0:
            continue
        canvas[ys0:ys1, xs0:xs1] += projection[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        blocks.append({"arc_px": s, "y": y, "x": x, "psi_deg": psi})
    return canvas, blocks


# --domains lattice: a KNOWN lattice pattern along each tube, to calibrate how short a
# segment can still tell compacted from expanded. C = 6DPV lattice, E = 6DPU lattice,
# same 6DPU atoms -- exactly the pair the Stage 6 competition compares.
LATTICE_STEMS = {"C": TEMPLATE_STEMS[("6dpv", 13)], "E": "6dpu_13pf_2rings_flatB"}
DOMAIN_ANGSTROM = 600.0  # tube 4 alternates C/E every 60 nm


def lattice_at(label: int, s_px: float) -> str:
    """Which lattice tube ``label`` has at arc position ``s_px`` (0 = its middle)."""
    if label == 1:
        return "C" if s_px < 0 else "E"      # one transition, mid-tube
    if label == 2:
        return "E"                           # expanded throughout
    if label == 3:
        return "C"                           # compacted throughout
    period = DOMAIN_ANGSTROM / PIXEL_SIZE    # tube 4: alternating 60 nm domains
    return "C" if int(np.floor(s_px / period)) % 2 == 0 else "E"


def tile_tube_domains(volumes, monomers, label, shape, curve_yx, tangent_rad, phi_deg,
                      flip):
    """Like ``tile_tube``, but each block takes the lattice its arc position calls for.

    Blocks step by four monomers of THEIR OWN lattice, so each domain's lattice is
    continuous; a junction mismatches by at most 2 x (41.9 - 40.9) A.
    """
    arc = own_arc_length(curve_yx)
    canvas = np.zeros(shape, dtype=np.float64)
    blocks = []
    limit = min(-arc[0], arc[-1])
    s = -limit + 2.0 * monomers["C"]
    while s < limit - 2.0 * monomers["C"]:
        lattice = lattice_at(label, s)
        y = float(np.interp(s, arc, curve_yx[:, 0]))
        x = float(np.interp(s, arc, curve_yx[:, 1]))
        angle = float(np.interp(s, arc, tangent_rad)) + (math.pi if flip else 0.0)
        psi = float(psi_from_tangent_angle(angle))
        projection = get_real_space_projections_from_volume(
            volumes[lattice], torch.tensor(phi_deg), torch.tensor(90.0),
            torch.tensor(psi)).numpy()
        height, width = projection.shape
        y0, x0 = int(round(y - height / 2)), int(round(x - width / 2))
        ys0, xs0 = max(y0, 0), max(x0, 0)
        ys1, xs1 = min(y0 + height, shape[0]), min(x0 + width, shape[1])
        if ys1 > ys0 and xs1 > xs0:
            canvas[ys0:ys1, xs0:xs1] += projection[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
            blocks.append({"arc_px": s, "y": y, "x": x, "psi_deg": psi,
                           "lattice": lattice})
        s += 4.0 * monomers[lattice]
    return canvas, blocks


def drawn_path(curve_yx, tangent_rad, error_px, phase, shape, periods, n_clicks=40):
    """The truth plus a smooth wander, sampled like hand clicks, inside the frame."""
    s = np.linspace(0.0, 1.0, len(curve_yx))
    wander = offset_along_normal(curve_yx, tangent_rad, 0.0) + (
        error_px * np.sin(2 * np.pi * periods * s + phase)
    )[:, None] * np.column_stack([np.cos(tangent_rad), -np.sin(tangent_rad)])
    inside = (
        (wander[:, 0] > 0) & (wander[:, 0] < shape[0] - 1)
        & (wander[:, 1] > 0) & (wander[:, 1] < shape[1] - 1)
    )
    visible = wander[inside]
    return visible[np.linspace(0, len(visible) - 1, n_clicks).astype(int)]


# pylint: disable=too-many-locals,too-many-statements
def main() -> None:
    """Build the bundle-plus-crossing micrograph and its ground truth."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=3072,
                        help="3072 puts all three crossings inside the searched area")
    parser.add_argument("--spacings-angstrom", type=float, nargs=2,
                        default=[411.0, 463.0])
    parser.add_argument("--bundle-psi", type=float, default=42.0)
    parser.add_argument("--crossing-psi", type=float, default=13.0)
    parser.add_argument("--draw-error-angstrom", type=float, default=50.0)
    parser.add_argument("--draw-periods", type=float, default=1.0,
                        help="wander periods along the tube; 1 tilts the drawn "
                             "tangent ~4 deg, about the sd-3 deg measured between "
                             "the GDP drawing and its data")
    parser.add_argument("--target-z", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="823like")
    parser.add_argument("--domains", default="none", choices=["none", "lattice"],
                        help="'lattice': known compacted/expanded domains per tube "
                             "(see lattice_at)")
    parser.add_argument("--rolls", type=float, nargs=4, default=[0.0, 47.0, 113.0, 200.0],
                        help="roll per tube; 0 happens to sit on the search grid and "
                             "scores ~1 z high, so avoid it when tubes are compared")
    parser.add_argument("--protofilaments", type=int, nargs=4, default=[13, 13, 13, 13],
                        help="protofilament number per tube (compacted lattice, the "
                             "compete_sites_curved rival templates): e.g. 13 13 14 14 "
                             "for a micrograph whose tubes differ in N")
    args = parser.parse_args()
    if args.domains == "lattice" and set(args.protofilaments) != {13}:
        parser.error("--domains lattice is 13-PF only")

    shape = (args.size, args.size)
    stem = TEMPLATE_STEMS[("6dpv", 13)]
    volumes_by_n, steps_by_n = {}, {}
    for n_pf in sorted(set(args.protofilaments)):
        # the same stems as compete_sites_curved.template_stem("compacted", n): each tube
        # is exactly one of the PF competition's rivals
        n_stem = f"6dpu_atoms_6dpv_lattice_{n_pf}pf_2rings_flatB"
        template_path = MT_ROOT / "maps" / f"{n_stem}_{PIXEL_SIZE}_bscale0.5.mrc"
        with mrcfile.open(template_path, permissive=True) as handle:
            volumes_by_n[n_pf] = torch.from_numpy(np.asarray(handle.data, dtype=np.float32))
        monomer = rise_from_template_autocorrelation(
            np.asarray(volumes_by_n[n_pf]), approximate_rise_px=41.98 / PIXEL_SIZE,
            search_fraction=0.1)
        steps_by_n[n_pf] = 4.0 * monomer
        print(f"template {template_path.name}, monomer {monomer * PIXEL_SIZE:.3f} Å")
    n_first = args.protofilaments[0]
    volume, step_px = volumes_by_n[n_first], steps_by_n[n_first]
    monomer_px = step_px / 4.0

    rng = np.random.default_rng(args.seed)
    centre = np.array([args.size / 2.0, args.size / 2.0])
    arc_len = args.size * 1.6

    base, base_tan, base_radius = arc_curve(arc_len, 70.0, 3001)
    base, base_tan = orient_curve(base, base_tan, args.bundle_psi)
    cross, cross_tan, cross_radius = arc_curve(arc_len, -50.0, 3001)
    cross, cross_tan = orient_curve(cross, cross_tan, args.crossing_psi)

    spacing_px = [d / PIXEL_SIZE for d in args.spacings_angstrom]
    tubes = {
        1: (base + centre, base_tan),
        2: (offset_along_normal(base, base_tan, spacing_px[0]) + centre, base_tan),
        3: (offset_along_normal(base, base_tan, -spacing_px[1]) + centre, base_tan),
        4: (cross + centre, cross_tan),
    }
    rolls = dict(zip((1, 2, 3, 4), args.rolls))
    n_of = dict(zip((1, 2, 3, 4), args.protofilaments))
    if args.domains == "lattice":
        volumes, monomers = {}, {}
        for key, lattice_stem in LATTICE_STEMS.items():
            path = MT_ROOT / "maps" / f"{lattice_stem}_{PIXEL_SIZE}_bscale0.5.mrc"
            with mrcfile.open(path, permissive=True) as handle:
                volumes[key] = torch.from_numpy(np.asarray(handle.data, dtype=np.float32))
            monomers[key] = rise_from_template_autocorrelation(
                np.asarray(volumes[key]), approximate_rise_px=41.4 / PIXEL_SIZE,
                search_fraction=0.1)
            print(f"  lattice {key}: {path.name}, monomer "
                  f"{monomers[key] * PIXEL_SIZE:.3f} Å")
    flips = {1: False, 2: True, 3: False, 4: True}

    signal = np.zeros(shape, dtype=np.float64)
    truth_tubes = []
    for label, (curve, tangent) in tubes.items():
        if args.domains == "lattice":
            canvas, blocks = tile_tube_domains(volumes, monomers, label, shape, curve,
                                               tangent, rolls[label], flips[label])
        else:
            canvas, blocks = tile_tube(volumes_by_n[n_of[label]], shape, curve, tangent,
                                       steps_by_n[n_of[label]], rolls[label], flips[label])
        signal += canvas
        truth_tubes.append({
            "label": label,
            "protofilaments": n_of[label],
            "phi_deg": rolls[label],
            "polarity_flipped": flips[label],
            "curve_yx": curve[::10].tolist(),
            "blocks": blocks,
        })
        print(f"tube {label}: {n_of[label]}-PF, {len(blocks)} blocks, roll {rolls[label]:.0f}°, "
              f"{'flipped' if flips[label] else 'forward'}, psi "
              f"{blocks[len(blocks) // 2]['psi_deg']:.1f}° mid")

    signal = apply_ctf(signal, shape)
    with mrcfile.open(NOISE_SOURCE, permissive=True) as handle:
        source = np.asarray(handle.data, dtype=np.float32)
    noise = phase_randomised(source[:args.size, :args.size], args.seed)
    sigma = float(noise.std())
    mid = truth_tubes[0]["blocks"][len(truth_tubes[0]["blocks"]) // 4]
    half = int(round(2.0 * monomer_px))
    window = signal[int(mid["y"]) - half:int(mid["y"]) + half,
                    int(mid["x"]) - half:int(mid["x"]) + half]
    scale = args.target_z * sigma / float(np.linalg.norm(window))

    out = MT_ROOT / "Frames" / f"synthetic_bundle_{args.tag}.mrc"
    with mrcfile.new(str(out), overwrite=True) as handle:
        handle.set_data((noise + scale * signal).astype(np.float32))
        handle.voxel_size = PIXEL_SIZE
    print(f"wrote {out}")

    width_px = BAND_ANGSTROM / PIXEL_SIZE
    error_px = args.draw_error_angstrom / PIXEL_SIZE
    paths = []
    for label, (curve, tangent) in tubes.items():
        clicks = drawn_path(curve, tangent, error_px, rng.uniform(0, 2 * np.pi), shape,
                            args.draw_periods)
        paths.append({"points": clicks.tolist(), "width_px": width_px,
                      "polarity": "both"})
    with open(out.with_suffix(".paths.json"), "w", encoding="utf-8") as handle:
        json.dump({"micrograph_path": str(out), "paths": paths}, handle, indent=2)

    with open(out.with_suffix(".truth.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "micrograph": str(out),
            "template": stem,
            "pixel_size_angstrom": PIXEL_SIZE,
            "monomer_rise_angstrom": float(monomer_px * PIXEL_SIZE),
            "spacings_angstrom": args.spacings_angstrom,
            "bundle_psi_deg": args.bundle_psi,
            "crossing_psi_deg": args.crossing_psi,
            "radius_of_curvature_px": {"bundle": base_radius, "crossing": cross_radius},
            "draw_error_angstrom": args.draw_error_angstrom,
            "draw_periods": args.draw_periods,
            "band_angstrom": BAND_ANGSTROM,
            "target_z": args.target_z,
            "domains": args.domains,
            "rolls_deg": args.rolls,
            "protofilaments": args.protofilaments,
            "tubes": truth_tubes,
        }, handle, indent=2)
    print(f"wrote {out.with_suffix('.paths.json').name}, "
          f"{out.with_suffix('.truth.json').name}")


if __name__ == "__main__":
    main()
