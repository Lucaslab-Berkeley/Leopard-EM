"""Simulate a BENT microtubule traversing a square micrograph, with full ground truth.

Every straight-filament result in ``STATUS.md`` rests on a single real tube, so there is
nothing to check a curved analysis against. This builds one where the answer is known:
protofilament number, lattice (expanded or compacted), monomer rise, polarity, seam
protofilament and the curve itself are all inputs, so each of the four readouts has a
truth to be scored against.

The tube is laid down by tiling the 2-ring template along a circular arc, rotating each
copy to the LOCAL tangent. A 2-ring block spans two dimers, so copies at that spacing
continue the lattice exactly -- the same trick ``make_synthetic_microtubule.py`` uses for
a straight tube, with psi now a function of arc length rather than a constant. Blocks are
short (~183 px) against the bend radius, so the within-block straightness costs a sagitta
of well under a pixel.

**What this does NOT contain: bend strain.** Each block is an internally unstrained
straight tube, so the real cos(phi) modulation of the rise -- u0[1 + R*kappa*cos(phi-phi0)]
-- is absent by construction. That is a useful ground truth in its own right (a strain
measurement run on this must return ZERO, so it tests that curvature is not *invented*),
but recovering a known non-zero strain needs a bent ATOMIC model, not tiled rigid blocks.

It also writes a ``.paths.json`` for the curve, in the format
``programs/constrained_search/napari_choose_filament_path.py`` saves. Because the curve
is known exactly, the manual drawing step can be skipped entirely: feed that file to
``make_filament_constraint.py`` and the constrained search is ready to run.

Usage:
    python run_scripts/simulate_curved_microtubule.py --n-pf 13 --lattice 6dpv
    python run_scripts/simulate_curved_microtubule.py --sagitta-px 0 --tag straight
"""

import argparse
import json
import math
import pathlib

import mrcfile
import numpy as np
import torch
from torch_ctf import calculate_ctf_2d
from torch_fourier_slice import project_3d_to_2d  # noqa: F401  (import check)

from leopard_em.analysis.filament_lattice import rise_from_template_autocorrelation
from leopard_em.utils.fourier_slice import get_real_space_projections_from_volume

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PIXEL_SIZE = 0.9194

# Optics and the CTF B-factor of the real micrograph, so the synthetic tube is damped
# exactly as the search damps its template. See make_synthetic_microtubule.py for the
# units trap: calculate_ctf_2d takes MICROMETRES.
DEFOCUS_A = 8212.763
ASTIGMATISM_A = 177.741
CTF_B_FACTOR = 60.0
OPTICS = {
    "astigmatism_angle": -0.864719,
    "voltage": 300.0,
    "spherical_aberration": 2.7,
    "amplitude_contrast": 0.07,
    "phase_shift": 0.0,
}

NOISE_SOURCE = MT_ROOT / "Frames" / (
    "2025-05-23_14.39.26_25May23_GMPCPP1_26-94_0005_X-1Y+1-0_sum_DW.mrc"
)  # one tube and mostly ice, so its power spectrum is close to plain background

TEMPLATE_STEMS = {
    ("6dpu", 13): "6dpu_13pf_2rings_flatB",
    ("6dpv", 13): "6dpu_atoms_6dpv_lattice_13pf_2rings_flatB",
    ("6dpu", 14): "6dpu_atoms_6dpu_lattice_2rings_flatB",
    ("6dpv", 14): "6dpu_atoms_6dpv_lattice_2rings_flatB",
}


def arc_curve(
    arc_length_px: float, sagitta_px: float, n_samples: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """A circular arc of given length and bulge, as points and tangent angles.

    ``sagitta_px`` is the perpendicular bulge at mid-arc, which is the intuitive handle:
    it is what you can see. The radius follows as ``L^2 / (8 * sagitta)``. A sagitta of
    zero gives a straight line, which is the control case.

    Returns points in ``(y, x)`` centred on the origin, the tangent angle at each point
    in radians, and the radius of curvature in pixels (``inf`` when straight).
    """
    s = np.linspace(-arc_length_px / 2.0, arc_length_px / 2.0, n_samples)
    if abs(sagitta_px) < 1e-9:
        return np.column_stack([np.zeros_like(s), s]), np.zeros_like(s), math.inf

    radius = arc_length_px**2 / (8.0 * abs(sagitta_px))
    sign = math.copysign(1.0, sagitta_px)
    angle = s / radius
    # Integrate the unit tangent (cos, sin) to get the curve; x runs along the chord.
    x = radius * np.sin(angle)
    y = sign * radius * (1.0 - np.cos(angle))
    return np.column_stack([y, x]), sign * angle, radius


def orient_curve(
    curve_yx: np.ndarray, tangent_rad: np.ndarray, orient_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the arc so its mid-tangent points along ``orient_deg`` in psi.

    A tube lying near psi = 0 makes its sweep straddle the 0/360 wrap, which is a real
    case but an unhelpful one to debug alongside everything else. Orienting the arc
    keeps the whole sweep in one branch.
    """
    shift = math.radians(orient_deg)
    rotation = np.array(
        [[math.cos(shift), -math.sin(shift)], [math.sin(shift), math.cos(shift)]]
    )
    return curve_yx @ rotation.T, tangent_rad - shift


def psi_from_tangent_angle(angle_rad: np.ndarray) -> np.ndarray:
    """In-plane ``psi`` in degrees for a tangent, image convention ``atan2(-dy, dx)``.

    The curve is stored as ``(y, x)`` with tangent ``(sin a, cos a)``, so this is
    ``-a`` in degrees -- which keeps the same convention as
    ``filament_psi_from_image_line`` and the drawing tool.
    """
    return np.degrees(-np.asarray(angle_rad, dtype=np.float64)) % 360.0


def tile_along_curve(
    volume: torch.Tensor,
    shape: tuple[int, int],
    centre_yx: np.ndarray,
    curve_yx: np.ndarray,
    tangent_rad: np.ndarray,
    arc_positions: np.ndarray,
    step_px: float,
    phi_deg: float,
) -> tuple[np.ndarray, list[dict]]:
    """Paste template projections along the curve, each at its own local tangent.

    One projection per block, because psi changes from block to block. The template box
    is far larger than a block, so the boxes overlap heavily -- but everything outside
    the tube is zero, so the sum is still exactly one continuous tube.
    """
    canvas = np.zeros(shape, dtype=np.float64)
    blocks: list[dict] = []
    n_blocks = int(arc_positions[-1] / step_px)

    for index in range(-n_blocks, n_blocks + 1):
        s = index * step_px
        if s < arc_positions[0] or s > arc_positions[-1]:
            continue
        y = float(np.interp(s, arc_positions, curve_yx[:, 0])) + centre_yx[0]
        x = float(np.interp(s, arc_positions, curve_yx[:, 1])) + centre_yx[1]
        angle = float(np.interp(s, arc_positions, tangent_rad))
        psi = float(psi_from_tangent_angle(angle))

        projection = get_real_space_projections_from_volume(
            volume,
            torch.tensor(phi_deg),
            torch.tensor(90.0),
            torch.tensor(psi),
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


def phase_randomised(image: np.ndarray, seed: int) -> np.ndarray:
    """Noise carrying the real micrograph's power spectrum but none of its structure."""
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft2(image)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape)
    return np.fft.irfft2(np.abs(spectrum) * np.exp(1j * phases), s=image.shape)


def apply_ctf(canvas: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """CTF and the search's own B-factor envelope, in one pass."""
    ctf = np.asarray(
        calculate_ctf_2d(
            defocus=DEFOCUS_A * 1e-4,
            astigmatism=ASTIGMATISM_A * 1e-4,
            **OPTICS,
            pixel_size=PIXEL_SIZE,
            image_shape=shape,
            rfft=True,
            fftshift=False,
        ).squeeze()
    )
    freq_y = np.fft.fftfreq(shape[0], d=PIXEL_SIZE)[:, None]
    freq_x = np.fft.rfftfreq(shape[1], d=PIXEL_SIZE)[None, :]
    envelope = np.exp(-CTF_B_FACTOR * (freq_y**2 + freq_x**2) / 4.0)
    return np.fft.irfft2(np.fft.rfft2(canvas) * ctf * envelope, s=shape)


# pylint: disable=too-many-locals,too-many-statements
def main() -> None:
    """Build one curved synthetic micrograph and its ground truth."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pf", type=int, default=13, choices=[13, 14])
    parser.add_argument("--lattice", default="6dpv", choices=["6dpu", "6dpv"],
                        help="6dpv = GDP/compacted, 6dpu = GMPCPP/expanded")
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--sagitta-px", type=float, default=100.0,
                        help="bulge at mid-arc; 0 gives a straight control")
    parser.add_argument("--orient-deg", type=float, default=300.0,
                        help="psi of the tube at mid-arc; 300 keeps the whole "
                             "sweep clear of the 0/360 wrap")
    parser.add_argument("--phi-deg", type=float, default=0.0,
                        help="roll of the tube; fixes where the seam projects")
    parser.add_argument("--target-z", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    shape = (args.size, args.size)
    stem = TEMPLATE_STEMS[(args.lattice, args.n_pf)]
    template_path = MT_ROOT / "maps" / f"{stem}_{PIXEL_SIZE}_bscale0.5.mrc"
    with mrcfile.open(template_path, permissive=True) as handle:
        volume = torch.from_numpy(np.asarray(handle.data, dtype=np.float32))
    print(f"template {template_path.name}  {tuple(volume.shape)}")

    # The template's own repeat, measured rather than assumed: a 2-ring block spans two
    # dimers, so the tiling step is four monomers.
    monomer_px = rise_from_template_autocorrelation(
        np.asarray(volume), approximate_rise_px=41.98 / PIXEL_SIZE, search_fraction=0.1
    )
    step_px = 4.0 * monomer_px
    print(f"monomer rise {monomer_px * PIXEL_SIZE:.3f} Å, "
          f"tiling step {step_px * PIXEL_SIZE:.3f} Å ({step_px:.1f} px)")

    # Run the tube right across the frame, with enough overhang that it traverses.
    arc_length = args.size * 1.25
    curve_yx, tangent_rad, radius = arc_curve(arc_length, args.sagitta_px, 2001)
    curve_yx, tangent_rad = orient_curve(curve_yx, tangent_rad, args.orient_deg)
    arc_positions = np.linspace(-arc_length / 2.0, arc_length / 2.0, 2001)
    centre = np.array([args.size / 2.0 - curve_yx[:, 0].mean(), args.size / 2.0])
    turn_deg = float(np.degrees(tangent_rad[-1] - tangent_rad[0]))
    print(f"arc {arc_length:.0f} px, sagitta {args.sagitta_px:.0f} px, "
          f"radius {radius:.0f} px ({radius * PIXEL_SIZE / 1e4:.2f} µm), "
          f"total turn {turn_deg:.1f}°")

    canvas, blocks = tile_along_curve(
        volume, shape, centre, curve_yx, tangent_rad, arc_positions,
        step_px, args.phi_deg,
    )
    sweep = np.unwrap(np.radians([b["psi_deg"] for b in blocks]))
    print(f"tiled {len(blocks)} blocks, psi {blocks[0]['psi_deg']:.2f}° → "
          f"{blocks[-1]['psi_deg']:.2f}° "
          f"(sweeps {abs(np.degrees(sweep[-1] - sweep[0])):.1f}°)")

    signal = apply_ctf(canvas, shape)

    with mrcfile.open(NOISE_SOURCE, permissive=True) as handle:
        source = np.asarray(handle.data, dtype=np.float32)
    patch = source[:args.size, :args.size]
    noise = phase_randomised(patch, args.seed)
    sigma = float(noise.std())

    # Scale so a matched filter over one template footprint reaches the target z.
    mid = blocks[len(blocks) // 2]
    half = int(round(2.0 * monomer_px))
    window = signal[
        max(int(mid["y"]) - half, 0):int(mid["y"]) + half,
        max(int(mid["x"]) - half, 0):int(mid["x"]) + half,
    ]
    norm = float(np.linalg.norm(window))
    scale = args.target_z * sigma / norm if norm > 0 else 1.0
    print(f"noise sigma {sigma:.4f}, signal norm {norm:.4f}, scale {scale:.4g}")

    tag = args.tag or f"{args.n_pf}pf_{args.lattice}_sag{int(args.sagitta_px)}"
    out_path = MT_ROOT / "Frames" / f"synthetic_curved_{tag}.mrc"
    with mrcfile.new(str(out_path), overwrite=True) as handle:
        handle.set_data((noise + scale * signal).astype(np.float32))
    print(f"wrote {out_path}")

    # Ground truth, and the curve in the drawing tool's own format so the manual
    # segmentation step can be skipped entirely.
    inside = (
        (curve_yx[:, 0] + centre[0] > 0) & (curve_yx[:, 0] + centre[0] < args.size)
        & (curve_yx[:, 1] + centre[1] > 0) & (curve_yx[:, 1] + centre[1] < args.size)
    )
    control_points = np.column_stack(
        [curve_yx[inside, 0] + centre[0], curve_yx[inside, 1] + centre[1]]
    )[:: max(int(inside.sum() // 12), 1)]

    paths_path = out_path.with_suffix(".paths.json")
    with open(paths_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "micrograph_path": str(out_path),
                "paths": [
                    {
                        "points": control_points.tolist(),
                        "width_px": 340.0,
                        "polarity": "both",
                    }
                ],
            },
            handle,
            indent=2,
        )

    truth_path = out_path.with_suffix(".truth.json")
    with open(truth_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "micrograph": str(out_path),
                "template": stem,
                "n_protofilaments": args.n_pf,
                "lattice": args.lattice,
                "monomer_rise_angstrom": float(monomer_px * PIXEL_SIZE),
                "dimer_repeat_angstrom": float(2.0 * monomer_px * PIXEL_SIZE),
                "pixel_size_angstrom": PIXEL_SIZE,
                "phi_deg": args.phi_deg,
                "orient_deg": args.orient_deg,
                "theta_deg": 90.0,
                "psi_deg_start": blocks[0]["psi_deg"],
                "psi_deg_end": blocks[-1]["psi_deg"],
                "radius_of_curvature_px": None if math.isinf(radius) else float(radius),
                "sagitta_px": args.sagitta_px,
                "total_turn_deg": turn_deg,
                "arc_length_px": arc_length,
                "curve_centre_yx": centre.tolist(),
                "bend_strain_present": False,
                "target_z": args.target_z,
                "blocks": blocks,
            },
            handle,
            indent=2,
        )
    print(f"wrote {paths_path}")
    print(f"wrote {truth_path}")


if __name__ == "__main__":
    main()
