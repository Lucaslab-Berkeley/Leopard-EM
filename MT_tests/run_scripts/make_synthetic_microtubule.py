"""Build a synthetic micrograph containing a microtubule of KNOWN protofilament number.

This is the positive control for the protofilament-number competition. On the real data
that competition returned 13, but with an outlier-driven mean and a per-site majority
pointing at 14, so it is unclear whether it is measuring structure or a bias towards
smaller templates. Running the identical competition against ground truth settles it:

* if the competition returns 14 on data built from a 14-protofilament tube, the method
  works and the 13 on real data has to be taken seriously;
* if it returns 13, the method is disqualified and every protofilament-number result
  from it should be discarded.

Construction, in the same geometry as the real data so the same constraint sidecar and
search config apply unchanged:

* the tube is tiled from the 14-PF template's own projection, at twice the dimer repeat
  -- the 2-ring unit spans exactly 2 dimers, so copies at 2 x 83.958 A continue the
  lattice with neither gaps nor double counting;
* it is laid along the same line, at the same psi/theta, as the real microtubule;
* the CTF is the micrograph's own, and the noise is the real crop with its phases
  randomised, so the power spectrum -- and therefore the whitening filter -- is realistic;
* the amplitude is set from the matched-filter relation z ~ ||signal|| / sigma.

``--target-z`` is NOT the z you get. That relation assumes white noise, but the noise
here carries the micrograph's power spectrum and the search whitens before correlating,
so the achieved z is lower. Measured empirically: ``--target-z 10`` produced max z 6.85,
i.e. a factor of 0.69. Ask for ~1.45x the z you actually want, and check the resulting
max z against the real data (216 peaks, max z 9.04 for the 14-PF template) before
trusting any comparison built on it -- a control that is not matched in SNR to the data
it is controlling for is not a control.

Usage:  python run_scripts/make_synthetic_microtubule.py [--n-pf 14] [--target-z 14]
"""

import argparse
import pathlib

import mrcfile
import numpy as np
import pandas as pd
import torch
from torch_ctf import calculate_ctf_2d

from leopard_em.utils.fourier_slice import get_real_space_projections_from_volume

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PIXEL_SIZE = 0.9194
DIMER_REPEAT = 83.958

CROP = MT_ROOT / "Frames" / (
    "2025-05-23_14.39.26_25May23_GMPCPP1_26-94_0005_X-1Y+1-0_sum_DW_cropped_4.mrc"
)
REFERENCE_PEAKS = MT_ROOT / "results_cropped" / "results_6dpu_14pf_2rings_flatB_crop.csv"

# Optics of the real micrograph, from configs/match_tm_crop_2rings.yaml.
#
# calculate_ctf_2d takes defocus and astigmatism in MICROMETRES, not Angstrom -- see
# leopard_em/utils/ctf_utils.py, which multiplies by 1e-4 at the call site. Passing
# Angstrom gives 8212 um of defocus, which turns the microtubule into a coarse grating
# across the whole field; the tell is the fringe spacing, ~150 px instead of the
# sqrt(lambda * defocus) ~ 14 px that this defocus implies.
DEFOCUS_A = 8212.763              # mean of defocus_u and defocus_v
ASTIGMATISM_A = 177.741           # half the difference
OPTICS = {
    "astigmatism_angle": -0.864719,
    "voltage": 300.0,
    "spherical_aberration": 2.7,
    "amplitude_contrast": 0.07,
    "phase_shift": 0.0,
}
# The search damps its template by this before correlating, so the synthetic tube must
# carry it too or the two differ at exactly the frequencies whitening weights most.
CTF_B_FACTOR = 60.0


def microtubule_line() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Centre, direction, psi and theta of the real microtubule, in crop pixels.

    Use ``pos_*_img``, not ``pos_*``. The raw columns are the top-left corner of the
    template box; the image columns add half the box width to give the particle centre.
    Getting this wrong puts the synthetic tube 300 px from the real one, outside the
    spatial constraint mask, and the search then finds nothing at all.
    """
    peaks = pd.read_csv(REFERENCE_PEAKS, index_col=0)
    peaks = peaks[peaks["scaled_mip"] > 7.5]
    pts = peaks[["pos_x_img", "pos_y_img"]].to_numpy(dtype=float)
    centre = pts.mean(axis=0)
    direction = np.linalg.svd(pts - centre)[2][0]
    if direction[1] < 0:
        direction = -direction
    return centre, direction, float(peaks["psi"].median()), float(peaks["theta"].median())


def phase_randomised(image: np.ndarray, seed: int) -> np.ndarray:
    """Noise with the real micrograph's power spectrum but none of its structure."""
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft2(image)
    phases = rng.uniform(0, 2 * np.pi, size=spectrum.shape)
    return np.fft.irfft2(np.abs(spectrum) * np.exp(1j * phases), s=image.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pf", type=int, default=14, help="ground-truth protofilaments")
    parser.add_argument("--target-z", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with mrcfile.open(CROP, permissive=True) as handle:
        crop = np.asarray(handle.data, dtype=np.float32)
    shape = crop.shape
    print(f"crop {shape}")

    template_path = (
        MT_ROOT / "maps" / f"6dpu_{args.n_pf}pf_2rings_flatB_{PIXEL_SIZE}_bscale0.5.mrc"
    )
    with mrcfile.open(template_path, permissive=True) as handle:
        volume = torch.from_numpy(np.asarray(handle.data, dtype=np.float32))
    print(f"template {template_path.name}  {tuple(volume.shape)}")

    centre, direction, psi, theta = microtubule_line()
    print(f"laying the tube along the real one: centre {np.round(centre,1)} "
          f"direction {np.round(direction,4)} psi {psi} theta {theta}")

    projection = get_real_space_projections_from_volume(
        volume,
        torch.tensor(0.0), torch.tensor(theta), torch.tensor(psi),
    ).numpy()
    print(f"projection {projection.shape}")

    # The 2-ring unit spans two dimers, so copies every 2 x 83.958 A continue the
    # lattice exactly. Enough copies to span the whole crop diagonal.
    step_px = 2 * DIMER_REPEAT / PIXEL_SIZE
    reach = int(np.hypot(*shape) / step_px) + 2
    canvas = np.zeros(shape, dtype=np.float64)
    ph, pw = projection.shape
    placed = 0
    for k in range(-reach, reach + 1):
        cx, cy = centre + direction * (k * step_px)
        x0, y0 = int(round(cx - pw / 2)), int(round(cy - ph / 2))
        xs0, ys0 = max(x0, 0), max(y0, 0)
        xs1, ys1 = min(x0 + pw, shape[1]), min(y0 + ph, shape[0])
        if xs1 <= xs0 or ys1 <= ys0:
            continue
        canvas[ys0:ys1, xs0:xs1] += projection[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        placed += 1
    print(f"tiled {placed} copies at {step_px:.1f} px ({2*DIMER_REPEAT:.1f} A) spacing")

    ctf = np.asarray(calculate_ctf_2d(
        defocus=DEFOCUS_A * 1e-4,
        astigmatism=ASTIGMATISM_A * 1e-4,
        **OPTICS, pixel_size=PIXEL_SIZE, image_shape=shape, rfft=True, fftshift=False
    ).squeeze())
    freq_y = np.fft.fftfreq(shape[0], d=PIXEL_SIZE)[:, None]
    freq_x = np.fft.rfftfreq(shape[1], d=PIXEL_SIZE)[None, :]
    envelope = np.exp(-CTF_B_FACTOR * (freq_y**2 + freq_x**2) / 4.0)
    signal = np.fft.irfft2(np.fft.rfft2(canvas) * ctf * envelope, s=shape)

    noise = phase_randomised(crop, args.seed)
    sigma = float(noise.std())

    # Matched filter: z ~ ||signal in one template footprint|| / sigma. Scale to hit the
    # target, which is chosen to match the real data's max z.
    footprint = signal[
        max(int(centre[1]) - ph // 2, 0):int(centre[1]) + ph // 2,
        max(int(centre[0]) - pw // 2, 0):int(centre[0]) + pw // 2,
    ]
    norm = float(np.linalg.norm(footprint))
    scale = args.target_z * sigma / norm if norm > 0 else 1.0
    print(f"noise sigma {sigma:.4f}   signal norm {norm:.4f}   scale {scale:.4g}")

    out = (noise + scale * signal).astype(np.float32)
    out_path = MT_ROOT / "Frames" / f"synthetic_{args.n_pf}pf_crop.mrc"
    with mrcfile.new(out_path, overwrite=True) as handle:
        handle.set_data(out)
        handle.voxel_size = PIXEL_SIZE
    print(f"wrote {out_path}")
    print(f"  image sd {out.std():.4f}, signal contributes "
          f"{100*np.abs(scale*signal).max()/out.std():.1f}% of sd at its peak")


if __name__ == "__main__":
    main()
