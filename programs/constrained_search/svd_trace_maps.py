"""SVD (Panther-EM) z and psi maps for the filament tracer: ``auto_trace_filaments.py --maps``.

The tracer's own coarse search matches a smooth 64 nm cylinder at 8-15 A. That responds
to any straight edge and to gold halos, which is most of what goes wrong with it. This
runs a real 2DTM search instead -- the RING template with its protofilaments, every
roll (phi), theta within ``--theta-half-range`` of in-plane, every psi -- at a few A per
pixel, through Panther-EM's SVD search, so it costs about as much as the cylinder did.
The z map is per-pixel normalised over all searched orientations (a true 2DTM z), and
the ring is axis-centred, so its ridge IS the tube axis.

Runs in the Panther-EM environment, not Leopard-EM's::

    source MT_tests/run_scripts/panther_env.sh   # from the Leopard-EM root
    CUDA_VISIBLE_DEVICES=1 python programs/constrained_search/svd_trace_maps.py \\
        --micrograph MT_tests/Frames/GDP_curved_mgraph.mrc \\
        --template MT_tests/maps/6dpu_atoms_6dpv_lattice_13pf_2rings_flatB_0.9194_bscale0.5.mrc \\
        --optics MT_tests/results_ctf/GDP_curved_mgraph_optics.yaml \\
        --outdir MT_tests/results_autotrace_svd --prefix gdp_curved

then trace in Leopard-EM's environment with ``auto_trace_filaments.py ... --maps
<outdir>/<prefix>_svd_maps.npz``.

Steps (all in ``<outdir>/<prefix>_svd_work/``)
---------------------------------------------
1. Bin the micrograph with the tracer's own ``fourier_crop`` (same factor, same grid).
2. Crop the template box to its mass (``--mass-fraction`` of the density within the
   radius, x ``--box-margin``) and Fourier-crop it by the same factor. Panther-EM rotates
   about the box centre, so the template must be centred: its centre-of-mass offset is
   checked and reported.
3. Panther-EM whitening (this micrograph), decomposition over HEALPix ``--nside``
   orientations restricted to the theta band (and ``--phi-range``), calibration tables,
   and one dense search stage at rank ``--rank K M``.
4. Write ``<prefix>_svd_maps.npz``: z, psi, theta, phi, defocus on the binned valid
   grid (top-left frame), plus ``half`` (binned box // 2), ``factor``, ``binned_pixel``.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import pathlib
import shutil
import signal
import sys
import time

import h5py
import mrcfile
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from auto_trace_filaments import fourier_crop, read_optics  # noqa: E402

# Disk space the SVD tools leave untouched (GB). The disk filled once (2026-11-12): every
# run keeps a work folder with the template's decomposition (~2 GB per template at the
# default rank, 4-7 GB at k 256 or nside 64), and folders of failed runs piled up.
MIN_FREE_GB = float(os.environ.get("SVD_MIN_FREE_GB", "30"))


def require_free_space(path: pathlib.Path, need_gb: float, what: str) -> float:
    """Stop before ``what`` writes ~``need_gb`` GB under ``path`` if that would leave less
    than ``SVD_MIN_FREE_GB`` (default 30) free on that filesystem. Returns the free GB."""
    probe = pathlib.Path(path).resolve()
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free / 1e9
    if free - need_gb < MIN_FREE_GB:
        raise SystemExit(
            f"{what}: {free:.1f} GB free on {probe}; it needs ~{need_gb:.0f} GB and leaves "
            f"{MIN_FREE_GB:.0f} GB spare (SVD_MIN_FREE_GB). Free space first -- old "
            "*_svd_work folders are the usual culprit (MT_tests/run_scripts/svd_work_dirs.py "
            "lists them) -- or lower SVD_MIN_FREE_GB.")
    return free


def remove_on_exit(path: pathlib.Path, keep: bool) -> None:
    """Delete the work folder ``path`` when the process ends -- normally, on an error or on
    SIGTERM -- unless ``keep``. (Not on SIGKILL: ``svd_work_dirs.py`` finds those.)"""
    if keep:
        return
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))


def write_mrc(path: pathlib.Path, data: np.ndarray, pixel: float) -> None:
    with mrcfile.new(str(path), overwrite=True) as handle:
        handle.set_data(np.ascontiguousarray(data, dtype=np.float32))
        handle.voxel_size = pixel


def mass_radius(volume: torch.Tensor, fraction: float) -> tuple[float, list[float]]:
    """Radius about the box centre holding ``fraction`` of the positive density, and the
    centre-of-mass offset (z, y, x) in voxels."""
    n = volume.shape[0]
    c = n // 2
    w = volume.clamp_min(0).double()
    total = w.sum()
    ax = torch.arange(n, device=volume.device, dtype=torch.float64) - c
    com = [float((w.sum(dim=tuple(d for d in range(3) if d != i)) * ax).sum() / total)
           for i in range(3)]
    r = torch.sqrt(ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2)
    hist = torch.bincount(r.round().long().ravel(), weights=w.ravel())
    cum = torch.cumsum(hist, 0) / total
    return float(torch.searchsorted(cum, torch.tensor(fraction, dtype=cum.dtype,
                                                      device=cum.device))), com


def prepare_template(path: pathlib.Path, factor: int, fraction: float, margin: float,
                     device: torch.device) -> tuple[np.ndarray, dict]:
    with mrcfile.open(str(path), permissive=True) as handle:
        volume = torch.from_numpy(np.asarray(handle.data, dtype=np.float32)).to(device)
        pixel = float(handle.voxel_size.x)
    n = volume.shape[0]
    _, com_in = mass_radius(volume, fraction)
    # Centre the mass on the box centre (whole voxels). For a ring the offset is along
    # the tube axis (z), which moves detections along the tube, never off its axis.
    shift = [-int(round(c)) for c in com_in]
    if any(shift):
        volume = torch.roll(volume, shifts=shift, dims=(0, 1, 2))
    radius, com = mass_radius(volume, fraction)
    # crop box: even multiple of 2*factor, so the centre n//2 lands on binned b//2
    box = int(np.ceil(2 * radius * margin / (2 * factor))) * 2 * factor
    box = min(box, n // (2 * factor) * 2 * factor)
    lo = n // 2 - box // 2
    cropped = volume[lo:lo + box, lo:lo + box, lo:lo + box].cpu().numpy()
    del volume
    binned = fourier_crop(cropped, factor)
    info = {"template_box_in": n, "template_pixel": pixel, "mass_radius_px": radius,
            "mass_fraction": fraction, "crop_box": box, "binned_box": binned.shape[0],
            "com_offset_in_px_zyx": com_in, "centring_shift_px_zyx": shift,
            "centring_shift_angstrom_zyx": [s * pixel for s in shift],
            "com_offset_px_zyx": com}
    return binned, info


def polar_options(box: int, convention: str) -> dict:
    """Where the polar grid -- the in-plane (psi) rotation -- is centred.

    Panther-EM's grids defaulted to ``(box / 2 - 0.5, box / 2 - 0.5)`` until 2026-11-12
    (fixed on branch jd/mt-correlation-table); a projection made by Fourier slice
    (Panther-EM's own and Leopard-EM's) is centred at ``box // 2``. For an even box the
    old default rotated every template about a point half a pixel off its centre,
    which moves the correlation peak by ``(I - R(-psi)) (1/2, 1/2)`` px -- up to 1.4 px,
    by tube direction (measured on 823: 15% of z lost on three of four tubes). ``fft``
    centres the grid at ``box // 2``, as Leopard-EM's positions assume, whatever the
    installed Panther-EM's default; ``panther`` leaves it to that default.
    """
    if convention == "fft":
        return {"center": [float(box // 2), float(box // 2)]}
    return {}


def b_envelope_3d(volume: np.ndarray, pixel: float, b_factor: float) -> np.ndarray:
    """``volume`` with Leopard-EM's CTF B-factor envelope exp(-B k^2 / 4) (k in 1/A)
    applied in 3-D. A projection's spectrum is a central slice of the volume's, with the
    same |k|, so this is exactly the envelope Leopard-EM puts on every projection's CTF
    (``ctf_B_factor`` in its optics group); Panther-EM's CTF planes carry none."""
    k = [np.fft.fftfreq(n) / pixel for n in volume.shape[:-1]] + [
        np.fft.rfftfreq(volume.shape[-1]) / pixel]
    kz, ky, kx = np.meshgrid(*k, indexing="ij")
    envelope = np.exp(-b_factor * (kz ** 2 + ky ** 2 + kx ** 2) / 4.0)
    out = np.fft.irfftn(np.fft.rfftn(volume) * envelope, s=volume.shape)
    return out.astype(np.float32)


def band_orientations(nside: int, theta_half: float, phi_range: float):
    """HEALPix pixel centres with |theta - 90| <= theta_half and phi < phi_range."""
    from panther_em.orientations import healpix_orientations

    phi, theta = healpix_orientations(nside)
    keep = (np.abs(theta - 90.0) <= theta_half) & (phi < phi_range)
    return phi[keep], theta[keep]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--micrograph", type=pathlib.Path, required=True)
    parser.add_argument("--template", type=pathlib.Path, required=True,
                        help="the RING template (tube axis on z, axis on the box centre)")
    parser.add_argument("--optics", type=pathlib.Path, required=True,
                        help="YAML with an optics_group (PICASSO *_optics.yaml or a config)")
    parser.add_argument("--outdir", type=pathlib.Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--target-pixel", type=float, default=2.8,
                        help="A/px to bin to (integer factor); the ring needs its "
                             "protofilaments, so finer than the cylinder's 4.5")
    parser.add_argument("--mass-fraction", type=float, default=0.99)
    parser.add_argument("--box-margin", type=float, default=1.15)
    parser.add_argument("--theta-half-range", type=float, default=20.0)
    parser.add_argument("--phi-range", type=float, default=360.0,
                        help="search phi in [0, this); 360/N_pf covers every roll up to "
                             "the lattice's own pseudo-symmetry")
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--defocus-offsets", type=float, nargs="+", default=[0.0])
    parser.add_argument("--rank", type=int, nargs=2, default=[48, 40], metavar=("K", "M"))
    parser.add_argument("--n-psi", type=int, default=256)
    parser.add_argument("--band-cutoff", type=float, default=0.45,
                        help="cycles per binned pixel (Nyquist 0.5)")
    parser.add_argument("--band-falloff", type=float, default=0.05)
    parser.add_argument("--no-invert", action="store_true",
                        help="do not negate the micrograph (Panther-EM's contrast is the "
                             "opposite of Leopard-EM's for these maps)")
    parser.add_argument("--no-seeding", action="store_true")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "tf32", "fp32"])
    parser.add_argument("--polar-center", choices=["fft", "panther"], default="fft",
                        help="centre of the in-plane rotation (see polar_options)")
    parser.add_argument("--b-factor", type=float, default=None,
                        help="CTF B-factor envelope on the template (A^2); default: the "
                             "optics file's ctf_B_factor, else 0")
    parser.add_argument("--keep-work", action="store_true",
                        help="keep <outdir>/<prefix>_svd_work (~0.7 GB) after the run")
    args = parser.parse_args()

    from panther_em.managers import calibrate, decompose, search, whitening
    from panther_em.managers import CalibrateConfig, DecomposeConfig, WhiteningConfig
    from panther_em.managers.search import SearchConfig

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    work = args.outdir / f"{args.prefix}_svd_work"
    require_free_space(args.outdir, 1.5, "svd_trace_maps")
    work.mkdir(parents=True, exist_ok=True)
    remove_on_exit(work, args.keep_work)
    times: dict[str, float] = {}

    optics = read_optics(args.optics)
    pixel = float(optics["pixel_size"])
    factor = max(1, int(round(args.target_pixel / pixel)))
    binned_pixel = pixel * factor

    # 1-2. bin the micrograph and the template
    t = time.perf_counter()
    with mrcfile.open(str(args.micrograph), permissive=True) as handle:
        micrograph = np.asarray(handle.data, dtype=np.float32)
    binned = fourier_crop(micrograph, factor)
    mic_path = work / f"{args.prefix}.mrc"  # the stem is the whitening's key
    write_mrc(mic_path, binned, binned_pixel)
    template, tinfo = prepare_template(args.template, factor, args.mass_fraction,
                                       args.box_margin, device)
    if args.b_factor is None:
        args.b_factor = float(optics.get("ctf_B_factor", 0.0) or 0.0)
    if args.b_factor > 0:
        template = b_envelope_3d(template, binned_pixel, args.b_factor)
    tpl_path = work / "template_binned.mrc"
    write_mrc(tpl_path, template, binned_pixel)
    b = template.shape[0]
    com = tinfo["com_offset_px_zyx"]
    print(f"{args.micrograph.name}: {pixel} A/px, bin {factor} -> {binned_pixel:.3f} A/px, "
          f"{micrograph.shape} -> {binned.shape}")
    print(f"{args.template.name}: {tinfo['template_box_in']}^3, {args.mass_fraction:.0%} of "
          f"mass within {tinfo['mass_radius_px']:.0f} px -> crop {tinfo['crop_box']}^3 -> "
          f"{b}^3 binned; centre of mass was off by (z,y,x) "
          + " ".join(f"{c:+.1f}" for c in tinfo["com_offset_in_px_zyx"])
          + f" px, shifted by {tinfo['centring_shift_px_zyx']}, now "
          f"{com[0]:+.1f} {com[1]:+.1f} {com[2]:+.1f}")
    if max(abs(v) for v in com) > 2 * factor:
        print("  WARNING: template not centred on its box -- Panther-EM rotates about the "
              "box centre, so an off-centre template costs rank and accuracy")
    times["bin"] = time.perf_counter() - t

    # 3. whitening, decomposition (theta band), calibration, search
    t = time.perf_counter()
    whitening.run(WhiteningConfig(micrographs=[mic_path], output=work / "whitening.h5",
                                  band_cutoff=args.band_cutoff,
                                  band_falloff=args.band_falloff, device=str(device)))
    times["whitening"] = time.perf_counter() - t

    t = time.perf_counter()
    phi, theta = band_orientations(args.nside, args.theta_half_range, args.phi_range)
    decompose.healpix_orientations = lambda nside: (phi, theta)  # the band, not the sphere
    k, m = args.rank
    num_angle = max(2 * 128, int(np.ceil(1.875 * b / 16)) * 16)
    num_radius = int(np.ceil(1.5 * b / 16)) * 16
    decompose.run(DecomposeConfig.model_validate({
        "volume": str(tpl_path), "whitening": str(work / "whitening.h5"),
        "output": str(work / "decomposition.h5"),
        "polar_grid": {"kind": "spiral", "num_angle": num_angle, "num_radius": num_radius,
                       "options": polar_options(b, args.polar_center)},
        "healpix_nside": args.nside, "k_max": 128, "eig_max": 128,
        "svd_method": "low_rank", "device": str(device)}))
    times["decompose"] = time.perf_counter() - t
    print(f"decomposed {len(phi)} orientations (theta 90 +/- {args.theta_half_range}, phi < "
          f"{args.phi_range}) on a {num_angle} x {num_radius} polar grid centred by "
          f"'{args.polar_center}'; template B-factor {args.b_factor} A^2")

    t = time.perf_counter()
    calibrate.run(CalibrateConfig.model_validate({
        "volume": str(tpl_path), "whitening": str(work / "whitening.h5"),
        "decomposition": str(work / "decomposition.h5"),
        "output_dir": str(work / "calibration"),
        "tables": {"trace": [[0, 0, k, m]]},
        "suggest": [{"max_error": 0.3}, {"max_error": 0.2}, {"max_error": 0.1}],
        "device": str(device)}))
    times["calibrate"] = time.perf_counter() - t
    report = json.loads((work / "calibration" / "calibration.json").read_text())
    print(f"rank {k} x {m}: truncation error {report['tables']['trace']}")

    stage = {"name": "trace", "rectangles": [[0, 0, k, m]], "table": "trace",
             "precision": args.precision, "n_psi": args.n_psi,
             "decision": {"threshold": "auto", "accept": True}}
    if not args.no_seeding:
        stage["seeding"] = {"level": 1, "rings": 2, "templates": "cell_mean",
                            "lattice": 2, "top_k": 3}
    mean_defocus = 0.5 * (float(optics["defocus_u"]) + float(optics["defocus_v"]))
    config = SearchConfig.model_validate({
        "decomposition": str(work / "decomposition.h5"),
        "whitening": str(work / "whitening.h5"),
        "tables": {"trace": str(work / "calibration" / "tables_trace.h5")},
        "optics": {"pixel_size": binned_pixel, "voltage": float(optics["voltage"]),
                   "spherical_aberration": float(optics["spherical_aberration"]),
                   "amplitude_contrast_ratio": float(optics["amplitude_contrast_ratio"]),
                   "phase_shift": float(optics.get("phase_shift", 0.0))},
        "micrographs": [{"path": str(mic_path), "defocus_u": float(optics["defocus_u"]),
                         "defocus_v": float(optics["defocus_v"]),
                         "astigmatism_angle": float(optics["astigmatism_angle"])}],
        "defocus_offsets": args.defocus_offsets, "invert_contrast": not args.no_invert,
        "stages": [stage], "weights_cells": [0, 0, k, m],
        "output_dir": str(work / "search"), "cache_micrographs": False,
        "overwrite": True, "keep_stage_results": False, "device": str(device)})
    t = time.perf_counter()
    (result_path,) = search.run(config).values()
    times["search"] = time.perf_counter() - t

    # 4. maps for the tracer
    with h5py.File(result_path, "r") as f:
        g = f["tensors"]
        maps = {"z": g["scaled_mip"][()], "psi": g["orientation_psi"][()],
                "theta": g["orientation_theta"][()], "phi": g["orientation_phi"][()],
                "defocus": g["relative_defocus"][()]}
    maps = {key: np.asarray(v, dtype=np.float32).reshape(v.shape[-2:]) for key, v in maps.items()}
    expected = (binned.shape[0] - b + 1, binned.shape[1] - b + 1)
    if maps["z"].shape != expected:
        raise RuntimeError(f"z map {maps['z'].shape}, expected the valid grid {expected}")
    meta = {"micrograph": str(args.micrograph.resolve()), "template": str(args.template),
            "optics": str(args.optics), "factor": factor, "binned_pixel": binned_pixel,
            "half": b // 2, "defocus_mean": mean_defocus, "n_orientations": int(len(phi)),
            "rank": [k, m], "args": {key: (str(v) if isinstance(v, pathlib.Path) else v)
                                     for key, v in vars(args).items()},
            "template_info": tinfo, "times_s": times,
            "truncation_error": report["tables"]["trace"]}
    out = args.outdir / f"{args.prefix}_svd_maps.npz"
    np.savez_compressed(out, **maps, half=b // 2, factor=factor, binned_pixel=binned_pixel,
                        meta=json.dumps(meta))
    (args.outdir / f"{args.prefix}_svd_maps.json").write_text(json.dumps(meta, indent=2))
    z = maps["z"]
    print(f"z map {z.shape}: median {np.median(z):.2f}, max {z.max():.2f}; times "
          + ", ".join(f"{key} {v:.1f} s" for key, v in times.items()))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
