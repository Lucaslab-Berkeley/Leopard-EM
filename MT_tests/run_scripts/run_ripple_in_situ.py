"""Motion-correct the in situ movies with RIPPLE, driven by their .mdoc files.

Every parameter that varies between movies comes out of the mdoc rather than the YAML,
so the same config runs the whole set:

    PixelSpacing   -> movie_config.pixel_size  (0.528 A, the SUPER-RESOLUTION pixel)
    Voltage        -> movie_config.voltage
    NumSubFrames   -> deformation_field_resolution[0], and fluence = n * per-frame

The one thing the mdoc does NOT carry is the dose: ExposureDose is 0 in every file.
That comes from ``--fluence-per-frame`` (1.0 e-/A^2/frame as collected).

Measured, the frames average 0.63-0.68 counts/A^2/frame with the brightest blocks at
0.75-0.89, so the recorded counts sit below the nominal 1.0. That is the usual mix of
detector DQE and specimen absorption and it is reported per movie rather than used to
override the nominal figure -- dose weighting wants the electrons that went through the
specimen, not the ones the camera registered.

Outputs per movie, into ``--outdir``:

    {stem}_DW.mrc          dose-weighted sum, Fourier-cropped to 1.056 A/px
    {stem}_def_field.csv   the deformation field

Usage:
    # everything, one GPU
    python run_scripts/run_ripple_in_situ.py

    # spread over four GPUs (run each in its own shell)
    python run_scripts/run_ripple_in_situ.py --gpu 0 --shard 0/4
    python run_scripts/run_ripple_in_situ.py --gpu 1 --shard 1/4
    ...
"""

import argparse
import pathlib
import time

import numpy as np
import tifffile
import torch
from ripple.managers import AlignFramesManager
from ripple.utils.data_io import load_array_from_path

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MT_ROOT / "configs" / "align_in_situ.yaml"
DEFAULT_FRAMES = MT_ROOT / "Frames" / "in_situ"
DEFAULT_OUTDIR = MT_ROOT / "results_in_situ"


def rotate_flip(array: np.ndarray, operation: int) -> np.ndarray:
    """Apply a SerialEM RotationAndFlip operation to the last two axes.

    ``operation`` is ``rotation // 90`` plus 4 when a flip is needed, so 7 is a 270
    degree rotation followed by a flip about Y (a left-right reversal of the column
    axis). The frames are written raw -- the TIFF's ImageDescription says
    "r/f 0 need 7" -- so the operation has never been applied and the movie, the
    deformation field and the sum all have to receive it together, which is why it
    happens here on the frames rather than afterwards on the sum.

    NOTE the ORDER. This rotates and then flips, as stated. If the micrographs come
    out point-reflected against another route through this data, the two steps are
    the other way round and that is a one-line change.
    """
    if operation == 0:
        return array
    rotated = np.rot90(array, k=operation & 3, axes=(-2, -1))
    if operation & 4:
        rotated = np.flip(rotated, axis=-1)
    return np.ascontiguousarray(rotated)


def read_mdoc(path: pathlib.Path) -> dict[str, str]:
    """Flatten a SerialEM mdoc to ``key -> value``; later frame sets win."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line or line.lstrip().startswith("["):
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def movie_jobs(frames_dir: pathlib.Path) -> list[tuple[pathlib.Path, dict[str, str]]]:
    """Every movie in the directory that has an mdoc beside it, in name order."""
    jobs = []
    for mdoc in sorted(frames_dir.glob("*.mdoc")):
        movie = mdoc.with_suffix("")  # strip ".mdoc", leaving ".tif"
        if not movie.exists():
            print(f"  skipping {mdoc.name}: no movie beside it")
            continue
        jobs.append((movie, read_mdoc(mdoc)))
    return jobs


def apply_mdoc(manager: AlignFramesManager, movie: pathlib.Path,
               mdoc: dict[str, str], fluence_per_frame: float,
               time_points: int | None = None) -> int:
    """Push the mdoc's optics and frame count into the manager. Returns n frames."""
    movie_config = manager.movie_config
    movie_config.movie_path = str(movie)
    movie_config.pixel_size = float(mdoc["PixelSpacing"])
    movie_config.voltage = float(mdoc["Voltage"])

    n_frames = int(mdoc["NumSubFrames"])
    movie_config.fluence_per_frame = fluence_per_frame
    movie_config.fluence = n_frames * fluence_per_frame

    # One temporal control point per frame unless told otherwise. On low-contrast
    # movies (thin ice, ~1 e-/A^2/frame) a single frame cannot pin its own shift, so
    # fewer points -- each fitted from several frames -- keep the trajectory smooth.
    resolution = list(manager.alignment_config.deformation_field_resolution)
    manager.alignment_config.deformation_field_resolution = (
        min(time_points or n_frames, n_frames), resolution[1], resolution[2])
    return n_frames


def main() -> None:
    """Align every mdoc-described movie in the directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frames", type=pathlib.Path, default=DEFAULT_FRAMES)
    parser.add_argument("--outdir", type=pathlib.Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--gpu", type=int, default=None,
                        help="override computational_config.gpu_id")
    parser.add_argument("--fluence-per-frame", type=float, default=1.0,
                        help="e-/A^2 per frame; the mdoc's ExposureDose is 0")
    parser.add_argument("--time-points", type=int, default=None,
                        help="temporal control points of the deformation field "
                             "(default: one per frame)")
    parser.add_argument("--no-xc-prepass", dest="xc_prepass", action="store_false",
                        help="start the optimizer from zero motion instead of the "
                             "per-frame cross-correlation seed, which locks onto noise "
                             "peaks on low-contrast frames")
    parser.add_argument("--gain", type=pathlib.Path, default=None,
                        help="override movie_config.gain_path: a dataset whose "
                             "directories each carry their own gain reference")
    parser.add_argument("--shard", default="0/1",
                        help="'i/n' -- take every n-th movie starting at i, so n "
                             "shells can share the movie list across GPUs")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many movies (a smoke test)")
    parser.add_argument("--redo", action="store_true",
                        help="re-align movies whose DW sum already exists")
    parser.add_argument("--no-rotate-flip", dest="rotate_flip", action="store_false",
                        help="leave the frames in raw camera orientation instead of "
                             "applying the mdoc's RotationAndFlip")
    args = parser.parse_args()

    index, _, total = args.shard.partition("/")
    index, total = int(index), int(total or 1)

    args.outdir.mkdir(parents=True, exist_ok=True)
    jobs = movie_jobs(args.frames)[index::total]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        raise SystemExit(f"no movies with mdocs under {args.frames}")

    manager = AlignFramesManager.from_yaml(str(args.config))
    if args.gpu is not None:
        manager.computational_config.gpu_id = args.gpu
    if args.gain is not None:
        manager.movie_config.gain_path = str(args.gain)
    if not args.xc_prepass:
        manager.alignment_config.use_xc_prepass = False
    print(f"shard {index}/{total}: {len(jobs)} movie(s) on "
          f"gpu {manager.computational_config.gpu_id}")

    # The gain is one file for the whole set, so read it once. Its orientation against
    # the raw frames was checked by gain-correcting a sum under all eight rotation/flip
    # combinations: as stored is the only one that flattens the column fixed pattern
    # (9.24% -> 0.43% of the mean), so no gain_rot/gain_flip is needed.
    gain = load_array_from_path(manager.movie_config.gain_path)
    print(f"gain {gain.shape}, median {float(np.median(gain)):.4f}, multiplicative\n")

    for number, (movie, mdoc) in enumerate(jobs, start=1):
        stem = movie.stem
        dw_path = args.outdir / f"{stem}_DW.mrc"
        if dw_path.exists() and not args.redo:
            print(f"[{number}/{len(jobs)}] {stem}: DW sum exists, skipping")
            continue

        n_frames = apply_mdoc(manager, movie, mdoc, args.fluence_per_frame,
                              args.time_points)
        manager.output_config.dw_sum_output_path = str(dw_path)
        manager.output_config.deformation_field_output_path = str(
            args.outdir / f"{stem}_def_field.csv")

        pixel = manager.movie_config.pixel_size
        output_pixel = pixel * manager.movie_config.super_resolution_factor
        print(f"[{number}/{len(jobs)}] {stem}")
        print(f"    {n_frames} frames ({manager.alignment_config.deformation_field_resolution[0]}"
              f" time points, xc prepass {manager.alignment_config.use_xc_prepass}), "
              f"{pixel} A/px super-res -> {output_pixel:.4f} "
              f"A/px out, {manager.movie_config.fluence:.0f} e-/A^2 total, "
              f"{manager.movie_config.voltage:.0f} kV")

        start = time.time()
        # Load as uint8 (4.7 GB) and orient BEFORE the float32 cast (18.8 GB).
        frames = tifffile.imread(str(movie))
        operation = int(mdoc.get("RotationAndFlip", 0)) if args.rotate_flip else 0
        frames = rotate_flip(frames, operation)
        oriented_gain = rotate_flip(gain, operation)
        print(f"    rotation/flip {operation} -> {frames.shape[-2:]} "
              f"(from {tifffile.TiffFile(str(movie)).pages[0].shape})")

        prepared = manager.prepare_movie(
            movie=torch.from_numpy(frames).float(),
            gain_map=torch.from_numpy(oriented_gain).to(
                manager.computational_config.gpu_device))
        del frames
        print(f"    prepared {tuple(prepared.shape)} in {time.time() - start:.0f}s")

        deformation_field, trajectory = manager.estimate_motion(prepared)
        manager.correct_and_save(prepared, deformation_field, trajectory)

        # (2, t, h, w) in super-resolution pixels; the excursion each control point
        # travels over the exposure is the honest one-number summary of the motion.
        field = deformation_field.data.detach().cpu().numpy() * pixel
        excursion = (field.max(axis=1) - field.min(axis=1))
        print(f"    done in {time.time() - start:.0f}s, control-point travel "
              f"{np.hypot(*excursion).mean():.1f} A mean, "
              f"{np.hypot(*excursion).max():.1f} A max -> {dw_path.name}\n")

        del prepared
        torch.cuda.empty_cache()


# NOTE: __main__ guard required for the GPU/multiprocessing setup inside RIPPLE.
if __name__ == "__main__":
    main()
