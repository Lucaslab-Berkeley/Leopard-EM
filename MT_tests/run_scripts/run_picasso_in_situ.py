"""PICASSO CTF estimation on the in situ dose-weighted sums.

The same thing PICASSO's own ``programs/ctf_estimation/run_ctf_estimation.py`` does,
but taking the config as an argument instead of a module-level constant, so the config
can live with the rest of this project's configs and the PICASSO checkout stays clean.

Afterwards it prints what the mdocs asked for beside what PICASSO measured. The mdoc
``Defocus`` is what SerialEM's own autofocus thought at the time and is an independent
check worth a glance -- agreement to a few hundred Angstroms is normal, a micrograph
off by microns is a fit that failed.

Usage:
    /home/jdickerson/git/LucasLab/PICASSO/.venv/bin/python \
        run_scripts/run_picasso_in_situ.py
"""

import argparse
import pathlib
import time

import pandas as pd
from picasso.managers import CTFEstimateManager

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MT_ROOT / "configs" / "ctf_in_situ.yaml"
DEFAULT_FRAMES = MT_ROOT / "Frames" / "in_situ"


def mdoc_defocus(frames_dir: pathlib.Path) -> dict[str, dict[str, float]]:
    """``stem -> {target, autofocus}`` in Angstroms, from each mdoc.

    Use TARGET, not the autofocus reading. SerialEM's measured ``Defocus`` is
    unusable on this set -- it records 0.0135 um on one micrograph and -4.0 um on
    another against targets of -1.1 and -0.9, and across the 26 it correlates with
    the fitted defocus at -0.03, which is to say not at all. Autofocus fails often on
    thick cellular specimens. ``TargetDefocus`` is what the scope was asked for and
    correlates at 0.88, so that is the honest cross-check.
    """
    values: dict[str, dict[str, float]] = {}
    for mdoc in sorted(frames_dir.glob("*.mdoc")):
        stem = mdoc.name.removesuffix(".tif.mdoc")
        entry: dict[str, float] = {}
        for line in mdoc.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "TargetDefocus":
                entry["target"] = abs(float(value)) * 1e4
            elif key.strip() == "Defocus":
                entry["autofocus"] = abs(float(value)) * 1e4
        values[stem] = entry
    return values


def main() -> None:
    """Estimate the CTF for every DW sum, then compare against the mdocs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frames", type=pathlib.Path, nargs="+", default=[DEFAULT_FRAMES],
                        help="directories holding the movies' .mdoc files (for the "
                             "TargetDefocus comparison); several if the set is split")
    args = parser.parse_args()

    manager = CTFEstimateManager.from_yaml(str(args.config))
    images = manager.input_config.discover_images()
    print(f"{len(images)} micrograph(s) at "
          f"{manager.optical_config.pixel_spacing_angstroms} A/px")
    for device, k in manager.worker_counts(len(images)):
        print(f"  {device}: {k} concurrent image(s)")

    start = time.time()
    results = manager.estimate_and_export()
    print(f"\nfinished {len(results)} image(s) in {time.time() - start:.0f}s")

    output_dir = pathlib.Path(manager.output_config.output_directory)
    optics = pd.read_csv(output_dir / manager.output_config.optics_csv_filename)
    metrics = pd.read_csv(output_dir / manager.output_config.metrics_csv_filename)
    nominal = {}
    for frames_dir in args.frames:
        nominal.update(mdoc_defocus(frames_dir))

    mean_defocus = (optics["defocus_u"] + optics["defocus_v"]) / 2
    astigmatism = (optics["defocus_u"] - optics["defocus_v"]).abs()
    stems = [pathlib.Path(p).stem.removesuffix("_DW")
             for p in optics.get("image_path", optics.iloc[:, 0])]

    print(f"\n{'micrograph':<28} {'PICASSO A':>10} {'target A':>9} {'ratio':>7} "
          f"{'astig A':>9} {'thick A':>9}")
    thickness = metrics.get("thickness_angstroms",
                            pd.Series([float("nan")] * len(optics)))
    ratios = []
    for stem, mean, astig, thick in zip(stems, mean_defocus, astigmatism, thickness):
        target = nominal.get(stem, {}).get("target")
        ratio = f"{mean / target:7.2f}" if target else f"{'-':>7}"
        if target:
            ratios.append(mean / target)
        print(f"{stem:<28} {mean:10.0f} {target if target else float('nan'):9.0f} "
              f"{ratio} {astig:9.0f} {thick:9.0f}")

    ceiling = manager.thickness_config.thickness_range_angstroms[1]
    railed = (thickness >= ceiling - 50).sum()
    print(f"\nmean defocus {mean_defocus.mean():.0f} +/- {mean_defocus.std():.0f} A, "
          f"astigmatism {astigmatism.mean():.0f} A")
    if ratios:
        ratios = pd.Series(ratios)
        print(f"fitted/target {ratios.mean():.2f} +/- {ratios.std():.2f}")
    # The strongest check available without a second estimator: the 1D fit on the mean
    # power spectrum and the 2D fit on patches are independent, so their agreement is
    # evidence the defocus is real rather than a shared artefact.
    if "defocus_1d_A" in metrics and "defocus_2d_mean_A" in metrics:
        gap = (metrics["defocus_1d_A"] - metrics["defocus_2d_mean_A"]).abs()
        print(f"1D vs 2D median disagreement {gap.median():.0f} A")
    if railed:
        print(f"{railed} micrograph(s) pinned at the {ceiling:.0f} A thickness ceiling -- that "
              f"is the grid railing, not a measurement. Defocus is unaffected; it is "
              f"frozen before thickness is fit.")

    print(f"\noptics CSV: {output_dir / manager.output_config.optics_csv_filename}")


if __name__ == "__main__":
    main()
