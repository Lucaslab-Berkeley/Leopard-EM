"""Turn drawn filament paths into a per-pixel angular constraint.

``napari_choose_filament_path.py`` always saves the paths it drew as
``<output>.paths.json``, and writes the constraint too when ``h5py`` is available. A
napari-only environment often has no ``h5py``, so this does the second half
separately -- the same split the membrane workflow uses (draw in the napari env,
export in the one with the heavy dependencies).

It is also the way to re-export at a different width, cone or polarity **without
redrawing**: edit the JSON or pass new options here.

Usage:
    python make_filament_constraint.py paths.json -o sidecar.yaml
    python make_filament_constraint.py paths.json -o sidecar.yaml --width-px 300
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# The drawing tool owns the geometry and the file formats; this only orchestrates.
from napari_choose_filament_path import (
    build_sidecar_payload,
    dump_sidecar_yaml,
    estimate_n_orientations,
    load_mrc_image,
    load_paths_json,
    paint_path_maps,
    preview_text,
    write_constraint_hdf5,
)


def main() -> None:
    """Rasterize saved paths and write the YAML sidecar plus its HDF5."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths_json", type=pathlib.Path)
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True,
                        help="sidecar YAML; the HDF5 is written alongside it")
    parser.add_argument("--micrograph", type=pathlib.Path, default=None,
                        help="override the micrograph recorded in the JSON")
    parser.add_argument("--width-px", type=float, default=None,
                        help="override every path's width")
    parser.add_argument("--polarity", default=None,
                        choices=["both", "positive", "negative"],
                        help="override every path's polarity")
    parser.add_argument("--cone-half-angle-deg", type=float, default=10.0)
    parser.add_argument("--theta-center-deg", type=float, default=90.0)
    parser.add_argument("--psi-step", type=float, default=1.5)
    parser.add_argument("--theta-step", type=float, default=2.5)
    parser.add_argument("--pixel-size-angstrom", type=float, default=None,
                        help="override the MRC header (cropped files carry none)")
    args = parser.parse_args()

    recorded_micrograph, paths = load_paths_json(str(args.paths_json))
    if not paths:
        parser.error(f"{args.paths_json} contains no paths.")

    micrograph = args.micrograph or recorded_micrograph
    if micrograph is None:
        parser.error("No micrograph in the JSON; pass --micrograph.")
    micrograph = pathlib.Path(micrograph).expanduser().resolve()

    for drawn in paths:
        if args.width_px is not None:
            drawn["width_px"] = float(args.width_px)
        if args.polarity is not None:
            drawn["polarity"] = args.polarity

    image, header_pixel_size = load_mrc_image(str(micrograph))
    pixel_size = args.pixel_size_angstrom or header_pixel_size

    eligible, region_id, psi_center, distance = paint_path_maps(image.shape, paths)

    yaml_path = args.output.expanduser().resolve()
    hdf5_path = yaml_path.with_suffix(".h5")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    write_constraint_hdf5(
        str(hdf5_path), eligible, region_id, psi_center,
        estimate_n_orientations(eligible, args.cone_half_angle_deg,
                                args.psi_step, args.theta_step,
                                args.theta_center_deg),
        signed_distance=distance, paths=paths,
        cone_half_angle_deg=args.cone_half_angle_deg,
        theta_center_deg=args.theta_center_deg,
        psi_step=args.psi_step, theta_step=args.theta_step,
        pixel_size_angstrom=pixel_size,
    )
    yaml_path.write_text(
        dump_sidecar_yaml(
            build_sidecar_payload(
                paths, args.cone_half_angle_deg, args.theta_center_deg,
                args.psi_step, args.theta_step,
                micrograph_path=str(micrograph),
                spatial_constraint_path=str(hdf5_path),
            )
        ),
        encoding="utf-8",
    )

    print(f"micrograph {micrograph.name}  {image.shape}"
          + (f"  pixel size {pixel_size:.4f} Å" if pixel_size else ""))
    print(preview_text(paths, eligible, pixel_size, args.cone_half_angle_deg))
    print(f"\nwrote {yaml_path}")
    print(f"wrote {hdf5_path}")


if __name__ == "__main__":
    main()
