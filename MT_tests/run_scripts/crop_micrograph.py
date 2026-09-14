"""Crop a micrograph to a smaller field for quick tests.

Follows the convention in ``docs/tutorials/match_template_intro.md``: the crop is
taken from the CENTRE of the frame (not a quadrant), and the division factor names
the output, so ``--factor 2`` writes ``<stem>_cropped_2.mrc`` and keeps half of each
axis -- a quarter of the area.

``mrcfile.new().set_data()`` writes a fresh header, so the **pixel size is not
carried over**. That is harmless here because every downstream config supplies it
(``optics_group.pixel_size``), but it does mean you cannot read the pixel size back
off a cropped file. The nominal value is printed so it is at least recorded.

Usage:
    python run_scripts/crop_micrograph.py Frames/GDP_curved_mgraph.mrc --factor 2
    python run_scripts/crop_micrograph.py Frames/foo.mrc --origin 1024 2048 \
        --size 2880 2046
"""

import argparse
import pathlib

import mrcfile
import numpy as np


def centre_window(shape: tuple[int, int], factor: int) -> tuple[int, int, int, int]:
    """Origin and size of a centred window covering 1/factor of each axis."""
    height, width = shape
    new_height, new_width = height // factor, width // factor
    return (
        height // 2 - new_height // 2,
        width // 2 - new_width // 2,
        new_height,
        new_width,
    )


def main() -> None:
    """Crop one micrograph and report what was taken."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("micrograph", type=pathlib.Path)
    parser.add_argument("--factor", type=int, default=2,
                        help="keep 1/factor of each axis, centred (default 2)")
    parser.add_argument("--origin", type=int, nargs=2, metavar=("Y", "X"),
                        help="top-left corner, overriding the centred window")
    parser.add_argument("--size", type=int, nargs=2, metavar=("H", "W"),
                        help="window size, used with --origin")
    parser.add_argument("-o", "--output", type=pathlib.Path,
                        help="output path (default <stem>_cropped_<factor>.mrc)")
    args = parser.parse_args()

    with mrcfile.open(str(args.micrograph), permissive=True) as handle:
        data = np.asarray(handle.data, dtype=np.float32)
        nominal_pixel_size = float(handle.voxel_size.x)

    if args.origin is not None:
        if args.size is None:
            parser.error("--origin requires --size")
        y0, x0 = args.origin
        height, width = args.size
        tag = f"{y0}_{x0}"
    else:
        y0, x0, height, width = centre_window(data.shape, args.factor)
        tag = str(args.factor)

    if y0 < 0 or x0 < 0 or y0 + height > data.shape[0] or x0 + width > data.shape[1]:
        parser.error(
            f"window y {y0}:{y0 + height}, x {x0}:{x0 + width} "
            f"falls outside the {data.shape} micrograph"
        )

    cropped = data[y0:y0 + height, x0:x0 + width]
    output = args.output or args.micrograph.with_name(
        f"{args.micrograph.stem}_cropped_{tag}.mrc"
    )

    with mrcfile.new(str(output), overwrite=True) as handle:
        handle.set_data(cropped.astype(np.float32))

    print(f"{args.micrograph}  {data.shape}  "
          f"header pixel size {nominal_pixel_size:.4f} Å")
    print(f"  crop y {y0}:{y0 + height}, x {x0}:{x0 + width}  ->  {cropped.shape}")
    print(f"  wrote {output}")
    print("  NOTE: the output header carries no pixel size; supply it in the YAML.")


if __name__ == "__main__":
    main()
