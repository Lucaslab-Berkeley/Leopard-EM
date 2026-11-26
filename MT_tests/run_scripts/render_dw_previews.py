"""Render every dose-weighted sum as a binned, contrast-enhanced PNG.

Screening images, not figures: 8x8 binned so a 5760 x 4092 sum becomes 720 x 511,
written at exactly that size with no axes or borders so one image pixel is one PNG
pixel.

Three contrast modes, because these are unroofed cells and the specimen itself puts a
strong thickness gradient across the frame -- bright thin edge, dark thick interior:

``stretch``
    Percentile clip and linear rescale. Faithful: relative density across the frame
    still means something. The gradient survives, so the thin end can look washed out.
``flatten`` (default)
    Divide out a heavily Gaussian-blurred copy first, then stretch. The blur radius is
    far larger than anything structural, so it removes illumination and thickness
    gradients while leaving filaments untouched. Best for actually seeing the
    cytoskeleton across the whole frame. Density is no longer comparable between one
    part of the frame and another -- that is the trade.
``clahe``
    Adaptive histogram equalisation. Strongest local contrast, most aggressive; it
    will amplify noise in empty regions and is the easiest of the three to
    over-interpret.

Usage:
    python run_scripts/render_dw_previews.py
    python run_scripts/render_dw_previews.py --mode stretch
    python run_scripts/render_dw_previews.py --contact-sheet
"""

import argparse
import pathlib

import mrcfile
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from skimage import exposure

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INDIR = MT_ROOT / "results_in_situ"
DEFAULT_OUTDIR = MT_ROOT / "results_in_situ" / "previews"


def bin_image(image: np.ndarray, factor: int) -> np.ndarray:
    """Mean-bin by an integer factor, cropping the remainder off the far edges."""
    height = image.shape[0] // factor * factor
    width = image.shape[1] // factor * factor
    cropped = image[:height, :width]
    return cropped.reshape(height // factor, factor,
                           width // factor, factor).mean(axis=(1, 3))


def enhance(image: np.ndarray, mode: str, low: float, high: float) -> np.ndarray:
    """Return the binned image as uint8 under the chosen contrast mode."""
    if mode == "flatten":
        # sigma is ~1/12 of the short side: far above any filament, far below the
        # scale of the specimen gradient being removed.
        background = gaussian_filter(image, sigma=min(image.shape) / 12.0)
        image = image - background

    if mode == "clahe":
        # np.clip is not redundant: rescale_intensity can land on 1.0000001 in
        # float32, and equalize_adapthist rejects anything above 1.
        rescaled = np.clip(exposure.rescale_intensity(
            image, in_range=tuple(np.percentile(image, (low, high))),
            out_range=(0.0, 1.0)), 0.0, 1.0)
        return (exposure.equalize_adapthist(rescaled, clip_limit=0.02) * 255
                ).astype(np.uint8)

    lo, hi = np.percentile(image, (low, high))
    if hi <= lo:
        hi = lo + 1e-6
    return (np.clip((image - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def main() -> None:
    """Render one PNG per DW sum, and optionally a contact sheet of all of them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indir", type=pathlib.Path, default=DEFAULT_INDIR)
    parser.add_argument("--outdir", type=pathlib.Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--pattern", default="*_DW.mrc")
    parser.add_argument("--bin", type=int, default=8, dest="binning")
    parser.add_argument("--mode", default="flatten",
                        choices=("flatten", "stretch", "clahe"))
    parser.add_argument("--low", type=float, default=0.5,
                        help="lower percentile for the stretch")
    parser.add_argument("--high", type=float, default=99.5,
                        help="upper percentile for the stretch")
    parser.add_argument("--contact-sheet", action="store_true",
                        help="also write one montage of every micrograph, labelled")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.indir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"no {args.pattern} under {args.indir}")

    tiles = []
    for number, path in enumerate(paths, start=1):
        with mrcfile.open(path, permissive=True) as handle:
            data = handle.data.astype(np.float32)
        binned = bin_image(data, args.binning)
        eight_bit = enhance(binned, args.mode, args.low, args.high)

        stem = path.stem.removesuffix("_DW")
        out = args.outdir / f"{stem}.png"
        Image.fromarray(eight_bit, mode="L").save(out)
        tiles.append((stem, eight_bit))
        print(f"[{number}/{len(paths)}] {stem}  {data.shape} -> {eight_bit.shape}  "
              f"{out.name}")

    print(f"\n{len(tiles)} PNG(s) at bin {args.binning}, mode '{args.mode}' "
          f"-> {args.outdir}")

    if args.contact_sheet:
        columns = 6
        rows = (len(tiles) + columns - 1) // columns
        scale = 3  # the contact sheet is for spotting, not for looking closely
        tile_h, tile_w = tiles[0][1].shape
        cell_h, cell_w = tile_h // scale, tile_w // scale
        label = 14
        sheet = Image.new("L", (columns * cell_w, rows * (cell_h + label)), 0)
        draw = ImageDraw.Draw(sheet)
        for index, (stem, tile) in enumerate(tiles):
            row, column = divmod(index, columns)
            thumb = Image.fromarray(tile, mode="L").resize(
                (cell_w, cell_h), Image.BILINEAR)
            y = row * (cell_h + label)
            sheet.paste(thumb, (column * cell_w, y))
            draw.text((column * cell_w + 3, y + cell_h + 1),
                      stem.replace("set1_", "").split("_Aug09_")[0], fill=255)
        sheet_path = args.outdir / f"contact_sheet_{args.mode}.png"
        sheet.save(sheet_path)
        print(f"contact sheet {sheet.size} -> {sheet_path}")


if __name__ == "__main__":
    main()
