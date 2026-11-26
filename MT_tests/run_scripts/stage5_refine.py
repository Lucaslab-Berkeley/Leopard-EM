"""Stage 5: refine the Stage 4 sites with the same ring, then re-measure the lattice.

``refine_template`` takes each site off the search grid -- phi, theta, psi, defocus and
position -- with COARSE steps equal to the search's own steps (read from the search
config), so it searches around each peak without re-searching what the search already
covered. Box sizes come from the template's header.

Everything run-specific is read, not assumed: micrograph, optics and template from the
search config; pixel size, table-to-micrograph offset and correlations per pixel from
the table.

After refinement each filament's lattice is re-measured along a curve fitted to the
REFINED sites (``continuous_sites`` again, on the refined columns), with the
per-detection sin(theta) correction.

Usage:
    python run_scripts/stage5_refine.py \\
        --table results_in_situ_823/output_correlation_table_in_situ_823_13pf.h5
    python run_scripts/stage5_refine.py --table ... --dry-run      # stack + config only
    python run_scripts/stage5_refine.py --table ... --measure-only # skip refinement
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import yaml

from leopard_em.analysis.filament_lattice import TemplateLatticeGeometry

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from continuous_unwrap_curved import continuous_sites  # noqa: E402
from frame_geometry import frame_geometry, legacy_frame_geometry  # noqa: E402
from refine_sites_curved import particle_stack, write_config  # noqa: E402
from stage3_read_table import RING_STEM  # noqa: E402
from stage4_sites import block_rises, bootstrap_threshold, reliability  # noqa: E402


def refined_frame(refined: pd.DataFrame) -> pd.DataFrame:
    """The refined sites in the column layout ``continuous_sites`` reads."""
    return pd.DataFrame({
        "x": refined["refined_pos_x"].to_numpy(dtype=float),
        "y": refined["refined_pos_y"].to_numpy(dtype=float),
        "phi": refined["refined_phi"].to_numpy(),
        "theta": refined["refined_theta"].to_numpy(),
        "psi": refined["refined_psi"].to_numpy(),
        "z_score": refined["refined_scaled_mip"].to_numpy(),
    })


def remeasure(refined, filaments, geometry, pixel_size):
    """Rise per filament along a curve fitted to the refined sites."""
    print(f"\n{'filament':<9} {'sites':>6} {'boot z':>7} {'rise A':>17} {'block SE':>9} "
          f"{'vs template':>12} {'dev rms':>8}")
    for label in sorted(set(filaments.tolist())):
        block = refined_frame(refined[filaments == label]).reset_index(drop=True)
        floor = float(block["z_score"].min()) - 0.01
        boot = bootstrap_threshold(block, floor=floor)
        if boot is None:
            print(f"{label:<9} {len(block):>6}  too few sites to re-measure")
            continue
        # These are already one clean detection per site, so bootstrapping lower is
        # safe -- and needed when the strongest 20 cluster in one stretch of a sparse
        # tube, leaving too little span to fit a curve through.
        sites, failure = None, None
        # at least the bootstrap cut itself, even when every site is stronger than it
        # (GMPCPP: all refined z > 8.4, so arange(8.0, 8.4) was empty)
        for cut in np.arange(boot, min(floor, boot - 1e-6), -0.25):
            try:
                sites = continuous_sites(block, geometry, float(cut),
                                         pixel_size=pixel_size)
                boot = float(cut)
                break
            except ValueError as error:
                failure = error
        if sites is None:
            print(f"{label:<9} {len(block):>6}  {failure}")
            continue
        rise, error = sites["rise"], sites["rise_standard_error"]
        rms = float(np.sqrt(np.mean(sites["deviation"] ** 2)))
        n_sites = len(sites["axial_index"])
        blocks, block_error = block_rises(
            block.iloc[sites["detection_index"]].reset_index(drop=True),
            sites["axial_index"], geometry, pixel_size)
        print(f"{label:<9} {n_sites:>6} {boot:>7.2f} "
              f"{rise:>8.3f} ± {error:6.3f} {block_error:>9.3f} "
              f"{100 * (rise / geometry.rise_angstrom - 1):>+11.2f}% {rms:>6.2f} A  "
              f"{reliability(n_sites, rms)}")
        if blocks:
            print(f"{'':<9} blocks: " + ", ".join(f"{r:.3f}" for r in blocks))


# pylint: disable=too-many-locals
def main() -> None:
    """Build the stack, refine it, and re-measure each filament."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=pathlib.Path, required=True)
    parser.add_argument("--search-config", type=pathlib.Path, default=None,
                        help="default configs/match_tm_<tag>.yaml")
    parser.add_argument("--stem", default=RING_STEM)
    parser.add_argument("--ctf-b-factor", type=float, default=60.0)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--in-vitro-legacy", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="write the stack and config, do not refine")
    parser.add_argument("--measure-only", action="store_true",
                        help="re-measure an existing refined CSV")
    parser.add_argument("--tag", default=None,
                        help="the SEARCH's tag (its map names and config); default "
                             "from the table name")
    parser.add_argument("--label", default=None,
                        help="name for this stage's outputs; default the tag")
    args = parser.parse_args()

    table = args.table if args.table.is_absolute() else MT_ROOT / args.table
    tag = args.tag or table.stem.removeprefix("output_correlation_table_")
    label = args.label or tag
    results = table.parent
    frame = (legacy_frame_geometry(table, args.stem) if args.in_vitro_legacy
             else frame_geometry(table))
    geometry = TemplateLatticeGeometry.from_pdb(str(MT_ROOT / "models" / f"{args.stem}.pdb"))
    search_config = args.search_config or MT_ROOT / "configs" / f"match_tm_{tag}.yaml"
    search = yaml.safe_load(search_config.read_text())
    optics = dict(search["optics_group"])
    optics.pop("label", None)
    optics["ctf_B_factor"] = args.ctf_b_factor
    template = search["template_volume_path"]
    micrograph = search["micrograph_path"]
    print(frame.describe())
    print(f"search config {search_config.name}: {micrograph}, {template}")

    sites = pd.read_csv(results / f"stage4_sites_{tag}.csv")
    out = results / f"refined_sites_{label}.csv"
    if not args.measure_only:
        print(f"{len(sites)} Stage 4 sites over {sites['filament'].nunique()} filament(s)")
        # particle_stack carries the segmented pipeline's labels; continuous sites are
        # one segment per filament.
        sites["region"] = sites["filament"]
        sites["segment"] = 1
        stack = particle_stack(sites, optics, micrograph, template,
                               str(results.relative_to(MT_ROOT)), tag, frame)
        stack["filament"] = sites["filament"].to_numpy()
        stack["axial_index"] = sites["axial_index"].to_numpy()
        stack_path = results / f"stack_sites_{label}.csv"
        stack.to_csv(stack_path, index=True)
        config_path = MT_ROOT / "configs" / f"refine_sites_{label}.yaml"
        write_config(stack_path, config_path, template, search_config, args.gpus)
        print(f"wrote {stack_path.relative_to(MT_ROOT)} and "
              f"{config_path.relative_to(MT_ROOT)}")
        if args.dry_run:
            return

        # pylint: disable=import-outside-toplevel
        from leopard_em.pydantic_models.managers import RefineTemplateManager

        manager = RefineTemplateManager.from_yaml(str(config_path))
        start = time.time()
        manager.run_refine_template(str(out), args.batch_size)
        print(f"refined in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")

    refined = pd.read_csv(out, index_col=0)
    stack = pd.read_csv(results / f"stack_sites_{label}.csv", index_col=0)
    # Stacks from the in vitro runs call it "region" (the drawn region it came from).
    column = "filament" if "filament" in stack.columns else "region"
    filaments = stack[column].to_numpy()[: len(refined)]
    gain = refined["refined_scaled_mip"] - refined["scaled_mip"]
    print(f"\nz {refined['scaled_mip'].mean():.3f} -> "
          f"{refined['refined_scaled_mip'].mean():.3f} (mean {gain.mean():+.3f}, "
          f"{(gain > 0).mean():.0%} improved)")
    for name in ("phi", "theta", "psi"):
        move = np.abs(((refined[f"refined_{name}"] - refined[name] + 180) % 360) - 180)
        print(f"  {name:<6} median move {move.median():.2f}°, "
              f"{refined[name].round(2).nunique()} -> "
              f"{refined[f'refined_{name}'].round(2).nunique()} distinct values")
    remeasure(refined, filaments, geometry, frame.pixel_size)


# NOTE: invoking under __main__ is required for the multiprocessing/GPU distribution.
if __name__ == "__main__":
    main()
