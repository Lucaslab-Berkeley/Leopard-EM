"""Stage 3 of PIPELINE.md: refine the reference RING on its own match_template peaks.

Polishing the angles is not cosmetic here. Everything downstream reads phi as the
azimuth about the tube: the protofilament harmonic, the helical unwrap, and the axis fit
all take it directly. Left on the coarse search grid it is quantised -- on an earlier
micrograph that quantisation produced only 19 distinct phi across 82 sites and faked a
supertwist slope that was not there.

The coarse steps in the YAML must equal the steps the match_template run actually used.
That is what ``coarse_*_step`` means: the grid this refine is filling in. Setting them
smaller silently leaves gaps, and this config checks them against the search config at
run time rather than trusting the YAML.

Usage:
    python run_scripts/run_refine_curved.py --tag gdp_curved_13pf
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import yaml

from leopard_em.pydantic_models.managers import RefineTemplateManager

MT_ROOT = Path(__file__).resolve().parents[1]
PARTICLE_BATCH_SIZE = 16


def check_coarse_steps(refine_config: Path, search_config: Path) -> None:
    """Fail loudly if the refine's coarse steps do not match the search that ran."""
    with open(refine_config, encoding="utf-8") as handle:
        refine = yaml.safe_load(handle)["orientation_refinement_config"]
    with open(search_config, encoding="utf-8") as handle:
        search = yaml.safe_load(handle)
    orientation = search["orientation_search_config"]
    defocus = search["defocus_search_config"]

    problems = []
    for name, refined, searched in (
        ("psi", refine["psi_step_coarse"], orientation["psi_step"]),
        ("theta", refine["theta_step_coarse"], orientation["theta_step"]),
    ):
        if abs(float(refined) - float(searched)) > 1e-9:
            problems.append(
                f"{name}_step_coarse is {refined} but match_template used {searched}"
            )
    with open(refine_config, encoding="utf-8") as handle:
        defocus_refine = yaml.safe_load(handle)["defocus_refinement_config"]
    half_cell = float(defocus["defocus_step"]) / 2.0
    span = max(abs(defocus_refine["defocus_min"]), abs(defocus_refine["defocus_max"]))
    if span < half_cell - 1e-9:
        problems.append(
            f"defocus refine spans +/-{span} but the coarse cell is +/-{half_cell}"
        )
    if problems:
        raise SystemExit(
            "refine steps do not match the search that produced these peaks:\n  "
            + "\n  ".join(problems)
        )
    print(f"coarse steps match the search: psi {orientation['psi_step']}, "
          f"theta {orientation['theta_step']}, defocus {defocus['defocus_step']}")


def main() -> None:
    """Refine one tag's peaks and report how far the angles actually moved."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="gdp_curved_13pf")
    parser.add_argument("--results", default="results_gdp_curved")
    parser.add_argument("--gpus", type=int, nargs="+", default=None)
    args = parser.parse_args()

    os.chdir(MT_ROOT)
    refine_config = MT_ROOT / "configs" / f"refine_{args.tag}.yaml"
    search_config = MT_ROOT / "configs" / f"match_tm_{args.tag}.yaml"
    if not refine_config.is_file():
        raise SystemExit(f"no refine config at {refine_config}")
    check_coarse_steps(refine_config, search_config)

    manager = RefineTemplateManager.from_yaml(str(refine_config))
    if args.gpus:
        manager.computational_config.gpu_ids = list(args.gpus)
    offsets = manager.orientation_refinement_config.euler_angles_offsets
    defocus = manager.defocus_refinement_config.defocus_values
    before = pd.read_csv(manager.particle_stack._df_path
                         if hasattr(manager.particle_stack, "_df_path")
                         else MT_ROOT / args.results / f"results_{args.tag}.csv",
                         index_col=0)
    print(f"{len(before)} particles, {offsets.shape[0]} orientation offsets x "
          f"{len(defocus)} defocus")

    out = MT_ROOT / args.results / f"refined_{args.tag}.csv"
    start = time.time()
    manager.run_refine_template(str(out), PARTICLE_BATCH_SIZE)
    print(f"wall time {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")

    after = pd.read_csv(out, index_col=0)
    print(f"wrote {out}  ({len(after)} particles)")

    # The point of the stage: did the angles come off the coarse grid?
    print(f"\n{'angle':<22} {'distinct before':>16} {'distinct after':>15} "
          f"{'median |move|':>14}")
    for name, coarse, refined in (
        ("phi (roll)", "phi", "refined_phi"),
        ("theta (tilt)", "theta", "refined_theta"),
        ("psi (in-plane)", "psi", "refined_psi"),
    ):
        if refined not in after.columns:
            continue
        old = after[coarse].to_numpy()
        new = after[refined].to_numpy()
        move = abs(((new - old + 180.0) % 360.0) - 180.0)
        print(f"  {name:<20} {len(set(old.round(2))):16d} "
              f"{len(set(new.round(2))):15d} {move.mean():13.2f}°")
    if "refined_scaled_mip" in after.columns:
        gain = after["refined_scaled_mip"] - after["scaled_mip"]
        print(f"\n  z: {after['scaled_mip'].mean():.3f} -> "
              f"{after['refined_scaled_mip'].mean():.3f} "
              f"(mean gain {gain.mean():+.3f}, {(gain > 0).mean():.0%} improved)")
        print(f"  max z {after['scaled_mip'].max():.2f} -> "
              f"{after['refined_scaled_mip'].max():.2f}")


# NOTE: invoking under __main__ is required for the multiprocessing/GPU distribution.
if __name__ == "__main__":
    main()
