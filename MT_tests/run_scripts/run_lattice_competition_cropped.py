"""Compete the expanded and compacted microtubule lattices on the cropped micrograph.

Runs the 2x2 of {atom source} x {lattice} built by ``models/build_mt_templates.py``
through one identical constrained search, so the only thing that differs between the
four runs is the template volume:

    6dpu_atoms_6dpu_lattice   6dpv_atoms_6dpu_lattice     rows  = atom source (model quality)
    6dpu_atoms_6dpv_lattice   6dpv_atoms_6dpv_lattice     cols  = lattice (the question)

The off-diagonal cells exist so the model-quality offset is *measured* rather than
assumed away: the lattice effect is the mean over atom source of (expanded - compacted),
and the quality effect is the mean over lattice of (6dpu atoms - 6dpv atoms).

The crop is legitimate here even though it is too short for the *estimator*: a wrong
lattice is stretched 2.7% axially, so it misregisters within the template itself and the
per-site signal does not depend on track length. The crop only limits how many paired
sites there are (~9-10), so it can confirm a decisive result but not settle a close one.

Same config and sidecar as run_match_cropped_ring_constrained.py.

Usage:  python run_scripts/run_lattice_competition_cropped.py [--native-b] [--only NAME]
"""

import argparse
import os
import time
from pathlib import Path

from leopard_em.pydantic_models.config import FilamentConstraint
from leopard_em.pydantic_models.managers import MatchTemplateManager

MT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MT_ROOT.parent

YAML_CONFIG_PATH = str(MT_ROOT / "configs" / "match_tm_crop_2rings.yaml")
CONSTRAINT_YAML_PATH = str(MT_ROOT / "configs" / "filament_constraint_box.yaml")
ORIENTATION_BATCH_SIZE = 8
STATS_FROM_VALID_ORIENTATIONS_DEFOCUS = False

PIXEL_SIZE = 0.9194
B_FACTOR_SCALING = 0.5
RESULT_DIR = "results_cropped"

# The 2x2. Order puts the two native cells first so a decisive result shows early.
CELLS = [
    ("6dpu", "6dpu"),
    ("6dpv", "6dpv"),
    ("6dpu", "6dpv"),
    ("6dpv", "6dpu"),
]

_RESULT_KEYS = [
    "mip_path",
    "scaled_mip_path",
    "orientation_psi_path",
    "orientation_theta_path",
    "orientation_phi_path",
    "relative_defocus_path",
    "correlation_average_path",
    "correlation_variance_path",
    "correlation_table_path",
]


def _resolve_existing_path(path: str) -> str:
    """Find a sidecar path exported from either MT_tests or the repo root."""
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate)
    for root in (Path.cwd(), MT_ROOT, REPO_ROOT):
        for alt in (root / path, root / "configs" / Path(path).name,
                    root / "MT_tests" / "configs" / Path(path).name):
            if alt.is_file():
                return str(alt)
    return path


def run_cell(
    stem: str,
    gpu_ids: list[int] | None = None,
    micrograph: str | None = None,
    tag_suffix: str = "crop",
) -> tuple[str, float]:
    """Run one template through the constrained search and write its peak table."""
    tag = f"{stem}_{tag_suffix}"
    template = MT_ROOT / "maps" / f"{stem}_{PIXEL_SIZE}_bscale{B_FACTOR_SCALING}.mrc"
    if not template.is_file():
        raise FileNotFoundError(
            f"{template} -- run run_scripts/simulate_lattice_competition.py first"
        )

    manager = MatchTemplateManager.from_yaml(YAML_CONFIG_PATH)
    manager.template_volume_path = str(template)
    if micrograph:
        manager.micrograph_path = micrograph
    if gpu_ids:
        manager.computational_config.gpu_ids = list(gpu_ids)
    for key in _RESULT_KEYS:
        ext = "h5" if key == "correlation_table_path" else "mrc"
        name = key.replace("_path", "")
        setattr(
            manager.match_template_result,
            key,
            f"{RESULT_DIR}/output_{name}_{tag}.{ext}",
        )
    manager.match_template_result.allow_file_overwrite = True

    constraint = FilamentConstraint.from_yaml(CONSTRAINT_YAML_PATH)
    if constraint.spatial_constraint_path:
        constraint.spatial_constraint_path = _resolve_existing_path(
            constraint.spatial_constraint_path
        )
    constraint.stats_from_valid_orientations_defocus = (
        STATS_FROM_VALID_ORIENTATIONS_DEFOCUS
    )
    manager.apply_filament_constraint(constraint)

    n_orient = int(manager.orientation_search_config.euler_angles.shape[0])
    print(f"\n=== {stem} ===")
    print(f"  template   {template.name}")
    print(f"  gpus       {manager.computational_config.gpu_ids}")
    print(f"  search     {n_orient} orientations (sidecar subsets MIP eligibility)")

    start = time.time()
    manager.run_match_template(
        orientation_batch_size=ORIENTATION_BATCH_SIZE, do_result_export=True
    )
    elapsed = time.time() - start
    print(f"  wall time  {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

    csv_path = MT_ROOT / RESULT_DIR / f"results_{tag}.csv"
    df = manager.results_to_dataframe(locate_peaks_kwargs={"false_positives": 1.0})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=True)
    print(f"  peaks      {len(df)}  -> {csv_path.relative_to(MT_ROOT)}")
    if len(df) and "scaled_mip" in df:
        print(f"  max z      {df['scaled_mip'].max():.3f}")
    return stem, elapsed


def main() -> None:
    """Run every cell of the 2x2 in turn."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native-b",
        action="store_true",
        help="use the native-B templates instead of the B-equalised ones",
    )
    parser.add_argument("--only", help="run a single cell, e.g. 6dpu_atoms_6dpv_lattice")
    parser.add_argument(
        "--stems",
        nargs="+",
        metavar="STEM",
        help="run these template stems instead of the 2x2, e.g. 6dpu_13pf_2rings_flatB",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        metavar="ID",
        help="GPUs to use, overriding the YAML (e.g. --gpus 1 2 3)",
    )
    parser.add_argument(
        "--micrograph",
        help="search this micrograph instead of the real crop, e.g. the synthetic "
        "ground-truth control from make_synthetic_microtubule.py",
    )
    parser.add_argument(
        "--tag",
        default="crop",
        help="suffix for output names, so a control run does not overwrite the real one",
    )
    args = parser.parse_args()

    os.chdir(MT_ROOT)
    suffix = "" if args.native_b else "_flatB"

    if args.stems:
        stems = args.stems
    else:
        cells = CELLS
        if args.only:
            cells = [(a, b) for a, b in CELLS if f"{a}_atoms_{b}_lattice" == args.only]
            if not cells:
                raise SystemExit(f"no cell matches {args.only!r}")
        stems = [f"{a}_atoms_{b}_lattice_2rings{suffix}" for a, b in cells]

    where = f"GPUs {args.gpus}" if args.gpus else "the GPUs in the YAML"
    what = args.micrograph or "the real crop"
    print(f"Competition on {what}: {len(stems)} templates on {where}")
    for stem in stems:
        run_cell(stem, args.gpus, args.micrograph, args.tag)
    print("\nDone. Compare with analyse_lattice_competition.py")


# NOTE: invoking from `if __name__ == "__main__"` is necessary
# for proper multiprocessing/GPU-distribution behavior
if __name__ == "__main__":
    main()
