"""Constrained match_template on the curved synthetic microtubule.

The constraint is the CONTINUOUS one: a per-pixel ``psi_center`` field built from the
known curve by ``programs/constrained_search/make_filament_constraint.py``. A single
global Euler box cannot describe this tube -- its psi sweeps about 17 degrees end to end,
which is well outside any one +/-10 degree cone.

As with every other run here, the YAML holds the full uniform SO(3) grid and restricts no
angle; the sidecar only gates which (pixel, orientation) tuples may win the MIP, and
``STATS_FROM_VALID_ORIENTATIONS_DEFOCUS = False`` keeps mean and variance over the whole
grid so the z-scores stay comparable to an unconstrained search.

Usage:
    python run_scripts/run_match_curved.py --tag truth_13pf_6dpv
    python run_scripts/run_match_curved.py --tag 14pf_6dpv --gpus 1 2 3
"""

import argparse
import os
import time
from pathlib import Path

from leopard_em.pydantic_models.config import FilamentConstraint
from leopard_em.pydantic_models.managers import MatchTemplateManager

MT_ROOT = Path(__file__).resolve().parents[1]
CONSTRAINT_YAML = MT_ROOT / "configs" / "filament_constraint_curved_sim.yaml"
ORIENTATION_BATCH_SIZE = 8
STATS_FROM_VALID_ORIENTATIONS_DEFOCUS = False


def main() -> None:
    """Run one constrained search on the curved synthetic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="truth_13pf_6dpv",
                        help="which configs/match_tm_<prefix><tag>.yaml to run")
    parser.add_argument("--prefix", default="curved_",
                        help="config name prefix; '' for a bare tag")
    parser.add_argument("--results", default="results_curved",
                        help="directory the correlation table is written to")
    parser.add_argument("--gpus", type=int, nargs="+", default=None)
    parser.add_argument("--constraint", type=Path, default=CONSTRAINT_YAML)
    args = parser.parse_args()

    os.chdir(MT_ROOT)
    config_path = MT_ROOT / "configs" / f"match_tm_{args.prefix}{args.tag}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    manager = MatchTemplateManager.from_yaml(str(config_path))
    if args.gpus:
        manager.computational_config.gpu_ids = list(args.gpus)
    table = f"{args.results}/output_correlation_table_{args.prefix}{args.tag}.h5"
    (MT_ROOT / args.results).mkdir(parents=True, exist_ok=True)
    manager.match_template_result.correlation_table_path = table

    constraint = FilamentConstraint.from_yaml(str(args.constraint))
    constraint.stats_from_valid_orientations_defocus = (
        STATS_FROM_VALID_ORIENTATIONS_DEFOCUS
    )
    manager.apply_filament_constraint(constraint)

    n_orient = int(manager.orientation_search_config.euler_angles.shape[0])
    print(constraint.preview_text())
    print(f"template {Path(manager.template_volume_path).name}")
    print(f"YAML search: {n_orient:,} orientations (sidecar subsets MIP eligibility)")
    print(f"GPUs: {manager.computational_config.gpu_ids}")

    start = time.time()
    manager.run_match_template(
        orientation_batch_size=ORIENTATION_BATCH_SIZE, do_result_export=True
    )
    print(f"wall time {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")

    frame = manager.results_to_dataframe(locate_peaks_kwargs={"false_positives": 1.0})
    out = MT_ROOT / args.results / f"results_{args.prefix}{args.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=True)
    print(f"wrote {out}  ({len(frame)} peaks)")


# NOTE: invoking under __main__ is required for the multiprocessing/GPU distribution.
if __name__ == "__main__":
    main()
