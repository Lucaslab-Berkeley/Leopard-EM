"""Constrained match template for the cropped GMPCPP test micrograph.

Uses ``configs/match_tm_crop_patch4.yaml`` as the FFT / mean-variance search
grid, and the napari sidecar at ``configs/filament_constraint_box.yaml`` to
subset which (pixel, orientation) tuples may win the MIP. The sidecar Euler
box constrains ``psi`` (in-plane filament angle) and ``theta``; ``phi`` stays
free.

Run from anywhere; this script cds into ``MT_tests`` so the crop YAML's
relative ``Frames/`` and ``maps/`` paths resolve.
"""

import os
import time
from pathlib import Path

from leopard_em.pydantic_models.config import FilamentConstraint
from leopard_em.pydantic_models.managers import MatchTemplateManager

MT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MT_ROOT.parent

YAML_CONFIG_PATH = str(MT_ROOT / "configs" / "match_tm_full_patch4.yaml")
CONSTRAINT_YAML_PATH = str(MT_ROOT / "configs" / "filament_constraint_full.yaml")
DATAFRAME_OUTPUT_PATH = str(
    MT_ROOT / "results_cropped" / "results_patch4_full.csv"
)
ORIENTATION_BATCH_SIZE = 8
STATS_FROM_VALID_ORIENTATIONS_DEFOCUS = False

_RESULT_PATHS = {
    "mip_path": "results_full/output_mip_patch4_full.mrc",
    "scaled_mip_path": "results_full/output_scaled_mip_patch4_full.mrc",
    "orientation_psi_path": (
        "results_full/output_orientation_psi_patch4_full.mrc"
    ),
    "orientation_theta_path": (
        "results_full/output_orientation_theta_patch4_full.mrc"
    ),
    "orientation_phi_path": (
        "results_full/output_orientation_phi_patch4_full.mrc"
    ),
    "relative_defocus_path": (
        "results_full/output_relative_defocus_patch4_full.mrc"
    ),
    "correlation_average_path": (
        "results_full/output_correlation_average_patch4_full.mrc"
    ),
    "correlation_variance_path": (
        "results_full/output_correlation_variance_patch4_full.mrc"
    ),
    "correlation_table_path": (
        "results_full/output_correlation_table_patch4_full.h5"
    ),
}


def _resolve_existing_path(path: str) -> str:
    """Find a sidecar path exported from either MT_tests or the repo root."""
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate)
    for root in (Path.cwd(), MT_ROOT, REPO_ROOT):
        alt = root / path
        if alt.is_file():
            return str(alt)
        named = root / "configs" / Path(path).name
        if named.is_file():
            return str(named)
        nested = root / "MT_tests" / "configs" / Path(path).name
        if nested.is_file():
            return str(nested)
    return path


def main() -> None:
    """Run constrained match template on the cropped GMPCPP micrograph."""
    os.chdir(MT_ROOT)

    mt_manager = MatchTemplateManager.from_yaml(YAML_CONFIG_PATH)
    for attr, path in _RESULT_PATHS.items():
        setattr(mt_manager.match_template_result, attr, path)

    constraint = FilamentConstraint.from_yaml(CONSTRAINT_YAML_PATH)
    if constraint.spatial_constraint_path:
        constraint.spatial_constraint_path = _resolve_existing_path(
            constraint.spatial_constraint_path
        )
    constraint.stats_from_valid_orientations_defocus = (
        STATS_FROM_VALID_ORIENTATIONS_DEFOCUS
    )
    mt_manager.apply_filament_constraint(constraint)

    n_orient = int(mt_manager.orientation_search_config.euler_angles.shape[0])
    print(constraint.preview_text())
    print(
        f"YAML search: {n_orient} orientations "
        "(sidecar subsets MIP eligibility)."
    )
    if constraint.spatial_constraint_path:
        print(f"Spatial constraint: {constraint.spatial_constraint_path}")
    if STATS_FROM_VALID_ORIENTATIONS_DEFOCUS:
        print(
            "Mean/variance: allowed (pixel, orientation, defocus) tuples "
            "only (per-pixel count)."
        )
    else:
        print(
            "Mean/variance: all orientations and defocus searched in this "
            "run (MIP still restricted to allowed tuples)."
        )

    print("Loaded configuration.")
    print("Running match template...")

    start_time = time.time()
    mt_manager.run_match_template(
        orientation_batch_size=ORIENTATION_BATCH_SIZE,
        do_result_export=True,
    )
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print("Finished core match_template call.")
    print(f"Match Template wall time: {elapsed}")

    print("Exporting results...")
    df = mt_manager.results_to_dataframe(
        locate_peaks_kwargs={"false_positives": 1.0}
    )
    Path(DATAFRAME_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATAFRAME_OUTPUT_PATH, index=True)
    print(f"Wrote {DATAFRAME_OUTPUT_PATH}")
    print("Done!")


# NOTE: invoking from `if __name__ == "__main__"` is necessary
# for proper multiprocessing/GPU-distribution behavior
if __name__ == "__main__":
    main()
