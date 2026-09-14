"""Time the constrained search at two grid sizes to extrapolate the full-grid cost."""

import sys
import time

from leopard_em.pydantic_models.config.orientation_search import OrientationSearchConfig
from leopard_em.pydantic_models.managers import MatchTemplateManager


def main() -> None:
    """Time two small grids and extrapolate."""
    yaml_path, = sys.argv[1:]
    timings = []
    for psi_step, theta_step in ((30.0, 30.0), (16.0, 16.0)):
        manager = MatchTemplateManager.from_yaml(yaml_path)
        manager.orientation_search_config = OrientationSearchConfig(
            base_grid_method="uniform", psi_step=psi_step, theta_step=theta_step
        )
        n = manager.orientation_search_config.euler_angles.shape[0]
        start = time.time()
        manager.run_match_template(orientation_batch_size=8, do_result_export=False)
        elapsed = time.time() - start
        timings.append((n, elapsed))
        print(f"  {n:7d} orientations -> {elapsed:7.1f} s", flush=True)

    (n1, t1), (n2, t2) = timings
    per = (t2 - t1) / (n2 - n1)
    fixed = t1 - per * n1
    full = fixed + per * 485856
    print(f"\nfixed overhead {fixed:.0f} s; {per * 1000:.2f} s per 1000 orientations")
    print(f"EXTRAPOLATED full 485,856-orientation grid: {full / 60:.0f} min")


if __name__ == "__main__":
    main()
