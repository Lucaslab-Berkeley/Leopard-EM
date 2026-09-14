"""Refine the orientation-aware multi-peak stack from the constrained MT search."""

from leopard_em.pydantic_models.managers import RefineTemplateManager

YAML_CONFIG_PATH = "configs/refine_mt_2rings.yaml"
DATAFRAME_OUTPUT_PATH = "results_cropped/multipeak_2rings_refined.csv"
PARTICLE_BATCH_SIZE = 16


def main() -> None:
    """Run refine_template over the multi-peak particle stack."""
    manager = RefineTemplateManager.from_yaml(YAML_CONFIG_PATH)
    offsets = manager.orientation_refinement_config.euler_angles_offsets
    defocus = manager.defocus_refinement_config.defocus_values
    print(f"orientation offsets {tuple(offsets.shape)}, defocus {tuple(defocus.shape)}")
    manager.run_refine_template(DATAFRAME_OUTPUT_PATH, PARTICLE_BATCH_SIZE)
    print("DONE")


# NOTE: Invoking under __main__ is necessary for multiprocessing
if __name__ == "__main__":
    main()
