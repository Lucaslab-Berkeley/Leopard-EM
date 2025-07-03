"""Runs pixel size optimization for a given template structure."""

from leopard_em.pydantic_models.managers import OptimizeTemplateManager

OPTIMIZE_YAML_PATH = "optimize_template_example_config.yaml"


def main() -> None:
    """Main function to run the optimize template program."""
    otm = OptimizeTemplateManager.from_yaml(OPTIMIZE_YAML_PATH)
    otm.run_optimize_template(output_text_path="results/optimize_template_results.txt")


if __name__ == "__main__":
    main()
