"""Script is used to optimize the template for a given pdb file."""

from leopard_em.pydantic_models.managers import CorrelateFramesManager

CORRELATE_YAML_PATH = "frame_search_example_config.yaml"


def main() -> None:
    """Main function to run the optimize template program."""
    cfm = CorrelateFramesManager.from_yaml(CORRELATE_YAML_PATH)
    cfm.run_correlate_frames(
        output_dataframe_path="results/correlate_frames_results.csv"
    )


if __name__ == "__main__":
    main()
