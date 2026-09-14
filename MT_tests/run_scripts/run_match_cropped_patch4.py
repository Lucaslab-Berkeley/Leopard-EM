from leopard_em.pydantic_models.managers import MatchTemplateManager

YAML_CONFIG_PATH = "configs/match_tm_crop_patch4.yaml"
ORIENTATION_BATCH_SIZE = 8

def main():
    mt_manager = MatchTemplateManager.from_yaml(YAML_CONFIG_PATH)
    mt_manager.run_match_template(ORIENTATION_BATCH_SIZE)
    df = mt_manager.results_to_dataframe(locate_peaks_kwargs={"false_positives": 1.0})
    df.to_csv("results_cropped/results_match_template_GMPCPP_patch4.csv")

# NOTE: invoking from `if __name__ == "__main__"` is necessary
# for proper multiprocessing/GPU-distribution behavior
if __name__ == "__main__":
    main()
