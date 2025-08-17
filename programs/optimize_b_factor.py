"""Script is used to optimize the B-factor for a given match template."""

import pandas as pd

from leopard_em.pydantic_models.managers import MatchTemplateManager

# The config file for the match template
MATCH_YAML_PATH = "match_template_manager_example_crop.yaml"
# the B-factor range to optimize over
B_MIN = 0
B_MAX = 200
B_STEP = 10
# the optimize metric can be:
# all: optimize mean SNR of peaks
# best: optimize mean SNR of best n peaks
# worst: optimize mean SNR of worst n peaks
# count: Maximise the number of peaks
OPTIMIZE_METRIC = "mean"
# the number of peaks to optimize, -1 for all
OPTIMIZE_N = -1


def get_metric(df: pd.DataFrame) -> float:
    """Get the optimization metric for the given dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to get the metric for.

    Returns
    -------
    float
        The metric for the given dataframe.
    """
    subtract_background = df["scaled_mip"]
    if OPTIMIZE_METRIC == "mean":
        return float(subtract_background.mean())
    if OPTIMIZE_METRIC == "best":
        return float(subtract_background.nlargest(OPTIMIZE_N).mean())
    if OPTIMIZE_METRIC == "worst":
        return float(subtract_background.nsmallest(OPTIMIZE_N).mean())
    if OPTIMIZE_METRIC == "count":
        return float(subtract_background.count())

    raise ValueError(f"Invalid optimize metric: {OPTIMIZE_METRIC}")


def main() -> None:
    """Main function to run the optimize b-factor program."""
    b_values = list(range(B_MIN, B_MAX + B_STEP, B_STEP))
    best_metric = -float("inf")
    best_b = 0
    consecutive_decreases = 0
    previous_metric = float("-inf")

    for b in b_values:
        mtm = MatchTemplateManager.from_yaml(MATCH_YAML_PATH)
        mtm.optics_group.ctf_B_factor = b
        mtm.run_match_template(
            orientation_batch_size=16, do_result_export=False, do_valid_cropping=False
        )
        df = mtm.results_to_dataframe()
        metric = get_metric(df)
        print(f"B-factor: {b}, Metric: {metric}")
        # Write results to CSV
        with open("optimize_B_results.csv", "a") as f:
            if b == b_values[0]:  # Write header for first iteration
                f.write(f"b_factor,{OPTIMIZE_METRIC}\n")
            f.write(f"{b},{metric}\n")
        if metric > best_metric:
            best_metric = metric
            best_b = b
        if metric > previous_metric:
            consecutive_decreases = 0
        else:
            consecutive_decreases += 1
            if consecutive_decreases >= 2:
                print(
                    "Metric decreased for two consecutive iterations. "
                    "Stopping B-factor search."
                )
                break

        previous_metric = metric
    print(f"Best B-factor: {best_b} with a {OPTIMIZE_METRIC} of {best_metric}")
    with open("optimize_B_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Best B-factor: {best_b} with a {OPTIMIZE_METRIC} of {best_metric}")


if __name__ == "__main__":
    main()
