"""Submodule for analyzing results during the template matching pipeline."""

from .analytic_null import (
    AnalyticNullPrediction,
    RefinedNullPrediction,
    predict_match_template_null,
    predict_refine_template_null,
)
from .empirical_null import (
    DistinctPeaks,
    NullPeaks,
    NullRateCurve,
    distinct_peaks,
    empirical_zscore_cutoff,
)
from .inspect_peaks_result import (
    InspectionResult,
    load_inspection_result,
    save_inspection_result,
)
from .match_template_peaks import (
    MatchTemplatePeaks,
    match_template_peaks_to_dataframe,
    match_template_peaks_to_dict,
)
from .pvalue_metric import extract_peaks_and_statistics_p_value
from .zscore_metric import (
    extract_peaks_and_statistics_zscore,
    gaussian_noise_zscore_cutoff,
)

__all__ = [
    "AnalyticNullPrediction",
    "DistinctPeaks",
    "InspectionResult",
    "MatchTemplatePeaks",
    "NullPeaks",
    "NullRateCurve",
    "RefinedNullPrediction",
    "distinct_peaks",
    "empirical_zscore_cutoff",
    "extract_peaks_and_statistics_p_value",
    "extract_peaks_and_statistics_zscore",
    "gaussian_noise_zscore_cutoff",
    "load_inspection_result",
    "match_template_peaks_to_dataframe",
    "match_template_peaks_to_dict",
    "predict_match_template_null",
    "predict_refine_template_null",
    "save_inspection_result",
]
