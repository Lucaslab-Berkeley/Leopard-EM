"""Submodule for analyzing results during the template matching pipeline."""

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
    "InspectionResult",
    "MatchTemplatePeaks",
    "extract_peaks_and_statistics_p_value",
    "extract_peaks_and_statistics_zscore",
    "gaussian_noise_zscore_cutoff",
    "load_inspection_result",
    "match_template_peaks_to_dataframe",
    "match_template_peaks_to_dict",
    "save_inspection_result",
]
