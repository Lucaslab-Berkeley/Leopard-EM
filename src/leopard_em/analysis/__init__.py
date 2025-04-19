"""Submodule for analyzing results during the template matching pipeline."""

from .match_template_peaks import (
    MatchTemplatePeaks,
    match_template_peaks_to_dataframe,
    match_template_peaks_to_dict,
)
from .pvalue_metric import extract_peaks_and_statistics_p_value
from .zscore_metric import extract_peaks_and_statistics_zscore, gaussian_noise_zscore_cutoff

__all__ = [
    "MatchTemplatePeaks",
    "match_template_peaks_to_dict",
    "match_template_peaks_to_dataframe",
    "extract_peaks_and_statistics_p_value",
    "extract_peaks_and_statistics_zscore",
    "gaussian_noise_zscore_cutoff",
]
