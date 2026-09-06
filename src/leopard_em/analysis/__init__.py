"""Submodule for analyzing results during the template matching pipeline."""

from .correlation_peaks import (
    CLUSTER_STATISTIC_COLUMNS,
    constrained_zscore_cutoff,
    find_peaks_orientation_aware,
    peaks_to_match_template_dataframe,
)
from .filament_lattice import (
    FilamentAxis,
    PolarityEstimate,
    TemplateLatticeGeometry,
    axis_points_from_peaks,
    estimate_lattice_rise,
    estimate_polarity,
    estimate_protofilament_number,
    filament_coordinates,
    fit_filament_axis,
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
    "CLUSTER_STATISTIC_COLUMNS",
    "FilamentAxis",
    "InspectionResult",
    "MatchTemplatePeaks",
    "PolarityEstimate",
    "TemplateLatticeGeometry",
    "axis_points_from_peaks",
    "constrained_zscore_cutoff",
    "estimate_lattice_rise",
    "estimate_polarity",
    "estimate_protofilament_number",
    "extract_peaks_and_statistics_p_value",
    "extract_peaks_and_statistics_zscore",
    "filament_coordinates",
    "find_peaks_orientation_aware",
    "fit_filament_axis",
    "gaussian_noise_zscore_cutoff",
    "load_inspection_result",
    "match_template_peaks_to_dataframe",
    "match_template_peaks_to_dict",
    "peaks_to_match_template_dataframe",
    "save_inspection_result",
]
