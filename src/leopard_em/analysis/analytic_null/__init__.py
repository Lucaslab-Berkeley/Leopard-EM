"""Analytic null: predict a search's null peak rate without a null search.

The null correlation field of a 2DTM search on a phase-randomised image is Gaussian,
with a covariance fixed by the template projections and the image power spectrum.
From that covariance, the expected number of distinct null peaks at or above each
z-score per micrograph follows (lattice local maxima, crowding, per-pixel
normalisation), and so does the threshold for a target number of false positives.

* :mod:`.field` and :mod:`.lattice` are engine-agnostic: another search engine can
  supply its own projections and grid neighbours.
* :func:`predict_match_template_null` is the Leopard-EM match_template predictor.
* :mod:`.refine` is the engine-agnostic refined null: the null rate after local
  refinement of a search's candidates, from the refinement gain of a null candidate
  conditioned on what the search established about it.
  :func:`predict_refine_template_null` is the Leopard-EM refine_template predictor.
"""

from .field import (
    AxisNeighbours,
    LatticeSamples,
    MeanTemplate,
    PowerWeighting,
    estimate_kappa,
    estimate_mean_template,
    sample_lattice,
)
from .lattice import (
    AnalyticNullPrediction,
    crossing,
    crowding_correction,
    lattice_maxima_rate,
    nominal_rate,
    normalization_factor,
    predict_null_rate,
)
from .match_template import predict_match_template_null
from .refine import (
    RefinedNullPrediction,
    RefinementGain,
    RefinementWindow,
    ScoreLayout,
    predict_refined_null_rate,
    sample_refinement_gain,
    score_layout,
    window_covariance,
)
from .refine_template import predict_refine_template_null, refine_window

__all__ = [
    "AnalyticNullPrediction",
    "AxisNeighbours",
    "LatticeSamples",
    "MeanTemplate",
    "PowerWeighting",
    "RefinedNullPrediction",
    "RefinementGain",
    "RefinementWindow",
    "ScoreLayout",
    "crossing",
    "crowding_correction",
    "estimate_kappa",
    "estimate_mean_template",
    "lattice_maxima_rate",
    "nominal_rate",
    "normalization_factor",
    "predict_match_template_null",
    "predict_null_rate",
    "predict_refine_template_null",
    "predict_refined_null_rate",
    "refine_window",
    "sample_lattice",
    "sample_refinement_gain",
    "score_layout",
    "window_covariance",
]
