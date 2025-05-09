"""Fit a General Extreme Value (GEV) distribution to calculate cutoff value."""

import warnings
from typing import Any

import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.stats import genextreme

from .match_template_peaks import MatchTemplatePeaks
from .zscore_metric import find_peaks_from_zscore


def gev_zscore_cutoff(zscore_map: torch.Tensor, false_positives: float = 1.0) -> float:
    """Calculate the z-score cutoff value based on a fitted GEV distribution."""
    (shape, loc, scale), _, _ = fit_gev_distribution(zscore_map)
    gev_opt = genextreme(shape, loc=loc, scale=scale)

    # False positive rate based on the number of search positions
    num_search_positions = zscore_map.numel()
    false_positive_rate = false_positives / num_search_positions

    # Calculate the z-score cutoff value
    zscore_cutoff = gev_opt.ppf(1.0 - false_positive_rate)

    return float(zscore_cutoff)


def genextreme_fit_function(
    x: np.ndarray, shape: float, loc: float, scale: float
) -> np.ndarray:
    """Helper function to evaluate when fitting GEV distn. to data histogram.

    Parameters
    ----------
    x: np.ndarray
        The coordinates to evaluate the GEV distribution at.
    shape: float
        The shape parameter of the GEV distribution.
    loc: float
        The location parameter of the GEV distribution.
    scale: float
        The scale parameter of the GEV distribution.

    Returns
    -------
    np.ndarray
        The evaluated GEV distribution at the given coordinates.
    """
    return genextreme.pdf(x, shape, loc=loc, scale=scale)


def fit_gev_distribution(
    zscore_map: torch.Tensor | np.ndarray,
    num_bins: int = 128,
) -> tuple[tuple[float, float, float], np.ndarray, dict[str, Any]]:
    """Fit a General Extreme Value (GEV) distribution to the z-score map.

    This function effectively acts as a wrapper around the scipy.optimize.curve_fit
    function to fit the GEV distribution to the histogram of z-scores. It returns
    the fitted parameters, the covariance matrix, and additional information about
    the fit. See scipy.optimize.curve_fit for more details.

    For more advanced usage (e.g. imposing user-defined bounds), consider
    re-implementing this function in your own codebase.

    Parameters
    ----------
    zscore_map : torch.Tensor | np.ndarray
        Input tensor/array containing z-score values. Can be 2D.
    num_bins : int, optional
        Optional number of bins to use for histogram. Default is 128.

    Returns
    -------
    popt : tuple
        Fitted parameters of the GEV distribution (shape, loc, scale).
    pcov : 2D array
        Covariance matrix of the fitted parameters.
    infodict : dict
        Dictionary containing additional information about the fit.
    """
    if isinstance(zscore_map, torch.Tensor):
        zscore_map = zscore_map.cpu().numpy()

    # Generate a histogram of the flattened z-score map
    counts, bin_edges = np.histogram(zscore_map.flatten(), bins=num_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Initial guess and bounds for the parameters
    initial_guess = (0.0, 5.0, 0.2)
    bounds = ([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf])

    popt, pcov, infodict = curve_fit(
        genextreme_fit_function,
        bin_centers,
        counts,
        p0=initial_guess,
        bounds=bounds,
        maxfev=10000,
        full_output=True,
    )

    return popt, pcov, infodict


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=duplicate-code
def extract_peaks_and_statistics_gev(
    mip: torch.Tensor,
    scaled_mip: torch.Tensor,
    best_psi: torch.Tensor,
    best_theta: torch.Tensor,
    best_phi: torch.Tensor,
    best_defocus: torch.Tensor,
    correlation_average: torch.Tensor,
    correlation_variance: torch.Tensor,
    total_correlation_positions: int,
    num_bins: int = 128,
    false_positives: float = 1.0,
    mask_radius: float = 5.0,
) -> MatchTemplatePeaks:
    """Returns peak locations, heights, and pose stats from match template results.

    Parameters
    ----------
    mip : torch.Tensor
        Maximum intensity projection of the match template results.
    scaled_mip : torch.Tensor
        Scaled maximum intensity projection of the match template results.
    best_psi : torch.Tensor
        Best psi angles for each pixel.
    best_theta : torch.Tensor
        Best theta angles for each pixel.
    best_phi : torch.Tensor
        Best phi angles for each pixel.
    best_defocus : torch.Tensor
        Best relative defocus values for each pixel.
    correlation_average : torch.Tensor
        Average correlation value for each pixel.
    correlation_variance : torch.Tensor
        Variance of the correlation values for each pixel.
    total_correlation_positions : int
        Total number of correlation positions calculated during template matching. Must
        be provided if `z_score_cutoff` is not provided (needed for the noise model).
    num_bins : int, optional
        Number of bins to use for histogram when fitting GEV distribution. Default is
        128.
    false_positives : float, optional
        Number of false positives to allow in the image (over all pixels). Default is
        1.0 which corresponds to a single false-positive.
    mask_radius : float, optional
        Radius of the mask to apply around the peak, in units of pixels. Default is 5.0.

    Returns
    -------
    MatchTemplatePeaks
        Named tuple containing the peak locations, heights, and pose statistics.
    """
    z_score_cutoff = gev_zscore_cutoff(scaled_mip, false_positives=false_positives)

    # Find the peak locations only in the scaled MIP
    pos_y, pos_x = find_peaks_from_zscore(scaled_mip, z_score_cutoff, mask_radius)

    # Raise warning if no peaks are found
    if len(pos_y) == 0:
        warnings.warn("No peaks found using z-score metric.", stacklevel=2)

    # Extract peak heights, orientations, etc. from other maps
    return MatchTemplatePeaks(
        pos_y=pos_y,
        pos_x=pos_x,
        mip=mip[pos_y, pos_x],
        scaled_mip=scaled_mip[pos_y, pos_x],
        psi=best_psi[pos_y, pos_x],
        theta=best_theta[pos_y, pos_x],
        phi=best_phi[pos_y, pos_x],
        relative_defocus=best_defocus[pos_y, pos_x],
        correlation_mean=correlation_average[pos_y, pos_x],
        correlation_variance=correlation_variance[pos_y, pos_x],
        total_correlations=total_correlation_positions,
    )
