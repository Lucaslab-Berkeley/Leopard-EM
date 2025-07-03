"""Fit a General Extreme Value (GEV) distribution to calculate cutoff value."""

import warnings
from typing import Optional

import numpy as np
import torch
from scipy.stats import genextreme
from scipy.stats._distn_infrastructure import rv_frozen

from .match_template_peaks import MatchTemplatePeaks
from .zscore_metric import find_peaks_from_zscore

LARGE_PEAK_WARNING_VALUE = 1000


def fit_gev_to_zscore(
    zscore_map: torch.Tensor,
    min_zscore_value: Optional[float] = None,
    max_zscore_value: Optional[float] = 8.5,
    num_samples: Optional[int] = 1_000_000,
) -> tuple[rv_frozen, tuple[float, float, float]]:
    """Helper function to fit a GEV distribution to the z-score map.

    See `gev_zscore_cutoff` for more details.
    """
    if isinstance(zscore_map, torch.Tensor):
        zscore_map = zscore_map.cpu().numpy()

    # Logic for handling optional parameters
    if min_zscore_value is None:
        min_zscore_value = zscore_map.min().item()
    if max_zscore_value is None:
        max_zscore_value = zscore_map.max().item()
    if num_samples is None or num_samples > zscore_map.size:
        num_samples = zscore_map.size

    # Get flattened and filtered data to fit the GEV distribution
    data = zscore_map.flatten()
    data = data[(data >= min_zscore_value) & (data <= max_zscore_value)]
    if len(data) > num_samples:  # type: ignore
        data = np.random.choice(data, num_samples, replace=False)

    # Fit the parameters of the GEV distribution
    shape, loc, scale = genextreme.fit(data)

    return genextreme(shape, loc=loc, scale=scale), (shape, loc, scale)


def gev_zscore_cutoff(
    zscore_map: torch.Tensor,
    false_positives: Optional[float] = 1.0,
    min_zscore_value: Optional[float] = None,
    max_zscore_value: Optional[float] = 8.5,
    num_samples: Optional[int] = 1_000_000,
) -> float:
    """Calculate the z-score cutoff value by fitting a GEV distn to the z-score map.

    NOTE: This function can take on the order of 10s to 100s of seconds to run when
    there are a large number of pixels in the z-score map. The 'num_samples' parameter
    can be set to fit only using a random subset of the z-score map.

    NOTE: Fitting with ~1,000,000 points seems to sufficiently capture the GEV behavior.
    Your fit results may vary depending on the data; inspecting the quality of your fit
    is recommended.

    NOTE: The 'max_zscore_value' parameter is set to 8.5 by default which performs well
    for a full orientation search (1.5 degrees in-plane and 2.5 degrees out-of-plane).
    Adjusting the search space parameters will require adjustment from the default
    value.

    Parameters
    ----------
    zscore_map: torch.Tensor
        The z-score map to fit the GEV distribution to.
    false_positives: float, optional
        The number of false positives to allow in the image (over all pixels). Default
        is 1.0 which corresponds to a single false-positive.
    min_zscore_value: float, optional
        The minimum z-score value to consider for fitting the GEV distribution. If
        None, the minimum value in the z-score map is used.
    max_zscore_value: float, optional
        The maximum z-score value to consider for fitting the GEV distribution. If
        None, the maximum value in the z-score map is used. Default is 8.5 and all
        values above this are ignored.
    num_samples: int, optional
        The number of samples to use for fitting the GEV distribution. If None, the
        number of samples is set to the number of pixels in the z-score map. The default
        is 1,000,000, and 1 million random pixels are sampled from the z-score map.

    Returns
    -------
    float
        The z-score cutoff value for the GEV distribution.
    """
    if isinstance(zscore_map, torch.Tensor):
        zscore_map = zscore_map.cpu().numpy()

    gev_opt, _ = fit_gev_to_zscore(
        zscore_map,
        min_zscore_value=min_zscore_value,
        max_zscore_value=max_zscore_value,
        num_samples=num_samples,
    )

    # False positive rate of the survival function
    false_positive_density = false_positives / zscore_map.size
    tmp = gev_opt.isf(false_positive_density)

    return float(tmp)


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

    # Raise warning if a very large number of peaks are found
    if len(pos_y) > LARGE_PEAK_WARNING_VALUE:
        warnings.warn(
            f"Found {len(pos_y)} peaks using the fitted GEV distribution. This is a "
            "lot and could indicate a poor fit to the data. You should inspect the fit "
            "before using these results. See the online documentation for details.",
            stacklevel=2,
        )

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
