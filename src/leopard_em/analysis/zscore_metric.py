"""Locates peaks in the scaled mip from a match template result."""

import warnings
from typing import Optional

import numpy as np
import torch
from scipy.special import erfcinv  # pylint: disable=no-name-in-module

from .match_template_peaks import MatchTemplatePeaks
from .utils import filter_peaks_by_distance


def gaussian_noise_zscore_cutoff(num_ccg: int, false_positives: float = 1.0) -> float:
    """Determines the z-score cutoff based on Gaussian noise model and number of pixels.

    NOTE: This procedure assumes that the z-scores (normalized maximum intensity
    projections) are distributed according to a standard normal distribution. Here,
    this model is used to find the cutoff value such that there is at most
    'false_positives' number of false positives in all of the pixels.

    Parameters
    ----------
    num_ccg : int
        Total number of cross-correlograms calculated during template matching. Product
        of the number of pixels, number of defocus values, and number of orientations.
    false_positives : float, optional
        Number of false positives to allow in the image (over all pixels). Default is
        1.0 which corresponds to a single false-positive.

    Returns
    -------
    float
        Z-score cutoff.
    """
    tmp = erfcinv(2.0 * false_positives / num_ccg)
    tmp *= np.sqrt(2.0)

    return float(tmp)


def find_peaks_from_zscore(
    zscore_map: torch.Tensor,
    zscore_cutoff: float,
    mask_radius: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Find peaks in a z-score map above a cutoff threshold using torch.

    The function returns a tensor of peak indices sorted in descending order by
    their z-score values. Peaks closer than mask_radius to an already picked peak
    are suppressed.

    Parameters
    ----------
    zscore_map : torch.Tensor
        Input tensor containing z-score values.
    zscore_cutoff : float
        Minimum z-score value to consider as a peak.
    mask_radius : float, optional
        Minimum allowed distance between peaks, default is 5.0.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Two tensors containing the y and x coordinates of the peaks.
    """
    # Find indices where zscore_map is above the cutoff
    peaks = torch.nonzero(zscore_map > zscore_cutoff, as_tuple=False)

    if peaks.shape[0] == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    # Retrieve the zscore values for these indices and sort descending
    peak_values = zscore_map[tuple(peaks.t())]

    picked_peaks = filter_peaks_by_distance(
        peak_values=peak_values,
        peak_locations=peaks,
        distance_threshold=mask_radius,
    )

    return picked_peaks[:, 0], picked_peaks[:, 1]


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=duplicate-code
def extract_peaks_and_statistics_zscore(
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
    z_score_cutoff: Optional[float] = None,
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
    z_score_cutoff : float, optional
        Z-score cutoff value for peak detection. If not provided, it is calculated using
        the Gaussian noise model. Default is None.
    mask_radius : float, optional
        Radius of the mask to apply around the peak, in units of pixels. Default is 5.0.

    Returns
    -------
    MatchTemplatePeaks
        Named tuple containing the peak locations, heights, and pose statistics.
    """
    if z_score_cutoff is None:
        z_score_cutoff = gaussian_noise_zscore_cutoff(
            num_ccg=mip.numel() * total_correlation_positions,
            false_positives=false_positives,
        )

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
