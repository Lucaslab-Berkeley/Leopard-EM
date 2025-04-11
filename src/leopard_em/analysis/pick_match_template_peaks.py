"""Locates peaks in the scaled mip from a match template result."""

from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
import torch
from scipy.special import erfcinv  # pylint: disable=no-name-in-module


class MatchTemplatePeaks(NamedTuple):
    """Helper class for return value of extract_peaks_and_statistics."""

    pos_y: torch.Tensor
    pos_x: torch.Tensor
    mip: torch.Tensor
    scaled_mip: torch.Tensor
    psi: torch.Tensor
    theta: torch.Tensor
    phi: torch.Tensor
    relative_defocus: torch.Tensor
    correlation_mean: torch.Tensor
    correlation_variance: torch.Tensor
    total_correlations: int


def match_template_peaks_to_dict(peaks: MatchTemplatePeaks) -> dict:
    """Convert MatchTemplatePeaks object to a dictionary."""
    return peaks._asdict()


def match_template_peaks_to_dataframe(peaks: MatchTemplatePeaks) -> pd.DataFrame:
    """Convert MatchTemplatePeaks object to a pandas DataFrame."""
    return pd.DataFrame(peaks._asdict())


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


def find_peaks_in_zscore(
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
    _, sort_indices = torch.sort(peak_values, descending=True)
    peaks = peaks[sort_indices]

    # Create a boolean mask to record which peaks are taken
    taken_mask = torch.zeros(peaks.size(0), dtype=torch.bool, device=peaks.device)
    picked_peaks = torch.tensor([], dtype=torch.long, device=peaks.device)

    for i in range(peaks.size(0)):
        if taken_mask[i]:
            continue

        picked_peaks = torch.cat((picked_peaks, peaks[i].unsqueeze(0)), dim=0)

        # Compute distances between current peak and all peaks
        distances = torch.norm(peaks - peaks[i].float(), dim=1)

        # Mark all peaks closer than mask_radius as taken
        taken_mask |= distances < mask_radius

    return picked_peaks[:, 0], picked_peaks[:, 1]


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def extract_peaks_and_statistics(
    mip: torch.Tensor,
    scaled_mip: torch.Tensor,
    best_psi: torch.Tensor,
    best_theta: torch.Tensor,
    best_phi: torch.Tensor,
    best_defocus: torch.Tensor,
    correlation_average: torch.Tensor,
    correlation_variance: torch.Tensor,
    total_correlation_positions: int,
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
            mip.numel() * total_correlation_positions
        )

    # Find the peak locations only in the scaled MIP
    pos_y, pos_x = find_peaks_in_zscore(scaled_mip, z_score_cutoff, mask_radius)

    # rase error if no peaks are found
    if len(pos_y) == 0:
        raise ValueError("No peaks found in scaled MIP.")

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
