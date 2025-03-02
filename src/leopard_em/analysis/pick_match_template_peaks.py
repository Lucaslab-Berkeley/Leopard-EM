"""Locates peaks in the scaled mip from a match template result."""

from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
import torch
from scipy.special import erfcinv


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
    refined_relative_defocus: torch.Tensor
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
    zscore_map: torch.Tensor, zscore_cutoff: float, mask_radius: Optional[float] = 5.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finds locations of peaks above a threshold using masking around each found peak.

    Parameters
    ----------
    zscore_map : torch.Tensor
        2D tensor of z-scores.
    zscore_cutoff : float
        Z-score cutoff value.
    mask_radius : float, optional
        Radius of the circular mask to apply around the peak. Default is 5.0.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tensors corresponding to the x and y coordinates of the peaks, respectively.
    """
    # Short circuit if the cutoff is too high
    if zscore_cutoff > zscore_map.max():
        return torch.tensor([]), torch.tensor([])

    zscore_map_copy = zscore_map.clone()
    H, W = zscore_map.shape

    # Convert to next highest integer
    mask_width = np.ceil(mask_radius).astype(int) * 2 + 1
    mask = torch.ones(mask_width, mask_width)
    x = torch.arange(mask_width) - mask_width // 2
    y = torch.arange(mask_width) - mask_width // 2
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    dist = torch.sqrt(xx**2 + yy**2)
    mask[dist <= mask_radius] = 0.0
    mask_radius = np.ceil(mask_radius).astype(int)

    found_peaks_x = []
    found_peaks_y = []
    # Iteratively find the highest peak in the map, and then mask the surrounding region
    while zscore_map_copy.max() >= zscore_cutoff:
        peak_loc = torch.argmax(zscore_map_copy)
        peak_loc = torch.unravel_index(peak_loc, zscore_map_copy.shape)
        peak_x, peak_y = peak_loc
        found_peaks_x.append(peak_x)
        found_peaks_y.append(peak_y)

        # Mask the region around the peak, taking into account image bounds
        start_x = max(0, peak_x - mask_radius)
        end_x = min(H, peak_x + mask_radius + 1)
        start_y = max(0, peak_y - mask_radius)
        end_y = min(W, peak_y + mask_radius + 1)

        # Calculate the valid range of the mask to apply
        mask_start_x = max(0, mask_radius - peak_x)
        mask_end_x = mask_width - max(0, (peak_x + mask_radius + 1) - H)
        mask_start_y = max(0, mask_radius - peak_y)
        mask_end_y = mask_width - max(0, (peak_y + mask_radius + 1) - W)

        zscore_map_copy[start_x:end_x, start_y:end_y] *= mask[
            mask_start_x:mask_end_x, mask_start_y:mask_end_y
        ]

    found_peaks_x = torch.tensor(found_peaks_x)
    found_peaks_y = torch.tensor(found_peaks_y)

    return found_peaks_x, found_peaks_y


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
    mask_radius: Optional[float] = 5.0,
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
    mip_peaks = mip[pos_y, pos_x]
    scaled_mip_peaks = scaled_mip[pos_y, pos_x]
    psi_peaks = best_psi[pos_y, pos_x]
    theta_peaks = best_theta[pos_y, pos_x]
    phi_peaks = best_phi[pos_y, pos_x]
    relative_defocus_peaks = best_defocus[pos_y, pos_x]
    refined_relative_defocus_peaks = relative_defocus_peaks
    correlation_average_peaks = correlation_average[pos_y, pos_x]
    correlation_variance_peaks = correlation_variance[pos_y, pos_x]

    return MatchTemplatePeaks(
        pos_y=pos_y,
        pos_x=pos_x,
        mip=mip_peaks,
        scaled_mip=scaled_mip_peaks,
        psi=psi_peaks,
        theta=theta_peaks,
        phi=phi_peaks,
        relative_defocus=relative_defocus_peaks,
        refined_relative_defocus=refined_relative_defocus_peaks,
        correlation_mean=correlation_average_peaks,
        correlation_variance=correlation_variance_peaks,
        total_correlations=total_correlation_positions,
    )
