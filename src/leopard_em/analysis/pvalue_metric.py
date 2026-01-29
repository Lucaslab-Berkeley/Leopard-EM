"""
P-value based peak detection using an anisotropic Gaussian model.

This module implements a faithful, vectorized Torch rewrite of the reference
anisotropic-Gaussian p-value metric described in:

    https://journals.iucr.org/m/issues/2025/02/00/eh5020/eh5020.pdf

The implementation matches the original NumPy reference exactly:
- Rank-based probit transform
- Moment-based anisotropic Gaussian estimation
- Explicit eigen decomposition and rotation
- Analytic p-value with quadrant constraints
"""

import math
import warnings

import torch
from scipy.special import erfinv  # pylint: disable=no-name-in-module

from .match_template_peaks import MatchTemplatePeaks
from .utils import filter_peaks_by_distance


def probit_transform_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Apply a rank-based probit transform to a 1D tensor.

    This function converts the empirical rank of each element into a quantile
    of the standard normal distribution using the inverse error function.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of arbitrary shape. The tensor is flattened internally.

    Returns
    -------
    torch.Tensor
        A 1D tensor containing the probit-transformed values.
    """
    x_flat = x.flatten()
    n = x_flat.numel()

    # Rank data (1-based indexing)
    ranks = torch.argsort(torch.argsort(x_flat)) + 1
    u = (ranks.float() - 0.5) / max(1, n)

    # Convert quantiles to standard normal scores
    probit = torch.sqrt(torch.tensor(2.0, device=x.device)) * torch.from_numpy(
        erfinv((2.0 * u - 1.0).cpu().numpy())
    ).to(x.device)

    return probit


def estimate_anisotropic_gaussian(
    pro_x1: torch.Tensor,
    pro_x2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate an anisotropic Gaussian using second-order moments.

    This function computes the eigenvalues and eigenvectors of the moment
    matrix ⟨xxᵀ⟩, where x = [pro_x1, pro_x2]. The result defines the orientation
    and anisotropy of the Gaussian distribution without fitting or mean
    subtraction.

    Parameters
    ----------
    pro_x1 : torch.Tensor
        Probit-transformed values of the first variable.
    pro_x2 : torch.Tensor
        Probit-transformed values of the second variable.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Eigenvalues and eigenvectors of the moment matrix.
    """
    tmp = torch.stack([pro_x1, pro_x2])
    c_inv = (tmp @ tmp.T) / pro_x1.numel()

    d_inv, u_inv = torch.linalg.eig(c_inv)  # pylint: disable=not-callable

    return d_inv.real, u_inv.real


# ============================================================
# Vectorized anisotropic p-value
# ============================================================


def anisotropic_pvalue(  # pylint: disable=too-many-locals
    x1: torch.Tensor,
    x2: torch.Tensor,
    d_inv: torch.Tensor,
    u_inv: torch.Tensor,
    quadrant: int = 1,
) -> torch.Tensor:
    """
    Compute anisotropic Gaussian p-values with quadrant constraints.

    This function evaluates an analytic p-value based on the radial distance
    in a rotated and scaled coordinate system defined by the anisotropic
    Gaussian.

    Parameters
    ----------
    x1 : torch.Tensor
        Probit-transformed values along the first dimension.
    x2 : torch.Tensor
        Probit-transformed values along the second dimension.
    d_inv : torch.Tensor
        Eigenvalues of the moment matrix.
    u_inv : torch.Tensor
        Eigenvectors of the moment matrix.
    quadrant : int, optional
        Quadrant constraint to apply:
        - 1: First quadrant only (x1 > 0 and x2 > 0)
        - 3: Three quadrants (x1 > 0 or x2 > 0)

    Returns
    -------
    torch.Tensor
        Array of negative log p-values.
    """
    # Sort eigenvalues to identify major/minor axes
    a = torch.sqrt(torch.max(d_inv))
    b = torch.sqrt(torch.min(d_inv))

    if d_inv[0] <= d_inv[1]:
        u_inv = u_inv[:, [1, 0]]

    # Ensure consistent orientation
    if (u_inv[1, 0] < 0) and (u_inv[0, 0] < 0):
        u_inv[:, 0] *= -1

    # Rotation angle
    w = torch.atan2(u_inv[1, 0], u_inv[0, 0])

    # Angular correction factor
    gamma = torch.atan2(
        torch.tensor(1.0, device=x1.device),
        0.5 * torch.sin(2 * w) * (b / a - a / b),
    )

    # Rotate coordinates
    cosw = torch.cos(w)
    sinw = torch.sin(w)

    x0 = cosw * x1 + sinw * x2
    x1r = -sinw * x1 + cosw * x2

    # Scale by anisotropy
    y0 = x0 / torch.clamp(a, min=1e-12)
    y1 = x1r / torch.clamp(b, min=1e-12)

    r2 = y0**2 + y1**2

    # Apply quadrant constraint
    if quadrant == 1:
        mask = (x1 > 0) & (x2 > 0)
    elif quadrant == 3:
        mask = (x1 > 0) | (x2 > 0)
    else:
        raise ValueError("quadrant must be 1 or 3")

    pvals = torch.ones_like(x1)

    pvals[mask] = torch.exp(-0.5 * r2[mask]) * gamma / (2 * math.pi)

    return -torch.log(pvals)


def find_peaks_from_pvalue(  # pylint: disable=too-many-locals
    mip: torch.Tensor,
    scaled_mip: torch.Tensor,
    p_value_cutoff: float = 0.01,
    mask_radius: float = 5.0,
    quadrant: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Identify peak locations using the anisotropic Gaussian p-value metric.

    Parameters
    ----------
    mip : torch.Tensor
        Maximum intensity projection of the match template results.
    scaled_mip : torch.Tensor
        Z-score scaled maximum intensity projection.
    p_value_cutoff : float, optional
        Threshold on p-values for peak detection.
    mask_radius : float, optional
        Minimum distance between detected peaks.
    quadrant : int, optional
        Quadrant constraint used in p-value calculation.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Y and X coordinates of detected peaks.
    """
    device = mip.device

    # Flatten inputs
    mip_flat = mip.flatten()
    z_flat = scaled_mip.flatten()

    # Probit transform
    pro_z = probit_transform_torch(z_flat)
    pro_m = probit_transform_torch(mip_flat)

    # Estimate anisotropic Gaussian
    d_inv, u_inv = estimate_anisotropic_gaussian(pro_z, pro_m)

    # Compute p-values
    neg_log_p = anisotropic_pvalue(pro_z, pro_m, d_inv, u_inv, quadrant=quadrant)

    pvals = neg_log_p.reshape(mip.shape)

    # Select peaks
    peaks = torch.nonzero(pvals > p_value_cutoff, as_tuple=False)
    peak_vals = pvals[tuple(peaks.t())]

    if peaks.numel() == 0:
        return (
            torch.tensor([], dtype=torch.long, device=device),
            torch.tensor([], dtype=torch.long, device=device),
        )

    peaks = filter_peaks_by_distance(
        peak_values=peak_vals,
        peak_locations=peaks,
        distance_threshold=mask_radius,
    )

    return peaks[:, 0], peaks[:, 1]


def extract_peaks_and_statistics_p_value(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    mip: torch.Tensor,
    scaled_mip: torch.Tensor,
    best_psi: torch.Tensor,
    best_theta: torch.Tensor,
    best_phi: torch.Tensor,
    best_defocus: torch.Tensor,
    correlation_average: torch.Tensor,
    correlation_variance: torch.Tensor,
    total_correlation_positions: int,
    p_value_cutoff: float = 0.01,
    mask_radius: float = 5.0,
    quadrant: int = 1,
) -> MatchTemplatePeaks:
    """
    Extract peak locations and associated statistics using the p-value metric.

    Parameters
    ----------
    mip : torch.Tensor
        Maximum intensity projection of the match template results.
    scaled_mip : torch.Tensor
        Z-score scaled maximum intensity projection.
    best_psi : torch.Tensor
        Best psi angles per pixel.
    best_theta : torch.Tensor
        Best theta angles per pixel.
    best_phi : torch.Tensor
        Best phi angles per pixel.
    best_defocus : torch.Tensor
        Best relative defocus values per pixel.
    correlation_average : torch.Tensor
        Mean correlation values per pixel.
    correlation_variance : torch.Tensor
        Variance of correlation values per pixel.
    total_correlation_positions : int
        Total number of correlation positions evaluated.
    p_value_cutoff : float, optional
        P-value threshold for peak detection.
    mask_radius : float, optional
        Radius for peak masking.
    quadrant : int, optional
        Quadrant constraint used in p-value calculation.
        - 1: First quadrant only (x1 > 0 and x2 > 0)
        - 3: Three quadrants (x1 > 0 or x2 > 0)
        Default is 1.

    Returns
    -------
    MatchTemplatePeaks
        Named tuple containing peak locations and associated statistics.
    """
    pos_y, pos_x = find_peaks_from_pvalue(
        mip=mip,
        scaled_mip=scaled_mip,
        p_value_cutoff=p_value_cutoff,
        mask_radius=mask_radius,
        quadrant=quadrant,
    )

    if len(pos_y) == 0:
        warnings.warn("No peaks found using p-value metric.", stacklevel=2)

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
