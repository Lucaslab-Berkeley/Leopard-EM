"""Utility functions associated with backend functions."""

import torch


def normalize_template_projection(
    projections: torch.Tensor,  # shape (batch, h, w)
    small_shape: tuple[int, int],  # (h, w)
    large_shape: tuple[int, int],  # (H, W)
) -> torch.Tensor:
    r"""Subtract mean of edge values and set variance to 1 (in large shape).

    This function uses the fact that variance of a sequence, Var(X), is scaled by the
    relative size of the small (unpadded) and large (padded with zeros) space. Some
    negligible error is introduced into the variance (~1e-4) due to this routine.

    Let $X$ be the large, zero-padded projection and $x$ the small projection each
    with sizes $(H, W)$ and $(h, w)$, respectively. The mean of the zero-padded
    projection in terms of the small projection is:
    .. math::
        \begin{align}
            \mu(X) &= \frac{1}{H \cdot W} \sum_{i=1}^{H} \sum_{j=1}^{W} X_{ij} \\
            \mu(X) &= \frac{1}{H \cdot W} \sum_{i=1}^{h} \sum_{j=1}^{w} X_{ij} + 0 \\
            \mu(X) &= \frac{h \cdot w}{H \cdot W} \mu(x)
        \end{align}
    The variance of the zero-padded projection in terms of the small projection can be
    obtained by:
    .. math::
        \begin{align}
            Var(X) &= \frac{1}{H \cdot W} \sum_{i=1}^{H} \sum_{j=1}^{W} (X_{ij} -
                \mu(X))^2 \\
            Var(X) &= \frac{1}{H \cdot W} \left(\sum_{i=1}^{h}
                \sum_{j=1}^{w} (X_{ij} - \mu(X))^2 +
                \sum_{i=h+1}^{H}\sum_{i=w+1}^{W} \mu(X)^2 \right) \\
            Var(X) &= \frac{1}{H \cdot W} \sum_{i=1}^{h} \sum_{j=1}^{w} (X_{ij} -
                \mu(X))^2 + (H-h)(W-w)\mu(X)^2
        \end{align}

    Parameters
    ----------
    projections : torch.Tensor
        Real-space projections of the template (in small space).
    small_shape : tuple[int, int]
        Shape of the template.
    large_shape : tuple[int, int]
        Shape of the image (in large space).

    Returns
    -------
    torch.Tensor
        Edge-mean subtracted projections, still in small space, but normalized
        so variance of zero-padded projection is 1.
    """
    h, w = small_shape
    H, W = large_shape

    # Extract edges while preserving batch dimensions
    top_edge = projections[..., 0, :]  # shape: (..., w)
    bottom_edge = projections[..., -1, :]  # shape: (..., w)
    left_edge = projections[..., 1:-1, 0]  # shape: (..., h-2)
    right_edge = projections[..., 1:-1, -1]  # shape: (..., h-2)
    edge_pixels = torch.concatenate(
        [top_edge, bottom_edge, left_edge, right_edge], dim=-1
    )

    # Subtract the edge pixel mean and calculate variance of small, unpadded projection
    edge_mean = edge_pixels.mean(dim=-1)
    projections -= edge_mean[..., None, None]

    # # Calculate variance like cisTEM (does not match desired results...)
    # variance = (projections**2).sum(dim=(-1, -2), keepdim=True) * relative_size - (
    #     projections.mean(dim=(-1, -2), keepdim=True) * relative_size
    # ) ** 2

    # Fast calculation of mean/var using Torch + appropriate scaling.
    relative_size = h * w / (H * W)
    mean = torch.mean(projections, dim=(-2, -1), keepdim=True) * relative_size
    mean *= relative_size

    # First term of the variance calculation
    variance = torch.sum((projections - mean) ** 2, dim=(-2, -1), keepdim=True)
    # Add the second term of the variance calculation
    variance += (H - h) * (W - w) * mean**2
    variance /= H * W

    return projections / torch.sqrt(variance)


def do_iteration_statistics_updates(
    cross_correlation: torch.Tensor,
    euler_angles: torch.Tensor,
    defocus_values: torch.Tensor,
    pixel_values: torch.Tensor,
    mip: torch.Tensor,
    best_phi: torch.Tensor,
    best_theta: torch.Tensor,
    best_psi: torch.Tensor,
    best_defocus: torch.Tensor,
    best_pixel_size: torch.Tensor,
    correlation_sum: torch.Tensor,
    correlation_squared_sum: torch.Tensor,
    H: int,
    W: int,
) -> None:
    """Helper function for updating maxima and tracked statistics.

    NOTE: The batch dimensions are effectively unraveled since taking the
    maximum over a single batch dimensions is much faster than
    multi-dimensional maxima.

    NOTE: Updating the maxima was found to be fastest and least memory
    impactful when using torch.where directly. Other methods tested were
    boolean masking and torch.where with tuples of tensor indexes.

    Parameters
    ----------
    cross_correlation : torch.Tensor
        Cross-correlation values for the current iteration. Has either shape
        (batch, H, W) or (defocus, orientations, H, W).
    euler_angles : torch.Tensor
        Euler angles for the current iteration. Has shape (orientations, 3).
    defocus_values : torch.Tensor
        Defocus values for the current iteration. Has shape (defocus,).
    pixel_values : torch.Tensor
        Pixel size values for the current iteration. Has shape (pixel_size_batch,).
    mip : torch.Tensor
        Maximum intensity projection of the cross-correlation values.
    best_phi : torch.Tensor
        Best phi angle for each pixel.
    best_theta : torch.Tensor
        Best theta angle for each pixel.
    best_psi : torch.Tensor
        Best psi angle for each pixel.
    best_defocus : torch.Tensor
        Best defocus value for each pixel.
    best_pixel_size : torch.Tensor
        Best pixel size value for each pixel.
    correlation_sum : torch.Tensor
        Sum of cross-correlation values for each pixel.
    correlation_squared_sum : torch.Tensor
        Sum of squared cross-correlation values for each pixel.
    H : int
        Height of the cross-correlation values.
    W : int
        Width of the cross-correlation values.
    """
    nCs, nDefoc, nAng = cross_correlation.shape[0:3]
    max_values, max_indices = torch.max(cross_correlation.view(-1, H, W), dim=0)
    max_cs_idx = (max_indices // (nDefoc * nAng)) % nCs
    max_defocus_idx = (max_indices // nAng) % nDefoc
    max_orientation_idx = max_indices % nAng

    # using torch.where directly
    update_mask = max_values > mip

    torch.where(update_mask, max_values, mip, out=mip)
    torch.where(
        update_mask, euler_angles[max_orientation_idx, 0], best_phi, out=best_phi
    )
    torch.where(
        update_mask, euler_angles[max_orientation_idx, 1], best_theta, out=best_theta
    )
    torch.where(
        update_mask, euler_angles[max_orientation_idx, 2], best_psi, out=best_psi
    )
    torch.where(
        update_mask, defocus_values[max_defocus_idx], best_defocus, out=best_defocus
    )
    torch.where(
        update_mask, pixel_values[max_cs_idx], best_pixel_size, out=best_pixel_size
    )

    correlation_sum += cross_correlation.view(-1, H, W).sum(dim=0)
    correlation_squared_sum += (cross_correlation.view(-1, H, W) ** 2).sum(dim=0)
