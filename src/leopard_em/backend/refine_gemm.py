"""GEMM-based scoring of refine-template hypotheses.

The FFT refine engine pads every (orientation x defocus) projection to the particle box,
correlates it with the particle by FFT, and then keeps only a small crop of the
correlogram. This module computes the same normalized cross-correlation directly at the
integer shifts that are needed, using matrix multiplications over Fourier coefficients.

Math
----
For one particle, let ``X_o`` be the (un-fftshifted, DC-zeroed, contrast-flipped)
Fourier slice for orientation ``o``, ``T_f`` the combined projective filter * CTF for
filter ``f``, and ``G_s = rfft2(fftshift(patch_s))`` the DFT of the template-sized
patch of the particle image at shift ``s``. With RFFT Hermitian weights ``W`` and
``hw = h * w``, the steps of the FFT engine reduce to (Parseval):

    raw[o, f, s] = Re sum_k X_o(k) * W_k T_f(k) conj(G_s(k)) / hw     <patch_s, p_of>
    edge[o, f]   = Re sum_k X_o(k) * W_k T_f(k) conj(E(k))   / hw     edge mean of p_of
    power[o, f]  =    sum_k |X_o(k)|^2 * W_k T_f(k)^2        / hw     sum of p_of^2

where ``p_of`` is the real-space projection and ``E`` the DFT of the edge-pixel mask
used by :func:`leopard_em.backend.utils.normalize_template_projection`. The bracketed
factors do not depend on the orientation, so each line is one real GEMM over the
``2 * h * (w // 2 + 1)`` interleaved (real, imag) coefficients. The projection
normalization (edge-mean subtraction, padded variance, zeroed DC of the padded
projection) only adds a closed-form scale and offset per (orientation, filter), see
:func:`normalize_gemm_correlation`. The result matches the FFT engine to float32
round-off.

One caveat: the Parseval identities assume each projection spectrum is Hermitian in
its zero and Nyquist columns. Slices of a zero-padded template are (the crop goes
through real space); slices of an unpadded template are only approximately, and the
inverse real FFT of the FFT engine handles the non-Hermitian residual in a
library-dependent way (cuFFT and pocketfft differ). For band-limited templates the two
backends still agree to ~1e-5 relative.

Every function here is plain PyTorch and differentiable. The windowed max search in
:func:`gemm_search_particle` runs without gradients; gradients of the refined scores are
taken by re-scoring the winning hypothesis (see :func:`score_hypotheses`).
"""

# pylint: disable=E1102

import math
from collections.abc import Callable, Iterator
from typing import NamedTuple

import torch

from leopard_em.utils.fourier_slice import extract_template_fourier_slices

# Bytes allowed for the per-particle shift operands (``b_raw``) built at once. Windows
# needing more are split into shift chunks.
_SHIFT_OPERAND_BYTES = 1 << 30
# Bytes allowed for caching all shift-chunk operands of one particle. Beyond this the
# operands are rebuilt for every orientation chunk instead.
_SHIFT_OPERAND_CACHE_BYTES = 4 << 30
# Bytes allowed for caching the slice operands of one particle between window
# re-centering attempts.
_SLICE_CACHE_BYTES = 2 << 30

SliceOperandFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def rfft_hermitian_weights(
    shape: tuple[int, int], device: torch.device | None = None
) -> torch.Tensor:
    """Weights so a full-spectrum sum equals a weighted RFFT-spectrum sum.

    For real images ``a`` and ``b``, ``sum_full conj(A) B`` equals
    ``sum_rfft W * Re(conj(A) B)``.

    Parameters
    ----------
    shape : tuple[int, int]
        Real-space shape (h, w).
    device : torch.device | None, optional
        Device of the returned tensor.

    Returns
    -------
    torch.Tensor
        Weights with shape (h, w // 2 + 1).
    """
    h, w = shape
    weights = torch.full((h, w // 2 + 1), 2.0, device=device)
    weights[:, 0] = 1.0
    if w % 2 == 0:
        weights[:, -1] = 1.0
    return weights


def edge_mask_rfft(
    shape: tuple[int, int], device: torch.device | None = None
) -> torch.Tensor:
    """RFFT of the fftshifted edge-mean mask of ``normalize_template_projection``.

    Parameters
    ----------
    shape : tuple[int, int]
        Real-space template shape (h, w).
    device : torch.device | None, optional
        Device of the returned tensor.

    Returns
    -------
    torch.Tensor
        Complex tensor with shape (h, w // 2 + 1).
    """
    h, w = shape
    mask = torch.zeros((h, w), device=device)
    mask[0, :] = 1.0
    mask[-1, :] = 1.0
    mask[1:-1, 0] = 1.0
    mask[1:-1, -1] = 1.0
    mask /= mask.sum()
    mask_dft: torch.Tensor = torch.fft.rfftn(
        torch.fft.fftshift(mask, dim=(-2, -1)), dim=(-2, -1)
    )
    return mask_dft


def prepare_slices_for_scoring(
    fourier_slices: torch.Tensor, rows_centered: bool = True
) -> torch.Tensor:
    """Turn template Fourier slices into the slices the correlation uses.

    Applies the same steps as the FFT engine: un-fftshift the rows, zero the DC term,
    and flip the contrast. Out-of-place, so it is safe under autograd.

    Parameters
    ----------
    fourier_slices : torch.Tensor
        RFFT slices with shape (..., h, w // 2 + 1).
    rows_centered : bool, optional
        True if the zero frequency is centered along dim -2 (the layout returned by
        :func:`extract_template_fourier_slices`), False if it is already at index 0.

    Returns
    -------
    torch.Tensor
        Slices with the zero frequency at [0, 0], DC zeroed, and contrast flipped.
    """
    if rows_centered:
        fourier_slices = torch.fft.ifftshift(fourier_slices, dim=(-2,))
    keep = torch.ones(fourier_slices.shape[-2:], device=fourier_slices.device)
    keep[0, 0] = 0.0
    return fourier_slices * (-keep)  # zero the DC and flip contrast


def slices_to_operands(slices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """GEMM operands for a stack of prepared slices.

    Parameters
    ----------
    slices : torch.Tensor
        Complex slices from :func:`prepare_slices_for_scoring`, shape (O, h, wr).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``x_cat`` with shape (O, 2 * h * wr), the interleaved (real, imag) values, and
        ``x_pow`` with shape (O, h * wr), the squared magnitudes.
    """
    num = slices.shape[0]
    slices = slices.contiguous()
    x_cat = torch.view_as_real(slices).reshape(num, -1)
    x_pow = (slices.real**2 + slices.imag**2).reshape(num, -1)
    return x_cat, x_pow


def _interleaved(z: torch.Tensor) -> torch.Tensor:
    """Complex (..., h, wr) -> real (..., 2hwr) so <x_cat, out> = Re(x conj(z))."""
    z = z.contiguous()
    return torch.view_as_real(z).reshape(*z.shape[:-2], -1)


def cut_patches(
    images: torch.Tensor, shifts: torch.Tensor, template_shape: tuple[int, int]
) -> torch.Tensor:
    """Cut template-sized patches out of real-space images.

    Parameters
    ----------
    images : torch.Tensor
        Real-space images, shape (H, W) or (N, H, W).
    shifts : torch.Tensor
        Integer top-left (y, x) corners in the 'valid' correlation frame. Shape (S, 2)
        for a single image, or (N, 2) for one patch per image.
    template_shape : tuple[int, int]
        Patch shape (h, w).

    Returns
    -------
    torch.Tensor
        Patches with shape (S, h, w) or (N, h, w).
    """
    h, w = template_shape
    shifts = shifts.to(images.device, torch.long)
    rows = shifts[:, 0, None] + torch.arange(h, device=images.device)
    cols = shifts[:, 1, None] + torch.arange(w, device=images.device)
    if images.dim() == 2:
        return images[rows[:, :, None], cols[:, None, :]]
    batch = torch.arange(images.shape[0], device=images.device)[:, None, None]
    return images[batch, rows[:, :, None], cols[:, None, :]]


def normalize_gemm_correlation(
    raw: torch.Tensor,
    edge: torch.Tensor,
    power: torch.Tensor,
    patch_sums: torch.Tensor,
    image_sum: torch.Tensor,
    template_shape: tuple[int, int],
    image_shape: tuple[int, int],
) -> torch.Tensor:
    """Normalized cross-correlation from the raw GEMM functionals.

    Reproduces :func:`leopard_em.backend.utils.normalize_template_projection` followed
    by the zero-padded forward FFT with the DC term zeroed, as in the FFT engine.

    Parameters
    ----------
    raw : torch.Tensor
        Patch / projection inner products, shape (..., S).
    edge : torch.Tensor
        Edge means of the projections, shape (...).
    power : torch.Tensor
        Sum of squares of the projections, shape (...).
    patch_sums : torch.Tensor
        Sums of the image patches, broadcastable to ``raw``.
    image_sum : torch.Tensor
        Sum of the whole particle image, broadcastable to ``edge``.
    template_shape : tuple[int, int]
        Template (projection) shape (h, w).
    image_shape : tuple[int, int]
        Particle image shape (H, W).

    Returns
    -------
    torch.Tensor
        Normalized cross-correlation with the shape of ``raw``.
    """
    h, w = template_shape
    big_h, big_w = image_shape
    hw, big_hw = h * w, big_h * big_w
    rel = hw / big_hw

    # Edge-mean subtraction. The projection sums to zero (DC zeroed), so its mean after
    # subtraction is -edge. 'normalize_template_projection' scales that mean by the
    # relative size twice; replicated here so the results match.
    mean = -edge * rel * rel
    variance = power + hw * (edge + mean) ** 2 + (big_h - h) * (big_w - w) * mean**2
    scale = math.sqrt(big_hw) / torch.sqrt(variance.clamp_min(1e-8))

    # Zeroing the DC term of the padded projection subtracts its mean over the box
    padded_mean = scale * (-hw * edge) / big_hw
    cc = scale[..., None] * (raw - edge[..., None] * patch_sums)
    return cc - (padded_mean * image_sum)[..., None]


class FilterOperands(NamedTuple):
    """Shift-independent GEMM operands for one particle."""

    weighted_filters: torch.Tensor  # (F, h, wr), W * T / hw
    b_edge: torch.Tensor  # (2K, F)
    b_pow: torch.Tensor  # (K, F)


class ShiftOperands(NamedTuple):
    """GEMM operands for one chunk of shifts of one particle."""

    b_raw: torch.Tensor  # (2K, F * S)
    patch_sums: torch.Tensor  # (S,)


def build_filter_operands(
    filters: torch.Tensor, template_shape: tuple[int, int]
) -> FilterOperands:
    """Build the shift-independent operands for one particle's filters.

    Parameters
    ----------
    filters : torch.Tensor
        Combined projective filter * CTF, shape (F, h, w // 2 + 1).
    template_shape : tuple[int, int]
        Template shape (h, w).

    Returns
    -------
    FilterOperands
        Operands for the edge and power functionals.
    """
    h, w = template_shape
    num_filters = filters.shape[0]
    weights = rfft_hermitian_weights(template_shape, filters.device) / (h * w)
    weighted = weights * filters
    edge = edge_mask_rfft(template_shape, filters.device)
    b_edge = _interleaved(weighted * edge).T  # conj(W T conj(E)) = W T E
    b_pow = (weighted * filters).reshape(num_filters, -1).T
    return FilterOperands(weighted, b_edge, b_pow)


def build_shift_operands(
    image: torch.Tensor,
    shifts: torch.Tensor,
    filter_operands: FilterOperands,
    template_shape: tuple[int, int],
) -> ShiftOperands:
    """Build the correlation operand for a chunk of shifts.

    Parameters
    ----------
    image : torch.Tensor
        Real-space particle image, shape (H, W).
    shifts : torch.Tensor
        Integer (y, x) shifts in the 'valid' frame, shape (S, 2).
    filter_operands : FilterOperands
        Output of :func:`build_filter_operands`.
    template_shape : tuple[int, int]
        Template shape (h, w).

    Returns
    -------
    ShiftOperands
        The correlation operand and patch sums.
    """
    patches = cut_patches(image, shifts, template_shape)
    patch_dft = torch.fft.rfftn(torch.fft.fftshift(patches, dim=(-2, -1)), dim=(-2, -1))
    weighted = filter_operands.weighted_filters
    num_filters, num_shifts = weighted.shape[0], patches.shape[0]
    # conj(W T conj(G)) = W T G, shape (F, S, h, wr) -> (2K, F * S)
    b_raw = _interleaved(weighted[:, None] * patch_dft[None])
    b_raw = b_raw.reshape(num_filters * num_shifts, -1).T
    return ShiftOperands(b_raw, patches.sum(dim=(-2, -1)))


class GemmSearchResult(NamedTuple):
    """Best hypothesis for one particle (indices into the searched grids)."""

    z_score: float
    cross_correlation: float
    angle_idx: int
    filter_idx: int
    pos_y: int
    pos_x: int


def _window(
    center: tuple[int, int],
    radius: int | None,
    crop_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Inclusive (y_lo, y_hi, x_lo, x_hi) of a window kept inside the crop."""
    crop_h, crop_w = crop_shape
    if radius is None:
        return 0, crop_h - 1, 0, crop_w - 1
    bounds = []
    for c, n in zip(center, crop_shape, strict=False):
        half = min(radius, (n - 1) // 2)
        c = min(max(c, half), n - 1 - half)  # shift the window to stay inside the crop
        bounds += [c - half, c + half]
    return bounds[0], bounds[1], bounds[2], bounds[3]


def _window_shifts(
    bounds: tuple[int, int, int, int], device: torch.device
) -> torch.Tensor:
    y_lo, y_hi, x_lo, x_hi = bounds
    ys = torch.arange(y_lo, y_hi + 1, device=device)
    xs = torch.arange(x_lo, x_hi + 1, device=device)
    return torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1).reshape(-1, 2)


def _on_inner_edge(
    pos: tuple[int, int],
    bounds: tuple[int, int, int, int],
    crop_shape: tuple[int, int],
) -> bool:
    """True if ``pos`` is on a window edge that is not also a crop edge."""
    y, x = pos
    y_lo, y_hi, x_lo, x_hi = bounds
    crop_h, crop_w = crop_shape
    on_top = y == y_lo and y_lo != 0
    on_bottom = y == y_hi and y_hi != crop_h - 1
    on_left = x == x_lo and x_lo != 0
    on_right = x == x_hi and x_hi != crop_w - 1
    return on_top or on_bottom or on_left or on_right


# pylint: disable=too-many-arguments,too-many-locals
@torch.no_grad()
def _search_window(
    slice_chunks: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor, int]]],
    shifts: torch.Tensor,
    image: torch.Tensor,
    image_sum: torch.Tensor,
    filter_operands: FilterOperands,
    corr_mean: torch.Tensor,
    corr_std: torch.Tensor,
    template_shape: tuple[int, int],
) -> GemmSearchResult:
    """Max z-score over (orientation, filter, shift) for one window of shifts."""
    image_shape = (image.shape[-2], image.shape[-1])
    num_filters = filter_operands.weighted_filters.shape[0]
    num_k2 = filter_operands.b_edge.shape[0]
    shift_chunk = max(1, _SHIFT_OPERAND_BYTES // (num_k2 * num_filters * 4))
    starts = list(range(0, shifts.shape[0], shift_chunk))
    cache_shift_ops = (
        num_k2 * num_filters * shifts.shape[0] * 4 <= _SHIFT_OPERAND_CACHE_BYTES
    )
    shift_ops: dict[int, ShiftOperands] = {}

    mean_s = corr_mean[shifts[:, 0], shifts[:, 1]]
    std_s = corr_std[shifts[:, 0], shifts[:, 1]]

    best_z = torch.tensor(-torch.inf, device=image.device)
    best_cc = torch.tensor(0.0, device=image.device)
    best_idx = torch.zeros(3, dtype=torch.long, device=image.device)  # (o, f, s)
    for x_cat, x_pow, o_start in slice_chunks():
        edge = x_cat @ filter_operands.b_edge  # (O, F)
        power = x_pow @ filter_operands.b_pow  # (O, F)
        for s_start in starts:
            s_stop = min(s_start + shift_chunk, shifts.shape[0])
            ops = shift_ops.get(s_start)
            if ops is None:
                ops = build_shift_operands(
                    image, shifts[s_start:s_stop], filter_operands, template_shape
                )
                if cache_shift_ops:
                    shift_ops[s_start] = ops
            raw = (x_cat @ ops.b_raw).reshape(x_cat.shape[0], num_filters, -1)
            cc = normalize_gemm_correlation(
                raw, edge, power, ops.patch_sums, image_sum, template_shape, image_shape
            )
            z = (cc - mean_s[s_start:s_stop]) / std_s[s_start:s_stop]
            z_max, arg = z.reshape(-1).max(dim=0)
            better = z_max > best_z
            o_i, f_i, s_i = torch.unravel_index(arg, z.shape)
            idx = torch.stack([o_i + o_start, f_i, s_i + s_start])
            best_z = torch.where(better, z_max, best_z)
            best_cc = torch.where(better, cc.reshape(-1)[arg], best_cc)
            best_idx = torch.where(better, idx, best_idx)

    o_best, f_best, s_best = best_idx.tolist()
    pos_y, pos_x = shifts[s_best].tolist()
    return GemmSearchResult(
        z_score=float(best_z),
        cross_correlation=float(best_cc),
        angle_idx=o_best,
        filter_idx=f_best,
        pos_y=pos_y,
        pos_x=pos_x,
    )


# pylint: disable=too-many-arguments,too-many-locals
@torch.no_grad()
def gemm_search_particle(
    slice_operands: SliceOperandFn,
    rotation_matrices: torch.Tensor,
    *,
    image: torch.Tensor,
    image_sum: torch.Tensor,
    filters: torch.Tensor,
    corr_mean: torch.Tensor,
    corr_std: torch.Tensor,
    center: tuple[int, int],
    shift_search_radius: int | None,
    orientation_batch_size: int,
    max_recenters: int = 2,
) -> GemmSearchResult:
    """Max z-score search over orientation, filter, and shift for one particle.

    Shifts are searched in a ``(2r + 1) x (2r + 1)`` window around ``center``. If the
    best shift lands on the window edge (and that edge is not the crop edge), the
    window is re-centered on it, up to ``max_recenters`` times, after which the whole
    valid crop is searched.

    Parameters
    ----------
    slice_operands : Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        Maps rotation matrices (O, 3, 3) to the slice operands ``(x_cat, x_pow)``,
        see :func:`slices_to_operands`.
    rotation_matrices : torch.Tensor
        Orientations to search, shape (O, 3, 3).
    image : torch.Tensor
        Real-space particle image, shape (H, W).
    image_sum : torch.Tensor
        Sum of the particle image (the DC term of its DFT).
    filters : torch.Tensor
        Combined projective filter * CTF, shape (F, h, w // 2 + 1).
    corr_mean : torch.Tensor
        Correlation mean over the valid crop, shape (H - h + 1, W - w + 1).
    corr_std : torch.Tensor
        Correlation standard deviation over the valid crop, same shape.
    center : tuple[int, int]
        Starting window center (y, x) in the valid crop.
    shift_search_radius : int | None
        Window half-width ``r`` in pixels. None searches the whole valid crop.
    orientation_batch_size : int
        Number of orientations per GEMM chunk.
    max_recenters : int, optional
        Maximum number of window re-centerings before falling back to the whole valid
        crop. Default is 2.

    Returns
    -------
    GemmSearchResult
        The best z-score, the cross-correlation at that hypothesis, and its indices.
    """
    crop_shape = (int(corr_mean.shape[-2]), int(corr_mean.shape[-1]))
    template_shape = (
        int(filters.shape[-2]),
        2 * (int(filters.shape[-1]) - 1),
    )
    filter_operands = build_filter_operands(filters, template_shape)

    num_orient = rotation_matrices.shape[0]
    num_k = filters.shape[-2] * filters.shape[-1]
    can_recenter = shift_search_radius is not None and max_recenters > 0
    cache_slices = can_recenter and num_orient * num_k * 12 <= _SLICE_CACHE_BYTES
    cached: list[tuple[torch.Tensor, torch.Tensor, int]] = []

    def slice_chunks() -> Iterator[tuple[torch.Tensor, torch.Tensor, int]]:
        if cached:
            yield from cached
            return
        for start in range(0, num_orient, orientation_batch_size):
            x_cat, x_pow = slice_operands(
                rotation_matrices[start : start + orientation_batch_size]
            )
            if cache_slices:
                cached.append((x_cat, x_pow, start))
            yield x_cat, x_pow, start

    def search(bounds: tuple[int, int, int, int]) -> GemmSearchResult:
        return _search_window(
            slice_chunks,
            _window_shifts(bounds, image.device),
            image,
            image_sum,
            filter_operands,
            corr_mean,
            corr_std,
            template_shape,
        )

    for _ in range(max_recenters + 1):
        bounds = _window(center, shift_search_radius, crop_shape)
        result = search(bounds)
        if shift_search_radius is None or not _on_inner_edge(
            (result.pos_y, result.pos_x), bounds, crop_shape
        ):
            return result
        center = (result.pos_y, result.pos_x)

    # Still on a window edge after re-centering: search the whole valid crop
    return search(_window(center, None, crop_shape))


def reference_slice_operands(
    template_dft: torch.Tensor,
    projection_shape: tuple[int, int],
    mag_matrix: torch.Tensor | None = None,
) -> SliceOperandFn:
    """Slice operand function using ``torch_fourier_slice`` (differentiable).

    Parameters
    ----------
    template_dft : torch.Tensor
        Template volume prepared by ``volume_to_rfft_fourier_slice`` (may be padded).
    projection_shape : tuple[int, int]
        Real-space template projection shape (h, w).
    mag_matrix : torch.Tensor | None, optional
        Anisotropic magnification matrix of shape (2, 2). Default is None.

    Returns
    -------
    Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        Function mapping rotation matrices (O, 3, 3) to ``(x_cat, x_pow)``.
    """

    def operands(rotation_matrices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        slices = extract_template_fourier_slices(
            template_dft=template_dft,
            rotation_matrices=rotation_matrices,
            projection_shape=projection_shape,
            mag_matrix=mag_matrix,
        )
        return slices_to_operands(prepare_slices_for_scoring(slices))

    return operands


# pylint: disable=too-many-arguments,too-many-locals
def score_hypotheses(
    template_dft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    filters: torch.Tensor,
    particle_images_dft: torch.Tensor,
    positions: torch.Tensor,
    mag_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalized cross-correlation of one hypothesis per particle (differentiable).

    Equals the FFT engine's correlation of particle ``n`` with the template at
    orientation ``rotation_matrices[n]``, filter ``filters[n]`` and valid-crop position
    ``positions[n]``. Gradients flow to the template, rotations, filters, and images.

    Parameters
    ----------
    template_dft : torch.Tensor
        Template volume prepared by ``volume_to_rfft_fourier_slice`` (may be padded).
    rotation_matrices : torch.Tensor
        One rotation matrix per particle, shape (N, 3, 3).
    filters : torch.Tensor
        One combined projective filter * CTF per particle, shape (N, h, w // 2 + 1).
    particle_images_dft : torch.Tensor
        RFFT of the particle images, shape (N, H, W // 2 + 1).
    positions : torch.Tensor
        Integer (y, x) positions in the valid crop, shape (N, 2).
    mag_matrix : torch.Tensor | None, optional
        Anisotropic magnification matrix of shape (2, 2). Default is None.

    Returns
    -------
    torch.Tensor
        Cross-correlation values, shape (N,).
    """
    template_shape = (int(filters.shape[-2]), 2 * (int(filters.shape[-1]) - 1))
    image_shape = (
        int(particle_images_dft.shape[-2]),
        2 * (int(particle_images_dft.shape[-1]) - 1),
    )
    h, w = template_shape

    slices = extract_template_fourier_slices(
        template_dft=template_dft,
        rotation_matrices=rotation_matrices,
        projection_shape=template_shape,
        mag_matrix=mag_matrix,
    )
    slices = prepare_slices_for_scoring(slices)

    images = torch.fft.irfftn(particle_images_dft, s=image_shape, dim=(-2, -1))
    image_sums = particle_images_dft[:, 0, 0].real
    patches = cut_patches(images, positions, template_shape)
    patch_dft = torch.fft.rfftn(torch.fft.fftshift(patches, dim=(-2, -1)), dim=(-2, -1))

    weighted = rfft_hermitian_weights(template_shape, filters.device) / (h * w)
    weighted = weighted * filters
    edge_dft = edge_mask_rfft(template_shape, filters.device)

    raw = (slices * weighted * patch_dft.conj()).real.sum(dim=(-2, -1))
    edge = (slices * weighted * edge_dft.conj()).real.sum(dim=(-2, -1))
    power = ((slices.real**2 + slices.imag**2) * weighted * filters).sum(dim=(-2, -1))

    cc = normalize_gemm_correlation(
        raw[:, None],
        edge,
        power,
        patches.sum(dim=(-2, -1))[:, None],
        image_sums,
        template_shape,
        image_shape,
    )
    return cc[:, 0]
