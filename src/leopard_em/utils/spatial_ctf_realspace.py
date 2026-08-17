"""Real-space PSF grids from ``torch_ctf`` and spatially varying convolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import roma
import torch
import torch.nn.functional as F
from torch_ctf import calc_LPP_ctf_2D, calculate_ctf_2d
from torch_fourier_filter.envelopes import b_envelope
from torch_fourier_slice import extract_central_slices_rfft_3d

from leopard_em.utils.search_utils import get_cs_range

if TYPE_CHECKING:
    from leopard_em.pydantic_models.data_structures.optics_group import OpticsGroup


def _nominal_cs_mm(optics: OpticsGroup, device: torch.device) -> float:
    cs_vals = get_cs_range(
        pixel_size=optics.pixel_size,
        pixel_size_offsets=torch.tensor([0.0], device=device),
        cs=optics.spherical_aberration,
    )
    return float(cs_vals[0].item())


def _ctf_rfft_single(
    optics: OpticsGroup,
    defocus_mean_angstrom: float,
    phase_shift_deg: float,
    image_shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """One CTF (half-plane rFFT) with B-factor envelope, shape ``(H, W//2+1)``."""
    astigmatism_angstrom = abs(optics.defocus_u - optics.defocus_v) / 2.0
    cs_val = _nominal_cs_mm(optics, device)
    mag_matrix = optics.mag_matrix_tensor

    defocus_um = torch.tensor(
        defocus_mean_angstrom * 1e-4, device=device, dtype=torch.float32
    )
    astig_um = torch.tensor(
        astigmatism_angstrom * 1e-4, device=device, dtype=torch.float32
    )

    if optics.laser_params is not None:
        lp = optics.laser_params
        tmp = calc_LPP_ctf_2D(
            defocus=defocus_um,
            astigmatism=astig_um,
            astigmatism_angle=optics.astigmatism_angle,
            voltage=optics.voltage,
            spherical_aberration=cs_val,
            amplitude_contrast=optics.amplitude_contrast_ratio,
            pixel_size=optics.pixel_size,
            image_shape=image_shape,
            rfft=True,
            fftshift=False,
            NA=lp.NA,
            laser_wavelength_angstrom=lp.laser_wavelength_angstrom,
            focal_length_angstrom=lp.focal_length_angstrom,
            laser_xy_angle_deg=lp.laser_xy_angle_deg,
            laser_xz_angle_deg=lp.laser_xz_angle_deg,
            laser_long_offset_angstrom=lp.laser_long_offset_angstrom,
            laser_trans_offset_angstrom=lp.laser_trans_offset_angstrom,
            laser_polarization_angle_deg=lp.laser_polarization_angle_deg,
            peak_phase_deg=lp.peak_phase_deg,
            dual_laser=lp.dual_laser,
            beam_tilt_mrad=None,
            even_zernike_coeffs=optics.even_zernikes,
            odd_zernike_coeffs=optics.odd_zernikes,
            transform_matrix=mag_matrix,
        )
    else:
        tmp = calculate_ctf_2d(
            defocus=defocus_um,
            astigmatism=astig_um,
            astigmatism_angle=optics.astigmatism_angle,
            voltage=optics.voltage,
            spherical_aberration=cs_val,
            amplitude_contrast=optics.amplitude_contrast_ratio,
            phase_shift=phase_shift_deg,
            pixel_size=optics.pixel_size,
            image_shape=image_shape,
            rfft=True,
            fftshift=False,
            even_zernike_coeffs=optics.even_zernikes,
            odd_zernike_coeffs=optics.odd_zernikes,
            transform_matrix=mag_matrix,
        )

    tmp = tmp.to(device=device)
    env = b_envelope(
        B=optics.ctf_B_factor,
        image_shape=image_shape,
        pixel_size=optics.pixel_size,
        rfft=True,
        fftshift=False,
        device=tmp.device,
    )
    return tmp * env


def apply_circular_ctf_multiply(
    image: torch.Tensor,
    optics: OpticsGroup,
    defocus_mean_angstrom: float,
    phase_shift_deg: float,
) -> torch.Tensor:
    """Apply CTF by circular convolution: ``irfft(rfft(image) * CTF)``.

    No kernel crop or sum-normalization. Matches a Fourier-domain CTF multiply
    on the full micrograph (uniform spatial field).
    """
    image_shape = (int(image.shape[-2]), int(image.shape[-1]))
    ctf = _ctf_rfft_single(
        optics,
        defocus_mean_angstrom,
        phase_shift_deg,
        image_shape,
        image.device,
    )
    image_dft = torch.fft.rfft2(image)  # pylint: disable=not-callable
    return torch.fft.irfft2(  # pylint: disable=not-callable
        image_dft * ctf, s=image_shape
    )


def rfft2_parseval_power(x: torch.Tensor) -> torch.Tensor:
    """Energy of an unshifted rFFT with interior columns doubled (Nyquist-aware)."""
    abs2 = torch.abs(x) ** 2
    return abs2.sum(dim=(-2, -1)) + abs2[..., :, 1:-1].sum(dim=(-2, -1))


def sample_template_slices_for_gain(
    template_dft: torch.Tensor,
    euler_angles: torch.Tensor,
    max_slices: int = 32,
) -> torch.Tensor:
    """Evenly sample Fourier slices for the plane-wise template-norm gain."""
    n_orient = int(euler_angles.shape[0])
    n_sample = min(int(max_slices), n_orient)
    idx = torch.linspace(0, n_orient - 1, n_sample).long()
    eulers = euler_angles[idx].to(device=template_dft.device, dtype=torch.float32)
    rotation_matrices = roma.euler_to_rotmat(
        "ZYZ", eulers, degrees=True, device=template_dft.device
    )
    slices = extract_central_slices_rfft_3d(
        volume_rfft=template_dft,
        rotation_matrices=rotation_matrices,
    )
    slices = torch.fft.ifftshift(slices, dim=-2)  # pylint: disable=not-callable
    slices[..., 0, 0] = 0 + 0j
    return slices


def template_ctf_unit_variance_gain(
    ctf_rfft: torch.Tensor,
    whitening_filter_template: torch.Tensor,
    template_slices_rfft: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scale so spatial raw MIP matches Fourier-2DTM template-side CTF norm.

    Fourier 2DTM unit-normalizes ``T · CTF · W``. Spatial puts CTF on the image
    and unit-normalizes ``T · W``. The ratio ``||T W|| / ||T CTF W||`` is the
    missing template gain (one scalar per defocus plane). If ``template_slices_rfft``
    is omitted, ``T`` is treated as white (``||W|| / ||CTF W||``).
    """
    whitening = whitening_filter_template.to(dtype=torch.float32)
    ctf = ctf_rfft.to(device=whitening.device)
    if template_slices_rfft is None:
        probe = whitening
        filtered = ctf * whitening
    else:
        slices = template_slices_rfft.to(device=whitening.device)
        probe = slices * whitening
        filtered = slices * ctf * whitening
    numerator = rfft2_parseval_power(probe)
    denominator = rfft2_parseval_power(filtered).clamp_min(1e-12)
    return torch.sqrt(numerator / denominator).mean()


def ctf_to_psf_crop(
    ctf_rfft: torch.Tensor,
    image_shape: tuple[int, int],
    kernel_size: int,
) -> torch.Tensor:
    """Inverse FFT and center-crop ``kernel_size``; do not sum-normalize.

    Sum-to-1 forces DC gain to 1 and destroys CTF oscillations. The crop is a
    truncated copy of ``irfft(CTF)``.
    """
    h, w = image_shape
    psf_full = torch.fft.irfft2(ctf_rfft, s=(h, w))  # pylint: disable=not-callable
    psf_shifted = torch.fft.fftshift(psf_full)  # pylint: disable=not-callable
    center_y, center_x = h // 2, w // 2
    side = kernel_size // 2
    cropped = psf_shifted[
        center_y - side : center_y + side + 1,
        center_x - side : center_x + side + 1,
    ]
    return cropped.to(dtype=torch.float32)


def _circular_pad_kernel(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Wrap-pad a 2D image so a ``kernel_size`` window is centered on every pixel."""
    pad_val = (kernel_size - 1) // 2
    return (
        F.pad(
            image.unsqueeze(0).unsqueeze(0),
            (pad_val, pad_val, pad_val, pad_val),
            mode="circular",
        )
        .squeeze(0)
        .squeeze(0)
    )


def build_psf_kernel_grid_defocus(
    optics: OpticsGroup,
    defocus_vertex_angstrom: torch.Tensor,
    phase_shift_deg: float,
    image_shape: tuple[int, int],
    kernel_size: int,
    device: torch.device,
) -> torch.Tensor:
    """``(nx, ny, kh, kw)`` truncated PSF crops (not sum-normalized).

    ``defocus_vertex_angstrom`` is ``(nx, ny)`` in Å.
    """
    if defocus_vertex_angstrom.ndim != 2:
        raise ValueError("defocus_vertex_angstrom must be 2D (nx, ny).")
    nx, ny = defocus_vertex_angstrom.shape
    dv = defocus_vertex_angstrom.to(device=device, dtype=torch.float32)
    kernels = []
    for i in range(nx):
        row_k = []
        for j in range(ny):
            ctf = _ctf_rfft_single(
                optics,
                float(dv[i, j].item()),
                phase_shift_deg,
                image_shape,
                device,
            )
            row_k.append(ctf_to_psf_crop(ctf, image_shape, kernel_size))
        kernels.append(torch.stack(row_k, dim=0))
    return torch.stack(kernels, dim=0)


def build_psf_kernel_grid_phase(
    optics: OpticsGroup,
    defocus_mean_angstrom: float,
    phase_vertex_deg: torch.Tensor,
    image_shape: tuple[int, int],
    kernel_size: int,
    device: torch.device,
) -> torch.Tensor:
    """``(nx, ny, kh, kw)`` PSFs for phase (deg) at each vertex.

    Mean defocus is fixed (Å).
    """
    if optics.laser_params is not None:
        raise NotImplementedError(
            "Spatial quadratic phase mode is not implemented for laser phase plate "
            "optics."
        )
    if phase_vertex_deg.ndim != 2:
        raise ValueError("phase_vertex_deg must be 2D (nx, ny).")
    nx, ny = phase_vertex_deg.shape
    pv = phase_vertex_deg.to(device=device, dtype=torch.float32)
    kernels = []
    for i in range(nx):
        row_k = []
        for j in range(ny):
            ctf = _ctf_rfft_single(
                optics,
                defocus_mean_angstrom,
                float(pv[i, j].item()),
                image_shape,
                device,
            )
            row_k.append(ctf_to_psf_crop(ctf, image_shape, kernel_size))
        kernels.append(torch.stack(row_k, dim=0))
    return torch.stack(kernels, dim=0)


def _lagrange_quadratic_weights_1d(s: torch.Tensor) -> torch.Tensor:
    """Quadratic Lagrange basis for nodes at 0, 1, 2; ``s`` in ``[0, 2]``."""
    basis_0 = 0.5 * (s - 1.0) * (s - 2.0)
    basis_1 = -s * (s - 2.0)
    basis_2 = 0.5 * s * (s - 1.0)
    return torch.stack([basis_0, basis_1, basis_2], dim=-1)


def _effective_kernel_row_bilinear(  # pylint: disable=too-many-locals
    kernel_grid: torch.Tensor,
    fx: torch.Tensor,
    fy_scalar: float,
) -> torch.Tensor:
    """``fx`` is ``(W,)`` in ``[0, nx-1]``; returns ``(W, kh, kw)``."""
    nx, ny, kh, kw = kernel_grid.shape
    wlen = fx.shape[0]
    device = fx.device
    dtype = fx.dtype

    if nx == 1 and ny == 1:
        return kernel_grid[0, 0].unsqueeze(0).expand(wlen, kh, kw).clone()

    if nx == 1:
        fy = torch.full((wlen,), fy_scalar, device=device, dtype=dtype)
        iy = torch.floor(fy).long().clamp(0, ny - 2)
        uy = fy - iy.float()
        k_lo = kernel_grid[0, iy]
        k_hi = kernel_grid[0, iy + 1]
        return (1.0 - uy).view(-1, 1, 1) * k_lo + uy.view(-1, 1, 1) * k_hi

    if ny == 1:
        ix = torch.floor(fx).long().clamp(0, nx - 2)
        ux = fx - ix.float()
        k_lo = kernel_grid[ix, 0]
        k_hi = kernel_grid[ix + 1, 0]
        return (1.0 - ux).view(-1, 1, 1) * k_lo + ux.view(-1, 1, 1) * k_hi

    ix = torch.floor(fx).long().clamp(0, nx - 2)
    fy_t = torch.tensor(fy_scalar, device=device, dtype=dtype)
    iy = int(torch.floor(fy_t).clamp(0, ny - 2).item())
    ux = fx - ix.float()
    uy = float(fy_scalar) - iy
    w00 = (1.0 - ux) * (1.0 - uy)
    w10 = ux * (1.0 - uy)
    w01 = (1.0 - ux) * uy
    w11 = ux * uy
    return (
        w00.view(-1, 1, 1) * kernel_grid[ix, iy]
        + w10.view(-1, 1, 1) * kernel_grid[ix + 1, iy]
        + w01.view(-1, 1, 1) * kernel_grid[ix, iy + 1]
        + w11.view(-1, 1, 1) * kernel_grid[ix + 1, iy + 1]
    )


def _effective_kernel_row_biquadratic(  # pylint: disable=too-many-locals
    kernel_grid: torch.Tensor,
    fx: torch.Tensor,
    fy_scalar: float,
) -> torch.Tensor:
    """Biquadratic tensor-product weights.

    Falls back to bilinear if ``nx < 3`` or ``ny < 3``.
    """
    nx, ny, kh, kw = kernel_grid.shape
    if nx < 3 or ny < 3:
        return _effective_kernel_row_bilinear(kernel_grid, fx, fy_scalar)

    wlen = fx.shape[0]
    device = fx.device
    dtype = fx.dtype

    i0x = torch.floor(fx).long().clamp(0, nx - 3)
    sx = (fx - i0x.float()).clamp(0.0, 2.0)
    wx = _lagrange_quadratic_weights_1d(sx)

    fy_t = torch.tensor(fy_scalar, device=device, dtype=dtype)
    i0y = int(torch.floor(fy_t).clamp(0, ny - 3).item())
    sy = float((fy_t - i0y).clamp(0.0, 2.0))
    wy = _lagrange_quadratic_weights_1d(torch.tensor(sy, device=device, dtype=dtype))

    acc = torch.zeros(wlen, kh, kw, device=device, dtype=dtype)
    for a in range(3):
        ii = (i0x + a).clamp(0, nx - 1)
        wxa = wx[:, a].view(-1, 1, 1)
        for b in range(3):
            jj = i0y + b
            jj = min(jj, ny - 1)
            wyb = wy[b].item()
            acc = acc + wxa * wyb * kernel_grid[ii, jj]
    return acc


def apply_spatial_psf_grid(  # pylint: disable=too-many-locals
    image: torch.Tensor,
    kernel_grid: torch.Tensor,
    *,
    grid_nx: int,
    grid_ny: int,
    blend: Literal["bilinear", "biquadratic"],
    kernel_size: int,
) -> torch.Tensor:
    """Convolve ``image`` with per-pixel PSFs from an ``(nx, ny, kh, kw)`` bank.

    Pixel normalized coordinates match ``pixel_norm_coords`` / ``make_positions_2d``.
    **bilinear** is used for linear defocus; **biquadratic** approximates per-pixel
    ``PSF(φ(p))`` more closely when phase varies smoothly (falls back to bilinear
    if ``nx < 3`` or ``ny < 3``). Convolution is circular (wrap-around), matching
    a Fourier CTF multiply on the full micrograph.
    """
    if kernel_grid.shape[:2] != (grid_nx, grid_ny):
        raise ValueError(
            f"kernel_grid leading dims {kernel_grid.shape[:2]} != "
            f"({grid_nx}, {grid_ny})"
        )
    kh, kw = kernel_grid.shape[-2:]
    if kh != kernel_size or kw != kernel_size:
        raise ValueError("kernel_grid trailing dims must equal kernel_size.")

    size_h, size_w = image.shape
    padded = _circular_pad_kernel(image, kernel_size)

    x_1d = torch.linspace(0.0, 1.0, size_w, device=image.device, dtype=torch.float32)
    if size_w == 1:
        x_1d = torch.tensor([0.5], device=image.device, dtype=torch.float32)
    y_1d = torch.linspace(0.0, 1.0, size_h, device=image.device, dtype=torch.float32)
    if size_h == 1:
        y_1d = torch.tensor([0.5], device=image.device, dtype=torch.float32)

    fx_scale = max(grid_nx - 1, 1)
    fy_scale = max(grid_ny - 1, 1)

    row_fn = (
        _effective_kernel_row_biquadratic
        if blend == "biquadratic"
        else _effective_kernel_row_bilinear
    )

    output_rows: list[torch.Tensor] = []
    for r in range(size_h):
        y_norm = float(y_1d[r].item())
        fy_s = y_norm * fy_scale
        fx = x_1d * fx_scale
        eff_k = row_fn(kernel_grid, fx, fy_s)
        # eff_k: (size_w, kh, kw) — one PSF per column (len(fx) == size_w).
        # row_patches: k rows from r, unfold width into k-by-k tiles, permute to
        # (size_w, kh, kw). Same as eff_k; multiply elementwise, sum over kh and kw.
        row_patches = (
            padded[r : r + kernel_size, :].unfold(1, kernel_size, 1).permute(1, 0, 2)
        )
        output_rows.append((row_patches * eff_k).sum(dim=(-1, -2)))
    return torch.stack(output_rows, dim=0)


# --- Legacy 1D stack API (tests / simple callers) ---


def _scalar_map_interp_range(scalar_map: torch.Tensor) -> tuple[float, float]:
    s_min = float(scalar_map.min().item())
    s_max = float(scalar_map.max().item())
    if s_max - s_min < 1e-9:
        s_max = s_min + 1e-6
    return s_min, s_max


def build_defocus_psf_stack(
    optics: OpticsGroup,
    defocus_levels_angstrom: torch.Tensor,
    phase_shift_deg: float,
    image_shape: tuple[int, int],
    kernel_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Legacy: ``(L, kh, kw)`` stack along one axis."""
    levels = defocus_levels_angstrom.to(device=device, dtype=torch.float32)
    kernels = []
    for i in range(levels.numel()):
        ctf = _ctf_rfft_single(
            optics,
            float(levels[i].item()),
            phase_shift_deg,
            image_shape,
            device,
        )
        kernels.append(ctf_to_psf_crop(ctf, image_shape, kernel_size))
    return torch.stack(kernels, dim=0)


def build_phase_psf_stack(
    optics: OpticsGroup,
    defocus_mean_angstrom: float,
    phase_levels_deg: torch.Tensor,
    image_shape: tuple[int, int],
    kernel_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Legacy phase stack ``(L, kh, kw)``."""
    if optics.laser_params is not None:
        raise NotImplementedError(
            "Spatial quadratic phase mode is not implemented for laser phase plate "
            "optics."
        )
    levels = phase_levels_deg.to(device=device, dtype=torch.float32)
    kernels = []
    for i in range(levels.numel()):
        ctf = _ctf_rfft_single(
            optics,
            defocus_mean_angstrom,
            float(levels[i].item()),
            image_shape,
            device,
        )
        kernels.append(ctf_to_psf_crop(ctf, image_shape, kernel_size))
    return torch.stack(kernels, dim=0)


def apply_spatially_varying_psf(  # pylint: disable=too-many-locals
    image: torch.Tensor,
    scalar_map: torch.Tensor,
    kernel_stack: torch.Tensor,
    *,
    kernel_size: int,
) -> torch.Tensor:
    """Legacy 1D interpolation along scalar min/max."""
    if image.shape != scalar_map.shape:
        raise ValueError(
            f"image {image.shape} and scalar_map {scalar_map.shape} must match."
        )
    lcount, kh, kw = kernel_stack.shape
    if kh != kernel_size or kw != kernel_size:
        raise ValueError("kernel_stack spatial dims must equal kernel_size.")
    size_h, _ = image.shape
    s_min, s_max = _scalar_map_interp_range(scalar_map)

    padded = _circular_pad_kernel(image, kernel_size)

    level_idx = (scalar_map - s_min) / (s_max - s_min) * (lcount - 1)
    level_lo = level_idx.clamp(0, lcount - 1.0001).long()
    level_hi = (level_lo + 1).clamp(max=lcount - 1)
    w_hi = (level_idx - level_lo.float()).clamp(0, 1)
    w_lo = 1.0 - w_hi

    output_rows: list[torch.Tensor] = []
    for r in range(size_h):
        row_patches = (
            padded[r : r + kernel_size, :].unfold(1, kernel_size, 1).permute(1, 0, 2)
        )
        k_lo = kernel_stack[level_lo[r]]
        k_hi = kernel_stack[level_hi[r]]
        kernels_r = (
            w_lo[r].unsqueeze(-1).unsqueeze(-1) * k_lo
            + w_hi[r].unsqueeze(-1).unsqueeze(-1) * k_hi
        )
        output_rows.append((row_patches * kernels_r).sum(dim=(-1, -2)))
    return torch.stack(output_rows, dim=0)


def linspace_levels(
    scalar_map: torch.Tensor, num_levels: int, device: torch.device
) -> torch.Tensor:
    """Legacy: ``num_levels`` values from min to max of ``scalar_map``."""
    s_min, s_max = _scalar_map_interp_range(scalar_map)
    return torch.linspace(s_min, s_max, num_levels, device=device, dtype=torch.float32)
