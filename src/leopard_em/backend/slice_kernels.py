# Triton has no type stubs, and its kernels take untyped pointer / scalar arguments
# mypy: disable-error-code="import-untyped,no-untyped-def,untyped-decorator"
"""Triton kernel for fast central-slice extraction from an RFFT template volume.

Replicates ``torch_fourier_slice.extract_central_slices_rfft_3d`` (trilinear
interpolation, Friedel flip for negative x frequencies, out-of-bounds samples set to
zero), but reads each complex trilinear corner of the volume as one 64-bit load instead
of two planar 32-bit loads, and writes the slices with the zero frequency at row index 0
(the layout the correlation uses). This is several times faster than the reference
implementation and is the dominant cost of GEMM refinement.

The kernel has no backward pass. It is only used for the (gradient-free) refine search;
see :func:`can_use_triton_slices`.
"""

# pylint: disable=E1102,invalid-name,too-many-arguments,too-many-locals
# pylint: disable=too-many-statements,too-many-positional-arguments

import torch

from leopard_em.utils.fourier_slice import _crop_rfft_slices

try:
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR: Exception | None = None
except ImportError as _err:  # pragma: no cover - depends on the environment
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = _err


def triton_available() -> bool:
    """True if Triton imported and a CUDA device is visible."""
    return triton is not None and torch.cuda.is_available()


def can_use_triton_slices(
    template_dft: torch.Tensor, mag_matrix: torch.Tensor | None = None
) -> bool:
    """Whether the Triton slice extractor supports this template.

    Requires a CUDA complex64 volume of a cubic, even-sized real-space template,
    Triton, and no anisotropic magnification transform.

    Parameters
    ----------
    template_dft : torch.Tensor
        Template volume prepared by ``volume_to_rfft_fourier_slice``, shape
        ``(D, D, D // 2 + 1)``.
    mag_matrix : torch.Tensor | None, optional
        Anisotropic magnification matrix. NOTE: Triton slice extraction does not support
        anisotropic magnification.

    Returns
    -------
    bool
        True if :class:`TritonSliceExtractor` can be used.
    """
    if not triton_available() or mag_matrix is not None:
        return False

    if template_dft.device.type != "cuda" or template_dft.dtype != torch.complex64:
        return False

    d, h, wr = template_dft.shape

    return d == h and d % 2 == 0 and wr == d // 2 + 1


if triton is not None:

    @triton.jit
    def _slice_extract_kernel(
        vol_ptr,
        rot_ptr,
        out_ptr,
        num_orient,
        K,
        NV,
        WR,
        BO: tl.constexpr,
        BK: tl.constexpr,
        INDEX_64: tl.constexpr,
    ):
        """Write a (BO, BK) tile of slice values as interleaved (re, im) pairs."""
        o_idx = tl.program_id(0) * BO + tl.arange(0, BO)
        k_idx = tl.program_id(1) * BK + tl.arange(0, BK)
        o_mask = o_idx < num_orient
        k_mask = k_idx < K

        # Slice frequencies, zero frequency at row 0 (fftfreq / rfftfreq order)
        ky_i = k_idx // WR
        kx_i = k_idx % WR
        ky = tl.where(ky_i < (NV + 1) // 2, ky_i, ky_i - NV).to(tl.float32) / NV
        kx = kx_i.to(tl.float32) / NV

        # Rotation matrices are (z, y, x) ordered; only columns (y, x) are needed
        base = rot_ptr + o_idx * 9
        r01 = tl.load(base + 1, mask=o_mask, other=0.0)[:, None]
        r02 = tl.load(base + 2, mask=o_mask, other=0.0)[:, None]
        r11 = tl.load(base + 4, mask=o_mask, other=0.0)[:, None]
        r12 = tl.load(base + 5, mask=o_mask, other=0.0)[:, None]
        r21 = tl.load(base + 7, mask=o_mask, other=0.0)[:, None]
        r22 = tl.load(base + 8, mask=o_mask, other=0.0)[:, None]
        ky = ky[None, :]
        kx = kx[None, :]
        cz = r01 * ky + r02 * kx
        cy = r11 * ky + r12 * kx
        cx = r21 * ky + r22 * kx

        # Friedel flip for negative x (torch-fourier-slice tie-breaks at x == 0)
        tie = tl.where(tl.abs(r22) <= 1e-7, r21 < 0, r22 < 0)
        conj = (cx < -1e-8) | ((tl.abs(cx) < 1e-8) & tie)
        sgn = tl.where(conj, -1.0, 1.0)

        # Frequencies -> array coordinates (z, y fftshifted; x is the RFFT dim)
        nvf = NV * 1.0
        az = (cz * sgn + 0.5) * nvf
        ay = (cy * sgn + 0.5) * nvf
        ax = cx * sgn * nvf
        tzy = (nvf - 1.0) * 5e-7
        tx = (WR - 1.0) * 5e-7
        inside = (az >= -tzy) & (az <= nvf - 1.0 + tzy)
        inside = inside & (ay >= -tzy) & (ay <= nvf - 1.0 + tzy)
        inside = inside & (ax >= -tx) & (ax <= WR - 1.0 + tx)
        az = tl.minimum(tl.maximum(az, 0.0), nvf - 1.0)
        ay = tl.minimum(tl.maximum(ay, 0.0), nvf - 1.0)
        ax = tl.minimum(tl.maximum(ax, 0.0), WR - 1.0)
        z0 = tl.minimum(az.to(tl.int32), NV - 2)
        y0 = tl.minimum(ay.to(tl.int32), NV - 2)
        x0 = tl.minimum(ax.to(tl.int32), WR - 2)
        fz = az - z0
        fy = ay - y0
        fx = ax - x0

        # Trilinear interpolation, one 64-bit (re, im) load per corner
        m = o_mask[:, None] & k_mask[None, :]
        if INDEX_64:  # volumes with more than 2**31 voxels
            z0 = z0.to(tl.int64)
        sy = WR
        sz = NV * WR
        i000 = (z0 * NV + y0) * WR + x0
        re = tl.zeros((BO, BK), dtype=tl.float32)
        im = tl.zeros((BO, BK), dtype=tl.float32)
        for corner in tl.static_range(8):
            dz = corner // 4
            dy = (corner // 2) % 2
            dx = corner % 2
            wz = fz if dz == 1 else 1.0 - fz
            wy = fy if dy == 1 else 1.0 - fy
            wx = fx if dx == 1 else 1.0 - fx
            v = tl.load(vol_ptr + i000 + dz * sz + dy * sy + dx, mask=m, other=0)
            c_re = v.to(tl.int32).to(tl.float32, bitcast=True)
            c_im = (v >> 32).to(tl.int32).to(tl.float32, bitcast=True)
            weight = wz * wy * wx
            re += weight * c_re
            im += weight * c_im

        keep = inside & m
        re = tl.where(keep, re, 0.0)
        im = tl.where(keep, im * sgn, 0.0)

        # Interleaved (re, im) pairs, stored as contiguous 8-byte values
        row = o_idx.to(tl.int64)[:, None, None] * (2 * K)
        col = 2 * k_idx[None, :, None] + tl.arange(0, 2)[None, None, :]
        tl.store(out_ptr + row + col, tl.join(re, im), mask=m[:, :, None])


class TritonSliceExtractor:
    """Central slices of a template volume via a Triton gather kernel.

    Parameters
    ----------
    template_dft : torch.Tensor
        Template volume prepared by ``volume_to_rfft_fourier_slice`` (may be padded),
        a CUDA complex64 tensor with shape (D, D, D // 2 + 1). Must satisfy
        :func:`can_use_triton_slices`.
    projection_shape : tuple[int, int]
        Real-space template projection shape (h, w). Slices of a padded volume are
        cropped back to it.
    """

    _BO = 16
    _BK = 128

    def __init__(
        self, template_dft: torch.Tensor, projection_shape: tuple[int, int]
    ) -> None:
        if not can_use_triton_slices(template_dft):
            raise ValueError(
                "Template volume is not supported by the Triton slice extractor."
            )
        self.nv = int(template_dft.shape[0])
        self.wr = int(template_dft.shape[2])
        self.projection_shape = (int(projection_shape[0]), int(projection_shape[1]))
        # Interleaved (re, im) float32 pairs viewed as one int64 per voxel (no copy
        # when already contiguous)
        self._volume = (
            torch.view_as_real(template_dft.contiguous())
            .view(torch.int64)
            .reshape(template_dft.shape)
        )

    @property
    def padded_shape(self) -> tuple[int, int]:
        """Real-space shape of the slices before cropping."""
        return (self.nv, self.nv)

    def __call__(self, rotation_matrices: torch.Tensor) -> torch.Tensor:
        """Extract slices for rotation matrices (..., 3, 3).

        Parameters
        ----------
        rotation_matrices : torch.Tensor
            Rotation matrices with shape (O, 3, 3), (x, y, z) convention.

        Returns
        -------
        torch.Tensor
            Complex slices with shape (O, h, w // 2 + 1) and the zero frequency at row
            index 0 (not fftshifted). No contrast flip or DC zeroing applied.
        """
        rot = rotation_matrices.reshape(-1, 3, 3).to(self._volume.device, torch.float32)
        # Match torch-fourier-slice: zero tiny entries, then flip to (z, y, x) order
        rot = torch.where(rot.abs() < 1e-8, torch.zeros_like(rot), rot)
        rot = torch.flip(rot, dims=(-2, -1)).contiguous()

        num = rot.shape[0]
        num_k = self.nv * self.wr
        out = torch.empty((num, self.nv, self.wr, 2), device=rot.device)
        if num > 0:
            grid = (triton.cdiv(num, self._BO), triton.cdiv(num_k, self._BK))
            with torch.cuda.device(rot.device):
                _slice_extract_kernel[grid](  # pylint: disable=E0606
                    self._volume,
                    rot,
                    out,
                    num,
                    num_k,
                    self.nv,
                    self.wr,
                    BO=self._BO,
                    BK=self._BK,
                    INDEX_64=self._volume.numel() >= 2**31 - 1,
                )
        slices = torch.view_as_complex(out)

        if self.padded_shape != self.projection_shape:
            slices = _crop_rfft_slices(
                slices, self.padded_shape, self.projection_shape, rows_centered=False
            )
        return slices
