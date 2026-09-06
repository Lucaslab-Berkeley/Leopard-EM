"""Tests for the Fourier-slice projection utilities.

Two things are pinned here:

1. ``get_real_space_projections_from_volume`` must agree with the projection path the
   match-template backend actually uses. It previously did not: the helper passed a full
   ``fftn`` where ``extract_central_slices_rfft_3d`` requires an ``rfftn``, so the
   extractor inferred a volume of depth ``2d - 2`` and returned a wrongly-sized,
   geometrically incorrect projection.
2. The meaning of the ZYZ Euler angles in that projection path. Because
   ``extract_central_slices_rfft_3d`` applies the rotation to the *sampling grid*, the
   object appears rotated by the transpose, which inverts the naive reading of
   ``roma.euler_to_rotmat``. The net convention is that ``psi`` is the in-plane
   direction and ``phi`` is a roll about the template z-axis. Getting this backwards
   silently transposes every downstream filament/orientation analysis.
"""

import numpy as np
import pytest
import roma
import torch
from torch_fourier_slice import extract_central_slices_rfft_3d

from leopard_em.utils.fourier_slice import get_real_space_projections_from_volume


def _offset_rod_volume(size: int, offset: int = 14) -> torch.Tensor:
    """A rod parallel to z, displaced from the axis, with a marker at its +z end.

    The displacement makes a roll about z visible as a moving centroid, and the marker
    breaks the front/back symmetry so the projection is not accidentally degenerate.
    """
    coords = torch.arange(size) - size // 2
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")

    rod = ((x - offset) ** 2 + y**2 < 9) & (z.abs() < size // 3)
    marker = ((x - offset) ** 2 + y**2 + (z - size // 4) ** 2) < 9

    return rod.float() + 3.0 * marker.float()


def _backend_projection(
    volume: torch.Tensor, phi: float, theta: float, psi: float
) -> np.ndarray:
    """Project via the same sequence of operations as the match-template backend."""
    shifted = torch.fft.fftshift(volume, dim=(-3, -2, -1))
    volume_rfft = torch.fft.rfftn(shifted, dim=(-3, -2, -1))
    volume_rfft = torch.fft.fftshift(volume_rfft, dim=(-3, -2))

    rotation = roma.euler_to_rotmat(
        "ZYZ", torch.tensor([[phi, theta, psi]]), degrees=True
    )
    fourier_slice = extract_central_slices_rfft_3d(
        volume_rfft=volume_rfft, rotation_matrices=rotation
    )
    fourier_slice = torch.fft.ifftshift(-fourier_slice, dim=(-2,))

    projection = torch.fft.irfftn(fourier_slice, dim=(-2, -1))
    projection = torch.fft.ifftshift(projection, dim=(-2, -1))

    return projection[0].numpy()


def _normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    return float((a * b).mean())


def _best_vertical_shift(
    image: np.ndarray, reference: np.ndarray, search: int = 25
) -> tuple[int, float]:
    """Find the integer y-shift of ``reference`` that best matches ``image``.

    Integer rolls are exact, which makes this a far more stable probe of the geometry
    than fitting second moments of a short, thick rod.
    """
    best_shift, best_corr = 0, -np.inf
    for shift in range(-search, search + 1):
        corr = _normalized_correlation(image, np.roll(reference, shift, axis=0))
        if corr > best_corr:
            best_shift, best_corr = shift, corr

    return best_shift, best_corr


@pytest.mark.parametrize("size", [32, 64])
def test_projection_helper_matches_backend(size: int) -> None:
    """The helper and the backend projection path must agree exactly.

    Regression test: before the ``fftn`` -> ``rfftn`` fix the helper returned an
    ``(N, 2N - 2)`` image instead of ``(N, N)``.
    """
    volume = _offset_rod_volume(size)
    expected = _backend_projection(volume, 0.0, 90.0, 30.0)

    actual = np.asarray(
        get_real_space_projections_from_volume(
            volume,
            torch.tensor(0.0),
            torch.tensor(90.0),
            torch.tensor(30.0),
        )
    )

    assert actual.shape == (size, size)
    assert actual.shape == expected.shape
    assert _normalized_correlation(actual, expected) == pytest.approx(1.0, abs=1e-5)


def test_psi_sets_the_in_plane_direction() -> None:
    """psi rotates the projected object within the image plane.

    This is the convention ``filament_psi_from_image_line`` relies on when it derives
    psi from a line drawn on the micrograph. Checking psi = 90 against ``rot90`` keeps
    the comparison exact -- no interpolation, and an unambiguous sign.
    """
    volume = _offset_rod_volume(64)
    at_zero = _backend_projection(volume, 0.0, 90.0, 0.0)
    at_ninety = _backend_projection(volume, 0.0, 90.0, 90.0)

    forwards = _normalized_correlation(at_ninety, np.rot90(at_zero, 1))
    backwards = _normalized_correlation(at_ninety, np.rot90(at_zero, -1))

    assert forwards > 0.85, f"psi=90 did not reproduce a rotated psi=0 ({forwards})"
    assert forwards > backwards + 0.25, (
        "psi rotates the projection in the opposite sense to the one assumed by "
        f"filament_psi_from_image_line (forwards={forwards}, backwards={backwards})"
    )


def test_phi_rolls_about_the_filament_axis() -> None:
    """phi leaves the projected direction fixed and rolls the object about its own axis.

    With psi = 0 and theta = 90 the rod's axis lies along image x, so the displacement
    of the rod from that axis rotates in the (y, depth) plane. Its visible component is
    therefore a pure translation in y of ``-offset * sin(phi)``: zero at phi = 0 and
    180 (displacement is purely in depth), and extremal with opposite signs at 90 and
    270. That is what makes phi the azimuth around a filament.
    """
    offset = 14
    volume = _offset_rod_volume(64, offset=offset)
    reference = _backend_projection(volume, 0.0, 90.0, 0.0)

    expected = {0.0: 0, 90.0: -offset, 180.0: 0, 270.0: offset}
    for phi, want in expected.items():
        shift, corr = _best_vertical_shift(
            _backend_projection(volume, phi, 90.0, 0.0), reference
        )
        assert abs(shift - want) <= 2, f"phi={phi}: y-shift {shift}, expected ~{want}"
        # A good match at the expected shift confirms it really is a translation,
        # i.e. phi did not rotate or foreshorten the projection.
        assert corr > 0.8, f"phi={phi}: projection is not a translation ({corr})"
