"""Tests for phase-randomised copies of an image."""

import numpy as np
import pytest
import torch

from leopard_em.utils.phase_randomization import phase_randomize_image

# Output of the reference implementation (the 2DTM statistics prototype that produced
# the validated null searches) for ``np.arange(24).reshape(4, 6) ** 1.5`` and seed 0.
REFERENCE = np.array(
    [
        [72.34703087176729, 82.04273905033718, 65.2566283679757, 60.89529830636761,
         62.347584929649656, 69.6863751899077],
        [51.40365122689042, 60.98903142662553, 34.90251535833866, 26.336294121324645,
         35.370432880593164, 41.18572093386447],
        [-11.8533331885728, 9.321209335820893, -13.112844506122904, -15.94457889519427,
         -11.083487248301818, -8.685567572550987],
        [78.02530025709996, 91.35314417081156, 72.65462492942315, 72.40461416689811,
         70.46927628310038, 74.21235091453593],
    ]
)  # fmt: skip


def test_matches_reference_implementation():
    image = np.arange(24, dtype=np.float64).reshape(4, 6) ** 1.5
    out = phase_randomize_image(image, 0).numpy()
    np.testing.assert_allclose(out, REFERENCE, rtol=0, atol=1e-12)


@pytest.mark.parametrize("shape", [(64, 64), (63, 80), (50, 33)])
def test_keeps_amplitudes_and_mean(shape):
    rng = np.random.default_rng(1)
    image = rng.gamma(2.0, size=shape)
    out = phase_randomize_image(image, 3).numpy()
    np.testing.assert_allclose(
        np.abs(np.fft.rfft2(out)), np.abs(np.fft.rfft2(image)), rtol=1e-9, atol=1e-9
    )
    assert out.mean() == pytest.approx(image.mean(), rel=1e-12)
    # the phases are new: the copy is uncorrelated with the image
    a = image - image.mean()
    b = out - out.mean()
    assert abs((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())) < 0.1


def test_seeded_and_dtype():
    image = torch.randn(32, 40, dtype=torch.float32)
    first = phase_randomize_image(image, 5)
    assert first.dtype == torch.float32
    assert torch.equal(first, phase_randomize_image(image, 5))
    assert not torch.equal(first, phase_randomize_image(image, 6))
    integer = phase_randomize_image(np.ones((8, 8), dtype=np.int32), 0)
    assert integer.dtype == torch.float64


def test_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        phase_randomize_image(np.zeros((2, 3, 4)), 0)
