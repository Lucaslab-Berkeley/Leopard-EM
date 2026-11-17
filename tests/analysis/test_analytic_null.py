"""Tests for the analytic null (predicted null peak rate without a null search)."""

import numpy as np
import pytest
import roma
import torch
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

from leopard_em.analysis import distinct_peaks, predict_match_template_null
from leopard_em.analysis.analytic_null import (
    AxisNeighbours,
    LatticeSamples,
    PowerWeighting,
    crossing,
    crowding_correction,
    estimate_kappa,
    estimate_mean_template,
    lattice_maxima_rate,
    normalization_factor,
    sample_lattice,
)
from leopard_em.backend.cross_correlation import (
    do_batched_orientation_cross_correlate_cpu,
)
from leopard_em.pydantic_models.config import (
    ComputationalConfigMatch,
    DefocusSearchConfig,
    FastFFTPaddingConfig,
    OrientationSearchConfig,
    PreprocessingFilters,
)
from leopard_em.pydantic_models.data_structures import OpticsGroup
from leopard_em.pydantic_models.managers import MatchTemplateManager
from leopard_em.utils.phase_randomization import phase_randomize_image

# --------------------------------------------------------------------------- #
# Lattice math
# --------------------------------------------------------------------------- #


def _independent_samples(n_axes, sided=2.0):
    axes = {
        f"a{i}": np.tile([0.0, 0.0, sided], (3, 1, 1)).reshape(3, 1, 3)
        for i in range(n_axes)
    }
    return LatticeSamples(variance=np.ones((3, 1)), axes=axes, plane_correlation=None)


@pytest.mark.parametrize(("n_axes", "sided"), [(3, 2.0), (2, 1.0)])
def test_lattice_rate_for_independent_neighbours(n_axes, sided):
    """Uncorrelated neighbours: P(local max >= u) = int_u phi(z) Phi(z)^n dz."""
    samples = _independent_samples(n_axes, sided)
    u = np.array([2.5, 3.0, 4.0])
    power = int(sided) * n_axes
    expected = [quad(lambda z: norm.pdf(z) * norm.cdf(z) ** power, t, 12)[0] for t in u]
    np.testing.assert_allclose(
        lattice_maxima_rate(samples, 1.0, u), expected, rtol=1e-4
    )


def test_crowding_normalization_and_crossing():
    area, radius = 1e6, 10.0
    beta = np.pi * radius**2 / area
    np.testing.assert_allclose(
        crowding_correction(np.array([1e-3]), area, radius), 1e-3, rtol=1e-6
    )
    assert crowding_correction(np.array([1e9]), area, radius)[0] == pytest.approx(
        1 / beta
    )
    np.testing.assert_allclose(normalization_factor(0.0, np.array([5.0, 7.0])), 1.0)
    assert normalization_factor(1e-3, np.array([7.0]))[0] < 1.0
    u = np.linspace(5, 9, 41)
    rate = 1e4 * np.exp(-3 * (u - 5))
    assert crossing(u, rate, 1.0) == pytest.approx(5 + np.log(1e4) / 3)
    assert np.isnan(crossing(u, rate, 1e6))


# --------------------------------------------------------------------------- #
# Field estimators, with a toy engine that is not Leopard-EM's
# --------------------------------------------------------------------------- #

BOX = (12, 12)


def _white_weighting():
    rng = np.random.default_rng(0)
    image = torch.from_numpy(rng.standard_normal((96, 96)))
    return PowerWeighting(torch.fft.rfftn(image), (96, 96), BOX, white=True)


def _zero_mean(p):
    return p - p.mean(dim=(-2, -1), keepdim=True)


def test_white_weighting_is_the_real_space_inner_product():
    weighting = _white_weighting()
    rng = np.random.default_rng(1)
    projections = _zero_mean(torch.from_numpy(rng.standard_normal((5, *BOX))).float())
    q = weighting.transform(projections)
    ratio = weighting.inner(q, q) / (projections**2).sum(dim=(-2, -1))
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-4)
    p = projections[0]
    lagged = (p[:, :-2] * p[:, 2:]).sum() / (p**2).sum()  # lag of 2 along x
    shifted = weighting.shift_correlation(q[0], 0, 2) / weighting.inner(q[0], q[0])
    assert float(shifted) == pytest.approx(float(lagged), abs=1e-5)


def test_mean_template_and_kappa_on_a_toy_engine():
    weighting = _white_weighting()
    rng = np.random.default_rng(2)
    base = _zero_mean(torch.from_numpy(rng.standard_normal(BOX)).float())
    direction = _zero_mean(torch.from_numpy(rng.standard_normal(BOX)).float())

    def project(signs):
        p = (
            base
            + torch.as_tensor(signs, dtype=torch.float32)[:, None, None] * direction
        )
        return weighting.transform(p)[None]  # one plane

    signs = [rng.choice([-1.0, 1.0], size=8) for _ in range(50)]
    mean_template = estimate_mean_template(project, weighting, signs)
    share = float((base**2).sum() / ((base**2).sum() + (direction**2).sum()))
    assert mean_template.rho_bar == pytest.approx(share, abs=0.03)
    # after removing the mean, every point is +-direction: correlations are +-1
    pairs = [(rng.choice([-1.0, 1.0], 4), rng.choice([-1.0, 1.0], 4)) for _ in range(4)]
    kappa, _ = estimate_kappa(project, weighting, weighting.transform(base), pairs)
    assert kappa == pytest.approx(1.0, abs=1e-4)


def test_sample_lattice_on_a_ring():
    """Projections cos(t) a + sin(t) b with orthogonal a, b correlate as cos(step)."""
    weighting = _white_weighting()
    rng = np.random.default_rng(3)
    a, b = torch.from_numpy(rng.standard_normal((2, *BOX))).float()
    a = _zero_mean(a)
    b = _zero_mean(b)
    b = b - (a * b).sum() / (a * a).sum() * a
    b = b * torch.sqrt((a * a).sum() / (b * b).sum())
    step = 0.3

    def project(angles):
        t = torch.as_tensor(angles, dtype=torch.float32)[:, None, None]
        return weighting.transform(torch.cos(t) * a + torch.sin(t) * b)[None]

    def neighbours(angles):
        return [
            AxisNeighbours("ring", tuple(angles + m * step for m in (1, -1, 2, -2)))
        ]

    angles = [np.linspace(0, 2 * np.pi, 6, endpoint=False)]
    samples = sample_lattice(
        project, weighting, torch.zeros_like(project(np.zeros(1))[0, 0]), angles,
        neighbours,
    )  # fmt: skip
    np.testing.assert_allclose(samples.axes["ring"][..., 0], np.cos(step), atol=1e-5)
    np.testing.assert_allclose(
        samples.axes["ring"][..., 1], np.cos(2 * step), atol=1e-5
    )
    assert samples.plane_correlation is None


# --------------------------------------------------------------------------- #
# Leopard-EM match_template: prediction against an actual null search
# --------------------------------------------------------------------------- #

IMAGE = 128
TEMPLATE = 16


def _coloured_image(seed):
    gen = torch.Generator().manual_seed(seed)
    ky = torch.fft.fftfreq(IMAGE)[:, None]
    kx = torch.fft.rfftfreq(IMAGE)[None, :]
    power = 300.0 * torch.exp(-torch.sqrt(ky**2 + kx**2) / 0.05) + 1.0
    white = torch.fft.rfftn(torch.randn(IMAGE, IMAGE, generator=gen))
    return torch.fft.irfftn(white * power.sqrt(), s=(IMAGE, IMAGE)).float()


def _manager(image, defocus_enabled=False):
    np.random.seed(0)
    volume = gaussian_filter(np.random.randn(TEMPLATE, TEMPLATE, TEMPLATE), sigma=1.0)
    manager = MatchTemplateManager.model_construct(
        optics_group=OpticsGroup(
            label="test",
            voltage=300.0,
            pixel_size=1.5,
            defocus_u=8000.0,
            defocus_v=8000.0,
            astigmatism_angle=0.0,
            spherical_aberration=2.7,
            amplitude_contrast_ratio=0.07,
        ),
        preprocessing_filters=PreprocessingFilters(),
        computational_config=ComputationalConfigMatch(),
        defocus_search_config=DefocusSearchConfig(
            enabled=defocus_enabled, defocus_min=-400, defocus_max=400, defocus_step=400
        ),
        orientation_search_config=OrientationSearchConfig(psi_step=24, theta_step=24),
    )
    manager.micrograph = image
    manager.template_volume = torch.tensor(volume, dtype=torch.float32)
    manager.fast_fft_padding = FastFFTPaddingConfig(enabled=False)
    return manager


def _null_zscore(manager):
    kwargs = manager.make_backend_core_function_kwargs()
    rot = roma.euler_to_rotmat("ZYZ", kwargs["euler_angles"].float(), degrees=True)
    correlation = do_batched_orientation_cross_correlate_cpu(
        image_dft=kwargs["image_dft"],
        template_dft=kwargs["template_dft"],
        rotation_matrices=rot,
        projective_filters=kwargs["ctf_filters"]
        * kwargs["whitening_filter_template"][None, None],
    )
    valid = IMAGE - TEMPLATE + 1
    correlation = correlation[..., :valid, :valid].reshape(-1, valid, valid)
    mip = correlation.max(dim=0).values
    return (mip - correlation.mean(dim=0)) / correlation.std(dim=0, correction=0)


def test_prediction_matches_a_small_null_search():
    """Distinct null peaks of 24 phase-randomised copies vs the predicted rate.

    The thresholds are where null peaks are rare (3-80 per copy), the regime the
    crowding model is for; much lower, suppression saturates.
    """
    image = _coloured_image(0)
    radius = 4.0
    thresholds = np.array([4.2, 4.4, 4.6])
    counts = np.zeros(thresholds.size)
    copies = 24
    for seed in range(copies):
        z = _null_zscore(_manager(phase_randomize_image(image, seed)))
        peaks = distinct_peaks(z, thresholds[0], mask_radius=radius)
        counts += (peaks.zscore[None, :] >= thresholds[:, None]).sum(axis=1)
    valid_area = (IMAGE - TEMPLATE + 1) ** 2
    kwargs = _manager(image).make_backend_core_function_kwargs()
    prediction = predict_match_template_null(
        kwargs,
        valid_area,
        mask_radius=radius,
        n_mean_samples=400,
        n_kappa_pairs=16,
        n_lattice_samples=64,
        thresholds=np.round(np.arange(2.0, 6.5, 0.01), 3),
    )
    predicted = np.array([prediction.rate_at(t) for t in thresholds]) * copies
    # Poisson scatter of the counts is 2-6% here; measured agreement is within 10%
    np.testing.assert_allclose(counts, predicted, rtol=0.25)
    assert 0 < prediction.kappa < 0.1
    assert 0 < prediction.rho_bar < 0.5
    # coarse 24-degree steps: orientation neighbours are nearly uncorrelated
    for r1, _ in prediction.neighbour_correlations.values():
        assert -1 < r1 < 1
    assert prediction.neighbour_correlations["x"][0] > 0.1
    assert np.all(np.diff(prediction.rate) <= 0)
    assert prediction.zscore_cutoff(1.0) < prediction.nominal_zscore_cutoff(1.0)


def test_prediction_with_defocus_planes():
    kwargs = _manager(
        _coloured_image(1), defocus_enabled=True
    ).make_backend_core_function_kwargs()
    prediction = predict_match_template_null(
        kwargs, 1e4, n_mean_samples=60, n_kappa_pairs=4, n_lattice_samples=8
    )
    assert prediction.details["planes"] == 3
    assert 0 < prediction.neighbour_correlations["defocus"][0] < 1
    assert np.isfinite(prediction.zscore_cutoff(1.0))
