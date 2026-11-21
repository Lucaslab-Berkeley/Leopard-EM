"""Tests for the refined null (null peak rate after refining a search's candidates)."""

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from leopard_em.analysis import predict_match_template_null
from leopard_em.analysis.analytic_null import (
    AnalyticNullPrediction,
    MeanTemplate,
    PowerWeighting,
    RefinementGain,
    RefinementWindow,
    predict_refine_template_null,
    predict_refined_null_rate,
    refine_window,
    sample_refinement_gain,
    score_layout,
    window_covariance,
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

BOX = (12, 12)


def _white_weighting():
    rng = np.random.default_rng(0)
    image = torch.from_numpy(rng.standard_normal((96, 96)))
    return PowerWeighting(torch.fft.rfftn(image), (96, 96), BOX, white=True)


def _ring(angles, a, b):
    t = torch.as_tensor(angles, dtype=torch.float32)[:, None, None]
    return torch.cos(t) * a + torch.sin(t) * b


def _ring_basis(seed):
    """Two smooth, zero-mean, orthogonal, equal-norm patterns (a ring of templates)."""
    rng = np.random.default_rng(seed)
    a, b = (
        torch.from_numpy(gaussian_filter(rng.standard_normal(BOX), 1.0)).float()
        for _ in range(2)
    )
    a = a - a.mean()
    b = b - b.mean()
    b = b - (a * b).sum() / (a * a).sum() * a
    b = b * torch.sqrt((a * a).sum() / (b * b).sum())
    return a, b


def _real_space_covariance(projections, layout):
    """Cov(CC(x1, s1), CC(x2, s2)) for a white image, by zero-filled real-space lags."""
    p = projections.double().numpy()
    r = int(max(np.abs(layout.dy).max(), np.abs(layout.dx).max()))
    pad = np.pad(p, ((0, 0), (2 * r, 2 * r), (2 * r, 2 * r)))
    h, w = BOX
    n = len(layout.projection)
    cov = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            dy = layout.dy[a] - layout.dy[b]
            dx = layout.dx[a] - layout.dx[b]
            shifted = pad[
                layout.projection[b],
                2 * r + dy : 2 * r + dy + h,
                2 * r + dx : 2 * r + dx + w,
            ]
            cov[a, b] = (p[layout.projection[a]] * shifted).sum()
    return cov


def test_score_layout():
    layout = score_layout(3, 2, 1, shift_radius=1, condition_radius=2)
    # 9 positions with 3 window + 2 search scores, 16 with 2 search + the own
    assert len(layout.projection) == 9 * 5 + 16 * 3
    assert layout.in_window.sum() == 27
    assert layout.beaten.sum() == 25 * 2 + 16 + 8
    c = layout.centre
    assert (layout.projection[c], layout.dy[c], layout.dx[c]) == (1, 0, 0)
    assert not layout.beaten[c] and layout.in_window[c]
    own = (layout.projection == 1) & ~layout.in_window
    assert np.all(np.maximum(np.abs(layout.dy[own]), np.abs(layout.dx[own])) == 2)


def test_window_covariance_is_the_real_space_covariance():
    weighting = _white_weighting()
    a, b = _ring_basis(1)
    window = _ring([-0.2, 0.0, 0.2], a, b)
    search = _ring([0.5, -0.5], a, b)
    q = weighting.transform(torch.cat([window, search]))
    sigma2 = float(weighting.inner(q[1], q[1]))
    mean = MeanTemplate(torch.zeros_like(q[0]), sigma2, sigma2, 0.0, 1)
    cov, layout = window_covariance(
        RefinementWindow(q[:3], 1, q[3:]), weighting, mean, 1, condition_radius=2
    )
    expected = _real_space_covariance(torch.cat([window, search]), layout)
    expected /= expected[layout.centre, layout.centre]
    np.testing.assert_allclose(cov.cpu().numpy(), expected, atol=2e-4)


def test_gain_matches_brute_force_conditioning():
    """Conditional draws vs rejection from unconditional draws of the same field.

    A ring of templates: refinement scores 5 offsets about the candidate at +-1 px;
    the search's grid has steps of 0.45, so the candidate beat its two neighbours and
    its own template, within +-2 px.
    """
    weighting = _white_weighting()
    a, b = _ring_basis(2)
    rng = np.random.default_rng(3)
    extra = torch.from_numpy(
        gaussian_filter(rng.standard_normal((7, *BOX)), (0, 1.0, 1.0))
    ).float()
    extra = 0.15 * (extra - extra.mean(dim=(-2, -1), keepdim=True)) / extra.std()
    window = _ring(np.linspace(-0.3, 0.3, 5), a, b) + extra[:5]
    search = _ring([-0.45, 0.45], a, b) + extra[5:]
    q = weighting.transform(torch.cat([window, search]))
    sigma2 = float(weighting.inner(q[2], q[2]))
    mean = MeanTemplate(torch.zeros_like(q[0]), sigma2, sigma2, 0.0, 1)
    u = 2.0
    gain = sample_refinement_gain(
        [RefinementWindow(q[:5], 2, q[5:])],
        weighting,
        mean,
        shift_radius=1,
        condition_radius=2,
        heights=np.array([u]),
        n_draws=100_000,
        seed=0,
    )
    layout = score_layout(5, 2, 2, shift_radius=1, condition_radius=2)
    cov = _real_space_covariance(torch.cat([window, search]), layout)
    centre = layout.centre
    cov /= cov[centre, centre]
    factor = np.linalg.cholesky(cov + 1e-10 * np.eye(len(cov)))
    rng = np.random.default_rng(0)
    # independent of the layout's masks: window = projections 0-4 within +-1 px
    small = np.maximum(np.abs(layout.dy), np.abs(layout.dx)) <= 1
    in_window = small & (layout.projection < 5)
    beaten = (layout.projection >= 5) | (layout.projection == 2)
    beaten[centre] = False
    g, n_near, n_keep = [], 0, 0
    for _ in range(24):
        draws = rng.standard_normal((500_000, len(cov))) @ factor.T
        x0 = draws[:, centre]
        near = np.abs(x0 - u) < 0.03
        keep = near & (draws[:, beaten] < x0[:, None]).all(axis=1)
        g.append(draws[keep][:, in_window].max(axis=1) - x0[keep])
        n_near += int(near.sum())
        n_keep += int(keep.sum())
    g = np.concatenate(g)
    assert n_keep > 3000
    for value in (0.05, 0.15, 0.3):
        assert gain.survival_at(u, value) == pytest.approx(
            np.mean(g >= value), abs=0.03
        )
    assert gain.accepted[0] == pytest.approx(n_keep / n_near, abs=0.01)


def test_gain_of_an_approximate_search_matches_brute_force():
    """The search scores its own (approximate) templates; refinement re-scores exactly.

    The candidate's score is its approximate template's; the gain can be negative.
    """
    weighting = _white_weighting()
    a, b = _ring_basis(4)
    rng = np.random.default_rng(5)
    error = torch.from_numpy(
        gaussian_filter(rng.standard_normal((3, *BOX)), (0, 0.7, 0.7))
    ).float()
    error = 0.35 * (error - error.mean(dim=(-2, -1), keepdim=True)) / error.std()
    window = _ring(np.linspace(-0.3, 0.3, 5), a, b)  # exact
    approximate = _ring([0.0, -0.45, 0.45], a, b) + error  # own, then neighbours
    projections = torch.cat([window, approximate[1:], approximate[:1]])
    q = weighting.transform(projections)
    unit = float(weighting.inner(q[2], q[2]))
    q = q / np.sqrt(unit)
    u = 2.0
    gain = sample_refinement_gain(
        [RefinementWindow(q[:5], 2, q[5:7], own=q[7])],
        weighting,
        None,
        shift_radius=1,
        condition_radius=1,
        heights=np.array([u]),
        n_draws=100_000,
        seed=0,
    )
    layout = score_layout(
        5, 2, 2, shift_radius=1, condition_radius=1, separate_own=True
    )
    cov = (
        _real_space_covariance(projections, layout)
        / (_real_space_covariance(projections[2:3], score_layout(1, 0, 0, 0, 0))[0, 0])
    )
    centre = layout.centre
    assert layout.projection[centre] == 7
    in_window = layout.projection < 5
    beaten = layout.projection >= 5
    beaten[centre] = False
    factor = np.linalg.cholesky(cov + 1e-10 * np.eye(len(cov)))
    g, n_near, n_keep = [], 0, 0
    for _ in range(16):
        draws = rng.standard_normal((500_000, len(cov))) @ factor.T
        x0 = draws[:, centre]
        near = np.abs(x0 - u) < 0.03
        keep = near & (draws[:, beaten] < x0[:, None]).all(axis=1)
        g.append(draws[keep][:, in_window].max(axis=1) - x0[keep])
        n_near += int(near.sum())
        n_keep += int(keep.sum())
    g = np.concatenate(g)
    assert n_keep > 2000
    assert np.mean(g < 0) > 0.1  # the exact re-score can lose
    for value in (-0.3, 0.0, 0.2):
        assert gain.survival_at(u, value) == pytest.approx(
            np.mean(g >= value), abs=0.03
        )
    assert gain.accepted[0] == pytest.approx(n_keep / n_near, abs=0.015)


def test_windows_pool_by_their_candidates_density():
    """A search point whose score varies more supplies more candidates at height u."""
    weighting = _white_weighting()
    a, b = _ring_basis(6)
    q = weighting.transform(_ring(np.linspace(-0.3, 0.3, 5), a, b))
    sigma2 = float(weighting.inner(q[2], q[2]))
    mean = MeanTemplate(torch.zeros_like(q[0]), sigma2, sigma2, 0.0, 1)
    v = 1.15
    plain = RefinementWindow(q[:5], 2, q[5:])
    wider = RefinementWindow(q[:5] * np.sqrt(v), 2, q[5:])
    heights = np.array([3.0])
    kw = {"shift_radius": 0, "condition_radius": 0, "heights": heights}
    kw["n_draws"] = 100_000
    g1 = sample_refinement_gain([plain], weighting, mean, **kw)
    g2 = sample_refinement_gain([wider], weighting, mean, **kw)
    both = sample_refinement_gain([plain, wider], weighting, mean, **kw)
    u = heights[0]
    w2 = np.exp(-(u**2) / 2 * (1 / v - 1)) / np.sqrt(v)
    a1, a2 = g1.accepted[0], w2 * g2.accepted[0]
    for g in (0.05, 0.2, 0.4):
        mix = (a1 * g1.survival_at(u, g) + a2 * g2.survival_at(u, g)) / (a1 + a2)
        assert both.survival_at(u, g) == pytest.approx(mix, abs=0.01)
    assert both.accepted[0] == pytest.approx((a1 + a2) / (1 + w2), abs=0.01)


def _search_prediction(rate):
    t = np.round(np.arange(2.0, 10.0, 0.01), 3)
    values = rate(t)
    return AnalyticNullPrediction(
        thresholds=t, rate=values, distinct_rate=values, lattice_rate=values,
        nominal_rate=values, normalization_factor=np.ones_like(t), n_points=1e9,
        valid_area=1e6, mask_radius=10.0, kappa=0.0, kappa_se=0.0, rho_bar=0.0,
        variance_sd=0.0, neighbour_correlations={},
    )  # fmt: skip


def _constant_gain(delta):
    edges = np.round(np.arange(0.0, 3.0001, 0.005), 4)
    survival = np.tile((edges <= delta + 1e-9).astype(float), (2, 1))
    return RefinementGain(
        heights=np.array([5.0, 9.0]), gain_edges=edges, survival=survival,
        accepted=np.ones(2), n_windows=1, n_draws=1, scores_per_window=1,
    )  # fmt: skip


@pytest.mark.parametrize("delta", [0.0, 0.3])
def test_refined_rate_with_a_constant_gain(delta):
    """Gain G = delta always: lambda_ref(z) = lambda(z - delta) above t_c + delta."""

    def rate(t):
        return 1e4 * np.exp(-3.0 * (t - 5.0))

    search = _search_prediction(rate)
    refined = predict_refined_null_rate(search, _constant_gain(delta), 6.0)
    for z in (6.5, 7.0, 8.0):
        assert refined.rate_at(z) == pytest.approx(rate(z - delta), rel=0.03)
    # below the candidate threshold (+ gain) every candidate counts
    assert refined.rate_at(5.0) == pytest.approx(rate(6.0), rel=0.01)
    assert refined.candidates == pytest.approx(rate(6.0), rel=1e-3)
    assert refined.zscore_cutoff(1.0) == pytest.approx(
        search.zscore_cutoff(1.0) + delta, abs=0.01
    )


def test_refine_window_reads_a_refine_config():
    config = {
        "orientation_refinement_config": {
            "enabled": True, "psi_step_coarse": 1.5, "psi_step_fine": 0.5,
            "theta_step_coarse": 2.5, "theta_step_fine": 0.5,
        },
        "defocus_refinement_config": {
            "enabled": True, "defocus_min": -100.0, "defocus_max": 100.0,
            "defocus_step": 20.0,
        },
    }  # fmt: skip
    offsets, defocus = refine_window(config, defocus_step=40.0)
    assert offsets.shape[1] == 3
    assert float(offsets.abs().sum(dim=1).min()) < 1e-6
    np.testing.assert_allclose(defocus, [-80, -40, 0, 40, 80])
    _, own = refine_window(config, defocus_step=None)
    assert len(own) == 11
    config["pixel_size_refinement_config"] = {"enabled": True}
    with pytest.raises(NotImplementedError):
        refine_window(config)


IMAGE = 96
TEMPLATE = 16


def _manager():
    gen = torch.Generator().manual_seed(0)
    ky = torch.fft.fftfreq(IMAGE)[:, None]
    kx = torch.fft.fftfreq(IMAGE)[None, :]
    colour = 1.0 / (0.02 + torch.sqrt(ky**2 + kx**2))
    image = torch.fft.ifft2(
        torch.fft.fft2(torch.randn((IMAGE, IMAGE), generator=gen)) * colour
    ).real
    np.random.seed(0)
    volume = gaussian_filter(np.random.randn(TEMPLATE, TEMPLATE, TEMPLATE), sigma=1.0)
    manager = MatchTemplateManager.model_construct(
        optics_group=OpticsGroup(
            label="test", voltage=300.0, pixel_size=1.5, defocus_u=8000.0,
            defocus_v=8000.0, astigmatism_angle=0.0, spherical_aberration=2.7,
            amplitude_contrast_ratio=0.07,
        ),
        preprocessing_filters=PreprocessingFilters(),
        computational_config=ComputationalConfigMatch(),
        defocus_search_config=DefocusSearchConfig(
            enabled=True, defocus_min=-400, defocus_max=400, defocus_step=400
        ),
        orientation_search_config=OrientationSearchConfig(psi_step=24, theta_step=24),
    )  # fmt: skip
    manager.micrograph = image.float()
    manager.template_volume = torch.tensor(volume, dtype=torch.float32)
    manager.fast_fft_padding = FastFFTPaddingConfig(enabled=False)
    return manager


def test_refine_template_prediction_runs_and_raises_the_null():
    manager = _manager()
    kwargs = manager.make_backend_core_function_kwargs()
    search = predict_match_template_null(
        kwargs, 1e4, n_mean_samples=60, n_kappa_pairs=4, n_lattice_samples=8
    )
    offsets = np.array([[0, 0, 0], [0, 0, 6], [0, 0, -6], [0, 6, 0], [0, -6, 0]], float)
    refined = predict_refine_template_null(
        kwargs, manager.optics_group, search, offsets, np.array([-200.0, 0.0, 200.0]),
        3.0, n_candidates=3, n_draws=4000, n_mean_samples=60,
        heights=np.array([3.0, 4.0, 5.0]),
    )  # fmt: skip
    gain = refined.gain
    assert gain.n_windows == 3
    assert gain.scores_per_window > 9 * 5 * 3  # window x positions, + search points
    assert np.all(np.diff(gain.survival, axis=1) <= 1e-12)
    assert np.all((gain.accepted > 0) & (gain.accepted <= 1))
    assert gain.quantile(4.0, 0.5) > 0
    above = refined.thresholds >= 4.0
    assert np.all(refined.rate[above] >= refined.search_rate[above] * 0.999)
    assert refined.zscore_cutoff(1.0) > refined.search_zscore_cutoff(1.0)
    assert refined.as_dict()["details"]["engine"] == "leopard_em.refine_template"
    with pytest.raises(ValueError, match="include zero"):
        predict_refine_template_null(
            kwargs, manager.optics_group, search, offsets[1:], np.array([0.0]), 3.0,
            n_candidates=1, n_draws=1000, n_mean_samples=60,
        )  # fmt: skip
