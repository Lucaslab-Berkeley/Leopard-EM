"""Tests for the GEMM refine backend and the unified, differentiable refine API.

The GEMM scoring must reproduce the FFT engine's normalized cross-correlation, so most
tests compare the two directly on small CPU problems.
"""

import pytest
import roma
import torch

from leopard_em.backend.core_refine_template import core_refine_template
from leopard_em.backend.cross_correlation import (
    do_batched_orientation_cross_correlate_cpu,
)
from leopard_em.backend.refine_gemm import (
    build_filter_operands,
    build_shift_operands,
    gemm_search_particle,
    normalize_gemm_correlation,
    reference_slice_operands,
    score_hypotheses,
)
from leopard_em.backend.slice_kernels import (
    TritonSliceExtractor,
    can_use_triton_slices,
)
from leopard_em.pydantic_models.config import (
    ComputationalConfigRefine,
    DefocusSearchConfig,
    MovieConfig,
    PixelSizeSearchConfig,
    PositionRefinementConfig,
    PreprocessingFilters,
    RefineOrientationConfig,
)
from leopard_em.pydantic_models.data_structures import export_particle_stack
from leopard_em.pydantic_models.managers import RefineTemplateManager
from leopard_em.utils.fourier_slice import (
    extract_template_fourier_slices,
    volume_to_rfft_fourier_slice,
)

SIZE = 16  # template
BOX = 24  # particle image
CROP = BOX - SIZE + 1
NUM_ORIENT = 5
NUM_FILTERS = 6


def _smooth_volume(size: int = SIZE, seed: int = 0) -> torch.Tensor:
    """Band-limited random volume (like a real template)."""
    gen = torch.Generator().manual_seed(seed)
    k = torch.fft.fftfreq(size)
    lowpass = torch.exp(
        -(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)
        / (2 * 0.12**2)
    )
    noise = torch.randn((size,) * 3, generator=gen)
    return torch.fft.ifftn(torch.fft.fftn(noise) * lowpass).real


def _radial_filters(num: int = NUM_FILTERS, size: int = SIZE) -> torch.Tensor:
    """Centrosymmetric RFFT filters, like CTF * whitening filters."""
    radius = (
        torch.fft.fftfreq(size)[:, None] ** 2 + torch.fft.rfftfreq(size)[None] ** 2
    ).sqrt()
    return torch.stack([torch.cos(7 * radius * (i + 1)) ** 2 + 0.1 for i in range(num)])


def _problem(pad_factor: float, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    volume = _smooth_volume(seed=seed)
    image = torch.randn((BOX, BOX), generator=gen)
    rotations = roma.random_rotmat(NUM_ORIENT, device=None)
    return {
        "template_dft": volume_to_rfft_fourier_slice(volume, pad_factor=pad_factor),
        "image": image,
        "image_dft": torch.fft.rfftn(image),
        "rotations": rotations,
        "filters": _radial_filters(),
    }


def _fft_engine_cc(problem: dict) -> torch.Tensor:
    """FFT-engine correlation over the valid crop, shape (O, F, CROP * CROP)."""
    filters = problem["filters"].reshape(2, 3, SIZE, SIZE // 2 + 1)  # (n_px, n_def)
    cc = do_batched_orientation_cross_correlate_cpu(
        image_dft=problem["image_dft"],
        template_dft=problem["template_dft"],
        rotation_matrices=problem["rotations"],
        projective_filters=filters,
    )[..., :CROP, :CROP]
    return cc.permute(2, 0, 1, 3, 4).reshape(NUM_ORIENT, NUM_FILTERS, -1)


def _all_shifts() -> torch.Tensor:
    grid = torch.meshgrid(torch.arange(CROP), torch.arange(CROP), indexing="ij")
    return torch.stack(grid, dim=-1).reshape(-1, 2)


@pytest.mark.parametrize("pad_factor", [1.0, 2.0])
def test_gemm_correlation_matches_fft_engine(pad_factor):
    torch.manual_seed(0)
    problem = _problem(pad_factor)
    reference = _fft_engine_cc(problem)

    filter_ops = build_filter_operands(problem["filters"], (SIZE, SIZE))
    shift_ops = build_shift_operands(
        problem["image"], _all_shifts(), filter_ops, (SIZE, SIZE)
    )
    x_cat, x_pow = reference_slice_operands(problem["template_dft"], (SIZE, SIZE))(
        problem["rotations"]
    )
    raw = (x_cat @ shift_ops.b_raw).reshape(NUM_ORIENT, NUM_FILTERS, -1)
    cc = normalize_gemm_correlation(
        raw,
        x_cat @ filter_ops.b_edge,
        x_pow @ filter_ops.b_pow,
        shift_ops.patch_sums,
        problem["image_dft"][0, 0].real,
        (SIZE, SIZE),
        (BOX, BOX),
    )

    torch.testing.assert_close(cc, reference, atol=1e-5 * reference.abs().max(), rtol=0)


def test_score_hypotheses_matches_fft_engine():
    torch.manual_seed(1)
    problem = _problem(2.0)
    reference = _fft_engine_cc(problem)

    filter_idx = torch.tensor([0, 1, 2, 5, 3])
    positions = torch.tensor([[0, 0], [8, 8], [3, 7], [5, 0], [1, 2]])
    cc = score_hypotheses(
        template_dft=problem["template_dft"],
        rotation_matrices=problem["rotations"],
        filters=problem["filters"][filter_idx],
        particle_images_dft=problem["image_dft"].expand(NUM_ORIENT, -1, -1),
        positions=positions,
    )

    expected = reference[
        torch.arange(NUM_ORIENT), filter_idx, positions[:, 0] * CROP + positions[:, 1]
    ]
    torch.testing.assert_close(cc, expected, atol=1e-5 * expected.abs().max(), rtol=0)


def _search(problem, corr_mean, corr_std, center, radius, batch=2):
    return gemm_search_particle(
        slice_operands=reference_slice_operands(problem["template_dft"], (SIZE, SIZE)),
        rotation_matrices=problem["rotations"],
        image=problem["image"],
        image_sum=problem["image_dft"][0, 0].real,
        filters=problem["filters"],
        corr_mean=corr_mean,
        corr_std=corr_std,
        center=center,
        shift_search_radius=radius,
        orientation_batch_size=batch,
    )


def test_gemm_search_full_crop_finds_global_argmax():
    torch.manual_seed(2)
    problem = _problem(2.0)
    gen = torch.Generator().manual_seed(3)
    corr_mean = torch.randn((CROP, CROP), generator=gen) * 0.1
    corr_std = torch.rand((CROP, CROP), generator=gen) + 0.5

    reference = _fft_engine_cc(problem)
    z_ref = (reference - corr_mean.reshape(-1)) / corr_std.reshape(-1)
    o_i, f_i, s_i = torch.unravel_index(z_ref.argmax(), z_ref.shape)

    best = _search(problem, corr_mean, corr_std, (CROP // 2, CROP // 2), None)

    assert (best.angle_idx, best.filter_idx) == (int(o_i), int(f_i))
    assert (best.pos_y, best.pos_x) == (int(s_i) // CROP, int(s_i) % CROP)
    assert best.z_score == pytest.approx(float(z_ref.max()), rel=1e-5)
    assert best.cross_correlation == pytest.approx(
        float(reference[o_i, f_i, s_i]), rel=1e-5
    )


@pytest.mark.parametrize(
    ("target", "start"),
    [
        ((4, 6), (3, 4)),  # reached by re-centering the window
        ((1, 7), (6, 1)),  # too far: falls back to the whole crop
    ],
)
def test_gemm_window_recenters_toward_the_peak(target, start):
    torch.manual_seed(4)
    problem = _problem(2.0)
    # A z-score landscape that rises steadily toward 'target' and dominates the
    # correlation, so the best position is 'target' whatever the orientation
    ys, xs = torch.meshgrid(torch.arange(CROP), torch.arange(CROP), indexing="ij")
    distance = ((ys - target[0]) ** 2 + (xs - target[1]) ** 2).float().sqrt()
    corr_mean = 100.0 * distance
    corr_std = torch.ones((CROP, CROP))

    best = _search(problem, corr_mean, corr_std, start, radius=1)

    assert (best.pos_y, best.pos_x) == target


def test_gemm_zero_radius_searches_only_the_start():
    torch.manual_seed(4)
    problem = _problem(2.0)
    # The z-score peak is far from the start; a zero-radius window must not move to
    # it (re-centering a one-pixel window, then falling back to the whole crop)
    ys, xs = torch.meshgrid(torch.arange(CROP), torch.arange(CROP), indexing="ij")
    distance = ((ys - 1) ** 2 + (xs - 7) ** 2).float().sqrt()
    corr_mean = 100.0 * distance
    corr_std = torch.ones((CROP, CROP))

    reference = reference_slice_operands(problem["template_dft"], (SIZE, SIZE))
    calls = []

    def counted(rotation_matrices):
        calls.append(rotation_matrices.shape[0])
        return reference(rotation_matrices)

    start = (6, 1)
    best = gemm_search_particle(
        slice_operands=counted,
        rotation_matrices=problem["rotations"],
        image=problem["image"],
        image_sum=problem["image_dft"][0, 0].real,
        filters=problem["filters"],
        corr_mean=corr_mean,
        corr_std=corr_std,
        center=start,
        shift_search_radius=0,
        orientation_batch_size=NUM_ORIENT,
    )

    assert (best.pos_y, best.pos_x) == start
    assert calls == [NUM_ORIENT]  # a single pass over the orientations


def test_score_hypotheses_gradient_matches_finite_differences():
    torch.manual_seed(5)
    problem = _problem(2.0)
    template_dft = problem["template_dft"].clone().requires_grad_()
    positions = torch.tensor([[2, 3]] * NUM_ORIENT)
    inputs = {
        "rotation_matrices": problem["rotations"],
        "filters": problem["filters"][:NUM_ORIENT],
        "particle_images_dft": problem["image_dft"].expand(NUM_ORIENT, -1, -1),
        "positions": positions,
    }
    with torch.enable_grad():
        score_hypotheses(template_dft=template_dft, **inputs).sum().backward()
    assert template_dft.grad is not None

    gen = torch.Generator().manual_seed(6)
    direction = torch.complex(
        torch.randn(template_dft.shape, generator=gen),
        torch.randn(template_dft.shape, generator=gen),
    )
    direction *= problem["template_dft"].abs().mean() / direction.abs().mean()
    eps = 1e-2
    plus = score_hypotheses(
        template_dft=problem["template_dft"] + eps * direction, **inputs
    )
    minus = score_hypotheses(
        template_dft=problem["template_dft"] - eps * direction, **inputs
    )
    numeric = float((plus.sum() - minus.sum()) / (2 * eps))
    # d/de f(x + e v) = Re <conj(grad), v> for complex inputs
    analytic = float((template_dft.grad.conj() * direction).real.sum())
    assert analytic == pytest.approx(numeric, rel=2e-2)


#########################################################
### End-to-end core_refine_template on a CPU workspace ###
#########################################################


def _workspace_kwargs(workspace) -> dict:
    workspace.df["refined_relative_defocus"] = 0.0
    stack = export_particle_stack(
        workspace.df,
        str(workspace.root / "stack.h5"),
        extracted_box_size=workspace.extracted_box_size,
        original_template_size=workspace.original_template_size,
    )
    manager = RefineTemplateManager(
        template_volume_path=str(workspace.template_path),
        particle_stack=stack,
        defocus_refinement_config=DefocusSearchConfig(
            enabled=True, defocus_min=-200.0, defocus_max=200.0, defocus_step=200.0
        ),
        pixel_size_refinement_config=PixelSizeSearchConfig(enabled=False),
        orientation_refinement_config=RefineOrientationConfig(
            enabled=True,
            psi_step_coarse=4.0,
            psi_step_fine=2.0,
            theta_step_coarse=4.0,
            theta_step_fine=2.0,
        ),
        preprocessing_filters=PreprocessingFilters(),
        computational_config=ComputationalConfigRefine(gpu_ids="cpu"),
        movie_config=MovieConfig(enabled=False),
    )
    kwargs = manager.make_backend_core_function_kwargs(prefer_refined_angles=False)
    return kwargs


def test_core_refine_template_backends_agree(workspace):
    kwargs = _workspace_kwargs(workspace)
    crop = kwargs["corr_mean"].shape[-1]
    assert crop > 3  # so a radius-1 window is smaller than the crop

    results = {
        "fft": core_refine_template(**kwargs, backend="fft"),
        "gemm_full": core_refine_template(
            **kwargs, backend="gemm", shift_search_radius=None, fft_centering=False
        ),
        # A window covering the whole crop must give the same answer
        "gemm_covering_window": core_refine_template(
            **kwargs, backend="gemm", shift_search_radius=crop // 2, fft_centering=True
        ),
    }
    expected = results["fft"]
    assert set(results["gemm_full"]) == set(expected)
    for name in ("gemm_full", "gemm_covering_window"):
        result = results[name]
        for key in (
            "angle_idx",
            "refined_pos_y",
            "refined_pos_x",
            "refined_defocus_offset",
            "refined_euler_angles",
        ):
            torch.testing.assert_close(result[key], expected[key], msg=f"{name} {key}")
        for key in ("refined_z_score", "refined_cross_correlation"):
            torch.testing.assert_close(
                result[key], expected[key], rtol=1e-4, atol=1e-4, msg=f"{name} {key}"
            )

    # A smaller window searches a subset of the hypotheses: never a better z-score,
    # and the same answer whenever it found the global position
    window = core_refine_template(
        **kwargs, backend="gemm", shift_search_radius=1, fft_centering=True
    )
    assert (window["refined_z_score"] <= expected["refined_z_score"] + 1e-4).all()
    found = (window["refined_pos_y"] == expected["refined_pos_y"]) & (
        window["refined_pos_x"] == expected["refined_pos_x"]
    )
    torch.testing.assert_close(
        window["refined_z_score"][found],
        expected["refined_z_score"][found],
        rtol=1e-4,
        atol=1e-4,
    )


def test_core_refine_template_is_differentiable(workspace):
    kwargs = _workspace_kwargs(workspace)
    plain = core_refine_template(**kwargs)
    assert not plain["refined_z_score"].requires_grad

    template_dft = kwargs["template_dft"].clone().requires_grad_()
    with torch.enable_grad():
        result = core_refine_template(**{**kwargs, "template_dft": template_dft})
        result["refined_z_score"].sum().backward()

    # Same search result; the re-scored values carry gradients
    torch.testing.assert_close(result["angle_idx"], plain["angle_idx"])
    torch.testing.assert_close(
        result["refined_z_score"].detach(),
        plain["refined_z_score"],
        rtol=1e-4,
        atol=1e-4,
    )
    assert template_dft.grad is not None
    assert torch.isfinite(template_dft.grad).all()
    assert template_dft.grad.abs().sum() > 0


def test_core_differentiable_refine_is_a_deprecated_alias(workspace):
    from leopard_em.backend import core_differentiable_refine

    kwargs = _workspace_kwargs(workspace)
    with pytest.warns(DeprecationWarning, match="core_refine_template"):
        result = core_differentiable_refine(**kwargs)
    torch.testing.assert_close(
        result["angle_idx"], core_refine_template(**kwargs)["angle_idx"]
    )


@pytest.mark.parametrize("backend", ["gemm", "fft"])
def test_core_refine_template_particle_batches_match_one_pass(workspace, backend):
    kwargs = _workspace_kwargs(workspace)
    num_particles = kwargs["particle_stack_dft"].shape[0]
    assert num_particles > 2  # so a batch of 2 leaves a partial last chunk

    expected = core_refine_template(**kwargs, backend=backend)
    for particle_batch_size in (1, 2, num_particles + 5):
        result = core_refine_template(
            **kwargs, backend=backend, particle_batch_size=particle_batch_size
        )
        assert set(result) == set(expected)
        for key, value in expected.items():
            torch.testing.assert_close(
                result[key], value, msg=f"batch of {particle_batch_size}: {key}"
            )


def test_core_refine_template_rejects_bad_options(workspace):
    kwargs = _workspace_kwargs(workspace)
    with pytest.raises(ValueError, match="backend"):
        core_refine_template(**kwargs, backend="streamed")
    with pytest.raises(ValueError, match="shift_search_radius"):
        core_refine_template(**kwargs, shift_search_radius=-1)
    with pytest.raises(ValueError, match="particle_batch_size"):
        core_refine_template(**kwargs, particle_batch_size=0)


def test_refine_config_backend_and_position_defaults():
    assert ComputationalConfigRefine().backend == "gemm"
    for alias in ("batched", "streamed", "fft"):
        assert ComputationalConfigRefine(backend=alias).backend == "fft"
    config = PositionRefinementConfig()
    assert (config.shift_search_radius, config.fft_centering) == (2, True)
    assert (
        PositionRefinementConfig(shift_search_radius=None).shift_search_radius is None
    )
    with pytest.raises(ValueError):
        PositionRefinementConfig(shift_search_radius=-1)


#####################################
### Triton slice kernel (CUDA only) ###
#####################################

requires_triton = pytest.mark.skipif(
    not can_use_triton_slices(
        torch.zeros((2, 2, 2), dtype=torch.complex64, device="cuda")
        if torch.cuda.is_available()
        else torch.zeros((2, 2, 2), dtype=torch.complex64)
    ),
    reason="needs CUDA and Triton",
)


@requires_triton
@pytest.mark.parametrize("pad_factor", [1.0, 2.0])
def test_triton_slices_match_torch_fourier_slice(pad_factor):
    device = torch.device("cuda")
    template_dft = volume_to_rfft_fourier_slice(
        _smooth_volume(32).to(device), pad_factor=pad_factor
    )
    rotations = roma.random_rotmat(64, device=device)
    padded = template_dft.shape[0]

    ours = TritonSliceExtractor(template_dft, (padded, padded))(rotations)
    reference = torch.fft.ifftshift(
        extract_template_fourier_slices(template_dft, rotations, (padded, padded)),
        dim=(-2,),
    )

    # torch-fourier-slice drops a sample now and then exactly at the volume
    # boundary (see slice_kernels docs); everything else agrees to round-off
    error = (ours - reference).abs()
    scale = reference.abs().max()
    boundary_drops = (reference == 0) & (ours != 0)
    assert boundary_drops.sum() <= 0.001 * reference.numel()
    assert error[~boundary_drops].max() <= 1e-5 * scale


@requires_triton
def test_triton_and_reference_gemm_search_agree():
    device = torch.device("cuda")
    torch.manual_seed(7)
    problem = {k: v.to(device) for k, v in _problem(2.0).items()}
    template_dft = problem["template_dft"]
    extractor = TritonSliceExtractor(template_dft, (SIZE, SIZE))
    reference_ops = reference_slice_operands(template_dft, (SIZE, SIZE))

    from leopard_em.backend.refine_gemm import (  # pylint: disable=C0415
        prepare_slices_for_scoring,
        slices_to_operands,
    )

    x_triton, p_triton = slices_to_operands(
        prepare_slices_for_scoring(extractor(problem["rotations"]), rows_centered=False)
    )
    x_ref, p_ref = reference_ops(problem["rotations"])
    torch.testing.assert_close(x_triton, x_ref, atol=1e-4 * x_ref.abs().max(), rtol=0)
    torch.testing.assert_close(p_triton, p_ref, atol=1e-4 * p_ref.abs().max(), rtol=0)
