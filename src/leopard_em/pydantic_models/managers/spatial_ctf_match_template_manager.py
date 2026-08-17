"""Match template with spatial real-space CTF pre-multiplication per defocus plane."""

from __future__ import annotations

import os
from typing import Any, ClassVar, Literal

import torch
import torch.distributed as dist
from pydantic import ConfigDict

from leopard_em.backend.process_results import (
    decode_global_search_index,
    merge_runs_pooled_zscore,
)
from leopard_em.pydantic_models.config.spatial_ctf_premultiply import (
    LinearDefocusSpatialConfig,
    QuadraticPhaseSpatialConfig,
    SpatialModelConfig,
    SpatialPsfConfig,
)
from leopard_em.pydantic_models.managers.match_template_manager import (
    MatchTemplateManager,
)
from leopard_em.utils.ctf_utils import calculate_ctf_filter_stack
from leopard_em.utils.data_io import load_mrc_image, write_mrc_from_tensor
from leopard_em.utils.spatial_ctf_fields import (
    defocus_linear_increment_vertex_grid,
    phase_quadratic_vertex_grid,
)
from leopard_em.utils.spatial_ctf_realspace import (
    _ctf_rfft_single,
    apply_spatial_psf_grid,
    build_psf_kernel_grid_defocus,
    build_psf_kernel_grid_phase,
    sample_template_slices_for_gain,
    template_ctf_unit_variance_gain,
)


class SpatialCtfMatchTemplateManager(MatchTemplateManager):
    """2DTM with per-plane spatial PSF pre-multiply, merged MIP and pooled statistics.

    Runs ``core_match_template`` once per entry in ``defocus_search_config`` with
    ``ctf_premultiplied=True`` and a single Fourier defocus plane (0 Å offset).
    ``relative_defocus`` maps store the winning spatial defocus offset (Å) per pixel.

    Every plane uses a vertex PSF grid (truncated kernel, not sum-normalized,
    circular convolution), including uniform fields. Image DFTs are scaled by
    ``||T W|| / ||T CTF(d_mean+d_k) W||`` so raw MIP is in Fourier-2DTM units.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    spatial_model: SpatialModelConfig
    spatial_psf: SpatialPsfConfig
    premultiplied_output_dir: str | None = None
    premultiplied_output_base_name: str | None = None
    ctf_premultiplied: Literal[True] = True

    def _mean_defocus_angstrom(self) -> float:
        og = self.optics_group
        return (og.defocus_u + og.defocus_v) / 2.0

    def _precompute_spatial_vertex_grids(
        self, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Vertex geometry shared across defocus planes (independent of ``d_k``).

        Returns ``(defocus_increment, phase_vertices)``; one is set per model type.
        """
        nx, ny = self.spatial_psf.grid_nx, self.spatial_psf.grid_ny
        sm = self.spatial_model
        if isinstance(sm, LinearDefocusSpatialConfig):
            inc = defocus_linear_increment_vertex_grid(
                nx,
                ny,
                sm.grad_mag_angstrom,
                sm.grad_angle_deg,
                device=device,
            )
            return (inc, None)
        if isinstance(sm, QuadraticPhaseSpatialConfig):
            phi_v = phase_quadratic_vertex_grid(
                nx,
                ny,
                sm.phase_c,
                sm.phase_g,
                sm.phase_k,
                sm.phase_alpha_deg,
                device=device,
            )
            return (None, phi_v)
        return (None, None)

    def premultiply_micrograph_for_offset(  # pylint: disable=too-many-locals
        self,
        d_k: float,
        device: torch.device,
        *,
        defocus_vertex_increment_grid: torch.Tensor | None = None,
        phase_vertex_grid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return real-space micrograph after spatial PSF correction.

        When looping over defocus offsets, pass grids from
        :meth:`_precompute_spatial_vertex_grids` so vertex geometry is not
        recomputed each iteration.

        Always uses the vertex PSF grid (circular convolution of truncated
        kernels). A full-image Fourier CTF multiply is available as
        :func:`apply_circular_ctf_multiply` for tests, not the production path.
        """
        img = load_mrc_image(self.micrograph_path).to(
            device=device, dtype=torch.float32
        )
        h, w = img.shape
        nx, ny = self.spatial_psf.grid_nx, self.spatial_psf.grid_ny
        ksz = self.spatial_psf.kernel_size
        image_shape = (h, w)
        d_mean = self._mean_defocus_angstrom()
        sm = self.spatial_model

        if isinstance(sm, LinearDefocusSpatialConfig):
            if defocus_vertex_increment_grid is None:
                inc = defocus_linear_increment_vertex_grid(
                    nx,
                    ny,
                    sm.grad_mag_angstrom,
                    sm.grad_angle_deg,
                    device=device,
                )
            else:
                inc = defocus_vertex_increment_grid.to(device=device)
            vertex_d = d_mean + d_k + inc
            kg = build_psf_kernel_grid_defocus(
                self.optics_group,
                vertex_d,
                self.optics_group.phase_shift,
                image_shape,
                ksz,
                device,
            )
            return apply_spatial_psf_grid(
                img,
                kg,
                grid_nx=nx,
                grid_ny=ny,
                blend="bilinear",
                kernel_size=ksz,
            )

        if isinstance(sm, QuadraticPhaseSpatialConfig):
            if self.optics_group.laser_params is not None:
                raise NotImplementedError(
                    "quadratic_phase is not implemented for laser_params optics."
                )
            if phase_vertex_grid is None:
                phi_v = phase_quadratic_vertex_grid(
                    nx,
                    ny,
                    sm.phase_c,
                    sm.phase_g,
                    sm.phase_k,
                    sm.phase_alpha_deg,
                    device=device,
                )
            else:
                phi_v = phase_vertex_grid.to(device=device)
            kg = build_psf_kernel_grid_phase(
                self.optics_group,
                d_mean + d_k,
                phi_v,
                image_shape,
                ksz,
                device,
            )
            return apply_spatial_psf_grid(
                img,
                kg,
                grid_nx=nx,
                grid_ny=ny,
                blend="biquadratic",
                kernel_size=ksz,
            )

        raise TypeError(f"Unsupported spatial_model: {type(sm)!r}")

    def _plane_template_ctf_gain(
        self,
        d_k: float,
        whitening_filter_template: torch.Tensor,
        template_shape: tuple[int, int],
        template_dft: torch.Tensor,
        euler_angles: torch.Tensor,
    ) -> torch.Tensor:
        """``||T W|| / ||T CTF(d_mean+d_k) W||`` averaged over sampled orientations."""
        sm = self.spatial_model
        if isinstance(sm, QuadraticPhaseSpatialConfig):
            ctf = _ctf_rfft_single(
                self.optics_group,
                self._mean_defocus_angstrom() + d_k,
                sm.phase_c,
                template_shape,
                whitening_filter_template.device,
            )
        else:
            ctf_stack = calculate_ctf_filter_stack(
                template_shape=template_shape,
                optics_group=self.optics_group,
                defocus_offsets=torch.tensor([d_k], dtype=torch.float32),
                pixel_size_offsets=torch.tensor([0.0], dtype=torch.float32),
            )
            ctf = ctf_stack[0, 0]
        slices = sample_template_slices_for_gain(template_dft, euler_angles)
        return template_ctf_unit_variance_gain(
            ctf, whitening_filter_template, slices
        )

    def make_backend_core_function_kwargs_with_image(
        self,
        image: torch.Tensor,
        *,
        defocus_values: torch.Tensor | None = None,
        whitening_ref_image: torch.Tensor | None = None,
        plane_defocus_offset: float = 0.0,
    ) -> dict[str, Any]:
        """Build backend kwargs from a premultiplied micrograph (Fourier defocus 0).

        Defocus is searched in real space; the Fourier stack uses a single plane unless
        ``defocus_values`` is passed explicitly. Whitening is taken from
        ``whitening_ref_image`` when provided (the unconvolved micrograph).

        ``plane_defocus_offset`` is the spatial search offset ``d_k`` (Å) used to
        scale the image DFT into Fourier-2DTM raw-MIP units.
        """
        if not self.ctf_premultiplied:
            raise ValueError(
                "SpatialCtfMatchTemplateManager requires ctf_premultiplied=True "
                "(real-space CTF correction is applied before correlation)."
            )
        dv = (
            defocus_values
            if defocus_values is not None
            else torch.tensor([0.0], dtype=torch.float32)
        )
        kwargs = super().make_backend_core_function_kwargs_with_image(
            image,
            defocus_values=dv,
            whitening_ref_image=whitening_ref_image,
        )
        template_h = int(kwargs["template_dft"].shape[-2])
        gain = self._plane_template_ctf_gain(
            plane_defocus_offset,
            kwargs["whitening_filter_template"],
            (template_h, template_h),
            kwargs["template_dft"],
            kwargs["euler_angles"],
        )
        image_dft = kwargs["image_dft"]
        kwargs["image_dft"] = image_dft * gain.to(
            device=image_dft.device, dtype=image_dft.real.dtype
        )
        return kwargs

    def _unconvolved_micrograph(self) -> torch.Tensor:
        """Load the original micrograph (no spatial PSF) as float32 CPU tensor."""
        img = load_mrc_image(self.micrograph_path)
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        return img.to(dtype=torch.float32)

    def run_premultiply_only(self) -> None:
        """Write spatially CTF-corrected micrographs for each defocus search value."""
        if (
            self.premultiplied_output_dir is None
            or self.premultiplied_output_base_name is None
        ):
            raise ValueError(
                "run_premultiply_only requires premultiplied_output_dir and "
                "premultiplied_output_base_name to be set."
            )
        os.makedirs(self.premultiplied_output_dir, exist_ok=True)
        device = torch.device(str(self.computational_config.gpu_devices[0]))
        defocus_inc, phase_v = self._precompute_spatial_vertex_grids(device)
        offsets = self.defocus_search_config.defocus_values
        n = int(offsets.numel())
        for k in range(n):
            d_k = float(offsets[k].item())
            out = self.premultiply_micrograph_for_offset(
                d_k,
                device,
                defocus_vertex_increment_grid=defocus_inc,
                phase_vertex_grid=phase_v,
            )
            path = os.path.join(
                self.premultiplied_output_dir,
                f"{self.premultiplied_output_base_name}_{k:03d}.mrc",
            )
            write_mrc_from_tensor(out.cpu(), path, overwrite=True)

    def run_spatial_match_template(  # pylint: disable=too-many-locals
        self,
        orientation_batch_size: int = 16,
        do_result_export: bool = True,
        do_valid_cropping: bool = False,
        compute_correlation_table: bool = False,
    ) -> None:
        """Premultiply per defocus plane, merge MIP and stats, 1 pooled z-score map.

        ``do_valid_cropping`` defaults to False: the match-template backend already
        writes valid-cropped statistic maps (same as ``run_match_template``).
        """
        offsets = self.defocus_search_config.defocus_values.cpu()
        n = int(offsets.numel())
        if n < 1:
            raise ValueError("defocus_search_config produced no defocus values.")

        device_mt = torch.device(str(self.computational_config.gpu_devices[0]))
        defocus_inc, phase_v = self._precompute_spatial_vertex_grids(device_mt)
        n_orient = int(self.orientation_search_config.euler_angles.shape[0])
        whitening_ref = self._unconvolved_micrograph()

        plane_runs: list[dict[str, Any]] = []

        # Loop over defocus search values.
        for k in range(n):
            d_k = float(offsets[k].item())
            img_corr = self.premultiply_micrograph_for_offset(
                d_k,
                device_mt,
                defocus_vertex_increment_grid=defocus_inc,
                phase_vertex_grid=phase_v,
            ).cpu()
            core_kwargs = self.make_backend_core_function_kwargs_with_image(
                img_corr,
                whitening_ref_image=whitening_ref,
                plane_defocus_offset=d_k,
            )
            results = self._invoke_core_match_template(
                core_kwargs,
                orientation_batch_size,
                compute_correlation_table=compute_correlation_table,
            )

            mip_k = results["mip"]
            if not isinstance(mip_k, torch.Tensor):
                mip_k = torch.tensor(mip_k, dtype=torch.float32)
            else:
                mip_k = mip_k.to(torch.float32).cpu()

            csum = results["correlation_sum"]
            csq = results["correlation_squared_sum"]
            bgi = results["best_global_index"]
            if not isinstance(csum, torch.Tensor):
                csum = torch.tensor(csum, dtype=torch.float32)
            else:
                csum = csum.cpu().to(torch.float32)
            if not isinstance(csq, torch.Tensor):
                csq = torch.tensor(csq, dtype=torch.float32)
            else:
                csq = csq.cpu().to(torch.float32)
            if not isinstance(bgi, torch.Tensor):
                bgi = torch.tensor(bgi, dtype=torch.int32)
            else:
                bgi = bgi.cpu().to(torch.int32)

            plane_runs.append(
                {
                    "mip": mip_k,
                    "best_global_index": bgi,
                    "correlation_sum": csum,
                    "correlation_squared_sum": csq,
                }
            )

        total_proj = n * n_orient
        merged_plane = merge_runs_pooled_zscore(
            plane_runs,
            pooled_total_correlation_positions=total_proj,
            run_tag_values=offsets,
        )
        mip_acc = merged_plane["mip"]
        mip_scaled = merged_plane["scaled_mip"]
        corr_mean = merged_plane["correlation_mean"]
        corr_var = merged_plane["correlation_variance"]
        best_gi = merged_plane["best_global_index"]
        best_spatial = merged_plane["best_run_tag"]

        pixel_values = torch.tensor([0.0], dtype=torch.float32)
        defocus_single = torch.tensor([0.0], dtype=torch.float32)
        euler = self.orientation_search_config.euler_angles.cpu().to(torch.float32)

        phi, theta, psi, _df, _px = decode_global_search_index(
            best_gi, pixel_values, defocus_single, euler
        )

        merged: dict[str, Any] = {
            "mip": mip_acc,
            "scaled_mip": mip_scaled,
            "best_phi": phi,
            "best_theta": theta,
            "best_psi": psi,
            "best_defocus": best_spatial,
            "correlation_mean": corr_mean,
            "correlation_variance": corr_var,
            "total_projections": total_proj,
            "total_orientations": n_orient,
            "total_defocus": n,
        }
        self._populate_match_template_result(
            merged,
            defocus_values=defocus_single,
            euler_angles=euler,
            do_result_export=do_result_export,
            do_valid_cropping=do_valid_cropping,
        )

    def run_spatial_match_template_distributed(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self,
        world_size: int,
        rank: int,
        local_rank: int,
        orientation_batch_size: int = 16,
        do_result_export: bool = True,
        do_valid_cropping: bool = False,
        compute_correlation_table: bool = False,
    ) -> None:
        """Distributed spatial match template; accumulation on rank 0 only.

        ``do_valid_cropping`` defaults to False (backend already valid-crops).
        """
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group has not been initialized.")

        offsets = self.defocus_search_config.defocus_values.cpu()
        n = int(offsets.numel())
        n_orient = int(self.orientation_search_config.euler_angles.shape[0])
        device_mt = torch.device(str(self.computational_config.gpu_devices[0]))
        local_device = torch.device(f"cuda:{local_rank}")
        defocus_inc, phase_v = self._precompute_spatial_vertex_grids(device_mt)
        whitening_ref = self._unconvolved_micrograph() if rank == 0 else None

        plane_runs: list[dict[str, Any]] = []

        for k in range(n):
            d_k = float(offsets[k].item())
            # pylint: disable=duplicate-code
            if rank == 0:
                img_corr = self.premultiply_micrograph_for_offset(
                    d_k,
                    device_mt,
                    defocus_vertex_increment_grid=defocus_inc,
                    phase_vertex_grid=phase_v,
                ).cpu()
                core_kwargs = self.make_backend_core_function_kwargs_with_image(
                    img_corr,
                    whitening_ref_image=whitening_ref,
                    plane_defocus_offset=d_k,
                )

            else:
                core_kwargs = {}

            results = self._invoke_core_match_template_distributed(
                world_size,
                rank,
                local_rank,
                local_device,
                core_kwargs,
                orientation_batch_size,
                compute_correlation_table=compute_correlation_table,
            )
            # pylint: enable=duplicate-code

            if rank != 0:
                continue

            mip_k = results["mip"]
            if not isinstance(mip_k, torch.Tensor):
                mip_k = torch.tensor(mip_k, dtype=torch.float32)
            else:
                mip_k = mip_k.to(torch.float32).cpu()

            csum = results["correlation_sum"]
            csq = results["correlation_squared_sum"]
            bgi = results["best_global_index"]
            if not isinstance(csum, torch.Tensor):
                csum = torch.tensor(csum, dtype=torch.float32)
            else:
                csum = csum.cpu().to(torch.float32)
            if not isinstance(csq, torch.Tensor):
                csq = torch.tensor(csq, dtype=torch.float32)
            else:
                csq = csq.cpu().to(torch.float32)
            if not isinstance(bgi, torch.Tensor):
                bgi = torch.tensor(bgi, dtype=torch.int32)
            else:
                bgi = bgi.cpu().to(torch.int32)

            plane_runs.append(
                {
                    "mip": mip_k,
                    "best_global_index": bgi,
                    "correlation_sum": csum,
                    "correlation_squared_sum": csq,
                }
            )

        if rank != 0:
            return

        total_proj = n * n_orient
        merged_plane = merge_runs_pooled_zscore(
            plane_runs,
            pooled_total_correlation_positions=total_proj,
            run_tag_values=offsets,
        )
        mip_acc = merged_plane["mip"]
        mip_scaled = merged_plane["scaled_mip"]
        corr_mean = merged_plane["correlation_mean"]
        corr_var = merged_plane["correlation_variance"]
        best_gi = merged_plane["best_global_index"]
        best_spatial = merged_plane["best_run_tag"]

        pixel_values = torch.tensor([0.0], dtype=torch.float32)
        defocus_single = torch.tensor([0.0], dtype=torch.float32)
        euler = self.orientation_search_config.euler_angles.cpu().to(torch.float32)

        phi, theta, psi, _df, _px = decode_global_search_index(
            best_gi, pixel_values, defocus_single, euler
        )

        merged: dict[str, Any] = {
            "mip": mip_acc,
            "scaled_mip": mip_scaled,
            "best_phi": phi,
            "best_theta": theta,
            "best_psi": psi,
            "best_defocus": best_spatial,
            "correlation_mean": corr_mean,
            "correlation_variance": corr_var,
            "total_projections": total_proj,
            "total_orientations": n_orient,
            "total_defocus": n,
        }
        self._populate_match_template_result(
            merged,
            defocus_values=defocus_single,
            euler_angles=euler,
            do_result_export=do_result_export,
            do_valid_cropping=do_valid_cropping,
        )
