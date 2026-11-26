"""Score given lattice sites with Panther-EM's SVD search -- a z at EVERY site, found or not.

The correlation table keeps a site only when one of its correlations clears the table's
cut (z 6.5). Once the lattice is known (Stage 4's curve, rise, phase and roll), every
site's pose is PREDICTED, so each site can be scored directly, whatever its score. This
script does that for a list of sites and any number of templates, on the SVD search's
own machinery: Panther-EM's features, CTF planes, and per-pixel moments over EVERY
template in closed form (the all-template normalisation of ``svd_match_filament.py``).
Only the reduction differs: in place of "every correlation above a cut" each site keeps

* ``z_center`` -- z at the grid hypothesis nearest its predicted pose, at its predicted
  pixel: nothing searched (Stage 6b's neutral-pose rule);
* ``z_max`` -- the maximum z over a small window around that pose: +/- ``--radius-px``
  pixels, +/- ``--angle-window`` degrees in phi, theta and psi, defocus planes within
  ``--defocus-window`` Angstroms; with where the maximum lies (``max_dx`` ... ``max_defocus``)
  and how many hypotheses the window held (``n_window``).

A pixel shared by two sites' windows (a front and a back protofilament can project onto
the same pixel) counts for each site with that site's own orientation window.

Runs in the Panther-EM environment (branch ``jd/mt-correlation-table``)::

    source MT_tests/run_scripts/panther_env.sh   # from the Leopard-EM root
    cd MT_tests
    CUDA_VISIBLE_DEVICES=1 python ../programs/constrained_search/svd_score_sites.py \\
        --config configs/match_tm_bundle_strong_patch.yaml \\
        --sites results_lattice_fill/sites_bundle_strong_patch.csv \\
        --template patch=maps/6dpu_atoms_6dpv_lattice_13pf_patch5_2rings_flatB_0.9194_bscale0.5.mrc \\
        --template patch_exp=maps/6dpu_13pf_patch5_2rings_flatB_0.9194_bscale0.5.mrc \\
        --out results_lattice_fill/scores_bundle_strong_patch.csv

Sites CSV: ``pos_x``, ``pos_y`` (template top-left in the micrograph, the frame of the
ORIGINAL template box -- Leopard-EM's ``pos_x``/``pos_y``), ``phi``, ``theta``, ``psi``
(degrees) and ``relative_defocus`` (Angstroms); other columns are carried through.
Output: one row per site, the input columns plus ``<key>_z_center``, ``<key>_z_max``,
... per template.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time

import mrcfile
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svd_match_filament import defocus_offsets  # noqa: E402
from svd_trace_maps import (  # noqa: E402
    b_envelope_3d,
    polar_options,
    prepare_template,
    remove_on_exit,
    require_free_space,
    write_mrc,
)


def wrap(values: torch.Tensor, period: float = 360.0) -> torch.Tensor:
    return torch.remainder(values + period / 2, period) - period / 2


class SiteTargets:
    """Per-site windows on the search's compact pixel columns, and the running results.

    ``slots`` is ``(n_columns, S)``: the sites whose pixel window holds each column
    (-1 padded). Per site: orientation window centre, psi index window, defocus plane
    window, centre hypothesis.
    """

    def __init__(self, sites: pd.DataFrame, columns: np.ndarray, grid: tuple[int, int],
                 shift: int, windows: list, n_psi: int, psi_half_width: int,
                 plane_offsets: np.ndarray, defocus_window: float, device: torch.device):
        n = len(sites)
        self.device = device
        self.n_sites = n
        self.n_psi = n_psi
        y = sites["pos_y"].to_numpy(np.int64) + shift
        x = sites["pos_x"].to_numpy(np.int64) + shift
        flat_of = {int(f): i for i, f in enumerate(columns)}
        # windows -> (column, site) pairs
        pairs_col, pairs_site = [], []
        for i in range(n):
            dy, dx = windows[i]
            yy, xx = y[i] + dy, x[i] + dx
            ok = (yy >= 0) & (xx >= 0) & (yy < grid[0]) & (xx < grid[1])
            for f in (yy[ok] * grid[1] + xx[ok]):
                pairs_col.append(flat_of[int(f)])
                pairs_site.append(i)
        pairs_col = np.asarray(pairs_col, np.int64)
        pairs_site = np.asarray(pairs_site, np.int64)
        order = np.lexsort((pairs_site, pairs_col))
        pairs_col, pairs_site = pairs_col[order], pairs_site[order]
        first = np.r_[0, np.flatnonzero(np.diff(pairs_col)) + 1]
        rank = np.arange(len(pairs_col)) - np.repeat(first, np.diff(np.r_[first, len(pairs_col)]))
        self.n_slots = int(rank.max()) + 1 if len(rank) else 1
        slots = np.full((len(columns), self.n_slots), -1, np.int64)
        slots[pairs_col, rank] = pairs_site
        self.slots = torch.from_numpy(slots).to(device)
        self.n_columns = len(columns)

        def t(values, dtype=torch.float32):
            return torch.as_tensor(np.asarray(values), dtype=dtype, device=device)

        self.phi = t(sites["phi"].to_numpy(float))
        self.theta = t(sites["theta"].to_numpy(float))
        psi_step = 360.0 / n_psi
        self.psi_index = t(np.round(sites["psi"].to_numpy(float) / psi_step).astype(np.int64) % n_psi,
                           torch.long)
        self.psi_offsets = torch.arange(-psi_half_width, psi_half_width + 1, device=device)
        defocus = sites["relative_defocus"].to_numpy(float)
        near = np.abs(defocus[:, None] - plane_offsets[None, :])
        plane_ok = near <= defocus_window + 1e-6
        plane_ok[np.arange(n), np.argmin(near, axis=1)] = True  # the nearest, always
        self.plane_ok = t(plane_ok, torch.bool)  # (n, D)
        self.center_plane = t(np.argmin(near, axis=1), torch.long)
        self.center_col = t([flat_of.get(int(yy * grid[1] + xx), -1) for yy, xx in zip(y, x)],
                            torch.long)
        self.center_j = torch.full((n,), -1, dtype=torch.long, device=device)  # set by .bind
        # results
        self.best_z = torch.full((n,), float("-inf"), device=device)
        self.best_col = torch.full((n,), -1, dtype=torch.long, device=device)
        self.best_j = torch.full((n,), -1, dtype=torch.long, device=device)
        self.best_plane = torch.full((n,), -1, dtype=torch.long, device=device)
        self.best_psi = torch.full((n,), -1, dtype=torch.long, device=device)
        self.center_z = torch.full((n,), float("nan"), device=device)
        self.center_value = torch.full((n,), float("nan"), device=device)
        self.center_mean = torch.full((n,), float("nan"), device=device)
        self.center_sd = torch.full((n,), float("nan"), device=device)
        self.n_window = torch.zeros(n, dtype=torch.long, device=device)

    def bind(self, row_phi: np.ndarray, row_theta: np.ndarray, angle_window: float,
             phi_window: float | None = None) -> None:
        """The stage's selected template rows (their phi, theta) and each site's
        centre row: the nearest in angle. ``phi_window`` (default ``angle_window``) is
        the roll's half-width -- e.g. half a protofilament period for a rival ring."""
        self.row_phi = torch.as_tensor(row_phi, dtype=torch.float32, device=self.device)
        self.row_theta = torch.as_tensor(row_theta, dtype=torch.float32, device=self.device)
        self.angle_window = float(angle_window)
        self.phi_window = float(angle_window if phi_window is None else phi_window)
        d2 = (wrap(self.row_phi[None] - self.phi[:, None]) ** 2
              + (self.row_theta[None] - self.theta[:, None]) ** 2)
        self.center_j = d2.argmin(dim=1)
        self.center_offset_deg = d2.min(dim=1).values.sqrt()

    @torch.no_grad()
    def reduce(self, corr: torch.Tensor, mean: torch.Tensor, sd: torch.Tensor,
               cols: torch.Tensor, d_local: int, j0: int, j1: int) -> None:
        """One batch: ``corr`` ``(P, H, n_psi)`` (psi axis as the irfft leaves it) at
        compact columns ``cols``, rows ``j0..j1`` of plane ``d_local``."""
        n_psi = self.n_psi
        valid = cols < self.n_columns
        safe = cols.clamp(max=self.n_columns - 1)
        row_phi, row_theta = self.row_phi[j0:j1], self.row_theta[j0:j1]
        for s in range(self.n_slots):
            sid = torch.where(valid, self.slots[safe, s], torch.full_like(cols, -1))
            r = (sid >= 0).nonzero()[:, 0]
            if r.numel() == 0:
                continue
            site = sid[r]
            keep = self.plane_ok[site, d_local]
            r, site = r[keep], site[keep]
            if r.numel() == 0:
                continue
            ang = ((wrap(row_phi[None] - self.phi[site, None]).abs() <= self.phi_window)
                   & ((row_theta[None] - self.theta[site, None]).abs() <= self.angle_window))
            has = ang.any(dim=1)
            r, site, ang = r[has], site[has], ang[has]
            if r.numel() == 0:
                continue
            psi_idx = (self.psi_index[site, None] + self.psi_offsets[None]) % n_psi  # (R, W)
            a = (n_psi - psi_idx) % n_psi
            block = corr[r]  # (R, H, n_psi)
            vals = block.gather(2, a[:, None, :].expand(-1, block.shape[1], -1))  # (R, H, W)
            z = (vals - mean[r, None, None]) / sd[r, None, None]
            z = z.masked_fill(~ang[:, :, None], float("-inf"))
            width = int(a.shape[1])
            zmax, arg = z.reshape(len(r), -1).max(dim=1)
            self.n_window.index_add_(0, site, ang.sum(dim=1) * width)
            self.best_z.scatter_reduce_(0, site, zmax, "amax")
            win = zmax >= self.best_z[site]
            if bool(win.any()):
                w_site = site[win]
                self.best_col[w_site] = cols[r[win]]
                self.best_j[w_site] = j0 + torch.div(arg[win], width, rounding_mode="floor")
                self.best_psi[w_site] = psi_idx[win].gather(1, (arg[win] % width)[:, None])[:, 0]
                self.best_plane[w_site] = d_local
            # the centre hypothesis, if it is in this batch
            cj = self.center_j[site]
            at = ((self.center_col[site] == cols[r]) & (self.center_plane[site] == d_local)
                  & (cj >= j0) & (cj < j1))
            if bool(at.any()):
                rr, ss = r[at], site[at]
                a0 = (n_psi - self.psi_index[ss]) % n_psi
                value = corr[rr, cj[at] - j0, a0]
                self.center_value[ss] = value
                self.center_mean[ss] = mean[rr]
                self.center_sd[ss] = sd[rr]
                self.center_z[ss] = (value - mean[rr]) / sd[rr]


def install(targets_ref: dict) -> None:
    """Route the emitting reduction into ``targets_ref['targets']``."""
    from panther_em.search.correlation import correlator, reduction
    from panther_em.search.featurization import store as store_module

    # Each block's first compact column, recorded as the store hands the block out
    # (before its loop runs). A stage can split its pixels into bands, each with its own
    # store whose blocks restart at column 0; the runner sets the sink's column_offset.
    for cls in (store_module.FeatureStore, store_module.ColumnSubset):
        if getattr(cls.blocks, "_records_start", False):
            continue
        def wrapped(self, groups, _original=cls.blocks):
            for block in _original(self, groups):
                targets_ref["block_start"] = int(block.col_range[0])
                yield block
        wrapped._records_start = True
        cls.blocks = wrapped

    class SiteBlockStats(reduction.EmittingBlockStats):
        """The emitting reduction with the cut replaced by the site windows."""

        def _consume(self, spectrum, batch, rows):
            n_psi = self.n_psi
            s1, s2, corr = reduction.moments_and_correlogram(spectrum.pnf(), n_psi)
            self.corr_sum[rows] += s1
            self.corr_sum2[rows] += s2
            flat = corr.reshape(corr.shape[0], -1)
            vmax, amax = flat.max(dim=1)
            best = self.best_corr[rows]
            improved = vmax > best
            hyp = batch.virtual_offset + torch.div(amax, n_psi, rounding_mode="floor")
            psi = (n_psi - amax % n_psi) % n_psi
            torch.where(improved, vmax, best, out=best)
            best_hyp = self.best_hypothesis[rows]
            torch.where(improved, hyp, best_hyp, out=best_hyp)
            best_psi = self.best_psi[rows]
            torch.where(improved, psi, best_psi, out=best_psi)
            if self.external is None:
                raise RuntimeError("site scoring needs the all-template moments up front")
            mean, sd = (v[rows] for v in self.external)
            sink = self.sink
            cols = (sink.column_offset + targets_ref["block_start"]
                    + torch.arange(rows.start, rows.start + corr.shape[0], device=corr.device))
            targets_ref["targets"].reduce(corr, mean, sd, cols, batch.d_local, batch.j0, batch.j1)

        @torch.no_grad()
        def finalize_into(self, outs, start, n_valid):
            if start != targets_ref["block_start"]:
                raise RuntimeError(f"block start {start} != {targets_ref['block_start']}")
            self.buffer.clear()
            super().finalize_into(outs, start, n_valid)

    original = correlator.make_block_stats

    def make(spec, *, fused):
        if reduction.emission_sink() is not None:
            return SiteBlockStats(n_sub=spec.n_sub, gemm_pixels=spec.gemm_pixels,
                                  n_psi=spec.n_psi, device=spec.device,
                                  sink=reduction.emission_sink())
        return original(spec, fused=fused)

    correlator.make_block_stats = make


def site_windows(sites: pd.DataFrame, a) -> list:
    """Per site, the ``(dy, dx)`` pixel offsets it is searched over: a square of
    +/-``--radius-px``, or -- with ``--along-px``/``--across-px`` and per-site unit tube
    directions ``axis_x``, ``axis_y`` in the CSV -- a rectangle aligned with the tube."""
    oriented = (a.along_px is not None and a.across_px is not None
                and {"axis_x", "axis_y"} <= set(sites.columns))
    reach = int(np.ceil(max(a.along_px, a.across_px))) if oriented else a.radius_px
    offs = np.arange(-reach, reach + 1)
    dy, dx = (v.ravel() for v in np.meshgrid(offs, offs, indexing="ij"))
    if not oriented:
        return [(dy, dx)] * len(sites)
    out = []
    for ux, uy in zip(sites["axis_x"].to_numpy(float), sites["axis_y"].to_numpy(float)):
        along = np.abs(dx * ux + dy * uy)
        across = np.abs(dx * uy - dy * ux)
        keep = (along <= a.along_px) & (across <= a.across_px)
        out.append((dy[keep], dx[keep]))
    return out


def score_template(key, template_path, cfg, sites, a, work_root, device, mic_path, shape,
                   targets_ref):
    from panther_em.managers import calibrate, decompose, whitening
    from panther_em.managers import CalibrateConfig, DecomposeConfig, WhiteningConfig
    from panther_em.managers.search import SearchConfig, _Run
    from panther_em.orientations import healpix_orientations
    from panther_em.search.correlation import reduction
    from panther_em.search.images import PixelSelection
    from panther_em.search.ledger import PixelLedger

    optics = cfg["optics_group"]
    pixel = float(optics["pixel_size"])
    offsets = np.asarray(defocus_offsets(cfg), float)
    work = work_root / key
    work.mkdir(parents=True, exist_ok=True)
    times = {}
    t = time.perf_counter()
    template, tinfo = prepare_template(template_path, 1, a.mass_fraction, a.box_margin, device)
    if any(tinfo["centring_shift_px_zyx"]):
        raise NotImplementedError(f"{template_path} is off-centre: {tinfo['com_offset_in_px_zyx']}")
    if a.b_factor > 0:
        template = b_envelope_3d(template, pixel, a.b_factor)
    box = template.shape[0]
    with mrcfile.mmap(str(template_path), permissive=True) as handle:
        out_box = int(handle.data.shape[-1])
    shift = out_box // 2 - box // 2
    tpl_path = work / "template.mrc"
    write_mrc(tpl_path, template, pixel)
    grid = (shape[0] - box + 1, shape[1] - box + 1)
    whitening_path = work_root / "whitening.h5"
    if not whitening_path.exists():
        whitening.run(WhiteningConfig(micrographs=[mic_path], output=whitening_path,
                                      band_cutoff=a.band_cutoff, band_falloff=a.band_falloff,
                                      device=str(device)))
    num_angle = max(256, 2 * a.k_max, int(np.ceil(1.875 * box / 16)) * 16)
    num_radius = int(np.ceil(1.5 * box / 16)) * 16
    decompose.run(DecomposeConfig.model_validate({
        "volume": str(tpl_path), "whitening": str(whitening_path),
        "output": str(work / "decomposition.h5"),
        "polar_grid": {"kind": "spiral", "num_angle": num_angle, "num_radius": num_radius,
                       "options": polar_options(box, a.polar_center)},
        "healpix_nside": a.nside, "k_max": a.k_max, "eig_max": a.eig_max, "svd_method": "low_rank",
        "device": str(device)}))
    k, m = a.rank
    calibrate.run(CalibrateConfig.model_validate({
        "volume": str(tpl_path), "whitening": str(whitening_path),
        "decomposition": str(work / "decomposition.h5"), "output_dir": str(work / "calibration"),
        "tables": {"table": [[0, 0, k, m]]}, "suggest": [{"max_error": 0.1}],
        "device": str(device)}))
    report = json.loads((work / "calibration" / "calibration.json").read_text())
    rho = report["tables"]["table"]["truncation_error"]["rho"]
    rho_k = report["tables"]["table"]["truncation_error"].get("rho_k", float("nan"))
    rho_m = report["tables"]["table"]["truncation_error"].get("rho_m", float("nan"))
    times["template+decompose+calibrate"] = time.perf_counter() - t

    # orientations a site window can reach: the stage's template rows
    phi_all, theta_all = healpix_orientations(a.nside)
    phi_s, theta_s = sites["phi"].to_numpy(float), sites["theta"].to_numpy(float)
    phi_window = a.phi_windows.get(key, a.angle_window)
    reach = np.zeros(len(phi_all), bool)
    step = 2.0 * np.degrees(np.sqrt(4 * np.pi / len(phi_all)))
    for lo in range(0, len(sites), 512):
        dphi = (phi_all[None] - phi_s[lo:lo + 512, None] + 180.0) % 360.0 - 180.0
        dth = theta_all[None] - theta_s[lo:lo + 512, None]
        # one grid step beyond the window, so every site's nearest row is in the set
        reach |= ((np.abs(dphi) <= phi_window + step)
                  & (np.abs(dth) <= a.angle_window + step)).any(axis=0)
    rows = np.flatnonzero(reach)

    # pixels: every site's window, in the search's (cropped-box) frame
    y = sites["pos_y"].to_numpy(np.int64) + shift
    x = sites["pos_x"].to_numpy(np.int64) + shift
    windows = site_windows(sites, a)
    mask = np.zeros(grid, bool)
    for yy, xx, (dy, dx) in zip(y, x, windows):
        wy, wx = yy + dy, xx + dx
        ok = (wy >= 0) & (wx >= 0) & (wy < grid[0]) & (wx < grid[1])
        mask[wy[ok], wx[ok]] = True
    outside = (y < 0) | (x < 0) | (y >= grid[0]) | (x >= grid[1])
    columns = np.flatnonzero(mask.ravel())

    stage = {"name": "sites", "rectangles": [[0, 0, k, m]], "table": "table",
             "precision": a.precision, "n_psi": a.n_psi,
             "orientations": [int(i) for i in rows],
             "decision": {"threshold": 1e9, "accept": False}}
    config = SearchConfig.model_validate({
        "decomposition": str(work / "decomposition.h5"), "whitening": str(whitening_path),
        "tables": {"table": str(work / "calibration" / "tables_table.h5")},
        "optics": {"pixel_size": pixel, "voltage": float(optics["voltage"]),
                   "spherical_aberration": float(optics["spherical_aberration"]),
                   "amplitude_contrast_ratio": float(optics["amplitude_contrast_ratio"]),
                   "phase_shift": float(optics.get("phase_shift", 0.0))},
        "micrographs": [{"path": str(mic_path), "defocus_u": float(optics["defocus_u"]),
                         "defocus_v": float(optics["defocus_v"]),
                         "astigmatism_angle": float(optics["astigmatism_angle"])}],
        "defocus_offsets": [float(v) for v in offsets], "invert_contrast": not a.no_invert,
        "stages": [stage], "weights_cells": [0, 0, k, m],
        "execution": {"show_progress": False}, "output_dir": str(work / "search"),
        "cache_micrographs": False, "overwrite": True, "keep_stage_results": False,
        "device": str(device)})
    t = time.perf_counter()
    state = _Run(config)
    micrograph = state.micrograph(config.micrographs[0])
    searcher = state.searcher((shape[0], shape[1]))
    session = searcher.session(micrograph)
    space = searcher.space
    template_ids, defocus_ids = searcher.stages[0].hypothesis_ids(space, searcher.ipix)
    if defocus_ids is not None:
        raise RuntimeError("expected every defocus plane in the stage")
    _, filt, orient = (v.numpy() for v in space.decode(template_ids.cpu()))
    if np.any(filt != 0):
        raise RuntimeError("expected one Fourier filter")
    targets = SiteTargets(sites, columns, grid, shift, windows, a.n_psi,
                          int(np.ceil(a.angle_window / (360.0 / a.n_psi))), offsets,
                          a.defocus_window, device)
    targets.bind(phi_all[orient], theta_all[orient], a.angle_window, phi_window)
    targets_ref["targets"] = targets
    initial = PixelSelection.from_mask(session.geometry, torch.from_numpy(mask)[None].to(device))
    if not np.array_equal(initial.index.cpu().numpy(), columns):
        raise RuntimeError("pixel selection is not the mask's sorted flat index")
    session.ledger = PixelLedger(session.geometry, session.ledger.device, initial=initial)
    session.pixels = initial
    times["prepare"] = time.perf_counter() - t
    t = time.perf_counter()
    previous = reduction._EMIT
    reduction._EMIT = reduction.EmittedEntries(z_cut=float("inf"), normalize="all")
    try:
        result = next(iter(session))
    finally:
        reduction._EMIT = previous
    times["score"] = time.perf_counter() - t
    if not np.array_equal(result.pixel_index.cpu().numpy(), columns):
        raise RuntimeError("the stage's compact column order is not the selection's")

    # check: no site's window maximum exceeds the stage's own per-pixel maximum
    stage_z = result.maps.zscore.cpu()
    best_col = targets.best_col.cpu()
    ok = best_col >= 0
    excess = (targets.best_z.cpu()[ok] - stage_z[best_col[ok]]).max().item() if ok.any() else 0.0
    if excess > 1e-3:
        raise RuntimeError(f"a site's window maximum exceeds the pixel's maximum by {excess:.4f}")

    missing = int((torch.isnan(targets.center_z).cpu().numpy() & ~outside).sum())
    if missing:
        raise RuntimeError(f"{missing} sites inside the grid never met their centre hypothesis")
    gy, gx = np.divmod(columns, grid[1])
    out = pd.DataFrame(index=sites.index)
    bc = targets.best_col.cpu().numpy()
    bj = targets.best_j.cpu().numpy()
    out[f"{key}_z_center"] = targets.center_z.cpu().numpy()
    out[f"{key}_z_max"] = targets.best_z.cpu().numpy()
    out[f"{key}_max_dx"] = np.where(bc >= 0, gx[np.maximum(bc, 0)] - x, 0)
    out[f"{key}_max_dy"] = np.where(bc >= 0, gy[np.maximum(bc, 0)] - y, 0)
    out[f"{key}_max_phi"] = np.where(bj >= 0, phi_all[orient][np.maximum(bj, 0)], np.nan)
    out[f"{key}_max_theta"] = np.where(bj >= 0, theta_all[orient][np.maximum(bj, 0)], np.nan)
    out[f"{key}_max_psi"] = targets.best_psi.cpu().numpy() * 360.0 / a.n_psi
    out[f"{key}_max_defocus"] = offsets[np.maximum(targets.best_plane.cpu().numpy(), 0)]
    out[f"{key}_n_window"] = targets.n_window.cpu().numpy()
    out[f"{key}_center_offset_deg"] = targets.center_offset_deg.cpu().numpy()
    out[f"{key}_center_mean"] = targets.center_mean.cpu().numpy()
    out[f"{key}_center_sd"] = targets.center_sd.cpu().numpy()
    out.loc[outside, [c for c in out.columns if "_z_" in c]] = np.nan
    print(f"[{key}] {template_path.name}: crop {box}^3 of {out_box}^3 (shift {shift} px); "
          f"rank {k}x{m}, k_max {a.k_max}, n_psi {a.n_psi}, band {a.band_cutoff}, B {a.b_factor}; truncation error "
          f"{rho:.3f} (angular {rho_k:.3f}, radial {rho_m:.3f}); {len(rows)} of {len(phi_all)} orientations "
          f"reachable; {len(columns):,} window pixels ({targets.n_slots} slot(s)); "
          f"{int(outside.sum())} sites outside the grid; window max <= pixel max "
          f"(excess {excess:+.1e}); " + ", ".join(f"{n} {v:.1f} s" for n, v in times.items()),
          flush=True)
    del session, searcher, state, result
    torch.cuda.empty_cache()
    if not a.keep_work:  # the decomposition alone is ~2 GB per template
        shutil.rmtree(work, ignore_errors=True)
    return out, {"box": box, "out_box": out_box, "shift": shift, "truncation_error": rho, "rho_k": rho_k, "rho_m": rho_m,
                 "n_rows": int(len(rows)), "n_pixels": int(len(columns)), "times_s": times}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=pathlib.Path, required=True, help="Leopard-EM match_tm_*.yaml")
    ap.add_argument("--sites", type=pathlib.Path, required=True)
    ap.add_argument("--template", action="append", required=True, metavar="KEY=MRC",
                    help="a template to score, repeatable")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--radius-px", type=int, default=3, help="pixel window half-width")
    ap.add_argument("--along-px", type=float, default=None,
                    help="with --across-px and per-site axis_x/axis_y columns: a window "
                         "aligned with the tube, +/- this along it")
    ap.add_argument("--across-px", type=float, default=None,
                    help="+/- this across the tube (see --along-px)")
    ap.add_argument("--angle-window", type=float, default=3.0,
                    help="+/- degrees in phi, theta and psi")
    ap.add_argument("--phi-window", action="append", default=[], metavar="KEY=DEG",
                    help="a template's own roll half-width (default --angle-window); "
                         "repeatable, e.g. 12pf=15 for half a 12-PF period")
    ap.add_argument("--defocus-window", type=float, default=200.0,
                    help="planes within this many Angstroms of the site's defocus")
    ap.add_argument("--rank", type=int, nargs=2, default=[96, 80], metavar=("K", "M"))
    ap.add_argument("--precision", default="fp32", choices=["fp16", "tf32", "fp32"])
    ap.add_argument("--nside", type=int, default=32)
    ap.add_argument("--k-max", type=int, default=128)
    ap.add_argument("--n-psi", type=int, default=256)
    ap.add_argument("--eig-max", type=int, default=128, help="radial components kept (M <= this)")
    ap.add_argument("--band-cutoff", type=float, default=0.3)
    ap.add_argument("--band-falloff", type=float, default=0.05)
    ap.add_argument("--mass-fraction", type=float, default=0.99)
    ap.add_argument("--box-margin", type=float, default=1.15)
    ap.add_argument("--no-invert", action="store_true")
    ap.add_argument("--polar-center", choices=["fft", "panther"], default="fft",
                    help="centre of the in-plane rotation: fft = box // 2 (Leopard-EM's "
                         "and the projector's), panther = the installed Panther-EM's own "
                         "default (box/2 - 0.5 before the 2026-11-12 fix)")
    ap.add_argument("--b-factor", type=float, default=None,
                    help="CTF B-factor envelope (A^2) on the template, as Leopard-EM's "
                         "ctf_B_factor (Panther-EM's own CTF has none); default: the "
                         "config's optics_group.ctf_B_factor")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep each template's decomposition, calibration and search files "
                         "(deleted after scoring by default: ~2 GB per template)")
    a = ap.parse_args()

    a.phi_windows = {k: float(v) for k, v in (spec.split("=", 1) for spec in a.phi_window)}
    cfg = yaml.safe_load(a.config.read_text())
    if a.b_factor is None:
        a.b_factor = float(cfg["optics_group"].get("ctf_B_factor", 0.0) or 0.0)
    mic_path = pathlib.Path(cfg["micrograph_path"])
    with mrcfile.mmap(str(mic_path), permissive=True) as handle:
        shape = tuple(int(n) for n in handle.data.shape)
    sites = pd.read_csv(a.sites, index_col=0)
    device = torch.device("cuda:0")
    work_root = a.out.parent / f"{a.out.stem}_svd_work"
    # one template's decomposition at a time (each folder is deleted after its template),
    # all of them with --keep-work
    require_free_space(a.out.parent, 3.0 * (len(a.template) if a.keep_work else 1),
                       "svd_score_sites")
    work_root.mkdir(parents=True, exist_ok=True)
    remove_on_exit(work_root, a.keep_work)
    targets_ref: dict = {}
    install(targets_ref)
    print(f"B-factor {a.b_factor} A^2, polar grid centred by '{a.polar_center}'")
    window = (f"+/-{a.along_px} px along x +/-{a.across_px} px across the tube"
              if a.along_px is not None and a.across_px is not None
              and {"axis_x", "axis_y"} <= set(sites.columns) else f"+/-{a.radius_px} px")
    print(f"{len(sites):,} sites; window {window}, +/-{a.angle_window} deg, "
          f"defocus +/-{a.defocus_window} A; normalised over every template", flush=True)
    parts, meta = [sites], {}
    for spec in a.template:
        key, path = spec.split("=", 1)
        part, info = score_template(key, pathlib.Path(path), cfg, sites, a, work_root, device,
                                    mic_path, shape, targets_ref)
        parts.append(part)
        meta[key] = {"template": path, **info}
    out = pd.concat(parts, axis=1)
    out.to_csv(a.out)
    (a.out.parent / f"{a.out.stem}_meta.json").write_text(json.dumps(
        {"args": {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(a).items()},
         "templates": meta}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
