"""Stage 2 on SVD: Panther-EM search of a filament band that keeps every detection.

Drop-in for ``MT_tests/run_scripts/run_match_curved.py``: same inputs (a Leopard-EM
``match_tm_*.yaml`` and the filament constraint YAML), but the search is Panther-EM's
SVD search and the output is every correlation with per-pixel z >= ``--z-cut``, not
only each pixel's maximum -- the raw material of Leopard-EM's correlation table.
``svd_entries_to_table.py`` (Leopard-EM's environment) then applies the constraint's
per-pixel orientation gate and writes the table itself.

Runs in the Panther-EM environment (Panther-EM branch ``jd/emit-entries``, which adds the
entry emission; mgiammar/Panther-EM#25)::

    source MT_tests/run_scripts/panther_env.sh   # from the Leopard-EM root
    cd MT_tests
    CUDA_VISIBLE_DEVICES=1 python ../programs/constrained_search/svd_match_filament.py \\
        --config configs/match_tm_gdp_curved_13pf.yaml \\
        --constraint configs/filament_constraint_gdp_curved.yaml \\
        --out results_svd/gdp_curved_13pf

Search
------
* Template: the config's ring, centred on its mass and cropped to it (see
  ``svd_trace_maps.prepare_template``); Panther-EM rotates about the box centre.
* Orientations: HEALPix ``--nside`` view directions, every phi (the roll is measured),
  ``--n-psi`` psi; the table stage correlates only theta inside the constraint's cone
  (+ ``--theta-margin``). Defocus planes: the config's defocus search.
* Normalisation (``--normalize``): ``all`` (default) -- each pixel's mean and sd over
  EVERY template of the full sphere, in closed form, as Leopard-EM's constrained search
  does (the constraint restricts what may be detected, not what z is measured
  against); ``band`` -- over the theta band only (the decomposition is then the band).
* Pixels: the constraint's eligible band only.
* Stage A (``--screen``, off by default): low rank ``--screen-rank``, fp16; the threshold
  test at the cut drops pixels whose exact maximum cannot reach it. Near z 6.5 noise
  maxima over millions of hypotheses sit too close to the cut for a low-rank test to
  reject anything (GDP: 99.99% kept), so it only pays at a high cut.
* Stage B (find): rank ``--rank`` on the fused kernel (fp16), every band orientation;
  keeps pixels whose maximum reaches ``--z-cut - --find-margin``. A pixel holds a table
  entry only if its maximum reaches the cut, so this bounds the emitting pass.
* Stage C (table): the same rank at ``--precision`` on the reference reduction (about 7x
  slower than the fused kernel, hence B), at B's survivors, emitting each (pixel, plane,
  orientation, psi) with z >= ``--z-cut`` under the chosen normalisation (the maps use
  the same).

Writes ``<out>_svd.h5``: the Panther-EM result (Leopard-EM maps in ``tensors``) plus
``entries/`` and ``orientations/`` groups and the run's metadata. The work folder
``<out>_svd_work`` (~2 GB: whitening, decomposition, calibration) is deleted when the run
ends, also on failure, unless ``--keep-work``; the run refuses to start when it would
leave less than ``SVD_MIN_FREE_GB`` (default 30) free.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import h5py
import mrcfile
import numpy as np
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svd_trace_maps import (b_envelope_3d, polar_options, prepare_template,  # noqa: E402
                            remove_on_exit, require_free_space, write_mrc)


def defocus_offsets(cfg: dict) -> list[float]:
    d = cfg.get("defocus_search_config") or {}
    if not d.get("enabled", False):
        return [0.0]
    lo, hi, step = float(d["defocus_min"]), float(d["defocus_max"]), float(d["defocus_step"])
    return [float(v) for v in np.arange(lo, hi + step / 2, step)]


def band(nside: int, centre: float, half: float):
    from panther_em.orientations import healpix_orientations

    phi, theta = healpix_orientations(nside)
    keep = np.abs(theta - centre) <= half
    return phi[keep], theta[keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=pathlib.Path, required=True, help="Leopard-EM match_tm_*.yaml")
    ap.add_argument("--constraint", type=pathlib.Path, required=True, help="filament constraint YAML")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="output prefix")
    ap.add_argument("--z-cut", type=float, default=6.5,
                    help="Leopard-EM's tables cut raw CC at 6.5 (noise sd ~1), i.e. about z 6.5")
    ap.add_argument("--find-margin", type=float, default=0.5,
                    help="stage B keeps pixels with max z >= z-cut - this (fp16, and its z "
                         "is over the band where --normalize all uses every template)")
    ap.add_argument("--margin", type=float, default=0.5,
                    help="z below the cut kept while a block's moments are still accumulating")
    ap.add_argument("--rank", type=int, nargs=2, default=[96, 80], metavar=("K", "M"),
                    help="rank of the emitting (table) pass")
    ap.add_argument("--find-rank", type=int, nargs=2, default=None, metavar=("K", "M"),
                    help="rank of the find pass (default: --rank)")
    ap.add_argument("--screen-rank", type=int, nargs=2, default=[48, 40], metavar=("K", "M"))
    ap.add_argument("--screen", action="store_true", help="add the low-rank stage A")
    ap.add_argument("--screen-calibration", type=float, nargs=4, default=[0.02, 0.98, 0.1, 10.0],
                    metavar=("ALPHA", "ZETA", "EPS_RHO", "B"))
    ap.add_argument("--precision", default="fp32", choices=["fp16", "tf32", "fp32"])
    ap.add_argument("--normalize", choices=["all", "band"], default="all",
                    help="z over every template (default) or over the theta band only")
    ap.add_argument("--nside", type=int, default=32)
    ap.add_argument("--k-max", type=int, default=128,
                    help="angular frequencies stored by the decomposition (k_stop <= n_psi/2 + 1)")
    ap.add_argument("--n-psi", type=int, default=256)
    ap.add_argument("--theta-margin", type=float, default=2.0,
                    help="degrees searched beyond the constraint's theta cone")
    ap.add_argument("--band-cutoff", type=float, default=0.3, help="cycles/px (Nyquist 0.5)")
    ap.add_argument("--band-falloff", type=float, default=0.05)
    ap.add_argument("--mass-fraction", type=float, default=0.99)
    ap.add_argument("--box-margin", type=float, default=1.15)
    ap.add_argument("--no-invert", action="store_true")
    ap.add_argument("--polar-center", choices=["fft", "panther"], default="fft",
                    help="centre of the in-plane (psi) rotation: fft = box // 2, where "
                         "Fourier-slice projections (Panther-EM's and Leopard-EM's) put the "
                         "template's centre; panther = the installed Panther-EM's own default "
                         "(box/2 - 0.5 before the 2026-11-12 fix: shifts every peak by "
                         "(I - R(-psi)) (1/2, 1/2) px)")
    ap.add_argument("--b-factor", type=float, default=None,
                    help="CTF B-factor envelope on the template, as Leopard-EM's "
                         "ctf_B_factor (Panther-EM's CTF has none); default: the config's")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep <out>_svd_work (~2 GB; 4-7 GB at k 256 or nside 64) after the run")
    a = ap.parse_args()

    from panther_em.managers import calibrate, decompose, whitening
    from panther_em.managers import CalibrateConfig, DecomposeConfig, WhiteningConfig
    from panther_em.managers import leopard
    from panther_em.managers.search import SearchConfig, _Run
    from panther_em.search.correlation.reduction import emit_entries
    from panther_em.search.images import PixelSelection
    from panther_em.search.ledger import PixelLedger

    cfg = yaml.safe_load(a.config.read_text())
    cons = yaml.safe_load(a.constraint.read_text())
    optics = cfg["optics_group"]
    pixel = float(optics["pixel_size"])
    mic_path = pathlib.Path(cfg["micrograph_path"])
    offsets = defocus_offsets(cfg)
    device = torch.device("cuda:0")
    work = a.out.parent / f"{a.out.name}_svd_work"
    require_free_space(a.out.parent, 4.0, "svd_match_filament")
    work.mkdir(parents=True, exist_ok=True)
    remove_on_exit(work, a.keep_work)
    times: dict[str, float] = {}

    # template: centred, cropped to its mass
    t = time.perf_counter()
    template, tinfo = prepare_template(pathlib.Path(cfg["template_volume_path"]), 1,
                                       a.mass_fraction, a.box_margin, device)
    if any(tinfo["centring_shift_px_zyx"]):
        raise NotImplementedError(
            f"{cfg['template_volume_path']} is off-centre by {tinfo['com_offset_in_px_zyx']} px: "
            "re-centring moves detections along the axis; correct positions before using it here")
    if a.b_factor is None:
        a.b_factor = float(optics.get("ctf_B_factor", 0.0) or 0.0)
    if a.b_factor > 0:
        template = b_envelope_3d(template, pixel, a.b_factor)
    box = template.shape[0]
    tpl_path = work / "template.mrc"
    write_mrc(tpl_path, template, pixel)
    with mrcfile.open(str(mic_path), permissive=True) as handle:
        shape = tuple(int(n) for n in handle.data.shape)
    grid = (shape[0] - box + 1, shape[1] - box + 1)
    half = box // 2
    print(f"template {pathlib.Path(cfg['template_volume_path']).name}: {tinfo['template_box_in']}^3 -> "
          f"crop {box}^3 (99% of mass within {tinfo['mass_radius_px']:.0f} px); micrograph {shape}, "
          f"valid grid {grid}")
    times["template"] = time.perf_counter() - t

    # whitening, decomposition over the theta band, error tables
    t = time.perf_counter()
    whitening.run(WhiteningConfig(micrographs=[mic_path], output=work / "whitening.h5",
                                  band_cutoff=a.band_cutoff, band_falloff=a.band_falloff,
                                  device=str(device)))
    theta_c, cone = float(cons.get("theta_center_deg", 90.0)), float(cons["cone_half_angle_deg"])
    from panther_em.orientations import healpix_orientations

    phi, theta = healpix_orientations(a.nside)
    in_band = np.abs(theta - theta_c) <= cone + a.theta_margin
    if a.normalize == "band":  # decompose the band only: it is the whole template set
        phi, theta = phi[in_band], theta[in_band]
        in_band = np.ones(len(phi), bool)
        decompose.healpix_orientations = lambda nside: (phi, theta)
    band_ids = np.nonzero(in_band)[0]
    num_angle = max(256, 2 * a.k_max, int(np.ceil(1.875 * box / 16)) * 16)
    num_radius = int(np.ceil(1.5 * box / 16)) * 16
    decompose.run(DecomposeConfig.model_validate({
        "volume": str(tpl_path), "whitening": str(work / "whitening.h5"),
        "output": str(work / "decomposition.h5"),
        "polar_grid": {"kind": "spiral", "num_angle": num_angle, "num_radius": num_radius,
                       "options": polar_options(box, a.polar_center)},
        "healpix_nside": a.nside, "k_max": a.k_max, "eig_max": 128, "svd_method": "low_rank",
        "device": str(device)}))
    times["whitening+decompose"] = time.perf_counter() - t
    t = time.perf_counter()
    (ks, ms), (k, m) = a.screen_rank, a.rank
    kf, mf = a.find_rank or a.rank
    tables = {"table": [[0, 0, k, m]]}
    if (kf, mf) != (k, m):
        tables["find"] = [[0, 0, kf, mf]]
    if a.screen:
        tables["screen"] = [[0, 0, ks, ms]]
    calibrate.run(CalibrateConfig.model_validate({
        "volume": str(tpl_path), "whitening": str(work / "whitening.h5"),
        "decomposition": str(work / "decomposition.h5"), "output_dir": str(work / "calibration"),
        "tables": tables, "suggest": [{"max_error": 0.2}, {"max_error": 0.1}, {"max_error": 0.05}],
        "device": str(device)}))
    times["calibrate"] = time.perf_counter() - t
    report = json.loads((work / "calibration" / "calibration.json").read_text())
    print(f"{len(phi)} orientations decomposed, {len(band_ids)} in the theta band {theta_c} +/- "
          f"{cone + a.theta_margin}; {len(offsets)} planes, {a.n_psi} psi; normalise over "
          f"{'every template' if a.normalize == 'all' else 'the band'}; truncation error: "
          + "; ".join(f"{tag} {r['truncation_error']['rho']:.3f}" for tag, r in report["tables"].items()))

    # the searcher (the manager's own construction)
    table_stage = {"name": "table", "rectangles": [[0, 0, k, m]], "table": "table",
                   "precision": a.precision, "n_psi": a.n_psi,
                   "decision": {"threshold": a.z_cut, "accept": True}}
    find_stage = {"name": "find", "rectangles": [[0, 0, kf, mf]],
                  "table": "find" if (kf, mf) != (k, m) else "table",
                  "precision": "fp16", "n_psi": a.n_psi,
                  "decision": {"threshold": a.z_cut - a.find_margin, "accept": False}}
    if a.normalize == "all":
        table_stage["orientations"] = [int(i) for i in band_ids]
        find_stage["orientations"] = [int(i) for i in band_ids]
    stages = [find_stage, table_stage]
    if a.screen:
        alpha, zeta, eps, bf = a.screen_calibration
        stages.insert(0, {"name": "screen", "rectangles": [[0, 0, ks, ms]], "table": "screen",
                          "precision": "fp16", "n_psi": a.n_psi,
                          "seeding": {"level": 1, "rings": 2, "templates": "cell_mean",
                                      "lattice": 2, "top_k": 3},
                          "decision": {"threshold": a.z_cut, "accept": False,
                                       "calibration": {"alpha": alpha, "zeta": zeta,
                                                       "eps_rho": eps, "b_factor": bf}}})
    config = SearchConfig.model_validate({
        "decomposition": str(work / "decomposition.h5"), "whitening": str(work / "whitening.h5"),
        "tables": {tag: str(work / "calibration" / f"tables_{tag}.h5") for tag in tables},
        "optics": {"pixel_size": pixel, "voltage": float(optics["voltage"]),
                   "spherical_aberration": float(optics["spherical_aberration"]),
                   "amplitude_contrast_ratio": float(optics["amplitude_contrast_ratio"]),
                   "phase_shift": float(optics.get("phase_shift", 0.0))},
        "micrographs": [{"path": str(mic_path), "defocus_u": float(optics["defocus_u"]),
                         "defocus_v": float(optics["defocus_v"]),
                         "astigmatism_angle": float(optics["astigmatism_angle"])}],
        "defocus_offsets": offsets, "invert_contrast": not a.no_invert, "stages": stages,
        "weights_cells": [0, 0, max(k, ks, kf), max(m, ms, mf)],
        "execution": {"show_progress": True}, "output_dir": str(work / "search"),
        "cache_micrographs": False, "overwrite": True, "keep_stage_results": False,
        "device": str(device)})
    t = time.perf_counter()
    state = _Run(config)
    micrograph = state.micrograph(config.micrographs[0])
    searcher = state.searcher((shape[0], shape[1]))
    session = searcher.session(micrograph)
    times["prepare"] = time.perf_counter() - t

    # the constraint's eligible band, moved into the search's (template top-left) frame
    with h5py.File(cons["spatial_constraint_path"], "r") as f:
        eligible = np.asarray(f["maps/eligible"][()], bool)
    band_mask = eligible[half:half + grid[0], half:half + grid[1]]
    initial = PixelSelection.from_mask(session.geometry, torch.from_numpy(band_mask)[None].to(device))
    session.ledger = PixelLedger(session.geometry, session.ledger.device, initial=initial)
    session.pixels = initial
    print(f"searching {len(initial):,} band pixels ({band_mask.mean():.1%} of the grid)")

    stages_run = iter(session)
    run_info = {}
    if a.screen:
        t = time.perf_counter()
        screen = next(stages_run)
        times["screen"] = time.perf_counter() - t
        run_info["screen_pixels"] = screen.n_pixels
        run_info["screen_survivors"] = len(session.pixels)
        print(f"screen: {screen.n_pixels:,} px in {times['screen']:.1f} s -> "
              f"{len(session.pixels):,} survivors ({len(session.pixels) / max(screen.n_pixels, 1):.2%})")
        del screen
    t = time.perf_counter()
    found = next(stages_run)
    times["find"] = time.perf_counter() - t
    run_info["find_pixels"] = found.n_pixels
    run_info["find_survivors"] = len(session.pixels)
    print(f"find: {found.n_pixels:,} px in {times['find']:.1f} s -> {len(session.pixels):,} with max z >= "
          f"{a.z_cut - a.find_margin} ({len(session.pixels) / max(found.n_pixels, 1):.3%})", flush=True)
    del found
    t = time.perf_counter()
    with emit_entries(a.z_cut, a.margin, normalize="all" if a.normalize == "all" else "stage") as sink:
        result_b = next(stages_run)
    times["table"] = time.perf_counter() - t
    print(f"table pass: {result_b.n_pixels:,} px in {times['table']:.1f} s, "
          f"{sink.n_entries:,} entries with z >= {a.z_cut}")

    # decode the entries: virtual id -> (plane, selected template) -> global id
    space = searcher.space
    template_ids, defocus_ids = searcher.stages[-1].hypothesis_ids(space, searcher.ipix)
    n_sel = space.n_filters * space.n_orientations if template_ids is None else len(template_ids)

    def to_global(virtual: torch.Tensor) -> torch.Tensor:
        v = virtual.long().cpu()
        d_local, j = v // n_sel, v % n_sel
        tpl = j if template_ids is None else template_ids.cpu()[j]
        d = d_local if defocus_ids is None else defocus_ids.cpu()[d_local]
        return d * (space.n_filters * space.n_orientations) + tpl

    maps = result_b.maps
    valid = (maps.best_virtual >= 0).cpu()
    if not torch.equal(to_global(maps.best_virtual.cpu()[valid]), result_b.best_index.cpu()[valid]):
        raise RuntimeError("virtual -> global decoding disagrees with the stage's own best_index")
    e = sink.gather()
    column = e["column"].long()
    pix = result_b.pixel_index.cpu()[column]
    y, x = (pix // grid[1]).numpy(), (pix % grid[1]).numpy()
    d_idx, _, o_idx = (v.numpy() for v in space.decode(to_global(e["virtual"])))
    # consistency: a pixel's largest entry is its map maximum
    if column.numel():
        top = torch.full((result_b.n_pixels,), -np.inf).scatter_reduce(0, column, e["value"].float(), "amax")
        mip = maps.mip.cpu()
        z_max = maps.zscore.cpu()
        check = z_max >= a.z_cut
        err = (top[check] - mip[check]).abs().max().item() if check.any() else 0.0
        print(f"check: {int(check.sum()):,} pixels reach the cut; max |top entry - map max| = {err:.2e}")
        run_info["max_entry_vs_map_error"] = err

    result = session.snapshot()
    out = a.out.parent / f"{a.out.name}_svd.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(out)
    leopard.write_match_template_maps(out, result)
    scale = float(maps.variance.clamp_min(0).sqrt().median())
    with h5py.File(out, "a") as f:
        g = f.create_group("entries")
        for name, arr in (("y", y.astype(np.int32)), ("x", x.astype(np.int32)),
                          ("defocus_index", d_idx.astype(np.int16)),
                          ("orientation_index", o_idx.astype(np.int32)),
                          ("psi_index", e["psi"].numpy().astype(np.int16)),
                          ("value", e["value"].numpy().astype(np.float32)),
                          ("z", e["z"].numpy().astype(np.float32)),
                          ("mean", e["mean"].numpy().astype(np.float32)),
                          ("std", e["std"].numpy().astype(np.float32))):
            g.create_dataset(name, data=arr, compression="gzip", compression_opts=1, shuffle=True)
        o = f.create_group("orientations")
        o.create_dataset("phi_deg", data=np.asarray(phi, np.float32))
        o.create_dataset("theta_deg", data=np.asarray(theta, np.float32))
        meta = {"config": str(a.config), "constraint": str(a.constraint), "micrograph": str(mic_path),
                "template": cfg["template_volume_path"], "pixel_size": pixel,
                "micrograph_shape": list(shape), "template_box": box, "half": half,
                "grid": list(grid), "n_psi": a.n_psi, "defocus_offsets": offsets,
                "z_cut": a.z_cut, "margin": a.margin, "rank": [k, m], "find_rank": [kf, mf], "normalize": a.normalize,
                "n_orientations_decomposed": int(len(phi)), "n_orientations_band": int(len(band_ids)),
                "screen_rank": [ks, ms] if a.screen else None, "find_margin": a.find_margin,
                "nside": a.nside,
                "theta_band": [theta_c, cone + a.theta_margin], "scale_median_std": scale,
                "template_info": tinfo, "truncation_error": report["tables"],
                "polar_center": a.polar_center, "b_factor": a.b_factor,
                "suggestions": report.get("suggestions"), "times_s": times, **run_info}
        f.attrs["svd_filament_meta"] = json.dumps(meta)
    print("times: " + ", ".join(f"{k_} {v:.1f} s" for k_, v in times.items()))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
