"""Panther-EM SVD entries -> Leopard-EM correlation table (format 2) + match-template maps.

Second half of SVD Stage 2 (``svd_match_filament.py`` writes ``<out>_svd.h5``). Runs in
Leopard-EM's environment, and gates the entries with Leopard-EM's own constraint code,
so a detection is kept exactly when ``match_template`` would have let it into the table:
its pixel is eligible, theta and phi lie in the constraint's box, and psi lies within the
cone of the ``psi_center`` (or its opposite pole, as the region's polarity allows) of ANY
region covering the pixel.

    python ../programs/constrained_search/svd_entries_to_table.py \\
        results_svd/gdp_curved_13pf_svd.h5 --tag gdp_curved_13pf

writes ``results_svd/output_correlation_table_<tag>.h5`` and the maps
``results_svd/output_{mip,scaled_mip,...}_<tag>.mrc``.

Units: Panther-EM correlations are divided by the median per-pixel correlation sd of the
search, so values sit on Leopard-EM's scale (noise sd ~1); z = (value - mean) / sd is
exact either way. The table's ``correlation_threshold`` is the z cut of the search
(Leopard-EM's own tables cut on the raw correlation).

Frame: by default the table and maps are written in the frame of the ORIGINAL template
box (the config's ``template_volume_path``), exactly as Leopard-EM's own search writes
them: the SVD search used a box cropped to the template's mass, so its top-left grid is
shifted by (original - cropped) // 2 and slightly larger; it is shifted back and cropped
to the original grid (positions within half the original box of the edge -- which
Leopard-EM never searches -- are dropped). Downstream code that crops the statistic maps
at a particle's top-left position (refine_template) then lines up. ``--frame search``
keeps the search's own (cropped-box) frame.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import h5py
import mrcfile
import numpy as np
import torch

from leopard_em.backend.utils import _circular_abs_diff_deg, _in_angle_range_torch
from leopard_em.pydantic_models.config import FilamentConstraint


def orientation_ok(constraint, maps, y, x, phi, theta, psi, atol=1e-3) -> np.ndarray:
    """Leopard-EM's per-(pixel, orientation) gate, per entry (stats-map coordinates)."""
    theta_c, cone = float(maps.psi_theta_center_deg), float(maps.psi_cone_half_angle_deg)
    phi_t, theta_t, psi_t = (torch.from_numpy(np.asarray(v, np.float64)) for v in (phi, theta, psi))
    theta_ok = _in_angle_range_torch(theta_t, max(0.0, theta_c - cone), min(180.0, theta_c + cone),
                                     period=None, atol=atol)
    phi_ok = _in_angle_range_torch(phi_t, float(maps.psi_phi_min), float(maps.psi_phi_max),
                                   period=360.0, atol=atol)
    ok = (theta_ok & phi_ok).numpy()
    if cone >= 180.0:
        return ok
    stacked = maps.stacked_psi_layers() if maps.n_psi_layers > 1 else None
    if stacked is not None:
        centres, poles = stacked
    else:
        centres = np.asarray(maps.psi_center, np.float32)[None]
        poles = (np.full(centres.shape, 3, np.uint8) if maps.pole_mask is None
                 else np.asarray(maps.pole_mask, np.uint8)[None])
    half = cone + atol
    psi_ok = np.zeros(len(y), bool)
    for layer in range(centres.shape[0]):
        c = torch.from_numpy(centres[layer][y, x].astype(np.float64))
        p = poles[layer][y, x]
        near = (_circular_abs_diff_deg(psi_t, c) <= half).numpy()
        far = (_circular_abs_diff_deg(psi_t, (c + 180.0) % 360.0) <= half).numpy()
        psi_ok |= (((p & 1) != 0) & near) | (((p & 2) != 0) & far)
    return ok & psi_ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("svd_h5", type=pathlib.Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", type=pathlib.Path, default=None, help="default: beside svd_h5")
    ap.add_argument("--frame", choices=["original", "search"], default="original")
    a = ap.parse_args()
    outdir = a.outdir or a.svd_h5.parent

    with h5py.File(a.svd_h5, "r") as f:
        meta = json.loads(f.attrs["svd_filament_meta"])
        e = {k: f["entries"][k][()] for k in f["entries"]}
        phi_o, theta_o = f["orientations/phi_deg"][()], f["orientations/theta_deg"][()]
        tensors = {k: f["tensors"][k][()] for k in f["tensors"]}
    n_psi, box = int(meta["n_psi"]), int(meta["template_box"])
    shape = tuple(meta["micrograph_shape"])
    psi_deg = 360.0 * np.arange(n_psi) / n_psi
    phi, theta, psi = phi_o[e["orientation_index"]], theta_o[e["orientation_index"]], psi_deg[e["psi_index"]]

    constraint = FilamentConstraint.from_yaml(meta["constraint"])
    maps = constraint.stats_maps_for_template(shape, box)
    y, x = e["y"].astype(np.int64), e["x"].astype(np.int64)
    pixel_ok = np.asarray(maps.eligible, bool)[y, x]
    keep = pixel_ok & orientation_ok(constraint, maps, y, x, phi, theta, psi)
    print(f"{len(y):,} entries with z >= {meta['z_cut']}; {pixel_ok.mean():.1%} on eligible pixels; "
          f"{keep.sum():,} pass the constraint's orientation gate")
    out_box, shift = box, 0
    if a.frame == "original":
        with mrcfile.mmap(meta["template"], permissive=True) as handle:
            out_box = int(handle.data.shape[-1])
        shift = out_box // 2 - box // 2
        grid = (shape[0] - out_box + 1, shape[1] - out_box + 1)
        y, x = y - shift, x - shift
        inside = (y >= 0) & (x >= 0) & (y < grid[0]) & (x < grid[1])
        print(f"frame: original {out_box}^3 box (search used {box}^3): shift {shift} px; "
              f"{(keep & ~inside).sum():,} gated entries within the original box's edge band dropped")
        keep &= inside
        tensors = {k: (v[..., shift:shift + grid[0], shift:shift + grid[1]] if v.ndim >= 2 else v)
                   for k, v in tensors.items()}

    n_orient = len(phi_o) * n_psi
    orient = e["orientation_index"].astype(np.int64) * n_psi + e["psi_index"]
    search_index = e["defocus_index"].astype(np.int64) * n_orient + orient
    scale = float(meta["scale_median_std"])
    euler = np.stack([np.repeat(phi_o, n_psi), np.repeat(theta_o, n_psi), np.tile(psi_deg, len(phi_o))], 1)
    table = outdir / f"output_correlation_table_{a.tag}.h5"
    with h5py.File(table, "w") as f:
        m = f.create_group("metadata")
        m.attrs["correlation_threshold"] = float(meta["z_cut"])
        m.attrs["num_observations"] = int(keep.sum())
        m.attrs["format_version"] = 2
        m.attrs["pixel_size"] = float(meta["pixel_size"])
        m.attrs["template_shape"] = np.asarray([out_box] * 3, np.int64)
        m.attrs["micrograph_shape"] = np.asarray(shape, np.int64)
        m.attrs["source"] = "panther-em svd (svd_match_filament.py); threshold is a z cut"
        m.attrs["svd_search_box"] = int(box)
        m.attrs["svd_meta"] = json.dumps({k: meta[k] for k in ("rank", "screen_rank", "nside", "n_psi",
                                                                "theta_band", "z_cut", "times_s")})
        s = f.create_group("search_space")
        s.create_dataset("euler_angles", data=euler.astype(np.float32))
        s.create_dataset("defocus_offsets", data=np.asarray(meta["defocus_offsets"], np.float32))
        s.create_dataset("phi_theta_angles", data=np.stack([phi_o, theta_o], 1).astype(np.float32))
        s.create_dataset("psi_angles", data=psi_deg.astype(np.float32))
        d = f.create_group("detections")
        d.create_dataset("search_index", data=search_index[keep].astype(np.int32))
        d.create_dataset("x", data=x[keep].astype(np.int32))
        d.create_dataset("y", data=y[keep].astype(np.int32))
        d.create_dataset("correlation_value", data=(e["value"][keep] / scale).astype(np.float32))
        d.create_dataset("correlation_mean", data=(e["mean"][keep] / scale).astype(np.float32))
        d.create_dataset("correlation_variance", data=(e["std"][keep] / scale).astype(np.float32))
    print(f"wrote {table} ({int(keep.sum()):,} detections)")

    for name, arr in tensors.items():
        if arr.ndim == 3:
            arr = arr[0]
        path = outdir / f"output_{name}_{a.tag}.mrc"
        with mrcfile.new(str(path), overwrite=True) as handle:
            handle.set_data(np.ascontiguousarray(arr, dtype=np.float32))
            handle.voxel_size = float(meta["pixel_size"])
    print(f"wrote {len(tensors)} maps: output_<map>_{a.tag}.mrc")


if __name__ == "__main__":
    main()
