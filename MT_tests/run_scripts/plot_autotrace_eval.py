"""Evaluation PNGs: cylinder tracer vs SVD tracer, per micrograph.

usage: python run_scripts/plot_autotrace_eval.py [frame ...]   (default: every frame below)
Run from MT_tests/ in the leopard-em env. Writes results_autotrace_svd/eval_png/<frame>.png.

Three panels: the current (cylinder) tracks and the SVD tracks on the same low-passed
micrograph, and the SVD z map. Blue = track on an annotated/true tube, orange = track
on none (on real frames that only means "not annotated" -- check it by eye), white
dashed = the annotation. Track numbers are the paths.json order, which is what
``curate_paths.py --keep`` takes.
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mrcfile  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter, maximum_filter  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "programs/constrained_search"))
from auto_trace_filaments import fourier_crop  # noqa: E402
from napari_choose_filament_path import load_paths_json  # noqa: E402
from score_autotrace import score, truth_paths  # noqa: E402

MATCHED, EXTRA, REF = "#2a78d6", "#eb6834", "#ffffff"
SVD = pathlib.Path("results_autotrace_svd")
OLD = pathlib.Path("results_autotrace")
FRAMES = {
    "gmpcpp": dict(mic="Frames/2025-05-23_14.39.26_25May23_GMPCPP1_26-94_0005_X-1Y+1-0_sum_DW.mrc",
                   pixel=0.9194, ref=SVD / "gmpcpp_2dtm_axis.paths.json",
                   ref_label="axis from the full-resolution 2DTM search", truth=False),
    "gdp_curved": dict(mic="Frames/GDP_curved_mgraph.mrc", pixel=0.9194,
                       ref="configs/gdp_curved.paths.json",
                       ref_label="hand-drawn (2 tubes; the field holds many more)", truth=False),
    "in_situ_823": dict(mic="results_in_situ/set1_823_Aug09_10.05.20_DW.mrc", pixel=1.056,
                        ref="configs/filament_constraint_in_situ_823.paths.json",
                        ref_label="hand-drawn (4 tubes)", truth=False),
    "bundle_strong": dict(mic="Frames/synthetic_bundle_lattice_domains_strong.mrc", pixel=0.9194,
                          ref="Frames/synthetic_bundle_lattice_domains_strong.truth.json",
                          ref_label="simulated truth (every tube)", truth=True),
    "bundle_matched": dict(mic="Frames/synthetic_bundle_lattice_domains_matched.mrc", pixel=0.9194,
                           ref="Frames/synthetic_bundle_lattice_domains_matched.truth.json",
                           ref_label="simulated truth (every tube, 823-like strength)", truth=True),
}


def display_image(mic: np.ndarray, fd: int, pixel: float) -> np.ndarray:
    img = fourier_crop(mic, fd)
    px = pixel * fd
    img = gaussian_filter(img, 20.0 / px) - gaussian_filter(img, 800.0 / px)  # 20 A low-pass
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo), 0, 1)


def draw_paths(ax, paths, fd, status=None, numbers=True, ref=False, thin=False):
    for k, p in enumerate(paths):
        pts = np.asarray(p["points"], float) / fd
        if ref:
            ax.plot(pts[:, 1], pts[:, 0], ls=(0, (4, 3)), color=REF, lw=1.2, alpha=0.9, zorder=3)
            continue
        color = MATCHED if status[k] else EXTRA
        if thin:  # on the z map: keep the ridge underneath visible
            ax.plot(pts[:, 1], pts[:, 0], color=color, lw=0.8, alpha=0.75, ls=(0, (6, 3)), zorder=5)
        else:
            ax.plot(pts[:, 1], pts[:, 0], color="white", lw=3.6, alpha=0.6, zorder=4)  # surface ring
            ax.plot(pts[:, 1], pts[:, 0], color=color, lw=2.0, zorder=5)
        if numbers:
            mid = pts[len(pts) // 2]
            ax.text(mid[1], mid[0], str(k + 1), color="white", fontsize=8, fontweight="bold",
                    ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="white", lw=0.6))


def track_status(auto, refs, pixel, shape) -> tuple[list[bool], dict]:
    s = score(auto, refs, pixel, shape, margin_px=37.0 * 10 / pixel, match_a=150.0)
    on = {t - 1 for r in s["per_ref"] for t in r["tracks"]}
    return [k in on for k in range(len(auto))], s


def panel_title(name, s, truth):
    word = "false" if truth else "extra"
    return (f"{name}\n{s['coverage']:.0%} of annotated length covered · {s['n_tracks']} tracks · "
            f"{s['n_extra']} {word} ({s['extra_length_nm']:.0f} nm)")


def plot_frame(frame: str, out_dir: pathlib.Path) -> pathlib.Path:
    cfg = FRAMES[frame]
    with mrcfile.open(cfg["mic"], permissive=True) as handle:
        mic = np.asarray(handle.data, dtype=np.float32)
    shape = mic.shape
    pixel = cfg["pixel"]
    refs = truth_paths(pathlib.Path(cfg["ref"])) if cfg["truth"] else load_paths_json(str(cfg["ref"]))[1]
    fd = max(1, int(round(5.0 / pixel)))  # display at ~5 A/px
    img = display_image(mic, fd, pixel)
    runs = []
    for label, path in (("Current tracer (64 nm cylinder, Leopard-EM)", OLD / f"{frame}.paths.json"),
                        ("SVD tracer (ring template, Panther-EM)", SVD / f"{frame}.paths.json")):
        auto = load_paths_json(str(path))[1] if path.exists() else []
        status, s = track_status(auto, refs, pixel, shape)
        runs.append((label, auto, status, s))
    maps = np.load(SVD / f"{frame}_svd_maps.npz")
    z, half, f = maps["z"], int(maps["half"]), int(maps["factor"])
    meta = json.loads(str(maps["meta"]))

    portrait = shape[0] > shape[1]
    w_panel = 7.0 if not portrait else 5.2
    h_panel = w_panel * shape[0] / shape[1]
    fig, axes = plt.subplots(1, 3, figsize=(3 * w_panel, h_panel + 1.3), dpi=150,
                             gridspec_kw=dict(wspace=0.04))
    for ax, (label, auto, status, s) in zip(axes[:2], runs):
        ax.imshow(img, cmap="gray", interpolation="lanczos")
        draw_paths(ax, refs, fd, ref=True)
        draw_paths(ax, auto, fd, status)
        ax.set_title(panel_title(label, s, cfg["truth"]), fontsize=9, loc="left", color="#0b0b0b")
    ax = axes[2]
    extent = [half * f / fd, (half + z.shape[1]) * f / fd, (half + z.shape[0]) * f / fd, half * f / fd]
    # max-pool to the display grid so thin ridges survive the downsampling
    pool = max(1, int(round(fd / f)))
    zd = maximum_filter(z, size=2 * pool + 1)[::pool, ::pool] if pool > 1 else z
    med = float(np.median(z)); sd = 1.4826 * float(np.median(np.abs(z - med)))
    lo, hi = med + 2 * sd, max(med + 9 * sd, float(np.percentile(z, 99.99)))
    ax.imshow(np.zeros(img.shape), cmap="gray", vmin=0, vmax=1)
    ax.imshow(zd, cmap="magma", extent=extent, vmin=lo, vmax=hi, interpolation="antialiased")
    label, auto, status, s = runs[1]
    draw_paths(ax, auto, fd, status, numbers=True, thin=True)
    ax.set_title(f"SVD z map (ring 2DTM, {meta['binned_pixel']:.2f} A/px, rank {meta['rank'][0]}x"
                 f"{meta['rank'][1]}); search {meta['times_s']['search']:.0f} s\n"
                 f"colour from median + 2 sd ({lo:.1f}) to {hi:.1f}; max {z.max():.1f}", fontsize=9, loc="left")
    for ax in axes:
        ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    # scale bar on the first panel: 100 nm
    bar = 1000.0 / (pixel * fd)
    axes[0].plot([img.shape[1] * 0.95 - bar, img.shape[1] * 0.95], [img.shape[0] * 0.96] * 2,
                 color="white", lw=3)
    axes[0].text(img.shape[1] * 0.95 - bar / 2, img.shape[0] * 0.94, "100 nm", color="white",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
    handles = [plt.Line2D([], [], color=MATCHED, lw=2, label="track on an annotated tube"),
               plt.Line2D([], [], color=EXTRA, lw=2,
                          label="track on no annotated tube" + (" (false)" if cfg["truth"] else
                                                                " (check by eye)")),
               plt.Line2D([], [], color="#888888", ls=(0, (4, 3)), lw=1.2, label=f"annotation: {cfg['ref_label']}")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"{frame}: {pathlib.Path(cfg['mic']).name}", fontsize=11, x=0.01, ha="left")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.88 if not portrait else 0.9, bottom=0.06)
    out = out_dir / f"{frame}.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    out_dir = SVD / "eval_png"
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in sys.argv[1:] or FRAMES:
        print("wrote", plot_frame(frame, out_dir), flush=True)


if __name__ == "__main__":
    main()
