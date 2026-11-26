"""Figures for ``spacing_profile.py``: a spacing map on the micrograph, and the profiles.

* ``spacing_map_<name>.png`` -- the micrograph (binned 4x, band-passed, dimmed) with each
  tube's axis coloured by its local spacing, on one diverging scale for every case:
  centred midway between the compacted and expanded templates' repeats (C 40.943,
  E 41.833 A), so blue is compacted, red expanded, and the colour bar marks both.
* ``spacing_profiles_<name>.png`` -- one panel per tube: local spacing against position
  along the tube, with its error band; dashed lines at C and E; a grey line for the
  spacing before the tilt correction where that differs; the truth through the same
  window where the run has one.

Several tables (a split run's children) go on one map, tubes numbered as in the parent.

Usage:
    python run_scripts/plot_spacing_profile.py --name gdp --tables \\
        results_svd_pipeline/output_correlation_table_pipe_gdp_autotrace_fil1-2-3-4-5-6-11-12-13_13pf_patch.h5 \\
        results_svd_pipeline/output_correlation_table_pipe_gdp_autotrace_fil7-8-9-10_14pf_patch.h5
"""

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
# pylint: disable=wrong-import-position
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.ndimage import gaussian_filter

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stage6b_local_lattice import C_REPEAT, E_REPEAT  # noqa: E402

BIN = 4
MIDPOINT = (C_REPEAT + E_REPEAT) / 2
HALF_RANGE = 1.2                      # A either side of the midpoint
Y_RANGE = (40.0, 42.7)
CMAP = LinearSegmentedColormap.from_list(
    "spacing", ["#123f78", "#2a78d6", "#9cc3f0", "#f0efec", "#f3a5a4", "#e34948",
                "#8f1f1f"])
INK, MUTED, GRID = "#16202B", "#5B6773", "#DDE1E5"


def load_image(path: pathlib.Path) -> np.ndarray:
    """Block-mean binned, band-passed (~3 to ~80 binned px), clipped, 0..1."""
    with mrcfile.open(path, permissive=True) as handle:
        img = np.asarray(handle.data, dtype=np.float32)
    h, w = img.shape[0] // BIN * BIN, img.shape[1] // BIN * BIN
    small = img[:h, :w].reshape(h // BIN, BIN, w // BIN, BIN).mean((1, 3))
    small = gaussian_filter(small, 3.0) - gaussian_filter(small, 80)
    lo, hi = np.percentile(small, [0.5, 99.5])
    return np.clip((small - lo) / (hi - lo), 0, 1)


def read(tables, window):
    """Profiles of all tables, one micrograph, and whether any has a truth column."""
    profiles, micrograph = [], None
    for table in tables:
        table = table if table.is_absolute() else MT_ROOT / table
        tag = table.stem.removeprefix("output_correlation_table_")
        profiles.append(pd.read_csv(table.parent / f"spacing_profile_{tag}_w{window}.csv"))
        refined = pd.read_csv(table.parent / f"stage7_refined_{tag}_every_in.csv",
                              usecols=["micrograph_path"], nrows=1)
        path = MT_ROOT / refined["micrograph_path"].iloc[0]
        if micrograph is not None and path != micrograph:
            raise SystemExit("the tables are not on one micrograph")
        micrograph = path
    return pd.concat(profiles, ignore_index=True), micrograph


def draw_map(ax, img, profile, column, cmap, norm):
    """Dimmed micrograph on ``ax``, each tube's axis coloured by ``column``, numbered.

    Rows with ``column`` NaN are left out of the colour (the black outline still runs).
    """
    h, w = img.shape
    ax.imshow(0.15 + 0.6 * img, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    width = 7.5
    radius = 22
    placed = []
    for tube, part in profile.groupby("tube"):
        part = part.sort_values("centre_index")
        xy = part[["x_img", "y_img"]].to_numpy() / BIN
        ax.plot(xy[:, 0], xy[:, 1], color="black", lw=width + 4, alpha=0.7,
                solid_capstyle="round")
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        values = (part[column].to_numpy()[:-1] + part[column].to_numpy()[1:]) / 2
        keep = np.isfinite(values)
        lines = LineCollection(segments[keep], cmap=cmap, norm=norm, linewidths=width,
                               capstyle="round")
        lines.set_array(values[keep])
        ax.add_collection(lines)
        # badge beside the tube, where it is farthest from badges already placed
        normal = np.array([-(xy[-1] - xy[0])[1], (xy[-1] - xy[0])[0]])
        normal = normal / (np.linalg.norm(normal) + 1e-9)
        best, best_d = None, -1.0
        for fraction in (0.5, 0.35, 0.65, 0.2, 0.8, 0.1, 0.9):
            for side in (1, -1):
                spot = xy[int(fraction * (len(xy) - 1))] + side * 34 * normal
                spot = np.clip(spot, radius + 2, [w - radius - 2, h - radius - 2])
                gap = min([np.hypot(*(spot - q)) for q in placed] + [1e9])
                if gap > best_d:
                    best, best_d = spot, gap
            if best_d > 3 * radius:
                break
        spot = best
        placed.append(spot)
        ax.add_patch(plt.Circle(spot, radius, facecolor="white", edgecolor=INK, lw=2,
                                zorder=5))
        ax.text(spot[0], spot[1] + 1, str(int(tube)), ha="center", va="center",
                fontsize=19, weight="bold", color=INK, zorder=6)


def spacing_map(profile, micrograph, out, title):
    """Dimmed micrograph, tube axes coloured by local spacing, numbered, colour bar."""
    img = load_image(micrograph)
    h, w = img.shape
    bar_px = 260
    fig = plt.figure(figsize=((w + bar_px) / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, w / (w + bar_px), 1])
    norm = Normalize(MIDPOINT - HALF_RANGE, MIDPOINT + HALF_RANGE)
    draw_map(ax, img, profile, "spacing_A", CMAP, norm)
    cax = fig.add_axes([(w + 70) / (w + bar_px), 0.12, 34 / (w + bar_px), 0.76])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    bar.set_ticks([C_REPEAT, E_REPEAT, MIDPOINT - HALF_RANGE, MIDPOINT + HALF_RANGE])
    bar.set_ticklabels([f"C {C_REPEAT:.2f}", f"E {E_REPEAT:.2f}",
                        f"{MIDPOINT - HALF_RANGE:.1f}", f"{MIDPOINT + HALF_RANGE:.1f}"])
    bar.ax.tick_params(labelsize=15, colors=INK)
    bar.outline.set_edgecolor(MUTED)
    cax.set_title("spacing\n(Å)", fontsize=15, color=INK, pad=12)
    fig.patch.set_facecolor("white")
    fig.text(0.005, 0.995, title, ha="left", va="top", fontsize=16, color="white",
             path_effects=[pe.withStroke(linewidth=3, foreground="black")])
    fig.savefig(out, dpi=100)
    plt.close(fig)


def profiles_figure(profile, out, title, window):
    """Small multiples: spacing vs position along the tube, one panel per tube."""
    tubes = sorted(profile["tube"].unique())
    cols = min(4, len(tubes))
    rows = int(np.ceil(len(tubes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(max(4.4 * cols, 9.0), 3.0 * rows + 1.0),
                             squeeze=False, sharey=True)
    handles = {}
    for ax in axes.ravel()[len(tubes):]:
        ax.axis("off")
    for ax, tube in zip(axes.ravel(), tubes):
        part = profile[profile["tube"] == tube].sort_values("centre_index")
        x = part["arc_nm"].to_numpy()
        y = part["spacing_A"].to_numpy()
        se = part["se_A"].to_numpy()
        for value, name in ((C_REPEAT, "C"), (E_REPEAT, "E")):
            ax.axhline(value, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.text(x.max(), value + 0.03, name, color=MUTED, fontsize=9, ha="right",
                    va="bottom")
        projected = part["projected"].to_numpy()
        if np.max(np.abs(projected - y)) > 0.1:
            handles["before tilt correction"] = ax.plot(
                x, projected, color="#AEB6BF", lw=1.2, zorder=2)[0]
        if "truth_A" in part:
            handles["truth, seen through the same window"] = ax.plot(
                x, part["truth_A"], color=INK, lw=1.4, ls=(0, (2, 2)), zorder=3)[0]
        handles["±1 error"] = ax.fill_between(x, y - se, y + se, color="#9cc3f0",
                                              alpha=0.55, lw=0, zorder=2)
        points = np.column_stack([x, y])
        lines = LineCollection(np.stack([points[:-1], points[1:]], axis=1), cmap=CMAP,
                               norm=Normalize(MIDPOINT - HALF_RANGE,
                                              MIDPOINT + HALF_RANGE),
                               linewidths=2.2, zorder=4)
        lines.set_path_effects([pe.Stroke(linewidth=3.6, foreground=INK, alpha=0.55),
                                pe.Normal()])   # the pale midpoint stays visible
        lines.set_array((y[:-1] + y[1:]) / 2)
        ax.add_collection(lines)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(*Y_RANGE)
        ax.set_title(f"tube {int(tube)}", fontsize=11, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=9, colors=MUTED)
        if ax in axes[:, 0]:
            ax.set_ylabel("local spacing (Å)", fontsize=10, color=INK)
        if ax in axes[-1, :] or tube == tubes[-1]:
            ax.set_xlabel("position along tube (nm)", fontsize=10, color=INK)
    top = 1 - 0.75 / fig.get_figheight()
    fig.suptitle(f"{title}: local spacing, {window}-index window (≈{window * 4.09:.0f} nm)",
                 fontsize=12, color=INK, x=0.01, y=0.995, ha="left", va="top")
    fig.legend(list(handles.values()), list(handles.keys()), loc="upper right",
               ncol=len(handles), fontsize=9, frameon=False,
               bbox_to_anchor=(0.995, 1 - 0.32 / fig.get_figheight()))
    fig.tight_layout()
    fig.subplots_adjust(top=top)   # tight_layout reserves suptitle room again
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    """Map and profiles for one case."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("--name", required=True, help="output name, e.g. gdp")
    parser.add_argument("--title", default=None)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None,
                        help="default: the first table's folder")
    args = parser.parse_args()
    profile, micrograph = read(args.tables, args.window)
    first = args.tables[0] if args.tables[0].is_absolute() else MT_ROOT / args.tables[0]
    out_dir = args.out_dir or first.parent
    title = args.title or args.name
    spacing_map(profile, micrograph, out_dir / f"spacing_map_{args.name}_w{args.window}.png",
                title)
    profiles_figure(profile, out_dir / f"spacing_profiles_{args.name}_w{args.window}.png",
                    title, args.window)
    print(f"wrote spacing_map_{args.name}_w{args.window}.png and "
          f"spacing_profiles_{args.name}_w{args.window}.png in {out_dir}")


if __name__ == "__main__":
    main()
