"""Plot the register test on the GDP frame, in the deck's style.

One panel per segment: site score against axial index, with alternate sites coloured,
so an alternation would show as a zig-zag. Plus a bar per segment of the period-2
amplitude divided by its own permutation chance level, against the 1.0 chance line and
the 2.19x the same method reads on the curved synthetic where the register is KNOWN.

Segments are plotted separately and never pooled: extract_lattice_sites phases the axial
index on each segment's own bootstrap, so index 0 is a different physical subunit in each
and pooling scrambles the parity rather than averaging it.

Usage:
    python run_scripts/plot_seam_curved.py
"""

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import h5py  # noqa: E402

from leopard_em.analysis.filament_lattice import (  # noqa: E402
    TemplateLatticeGeometry,
    extract_lattice_sites,
)
from leopard_em.pydantic_models.results.correlation_table import (  # noqa: E402
    detections_from_hdf5,
)

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyse_gdp_curved as A  # noqa: E402
from seam_from_patch_search import TEMPLATES, alternation, segments_for  # noqa: E402
from continuous_unwrap_curved import continuous_sites  # noqa: E402
from frame_geometry import legacy_frame_geometry  # noqa: E402

INK, MUTED, BLUE, AMBER, RED, LIGHT = (
    "#1a1a1a", "#6b6b6b", "#1f4e79", "#ef6c00", "#c62828", "#e8eef4")
SYNTHETIC_PATCH = 2.19   # same method, known truth
SYNTHETIC_RING = 0.69


def collect(key, results, constraint, min_z, rng):
    """Per-segment (label, index, score, ratio, control ratio) for one template."""
    tag, stem = TEMPLATES[key]
    table = MT_ROOT / results / f"output_correlation_table_{tag}.h5"
    if not table.is_file():
        return []
    geometry = TemplateLatticeGeometry.from_pdb(str(MT_ROOT / "models" / f"{stem}.pdb"))
    detections = detections_from_hdf5(str(table), min_z_score=min_z).reset_index(drop=True)
    frame = legacy_frame_geometry(table, stem)
    with h5py.File(constraint, "r") as handle:
        region_map = handle["maps/region_id"][:]
        drawn = {int(n): handle[f"regions/{n}/path"][:] for n in handle["regions"]}

    out = []
    for region, part, k, piece in segments_for(detections, geometry, drawn, region_map,
                                               frame):
        boot = A.pick_bootstrap(piece["z_score"].to_numpy())
        if boot is None:
            continue
        try:
            sites = extract_lattice_sites(
                piece, geometry, frame.pixel_size, bootstrap_score_threshold=boot)
            control = extract_lattice_sites(
                piece, geometry, frame.pixel_size, bootstrap_score_threshold=boot,
                initial_rise_angstrom=2.0 * geometry.rise_angstrom)
        except ValueError:
            continue
        index = np.asarray(sites.axial_index).astype(int)
        if len(index) < 8:
            continue
        score = np.asarray(sites.score)
        order = np.argsort(index)
        amplitude, null = alternation(index, score, rng)
        c_index = np.asarray(control.axial_index).astype(int)
        c_ratio = np.nan
        if len(c_index) >= 8:
            c_amp, c_null = alternation(c_index, np.asarray(control.score), rng)
            c_ratio = c_amp / c_null if c_null else np.nan
        out.append({
            "label": f"R{region} seg{part}/{k}", "index": index[order],
            "score": score[order], "ratio": amplitude / null if null else np.nan,
            "control": c_ratio, "rise": float(sites.rise_angstrom),
        })
    return out


def collect_refined(key, results, rng):
    """Blocks from the refined, continuously-indexed sites written by
    refine_patch_sites.py -- score and index both from this template."""
    import pandas as pd
    tag, _stem = TEMPLATES[key]
    refined = MT_ROOT / results / f"refined_continuous_{tag}.csv"
    stack = MT_ROOT / results / f"stack_continuous_{tag}.csv"
    if not refined.is_file() or not stack.is_file():
        return []
    after = pd.read_csv(refined, index_col=0)
    labels = pd.read_csv(stack, index_col=0)
    region = labels["region"].to_numpy()
    index_all = labels["axial_index"].to_numpy().astype(int)
    score_all = after["refined_scaled_mip"].to_numpy()

    out = []
    for value in sorted(set(region.tolist())):
        keep = region == value
        index, score = index_all[keep], score_all[keep]
        if len(index) < 8:
            continue
        order = np.argsort(index)
        amplitude, null = alternation(index, score, rng)
        c_amp, c_null = alternation(index // 2, score, rng)
        out.append({
            "label": f"region {value}", "index": index[order], "score": score[order],
            "ratio": amplitude / null if null else np.nan,
            "control": c_amp / c_null if c_null else np.nan,
        })
    return out


def collect_continuous(key, results, constraint, min_z, rng):
    """One block per REGION, indexed continuously along the fitted curve."""
    import h5py as _h5
    tag, stem = TEMPLATES[key]
    table = MT_ROOT / results / f"output_correlation_table_{tag}.h5"
    if not table.is_file():
        return []
    geometry = TemplateLatticeGeometry.from_pdb(str(MT_ROOT / "models" / f"{stem}.pdb"))
    detections = detections_from_hdf5(str(table), min_z_score=min_z).reset_index(drop=True)
    frame = legacy_frame_geometry(table, stem)
    with _h5.File(constraint, "r") as handle:
        region_map = handle["maps/region_id"][:]
        drawn = {int(n): handle[f"regions/{n}/path"][:] for n in handle["regions"]}
    y = np.clip(detections["y"].to_numpy() + frame.half_width, 0, region_map.shape[0] - 1)
    x = np.clip(detections["x"].to_numpy() + frame.half_width, 0, region_map.shape[1] - 1)
    owner = region_map[y, x]

    out = []
    for region in sorted(drawn):
        mask = owner == region
        if mask.sum() < 200:
            continue
        subset = detections[mask].reset_index(drop=True)
        boot = A.pick_bootstrap(subset["z_score"].to_numpy(), want=20)
        if boot is None:
            continue
        try:
            sites = continuous_sites(subset, geometry, boot, pixel_size=frame.pixel_size)
        except ValueError:
            continue
        index, score = sites["axial_index"], sites["score"]
        order = np.argsort(index)
        amplitude, null = alternation(index, score, rng)
        c_amp, c_null = alternation(index // 2, score, rng)
        out.append({
            "label": f"region {region}", "index": index[order], "score": score[order],
            "ratio": amplitude / null if null else np.nan,
            "control": c_amp / c_null if c_null else np.nan,
            "rise": sites["rise"],
            "occupancy": len(index) / max(sites["n_predicted"], 1),
            "deviation": float(np.sqrt((sites["deviation"] ** 2).mean())),
        })
    return out


def main() -> None:
    """Draw the register figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results_gdp_curved")
    parser.add_argument("--constraint", type=pathlib.Path,
                        default=MT_ROOT / "configs"
                        / "filament_constraint_gdp_curved.h5")
    parser.add_argument("--min-z", type=float, default=6.0)
    parser.add_argument("--out", default="seam_register_gdp_curved.png")
    parser.add_argument("--segmented", action="store_true",
                        help="use the old per-segment indexing instead of continuous")
    parser.add_argument("--unrefined", action="store_true",
                        help="use search-grid scores instead of the refined ones")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    if args.segmented:
        patch = collect("patch", args.results, args.constraint, args.min_z, rng)
        ring = collect("ring", args.results, args.constraint, args.min_z, rng)
    elif args.unrefined:
        patch = collect_continuous("patch", args.results, args.constraint, args.min_z, rng)
        ring = collect_continuous("ring", args.results, args.constraint, args.min_z, rng)
    else:
        patch = collect_refined("patch", args.results, rng)
        ring = collect_refined("ring", args.results, rng)
    if not patch:
        raise SystemExit("no patch segments -- has the patch search finished?")

    n = len(patch)
    columns = min(4, n)
    rows = int(np.ceil(n / columns))
    # Wide enough that the per-panel titles fit: with only two regions the continuous
    # mode has long titles and a narrow figure clips them.
    width = max(6.2 * columns, 13.0)
    fig = plt.figure(figsize=(width, 3.4 * rows + 3.6))
    how = ("per segment" if args.segmented else
           "continuous arc-length index, unrefined" if args.unrefined else
           "continuous arc-length index, refined")
    fig.suptitle(f"Monomer register on the real GDP micrograph — patch template, {how}",
                 fontsize=15, color=INK, x=0.5, y=0.985)
    fig.text(0.5, 0.945, "Sites indexed at the MONOMER spacing along the fitted curve, so "
             "alternate sites have the template one monomer out of register — a seam "
             "would zig-zag.", fontsize=11, color=MUTED, ha="center")

    for i, block in enumerate(patch):
        ax = fig.add_subplot(rows + 1, columns, i + 1)
        index, score = block["index"], block["score"]
        even, odd = index % 2 == 0, index % 2 == 1
        ax.plot(index, score, lw=1.0, color=MUTED, zorder=1)
        ax.scatter(index[even], score[even], s=42, color=BLUE, zorder=3,
                   label="one register")
        ax.scatter(index[odd], score[odd], s=42, color=AMBER, zorder=3,
                   label="one monomer out")
        gap = abs(score[even].mean() - score[odd].mean()) if even.any() and odd.any() \
            else float("nan")
        extra = (f"\nocc {block['occupancy']:.0%}   dev {block['deviation']:.1f} Å"
                 f"   rise {block['rise']:.3f} Å" if "occupancy" in block else "")
        ax.set_title(f"{block['label']}   n={len(index)}   "
                     f"{block['ratio']:.2f}× chance   "
                     f"|Δz| {gap:.2f} ({100 * gap / score.mean():.1f}%){extra}",
                     fontsize=11, color=INK, loc="left")
        ax.set_xlabel("site number (one monomer apart)", fontsize=9)
        ax.set_ylabel("z-score", fontsize=9)
        ax.grid(alpha=0.25, ls=":")
        ax.tick_params(labelsize=8.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # Summary bars across the bottom.
    ax = fig.add_subplot(rows + 1, 1, rows + 1)
    labels = [b["label"] for b in patch]
    x = np.arange(len(labels))
    ring_by_label = {b["label"]: b["ratio"] for b in ring}
    ax.bar(x - 0.21, [b["ratio"] for b in patch], 0.2, color=BLUE, label="patch")
    ax.bar(x, [b["control"] for b in patch], 0.2, color="#c8d6e5",
           edgecolor=BLUE, lw=0.7, label="patch at DIMER spacing — THE null")
    ax.bar(x + 0.21, [ring_by_label.get(l, np.nan) for l in labels], 0.2,
           color=MUTED, alpha=0.6,
           label="ring (~40% of the patch's signal, NOT a null)")
    ax.axhline(1.0, color=RED, lw=2.0)
    ax.text(-0.45, 1.04, "chance", fontsize=10, color=RED, va="bottom")
    ax.axhline(SYNTHETIC_PATCH, color="#2e7d32", lw=2.0, ls="--")
    ax.text(-0.45, SYNTHETIC_PATCH + 0.04,
            f"{SYNTHETIC_PATCH}× — same method on the synthetic, where the register IS there",
            fontsize=10, color="#2e7d32", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("alternation ÷ chance")
    ax.set_ylim(0, max(SYNTHETIC_PATCH + 0.45, 1.5))
    ax.legend(frameon=False, fontsize=10, ncol=3, loc="upper right")
    ax.grid(alpha=0.25, ls=":", axis="y")
    best = np.nanmax([b["ratio"] for b in patch])
    ax.set_title(f"Highest {best:.2f}× against the 2.19× the same method reads on known "
                 f"truth — no register detected",
                 fontsize=12, color=INK, loc="left")

    fig.tight_layout(rect=[0, 0, 1, 0.925], h_pad=2.4)
    out = MT_ROOT / args.out
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    print(f"patch median {np.nanmedian([b['ratio'] for b in patch]):.2f}x over {n} "
          f"segments; ring median "
          f"{np.nanmedian([b['ratio'] for b in ring]):.2f}x over {len(ring)}")


if __name__ == "__main__":
    main()
