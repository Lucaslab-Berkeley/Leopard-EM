"""Score EVERY lattice site, found or not, and see what that buys Stage 6b.

The table keeps a site only when one of its correlations clears the table's cut; Stages
4-7 then work on those found sites. But once Stage 4 has the lattice (curve, rise, phase,
roll), every site's pose is PREDICTED, and an SVD search can score each one directly
(``programs/constrained_search/svd_score_sites.py``). This is the prototype of that,
next to the old route, which it does not touch:

``sites``    every (axial index[, protofilament]) in each filament's span at its neutral
             pose (Stage 6b's ``neutral_stack`` with ``all_sites``; found sites keep their
             labels), plus DECOYS: each site's pose moved to a random background pixel
             (>= half a template box from any filament band), the null for "the best z in
             a window around a predicted pose" when nothing is there.
             -> ``lattice_fill_sites_<tag>.csv``
(score)      ``svd_score_sites.py`` in the Panther-EM environment, both rival templates.
             -> ``lattice_fill_scores_<tag>.csv``
``analyse``  the decoy null (how far above noise each site is), and Stage 6b's segment
             competition on: the OLD route (Leopard-EM refine at found sites, neutral
             pose), the SVD scores at found sites only (same sites: checks the scorer), at
             every site, at every site with the window maximum, and at every site that
             clears the decoy null. With ``--truth`` each is scored against the truth.

Usage (patch, strong known-domain synthetic):
    python run_scripts/lattice_fill.py sites \\
        --table results_bundle/output_correlation_table_bundle_strong_patch.h5 --patch \\
        --constraint configs/filament_constraint_bundle_domains_strong.h5 \\
        --truth Frames/synthetic_bundle_lattice_domains_strong.truth.json
    python run_scripts/lattice_fill.py analyse \\
        --table results_bundle/output_correlation_table_bundle_strong_patch.h5 --patch \\
        --truth Frames/synthetic_bundle_lattice_domains_strong.truth.json
"""

import argparse
import json
import pathlib
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from leopard_em.analysis.filament_lattice import TemplateLatticeGeometry

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from frame_geometry import frame_geometry  # noqa: E402
from stage6b_local_lattice import (  # noqa: E402
    PATCH_PAIRS,
    LATTICE_PAIRS,
    PURE,
    analyse,
    neutral_stack,
    reference_stems,
    rise_models,
    truth_labels,
)

DECOY_LABEL = 2_000_000
SIZES = (8, 12, 16, 20, 24, 32)


def load_found(results, tag, patch):
    """Stage 6b's inputs: the found sites' stack and their refined poses."""
    if patch:
        stack = pd.read_csv(results / f"stage7_stack_{tag}_in.csv", index_col=0)
        refined = pd.read_csv(results / f"stage7_refined_{tag}_in.csv", index_col=0)
        labels = pd.read_csv(results / f"stage4_sites_{tag}.csv")
        if len(labels) != len(stack) or not np.array_equal(
                labels["x"].to_numpy(), stack["pos_x"].to_numpy()):
            raise SystemExit("Stage 7's stack does not match Stage 4's sites row for row")
        for column in ("filament", "axial_index", "protofilament"):
            stack[column] = labels[column].to_numpy()
    else:
        stack = pd.read_csv(results / f"stack_sites_{tag}.csv", index_col=0)
        refined = pd.read_csv(results / f"refined_sites_{tag}.csv", index_col=0)
    return stack, refined


def setup(args):
    table = args.table if args.table.is_absolute() else MT_ROOT / args.table
    tag = table.stem.removeprefix("output_correlation_table_")
    ring_stem, patch_stem = reference_stems(args.rivals)
    stem = patch_stem if args.patch else ring_stem
    geometry = TemplateLatticeGeometry.from_pdb(str(MT_ROOT / "models" / f"{stem}.pdb"))
    keys = PATCH_PAIRS[args.rivals] if args.patch else LATTICE_PAIRS[args.rivals][:2]
    return table, tag, table.parent, geometry, frame_geometry(table), keys


MIN_BACKGROUND = 0.01   # fraction of the frame; every run before 2026-11-18 had >= 2%


def decoys(sites, eligible, half_width, shape, rng, min_distance):
    """Each site's pose at a random background pixel: >= ``min_distance`` px from any
    filament band, and a whole template box inside the micrograph.

    A crowded frame (a micrograph full of bundles: one had no pixel 242 px from every band,
    at most 238) has too little such background. Then, and only then -- less
    than ``MIN_BACKGROUND`` of the frame -- the distance comes down in 8 px steps until
    there is, but not below half of it: a decoy's patch (support within ~110 px of its
    centre) then still stays >= ~120 px off the nearest tube wall (band half-width 228 px,
    wall ~115 px from the axis). Frames with enough background are untouched (same
    pixels, same draws)."""
    distance = distance_transform_edt(~eligible)
    ok = distance >= min_distance
    inner = np.zeros_like(ok)
    inner[half_width:shape[0] - half_width, half_width:shape[1] - half_width] = True
    used = min_distance
    if (ok & inner).mean() < MIN_BACKGROUND:
        floor = min_distance / 2
        while (ok & inner).mean() < MIN_BACKGROUND and used - 8 >= floor:
            used -= 8
            ok = distance >= used
        print(f"  crowded frame: < {MIN_BACKGROUND:.0%} of it is >= {min_distance} px from "
              f"every band; decoys at >= {used} px instead ({(ok & inner).mean():.1%})")
    ys, xs = np.nonzero(ok & inner)
    if len(ys) == 0:
        raise SystemExit("no background left for decoys")
    pick = rng.integers(0, len(ys), len(sites))
    out = sites.copy()
    out["pos_y_img"], out["pos_x_img"] = ys[pick], xs[pick]
    out["pos_y"], out["pos_x"] = ys[pick] - half_width, xs[pick] - half_width
    out["decoy"] = True
    out["detected"] = False
    out.index = DECOY_LABEL + np.arange(len(out))
    return out, float((ok & inner).mean()), used


def path_band(paths_file, shape):
    """Pixels within half its width of any path in a ``*.paths.json`` (napari (y, x)
    points) -- e.g. an autotrace's extra tubes that no constraint band covers."""
    payload = json.loads(pathlib.Path(paths_file).read_text())
    band = np.zeros(shape, bool)
    for path in payload["paths"]:
        points = np.asarray(path["points"], float).reshape(-1, 2)
        line = np.zeros(shape, bool)
        for a, b in zip(points[:-1], points[1:]):
            n = int(np.ceil(np.hypot(*(b - a)))) + 1
            yx = np.rint(a + np.linspace(0, 1, n)[:, None] * (b - a)).astype(int)
            ok = ((yx >= 0) & (yx < np.asarray(shape))).all(axis=1)
            line[yx[ok, 0], yx[ok, 1]] = True
        band |= distance_transform_edt(~line) <= float(path.get("width_px", 300)) / 2
    print(f"  avoiding {len(payload['paths'])} path(s) from {paths_file}")
    return band


def cmd_sites(args):
    table, tag, results, geometry, frame_geo, keys = setup(args)
    stack, refined = load_found(results, tag, args.patch)
    print(frame_geo.describe())
    print(f"{len(stack)} found sites; every lattice "
          f"{'(index, protofilament)' if args.patch else 'index'} in each span:")
    sites = neutral_stack(stack, refined, geometry, frame_geo.pixel_size,
                          frame_geo.half_width, all_sites=True, resolve=args.patch)
    sites["decoy"] = False
    if args.truth is not None:
        sites["truth"] = truth_labels(sites, args.truth)
    with h5py.File(args.constraint, "r") as handle:
        eligible = np.asarray(handle["maps/eligible"][()], bool)
    for paths_file in args.avoid_paths:
        eligible |= path_band(paths_file, eligible.shape)
    fake, fraction, used = decoys(sites, eligible, frame_geo.half_width, eligible.shape,
                                  np.random.default_rng(args.seed), frame_geo.half_width)
    if "truth" in fake:
        fake["truth"] = "-"
    keep = ["filament", "axial_index", "protofilament", "detected", "decoy", "pos_x",
            "pos_y", "pos_x_img", "pos_y_img", "phi", "theta", "psi", "relative_defocus",
            "roll_abs"] + (["truth"] if "truth" in sites else [])
    out = pd.concat([sites[keep], fake[keep]])
    path = results / f"lattice_fill_sites_{tag}.csv"
    if args.out is not None:
        path = args.out if args.out.is_absolute() else MT_ROOT / args.out
    out.to_csv(path)
    for label, part in sites.groupby("filament"):
        span = part["axial_index"].max() - part["axial_index"].min() + 1
        print(f"  filament {label}: {len(part)} sites over {span} indices, "
              f"{int(part['detected'].sum())} found ({part['detected'].mean():.0%})")
    print(f"{len(fake)} decoys on background ({fraction:.0%} of the micrograph is >= "
          f"{used} px from every filament band)")
    print(f"wrote {path.relative_to(MT_ROOT)}")
    print("score it (Panther-EM environment), e.g.:\n"
          f"  CUDA_VISIBLE_DEVICES=0 python ../programs/constrained_search/svd_score_sites.py "
          f"--config configs/match_tm_{tag}.yaml --sites {path.relative_to(MT_ROOT)} "
          + " ".join(f"--template {k}=<{k} map>" for k in keys)
          + f" --out {(results / f'lattice_fill_scores_{tag}.csv').relative_to(MT_ROOT)}")


def competition_rates(frame, sizes, adaptive, geometry):
    """{size: (right, wrong, undecided, n)} of the competition call on windows that are
    >= PURE one lattice (Stage 6b's calibration, competition readout only)."""
    rates = {}
    for size in list(sizes) + ["adaptive"]:
        table = (analyse(frame, size, geometry, {}, step=max(size // 2, 1))
                 if size != "adaptive" else analyse(frame, adaptive, geometry, {}))
        if table.empty or "purity" not in table:
            continue
        pure = table[table["purity"] >= PURE]
        if pure.empty:
            continue
        calls = pure["comp_call"].fillna("?")
        rates[size] = (float((calls == pure["truth"]).mean()),
                       float(((calls != "?") & (calls != pure["truth"])).mean()),
                       float((calls == "?").mean()), len(pure))
    return rates


def cmd_analyse(args):
    table, tag, results, geometry, frame_geo, keys = setup(args)
    sites = pd.read_csv(results / f"lattice_fill_sites_{tag}.csv", index_col=0)
    scores = pd.read_csv(args.scores or results / f"lattice_fill_scores_{tag}{args.suffix}.csv",
                         index_col=0)
    scores = scores[[c for c in scores.columns if c.startswith(tuple(f"{k}_" for k in keys))]]
    data = sites.join(scores)
    c, e = keys
    real, fake = data[~data["decoy"]], data[data["decoy"]]
    found = real["detected"].astype(bool)

    # ---- the decoy null ------------------------------------------------------------
    print(f"{len(real)} lattice sites ({int(found.sum())} found, {int((~found).sum())} "
          f"not), {len(fake)} decoys; templates {c} / {e}")
    print(f"window: {int(np.median(data[f'{c}_n_window']))} hypotheses per site (median); "
          f"centre pose within {data[f'{c}_center_offset_deg'].median():.2f} deg of the grid "
          f"(median)\n")
    sym = {kind: (data[f"{c}_z_{kind}"] + data[f"{e}_z_{kind}"]) / 2
           for kind in ("center", "max")}
    print(f"{'z (mean of both templates)':<30}{'decoys':>22}{'found':>22}{'not found':>22}")
    print(f"{'':<30}" + f"{'median   p99  p99.9':>22}" * 3)
    thresholds = {}
    for kind in ("center", "max"):
        z = sym[kind]
        null = z[data["decoy"]].dropna()
        thresholds[kind] = float(np.percentile(null, 99))
        cells = []
        for part in (data["decoy"], ~data["decoy"] & data["detected"],
                     ~data["decoy"] & ~data["detected"]):
            values = z[part].dropna()
            cells.append(f"{values.median():>8.2f}{np.percentile(values, 99):>7.2f}"
                         f"{np.percentile(values, 99.9):>7.2f}")
        print(f"{kind:<30}" + "".join(cells))
    for kind in ("center", "max"):
        cut = thresholds[kind]
        above = sym[kind] >= cut
        print(f"  above the decoys' 99th percentile ({kind}, z {cut:.2f}): found "
              f"{(above & ~data['decoy'] & data['detected']).sum()}/{int(found.sum())}, "
              f"not found {(above & ~data['decoy'] & ~data['detected']).sum()}/"
              f"{int((~found).sum())}, decoys "
              f"{(above & data['decoy']).sum()}/{len(fake)}")
    for kind in ("center", "max"):
        dz = (fake[f"{c}_z_{kind}"] - fake[f"{e}_z_{kind}"]).dropna()
        print(f"  decoy dz ({kind}): {dz.mean():+.3f} +/- {dz.std(ddof=1) / np.sqrt(len(dz)):.3f}"
              f" (sd {dz.std(ddof=1):.3f}) -- a template bias on noise would show here")

    # ---- the competition, old and new ---------------------------------------------
    variants = {}
    old_summary = results / f"compete_summary_{tag}_neutral.csv"
    if old_summary.is_file():
        old = pd.read_csv(old_summary, index_col=0)
        old_stack = pd.read_csv(results / f"stack_sites_{tag}_neutral.csv", index_col=0)
        frame = pd.DataFrame({"filament": old_stack["filament"].to_numpy(),
                              "axial_index": old_stack["axial_index"].to_numpy(),
                              "dz": (old[c] - old[e]).to_numpy()}, index=old_stack.index)
        if args.truth is not None:
            frame["truth"] = truth_labels(old_stack, args.truth)
        variants["OLD: Leopard refine, found, neutral"] = frame
        both = frame.index.intersection(real.index[found])
        svd_dz = real.loc[both, f"{c}_z_center"] - real.loc[both, f"{e}_z_center"]
        r = np.corrcoef(frame.loc[both, "dz"], svd_dz)[0, 1]
        # compete summaries are positional (row i = row i of the stack), as in Stage 6b
        oz = pd.Series(old[c].to_numpy(), index=old_stack.index).loc[both]
        line = (f"\nSVD scorer vs Leopard refine at the same {len(both)} found sites: dz r = "
                f"{r:.2f}, mean dz {svd_dz.mean():+.3f} vs {frame.loc[both, 'dz'].mean():+.3f}")
        rz = np.corrcoef(oz, real.loc[both, f"{c}_z_center"])[0, 1]
        line += (f"; z({c}) r = {rz:.2f}, mean {real.loc[both, f'{c}_z_center'].mean():.2f}"
                 f" vs {oz.mean():.2f}")
        print(line)

    def frame_of(part, kind):
        out = pd.DataFrame({"filament": part["filament"].to_numpy(),
                            "axial_index": part["axial_index"].to_numpy(),
                            "dz": (part[f"{c}_z_{kind}"] - part[f"{e}_z_{kind}"]).to_numpy()},
                           index=part.index)
        if "truth" in part:
            out["truth"] = part["truth"].to_numpy()
        return out

    variants["SVD centre, found sites"] = frame_of(real[found], "center")
    variants["SVD centre, every site"] = frame_of(real, "center")
    variants["SVD window max, every site"] = frame_of(real, "max")
    gated = sym["center"][real.index] >= thresholds["center"]
    variants["SVD centre, every site > decoy p99"] = frame_of(real[gated], "center")

    # segment lengths: Stage 6b's adaptive ones (from the found sites' rise scatter)
    stack, refined = load_found(results, tag, args.patch)
    models = rise_models(stack, refined, geometry, frame_geo.pixel_size, args.patch)
    adaptive = {}
    for label in sorted(real["filament"].unique()):
        n_index = int(real.loc[real["filament"] == label, "axial_index"].nunique())
        model = models.get(label)
        adaptive[label] = model.resolving_size(n_index) if model is not None else None
    print(f"segments: Stage 6b's adaptive lengths {adaptive} (indices)")

    rows = []
    for name, frame in variants.items():
        n_index = frame.groupby("filament")["axial_index"].nunique().sum()
        print(f"\n== {name}: {len(frame)} sites, {frame['dz'].notna().sum()} scored, "
              f"{n_index} indices ({len(frame) / max(n_index, 1):.1f} sites / index)")
        segments = analyse(frame, adaptive, geometry, {})
        for label in sorted(segments["filament"].unique()):
            part = segments[segments["filament"] == label]
            truth = ("".join(t if p >= PURE else t.lower()
                             for t, p in zip(part["truth"], part["purity"]))
                     if "truth" in part else "")
            print(f"  filament {label}: call {''.join(part['comp_call']):<6} "
                  f"comp z {' '.join(f'{z:+.1f}' for z in part['z_comp']):<22} "
                  f"truth {truth}")
        if args.truth is not None:
            rates = competition_rates(frame, SIZES, adaptive, geometry)
            print("  right / wrong / undecided by window (indices): " + "; ".join(
                f"{s}: {r:.0%}/{w:.0%}/{u:.0%}" for s, (r, w, u, _) in rates.items()))
            for size, (right, wrong, undecided, n) in rates.items():
                rows.append({"variant": name, "size": size, "right": right, "wrong": wrong,
                             "undecided": undecided, "windows": n})
    shift_path = results / f"lattice_fill_scores_{tag}{args.suffix}_shift1.csv"
    register_rows = []
    if args.patch and shift_path.is_file():
        register_rows = register(data.join(pd.read_csv(shift_path, index_col=0)[
            ["shift1_z_center", "shift1_z_max"]]), results, tag, c, sym, thresholds)
    if rows:
        out = results / f"lattice_fill_rates_{tag}{args.suffix}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nwrote {out.relative_to(MT_ROOT)}")
    if args.plot:
        plot(data, sym, thresholds, rows, results / f"lattice_fill_{tag}{args.suffix}.png", c, e,
             register_rows)


def register(data, results, tag, key, sym, thresholds):
    """Stage 7's estimator (half the even-minus-odd paired difference, in - shift1) on
    the old found sites and on every site; decoys as a control that must be null."""
    from stage7_register import paired_register  # pylint: disable=import-outside-toplevel

    print("\n== REGISTER (Stage 7 estimator: in-register patch vs one monomer out)")
    print(f"{'':<38}{'fil':>4}{'sites':>7}{'register z':>17}{'t':>7}{'offset':>8}"
          f"{'control t':>10}")

    rows = []

    def show(name, frame, first):
        for label in sorted(frame["filament"].unique()):
            part = frame[frame["filament"] == label].dropna()
            if len(part) < 10:
                continue
            r = paired_register(part["axial_index"], part["in"], part["out"])
            control = paired_register(part["axial_index"] // 2, part["in"], part["out"])
            rows.append({"variant": name, "filament": int(label), "t": r["t"]})
            print(f"{name if first else '':<38}{label:>4}{r['n']:>7}{r['mean']:>9.3f} ± "
                  f"{r['sem']:.3f}{r['t']:>7.2f}{r['offset']:>+8.3f}{control['t']:>10.2f}")
            first = False

    old_path = results / f"stage7_register_{tag}.csv"
    if old_path.is_file():
        old = pd.read_csv(old_path).rename(columns={"in_register": "in",
                                                    "out_of_register": "out"})
        show("OLD: Leopard refine, found (search)", old, True)
    real = data[~data["decoy"]]
    found = real["detected"].astype(bool)
    gated = sym["center"][real.index] >= thresholds["center"]
    for name, part, kind in (("SVD centre, found sites", real[found], "center"),
                             ("SVD centre, every site", real, "center"),
                             ("SVD window max, every site", real, "max"),
                             ("SVD centre, every site > decoy p99", real[gated], "center"),
                             ("DECOYS centre (must be null)", data[data["decoy"]], "center"),
                             ("DECOYS window max (must be null)", data[data["decoy"]], "max")):
        frame = pd.DataFrame({"filament": part["filament"], "axial_index": part["axial_index"],
                              "in": part[f"{key}_z_{kind}"], "out": part[f"shift1_z_{kind}"]})
        show(name, frame, True)
    print("  register z: |half the even-minus-odd difference of (in - out)|; offset: mean"
          " (in - out);\n  control: the same on a two-dimer period (index // 2), must be"
          " null.")
    return rows


def plot(data, sym, thresholds, rows, path, c, e, register_rows=()):
    import matplotlib  # pylint: disable=import-outside-toplevel
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    groups = (("decoys", data["decoy"], "#9e9e9e"),
              ("found sites", ~data["decoy"] & data["detected"], "#1f4e79"),
              ("not found", ~data["decoy"] & ~data["detected"], "#d17a00"))
    for ax, kind in zip(axes[:2], ("center", "max")):
        bins = np.linspace(np.nanmin(sym[kind]), np.nanmax(sym[kind]), 70)
        for name, part, colour in groups:
            ax.hist(sym[kind][part].dropna(), bins=bins, histtype="step", lw=1.6,
                    color=colour, label=f"{name} ({int(part.sum())})", density=True)
        ax.axvline(thresholds[kind], color="black", ls="--", lw=0.8)
        ax.set_xlabel(f"z, mean of {c} and {e} ({'predicted pose' if kind == 'center' else 'window maximum'})")
        ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"{'centre' if kind == 'center' else 'window max'}: dashed = decoy 99th pct",
                     fontsize=9)
    if rows:
        rates = pd.DataFrame(rows)
        rates = rates[rates["size"] != "adaptive"]
        for name, part in rates.groupby("variant", sort=False):
            axes[2].plot(part["size"].astype(int), 100 * part["right"], "o-", ms=4, lw=1.4,
                         label=name)
        axes[2].set_xlabel("window (lattice indices)")
        axes[2].set_ylabel("competition right (%)")
        axes[2].set_title("known-domain synthetic: windows >= 75% one lattice", fontsize=9)
        axes[2].legend(frameon=False, fontsize=7)
    elif register_rows:
        reg = pd.DataFrame(register_rows)
        shown = [v for v in ("OLD: Leopard refine, found (search)", "SVD centre, every site",
                             "SVD window max, every site", "DECOYS window max (must be null)")
                 if v in set(reg["variant"])]
        labels = sorted(reg["filament"].unique())
        width = 0.8 / max(len(shown), 1)
        colours = ("#1f4e79", "#2a9d8f", "#d1495b", "#9e9e9e")
        for i, (name, colour) in enumerate(zip(shown, colours)):
            part = reg[reg["variant"] == name].set_index("filament").reindex(labels)
            axes[2].bar(np.arange(len(labels)) + (i - (len(shown) - 1) / 2) * width,
                        part["t"], width * 0.92, color=colour, label=name)
        axes[2].set_xticks(np.arange(len(labels)), [f"fil {k}" for k in labels])
        axes[2].set_ylabel("register t (Stage 7 estimator)")
        axes[2].set_title("register: in-register vs one monomer out", fontsize=9)
        axes[2].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path.relative_to(MT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("sites", "analyse"))
    parser.add_argument("--table", type=pathlib.Path, required=True)
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--rivals", default="compacted13")
    parser.add_argument("--truth", type=pathlib.Path, default=None)
    parser.add_argument("--constraint", type=pathlib.Path, default=None,
                        help="the search's filament constraint .h5 (sites: decoy placement)")
    parser.add_argument("--avoid-paths", type=pathlib.Path, nargs="*", default=[],
                        help="sites: *.paths.json whose tubes decoys must also avoid (e.g. "
                             "an autotrace's tubes no constraint band covers)")
    parser.add_argument("--scores", type=pathlib.Path, default=None)
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="sites: write here instead of lattice_fill_sites_<tag>.csv "
                             "next to the table (e.g. to compare with a finished run)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suffix", default="",
                        help="analyse: read lattice_fill_scores_<tag><suffix>.csv (and its "
                             "_shift1); outputs get the suffix too")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.command == "sites":
        if args.constraint is None:
            raise SystemExit("sites needs --constraint (decoys go off every filament band)")
        cmd_sites(args)
    else:
        cmd_analyse(args)


if __name__ == "__main__":
    main()
