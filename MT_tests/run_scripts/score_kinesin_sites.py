"""Kinesin occupancy on real tubes: the kinesin templates scored at the neutral poses.

Every site of a Stage 6b neutral stack (lattice prediction + neighbours' offsets and
roll; nothing searched, so no finder bias) is scored, exactly as the neutral competition
scores it (``compete_sites_curved.write_config``, ``neutral_pose``: position pinned, no
orientation or defocus refinement), with the fixed-neighbourhood (``_nb61.18``) set:

    bare             14-PF ring
    decorated        + a motor on every beta                  (register A)
    decorated_shift  + the motors one monomer along: on alpha (register B)
    decorated_avg    + half-weight motors on every monomer    (no register, no seam)
    motors, motors_shift, motors_avg     the same motors without the tubulin

Each template is normalised by its OWN background: poses moved to empty ice (>
``--clear-nm`` from every tube), ``z = (cc - mean_bg) / sd_bg``. The background is one
fixed set -- poses from the plain neutral stack, ``--n-background`` positions -- shared by
every stack scored, so two stacks' z differ only by their sites (a fresh 400-position
background per run moved the SD estimate by 5%, as much as the effects compared). Leopard's z cannot be
used here: it subtracts the bare-ring search's per-pixel mean, which at a tube pixel is
far from zero and would offset a motors-only template's z by something that has nothing
to do with motors. This is the normalisation the synthetic calibration uses
(``simulate_kinesin_occupancy.py``: noise-only SD per template).

Occupancy is read from the ratio of a template's mean z to the bare ring's: noise scale,
contrast and model mismatch common to both cancel, and the synthetic gives the ratio as a
function of occupancy (its noise-free ``signal_*`` columns). ``motors_avg`` is the
primary readout: it needs neither the register nor the seam, which are unknown here.
``motors`` against ``motors_shift`` then says which register the decoration sits in,
attenuated by however many protofilaments the unknown seam puts in the other. Neutral
sites sit one MONOMER apart, so moving one site along swaps the template's register:
the difference is signed by the parity of the site's lattice index (and its overall sign
is arbitrary per tube, since so is the index origin).

Sites one monomer apart share most of their pixels (the template spans ~7 monomers), so
their scores are far from independent: errors are a bootstrap over blocks of
``--block-monomers`` consecutive lattice indices of a tube, not over sites.

Usage:
    python run_scripts/score_kinesin_sites.py --tag kinesin_2_0031_14pf
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "programs"
                       / "constrained_search"))
from compete_sites_curved import write_config  # noqa: E402
from napari_choose_filament_path import load_paths_json, spline_through_points  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from leopard_em.pydantic_models.managers import RefineTemplateManager  # noqa: E402

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
TAG = "_1.13_bscale0.5_nb61.18"
LATTICE = "6dpu_14pf_2rings_flatB"
TEMPLATES = {
    "bare": LATTICE,
    "decorated": f"{LATTICE}_kin3j8x",
    "decorated_shift": f"{LATTICE}_kin3j8x_shift",
    "decorated_avg": f"{LATTICE}_kin3j8x_avg",
    "motors": f"{LATTICE}_kin3j8x_motorsonly",
    "motors_shift": f"{LATTICE}_kin3j8x_motorsonly_shift",
    "motors_avg": f"{LATTICE}_kin3j8x_motorsavg",
}
HALF_BOX = 244


def background_stack(stack: pd.DataFrame, paths_jsons: list[pathlib.Path], shape,
                     pixel: float, clear_nm: float, n_sites: int, rng,
                     min_row: int = 0) -> pd.DataFrame:
    """The stack's poses moved to empty ice: > ``clear_nm`` from every path given.

    In a crowded field use every traced filament (``--background-paths``), not just the
    scored tubes, and ``min_row`` to keep off carbon above a hole edge. Fewer positions
    than asked are used if that is all the frame has (at least 200).
    """
    curves = []
    for paths_json in paths_jsons:
        _, paths = load_paths_json(str(paths_json))
        curves += [spline_through_points(p["points"], smooth_px=5.0)[0] for p in paths]
    tree = cKDTree(np.vstack(curves))
    height, width = shape
    margin = HALF_BOX + 8
    ys, xs = np.mgrid[max(margin, min_row):height - margin:40, margin:width - margin:40]
    candidates = np.column_stack([ys.ravel(), xs.ravel()]).astype(float)
    clear = tree.query(candidates)[0] > clear_nm * 10 / pixel
    candidates = candidates[clear]
    if len(candidates) < 200:
        raise SystemExit(f"only {len(candidates)} background positions clear of the tubes")
    if len(candidates) < n_sites:
        print(f"  only {len(candidates)} background positions clear of every path: using all")
        n_sites = len(candidates)
    chosen = candidates[rng.choice(len(candidates), n_sites, replace=False)]
    rows = stack.iloc[rng.integers(0, len(stack), n_sites)].copy().reset_index(drop=True)
    rows["pos_y_img"], rows["pos_x_img"] = chosen[:, 0].astype(int), chosen[:, 1].astype(int)
    rows["pos_y"], rows["pos_x"] = rows.pos_y_img - HALF_BOX, rows.pos_x_img - HALF_BOX
    rows["pos_y_img_angstrom"] = rows.pos_y_img * pixel
    rows["pos_x_img_angstrom"] = rows.pos_x_img * pixel
    rows["filament"] = -1
    return rows


def score(stack_path: pathlib.Path, template: pathlib.Path, label: str, args,
          suffix: str) -> np.ndarray:
    """Cross-correlation (Leopard's refined_mip) of every row, at its pose as given.

    Reused from disk unless ``--rescore``: a stack, a template and a low-pass fix it.
    """
    out = stack_path.parent / f"{stack_path.stem}_{label}{suffix}_refined.csv"
    if out.is_file() and not args.rescore:
        return pd.read_csv(out, index_col=0)["refined_mip"].to_numpy()
    out.unlink(missing_ok=True)
    config = MT_ROOT / "configs" / f"kinesin_score_{stack_path.stem}_{label}{suffix}.yaml"
    write_config(stack_path, config, str(template.relative_to(MT_ROOT)),
                 MT_ROOT / "configs" / f"match_tm_{args.tag}.yaml", args.gpus,
                 neutral_pose=True)
    if args.lowpass_A is not None:
        settings = yaml.safe_load(config.read_text())
        settings["preprocessing_filters"]["bandpass_filter"] = {
            "enabled": True, "low_freq_cutoff": 0.0,
            "high_freq_cutoff": args.pixel / args.lowpass_A, "falloff": 0.01}
        config.write_text(yaml.safe_dump(settings, sort_keys=False))
    RefineTemplateManager.from_yaml(str(config)).run_refine_template(str(out), 80)
    return pd.read_csv(out, index_col=0)["refined_mip"].to_numpy()


def occupancy_from_ratio(ratio: np.ndarray, calibration: pd.DataFrame, name: str):
    """Occupancy whose noise-free signal ratio (template / bare) equals ``ratio``."""
    curve = (calibration[f"signal_{name}"] / calibration["signal_bare"]).to_numpy()
    occupancy = calibration["occupancy"].to_numpy()
    order = np.argsort(curve)
    # beyond the calibrated range: extend linearly from the end points
    slope_lo = (curve[order][1] - curve[order][0]) / (occupancy[order][1] - occupancy[order][0])
    slope_hi = (curve[order][-1] - curve[order][-2]) / (occupancy[order][-1] - occupancy[order][-2])
    ratio = np.atleast_1d(ratio)
    out = np.interp(ratio, curve[order], occupancy[order])
    low, high = ratio < curve[order][0], ratio > curve[order][-1]
    out[low] = occupancy[order][0] + (ratio[low] - curve[order][0]) / slope_lo
    out[high] = occupancy[order][-1] + (ratio[high] - curve[order][-1]) / slope_hi
    return out


def summarise(frame: pd.DataFrame, calibration: pd.DataFrame, label: str, rng, n_boot: int,
              block: int):
    """Mean z per template, occupancy from motors_avg, register preference; with CIs."""
    z = {name: frame[f"z_{name}"].to_numpy() for name in TEMPLATES}
    n = len(frame)
    # resample whole blocks of consecutive lattice indices, sites within them kept
    keys = list(zip(frame["filament"], frame["axial_index"] // block))
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    members = list(groups.values())
    picks = rng.integers(0, len(members), (n_boot, len(members)))

    def ratio(name, pick=None):
        if pick is None:
            return z[name].mean() / z["bare"].mean()
        index = np.concatenate([members[j] for j in pick])
        return z[name][index].mean() / z["bare"][index].mean()

    out = {"set": label, "sites": n}
    for name in TEMPLATES:
        out[f"z_{name}"] = z[name].mean()
    for name in [n for n in ("motors_avg", "motors", "decorated_avg") if n in TEMPLATES]:
        point = occupancy_from_ratio(ratio(name), calibration, name)[0]
        boot = occupancy_from_ratio(np.array([ratio(name, pick) for pick in picks]),
                                    calibration, name)
        out[f"occ_{name}"] = point
        out[f"occ_{name}_lo"], out[f"occ_{name}_hi"] = np.percentile(boot, [2.5, 97.5])
    out["blocks"] = len(members)
    if "motors" not in TEMPLATES or "motors_shift" not in TEMPLATES:
        return out
    sign = np.where(frame["axial_index"].to_numpy() % 2 == 0, 1.0, -1.0)
    diff = (z["motors"] - z["motors_shift"]) * sign
    boot = np.array([diff[np.concatenate([members[j] for j in pick])].mean()
                     for pick in picks[:1000]])
    out["register_A_minus_B"] = diff.mean()
    out["register_t"] = diff.mean() / boot.std(ddof=1)
    return out


def main() -> None:
    """Score the neutral stack and a background stack with every kinesin template."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="kinesin_2_0031_14pf")
    parser.add_argument("--stack-suffix", default="neutral",
                        help="score stack_sites_<tag>_<suffix>.csv, e.g. neutral_rollfit")
    parser.add_argument("--results", type=pathlib.Path,
                        default=MT_ROOT / "results_kinesin_2_0031")
    parser.add_argument("--paths", type=pathlib.Path,
                        default=MT_ROOT / "configs" / "filament_constraint_kinesin_2_0031.paths.json")
    parser.add_argument("--calibration", type=pathlib.Path,
                        default=MT_ROOT / "results_kinesin_synthetic" / "occupancy_summary.csv")
    parser.add_argument("--background-paths", type=pathlib.Path, nargs="*", default=[],
                        help="further paths the background must clear (e.g. every traced "
                             "filament of a crowded frame)")
    parser.add_argument("--background-min-row", type=int, default=0,
                        help="background only below this row (e.g. off carbon above a hole "
                             "edge)")
    parser.add_argument("--clear-nm", type=float, default=60.0)
    parser.add_argument("--max-x-fraction", type=float, default=0.78,
                        help="background only left of this fraction of the width (the "
                             "right edge of 2_0031 has uncurated striations)")
    parser.add_argument("--n-background", type=int, default=1500)
    parser.add_argument("--rescore", action="store_true",
                        help="recompute scores already on disk")
    parser.add_argument("--block-monomers", type=int, default=8,
                        help="bootstrap block: consecutive lattice indices resampled together")
    parser.add_argument("--segment-monomers", type=int, default=40,
                        help="segment length along a tube, in lattice indices (40 = 167 nm)")
    parser.add_argument("--gpus", type=int, nargs="+", default=[3])
    parser.add_argument("--lowpass-A", type=float, default=None,
                        help="low-pass image and templates at this resolution (Leopard's "
                             "bandpass); pair with --calibration from the same low-pass")
    parser.add_argument("--templates", nargs="+", default=list(TEMPLATES),
                        help="score only these (bare is always included)")
    parser.add_argument("--bootstrap", type=int, default=4000)
    args = parser.parse_args()
    args.lowpass = "" if args.lowpass_A is None else f"_lp{args.lowpass_A:g}"
    args.suffix = ("" if args.stack_suffix == "neutral" else f"_{args.stack_suffix}") + args.lowpass
    wanted = ["bare"] + [t for t in args.templates if t != "bare"]
    for name in list(TEMPLATES):
        if name not in wanted:
            del TEMPLATES[name]

    import mrcfile  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(0)
    stack_path = args.results / f"stack_sites_{args.tag}_{args.stack_suffix}.csv"
    stack = pd.read_csv(stack_path, index_col=0)
    micrograph = MT_ROOT / stack["micrograph_path"].iloc[0]
    with mrcfile.open(str(micrograph), permissive=True, header_only=True) as handle:
        shape = (int(handle.header.ny), int(handle.header.nx))
    pixel = float(stack["pixel_size"].iloc[0])
    args.pixel = pixel
    bg_path = args.results / f"kinesin_background_{args.tag}_n{args.n_background}.csv"
    if not bg_path.is_file() or args.rescore:
        poses = pd.read_csv(args.results / f"stack_sites_{args.tag}_neutral.csv", index_col=0)
        background = background_stack(poses, [args.paths, *args.background_paths],
                                      (shape[0], int(shape[1] * args.max_x_fraction)),
                                      pixel, args.clear_nm, args.n_background, rng,
                                      args.background_min_row)
        background.to_csv(bg_path)
    background = pd.read_csv(bg_path, index_col=0)
    print(f"{len(stack)} neutral sites on {stack.filament.nunique()} tubes, "
          f"{len(background)} background positions > {args.clear_nm:g} nm from every tube")

    frame = stack[["filament", "segment", "axial_index"]].copy()
    for label, stem in TEMPLATES.items():
        template = MT_ROOT / "maps" / f"{stem}{TAG}.mrc"
        start = time.time()
        sites = score(stack_path, template, label, args, args.suffix)
        null = score(bg_path, template, label, args, args.lowpass)
        mean, sd = null.mean(), null.std(ddof=1)
        frame[f"cc_{label}"] = sites
        frame[f"z_{label}"] = (sites - mean) / sd
        print(f"  {label:<16} background cc {mean:+.4f} +/- {sd:.4f}; sites mean z "
              f"{frame[f'z_{label}'].mean():+.3f}  ({time.time() - start:.0f} s)")
    frame.to_csv(args.results / f"kinesin_scores_{args.tag}{args.suffix}.csv")

    calibration = pd.read_csv(args.calibration)
    rows = [summarise(frame, calibration, "all tubes", rng, args.bootstrap, args.block_monomers)]
    for fil, group in frame.groupby("filament"):
        rows.append(summarise(group, calibration, f"tube {fil}", rng, args.bootstrap,
                              args.block_monomers))
    table = pd.DataFrame(rows)
    table.to_csv(args.results / f"kinesin_occupancy_{args.tag}{args.suffix}.csv", index=False)
    print("\nmean z (own-background normalised):")
    print(table[["set", "sites"] + [f"z_{n}" for n in TEMPLATES]].to_string(
        index=False, float_format=lambda v: f"{v:.2f}"))
    print("\noccupancy (95% bootstrap over sites), from each template's ratio to the bare "
          "ring, calibrated on the synthetic:")
    for _, row in table.iterrows():
        print(f"  {row['set']:<10} n={row['sites']:3d} ({row['blocks']:2d} blocks)  motors_avg {row.occ_motors_avg:5.2f} "
              f"[{row.occ_motors_avg_lo:5.2f}, {row.occ_motors_avg_hi:5.2f}]   "
              f"decorated_avg {row.occ_decorated_avg:5.2f} [{row.occ_decorated_avg_lo:5.2f}, "
              f"{row.occ_decorated_avg_hi:5.2f}]   register A-B {row.register_A_minus_B:+.2f} "
              f"(t {row.register_t:+.1f})")

    # along each tube: segments of --segment-monomers lattice indices
    segments = []
    for fil, group in frame.groupby("filament"):
        first = int(group.axial_index.min())
        for start in range(first, int(group.axial_index.max()) + 1, args.segment_monomers):
            part = group[(group.axial_index >= start)
                         & (group.axial_index < start + args.segment_monomers)]
            if len(part) < args.segment_monomers // 4:
                continue
            row = summarise(part, calibration, f"tube {fil}", rng, 1000, args.block_monomers)
            row["axial_from"], row["axial_to"] = part.axial_index.min(), part.axial_index.max()
            segments.append(row)
    seg = pd.DataFrame(segments)
    seg.to_csv(args.results / f"kinesin_occupancy_segments_{args.tag}{args.suffix}.csv",
               index=False)
    print(f"\nalong the tubes ({args.segment_monomers}-monomer segments, motors_avg):")
    for _, row in seg.iterrows():
        print(f"  {row['set']:<8} axial {int(row.axial_from):4d}..{int(row.axial_to):<4d} "
              f"n={row['sites']:2d}  occupancy {row.occ_motors_avg:5.2f} "
              f"[{row.occ_motors_avg_lo:5.2f}, {row.occ_motors_avg_hi:5.2f}]")


if __name__ == "__main__":
    main()
