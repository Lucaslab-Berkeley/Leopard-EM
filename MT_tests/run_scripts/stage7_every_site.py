"""Stage 7's register readout at EVERY predicted lattice site, refined -- and at decoys.

Stage 7 scores the in-register patch and its one-monomer-out partner only where Stage 4
found a site (23-81% of the lattice). ``lattice_fill.py sites`` has already predicted
EVERY (axial index, protofilament) in each filament's span at Stage 6b's neutral pose,
plus decoys (the same poses moved to empty background). Here both Stage 7 templates are
refined at all of them with Stage 7's own ``score()`` -- the same refine config, the
same local search, the patch table's own statistic maps -- under new labels, and the
register is read with Stage 7's estimator (``paired_register``) on:

* every lattice site; the found sites only (``detected``); the never-found ones;
* every site whose refined z (mean of the two templates) clears the decoys' 99th pct;
* the decoys: the same estimator on background, which must be null.

It also reports how far refine moved each site from its predicted position (found vs
never-found vs decoys), since a never-found site that slides half a monomer is read at
the wrong parity.

Usage:
    python run_scripts/stage7_every_site.py \\
        --table results_bundle/output_correlation_table_bundle_strong_patch.h5 --gpus 2 3
"""

import argparse
import pathlib
import sys
import time

import mrcfile
import numpy as np
import pandas as pd
import yaml

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from frame_geometry import frame_geometry, map_for  # noqa: E402
from paired_register_test import build_stack  # noqa: E402
from refine_sites_curved import EXTRACTION_MARGIN_PX  # noqa: E402
from stage7_register import PATCH, paired_register, score  # noqa: E402


def stage4_rows(part):
    """Lattice-fill rows in the columns ``build_stack`` reads from Stage 4's sites.

    The correlation columns are not refine inputs (refine re-reads the statistic maps
    at each site and recomputes z); they are only carried through, so NaN.
    """
    return pd.DataFrame({
        "x": part["pos_x"].astype(int).to_numpy(),
        "y": part["pos_y"].astype(int).to_numpy(),
        "phi": part["phi"].to_numpy(), "theta": part["theta"].to_numpy(),
        "psi": part["psi"].to_numpy(),
        "relative_defocus": part["relative_defocus"].to_numpy(),
        "correlation_value": np.nan, "z_score": np.nan,
        "correlation_mean": np.nan, "correlation_variance": np.nan,
    })


def refined(stem, stack, label, results, search_config, pixel_size, gpus, batch):
    """Stage 7's ``score`` under a new label; never overwrites. Returns (frame, seconds).

    If this label's refined output already exists it is read back instead (refine
    writes it only on completion); a stack or config without it is a partial run and
    stops the script rather than being overwritten.
    """
    out = results / f"stage7_refined_{label}.csv"
    stack_path = results / f"stage7_stack_{label}.csv"
    config_path = MT_ROOT / "configs" / f"stage7_{label}.yaml"
    if out.is_file():
        saved = pd.read_csv(stack_path, index_col=0)
        if (len(saved) != len(stack)
                or not np.array_equal(saved["pos_x"].to_numpy(), stack["pos_x"].to_numpy())
                or not np.array_equal(saved["pos_y"].to_numpy(), stack["pos_y"].to_numpy())):
            raise SystemExit(f"{out.name} exists but its stack is not this one")
        print(f"    {label}: reusing {out.relative_to(MT_ROOT)}")
        return pd.read_csv(out, index_col=0), float("nan")
    for path in (stack_path, config_path):
        if path.is_file():
            raise SystemExit(f"{path} exists without {out.name}: not overwriting it")
    start = time.time()
    score(stem, stack, label, results, search_config, pixel_size, gpus, batch)
    return pd.read_csv(out, index_col=0), time.time() - start


UNSEARCHED_VARIANCE = 1e9   # constrained searches write 1e10 where they never searched


def data_positions(sites):
    """(x, y) table positions -- the statistic maps' own frame (refine's top-left)."""
    return sites[["pos_x", "pos_y"]].to_numpy(dtype=int)


def axial_step(sites):
    """Per site, the in-plane displacement (px) of one lattice index along its own
    protofilament -- the local axis direction and the projected rise."""
    step = np.full((len(sites), 2), np.nan)
    xy = sites[["pos_x", "pos_y"]].to_numpy(dtype=float)
    k = sites["axial_index"].to_numpy(dtype=float)
    keys = sites["filament"].astype(str) + "/" + sites["protofilament"].astype(str)
    for _, rows in sites.groupby(keys.to_numpy()).indices.items():
        if len(rows) < 3:
            continue
        rows = rows[np.argsort(k[rows])]
        for axis in (0, 1):
            step[rows, axis] = np.gradient(xy[rows, axis], k[rows])
    return step


def moves(ref, pixel_size, step):
    """Refined minus starting position: total (A) and along the axis (lattice indices)."""
    dx = (ref["refined_pos_x"] - ref["pos_x"]).to_numpy(dtype=float)
    dy = (ref["refined_pos_y"] - ref["pos_y"]).to_numpy(dtype=float)
    total = np.hypot(dx, dy) * pixel_size
    axial = (dx * step[:, 0] + dy * step[:, 1]) / np.sum(step ** 2, axis=1)
    edge = np.maximum(np.abs(dx), np.abs(dy)) >= EXTRACTION_MARGIN_PX // 2
    return total, axial, edge


def show(name, frame, rows):
    """Stage 7's estimator per filament, with its two-dimer control."""
    first = True
    for label in sorted(frame["filament"].unique()):
        part = frame[frame["filament"] == label].dropna(subset=["in", "out"])
        if len(part) < 10:
            continue
        r = paired_register(part["axial_index"], part["in"], part["out"])
        control = paired_register(part["axial_index"] // 2, part["in"], part["out"])
        rows.append({"variant": name, "filament": int(label), "n": r["n"],
                     "register_z": r["mean"], "sem": r["sem"], "t": r["t"],
                     "offset": r["offset"], "control_t": control["t"]})
        print(f"{name if first else '':<40}{label:>4}{r['n']:>7}{r['mean']:>9.3f} ± "
              f"{r['sem']:.3f}{r['t']:>7.2f}{r['offset']:>+8.3f}{control['t']:>10.2f}")
        first = False


# pylint: disable=too-many-locals,too-many-statements
def main() -> None:
    """Refine the Stage 7 pair at every lattice site and at the decoys; report."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=pathlib.Path, required=True,
                        help="the PATCH table (lattice_fill.py sites run on it)")
    parser.add_argument("--stem", default=PATCH, help="the in-register patch")
    parser.add_argument("--search-config", type=pathlib.Path, default=None)
    parser.add_argument("--ctf-b-factor", type=float, default=60.0)
    parser.add_argument("--gpus", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--label", default=None,
                        help="name of this run's outputs (default: the table's tag), e.g. "
                             "<tag>_4rings to score another patch pair (--stem) at the same "
                             "sites: refined scores are reused BY NAME, so a second pair "
                             "under the default name would read the first pair's back")
    args = parser.parse_args()

    table = args.table if args.table.is_absolute() else MT_ROOT / args.table
    tag = table.stem.removeprefix("output_correlation_table_")
    run = args.label or tag
    results = table.parent
    frame = frame_geometry(table)
    search_config = args.search_config or MT_ROOT / "configs" / f"match_tm_{tag}.yaml"
    search = yaml.safe_load(search_config.read_text())
    optics = dict(search["optics_group"])
    optics.pop("label", None)
    optics["ctf_B_factor"] = args.ctf_b_factor
    partner = f"{args.stem}_shift1"
    for stem in (args.stem, partner):
        if not map_for(stem, frame.pixel_size).is_file():
            raise SystemExit(f"no map for {stem} at {frame.pixel_size} A/px -- simulate it")
    out_csv = results / f"stage7_every_register_{run}.csv"
    summary = results / f"stage7_every_register_summary_{run}.csv"
    for path in (out_csv, summary):
        if path.is_file():
            raise SystemExit(f"{path} exists: not overwriting it")

    sites = pd.read_csv(results / f"lattice_fill_sites_{tag}.csv", index_col=0)
    sites["detected"] = sites["detected"].astype(bool)
    sites["decoy"] = sites["decoy"].astype(bool)
    print(frame.describe())
    groups = {"every": sites[~sites["decoy"]], "decoy": sites[sites["decoy"]]}
    real = groups["every"]
    print(f"{len(real)} lattice sites ({int(real['detected'].sum())} found, "
          f"{int((~real['detected']).sum())} never found) over "
          f"{real['filament'].nunique()} filament(s), {len(groups['decoy'])} decoys; "
          f"pair {args.stem} / {partner}; GPUs {args.gpus}")

    scored, walls = {}, {}
    for group, part in groups.items():
        stack = build_stack(stage4_rows(part), optics, search["micrograph_path"],
                            search["template_volume_path"],
                            str(results.relative_to(MT_ROOT)), tag, frame)
        for side, stem in (("in", args.stem), ("out", partner)):
            label = f"{run}_{group}_{side}"
            ref, walls[label] = refined(stem, stack, label, results, search_config,
                                        frame.pixel_size, args.gpus, args.batch_size)
            if len(ref) != len(part) or not (
                    np.array_equal(ref["pos_x"].to_numpy(), part["pos_x"].to_numpy())
                    and np.array_equal(ref["pos_y"].to_numpy(), part["pos_y"].to_numpy())):
                raise SystemExit(f"{label}: refined rows are not the sites row for row")
            ref.index = part.index   # refine's output is positional, row i = site i
            scored[(group, side)] = ref
    print("  wall: " + ", ".join(f"{k} {v / 60:.1f} min" for k, v in walls.items()
                                 if np.isfinite(v)))

    data = sites[[c for c in ("filament", "axial_index", "protofilament", "detected",
                              "decoy", "truth") if c in sites]].copy()
    step = axial_step(real)
    step = np.vstack([step, step])   # decoy i is site i moved (lattice_fill ``decoys``)
    if (not np.array_equal(data["decoy"].to_numpy(),
                           np.repeat([False, True], len(real)))
            or not np.array_equal(groups["decoy"]["axial_index"].to_numpy(),
                                  real["axial_index"].to_numpy())):
        raise SystemExit("decoys are not the sites in order")
    # A constrained search writes its statistic maps only inside its band (mean 0,
    # variance 1e10 elsewhere), so refine's z at a site outside it is cc * 1e-5. There
    # the raw cc stands in: inside the band the maps are mean ~0, sd ~1, so the cc is in
    # z units. (The register t and control t are scale-free either way.)
    with mrcfile.open(MT_ROOT / stack["correlation_variance_path"].iloc[0],
                      permissive=True) as handle:
        variance = np.asarray(handle.data)
    with mrcfile.open(MT_ROOT / stack["correlation_average_path"].iloc[0],
                      permissive=True) as handle:
        average = np.asarray(handle.data)
    y = np.clip(data_positions(sites)[:, 1], 0, variance.shape[0] - 1)
    x = np.clip(data_positions(sites)[:, 0], 0, variance.shape[1] - 1)
    data["unsearched"] = variance[y, x] >= UNSEARCHED_VARIANCE
    if data["unsearched"].any():
        band = variance < UNSEARCHED_VARIANCE
        print(f"\nNOTE: {int(data['unsearched'][data['decoy']].sum())} decoys and "
              f"{int(data['unsearched'][~data['decoy']].sum())} lattice sites lie where the "
              f"table never searched (statistic maps: variance >= {UNSEARCHED_VARIANCE:g});"
              f" there the raw cc (refined_mip) is used as the score. In the searched band "
              f"the maps are mean {np.median(average[band]):+.4f}, sd "
              f"{np.median(np.sqrt(variance[band])):.4f} (median), so the cc is in z units.")
    for side in ("in", "out"):
        ref = pd.concat([scored[("every", side)], scored[("decoy", side)]]).loc[data.index]
        data[f"{side}_z"] = ref["refined_scaled_mip"].to_numpy()
        data[f"{side}_mip"] = ref["refined_mip"].to_numpy()
        data[side] = np.where(data["unsearched"], data[f"{side}_mip"], data[f"{side}_z"])
        total, axial, edge = moves(ref, frame.pixel_size, step)
        data[f"{side}_shift_A"], data[f"{side}_axial_shift"] = total, axial
        data[f"{side}_at_edge"] = edge
        for angle in ("phi", "theta", "psi"):
            data[f"{side}_d{angle}"] = (ref[f"refined_{angle}"] - ref[angle] + 180) % 360 - 180

    # ---- scores against the decoy null ---------------------------------------------
    sym = (data["in"] + data["out"]) / 2
    kinds = {"decoys": data["decoy"], "found": ~data["decoy"] & data["detected"],
             "never found": ~data["decoy"] & ~data["detected"]}
    print(f"\n{'refined z':<24}" + "".join(f"{k:>22}" for k in kinds))
    print(f"{'':<24}" + f"{'median   p99  p99.9':>22}" * len(kinds))
    for name, values in (("in-register", data["in"]), ("one monomer out", data["out"]),
                         ("mean of the two", sym)):
        print(f"{name:<24}" + "".join(
            f"{values[m].median():>8.2f}{np.percentile(values[m], 99):>7.2f}"
            f"{np.percentile(values[m], 99.9):>7.2f}" for m in kinds.values()))
    cut = float(np.percentile(sym[data["decoy"]], 99))
    above = sym >= cut
    print(f"  above the decoys' 99th percentile (mean of the two, z {cut:.2f}): found "
          f"{int((above & kinds['found']).sum())}/{int(kinds['found'].sum())}, never found "
          f"{int((above & kinds['never found']).sum())}/{int(kinds['never found'].sum())}, "
          f"decoys {int((above & data['decoy']).sum())}/{int(data['decoy'].sum())}")

    # ---- how far refine moved each site --------------------------------------------
    print(f"\nrefine's move from the predicted pose (search window +/-"
          f"{EXTRACTION_MARGIN_PX // 2} px = {EXTRACTION_MARGIN_PX // 2 * frame.pixel_size:.0f}"
          f" A per axis)")
    print(f"{'':<14}{'template':<9}{'shift A med':>12}{'p90':>7}{'|axial| idx med':>16}"
          f"{'p90':>6}{'>0.5 idx':>9}{'at edge':>9}{'|dpsi| p90':>11}{'|dtheta| p90':>13}")
    for name, mask in kinds.items():
        for side in ("in", "out"):
            part = data[mask]
            axial = part[f"{side}_axial_shift"].abs()
            print(f"{name:<14}{side:<9}{part[f'{side}_shift_A'].median():>12.1f}"
                  f"{np.percentile(part[f'{side}_shift_A'], 90):>7.1f}"
                  f"{axial.median():>16.2f}{np.nanpercentile(axial, 90):>6.2f}"
                  f"{(axial > 0.5).mean():>9.1%}{part[f'{side}_at_edge'].mean():>9.1%}"
                  f"{np.percentile(part[f'{side}_dpsi'].abs(), 90):>11.2f}"
                  f"{np.percentile(part[f'{side}_dtheta'].abs(), 90):>13.2f}")
    print("  axial: the move along the site's own protofilament, in lattice indices (one "
          "index = one monomer);\n  decoys use their source site's axis.")

    # ---- the register --------------------------------------------------------------
    print("\n== REGISTER (Stage 7 estimator: in-register patch vs one monomer out)")
    print(f"{'':<40}{'fil':>4}{'sites':>7}{'register z':>17}{'t':>7}{'offset':>8}"
          f"{'control t':>10}")
    rows = []
    old_path = results / f"stage7_register_{tag}.csv"
    if old_path.is_file():
        old = pd.read_csv(old_path).rename(columns={"in_register": "in",
                                                    "out_of_register": "out"})
        show("OLD: Stage 7, found sites (Stage 4 pose)", old, rows)
    lattice = data[~data["decoy"]]
    for name, part in (
            ("REFINE every site", lattice),
            ("REFINE found sites (neutral pose)", lattice[lattice["detected"]]),
            ("REFINE never-found sites", lattice[~lattice["detected"]]),
            ("REFINE every site > decoy p99", lattice[above[lattice.index]]),
            ("REFINE every site, |axial move| <= 0.5",
             lattice[(lattice["in_axial_shift"].abs() <= 0.5)
                     & (lattice["out_axial_shift"].abs() <= 0.5)]),
            ("DECOYS (must be null)", data[data["decoy"]])):
        show(name, part, rows)
    print("  register z: |half the even-minus-odd difference of (in - out)|; offset: mean"
          " (in - out);\n  control: the same on a two-dimer period (index // 2), must be"
          " null.")

    data.to_csv(out_csv)
    pd.DataFrame(rows).to_csv(summary, index=False)
    print(f"\nwrote {out_csv.relative_to(MT_ROOT)} and {summary.relative_to(MT_ROOT)}")


# NOTE: invoking under __main__ is required for the multiprocessing/GPU distribution.
if __name__ == "__main__":
    main()
