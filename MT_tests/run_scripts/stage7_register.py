"""Stage 7: the monomer register as a PAIRED competition, per filament, on the patch.

At every site of the PATCH's own search, score the in-register patch and the same patch
built one monomer out of register (``--monomer-shift 1``), and take the difference. The
two are matched to the atom and differ only in which monomer is alpha, so the per-site
difference is alpha/beta discrimination with ice, defocus and occupancy cancelled.

The in-register template should win at even axial indices and the shifted one at odd,
so the register is the DIFFERENCE between the even-site and odd-site means of the
paired difference (halved). That, not the parity-signed mean, is the estimator: the
in-register template scores ~0.5 z higher EVERYWHERE -- the sites were found and placed by
it, so its partner refines from someone else's optimum -- and a parity-signed mean turns
that offset times any even/odd count imbalance into signal. On 823 filament 3 (487 even,
291 odd) that inflated t from 24.9 to 27.6, and shuffled indices reached a median t of
3.6. Which parity is alpha is set by each lattice's own phase origin, so the sign is
arbitrary per filament and the magnitude is what counts.

Sites come from Stages 3-4 run on the patch table (``--stem`` the patch,
``--resolve-protofilaments`` in Stage 4): membership from each detection's own axis
point -- essential for a patch, whose box centre sits 85.8 A off the axis -- and one site
per (axial index, protofilament). Everything run-specific is read from the search config
and the table; the partner map is chosen by the search's pixel size.

Usage:
    python run_scripts/stage7_register.py \\
        --table results_in_situ_823/output_correlation_table_in_situ_823_13pf_patch.h5
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import yaml

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from frame_geometry import frame_geometry, legacy_frame_geometry, map_for  # noqa: E402
from paired_register_test import build_stack  # noqa: E402
from refine_sites_curved import write_config  # noqa: E402

PATCH = "6dpu_atoms_6dpv_lattice_13pf_patch5_2rings_flatB"


def paired_register(axial_index, in_register, out_of_register):
    """Register of ONE lattice: half the even-minus-odd difference of paired deltas.

    Immune to a uniform offset between the two templates, whatever the even/odd counts.
    Returns the balanced delta, its standard error, t, the percentage of score, and --
    as a diagnostic only -- the fraction of sites where the parity-expected template
    wins (0.5 = no discrimination).
    """
    index = np.asarray(axial_index).astype(int)
    raw = np.asarray(in_register) - np.asarray(out_of_register)
    even, odd = raw[index % 2 == 0], raw[index % 2 == 1]
    delta = 0.5 * (even.mean() - odd.mean())
    sem = 0.5 * float(np.sqrt(even.var(ddof=1) / len(even) + odd.var(ddof=1) / len(odd)))
    sign = np.where(index % 2 == 0, 1.0, -1.0) * np.sign(delta)
    agreement = float(np.mean(np.sign(raw - raw.mean()) == sign))
    return {"mean": float(abs(delta)), "sem": sem,
            "t": float(abs(delta) / sem) if sem > 0 else float("nan"),
            "percent": float(100 * abs(delta) / np.mean(in_register)),
            "agreement": agreement, "n": len(raw),
            "n_even": len(even), "n_odd": len(odd),
            "offset": float(raw.mean())}


def score(stem, stack, label, results, search_config, pixel_size, gpus, batch):
    """Refine one template over the stack; refined z per site."""
    # pylint: disable=import-outside-toplevel
    from leopard_em.pydantic_models.managers import RefineTemplateManager

    template = str(map_for(stem, pixel_size).relative_to(MT_ROOT))
    stack = stack.copy()
    stack["template_path"] = template
    stack_path = results / f"stage7_stack_{label}.csv"
    stack.to_csv(stack_path, index=True)
    config_path = MT_ROOT / "configs" / f"stage7_{label}.yaml"
    write_config(stack_path, config_path, template, search_config, gpus)
    out = results / f"stage7_refined_{label}.csv"
    start = time.time()
    RefineTemplateManager.from_yaml(str(config_path)).run_refine_template(
        str(out), batch, allow_file_overwrite=True)
    print(f"    {label}: {len(stack)} sites in {time.time() - start:.0f}s")
    return pd.read_csv(out, index_col=0)["refined_scaled_mip"].to_numpy()


def report(sites, in_z, out_z):
    """Per filament, then pooled with each filament's own sign."""
    print(f"\n{'filament':<9} {'sites':>6} {'even/odd':>9} {'register z':>16} {'t':>7} "
          f"{'% score':>8} {'offset z':>9} {'control t':>10}")
    for label in sorted(sites["filament"].unique()):
        mine = (sites["filament"] == label).to_numpy()
        if mine.sum() < 10:
            print(f"{label:<9} {mine.sum():>6}  too few sites")
            continue
        index = sites["axial_index"].to_numpy()[mine]
        r = paired_register(index, in_z[mine], out_z[mine])
        control = paired_register(index // 2, in_z[mine], out_z[mine])
        print(f"{label:<9} {r['n']:>6} {r['n_even']:>4}/{r['n_odd']:<4} "
              f"{r['mean']:8.3f} ± {r['sem']:.3f} {r['t']:>7.2f} {r['percent']:>7.1f}% "
              f"{r['offset']:>+9.3f} {control['t']:>10.2f}")
    print("\n  register: half the even-minus-odd difference of (in - out); its sign is set by")
    print("  each lattice's phase origin, so only the magnitude is reported.")
    print("  offset: mean (in - out) -- a uniform advantage for the template that found the")
    print("  sites, which the balanced estimator cancels.")
    print("  control: the same estimator on a two-dimer period, which must be null.")
    print("  Reference: the curved synthetic, whose register is known, reads 8.2% of score"
          " at t = 20.5 by this estimator.")


# pylint: disable=too-many-locals
def main() -> None:
    """Score the pair at every Stage 4 patch site and report per filament."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=pathlib.Path, required=True,
                        help="the PATCH search's table, with Stages 3-4 run on it")
    parser.add_argument("--stem", default=PATCH, help="the in-register patch")
    parser.add_argument("--search-config", type=pathlib.Path, default=None)
    parser.add_argument("--ctf-b-factor", type=float, default=60.0)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--in-vitro-legacy", action="store_true")
    parser.add_argument("--report-only", action="store_true",
                        help="re-read stage7_register_<tag>.csv; no refinement")
    args = parser.parse_args()

    table = args.table if args.table.is_absolute() else MT_ROOT / args.table
    tag = table.stem.removeprefix("output_correlation_table_")
    results = table.parent
    if args.report_only:
        saved = pd.read_csv(results / f"stage7_register_{tag}.csv")
        report(saved, saved["in_register"].to_numpy(), saved["out_of_register"].to_numpy())
        return
    frame = (legacy_frame_geometry(table, args.stem) if args.in_vitro_legacy
             else frame_geometry(table))
    search_config = args.search_config or MT_ROOT / "configs" / f"match_tm_{tag}.yaml"
    search = yaml.safe_load(search_config.read_text())
    optics = dict(search["optics_group"])
    optics.pop("label", None)
    optics["ctf_B_factor"] = args.ctf_b_factor
    partner = f"{args.stem}_shift1"
    for stem in (args.stem, partner):
        if not map_for(stem, frame.pixel_size).is_file():
            raise SystemExit(f"no map for {stem} at {frame.pixel_size} A/px -- simulate it")

    sites = pd.read_csv(results / f"stage4_sites_{tag}.csv")
    if (sites["protofilament"] < 0).all():
        print("WARNING: sites are not protofilament-resolved; run Stage 4 with "
              "--resolve-protofilaments for the patch")
    print(frame.describe())
    print(f"{len(sites)} patch sites over {sites['filament'].nunique()} filament(s); "
          f"pair {args.stem} / {partner}")

    stack = build_stack(sites, optics, search["micrograph_path"],
                        search["template_volume_path"], str(results.relative_to(MT_ROOT)),
                        tag, frame)
    in_z = score(args.stem, stack, f"{tag}_in", results, search_config,
                 frame.pixel_size, args.gpus, args.batch_size)
    out_z = score(partner, stack, f"{tag}_out", results, search_config,
                  frame.pixel_size, args.gpus, args.batch_size)
    report(sites, in_z, out_z)
    frame_out = sites[["filament", "axial_index", "protofilament"]].copy()
    frame_out["in_register"], frame_out["out_of_register"] = in_z, out_z
    frame_out.to_csv(results / f"stage7_register_{tag}.csv", index=False)


# NOTE: invoking under __main__ is required for the multiprocessing/GPU distribution.
if __name__ == "__main__":
    main()
