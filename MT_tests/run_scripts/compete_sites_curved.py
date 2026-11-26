"""Stage 6 of PIPELINE.md: compete rival RING templates at the reference's own sites.

No new match_template runs. Every rival is scored at the sites the reference already
found, paired site by site, which is what makes the comparison valid: ice, defocus and
near/far wall live in the per-pixel correlation mean and variance, and both templates see
the same pixel, so those cancel. Validated on the lattice 2x2 -- raw CC reproduced the
per-pixel z ranking exactly, contrasts within 6%, per-site sign agreement 97-100%.

Every rival is scored against the SAME normalisation (the reference's own correlation
mean and variance maps), so the reported z is a strictly monotonic function of raw CC at
each site. Ranking by it is therefore identical to ranking by raw CC, and no rival needs
correlation maps of its own -- which is the entire cost saving.

THE ROLL. phi is the azimuth about the tube, and a rival with a different protofilament
number has its protofilaments at different azimuths: measured, rivals sit 6-8 degrees
from the reference once their own N-fold period is removed. Scored at the reference's
phi they would lose for a reason that has nothing to do with whether their model is
right. Left-multiplying a rotation about lab z adds exactly to phi (verified), so each
site is replicated at a spread of roll seeds covering +/-180/N and refine_template
polishes each; the best seed wins. Both the reference and every rival get the same
treatment, so the extra freedom cannot favour one of them.

NEUTRAL POSE (``--neutral-pose``). Refinement lets each template climb to its own best
pose near a site, but the sites were FOUND by one of the templates, at its noise peaks:
on the known-domain synthetic the template that found a site wins it by +0.17 z
(compacted-found) or -0.12 z (expanded-found), the size of the lattice signal itself.
With ``--neutral-pose`` the stack is taken as given -- a pose no single detection chose
(Stage 6b builds it from the fitted lattice and each site's neighbours) -- and both
templates are scored AT it, with nothing searched: position pinned (no search window),
roll, theta, psi and defocus fixed. Even a roll search alone lets the finder climb back
towards its own noise peak (it left +0.06 / -0.10 z of the bias).

Usage:
    python run_scripts/compete_sites_curved.py --tag gdp_curved_13pf
    python run_scripts/compete_sites_curved.py --tag <tag>_neutral --neutral-pose
"""

import argparse
import pathlib
import re
import sys
import time

import numpy as np
import pandas as pd
import yaml

from leopard_em.analysis.filament_lattice import TemplateLatticeGeometry
from leopard_em.pydantic_models.managers import RefineTemplateManager

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyse_gdp_curved as A  # noqa: E402
from frame_geometry import map_for, template_half_width  # noqa: E402

ROLL_STEP_DEG = 4.0   # refine covers ~5 deg of phi after composition, so seeds overlap
POSITION_SLACK_PX = 20       # refine's search window about each site: +/- this

# The competition, per REFERENCE: which N and lattice the search used decides the rivals.
# Rings by default -- the patches are scored only when asked for (--only), at the patch's
# own sites, for Stage 6b's patch readout.
#
# One set per lattice and N, named <lattice><N>: compacted13 for GDP and in situ (6DPV
# lattice), expanded14 for GMPCPP (6DPU lattice), and the others for when the PF
# competition picks another N than the ring's (pf_switch.py then restarts the pipeline
# with that N's set; GMPCPP frame 0005 is expanded13). Each set competes the rings of
# its own lattice at 12-15 PF (the PDB format caps a 4-chain column at 15), the other
# lattice's ring at its N -- the same 6DPU atoms, only the repeat differs -- and both
# lattices' 5-PF patches at its N.
PF_RANGE = (12, 13, 14, 15)
LATTICES = ("compacted", "expanded")
_OTHER = {"compacted": "expanded", "expanded": "compacted"}
_SUFFIX = {"compacted": "cmp", "expanded": "exp"}   # key suffix of the OTHER lattice


def template_stem(lattice: str, n_protofilaments: int, part: str = "") -> str:
    """A built template's stem: 6DPU atoms on ``lattice`` (6DPV's repeat for compacted,
    6DPU's for expanded); ``part`` e.g. ``"_patch5"``."""
    source = "6dpu_atoms_6dpv_lattice" if lattice == "compacted" else "6dpu"
    return f"{source}_{n_protofilaments}pf{part}_2rings_flatB"


def parse_rival_set(name: str) -> tuple[str, int]:
    """``("compacted", 13)`` from ``"compacted13"``."""
    match = re.fullmatch(r"(compacted|expanded)(\d+)", name)
    if match is None:
        raise ValueError(f"not a rival set name: {name!r} (e.g. compacted13, expanded14)")
    return match.group(1), int(match.group(2))


def _rival_set(lattice: str, n: int) -> list[tuple[str, str, str]]:
    other, suffix = _OTHER[lattice], _SUFFIX[_OTHER[lattice]]
    entries = [(f"{k}pf", template_stem(lattice, k),
                "PF competition (reference)" if k == n else "PF competition")
               for k in PF_RANGE]
    entries.append((f"{n}pf_{suffix}", template_stem(other, n), f"{other} lattice at N={n}"))
    patches = {lattice: ("patch", template_stem(lattice, n, "_patch5"),
                         f"{lattice} {n}-PF patch (reference lattice)"),
               other: (f"patch_{suffix}", template_stem(other, n, "_patch5"),
                       f"{other} {n}-PF patch")}
    return entries + [patches["compacted"], patches["expanded"]]


RIVAL_SETS = {f"{lattice}{n}": _rival_set(lattice, n) for lattice in LATTICES for n in PF_RANGE}
DEFAULT_KEYS = {name: tuple(f"{k}pf" for k in PF_RANGE)
                + (f"{parse_rival_set(name)[1]}pf_{_SUFFIX[_OTHER[parse_rival_set(name)[0]]]}",)
                for name in RIVAL_SETS}
# (compacted key, expanded key, N): the lattice question, at the reference's N, between
# two templates built from the SAME atoms.
LATTICE_PAIRS = {name: ((f"{n}pf", f"{n}pf_exp", n) if lattice == "compacted"
                        else (f"{n}pf_cmp", f"{n}pf", n))
                 for name, (lattice, n) in ((s, parse_rival_set(s)) for s in RIVAL_SETS)}
PATCH_PAIRS = {name: (("patch", "patch_exp") if name.startswith("compacted")
                      else ("patch_cmp", "patch"))
               for name in RIVAL_SETS}


def roll_seeds(n_protofilaments: int) -> np.ndarray:
    """Roll offsets spanning one protofilament period, centred on zero."""
    half = 180.0 / n_protofilaments
    count = max(int(np.ceil(2 * half / ROLL_STEP_DEG)), 1)
    return np.linspace(-half, half, count + 1)[:-1]


def replicate(stack: pd.DataFrame, seeds: np.ndarray) -> pd.DataFrame:
    """One row per (site, roll seed), ordered so site = index // len(seeds)."""
    blocks = []
    for _, row in stack.iterrows():
        for seed in seeds:
            copy = row.copy()
            copy["phi"] = (copy["phi"] + seed) % 360.0
            blocks.append(copy)
    out = pd.DataFrame(blocks).reset_index(drop=True)
    out["particle_index"] = np.arange(len(out))
    return out


def write_config(stack_path, config_path, template, search_config, gpus,
                 neutral_pose=False):
    """Refine config with coarse steps taken from the search that found the sites.

    With ``neutral_pose`` nothing is refined: the valid correlation region is one pixel
    (position pinned) and orientation and defocus refinement are off, so each row is
    scored exactly at the pose it carries.
    """
    with open(search_config, encoding="utf-8") as handle:
        search = yaml.safe_load(handle)
    psi_step = float(search["orientation_search_config"]["psi_step"])
    theta_step = float(search["orientation_search_config"]["theta_step"])
    defocus_step = float(search["defocus_search_config"]["defocus_step"])
    # From the template's header: 600 -> 640 in vitro, 528 -> 568 in situ. The valid
    # region is extracted - template + 1, so a box equal to the template's pins it.
    box = 2 * template_half_width(MT_ROOT / template)
    slack = 0 if neutral_pose else 2 * POSITION_SLACK_PX
    config = {
        "template_volume_path": template,
        "particle_stack": {
            "df_path": str(stack_path.relative_to(MT_ROOT)),
            "extracted_box_size": [box + slack, box + slack],
            "original_template_size": [box, box],
        },
        "defocus_refinement_config": {
            "enabled": not neutral_pose, "defocus_min": -defocus_step / 2.0,
            "defocus_max": defocus_step / 2.0, "defocus_step": defocus_step / 4.0,
        },
        "pixel_size_refinement_config": {
            "enabled": False, "pixel_size_min": -0.01,
            "pixel_size_max": 0.01, "pixel_size_step": 0.01,
        },
        "orientation_refinement_config": {
            "enabled": not neutral_pose,
            "phi_step_coarse": psi_step, "phi_step_fine": 0.4,
            "theta_step_coarse": theta_step, "theta_step_fine": 0.4,
            "psi_step_coarse": psi_step, "psi_step_fine": 0.4,
            "base_grid_method": "uniform",
        },
        "preprocessing_filters": {
            "whitening_filter": {"enabled": True, "do_power_spectrum": True,
                                 "max_freq": 1.0},
            "bandpass_filter": {"enabled": False},
        },
        "computational_config": {"gpu_ids": gpus, "num_cpus": 8},
        "movie_config": {"enabled": False, "movie_path": "",
                         "deformation_field_path": "", "pre_exposure": 0.0,
                         "fluence_per_frame": 1.0},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))


def paired(values: np.ndarray, label: str) -> None:
    """A per-site contrast as mean, standard error and sign test."""
    n = len(values)
    mean = values.mean()
    sem = values.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    print(f"    {label:<28} {mean:+7.3f} +/- {sem:5.3f} z   "
          f"t = {mean / sem if sem else float('inf'):+7.2f}   "
          f"{int((values > 0).sum())}/{n} sites positive")


def report(scores: dict, label: str, n_sites: int,
           lattice_pair: tuple[str, str, int] = LATTICE_PAIRS["compacted13"]) -> str:
    """Rank the rivals over one set of sites and return the winner."""
    order = sorted(scores, key=lambda k: scores[k].mean(), reverse=True)
    print(f"\n{'=' * 78}\n{label}  ({n_sites} sites)\n{'=' * 78}")
    print("  ranking: " + " > ".join(f"{k} ({scores[k].mean():.3f})" for k in order))

    winner = order[0]
    print(f"\n  paired against {winner}, per site:")
    for key in order[1:]:
        paired(scores[winner] - scores[key], f"{winner} - {key}")

    pf = sorted([k for k in order if k.endswith("pf")],
                key=lambda k: int(k.replace("pf", "")))
    if len(pf) >= 3:
        means = [scores[k].mean() for k in pf]
        rising = all(b > a for a, b in zip(means, means[1:]))
        falling = all(b < a for a, b in zip(means, means[1:]))
        print("\n  mean z against N: "
              + ", ".join(f"{k}={scores[k].mean():.3f}" for k in pf))
        print("    monotonic in N -> the trend is template MASS, not structure"
              if rising or falling else
              "    middle-peaked -> structural, not a mass artefact")
        print(f"    best N = {pf[int(np.argmax(means))]}")

    # The lattice question, asked only at the winning N so the two are not confounded.
    compacted, expanded, n_lattice = lattice_pair
    if compacted in scores and expanded in scores:
        delta = scores[compacted] - scores[expanded]
        verdict = "COMPACTED" if delta.mean() > 0 else "EXPANDED"
        print(f"\n  lattice at N={n_lattice} ({compacted} is 6DPV/compacted, "
              f"{expanded} is 6DPU/expanded; same atoms):")
        paired(delta, f"{compacted} - {expanded}")
        print(f"    -> {verdict}")
    return winner


def main() -> None:
    """Score every rival at the reference's sites and rank them, paired."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="gdp_curved_13pf")
    parser.add_argument("--results", default="results_gdp_curved")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--only", nargs="+", default=None,
                        help="restrict to these rival keys")
    parser.add_argument("--report-only", action="store_true",
                        help="re-read saved results and re-report; no GPU work")
    parser.add_argument("--rivals", choices=sorted(RIVAL_SETS), default="compacted13",
                        help="the rival set for the search's reference: compacted13 "
                             "(GDP, in situ), expanded14 or expanded13 (GMPCPP)")
    parser.add_argument("--map-pixel-size", type=float, default=None,
                        help="render scale of the templates (default: the search's). "
                             "The optics, and so the CTF, stay at the search's pixel "
                             "size: only the object is rescaled (fit_pixel_size.py)")
    parser.add_argument("--neutral-pose", action="store_true",
                        help="score every template AT the stack's pose (Stage 6b's "
                             "neutral one): no refinement, roll searched only")
    args = parser.parse_args()

    base = pd.read_csv(
        MT_ROOT / args.results / f"stack_sites_{args.tag}.csv", index_col=0
    )
    if not args.neutral_pose:
        refined = pd.read_csv(
            MT_ROOT / args.results / f"refined_sites_{args.tag}.csv", index_col=0
        )
        # Start every rival from the REFINED angles and positions, so none of them is
        # handicapped by the reference's coarse grid.
        for column in ("phi", "theta", "psi", "pos_x", "pos_y", "relative_defocus"):
            base[column] = refined[f"refined_{column}"].to_numpy()
    # refine_template reads refined_pos_* in preference to pos_*, if a stack has them.
    base = base.drop(columns=[c for c in base.columns if c.startswith("refined_pos_")])
    base["pos_x"] = base["pos_x"].round().astype(int)
    base["pos_y"] = base["pos_y"].round().astype(int)
    # Rival maps must be simulated at the pixel size the sites were found at; the
    # stack carries the search's optics.
    pixel_size = float(base["pixel_size"].iloc[0])
    map_pixel_size = args.map_pixel_size or pixel_size
    print(f"{len(base)} sites from {args.tag}, {pixel_size} A/px"
          + (f"; templates rendered at {map_pixel_size}" if args.map_pixel_size else "")
          + "\n")

    scores, rolls = {}, {}
    for key, stem, purpose in RIVAL_SETS[args.rivals]:
        if (args.only and key not in args.only) or (not args.only
                                                     and key not in DEFAULT_KEYS[args.rivals]):
            continue
        pdb = MT_ROOT / "models" / f"{stem}.pdb"
        if not pdb.is_file():
            print(f"  {key:<10} no model at {pdb.name}, skipped")
            continue
        template = map_for(stem, map_pixel_size)
        if not template.is_file():
            print(f"  {key:<10} no map at {template.name}: simulate it with "
                  f"simulate_lattice_competition.py --pixel-size {map_pixel_size}, "
                  "skipped")
            continue
        geometry = TemplateLatticeGeometry.from_pdb(str(pdb))
        seeds = (np.zeros(1) if args.neutral_pose
                 else roll_seeds(geometry.n_protofilaments))
        stack = replicate(base, seeds)
        # Tagged, so two runs sharing a results folder cannot overwrite each other.
        stack_path = MT_ROOT / args.results / f"compete_stack_{args.tag}_{key}.csv"
        stack.to_csv(stack_path, index=True)

        config_path = MT_ROOT / "configs" / f"compete_{args.tag}_{key}.yaml"
        write_config(stack_path, config_path,
                     str(template.relative_to(MT_ROOT)),
                     MT_ROOT / "configs" / f"match_tm_{args.tag}.yaml", args.gpus,
                     neutral_pose=args.neutral_pose)

        out = MT_ROOT / args.results / f"compete_refined_{args.tag}_{key}.csv"
        legacy = MT_ROOT / args.results / f"compete_refined_{key}.csv"
        if args.report_only and not out.is_file() and legacy.is_file():
            out = legacy  # written before outputs were tagged
        start = time.time()
        if not (args.report_only and out.is_file()):
            manager = RefineTemplateManager.from_yaml(str(config_path))
            manager.run_refine_template(str(out), args.batch_size)
        result = pd.read_csv(out, index_col=0)

        # Best roll seed per site.
        z = result["refined_scaled_mip"].to_numpy().reshape(len(base), len(seeds))
        best = z.max(axis=1)
        chosen = seeds[z.argmax(axis=1)]
        scores[key] = best
        rolls[key] = chosen
        print(f"  {key:<10} N={geometry.n_protofilaments:2d}  {len(seeds)} roll seeds "
              f"({seeds.min():+.1f}..{seeds.max():+.1f}°)  mean z {best.mean():6.3f}  "
              f"median |roll| {np.median(np.abs(chosen)):4.1f}°  "
              f"{time.time() - start:5.1f}s   {purpose}")

    if len(scores) < 2:
        return

    # The two drawn regions are different microtubules. They need not share a
    # protofilament number or a nucleotide state, so pooling them could average away a
    # real difference -- or invent one. Report each on its own, then the pool.
    region = base["region"].to_numpy()
    for value in sorted(set(region.tolist())):
        mask = region == value
        if mask.sum() < 8:
            print(f"\nREGION {value}: only {int(mask.sum())} sites, not reported")
            continue
        report({k: v[mask] for k, v in scores.items()},
               f"REGION {value}", int(mask.sum()), LATTICE_PAIRS[args.rivals])
        print("  median |roll| chosen: " + ", ".join(
            f"{k} {np.median(np.abs(rolls[k][mask])):.1f}°" for k in scores))

    report(scores, "BOTH REGIONS POOLED", len(base), LATTICE_PAIRS[args.rivals])

    frame = pd.DataFrame(scores)
    frame["region"] = region
    frame["segment"] = base["segment"].to_numpy()
    frame.to_csv(MT_ROOT / args.results / f"compete_summary_{args.tag}.csv", index=True)


# NOTE: invoking under __main__ is required for the multiprocessing/GPU distribution.
if __name__ == "__main__":
    main()
