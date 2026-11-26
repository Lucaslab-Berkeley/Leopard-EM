"""After the PF competition: keep the ring's protofilament number, or switch -- per filament.

Everything after Stage 6 asks its question AT THE RING'S N: the lattice pair (Stage 6b),
the patch the register is read with (Stage 7) and its one-monomer-shifted partner, and
every pose they are scored at comes from the ring's search. When the PF competition
(``stage6_svd.py``) decides that a tube has another N, those answers are for the wrong
tube. GMPCPP frame 0005, searched with the 14-PF ring: 13 beat 14 by +1.04 z (t 9.0,
71/81 sites); searched again with the 13-PF ring, 13 won by +2.19 z (t 25.8, 81/81).

The decision, from the competition's per-site scores (``compete_svd_summary_<tag>.csv``,
every N at every site, with the site's filament as ``region``) -- one rule, applied twice:
a winner other than N replaces N only if it beats N with paired t >= ``--min-t`` and the
scores are not monotonic in N (that is template MASS, not structure: see
compete_sites_curved.report); otherwise N stays (with a warning if another N led).

1. All filaments pooled, against the ring's N  -> the micrograph's N.
2. Each filament with >= MIN_SITES sites, against the micrograph's N -> its own N. A
   filament with fewer sites follows the micrograph's N.
3. Each filament where another N LEADS, not monotonically, but below ``--min-t`` is
   probed: its filaments (grouped by the N they lead to) run at that N, over their own
   constraint, up to their own PF check (``run_svd_pipeline.sh`` with STOP_AFTER=pf_check,
   PF_SWITCH=0), and move there if, at those sites, that N beats theirs with t >=
   ``--min-t``. Needed because the sites come from the ring's search: on a synthetic with
   13- and 14-PF tubes searched with the 13-PF ring, the 14-PF tubes' sites lay 11.5 Å rms
   off their lattice and 14 led 13 there by only t 0.2 and 1.7; searched with the 14-PF
   ring, their sites lay 2.0 Å off and 14 won by t 23.3 and 27.4. A run that goes ahead at
   the probed N reuses the probe (same tag), so a confirmed probe costs nothing extra; one
   that is not confirmed costs its ring search and Stages 3-6 (~10 min for two tubes).
   Not done with PF_SWITCH=0 or in a run that was itself switched or split.

Then:

* every filament at the ring's N          -> carry on (SWITCH=0)
* every filament at one other N           -> restart the whole run at that N (SWITCH=1)
* filaments at different N                -> one run per N, each over only its own
                                             filaments (SWITCH=split)
* a run that was itself switched or split (``--switched-from``) never switches again: if
  its own competition puts any filament at another N, it stops (exit 1) -- no back and
  forth

Each new run gets everything it needs:

1. every template of its rival set (``compete_sites_curved.RIVAL_SETS``, the same lattice
   at the new N), its patch and the patch's ``_shift1`` partner: models built with
   ``models/build_mt_templates.py`` / ``build_patch_template.py`` where missing
   (deterministic: a rebuilt 13-PF patch is byte-identical to the stored one), maps
   simulated with ``simulate_lattice_competition`` at the search's pixel size, box and
   voltage where missing -- only after the CURRENT ring, re-simulated with those settings,
   reproduces its own map (cc >= 0.999); on the CPU if the GPU is short of memory.
2. ``configs/match_tm_<new tag>.yaml`` and ``..._patch.yaml``: the given configs with only
   the template (and the N in the Leopard result paths) changed. An existing one is reused
   if it searches the same micrograph with the same settings and the right template.
3. split only: its own constraint, ``<constraint>_fil<labels>.{paths.json,yaml,h5}`` -- the
   parent's drawn paths of its filaments, rasterised by ``make_filament_constraint.py`` with
   the parent's cone, angular steps and theta centre. Its filaments are renumbered 1, 2,
   ... in path order (parent filaments 3 and 4 are the split run's 1 and 2).

and ``<results>/pf_switch_<tag>.env`` tells ``run_svd_pipeline.sh`` what to do: SWITCH and
REASON, and NEW_TAG, NEW_RIVALS, NEW_RING_STEM, NEW_PATCH_STEM, NEW_RING_CONFIG,
NEW_PATCH_CONFIG for a switch, or N_CHILDREN and CHILD_<i>_{TAG, RIVALS, RING_STEM,
PATCH_STEM, RING_CONFIG, PATCH_CONFIG, CONSTRAINT, FILAMENTS} for a split.

Tags: the old one with its ``_<N>pf`` replaced (``pipe_gmpcpp_14pf`` -> ``pipe_gmpcpp_13pf``)
or ``_<N>pf`` appended; a split run's also names its filaments (``pipe_x_fil3-4_14pf``).
The lattice (compacted / expanded) is kept: it is Stage 6b's question, asked after the
switch at the right N.

Usage (from MT_tests/; run_svd_pipeline.sh calls it after stage6_pf):
    python run_scripts/pf_switch.py --tag pipe_gmpcpp_14pf --results results_svd_pipeline \\
        --rivals expanded14 --ring-config configs/match_tm_gmpcpp_14pf.yaml \\
        --patch-config configs/match_tm_gmpcpp_14pf_patch.yaml \\
        --ring-stem 6dpu_14pf_2rings_flatB --patch-stem 6dpu_14pf_patch5_2rings_flatB \\
        --constraint configs/filament_constraint_gmpcpp
"""

import argparse
import json
import gc
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time

import h5py
import mrcfile
import numpy as np
import pandas as pd
import yaml

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from compete_sites_curved import (  # noqa: E402
    PF_RANGE, RIVAL_SETS, parse_rival_set,
)
from frame_geometry import map_for  # noqa: E402

PIPELINE = MT_ROOT / "run_scripts" / "run_svd_pipeline.sh"
MAKE_CONSTRAINT = MT_ROOT.parent / "programs" / "constrained_search" / "make_filament_constraint.py"
MIN_SITES = 8        # a filament's own verdict needs this many sites (stage6_svd's report)
# ttsim3d's detector model per voltage: the K3 at 300 kV (in vitro, in situ) and in CDS
# mode at 200 kV (the Talos Arctica kinesin set) -- simulate_lattice_competition.py.
MTF_FOR_KV = {300: "k3_300kV_FL2", 200: "k3_200kV_CDS"}
EXPOSURE = 50.0
SETTINGS = ("micrograph_path", "optics_group", "orientation_search_config",
            "defocus_search_config", "preprocessing_filters")
STEM_RE = re.compile(r"^6dpu(?:_atoms_(6dpv)_lattice)?_(\d+)pf(?:_patch(\d+))?_(\d+)rings_flatB"
                     r"(?:_shift(\d+))?$")


def paired_t(diff: np.ndarray) -> float:
    sem = diff.std(ddof=1) / np.sqrt(len(diff))
    return float(diff.mean() / sem) if sem > 0 else float("inf")


def verdict(scores: pd.DataFrame, against: str, min_t: float) -> tuple[str, str, str | None]:
    """(the N key these sites belong at, one line why, an N to probe): ``against`` unless
    another N beats it decisively and not monotonically. An N that leads, not monotonically,
    but below ``min_t`` comes back third: sites placed by the wrong ring hold the right N
    back (a 14-PF tube at 13-PF sites: 14 led by t 0.2 and 1.7; at its own sites, t 23-27)."""
    keys = sorted((c for c in scores.columns if re.fullmatch(r"\d+pf", c)),
                  key=lambda k: int(k[:-2]))
    if against not in keys:
        raise SystemExit(f"{against} is not among the scored rivals {keys}")
    means = scores[keys].mean()
    winner = str(means.idxmax())
    ranking = " > ".join(f"{k} ({means[k]:.3f})" for k in means.sort_values(ascending=False).index)
    if winner == against:
        runner = means.drop(against).idxmax()
        d = scores[against] - scores[runner]
        return against, (f"KEEP {against}: {ranking}; over {runner} {d.mean():+.3f} z, "
                         f"t {paired_t(d.to_numpy()):.1f}, {int((d > 0).sum())}/{len(d)} sites"
                         ), None
    d = (scores[winner] - scores[against]).to_numpy()
    stats = (f"{winner} - {against} {d.mean():+.3f} z, t {paired_t(d):.1f}, "
             f"{int((d > 0).sum())}/{len(d)} sites")
    steps = np.diff(means[keys].to_numpy())
    if np.all(steps > 0) or np.all(steps < 0):
        return against, (f"KEEP {against} (WARNING): {winner} leads ({ranking}; {stats}) "
                         f"but the scores are monotonic in N -- template mass, not structure"
                         ), None
    if paired_t(d) < min_t:
        return against, (f"KEEP {against} (WARNING): {winner} leads ({ranking}) but not "
                         f"decisively: {stats}, below t {min_t:g}"), winner
    return winner, f"SWITCH {against} -> {winner}: {ranking}; {stats}", None


def decide(scores: pd.DataFrame, ring_key: str, min_t: float
           ) -> tuple[dict[str, list[int] | None], list[str], dict[str, list[int]]]:
    """{N key: its filaments (None = all, no per-filament column)}, the report lines, and
    {N key: the filaments that lead to it but not decisively} -- to probe."""
    target, line, _ = verdict(scores, ring_key, min_t)
    lines = [f"PF number, all filaments, against the ring's {ring_key}: {line}"]
    if "region" not in scores.columns:
        return {target: None}, lines, {}
    groups: dict[str, list[int]] = {}
    probes: dict[str, list[int]] = {}
    for label in sorted(scores["region"].unique()):
        mine = scores[scores["region"] == label]
        if len(mine) < MIN_SITES:
            own, why, lead = target, f"{len(mine)} sites, too few for its own verdict: {target}", None
        else:
            own, why, lead = verdict(mine, target, min_t)
        lines.append(f"PF number, filament {int(label)}, against {target}: {why}")
        groups.setdefault(own, []).append(int(label))
        if lead is not None:
            probes.setdefault(lead, []).append(int(label))
    return groups, lines, probes


def build_model(stem: str) -> None:
    """Build ``models/<stem>.pdb`` with the generator every template here came from."""
    match = STEM_RE.match(stem)
    if match is None:
        raise SystemExit(f"models/{stem}.pdb is missing and is not a stem the builders make")
    lattice, n, keep, rings, shift = match.groups()
    if keep:
        cmd = ["models/build_patch_template.py", "--protofilaments", n, "--keep", keep]
    else:
        cmd = ["models/build_mt_templates.py", "--protofilaments", n]
    cmd += ["--rings", rings]
    if lattice:
        cmd += ["--lattice", lattice]
    if shift:
        cmd += ["--monomer-shift", shift]
    print(f"  building models/{stem}.pdb: {' '.join(cmd)}")
    subprocess.run([sys.executable, *cmd], cwd=MT_ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    if not (MT_ROOT / "models" / f"{stem}.pdb").is_file():
        raise SystemExit(f"the builder did not write models/{stem}.pdb")


def simulate_missing(stems: list[str], ring_stem: str, ring_map: pathlib.Path,
                     pixel_size: float, voltage: float, device: str) -> None:
    """Simulate the maps of ``stems`` that do not exist, with settings that reproduce the
    current ring's map."""
    # pylint: disable=import-outside-toplevel
    import simulate_lattice_competition as slc
    import torch
    from ttsim3d.models import Simulator

    missing = [s for s in stems if not map_for(s, pixel_size).is_file()]
    if not missing:
        return
    kv = int(round(voltage))
    if kv not in MTF_FOR_KV:
        raise SystemExit(f"no detector model known for {voltage} kV; simulate "
                         f"{', '.join(missing)} by hand (simulate_lattice_competition.py)")
    with mrcfile.open(ring_map, permissive=True) as handle:
        existing = np.asarray(handle.data, dtype=np.float32)
    box = existing.shape[0]
    config = slc.sim_config(voltage, MTF_FOR_KV[kv], EXPOSURE)
    simulator = Simulator(pdb_filepath=str(MT_ROOT / "models" / f"{ring_stem}.pdb"),
                          pixel_spacing=pixel_size, volume_shape=existing.shape,
                          center_atoms=False, remove_hydrogens=True,
                          b_factor_scaling=slc.B_FACTOR_SCALING, additional_b_factor=0,
                          simulator_config=config)

    def on_gpu_or_cpu(run):
        """``run(device)``, on the CPU if the GPU is short of memory: a 600^3 map takes
        ~33 GB on the GPU (upsampled), more than is free beside another job's refine."""
        nonlocal device
        try:
            return run(device)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  {device} out of memory; simulating on the CPU (slower, same maps: "
                  f"simulate_lattice_competition.py's GPU and CPU agree to cc 1.000000)")
            device = "cpu"
            return run(device)

    again = on_gpu_or_cpu(lambda dev: simulator.run(device=dev).cpu().numpy())
    a, b = again.ravel() - again.mean(), existing.ravel() - existing.mean()
    cc = float(a @ b / np.sqrt((a @ a) * (b @ b)))
    print(f"  settings check: {ring_stem} re-simulated at {pixel_size} A/px, {box}^3, "
          f"{kv} kV, {MTF_FOR_KV[kv]}, {EXPOSURE:g} e/A^2: cc {cc:.6f} with "
          f"{ring_map.name}")
    if cc < 0.999:
        raise SystemExit(f"these settings do not reproduce the ring's map (cc {cc:.4f}); "
                         f"simulate {', '.join(missing)} by hand with the ones it was made with")
    for stem in missing:
        path = on_gpu_or_cpu(
            lambda dev, s=stem: slc.simulate(s, dev, pixel_size, (box,) * 3, config))
        print(f"  simulated {path.relative_to(MT_ROOT)}")
    # Hand the GPU back before anything else runs on it. PyTorch keeps what the
    # simulations used (~28 GB at 484^3) reserved in this process, and the probe started
    # next on the same GPU then ran out of memory (2026-11-18).
    del simulator
    gc.collect()
    torch.cuda.empty_cache()


def switched_config(source: pathlib.Path, target: pathlib.Path, template: pathlib.Path,
                    old_n: int, new_n: int, note: str) -> None:
    """``source`` with the template (and the N in its result paths) changed, at ``target``;
    an existing ``target`` is reused only if it is the same search with that template."""
    want = str(template.relative_to(MT_ROOT))
    if target.is_file():
        have, base = yaml.safe_load(target.read_text()), yaml.safe_load(source.read_text())
        same = all(have.get(k) == base.get(k) for k in SETTINGS)
        if same and have.get("template_volume_path") == want:
            print(f"  reusing {target.relative_to(MT_ROOT)} (same search, {template.name})")
            return
        raise SystemExit(f"{target.relative_to(MT_ROOT)} exists but is not {source.name} with "
                         f"{template.name}; move it or pick another tag")
    lines = [f"# {note}\n"]
    for line in source.read_text().splitlines(keepends=True):
        if line.startswith("template_volume_path:"):
            line = f"template_volume_path: {want}\n"
        elif re.match(r"\s+\w+_path:", line):
            line = line.replace(f"_{old_n}pf", f"_{new_n}pf")
        lines.append(line)
    target.write_text("".join(lines))
    print(f"  wrote {target.relative_to(MT_ROOT)}")


def sub_constraint(constraint: pathlib.Path, labels: list[int], pixel_size: float) -> str:
    """The parent constraint over only ``labels`` (its region ids = path order, from 1):
    ``<constraint>_fil<labels>``, rasterised as the parent was."""
    parent = yaml.safe_load(constraint.with_name(constraint.name + ".yaml").read_text())
    drawn = json.loads(constraint.with_name(constraint.name + ".paths.json").read_text())
    if max(labels) > len(drawn["paths"]):
        raise SystemExit(f"filament {max(labels)} is not among the "
                         f"{len(drawn['paths'])} drawn paths of {constraint.name}")
    out = constraint.with_name(f"{constraint.name}_fil{'-'.join(map(str, labels))}")
    drawn["paths"] = [drawn["paths"][label - 1] for label in labels]
    out.with_name(out.name + ".paths.json").write_text(json.dumps(drawn, indent=2))
    cmd = [sys.executable, str(MAKE_CONSTRAINT), str(out.with_name(out.name + ".paths.json")),
           "-o", str(out.with_name(out.name + ".yaml")),
           "--cone-half-angle-deg", str(parent["cone_half_angle_deg"]),
           "--theta-center-deg", str(parent["theta_center_deg"]),
           "--psi-step", str(parent["psi_step"]), "--theta-step", str(parent["theta_step"]),
           "--pixel-size-angstrom", str(pixel_size)]
    subprocess.run(cmd, cwd=MT_ROOT, check=True, stdout=subprocess.DEVNULL)
    with h5py.File(out.with_name(out.name + ".h5"), "r") as handle:
        n_regions = len(handle["regions"])
    if n_regions != len(labels):
        raise SystemExit(f"{out.name}.h5 has {n_regions} regions, expected {len(labels)}")
    print(f"  wrote {out.relative_to(MT_ROOT)}.{{paths.json,yaml,h5}}: filament(s) "
          f"{', '.join(map(str, labels))} -> {', '.join(map(str, range(1, len(labels) + 1)))}")
    return str(out.relative_to(MT_ROOT))


# pylint: disable=too-many-locals,too-many-statements
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--results", type=pathlib.Path, required=True)
    parser.add_argument("--rivals", required=True, choices=sorted(RIVAL_SETS))
    parser.add_argument("--ring-config", type=pathlib.Path, required=True)
    parser.add_argument("--patch-config", type=pathlib.Path, required=True)
    parser.add_argument("--ring-stem", required=True)
    parser.add_argument("--patch-stem", required=True)
    parser.add_argument("--constraint", type=pathlib.Path, required=True,
                        help="the run's constraint base (<base>.yaml, .h5, .paths.json)")
    parser.add_argument("--min-t", type=float, default=3.0,
                        help="paired t a winner needs over the N it would replace")
    parser.add_argument("--switched-from", default="",
                        help="the tag this run was switched or split from: never again")
    parser.add_argument("--no-switch", action="store_true",
                        help="report the decision only (PF_SWITCH=0)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    results = args.results if args.results.is_absolute() else MT_ROOT / args.results
    constraint = args.constraint if args.constraint.is_absolute() else MT_ROOT / args.constraint
    env_path = results / f"pf_switch_{args.tag}.env"
    lattice, n_ring = parse_rival_set(args.rivals)
    ring_key = f"{n_ring}pf"
    scores = pd.read_csv(results / f"compete_svd_summary_{args.tag}.csv", index_col=0)
    groups, lines, probes = decide(scores, ring_key, args.min_t)
    for line in lines:
        print(line)

    ring_cfg = yaml.safe_load(args.ring_config.read_text())
    pixel_size = float(ring_cfg["optics_group"]["pixel_size"])
    voltage = float(ring_cfg["optics_group"]["voltage"])
    ring_map = MT_ROOT / ring_cfg["template_volume_path"]
    base_tag = re.sub(rf"_{n_ring}pf$", "", args.tag)
    everyone = sorted(label for members in groups.values() if members for label in members)
    prepared: dict[tuple, dict] = {}

    def prepare(key: str, members: list[int] | None, why: str) -> dict:
        """Templates, configs and (for some filaments only) a constraint for one run at
        ``key``'s N over ``members``: the whole run's tag and constraint if they are all."""
        whole = members is None or members == everyone
        cache = (key, None if whole else tuple(members))
        if cache in prepared:
            return prepared[cache]
        n_new = int(key[:-2])
        if n_new not in PF_RANGE:
            raise SystemExit(f"no rival set for N={n_new}")
        if f"_{n_ring}pf_" not in args.patch_stem:
            raise SystemExit(f"cannot tell the N of the patch stem {args.patch_stem}")
        rivals = f"{lattice}{n_new}"
        rival_set = RIVAL_SETS[rivals]
        ring = next(stem for _, stem, purpose in rival_set if "(reference)" in purpose)
        patch = args.patch_stem.replace(f"_{n_ring}pf_", f"_{n_new}pf_")
        tag = (f"{base_tag}_{n_new}pf" if whole else
               f"{base_tag}_fil{'-'.join(map(str, members))}_{n_new}pf")
        print(f"\n{args.tag} -> {tag}: {why}; rivals {rivals}, ring {ring}, patch {patch}")
        stems = list(dict.fromkeys([ring, patch, f"{patch}_shift1"]
                                   + [stem for _, stem, _ in rival_set]))
        for stem in stems:
            if not (MT_ROOT / "models" / f"{stem}.pdb").is_file():
                build_model(stem)
        simulate_missing(stems, args.ring_stem, ring_map, pixel_size, voltage, args.device)
        print(f"  {len(stems)} templates present at {pixel_size} A/px")
        run_constraint = (str(args.constraint) if whole else
                          sub_constraint(constraint, members, pixel_size))
        configs = {}
        for kind, source, stem in (("", args.ring_config, ring),
                                   ("_patch", args.patch_config, patch)):
            target = MT_ROOT / "configs" / f"match_tm_{tag}{kind}.yaml"
            note = (f"written by pf_switch.py from {source.name}: the {n_new}-PF template, "
                    f"for {args.tag}'s {why}")
            switched_config(source, target, map_for(stem, pixel_size), n_ring, n_new, note)
            configs[kind] = target.relative_to(MT_ROOT)
        prepared[cache] = {
            "TAG": tag, "RIVALS": rivals, "RING_STEM": ring, "PATCH_STEM": patch,
            "RING_CONFIG": configs[""], "PATCH_CONFIG": configs["_patch"],
            "CONSTRAINT": run_constraint,
            "FILAMENTS": "all" if whole else " ".join(map(str, members)),
            # the run's region id of each parent filament: its own if the whole run,
            # else 1, 2, ... in path order (sub_constraint)
            "_REGION": ({label: label for label in everyone} if whole else
                        {label: i for i, label in enumerate(members, start=1)})}
        return prepared[cache]

    def probe(key: str, members: list[int], target: str) -> list[int]:
        """Run ``members`` at ``key``'s N up to their own PF check; the filaments whose
        own sites put them there, beating ``target`` with t >= min_t."""
        run = prepare(key, members, f"probe of filament{'s' if len(members) > 1 else ''} "
                                    f"{', '.join(map(str, members))} at {key}")
        log = MT_ROOT / "logs" / f"{run['TAG']}_probe.log"
        env = dict(os.environ, RESULTS=str(args.results), RIVALS=run["RIVALS"],
                   RING_STEM=run["RING_STEM"], PATCH_STEM=run["PATCH_STEM"],
                   PF_SWITCH="0", PF_SWITCHED_FROM=args.tag, STOP_AFTER="pf_check")
        print(f"  probing: {run['TAG']} up to its PF check (log {log.relative_to(MT_ROOT)})",
              flush=True)
        start = time.time()
        with open(log, "w", encoding="utf-8") as handle:
            done = subprocess.run(
                ["bash", str(PIPELINE), run["TAG"], str(run["RING_CONFIG"]),
                 str(run["PATCH_CONFIG"]), run["CONSTRAINT"]],
                cwd=MT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if done.returncode != 0:
            raise SystemExit(f"the probe {run['TAG']} failed: see {log.relative_to(MT_ROOT)}")
        own = pd.read_csv(results / f"compete_svd_summary_{run['TAG']}.csv", index_col=0)
        confirmed = []
        for label in members:
            mine = own[own["region"] == run["_REGION"][label]]
            if len(mine) < MIN_SITES:
                result, why = target, f"{len(mine)} sites, too few: stays at {target}"
            else:
                result, why, _ = verdict(mine, target, args.min_t)
            line = (f"PF number, filament {label}, probed at its own {key} sites: {why}"
                    + ("" if result == key else f" -> stays at {target}"))
            print(line)
            lines.append(line)
            if result == key:
                confirmed.append(label)
        print(f"  probe took {(time.time() - start) / 60:.1f} min")
        return confirmed

    if probes and not (args.no_switch or args.switched_from):
        for key, members in sorted(probes.items(), key=lambda item: int(item[0][:-2])):
            target = next(k for k, m in groups.items() if m and members[0] in m)
            for label in probe(key, members, target):
                groups[target].remove(label)
                groups.setdefault(key, []).append(label)
            groups = {k: sorted(m) for k, m in groups.items() if m}
    elif probes:
        print("PF number: leads below t not probed ("
              + ("PF_SWITCH=0" if args.no_switch else f"switched from {args.switched_from}")
              + ")")

    def describe(key: str) -> str:
        members = groups[key]
        return (f"{key}" if members is None or len(groups) == 1 else
                f"filament{'s' if len(members) > 1 else ''} "
                f"{', '.join(map(str, members))} -> {key}")

    if set(groups) == {ring_key}:
        decision = f"KEEP {ring_key}"
    elif len(groups) == 1:
        decision = f"SWITCH {ring_key} -> {next(iter(groups))}"
    else:
        decision = "SPLIT " + "; ".join(describe(k) for k in groups)
    print(f"PF number: {decision}")

    def write_env(**values) -> None:
        env_path.write_text("".join(f"{k}={shlex.quote(str(v))}\n" for k, v in values.items()))

    reason = " | ".join([decision] + lines)
    if set(groups) == {ring_key} or args.no_switch:
        if set(groups) != {ring_key}:
            print(f"PF number: not acting on it (PF_SWITCH=0); later stages ask at N={n_ring}")
        write_env(SWITCH=0, REASON=reason)
        return
    if args.switched_from:
        write_env(SWITCH=0, REASON=reason)
        raise SystemExit(f"PF number: CONFLICT -- this run was made from {args.switched_from} "
                         f"at N={n_ring}, and its own competition now says {decision}. Not "
                         f"switching again; check the PF number by hand (REFINE_STAGE6=1).")

    # the ring's own N first (if any filament stays there), then the others by N
    runs = [prepare(key, groups[key], describe(key))
            for key in sorted(groups, key=lambda k: (k != ring_key, int(k[:-2])))]
    runs = [{k: v for k, v in run.items() if not k.startswith("_")} for run in runs]
    if len(runs) == 1:
        write_env(SWITCH=1, REASON=reason,
                  **{f"NEW_{k}": v for k, v in runs[0].items()
                     if k not in ("CONSTRAINT", "FILAMENTS")})
    else:
        write_env(SWITCH="split", REASON=reason, N_CHILDREN=len(runs),
                  **{f"CHILD_{i}_{k}": v for i, run in enumerate(runs, start=1)
                     for k, v in run.items()})
    print(f"wrote {env_path}")

if __name__ == "__main__":
    main()
