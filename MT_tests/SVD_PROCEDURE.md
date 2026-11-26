# The four microtubule parameters — current best procedure, with SVD

Polarity, protofilament (PF) number, lattice spacing and seam/register, from one
micrograph. This is the working procedure as of **2026-11-12**: the expensive searches
run on Panther-EM's SVD search, every number that is reported comes from Leopard-EM's
`refine_template`. `PIPELINE.md` holds the reasoning behind each stage; this file says what
to run now and what each parameter rests on.

## The rule

> **SVD finds, `refine_template` measures.**

* The SVD search replaces the two full `match_template` runs (ring and patch): on in situ
  823 the ring search took **~14 min on 1 GPU** against **5 h 39 min on 4 GPUs**, the patch
  **~16 min** against **5 h 40 min**.
* Its tables give the same polarity, the same PF call, 97–99% of Leopard's detections and
  the same sites. `refine_template` run on those sites gives the same rises as on
  Leopard's sites (within **0.03 Å** on every reliable filament), so the final numbers do
  not depend on which search found the sites.
* SVD correlation values were ~15% low and only r ≈ 0.8 with Leopard's on 823 for two
  reasons, both found and fixed in `svd_score_sites.py` on 2026-11-12 (see *Two SVD
  defects*): Panther-EM's polar grid rotates about a point half a pixel off the template's
  centre, and its CTF has no B-factor envelope. Fixed, the scorer agrees with Leopard at
  r 0.95 with z ratios 1.00–1.02. The Stage 2 search (`svd_match_filament.py`) has both
  fixes by default since the same day (`--polar-center fft`, `--b-factor` from the
  config): on 823 it then keeps **2.1×** the table entries (34% more on the synthetic),
  and Stage 5 rises are unchanged within 0.01–0.03 Å. Tables made before that lack the
  fixes; their sites are still fine, since `refine_template` re-searches the position.

Template rules from `PIPELINE.md` still hold: **ring for everything geometric, patch only
for the seam.** The SVD search needs a template whose mass is centred in its box (it
refuses otherwise). GMPCPP's `GMPCPP_2rings_0.9194` is 94 px off-centre along its axis;
`run_scripts/simulate_gmpcpp_ring_centred.py` re-simulates the same atoms moved 94 voxels
(86.424 Å) along the axis → `maps/GMPCPP_2rings_centred_0.9194_bscale0.5.mrc` (centre of
density 0.16 / −0.29 / 0.32 voxels; the re-simulated original rolled 94 voxels matches it
at cc 1.000000) and `models/GMPCPP_2rings_centred.pdb` (14 PF, 3-start, rise 41.979 Å —
the deposited 6DPU). Simulated with today's ttsim3d and the notebook's call, like every
rival map at 0.9194 Å and `GMPCPP_4patches_centred`: same structure as the June 2025 map
shell by shell (FSC ≥ 0.9992) but 4–8 Å weighted ~0.6× relative to low resolution
(ttsim3d changed since; the old map is also mean-subtracted).

**Use it for searching, not as the pipeline's ring.** Its 56 chains are a different slice
of the lattice from the built templates' (same polarity and hand — β→α 41.3 Å, α→β 42.6 Å
up the axis in both, −9.0 Å per protofilament — but ~19 Å out of axial register with the
built 14-PF ring, plus one non-helical step, its seam). Later stages score the built
templates (PF rivals, both lattices, the patch) at the ring's poses, so with it as the ring
(run `pipe_gmpcpp_centred`, 2026-11-13) every built rival scored z ~5 against the ring's
own 10.4 and Stage 4's sites deviated 8.7 Å rms (823: 4–5 Å); Stage 6b then found no
continuous lattice. The pipeline's ring must be the rival set's reference, as on 823:
for GMPCPP `6dpu_14pf_2rings_flatB` (6DPU atoms and lattice, already centred).

## Environments

```bash
# Panther-EM (SVD search, site scoring): branch jd/emit-entries on
# github.com/jdickerson95/Panther-EM (emit_entries on mdg-seed-local-stage1;
# PR mgiammar/Panther-EM#25).
# PANTHER_ROOT defaults to a Panther-EM checkout next to this Leopard-EM one; the
# other overrides are in run_scripts/panther_env.sh.
source run_scripts/panther_env.sh
# Leopard-EM (tables, Stages 3-7, refine_template): a Python with this branch installed
PY=${LEOPARD_PY:-$HOME/miniconda3/envs/leopard-em/bin/python}
```

Run everything from `MT_tests/`. Examples below use set1_823 (tag `in_situ_823_13pf`, SVD
outputs in `results_svd/` with the prefix `svd_`). Later stages read the search config as
`configs/match_tm_<tag>.yaml`, so the SVD tag needs its own copy
(`configs/match_tm_svd_in_situ_823_13pf.yaml`, same contents as the search's).

## One command: `run_scripts/run_svd_pipeline.sh`

```bash
AVOID_PATHS=results_autotrace_svd/in_situ_823.paths.json \
run_scripts/run_svd_pipeline.sh pipe_in_situ_823_13pf configs/match_tm_in_situ_823_13pf.yaml \
    configs/match_tm_in_situ_823_13pf_patch.yaml configs/filament_constraint_in_situ_823
```

Ring search → Stages 3, 4, 5 → PF competition on the SVD scorer → Stage 6b (refine, neutral
poses) → constraint re-fit → patch search → patch Stages 3, 4 → Stage 7 (found sites) →
Stage 6b on the patch at every site → every-site register → register per protofilament and
the seam (`stage7_seam.py`). Each step is timed
(`<results>/timings_<tag>.tsv`), logged (`logs/<tag>_<step>.log`) and skipped when re-run
once done; a summary prints at the end. Options in the script header (GPUs, rivals, patch
stem, ring stem, refit width, `KEEP_WORK`, `REFINE_STAGE6`, `MIN_FREE_GB`). Outputs go to
`results_svd_pipeline/` under the tag, with `configs/match_tm_<tag>[_patch].yaml` copied from
the given configs.

GMPCPP (in vitro, the expanded lattice; the ring is its rival set's reference). Frame 0005
is 13-PF: started from the 14-PF ring, the run switches itself to 13 after Stage 6 (see
**PF number switch**); started at 13 it goes straight through:

```bash
RIVALS=expanded14 PATCH_STEM=6dpu_14pf_patch5_2rings_flatB RING_STEM=6dpu_14pf_2rings_flatB \
run_scripts/run_svd_pipeline.sh pipe_gmpcpp_14pf configs/match_tm_gmpcpp_14pf.yaml \
    configs/match_tm_gmpcpp_14pf_patch.yaml configs/filament_constraint_gmpcpp
# -> "PF number: SWITCH 14pf -> 13pf ..." and it carries on as pipe_gmpcpp_13pf; or directly:
RIVALS=expanded13 PATCH_STEM=6dpu_13pf_patch5_2rings_flatB RING_STEM=6dpu_13pf_2rings_flatB \
run_scripts/run_svd_pipeline.sh pipe_gmpcpp_13pf configs/match_tm_gmpcpp_13pf.yaml \
    configs/match_tm_gmpcpp_13pf_patch.yaml configs/filament_constraint_gmpcpp
```

### PF number switch

Everything after Stage 6 is asked at the RING's N: the lattice pair (6b), the patch and its
one-monomer-shifted partner (Stage 7), and the poses all of them are scored at. So right
after the PF competition, `run_scripts/pf_switch.py` (step `pf_check`) settles each
filament's N from the competition's per-site scores, with one rule applied twice -- another
N replaces N only if it beats N with paired t ≥ `PF_SWITCH_T` (3) and the scores are not
monotonic in N (that is template mass): first all filaments pooled against the ring's N,
then each filament (≥ 8 sites) against that. A filament where another N LEADS, not
monotonically, but below t 3 is then **probed**: its filaments (grouped by the N they lead
to) run at that N over their own constraint up to their own `pf_check` (`PF_SWITCH=0`,
`STOP_AFTER=pf_check`, log `logs/<probe tag>_probe.log`), and move there if at those sites
that N beats theirs with t ≥ 3. The probe is needed because the sites come from the
ring's search, and a wrong-N ring's sites hold the right N back (the mixed synthetic
below: 14 led 13 by t 0.2 and 1.7 at the 13-PF ring's sites, and won by t 23–27 at the
14-PF ring's own). A probe has the tag the run at that N would have, so a confirmed probe
is reused, not repeated; a rejected one costs a ring search and Stages 3–6 (~10 min for
two tubes). No probes with `PF_SWITCH=0` or in a switched/split run. Then:

* every filament at the ring's N → carries on (a non-decisive leader is probed first);
* every filament at one other N → stops this tag and restarts the whole run (`exec`, so
  one command) at that N: the same lattice's rival set for it, its ring and patch,
  `configs/match_tm_<new tag>[_patch].yaml` (the given configs with only the template and
  the N in the result paths changed), tag `_<N>pf` replaced;
* filaments at different N → stops this tag and runs once per N over only its filaments:
  a constraint of their drawn paths, `configs/<constraint>_fil<labels>.{paths.json,yaml,h5}`,
  rasterised with the parent's cone and steps (its filaments renumbered 1, 2, … in path
  order), tag `<base>_fil<labels>_<N>pf`;
* a run that was itself switched or split (`PF_SWITCHED_FROM`) and whose own competition
  disagrees → stops: no back and forth. `PF_SWITCH=0` reports the decision and carries on.

`STOP_AFTER=<step>` stops a run (and the runs it switches or splits into) after that step,
e.g. `pf_check` for polarity, PF number and rise only.

Missing templates of the new N are made, not assumed: models with the builders
(`models/build_mt_templates.py`, `build_patch_template.py --monomer-shift 1`; deterministic —
a rebuilt 13-PF patch is byte-identical to the stored one), maps with
`simulate_lattice_competition` at the search's pixel size, box and voltage, after the
CURRENT ring re-simulated with those settings reproduces its own map (cc ≥ 0.999; checked on
GDP's compacted 13-PF ring: cc 1.000000) — on the CPU if the GPU is short of memory (a 600³
map takes ~33 GB on the GPU). `compete_sites_curved.RIVAL_SETS` now has a set for every
lattice and N, `compacted12`–`15` and `expanded12`–`15` (the three hand-written ones,
compacted13, expanded14 and expanded13, reproduced exactly): the lattice's rings at 12–15
PF, the other lattice's ring at N, both lattices' 5-PF patches at N.

Tested on GMPCPP: the `pipe_gmpcpp_14pf` command above decided `SWITCH 14pf -> 13pf` (t 9.0,
71/81 sites), restarted as `pipe_gmpcpp_13pf`, whose own competition kept 13 (`KEEP 13pf`,
t 13.4 over 12); on a made-up compacted 13 → 14 case, which built and simulated the one
missing template (`6dpu_atoms_6dpv_lattice_14pf_patch5_2rings_flatB_shift1`, now in
`models/` and `maps/` at 0.9194 Å); on 823 (every filament keeps 13 on its own, t 11–19);
and on 823's scores with filaments 3 and 4 swapped to read 14-PF: `SPLIT filaments 1, 2 ->
13pf; filaments 3, 4 -> 14pf` (pooled 14 by t 3.0, then filaments 1–2 back to 13 at t 16
and 24).

End to end on a synthetic with tubes of both N (`simulate_microtubule_bundle.py
--protofilaments 13 13 14 14 --rolls 11 47 113 200 --size 4000 --target-z 14.5 --tag
mixed_pf`, configs `match_tm_bundle_mixed_pf_13pf_{ring,patch}.yaml`, constraint
`filament_constraint_bundle_mixed_pf`; tag `pipe_bundle_mixed_pf_13pf`, `STOP_AFTER=pf_check`;
2026-11-13): **passes, with the probe** -- without it, it did not split.

| tube (truth) | 13-PF ring's sites: rise, dev rms | 14 vs 13 there | → | own run's sites: rise, dev rms | own N vs the other there |
|---|---|---|---|---|---|
| 1 (13) | 40.992 Å, 3.95 Å | 13 by t 33.5 | `fil1-2_13pf` | 40.992 Å, 3.95 Å | KEEP 13, t 33.5 |
| 2 (13) | 40.954 Å, 2.39 Å | 13 by t 28.0 | `fil1-2_13pf` | 40.954 Å, 2.39 Å | KEEP 13, t 28.0 |
| 3 (14) | 38.29 Å (−6.2%), 11.5 Å, UNRELIABLE | 14 by t **0.2** | probe → `fil3-4_14pf` | 40.961 Å, 2.05 Å | 14 by t 23.3 |
| 4 (14) | 44.00 Å (+7.8%), 11.9 Å, UNRELIABLE | 14 by t **1.7** | probe → `fil3-4_14pf` | 40.983 Å, 2.02 Å | 14 by t 33.1 |

(truth: monomer 40.943 Å for both N.) At the 13-PF ring's sites the 14-PF tubes only
LEANED to 14, so the rule as first written kept 13 for every tube (with warnings). Not the
competition's window: re-scored with ±23 px along the tube instead of ±8 (half a monomer;
`stage6_svd.py --along-px 23`, kept as `compete_svd_*_pipe_bundle_mixed_pf_13pf_along23.csv`)
14 still led by only t 0.6 / 1.8; and no other edge binds more there than on the 13-PF
tubes (14-PF ring at its roll edge at 16–17% of sites against 17–19%; theta, psi, defocus,
across ≤ 9%). Not the synthetic: its 14-PF tubes are tiled exactly as the
13-PF ones (parallel PFs, steps of four 40.943 Å monomers). The sites: the wrong ring put
them off the 14-PF lattice, and the 14-PF ring at those sites could not recover. Hence the
probe: the parent's `pf_check` ran tubes 3–4 at 14 PF up to their own check (9.5 min:
ring search 4.7, Stages 3–6 4.8), which confirmed 14; `SPLIT filaments 1, 2 -> 13pf;
filaments 3, 4 -> 14pf`; `fil1-2_13pf` ran (9 min) and kept 13; `fil3-4_14pf` reused the
probe, every step skipped. Logs `logs/pipe_bundle_mixed_pf_pipeline_probe.log`,
`logs/pipe_bundle_mixed_pf_fil3-4_14pf_probe.log`. (A manual 14-PF run of tubes 3–4 before
the probe existed gave the same numbers; its outputs are kept with a `manual_` prefix and
`.done_…_manual`.)

Its constraint came from the SVD autotrace with the centred deposited ring
(`results_autotrace_svd_fix/gmpcpp_centred*`: one track, 100% of the hand-checked axis
covered, median 2 Å off it and 0.4°, against 14 Å with the old template auto-centred)
→ `curate_paths.py --keep 1 --extend-to-edge` → `make_filament_constraint.py` (cone 20°,
psi 1.5°, theta 2.5°). Optics from Stage 0 (PICASSO) with `ctf_B_factor` 60.

**GMPCPP end to end (2026-11-13).** First with the 14-PF ring (`pipe_gmpcpp_14pf`, 52.8
min, from Stage 5 on one GPU — the user's PICASSO jobs held the other three), whose PF
competition said 13; then with the 13-PF ring and the `expanded13` set (`pipe_gmpcpp_13pf`,
23.2 min on 4 GPUs). Against the earlier Leopard-EM analysis of this frame (STATUS.md):

| | 14-PF ring | **13-PF ring** | earlier (Leopard-EM) |
|---|---|---|---|
| polarity | ψ 279.2°, 65.9 : 1 | ψ 279.2°, 74.0 : 1 | ψ 279.4°, 2633 : 1 |
| rise (Stage 5, refined) | 41.938 ± 0.072 Å, dev 5.0 Å rms (80 sites) | **41.863 ± 0.005 Å** (block SE 0.056; −0.28% vs template), dev **1.2 Å** (81 sites) | 41.86 Å (straight axis) |
| PF number (Stage 6, SVD) | 13 > 14 > 12 > 15; 13 − 14 +1.04 z, t 9.0, 71/81 | **13** > 12 > 14 > 15; 13 − 12 +1.31 z, t 13.4, 76/81; 13 − 14 +2.19 z, t 25.8, 81/81 | 13 at 7.6σ (crop competition) |
| lattice, ring (6b) | EE (2 segments) | EEEEE?E, competition EEEEEEE (7 segments) | expanded, t = 10 |
| lattice, patch at every site (6b) | EE | EEEE | |
| register, every site (Stage 7) | none: t 0.37 (1,134 sites) | **t 3.11** (1,053 sites; decoys 0.97, two-dimer control 0.40; found 2.47, never-found 4.13) | not located |
| outputs | 8.3 GB (patch Stage 3 CSV 4.6 GB) | 4.2 GB | |

At the right N everything tightens: the sites lie 1.2 Å rms from the fitted lattice instead
of 5.0, the rise matches the Leopard-EM value, 6b resolves 7 segments instead of 2 — all
expanded — and a register appears. At t 3.1 it is as strong as 823 filament 2's (t 3.5–3.6,
"provisional") and above its decoys, so: present but weak, provisional.

A longer patch does not strengthen it. `6dpu_13pf_patch5_4rings_flatB` (+ `_shift1`, maps
at 0.9194 Å) scored at the same every-site lattice (its map coincides with the 2-ring
patch's at zero axial shift, cc 0.71 ≈ the 1/√2 the half-length allows, so the sites need no
offset): `stage7_every_site.py --table results_svd_pipeline/output_correlation_table_pipe_gmpcpp_13pf_patch.h5
--stem 6dpu_13pf_patch5_4rings_flatB --label pipe_gmpcpp_13pf_patch_4rings` (`--label` keeps
its refined scores apart: they are reused by name; 3.7 min on 4 GPUs).

| every site (1,053) | 2-ring patch | 4-ring patch |
|---|---|---|
| found sites, median refined z | 11.44 | 16.42 (×1.44 ≈ √2) |
| register z | 0.143 ± 0.046 | 0.198 ± 0.063 |
| **t** | **3.11** | **3.13** |
| found (978) / never found (75) | 2.47 / 4.13 | 2.39 / 4.88 |
| decoys (must be null) / two-dimer control | 0.97 / 0.40 | 0.46 / 0.07 |
| in − out, even / odd axial index | −0.019 / +0.267 | −0.106 / +0.290 |

The same register (odd indices favour in-register in both), and per site the two patches'
in − out agree at r = 0.975: the 4-ring adds signal and scatter in proportion, so the scatter
is not noise a longer template averages away; it is the sites' own variation. (On GDP the
4-ring lifted t 1.5 → 2.2, STATUS.md.) The GMPCPP register stays at t 3.1: present,
consistent across patch lengths, weak, provisional.

Two fixes it needed: (1) Stages 4/5/6b stepped z cuts down from 8.0 to the weakest site
with `np.arange(8.0, floor)`, which is EMPTY when every refined site is above 8 (here all
> 8.4) — "too few sites to re-measure" / "no continuous lattice"; now at least the top cut
is tried (`stage4_sites.py`, `stage5_refine.py`, `stage6b_local_lattice.py`; unchanged
whenever any site is below it). (2) `stage6_svd.py` dropped `LATTICE_PAIRS[r][1]`, which for
expanded14 is the 14-PF reference itself; it now drops the non-`<N>pf` member. Do not edit
`run_svd_pipeline.sh` while it runs: bash reads it as it goes.

### Disk

One micrograph's outputs come to ~5 GB (823: 4.6 GB, of which the two `*_svd.h5` 1.8 GB
and the patch Stage 3 membership CSV 1.1 GB). The disk filled on 2026-11-12 from the SVD
tools' work folders (whitening + decomposition + calibration, ~2 GB per template at the
default rank, 4–7 GB at k 256 or nside 64), which `svd_match_filament.py` kept and failed
runs left behind. Since 2026-11-13:

* `svd_match_filament.py`, `svd_score_sites.py` and `svd_trace_maps.py` delete their work
  folder when they end — also on an error or SIGTERM — unless `--keep-work`, and refuse to
  start when the run would leave less than `SVD_MIN_FREE_GB` (default 30) free.
* `run_svd_pipeline.sh` checks the free space before every step and stops (resumably)
  below `MIN_FREE_GB` (default 50); `timings_<tag>.tsv` records the free GB after each step.
* `python run_scripts/svd_work_dirs.py` lists leftover `*_svd_work` folders under MT_tests
  (largest first; skips any touched in the last hour) and `--delete` removes them. On
  2026-11-13 it found 39 from earlier runs, 44.8 GB, all regenerable; deleted that day at
  the user's word (with the abandoned `pipe_gmpcpp_centred` run, 1.5 GB): 263 GB free.

**823 end to end (2026-11-12, both SVD fixes):** 93 min on 4 GPUs (+ the refine Stage 6
it also ran, 13 min), against ~11 h for the two Leopard-EM searches alone. Polarity and
13 PF on all four tubes; Stage 5 rises 41.194 / 40.515 / 40.743 / 40.754 Å; PF ranking
13 > 12 > 14 > 15 (SVD, same as refine); patch 6b at every site C?? / CCCC / CC / CCC;
register t 8.5 / 0.9 / 23.5 / 16.2 (found sites) → **13.8 / 3.5 / 40.1 / 18.6** (every
site). Where the time goes: searches 26 min, patch Stage 3 17 min (CPU: membership of the
fixed patch table's 6.5 M detections, 2.7× the unfixed table's), every-site register
25 min, Stage 7 9 min — the next things to make faster.

## The run, in order

| step | tool | env | writes | 823 cost |
|---|---|---|---|---|
| 0 CTF | PICASSO | — | optics | seconds |
| 1 filaments | `run_svd_autotrace.sh` (+ human check), or a hand drawing | both | `*.paths.json`, constraint | ~1 min + review |
| 2 ring search | `svd_match_filament.py` → `svd_entries_to_table.py` | Panther → Leopard | v2 correlation table + maps | ~14 min, 1 GPU |
| 3 read table | `stage3_read_table.py` | Leopard | membership; **polarity**; PF harmonic | seconds |
| 4 sites | `stage4_sites.py` | Leopard | site list; whole-tube rise | seconds |
| 5 refine | `stage5_refine.py` | Leopard | refined sites; **rise** | ~1 min |
| 6 rivals | `stage6_svd.py` (SVD scorer; refine: `compete_sites_curved.py`) | both | **PF number** | ~5 min, 4 GPUs (refine 13 min) |
| 6b segments | `stage6b_local_lattice.py` | Leopard | **lattice state + rise per segment** | minutes |
| 7a patch search | `refit_constraint_from_ring.py`, `svd_match_filament.py` → `svd_entries_to_table.py` | both | patch table | ~16 min, 1 GPU |
| 7b register | `stage3`/`stage4 --resolve-protofilaments`/`stage7_register.py` | Leopard | **seam / register** | ~5 min, 4 GPUs |
| 7c seam | `stage7_seam.py` (after the every-site register) | Leopard | **seam position; each site's own register** | seconds, CPU |

```bash
# 2. ring search (Panther env), then the table (Leopard env)
CUDA_VISIBLE_DEVICES=0 python ../programs/constrained_search/svd_match_filament.py \
    --config configs/match_tm_in_situ_823_13pf.yaml \
    --constraint configs/filament_constraint_in_situ_823.yaml \
    --out results_svd/in_situ_823_13pf
$PY ../programs/constrained_search/svd_entries_to_table.py \
    results_svd/in_situ_823_13pf_svd.h5 --tag svd_in_situ_823_13pf
T=results_svd/output_correlation_table_svd_in_situ_823_13pf.h5

# 3-6b on the ring (Leopard env)
$PY run_scripts/stage3_read_table.py --table $T \
    --constraint configs/filament_constraint_in_situ_823.h5 \
    --micrograph results_in_situ/set1_823_Aug09_10.05.20_DW.mrc
$PY run_scripts/stage4_sites.py --table $T
$PY run_scripts/stage5_refine.py --table $T
$PY run_scripts/compete_sites_curved.py --tag svd_in_situ_823_13pf --results results_svd
$PY run_scripts/stage6b_local_lattice.py --table $T

# 7a. patch: constraint re-fitted from the ring's refined sites, then the patch search
$PY run_scripts/refit_constraint_from_ring.py --tag svd_in_situ_823_13pf --results results_svd \
    --source configs/filament_constraint_in_situ_823.h5 \
    --paths configs/filament_constraint_in_situ_823.paths.json \
    --out filament_constraint_in_situ_823_svd_refit --restrict-polarity --width-px 490 \
    --micrograph results_in_situ/set1_823_Aug09_10.05.20_DW.mrc
# --width-px must be the band the ring search used (823: 490); psi/theta steps default to
# the in situ grid (1.5 / 2.5)
CUDA_VISIBLE_DEVICES=0 python ../programs/constrained_search/svd_match_filament.py \
    --config configs/match_tm_in_situ_823_13pf_patch.yaml \
    --constraint configs/filament_constraint_in_situ_823_svd_refit.yaml \
    --out results_svd/in_situ_823_13pf_patch
$PY ../programs/constrained_search/svd_entries_to_table.py \
    results_svd/in_situ_823_13pf_patch_svd.h5 --tag svd_in_situ_823_13pf_patch
P=results_svd/output_correlation_table_svd_in_situ_823_13pf_patch.h5
PATCH=6dpu_atoms_6dpv_lattice_13pf_patch5_2rings_flatB

# 7b. register (and optionally 6b on the patch's sites)
$PY run_scripts/stage3_read_table.py --table $P --stem $PATCH \
    --constraint configs/filament_constraint_in_situ_823_svd_refit.h5
$PY run_scripts/stage4_sites.py --table $P --stem $PATCH --resolve-protofilaments
$PY run_scripts/stage7_register.py --table $P
$PY run_scripts/stage6b_local_lattice.py --table $P --patch
```

The 823 SVD patch search that has been run used the Leopard-made
`filament_constraint_in_situ_823_refit` constraint; the `_svd_refit` name above is what a
fully SVD run would make. SVD search defaults that worked: rank 96×80, HEALPix nside 32,
256 psi, z cut 6.5, normalisation over **every** template (`--normalize all`, as
Leopard's constrained search does; `band` is the test option).

---

## 1. Polarity

**Procedure.** Stage 3 (`estimate_polarity`, ψ alone, so bending cannot bias it), read per
~100 nm stretch along the data-fitted curve and checked against that curve's tangent.
Membership comes from each detection's own axis point, not the drawn band.

**Second line.** The drawn/autotraced tangent; Stage 6b's neutral poses use one polarity
per tube (the side most found sites take), and report how many found sites sit in the
other.

**SVD.** Same polarity as Leopard on every test set: GDP, both known-domain synthetics
(truth), in situ 823. Nothing SVD-specific to watch.

**Then** restrict the patch constraint to the measured pole (`--restrict-polarity`) — but
check the pole per region first: a band holding two tubes of opposite polarity has equal
strength at both poles.

## 2. Protofilament number

**Procedure.** Two independent lines:

1. **Stage 3 harmonic** — `estimate_protofilament_number`, a pure azimuth harmonic on φ,
   axis- and refinement-free. Read the margin over the runner-up, not just the winner.
2. **Stage 6 ring competition** — 12/13/14/15 PF rings at the refined sites, paired per
   site. Monotonic in N means a template-mass artefact; a middle-peaked ranking means
   structure. **Run on the SVD scorer** (`stage6_svd.py --table <ring table> --gpus 0 1 2
   3`, since 2026-11-12): per site and rival, the maximum z over the rival's own roll
   period (±180/N), ±3° θ/ψ, defocus planes, and a window aligned with the tube, ±8 px
   along × ±20 px across — a ring of another N lines up with one wall of the tube, so
   refine moved 12-PF rivals a median 10.9 px *across* the tube (3.3 along); a ±3 px
   square under-scored the rivals and inflated 13 PF's margin. Validated against refine's
   Stage 6 on the same sites:

   | sites | refine ranking | SVD ranking | 13pf − rival t, refine / SVD |
   |---|---|---|---|
   | 823 (4 site sets) | 13 > 13_exp > 12 > 14 > 15 | identical | 26–35 / 16–39 |
   | GDP | 13 > 13_exp > 14 > 12 > 15 | 12 and 14 swapped (refine: 6.382 vs 6.384, a tie) | 17–22 / 15–20 |
   | matched / domain synthetics (truth 13) | 13 > 13_exp | 13 first by a wide margin | — |

   Per-rival mean z within ~0.1–0.6 of refine's, per-site r 0.76–0.87. ~5 min on 4 GPUs
   (one rival per GPU: ~70 s decomposition + ~185 s scoring) against refine's 13 min.
   The expanded ring is left out (`--with-lattice` adds it as a cross-check): the lattice
   is decided by refine at neutral poses (Stage 6b), where the sensitivity is needed.
   `compete_sites_curved.py` (refine, 6–8 roll seeds per site) remains available —
   `REFINE_STAGE6=1` in the pipeline script.

**SVD — use Stage 6 as the deciding line.** On SVD tables the Stage 3 harmonic's
runner-up is a **2-fold** component, and its margins are lower: on the strong synthetic
**1.1–2.1×** (vs 2) against Leopard's **2.6–6.9×** (vs 12), on 823 3.4–7.0× against
9–27×. The winner (13) is right everywhere tested. Where the detections sit in roll is
*not* biased: every table — Leopard, old SVD, fixed SVD — clusters at each synthetic
tube's true roll phase (to 0.2°). The 2-fold component is not rank, normalisation,
whitening band, nside, sub-maximal rows, angular bandwidth, the polar-grid centre or the
B-factor (unchanged after both fixes); still open. Stage 6 does not see it, because `refine_template` re-searches the roll: on 823
both routes rank **13pf > 13pf_exp > 12pf > 14pf > 15pf** (13pf mean z 9.35 Leopard /
9.56 SVD; 13pf − 13pf_exp t 17.4 / 19.0; 406 / 410 sites).

**Caveat (both routes).** Stage 6's scores favour the template that found the sites. Its
PF margins on data (t 11–23) are far larger than that bias, but a neutral-pose PF
competition is not available (each rival needs its own roll).

**Re-run condition.** If Stage 6 picks a different N by a wide margin, redo Stage 2 at that
N — a wrong-N ring finds the tube far more weakly. On SVD that redo costs ~14 min.

## 3. Lattice spacing

Three readouts, coarse to fine:

1. **Whole-tube rise, Stage 4** (continuous, along a curve fitted to the detections;
   sin θ applied once). Read a filament only with **≥ 20 sites and ≤ 7 Å rms
   deviation**; quote the **block** error (contiguous blocks of 25+ sites), not the fit's.
2. **Refined rise, Stage 5** (`refine_template` on the Stage 4 sites, re-measured along a
   curve fitted to the refined sites). This is the number to report.
3. **Compacted vs expanded**: once per tube in Stage 6 (13pf vs 13pf_exp ring, same atoms),
   and **per segment in Stage 6b** — local rise (the headline call) and a neutral-pose
   competition (second line), segment length computed per tube from its own scatter;
   read the running profile, not only the tiles. With `--patch` the same on the patch's
   protofilament-resolved sites (shorter windows: 8–11 sites per index).

**SVD.** Stage 5 on SVD sites reproduces Leopard's rises within 0.03 Å on every reliable
filament (823, GDP filament 2, the synthetic pure tubes), and 823 tube 4 becomes
*reliable* on the SVD path (40.760 Å; its Leopard ring fit was flagged at 7.9 Å). Stage 6b
on 823: tubes 2 and 3 CCC / CC on both; tube 1 C?? (Leopard) vs ??C? (SVD); tube 4 C
(Leopard, competition only) vs C?CC (SVD, 1.61 Å deviation). The matched known-domain
synthetic makes no wrong call on either route.

**Must have:** the curve-fit guard in `continuous_unwrap_curved.py` (`_overshoots`, added
2026-11-12). Without it, sparse refined sites let the cubic spline extrapolate and loop
(87,663 px of curve on a 3,400 px tube) and the lattice index jumps. Output is unchanged
wherever the guard does not fire.

**Pixel size.** Both readouts measure spacing against templates rendered at the search's
pixel size, so a 1% scale error is 0.41 Å — nearly the C/E half-gap. On an uncalibrated
dataset run `fit_pixel_size.py --patch` and, if it differs, Stage 6b with
`--pixel-size <fitted>`.

## 4. Seam / monomer register

**Procedure.** Its own patch search (5-PF patch at the winning N and lattice, constraint
re-fitted from the ring and restricted to the measured pole), Stage 3 membership from each
detection's own axis point, Stage 4 with `--resolve-protofilaments` (one site per (axial
index, protofilament)), then Stage 7: the in-register patch and the same patch one monomer
out (`_shift1`) refined at every site, and the register read as **half the even-minus-odd
difference of (in − out)** per filament. Check every filament against the two-dimer-period
control (`index // 2`), which must be null. Never score the patch at the ring's sites.

**SVD.** Same register as Leopard within noise on 823 (same estimator, `--report-only`
re-reads saved scores): t **8.9 / 0.7 / 25.2 / 15.8** (SVD sites) vs **8.9 / 0.4 / 24.9 /
13.1** (Leopard sites) on filaments 1–4; filament 2 reads no register either way.

**Expect this readout to be the weakest.** It rests on a ~10% α/β score difference and is
short of sites; that is what every-site scoring (below) addresses — and it is now the
recommended route: Stage 7 at **every** lattice site with `refine_template` lifts 823's
register t from 8.9 / 0.7 / 25.2 / 15.8 to **15.7 / 3.6 / 38.5 / 18.8** (SVD-found sites).

**Per protofilament: the seam (`stage7_seam.py`, 2026-11-17; a pipeline step after the
every-site register).** Stage 7 reads ONE parity per tube, and two things flip it around
the tube, so pooling every protofilament partly cancels the register:

* the labels: a site's axial index comes from the helical unwrap of its own roll, and a
  full turn adds N × the lateral rise = 3 monomers (13_3, 14_3) — odd — so a protofilament
  is indexed in one parity just above roll 0 and the other just below 360 (with the
  supertwist, sometimes partway along a tube);
* the seam, where lateral neighbours are α–β: the register really flips.

The step puts every site in one labelling (roll less the tube's roll phase, smoothed and
unwrapped along the tube; the index moved by 3 per turn), then fits
in − out = offset + A (−1)^k g_s(p) for every seam position s: g flips at the label cut and
at the seam, and is smeared over the patch's 5 protofilaments near the seam (a straddling
patch is part in, part out of register). Width 5 fits 32 real tubes best (RSS: 1 +594,
3 +92, 7 +52, 9 +280). Uncertainty from a block bootstrap (8 axial indices), each half of
the tube alone, and the decoys with the tube's own pattern (null: |t| ≤ 2.4 everywhere).
It writes `stage7_seam_<tag>.csv` (per filament), `stage7_seam_sites_<tag>.csv` (per site:
the register that is right THERE and whether its patch straddles the seam — what scoring
"at the right register" needs) and a figure.

| | pooled t (Stage 7, every site) | register t, per-site register | boot t | seam |
|---|---|---|---|---|
| 823 filaments 1–4 (pipeline) | 13.8 / **3.5** / 40.1 / 18.6 | 35.6 / **44.0** / 52.3 / 31.9 | 11.8 / 16.1 / 23.1 / 10.3 | 3\|4, 5\|6, 1\|2, 9\|10 |
| GMPCPP 13 PF / 14 PF (one tube) | 3.1 / 0.4 | 59.2 / 54.2 | 14.2 / 14.3 | 4\|5, 5\|6 |
| GDP 14-PF children, filaments 1–4 | 8.3 / **1.4** / 33.3 / 7.5 | 18.2 / **38.6** / 33.3 / 28.9 | 3.1 / 11.5 / 7.1 / 7.9 | |

* 823 filament 2's "no register" was the seam: its two arcs read opposite signs. 823
  analysed three ways (Leopard-found, SVD-found, pipeline) gives the same seams to within
  one protofilament.
* "register t" treats sites as independent, as Stage 7 does; neighbouring sites share
  pixels, and the block bootstrap gives about a third of it.
* Only the seam's place relative to the patch is identifiable (centring the patch window
  one protofilament off fits identically, shifted by one), so seam labels are protofilaments
  counted in roll from the tube's roll phase. A seam inside a stretch of roll with no sites
  (some rolls are never found) is placed only within that gap.
* Validation, 13-PF strong synthetic (known rolls, a bundle with a crossing tube): tubes 2
  and 4 are 5.5 protofilament steps apart in roll and 5 apart in fitted seam; tubes 1 and 3
  (3.7 apart) fit 2 apart, tube 3's runner-up 3 — both in the bundle. A clean single-tube
  synthetic with a known seam is still to do.

---

## Every lattice site, not only the found ones (prototype, 2026-11-12)

The table keeps a site only if one of its correlations clears z 6.5. Once Stage 4 has the
lattice, every site's pose is **predicted**, and each one can be scored directly —
found or not.

* `lattice_fill.py sites` — every (index, protofilament) in each filament's span at
  Stage 6b's neutral pose, plus **decoys** (the same poses at random background pixels
  ≥ half a box from any band, and from `--avoid-paths`, e.g. the autotrace's extra tubes).
* `svd_score_sites.py` (Panther env) — per site and template: z at the predicted pose, and
  the maximum z in a window (±3 px, ±3° in φ/θ/ψ, defocus planes within ±200 Å), with
  normalisation over every template. ~2.5 min per template for ~8,500 sites (3 defocus
  planes), ~6.5 min on 823 (9 planes); work files are deleted afterwards (`--keep-work`).
* `lattice_fill.py analyse` — the decoy null, then Stage 6b's own segment competition and
  Stage 7's register estimator on: the old route, SVD at found sites, at every site, the
  window maximum at every site, and every site above the decoys' 99th percentile.

```bash
$PY run_scripts/lattice_fill.py sites --table $P --patch \
    --constraint configs/filament_constraint_in_situ_823_refit.h5 \
    --avoid-paths results_autotrace_svd/in_situ_823.paths.json
CUDA_VISIBLE_DEVICES=0 python ../programs/constrained_search/svd_score_sites.py \
    --config configs/match_tm_svd_in_situ_823_13pf_patch.yaml \
    --sites results_svd/lattice_fill_sites_svd_in_situ_823_13pf_patch.csv \
    --template patch=maps/6dpu_atoms_6dpv_lattice_13pf_patch5_2rings_flatB_1.056_bscale0.5.mrc \
    --template patch_exp=maps/6dpu_13pf_patch5_2rings_flatB_1.056_bscale0.5.mrc \
    --out results_svd/lattice_fill_scores_svd_in_situ_823_13pf_patch.csv
# and --template shift1=..._shift1_1.056_bscale0.5.mrc to ..._patch_shift1.csv for the register
$PY run_scripts/lattice_fill.py analyse --table $P --patch --plot   # --truth <json> on a synthetic
```

**What it gave.**

| | strong known-domain synthetic (patch) | in situ 823 (patch, both anchor sets) |
|---|---|---|
| sites: found / total | 1,756 / 4,238 | 3,466–3,816 / ~5,300 |
| never-found sites, median z at the pose | 3.7 (decoys 0.0 ± 1.0) | 2.4–2.5 (decoys 0.0) |
| … above the decoy 99th percentile | 77% | ~half (74% with window max) |
| 6b competition | right-call rate up: adaptive 50% → 60% (pose) / 80% (max), **0% wrong** | **no change**: tube 1 still ????, tubes 2–4 C |
| Stage 7 register t, fil 1–4 | 13.0 / 11.3 / 4.1 / 3.2 → **22.8 / 28.3 / 21.4 / 6.4** (max) | 8.9 / 0.4–0.7 / 25 / 13–16 → **12.7–13.8 / 4.4–4.8 / 39 / 14–16** (max) |
| decoys / two-dimer control | null / null | null / null |

So the **window maximum** at every site is the variant to use for the register, and the
bias towards the template that found the sites disappears (offset +0.4–0.65 z → ~0). For
the lattice-state competition it helps on the synthetic but not on 823, where SVD's weaker
per-site contrast cancels the extra sites. 823 filament 2's register (t 4.4–4.8 where the
old route reads nothing) is the result to treat with most caution.

### Recommended: `refine_template` at every lattice site (run 2026-11-12)

SVD (or any search) **finds** the lattice; `refine_template` **scores** every predicted
site, found or not, so the extra sites come with Leopard's per-site fidelity:

```bash
# Stage 6b competition at every (index, protofilament), neutral pose, no search
$PY run_scripts/stage6b_local_lattice.py --table $P --patch --start lattice
# Stage 7 register at every site (and the decoys as a null), Stage 7's local search
$PY run_scripts/lattice_fill.py sites --table $P --patch --constraint <constraint .h5>
$PY run_scripts/stage7_every_site.py --table $P
```

Outputs carry `_lattice` / `_every` / `_decoy` and never overwrite the found-site run.
~2.5 min per 6b run; Stage 7 ~5.6 min per template on 823 with 2 GPUs.

| | strong synthetic | 823, SVD-found | 823, Leopard-found |
|---|---|---|---|
| **register t**, old route (found) | 13.0 / 11.3 / 4.1 / 3.2 | 8.9 / 0.7 / 25.2 / 15.8 | 8.9 / 0.4 / 24.9 / 13.1 |
| **register t, refine every site** | **24.1 / 27.0 / 19.5 / 7.3** | **15.7 / 3.6 / 38.5 / 18.8** | **15.0 / 4.0 / 40.8 / 16.5** |
| register t, decoys | ≤ 1.8 | ≤ 1.3 | ≤ 0.8 |
| 6b competition, % right by window 8/16/24/32/adaptive | 16/29/36/22/50 → **22/37/53/73/60**, 0% wrong | calls C?? / CCCC / CC / CCC, no E | ???? / CCC / CC / CCC (= old) |

* The register's found/never-found split shows why it works: the template that found the
  sites wins by +0.08 to +0.40 z at them and loses by 0.02–0.48 at the never-found ones —
  selection bias — and the two cancel when every site is scored (offset ≈ 0). The signed
  register agrees between found and never-found sites on every filament.
* 823 filament 2 is the one change of verdict: no register on the old route; never-found
  sites read t 6.0 with the same sign as the found ones, all sites t 3.6. Treat as
  provisional.
* About the top 10% of never-found sites drift to the edge of refine's ±20 px window
  (probably empty sites); excluding them leaves t unchanged.
* On the 823 SVD table, sites outside the constrained search's band have no statistic
  maps (mean 0, variance 1e10), so `stage7_every_site.py` scores those with the raw cc,
  which is in z units inside the band. Lattice sites are unaffected; decoys are.
* The ring route gains little: its table already finds ~100% of ring indices on the
  synthetic, and `--start lattice` there was "no better".

### Continuous lattice spacing along each tube (prototype, 2026-11-14)

Stage 6b reads the spacing in segments, and only on tubes whose sites fit ONE lattice end
to end (≤ 7 A rms): on GDP 10 of 13 tubes sat 7–11.5 A off it and got no rise. The
cause is the bookkeeping, not the lattice: the every-site step predicts each
(index, protofilament) from one whole-tube lattice, and where that drifts more than half a
monomer refine walks the site onto the neighbouring subunit, so the LABEL slips while the
refined POSITION stays good. `spacing_profile.py` ignores the labels for measuring: from
the every-site refines already on disk (no GPU) it takes the right-register patch's pose
per site (parity from Stage 7's sign), keeps sites above the decoys' p99, puts each on one
1-D axial lattice (axis point → smooth curve → arc length → helical unwrap by roll),
re-indexes by clustering (subunits 40.8 A apart, a subunit's sites agree to 1.0–2.8 A),
ring-averages each cluster over its protofilaments, and fits the slope in a sliding
window (default 12 indices ≈ 49 nm), divided by the window's sin(theta). Error: 6b's
formula from the tube's own scatter and neighbour correlation.

```bash
$PY run_scripts/spacing_profile.py --table $P [--truth <synthetic truth.json>] [--window 12]
$PY run_scripts/plot_spacing_profile.py --name <case> --tables $P [$P2 ...]   # split children on one map
```

Writes `spacing_{clusters,profile,summary}_<tag>_w12.csv` and
`spacing_{map,profiles}_<case>_w12.png` next to the table. `--register in` and
`--found-only` are the comparisons below.

| strong synthetic (truth through the same window) | rms error | windows with full coverage |
|---|---|---|
| every site, right register (default) | **0.188 A** | 274 |
| every site, in-register patch everywhere | 0.216 A | 275 |
| found sites only, right register | 0.227 A | 135 |

* Pure C reads 40.97–40.99 (truth 40.943), pure E 41.86–41.95 (41.833); tube 4's
  alternating 60 nm domains are resolved. Pull sd 1.07 / 1.26 on the pure tubes (error
  bars right), 0.57–0.66 on tubes with domains (conservative: a real lattice change
  inflates the noise estimate). Tube 4 matches the truth best shifted by 2 indices (8 nm);
  not explained (the patch models are centred on their atoms).
* The right register tightens a site about its subunit (synthetic 2.45 vs 2.46–2.86 A) and
  costs nothing; most of the gain is from every site (continuous coverage).
* Split-half (even- vs odd-PF ring averages) is reported but is a LOWER bound on the error:
  on the synthetic it is 2–3× below the true error, because both halves share the curve
  and ring-wide wander.
* Noise through a 12-index window looks like smooth wiggles about one window long. Read
  only features well outside the band and longer than the window.

Results (12-index window): **823** 40.1–41.7 A, tube 1 up to ~41.6–41.7 A at 170–190 and
300–330 nm (6b's running profile had found an intermediate 41.5 A stretch there);
**GMPCPP** 41.65–42.12 A, whole tube 41.857 (Stage 5: 41.863); **GDP** all 13 tubes now
profiled (6b: 3), 40.1–41.5 A; label slips repaired on tubes 1, 2, 7. GDP tubes 1, 2 and 7
tilt out of the image plane along their length (theta 72–88°, 74.5–88.4°, 76.8–88.6°):
their projected spacing shrinks up to 5%, which is what broke one-lattice indexing, and
the tilt correction takes their profile spread from 0.83 / 0.53 / 0.61 A to 0.21 / 0.24 /
0.22 A. Tubes 11 and 13 keep few sites (166/349, 122/724) and have wide bands.
Absolute values scale with the pixel size (0.9194 A/px in vitro, 1.056 in situ); changes
along a tube do not.

## Open issues

* **Two SVD defects (found and fixed 2026-11-12, in the scorer, the Stage 2 search and the
  autotrace maps).** Every script takes the B-factor from its optics/config file
  (PICASSO optics files say 0, the `match_tm` configs 60). Autotrace with the centre fix
  (`results_autotrace_svd_fix/`): 823 identical tracks, median offsets 22/16/31/11 →
  18/12/32/10 Å; GDP's curved path 1 covered 57% → 79% at 39 → 15 Å median offset.
  **The centre default was in Panther-EM itself** (`coordinates/*_polar.py` `from_image`)
  and is now fixed there too (on main as #16, and on branch `jd/emit-entries`): default
  `(h // 2, w // 2)`; odd boxes unchanged; Panther's 502 tests pass. On an in situ
  micrograph of yeast ribosomes (LSU, 300 kV; 512³ box, same configs): recall and
  precision unchanged, z at the reference picks up 2.6% on average — ≈ 0 for ψ 0–60° and
  300–360°, largest for ψ 120–240°, the predicted pattern. Earlier Panther-EM results on
  even boxes carry the ψ-dependent loss.
  1. *In-plane rotation centre.* Panther-EM's polar grids default to centre
     `(box/2 − 0.5, box/2 − 0.5)`; Fourier-slice projections (Panther's and Leopard's) are
     centred at `box // 2`. Every template is rotated in ψ about a point half a pixel off,
     which moves the correlation peak by `(I − R(−ψ))·(½, ½)` px — up to 1.4 px by tube
     direction (823: ~15% of z lost on the three tubes at ψ ≈ 194–225°, none on the one at
     42°). Shown with Leopard projections placed at known positions: peaks off on exactly
     the predicted poses, and on none once the grid is centred at `box // 2` (passed as
     the decomposition's `polar_grid.options.center`; `svd_score_sites.py
     --polar-center fft`, the default).
  2. *No B-factor.* Panther's CTF planes carry no envelope; Leopard multiplies by
     exp(−B k²/4) with `ctf_B_factor: 60`. Applied to the 3-D template before
     decomposition (exact: projection keeps |k|); `--b-factor`, default from the config.

  Together on 823: SVD vs Leopard z r 0.80 → **0.95**, ratio 1.00–1.02 per filament, C/E
  contrast per site 0.56 → 0.71 (Leopard 0.76), never-found sites median z 2.4 → 3.2.
  Ruled out on the way: normalisation (raw correlations disagree as much), whitening
  (identical curves), rank (truncation 0.154 → 0.003 changes nothing), orientation grid
  (nside 64), and refine vs match_template frames (≤ 0.06 px).

  Stage 2 with both fixes (`results_svd/*_fix`, tags `svd_fix_*`): strong synthetic ring
  1.40 M → 1.89 M detections, 823 ring 0.93 M → 1.96 M; truncation error at 96×80 on 823
  0.246 → 0.137 (the envelope removes what the rank could not hold); Stage 3 polarity and
  PF unchanged; Stage 5 rises 41.826 / 40.942 / 41.464 (synthetic fils 2–4; Leopard 41.835
  / 40.932 / 41.470) and 41.194 / 40.515 / 40.743 / 40.754 (823; Leopard 41.191 / 40.499 /
  40.765 / unreliable). The B-factor costs a little position precision: Stage 4 site
  deviation on the synthetic 1.1–2.4 Å (old SVD) → 2.0–3.6 Å (fixed; Leopard 2.0–3.2).
* **2-fold harmonic** in SVD tables' Stage 3 PF readout (section 2): harmless for the
  calls tested, unchanged by both fixes, not understood.
* `results_kinesin_2_0031/stage4_sites_kinesin_2_0031_14pf.csv` filament 4 still carries
  the looped-curve index (left, per "skip kinesin for testing").
* Disk: see *Disk* above. `pipe_gmpcpp_14pf` (8.3 GB) is superseded by `pipe_gmpcpp_13pf`
  but kept: it is the record of the PF switch.
* Where the code is (2026-11-25): Leopard-EM branch `jd_constrained_segment`
  (Lucaslab-Berkeley/Leopard-EM#138) and Panther-EM branch `jd/emit-entries`
  (mgiammar/Panther-EM#25, onto `mdg-seed-local-stage1`, which the seeded searches need).
