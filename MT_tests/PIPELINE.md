# Determining the four microtubule parameters from one micrograph

Polarity, protofilament number, lattice spacing, seam — the pipeline, in order, with the
template type stated at every step.

**The one rule that decides ring or patch:**

> **RING for everything geometric. PATCH only for the seam.**
>
> A ring averages the whole circumference, which is what makes it robust to bend strain
> (`u(φ) = u₀[1 + Rκ·cos(φ − φ₀)]` integrates to zero around the tube) and what makes it
> able to compete on protofilament number at all. That same averaging destroys the monomer
> register: a 3-start helix spreads N protofilaments' α/β phases over 1.5 dimer periods, so
> only **21.7%** of the dimer order survives projection for a 13-PF ring against **54.8%**
> for a 5-PF patch. It is *projection*, not the template — their axial autocorrelations are
> identical, 0.6193 against 0.6194.

Measured consequence on the curved synthetic with known truth: the patch reads the register
at **2.19× chance**, the ring at **0.69×**.

⚠️ **The ring is NOT a null, and must not be described as one.** Shift a ring one monomer
axially and every protofilament does go α-on-β — it is perfectly sensitive to the register
in principle. What happens is attenuation in *projection*: 21.7% of the dimer order
survives for the ring against 54.8% for the patch, so the ring carries roughly **40% of the
patch's signal** and falls under the noise at ordinary exposure. It would read the register
given about 2.5× the signal. On the GDP frame the ring gives 0.70× where the patch gives
1.12× — a ratio of 0.63, which is what an attenuated positive looks like, not a null.

**THE null is the same patch re-indexed at the DIMER spacing**, where every site shares a
register by construction, so any alternation there is method artefact.

---

## Stage 0 — CTF, per micrograph  ·  **Template: NONE**

Estimate defocus and astigmatism **from this micrograph**, with PICASSO.

**Pixel size is not in question: it is 0.9194 Å**, externally calibrated. The MRC header
says 0.9432 Å and the header is **wrong** — the microscope was miscalibrated. Do not
"correct" configs back to the header value. Every template in `maps/` is simulated at
0.9194, so changing it would invalidate the whole template library as well as the search.

**CTF: use PICASSO** (`/home/jdickerson/git/LucasLab/PICASSO/programs/ctf_estimation`), which
writes Leopard-EM optics columns directly — `ctf_estimate_optics.csv` plus a per-micrograph
`*_optics.yaml` that drops straight into the `optics_group` block.

Copy `ctf_estimate_example_config.yaml` and set:

| knob | value | why |
|---|---|---|
| `optical_config.pixel_spacing_angstroms` | `0.9194` | our calibration, not the header |
| `target_pixel_spacing_angstroms` | `1.4` | Nyquist 2.8 Å keeps the 4 Å thickness cutoff in band |
| `fitting_config.frequency_fit_range_angstroms` | `[30, 4]` | defocus band |
| `patch_sidelength` | `512` | matches a typical CTFFind box |
| `defocus_grid_resolution` | `[1, 3, 3]` | gentle spatial field, untilted |
| `defocus_range_microns` | `[0.0, 5.0]` | **widen if the fit rails at the edge** — this frame's defocus is unknown |
| `optimize_envelope_1d` | `false` | envelope competes with thickness |
| `thickness_config.refine_dim` | `"thickness"` | freeze the 2D defocus; joint refinement pulls defocus off |

Run from the PICASSO repo root, always `uv run --no-sync`:

```bash
uv run --no-sync python programs/ctf_estimation/run_ctf_estimation.py
```

The **2D** defocus is the reported one (`defocus_u`/`defocus_v` in the optics CSV). On a
5-micrograph set it tracked CTFFind5 to within ~40 Å. `thickness_angstroms` is PICASSO's own
CTF\(_t\) number and is systematically ~170–270 Å thicker than CTFFind — do not treat it as
validated ice thickness, and do not jointly refine it with defocus.

**Gate — do not skip.** A defocus fit with sensible astigmatism, not railed against the
search bounds.

### Cross-checking against CTFFIND — and the trap in it

The lab's CTFFIND setup lives in `/home/data/jdickerson/MT_data/23May25_SPA_bare`
(`ctffind_all.py` substitutes `&` in `ctffind_base.txt` and runs the result). It already
uses pixel size 0.9194, so it is a genuine independent check. **But CTFFIND5 fits sample
thickness by default, and on a low-defocus frame that breaks the defocus.**

On `GDP_curved_mgraph` the thickness brute force railed at the top of its range (4000 Å) and
the 2D refinement failed to recover it (3478 Å); a thickness model that wrong drags the
defocus with it. Turning thickness off fixes it completely:

| method | defocus_u | defocus_v | **mean** | astig angle |
|---|---|---|---|---|
| PICASSO | 2104.3 | 1873.9 | **1989.1** | −8.40° |
| CTFFIND5, thickness **off** | 2183.0 | 1807.7 | **1995.3** | −9.92° |
| independent 1D scan of CTFFIND's own radial spectrum | — | — | **2010** | — |
| CTFFIND5, thickness on, 30–5 Å band | 4015.0 | 3243.9 | 3629.4 | +2.77° |
| CTFFIND5, thickness on, 30–3.5 Å band | 3110.4 | 3045.0 | 3077.7 | +9.39° |

The two thickness-on runs disagree with *each other* by 550 Å, which is the tell. Low defocus
is exactly when this bites: at 1989 Å the first CTF zero is at 6.10 Å, so a 30–5 Å band holds
a single zero at its very edge — not enough oscillation to pin thickness and defocus at once.
On the GMPCPP control (8213 Å) there are six zeros in band and thickness-on is fine.

⚠️ **`Thon rings with good fit up to ...` is not a quality flag on this build.** It returned
the byte-identical `717.131958` on all three GDP runs, including ones that produced different
defocus values.

**Verdict for this frame: `GDP_curved_mgraph` is at ~1989 Å (0.199 µm)**, astigmatism 230 Å
at −8.40°, thickness ~674 Å. The PICASSO per-micrograph YAML in `results_ctf/` is the value
to use. Note the defocus *field* spans 1693–2291 Å across the frame (std 178 Å), so the
±600 Å defocus search in the match_template config is both necessary and sufficient.

> **This stage did not exist in the first version of this pipeline, and skipping it cost
> ~4 GPU-hours for zero detections.** `setup_gdp_curved.py` copied another micrograph's
> CTF: `defocus_u: 8390.503906`, `defocus_v: 8035.022461`,
> `astigmatism_angle: -0.864719` appear in *every* config in `configs/`, GMPCPP and GDP
> alike. Three full searches completed and found **0 peaks** (max z 6.67 / 6.81 / 6.84
> against a 7.62 threshold). With the pixel size known to be right, **the wrong defocus is
> the sole remaining explanation** — and the search only scanned ±600 Å around it, so a
> frame at a genuinely different defocus could not have been found.

Cost: seconds.

---

## Stage 1 — Annotation  ·  **Template: NONE (but sized for the PATCH)**

`programs/constrained_search/napari_choose_filament_path.py` → editable polylines with a
per-path width → `eligible`, `region_id`, `psi_center` (from the local **tangent**, not the
normal) via `make_filament_constraint.py`.

**The width must be set for the patch even though this stage uses no template.** A ring is
axis-centred (axis offset **0.0 Å**), but the 5-PF patch's box centre sits **85.8 Å =
93.4 px** off the axis, and that offset *rotates with φ*, so patch detections sweep a 93 px
circle about the axis. A ±160 px band leaves only 67 px for drawing error; **560 px width**
is the working value here (28.30% of the frame eligible).

Also fix `--smooth-px` (default 5): an interpolating spline turns hand jitter into real
wiggle in the tangent, which *is* `psi_center` — measured 3.40° rms / 14.07° peak before
smoothing, 0.76° / 3.56° after.

### Running it

The tool is standalone and does not import `leopard_em`, so it runs in a napari-only
environment. It needs `napari`, `numpy`, `scipy`, `mrcfile`, `pyyaml`, `qtpy` **and
`h5py`** -- h5py is imported lazily inside the exporter, so a missing one does not fail
until the moment you press export, after the drawing is done. Check it first.

Drawing happens on a laptop, so the repo gets mounted over sshfs from albatross:

```bash
# --- on the laptop ---
mkdir -p ~/mnt/albatross
sshfs jdickerson@albatross:/home/jdickerson/git/LucasLab/Leopard-EM ~/mnt/albatross \
    -o follow_symlinks,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
# macOS/macFUSE also accepts -o volname=albatross
# unmount with:  fusermount -u ~/mnt/albatross   (Linux)
#                umount ~/mnt/albatross          (macOS)

conda activate napari
cd ~/mnt/albatross
python programs/constrained_search/napari_choose_filament_path.py \
    --micrograph MT_tests/results_in_situ/set1_823_Aug09_10.05.20_DW.mrc \
    --output MT_tests/configs/filament_constraint_in_situ_823.yaml \
    --pixel-size-angstrom 1.056 \
    --width-px 490 \
    --cone-half-angle-deg 20 \
    --theta-center-deg 90 \
    --smooth-px 5
```

Writes the sidecar YAML, the HDF5 beside it, and `<output>.paths.json`. Reopen with
`--paths MT_tests/configs/filament_constraint_in_situ_823.paths.json` to keep editing
the same curves rather than redrawing them.

### Semi-automatic: propose, pick, extend (the route going forward)

Drawing every tube by hand is slow; most of it can be proposed. Three steps, all on
albatross except the optional touch-up:

```bash
# 1. propose tracks: a coarse template-matching tracer (~1 min on one GPU)
python ../programs/constrained_search/auto_trace_filaments.py \
    --micrograph results_kinesin/20260916_lysates_2_0031_DW.mrc \
    --template maps/6dpu_14pf_2rings_flatB_1.13_bscale0.5.mrc \
    --optics results_ctf_kinesin/20260916_lysates_2_0031_DW_optics.yaml \
    --outdir results_autotrace --prefix kinesin_2_0031 --gpu 3
#    -> results_autotrace/kinesin_2_0031_overlay.png (numbered tracks) + .paths.json

# 2. a person reads the overlay and says which tracks are microtubules; keep those,
#    join pieces of one tube with +, extend tubes that cross the frame
python ../programs/constrained_search/curate_paths.py \
    results_autotrace/kinesin_2_0031.paths.json \
    --micrograph results_kinesin/20260916_lysates_2_0031_DW.mrc --pixel-size 1.13 \
    --keep 11 7 15 14 13 --extend-to-edge \
    -o configs/filament_constraint_kinesin_2_0031
#    -> .paths.json, _curated.png, .curation.json (source, kept, dropped, joined)

# 3. export the sidecar the search uses (as for a hand drawing)
python ../programs/constrained_search/make_filament_constraint.py \
    configs/filament_constraint_kinesin_2_0031.paths.json \
    -o configs/filament_constraint_kinesin_2_0031.yaml \
    --micrograph results_kinesin/20260916_lysates_2_0031_DW.mrc \
    --cone-half-angle-deg 20 --theta-center-deg 90 \
    --psi-step 1.5 --theta-step 2.5 --pixel-size-angstrom 1.13
```

Anything still wrong after step 2 is fixed in the drawing tool: reopen the curated file
with `napari_choose_filament_path.py --paths configs/<name>.paths.json`.

**What the tracer is good and bad at** (`auto_trace_filaments.py`'s docstring has the
method): a 64 nm smooth hollow cylinder of the dataset's own radial profile, at a
resolution set by the defocus, traced along the z map's ridges. In vitro it finds every
tube but over-picks -- on kinesin_2_0031, 16 tracks for 5 tubes: pieces of one tube,
and wall side-lobes ~21 nm off the axis (the 9 and 12 either side of 14). In cells it
cannot tell a microtubule from a ~25 nm bundle of thinner filaments at 8-15 A, and in
crowded near-focus fields (GDP) it fragments. Hence the person in step 2.

**Every curated micrograph is a training example.** Curate *every* microtubule in the
frame, not just the ones to be analysed -- an untraced tube teaches a model that tubes
are background -- and keep the `.curation.json`. With ~10-30 curated micrographs per
imaging condition a filament picker can be fine-tuned; ~50-100 across conditions would
train one from scratch.

**Export on albatross, not over the mount.** A sidecar written through sshfs records the
laptop's paths (`/Users/.../mnt/albatross/...`), which the search cannot open. Re-export
from the saved paths with the headless tool, which also means the laptop's napari env
never needs `h5py`:

```bash
cd MT_tests
python ../programs/constrained_search/make_filament_constraint.py \
    configs/filament_constraint_in_situ_823.paths.json \
    -o configs/filament_constraint_in_situ_823.yaml \
    --micrograph results_in_situ/set1_823_Aug09_10.05.20_DW.mrc \
    --cone-half-angle-deg 20 --theta-center-deg 90 \
    --psi-step 1.5 --theta-step 2.5 --pixel-size-angstrom 1.056
```

**Overlapping bands keep every filament.** Where bands overlap -- a bundle, or one tube
crossing another -- the export records EVERY covering path per pixel (`psi_center` /
`region_id` for the nearest, `*_extra` layers for the rest), and the search accepts an
orientation any of them allows. The old rule kept only the nearest path, so at a
crossing a pixel owned by one tube rejected the other's genuine detections. On 823,
29.4% of eligible pixels are covered 2-3 deep. Validated on a synthetic with 823's layout
(`simulate_microtubule_bundle.py`, `validate_overlap_architecture.py`): axis coverage in
the crossing zones rose from 80% to 93% on two tubes and from 38% to 48% on the
crossing tube, and the old-gate table is an exact subset of the new one.

**Why these values differ from the in vitro run.** Every pixel figure in this stage is
a physical length divided by the pixel size, and the in situ set is **1.056 A/px**
against the in vitro **0.9194**, so nothing carries over unconverted:

| | in vitro (0.9194 A/px) | in situ (1.056 A/px) |
|---|---|---|
| patch axis offset 85.8 A | 93.4 px | **81.3 px** |
| band width ~515 A | 560 px | **490 px** |

`--pixel-size-angstrom` is belt and braces: RIPPLE does write 1.056 into the MRC
header, so the tool would read it, but the cropped files elsewhere in this project
carry none and the flag costs nothing.

`--cone-half-angle-deg 20`, not the 10 used in vitro. On the GDP frame a cone of 10 put
theta hard against its [80, 100] limit while the data wanted 79.0-99.1 -- the constraint
was too tight to begin with and could not be discovered to be wrong from inside itself.
In situ tubes sit at more varied tilts and the drawing is over noisier data, so start
wide. **Remember `cone_half_angle_deg` gates BOTH `theta_center +/- cone` AND
`psi_center +/- cone`** -- one knob, two angles -- so 20 is already generous on psi for
a hand-drawn curve. Narrow it later by refitting from the ring search
(`refit_constraint_from_ring.py`); widening it afterwards means redoing the search.

**Determines:** where the filaments are, and the drawn tangent used later as
semi-ground-truth. Cost: minutes of human time.

---

## Stage 2 — ONE full `match_template`  ·  **Template: RING**

The only expensive step. One reference ring at the best-guess N, lattice matching the
nucleotide state (GDP → compacted/6DPV, GMPCPP → expanded/6DPU), with the Stage 1 sidecar.

**Why ring:** it is axis-centred so every detection predicts the axis directly; it is
N-fold pseudo-symmetric so it finds the tube whatever the roll; it carries the most mass, so
it is the most sensitive detector available.

**Gate: peaks above the analytic threshold.** If zero, **stop and return to Stage 0** — do
not run further templates. The threshold is `erfcinv(2·FP/num_ccg)·√2`, ≈ **7.62** for a
5760×4092 frame at 485,856 orientations × 7 defocus.

Cost: ~1.3 h on 4 GPUs.

### It emits three products, and they are not interchangeable

| product | contents | what it is for |
|---|---|---|
| **correlation table** `.h5` | every detection above raw CC 6.5, **including sub-maximal orientations at the same pixel** | **the analysis input** |
| maps: `mip`, `scaled_mip`, `orientation_*`, `correlation_average`, `correlation_variance` | one hypothesis per pixel | `refine_template`'s normalisation; the paired template comparison in Stage 6 |
| peak CSV | the thresholded, de-duplicated peak list | a convenient particle stack — **not** the extraction input |

The table exists because the maps keep one hypothesis per pixel, so the near and far walls
of a tube annihilate each other and every sub-maximal orientation is thrown away — and
orientation *is* the measurement here. On `GDP_curved_mgraph` the table holds **27,544
detections** against **288** in the peak CSV. Analysis reads the table.

**Coordinate frames.** The table's `x`/`y` are the correlation-map frame; the CSV's
`pos_*_img` are the micrograph frame. They differ by exactly half the template box: 300
for the in vitro 600³ templates, **264** for the in situ 528³ one, and 256 for the two
early crop searches that used a 512³ map. The drawn curve from Stage 1 is in the
micrograph frame. **Never hard-code the offset**: every table now records its pixel size,
template shape and micrograph shape, and `run_scripts/frame_geometry.py` reads them
(`frame_geometry(table)`). Tables written before that need the template map the search
actually used, and the helper refuses to guess.

### In situ (set1_823)

| | value | why |
|---|---|---|
| template | `6dpu_atoms_6dpv_lattice_13pf_2rings_flatB_1.056_bscale0.5.mrc`, 528³ | same physical box (558 Å) as the in vitro 600³ at 0.9194; rise 40.947 Å vs 40.943, axis on the box centre |
| grid | ψ 1.5° / θ 2.5°, uniform: **1,584,480** orientations | finer than in vitro (485,856) |
| defocus | ±800 Å in 200 Å steps (9 planes) about PICASSO's 9855/9806 Å | spatial field spans 808 Å; ice 1077 Å |
| constraint | `filament_constraint_in_situ_823.yaml`, cone 20°, 4 paths, layered | |
| thresholds | full frame **7.78**, constrained **7.32** | 14.26 M correlations per pixel |
| cost | **~5.6 h** on 4 GPUs (~700 correlations/s) | 4.2x the in vitro grid |

```bash
cd MT_tests
python run_scripts/simulate_lattice_competition.py --pixel-size 1.056 --box 528 \
    --stems 6dpu_atoms_6dpv_lattice_13pf_2rings_flatB
python run_scripts/run_match_curved.py --prefix "" --tag in_situ_823_13pf \
    --results results_in_situ_823 \
    --constraint configs/filament_constraint_in_situ_823.yaml
```

---

## Stage 3 — Read the table  ·  **Template: RING** (its detections)

Two readouts come straight off the table with no axis and no refinement.

- **Polarity** — `estimate_polarity` uses ψ alone, so bending cannot bias it. Report per
  segment and check each against the drawn tangent.
- **PF number, first line** — `estimate_protofilament_number` is a pure azimuth harmonic on
  φ. Coarse φ is adequate here: the search samples φ at ~3.5°, which is 8 samples per a
  13-PF period of 27.7°. It is refinement-free *and* axis-free, which is what makes it an
  independent line of evidence against the Stage 6 competition.

**Membership comes from the data, not the drawing.** Which detections belong to which
filament used to be the drawn region at each detection's box centre. That takes geometry
from the drawing, breaks where bands overlap, and uses the wrong point for a patch (its
box centre sits 85.8 Å off the axis and swings with φ). `assign_detections_to_filaments`
(`filament_lattice.py`) instead uses each detection's OWN predicted axis point and
in-plane direction against curves fitted to the detections, the drawn paths only seeding
who starts where; position separates parallel tubes, direction separates a crossing.
Polarity is then read per ~100 nm stretch along the DATA-fitted curve and checked against
that curve's tangent.

```bash
python run_scripts/stage3_read_table.py \
    --table results_in_situ_823/output_correlation_table_in_situ_823_13pf.h5 \
    --constraint configs/filament_constraint_in_situ_823.h5 \
    --micrograph results_in_situ/set1_823_Aug09_10.05.20_DW.mrc
```

Writes `stage3_membership_<tag>.csv` (every detection with its filament) for Stage 4.
Checked before use: on the bundle synthetic it recovers 13 PF and the right polarity for
all four tubes (two of them reversed), with 99.95% of on-tube detections in the right
tube against 95.8% for the drawn-region rule; on GDP it reproduces the recorded readouts
exactly -- 13 PF at margins 1.51× and 2.84×, ∞:1 in every readable stretch -- with no
detection changing filament, as expected for two well-separated tubes.

Cost: seconds.

---

## Stage 4 — Curve and site extraction  ·  **Template: RING** (ring detections only)

Project the **table's** detections onto the drawn curve, segment by arc length, then
`extract_lattice_sites` per segment. This is the top-down step, and it reads the full
table — not the peak list — because that is the whole point of the table.

`extract_lattice_sites` does it in four moves: bootstrap an axis from a *high* cut (z > 8)
using orientation-aware peak finding that suppresses only detections within **12 px AND
15°** — so the two walls of the tube survive as separate hits; unwrap the helix by
`s_corr = s + (lateral_offset/angular_spacing)·φ`; assign every detection to its nearest
lattice site with the **best-scoring one in each bin winning, however weak**; refit the
rise and repeat.

**Why ring:** bend strain is `u(φ) = u₀[1 + Rκ·cos(φ − φ₀)]`, which averages to zero around
the circumference. A ring is therefore unbiased by it and a patch is not. Lattice spacing
must be read from the ring. A ring's `axis_offset` is also 0, so an axis point simply *is*
the detection position — angles do not enter the axis fit at all, which is why this stage
runs fine on unrefined angles and why refinement comes after it, not before.

**Segment length is set by axial distortion, not by the bow.** `extract_lattice_sites`
discards the transverse coordinate entirely (`along, _ = filament_coordinates(...)`) and
applies no transverse gate, so sideways bow costs nothing. What breaks is site *assignment*,
which rounds the axial coordinate to the nearest repeat: the lattice is uniform in **arc**
length, the measurement is in **chord** length, the best linear relation between them is the
shortening (absorbed into the fitted rise), and only the **residual** mis-indexes sites.

Cut segments so that residual stays under **5 Å** (a quarter of the ±20.5 Å rounding window;
the curved synthetic sits at ≈2.3 Å and recovered its rise to 0.006 Å). On the two drawn GDP
paths that gives 5 segments of 28 repeats and 4 of 34. Over a whole path the residual
reaches **110 Å — 2.7 monomers**, so one straight axis there assigns sites to the wrong
subunit entirely.

Measure the shortening directly off the drawn curve — fit a line over the window, project,
take (projected span)/(arc length) − 1. Do **not** use the −L²/(40R²) law on real tubes:
these are not circular arcs (region 2 turns 60° yet its arc exceeds its chord by 1.66%, an
S-bend), and the law predicts −4.66% there against a measured −1.64%.

**Determines:** the **clean lattice site list** — indexed, one detection per site — and a
first **lattice spacing** per segment, shortening-corrected, with the between-segment
scatter as the error bar. Cost: seconds.

**Anti-circularity check.** The assignment window is a full half-repeat (±rise/2 = ±20.9 Å)
with no transverse and no angular gate, so positions are free to land anywhere;
`axial_deviation_angstrom` records where they actually did. On the GMPCPP frame that was
**2.92 Å rms against ±20.9 Å**, which says the data and not the model set the positions.
Re-run from different bootstrap thresholds and starting repeats — the answer should not
move (spread 0.008 Å, against 0.083 Å for threshold-based picking).

### Running it: continuous, per filament (`stage4_sites.py`)

Segments are the fallback. The default indexes each filament CONTINUOUSLY along a curve
fitted to its own detections (Stage 7 explains why), using Stage 3's membership:

```bash
python run_scripts/stage4_sites.py \
    --table results_in_situ_823/output_correlation_table_in_situ_823_13pf.h5
```

The bootstrap cut is the highest z leaving **20 distinct peaks** under the same 12 px /
15° suppression `continuous_sites` applies. Counting table ROWS, as `pick_bootstrap` did,
fails on weak tubes: the bundle synthetic's strongest tube had 20,057 rows but only 17
peaks above the cut rows chose. Each rise comes with its statistical error and a flag:

**Read a filament's rise only with 20+ sites and ≤ 7 Å rms deviation.** Empirical, from
the bundle synthetic and GDP: filaments that measured right had 23–65 sites at 3.2–6.6 Å;
the two that did not had 7.9–10.1 Å (a 16-site tube came out 3.4σ from the truth, and GDP
region 1 was already "unresolved"). Large deviation means sites are landing on the wrong
subunit, and the quoted error does not see it.

On the bundle synthetic (truth 40.943 Å) the three trustworthy tubes read 40.946 ± 0.035,
40.992 ± 0.037 and 41.025 ± 0.070; GDP filament 2 reproduces `continuous_unwrap_curved`
exactly (63 sites, 97%, 40.774 Å).

**Quote the BLOCK error, not the fit's.** The fit's ± is statistical and blind to the rise
varying along a tube -- and on real data it does. Halves of one filament agree to 0.09%
on the synthetic (uniform by construction) but differ by 0.96% on GDP and 0.4–1.9% on in
situ 823, so the variation is in the data. Both stages therefore re-measure contiguous
blocks of 25+ sites and report the error of their mean. On 823 that is 6–14x the fit's
error. Whether the variation is real lattice heterogeneity or a real-data systematic the
synthetic lacks (local scale distortion from motion correction is a candidate) is open.

---

## Stage 5 — `refine_template` on the extracted sites  ·  **Template: RING** (the same one)

Build a particle stack **from the Stage 4 sites** and refine it. Not from the peak list:
the sites are the objects every later measurement is about, and they include weak
detections the peak list threw away.

Polish φ/θ/ψ and defocus within **one coarse cell** at a fine step. `coarse_*_step` means
the step of the `match_template` grid you already ran (ψ 2.5°, θ 3.5°, defocus 200 Å) — the
refine fills in what the coarse search skipped and nothing it already did. It is not a
range knob. `run_refine_curved.py` reads both YAMLs and refuses to run if they disagree.

**Why it has to be here and not earlier:** site assignment (Stage 4) is one-dimensional
with a ±20.9 Å window and no angular gate, so coarse angles are tolerable there, and a ring
axis fit ignores angles entirely. φ precision only becomes the limiting factor for the
φ-*drift* measurements — supertwist, moiré, per-protofilament azimuth — which come after.

**Not optional for those:** φ quantisation previously faked a supertwist slope from 19
distinct φ across 82 sites; refinement at 0.4° took it to 59 and raised mean z 13.26 →
15.51.

Two defects of `get_local_high_resolution_angles`'s `uniform` branch, worth knowing:
φ's *range* comes from `coarse_psi_step` while `coarse_phi_step`/`fine_phi_step` are inert;
and **the offset grid does not contain the identity** (nearest point 2.7° away, φ pinned at
its minimum), so a particle already at its optimum can only move off it. Expect ~30% of
scores to fall slightly while the rest gain much more — on the GDP ring, −0.270 against
+0.988, net +0.586.

**Determines:** sharp angles for every φ-dependent readout, and a better lattice spacing.
Cost: seconds to minutes.

### Running it (`stage5_refine.py`)

```bash
python run_scripts/stage5_refine.py \
    --table results_in_situ_823/output_correlation_table_in_situ_823_13pf.h5
```

Nothing run-specific is hard-coded: micrograph, optics and template come from the search
config; the coarse steps from the search grid (in situ ψ 1.5°, θ 2.5°); the box sizes from
the template header (528 → 568 extracted); pixel size, offset and `total_correlations`
from the table. After refining, each filament is re-measured along a curve fitted to the
REFINED sites, with the same reliability flag and block error. On GDP's existing refined
sites filament 2 gives **40.834 Å, block error ± 0.199** -- comparable to the segmented
route's 40.818 ± 0.295, which was also a between-segment scatter; the fit's own ± 0.022
is not comparable to either -- and filament 1 is flagged (+1.65%, 10.1 Å deviation),
matching its recorded "unresolved".

---

## Stage 6 — Competition at those sites  ·  **Template: RING for N and lattice; PATCH built here for Stage 7**

`constrained_search` at the reference's positions and orientations. **No new
`match_template` runs.**

| question | template | why |
|---|---|---|
| protofilament number, 12/13/14/15 | **RING** | a complete tube is what differs with N; a 5-PF patch barely distinguishes it |
| expanded vs compacted at the winning N | **RING** | bend-strain robust; both built from the *same* 6DPU atoms, so no model-quality confound |
| register / seam | **PATCH** (5 PF) | see the rule at the top — the ring cannot do this test |

Configuration:
- **φ range ±180/N.** Rivals roll onto their own protofilament: measured |dφ| of 67–89°
  between PF templates, but only **6–8° once each template's own period is removed**
  (30.0°/27.7°/25.7°/24.0°). ψ and θ agree to one grid step, so hold them tight (±2–3°) —
  ψ is already known from the constraint.
- **`center_vector`: 0 for rings, from `utils/get_center_vector.py` for the patch.**
- For the lattice 2×2 the four templates agree on orientation to **0.0° exactly**, so that
  comparison needs essentially no angular search at all.

**Score in raw CC, paired per site.** The mean/variance maps are the *only* thing that would
force a rival to have its own `match_template`, and for a paired comparison at identical
pixels they cancel: ice, defocus and near/far wall live in μ(p) and σ(p), which both
templates see at the same pixel. Validated on the lattice 2×2 —

| contrast | true per-pixel z | raw CC | per-site sign agreement |
|---|---|---|---|
| 6dpv@6dpu − 6dpu@6dpu | +0.257 ± 0.017 | +0.259 | 97.0% |
| 6dpv@6dpu − 6dpu@6dpv | +0.640 ± 0.022 | +0.680 | 98.7% |
| 6dpv@6dpu − 6dpv@6dpv | +0.510 ± 0.021 | +0.545 | 100.0% |

— identical ordering, and the correct winner on both known-truth synthetics (13pf on
synth13, 14pf on synth14). Per-template offset/scale corrections were tested and **never
changed an ordering**, so they are second-order.

**Boundary:** same pixel, same micrograph. This does *not* license comparing across
micrographs or defocus groups, where the per-pixel terms stop cancelling. Absolute
significance ("is this peak real at all?") still needs a true z and therefore the maps —
but none of the four readouts asks that.

**Determines:** **PF number** (second, independent line) and **lattice spacing**
(expanded vs compacted). Cost: minutes per rival.

⚠️ **These scores favour the template that found the sites** -- by +0.17 / −0.12 z on the
known-domain synthetic, as large as the lattice signal -- because each rival starts from
the winning detection's pose and refinement lets the finder climb back to its own noise
peak. For compacted vs expanded use Stage 6b's neutral-pose scores. The PF competition has
the same bias in the same direction (towards the reference N); its margins on data (t
11–23) are far larger than 0.17 z per site would produce, but a neutral PF competition
needs each rival's own roll, which a neutral pose cannot supply without a search.

---

## Stage 6b — Lattice state per segment along each tube  ·  **Template: RING** (both lattices)

Stage 6 says compacted or expanded once per tube. The lattice can change along a tube, so
this reads it in contiguous segments, **as short as each tube resolves**. Standard: run it
after Stage 6 on every micrograph.

```bash
python run_scripts/stage6b_local_lattice.py \
    --table results_in_situ_823/output_correlation_table_in_situ_823_13pf.h5
# finer, below the guaranteed resolution (a secondary view, not the call):
python run_scripts/stage6b_local_lattice.py --table ... --size 12
# on a known-domain synthetic, also score every length against the truth:
python run_scripts/stage6b_local_lattice.py --table ... --truth Frames/<synthetic>.truth.json
# from the PATCH's protofilament-resolved sites (after Stage 7), patches competed:
python run_scripts/stage6b_local_lattice.py \
    --table results_in_situ_823/output_correlation_table_in_situ_823_13pf_patch.h5 --patch
```

Reads Stage 5's `stack_sites_<tag>.csv` / `refined_sites_<tag>.csv` (with `--patch`: Stage
4's protofilament-resolved sites and Stage 7's in-register refine of them); writes
`stage6b_segments_<tag>_neutral.csv` and `.png` (`_n<size>` with `--size`), and
**`stage6b_profile_<tag>_neutral.csv`** -- the same readouts in a window of the tube's
length centred every length/8 indices. The neutral scoring (below) runs
`compete_sites_curved.py --neutral-pose` itself -- one pose per site, seconds on the GPUs
-- and is reused on later runs (`--rescore` to redo).

**Read the profile, not only the tiles.** Non-overlapping tiles depend on where their
boundaries fall: on 823 tube 1 they split an intermediate 41.5 Å stretch between two
segments and averaged it away. The profile keeps the same resolution (the window) without
that dependence; colour each lattice index by the window centred on it.

**Ring or patch.** The patch has 8–11 sites per lattice index to the ring's one, so its
per-index scatter is smaller and it resolves shorter windows (823: 98–139 nm against
106–229 nm). Both readouts are averaged per lattice index before any slope, noise or
correlation is estimated, and the patch's neutral pose takes its neighbours from the same
protofilament (its roll picks one out). Its lattice signal per site is small (+0.07 z
against the ring's 0.27 on the synthetic), so the patch competition leans on numbers.

### Two readouts per segment, both signed positive = compacted

| readout | what | error | role |
|---|---|---|---|
| **local rise** | slope of the refined sites' deviation from the tube's continuous lattice against index, plus the spacing it was measured against, **÷ the segment's own sin θ** | slope error × √(variance inflation of the deviations); Student t on n − 2, converted to a normal z | **the call** |
| **competition** | mean dz = z(compacted) − z(expanded), both scored at a **neutral pose** | noise sd from the detrended dz × √(its variance inflation / n) | second, independent line |

The call is which side of the midpoint (41.388 Å) the local rise lies, by |z| > 3.
**Each segment also says which repeat its rise is consistent with** (C 40.943, E 41.833,
both, neither): a segment between the two is undecided because it is *measured* to lie
between them -- a mixture below the resolution, or an intermediate lattice -- not because
it is noisy.

### The resolution is computed per tube, not guessed

The rise's target is known before looking: the two repeats sit **0.445 Å** either side of
the midpoint. A contiguous run of n sites has a rise error
σ·√(12 / n(n² − 1))·√inflation / sin θ, with σ the tube's own detrended site scatter. The
segment length is the smallest n whose expected |z| clears 3 with 90% power, so it differs
between tubes: 26–56 sites (106–229 nm) on set1_823, 16–43 on the matched synthetic. The
competition's power is not known in advance (its separation depends on the data's SNR), so
it does not set the length.

### Rules

- **Score the competition at a neutral pose; search nothing.** Stage 6's scores start from
  the pose of the detection that won the site -- chosen by one template at its own noise
  peak -- and refinement lets that template climb back to it. On the known-domain synthetic
  that is **+0.17 z towards compacted for compacted-found sites and −0.12 z towards expanded
  for expanded-found**, the size of the lattice signal (0.12–0.15). Starting from the
  lattice but searching the roll still left +0.06 / −0.10: a roll search reaches the peak
  too. The neutral pose: position where the tube's continuous lattice puts the site, plus
  its **neighbours'** median offset from theirs; roll phase (φ mod one protofilament), θ,
  ψ, defocus from the neighbours; **the site itself excluded**. Offset then +0.002 ± 0.031 /
  −0.061 ± 0.043 (the finder bias, the difference between the two, 0.06 ± 0.05 from 0.29),
  signal kept. `--start lattice` scores every lattice index, found or not: no better.
- **One polarity per tube.** ψ is the neighbours' AXIS (ψ mod 180) put on the tube's
  polarity (the side most of its found sites take); roll, θ, defocus and offsets come only
  from neighbours of that polarity. Before 25 Oct 2026, ψ was a median of differences
  wrapped at 360: where found sites mix polarity (kinesin_2_0031: 15–28% per tube) it
  landed on either, or at ±90° -- the template across the tube (15% of 2_0031's sites).
  823's stacks were clean; the synthetic bundle PATCH stacks were not.
- **`--refit-roll`** (ring, `--start neutral`) replaces the neutral rolls -- each a mean of
  a few found rolls, errors of 1–6° -- by one smooth roll curve per tube (straight under
  50 indices, quadratic beyond) fitted to the reference ring's summed z over a ±12° grid
  at every site, positions moving with the roll along the lattice. On 2_0031 it fixes the
  strong tubes' roll to ±0.3–0.4° (bootstrap); tubes that fit no smooth 14-PF roll (weak,
  or not 14-PF) stay at ±4°. Needed for anything scored OUTSIDE the wall: a decorated
  template loses all its motor contrast at a 5° roll error that keeps 44% of the bare z.
  Outputs get `_rollfit`.
- **Neighbouring sites are not independent.** A 2-ring template spans four monomers, and
  neighbouring refined positions wander together (823: lag-1 correlation 0.25–0.57). Both
  errors are widened by 1 + 2Σρ over lags 1–6, from residuals after a 24-site running mean
  (so a real lattice change is not counted as noise).
- **Always the per-segment sin θ.** The deviations are projected Å; only after dividing
  by the segment's own sin θ is the local spacing a rise.
- **Stage 4's reliability rule holds here too.** A tube whose sites deviate more than
  7 Å rms gets no rise readout, only the competition (823 tube 4's ring fit, 7.9 Å, read
  40.19; its patch reads 40.83).
- **The labels are only as good as the pixel size -- so fit it.** Both readouts measure
  spacing in pixels against templates rendered at the search's pixel size, so a scale
  error moves them together: 1% is 0.41 Å, nearly the half-gap. On an uncalibrated
  dataset run `fit_pixel_size.py --patch` after Stage 6b (both patches over a grid of
  render scales, scored at the neutral poses; protofilament spacing fixes the scale),
  then Stage 6b again with `--pixel-size <fitted>` if it differs from nominal. **Use the
  patch, not the ring**: both recover a synthetic's truth (patch +0.01%, ring +0.09%), but
  on set1_823 the ring read 1.047 and the patch 1.056 (= nominal) -- the ring's scale is
  set by the whole projected width, which a slightly flattened or wider tube changes.
  Compacted won at every scale in both. And a stronger competition at a trial scale is
  not evidence for it: shrinking the pixel size widens the compacted-minus-expanded gap
  by itself.

### Validated on the 823-matched known-domain synthetic

`simulate_microtubule_bundle.py --domains lattice` at 823's layout and strength (tube 1
C | E halves, 2 all E, 3 all C, 4 alternating every 600 Å), Stages 2–6 run as on data.
Rise pulls (measured − true)/error have sd **0.92–1.07** at 8–20 sites (1.2–1.3 at 24–32,
from 8–13 windows), mean +0.1 to +0.6 (a +0.08 Å high bias on compacted, a tenth of the
gap). **No readout made a wrong call at any length.** Tube 2 resolves at 16 sites (65 nm,
EEEE), tube 3 at 29 (CC); tubes 1 and 4 read "neither" -- correctly, their segments are
mixtures (tube 4's 600 Å domains are 15 sites, below its 43). At this SNR the competition
alone needs ~60 sites; on 823 it is 3–4× stronger per site.

**At 823's real strength** (that synthetic undershot 823 by ~1 z; re-simulated 16%
stronger, refined ring z 8.1 → 8.8 against 823's 9.0): site scatter 1.3–2.1 Å, windows
**16–31 indices (65–127 nm)**. Tube 1's halves are now resolved -- **E E C C**, truth
EECC -- tube 2 EE, tube 3 CCCCC, tube 4 (600 Å domains) "neither", as it should be. Rise
right 95% at 24 sites and 100% at 32 and adaptive; **no wrong call by any readout**;
pulls sd 0.91–1.14 at 8–20 sites. Two cautions from it:

- **One end segment read 0.4 Å low (pull −7)**: tube 3's last six sites, where the tube
  leaves the frame, compress progressively to −5.9 Å although the simulated spacing is
  uniform to the last block. Not the crossing; other tube ends at the border read right.
  Read a segment at a tube end that leaves the image with care.
- **The neutral-pose competition offset here is +0.143 ± 0.027 z** (compacted-found; C
  +0.46, E −0.18). The finder bias and any fixed difference between the two templates
  cannot be separated without an expanded-found run; it never produced a wrong call, but
  it is why the rise, not the competition, makes the call.

**Determines:** compacted / expanded per segment, and each segment's rise. Cost: seconds.

---

## Stage 7 — Seam / monomer register  ·  **Template: PATCH** (ring = negative control)

### Running it (`stage7_register.py`), and the estimator

```bash
python run_scripts/stage3_read_table.py --table <patch table> --stem <patch> \
    --constraint configs/filament_constraint_<frame>_refit.h5
python run_scripts/stage4_sites.py --table <patch table> --stem <patch> --resolve-protofilaments
python run_scripts/stage7_register.py --table <patch table>
```

The patch gets its own search with the constraint refitted from the ring
(`refit_constraint_from_ring.py --restrict-polarity`: ψ from the ring's refined angles,
each region's pole fixed when ≥ 90% of its sites agree, θ box from the measured tails).
Membership comes from each detection's own axis point -- on 823 the drawn-region rule would
have misfiled 4.1% of patch detections.

**Estimator: half the even-minus-odd difference of (in − out), per filament.** Not the
parity-signed mean: the in-register patch scores ~0.5 z higher at every site because it
found and placed them, and the signed mean converts that offset times any even/odd count
imbalance into apparent register. Check each filament against the two-dimer-period
control, which must be null. On 823 the register reads 3.6% / 9.9% / 5.2% of score
(t 8.9 / 24.9 / 13.1) on filaments 1, 3, 4 and nothing on filament 2; the known-register
synthetic reads 8.2% (t 20.5); GDP 0.9% (t 1.5).

⚠️ One parity per tube is pooled over the seam (and over the labels' flip at roll 0/360):
filament 2's "nothing" was its two arcs cancelling — read per protofilament it is t 44.
`stage7_seam.py` (SVD_PROCEDURE.md §4) fits the seam and gives every site its own register.


Period-2 alternation of patch score against the axial index from Stage 4, on the refined
sites from Stage 5.

**Run it LAST, and use what the earlier stages established.** By Stage 7 the protofilament
number, the lattice and the polarity are all known, so the patch search is far better posed
than it would have been at the start:

* pick the patch at the **winning N** and the **winning lattice** — no competition needed,
  and no confounding of N with lattice;
* **restrict the constraint to one pole.** Polarity is measured by then, so half the psi
  range is known to be wrong. This does not lower the analytic threshold (that is computed
  from the YAML grid, and the sidecar gates MIP eligibility rather than the search), but it
  stops wrong-pole orientations winning the per-pixel maximum, which matters most for a
  weak template like the patch.

⚠️ **Check the pole per region before restricting, and do not assume noise.** On the GDP
frame region 2's minority-pole sites were plainly weak (z 6.46 against 8.36) and safe to
drop, but region 1's were *not* (7.21 against 7.08) — equal strength at both poles, which a
560 px band containing two tubes of opposite polarity would produce. Region 2 was restricted
and region 1 left at `both`.

⚠️ **Do not tighten theta or psi on this evidence.** Both were checked against the refined
sites and both are already binding, not generous: theta has sd 5.28° with only 58.5% inside
±5°, and its extremes (79.0°, 99.1°) sit on the ±10° box edges; psi's largest deviation from
the drawn tangent is 9.9° against a 10° cone, 99th percentile 9.4°. The distributions are
being clipped by the box, so narrowing it would cut real signal rather than noise.

⚠️ **The patch needs its OWN `match_template` run. Do not score it at the ring's sites.**
That shortcut was tried and validated against the curved synthetic, where a full patch
search reads the register at 2.19×: the shortcut returns **0.06×**. The machinery is not at
fault (the ring control returns 0.70× against its known 0.69×) and neither is the placement
(the placed patch scores mean z 11.94 against the full search's 11.73, i.e. slightly
better). The **index** is: the full patch extraction indexes each detection by its own φ
through `s_corr = s + (lateral_offset/angular_spacing)·φ`, whereas the shortcut scores the
patch at one φ and indexes it by the ring's — score and index come from different
templates, so the parity is scrambled. Run the patch search, extract sites from the PATCH
table with the patch geometry, and test the alternation on those.
`run_scripts/validate_seam_shortcut.py` is the regression test.

### Index CONTINUOUSLY along the curve — this is the default

Measure the axial coordinate as **arc length along a fitted curve**, not as chord length
along a straight axis. `extract_lattice_sites` projects onto a straight `FilamentAxis`, so
on a bent tube its coordinate is chord length and the non-linear part mis-indexes sites —
110 Å over region 1 of the GDP frame, which is 2.7 monomers. Segmenting the tube works
around that but splits the sites into groups that cannot be compared, which is fatal for
the one readout that is short of sites.

Validated against known truth (`run_scripts/validate_continuous_unwrap.py`): on the curved
synthetic the continuous index reads the register at **2.26×** against the segmented
route's 2.19×, recovers the rise to **+0.17%** of truth, and raises occupancy 95% → 100%.
On the GDP frame region 2 goes from three tests of 16–34 sites to **one test of 74 sites at
97% occupancy**.

⚠️ **PIN THE SIGN OF THE ARC.** The arc runs whichever way the path was drawn, which need
not be the direction the filament points. The helical unwrap adds
`(lateral_offset/angular_spacing)·φ`, so a reversed axial coordinate makes that term fight
the lattice instead of removing it — and it fails *almost silently*: on the synthetic the
best-fit period stayed correct at 40.89 Å while the lattice coherence collapsed from
|R| 0.529 to 0.130, destroying the register while every other number still looked
plausible. Pin it against the direction the detections' own orientations imply, exactly as
`fit_filament_axis` does for the straight case.

Three further traps, all found by that validation and all fixed in
`continuous_unwrap_curved.py`:

* **`splprep` pulls endpoints inward.** The fitted curve spanned 1628 px where the data
  spanned 1777 (8.4% short), compressing the arc to a −10.6% rise. Fit x and y against a
  monotone ordering coordinate so the curve spans the data by construction.
* **Bootstrap with orientation-aware peak finding, not a raw score cut.** A raw cut keeps
  many detections per site, biases the lattice phase, and the rise then walks away across
  iterations (40.8 → 36.6 → 33.5).
* **Correct the tilt once.** Dividing by sin θ *and* passing `theta_deg` to
  `estimate_lattice_rise` applies it twice.

**Watch the deviation as the alarm.** A sound continuous index leaves ~2–5 Å rms against
the ±20.5 Å window. Region 1 of the GDP frame gives 10.13 Å at 69% occupancy — that says
the fitted curve does not describe one tube, which is consistent with its band holding two
of opposite polarity.

⚠️ **If you fall back to segments, do not pool them.** `extract_lattice_sites` phases the axial index
on each segment's own bootstrap, so index 0 is a different physical subunit in every
segment. Pooling does not average the alternation, it scrambles it — on the GDP frame,
pooling six segments gave amplitude 0.005 against 0.086–0.108 within regions, measuring the
mixing rather than the lattice. Tying segments to a common phase needs one axis spanning
the whole tube, i.e. the curved axis that does not exist yet.

**Expect this readout to fail before the other three do.** It rests on a ~10% score
difference between α- and β-aligned sites, and segmenting for curvature cuts the sites per
test by 4–5×. On the GDP frame (max z 10.06, 14–34 sites per segment) every segment came
out below chance — patch median 0.48×, ring 0.24× — while polarity, protofilament number
and lattice all resolved cleanly.

- Control: index at the **dimer** rate instead, which puts every site in the same register.
  Any alternation there is an artefact of the method — it must come out ≈1.
- Run the **ring** alongside as the negative control. On real data the ring gives ~3% and
  sits below its own chance level; the patch gives ~10% (1.35 z on 13.3).
- Do **not** centre the patch on the tube axis. It was tried and is worse on every measure:
  axis residual 21.7 Å vs 10.3 Å, |R| 0.39 vs 0.59, 3× more detections.

**Determines:** **seam**. Cost: seconds.

---

## Summary

| stage | template | determines | cost |
|---|---|---|---|
| 0 CTF (PICASSO) | none | — (gate) | seconds |
| 1 annotation | none, sized for patch | filament paths, ψ field | minutes (human) |
| 2 full `match_template` | **RING** | table + maps | ~1.3 h |
| 3 read the table | **RING** | polarity; PF number (harmonic) | seconds |
| 4 curve + sites (from the TABLE) | **RING** | site list; lattice spacing | seconds |
| 5 refine sites | **RING** | sharp angles for phi-dependent readouts | seconds |
| 6 competition | **RING** (+ patch built) | PF number; expanded vs compacted | minutes |
| 6b per segment | **RING** (both lattices, neutral pose) | lattice state + rise along each tube | seconds |
| 7 register | **PATCH** | seam | seconds |

**One full `match_template` for the three geometric readouts, plus a second one with the
PATCH if you want the seam** (Stage 7 — scoring the patch at the ring's sites was tried and
does not work). Everything after it is minutes or
seconds.

**Every readout has two independent lines:** polarity against the drawn tangent; PF number
from the axis-free harmonic *and* the ring competition (monotonic in N = template-mass
artefact, middle-peaked = structure); spacing from between-segment scatter *and* the curve's
measured shortening; seam against its dimer-rate control and the ring.

**One re-run condition:** if Stage 6 says a different N wins by a wide margin, the Stage 2
reference was wrong and the site list is biased — redo Stage 2 once at that N. A wrong-N
ring does still find the tube, but far more weakly (on synth14, z ≥ 7: 10,913 detections for
the right template against 297–882 for the wrong ones).

---

## Fallback: constrained search as the primary detector  ·  **Template: RING**

If Stage 2 finds nothing even after Stage 0 is correct, the tubes may sit between the two
thresholds. A full search of this frame needs **z ≥ 7.62**; a constrained search over ~200
sites × ~2000 orientations needs only **z ≥ 5.57**, because the threshold depends on the
*number* of correlations and nothing else. The observed maxima on the GDP frame were 6.67 /
6.81 / 6.84 — below the full-search threshold, above the constrained one.

Seed a reference stack directly from the Stage 1 drawing: positions sampled along the
centreline, ψ from the local tangent, θ ≈ 90°, and sweep φ over the full 360° (the roll is
the one angle the drawing cannot supply). This is Stage 6's machinery used as the detector
rather than as a follow-up, and it uses the **RING** for the same reasons as Stage 2.
