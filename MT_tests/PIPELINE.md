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
at **2.19× chance**, the ring at **0.69×** — i.e. the ring sits *below* its own chance
level and cannot do this test at any exposure.

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

**Determines already:**
- **Polarity** — `estimate_polarity` uses ψ alone and takes no axis, so bending cannot bias
  it. Report per segment, and check against the drawn tangent.
- **PF number, first line** — `estimate_protofilament_number` is a pure azimuth harmonic on
  φ. Axis-free, and independent of the template competition in Stage 5.

Cost: ~1.3 h on 4 GPUs.

---

## Stage 3 — `refine_template` on the reference peaks  ·  **Template: RING** (the same one)

Polish φ/θ/ψ and defocus within **one coarse cell** at a fine step. `coarse_*_step` means
the step of the `match_template` grid you already ran (ψ 2.5°, θ 3.5°) — the refine fills in
what the coarse search skipped and nothing it already did. It is not a range knob.

**Not optional:** φ grid quantisation previously faked a supertwist slope — 19 distinct φ
values across 82 sites. Refinement at 0.4° removed it.

Known defects in this path, worth avoiding: in the `uniform` branch φ's *range* is taken
from `coarse_psi_step`, while `coarse_phi_step` and `fine_phi_step` are dead; and
`base_grid_method` accepts `"basic"` (raises `ValueError`) but not `"cartesian"`, which is
the branch that would honour `coarse_phi_step`.

**Determines:** sharpened polarity, a much better azimuth harmonic, and the per-peak angles
the axis fit needs. Cost: minutes.

---

## Stage 4 — Curve and site extraction  ·  **Template: RING** (ring detections only)

Project the refined detections onto the drawn curve, segment by arc length, then
`extract_lattice_sites` per segment.

**Why ring:** bend strain is `u(φ) = u₀[1 + Rκ·cos(φ − φ₀)]`, which averages to zero around
the circumference. A ring is therefore unbiased by it and a patch is not. Lattice spacing
must be read from the ring.

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

**Determines:** **lattice spacing**, per segment, shortening-corrected, with the
between-segment scatter as the error bar. Plus the indexed site list Stage 6 needs.
Cost: seconds.

---

## Stage 5 — Competition at those sites  ·  **Template: RING for N and lattice; PATCH built here for Stage 6**

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

---

## Stage 6 — Seam / monomer register  ·  **Template: PATCH** (ring = negative control)

Period-2 alternation of patch score against the axial index from Stage 4.

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
| 2 full `match_template` | **RING** | polarity; PF number (harmonic) | ~1.3 h |
| 3 `refine_template` | **RING** | sharpened angles | minutes |
| 4 curve + sites | **RING** | lattice spacing | seconds |
| 5 competition | **RING** (+ patch built) | PF number; expanded vs compacted | minutes |
| 6 register | **PATCH** | seam | seconds |

**Exactly one full `match_template` per micrograph.** Everything after it is minutes or
seconds.

**Every readout has two independent lines:** polarity against the drawn tangent; PF number
from the axis-free harmonic *and* the ring competition (monotonic in N = template-mass
artefact, middle-peaked = structure); spacing from between-segment scatter *and* the curve's
measured shortening; seam against its dimer-rate control and the ring.

**One re-run condition:** if Stage 5 says a different N wins by a wide margin, the Stage 2
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
the one angle the drawing cannot supply). This is Stage 5's machinery used as the detector
rather than as a follow-up, and it uses the **RING** for the same reasons as Stage 2.
