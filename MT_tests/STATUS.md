# Microtubule 2DTM — what has been tried, and what to do next

Working notes for the microtubule lattice analysis built on top of the constrained
match-template search. Written so the next person (or the next you) does not repeat
the dead ends.

**Short version.** Polarity and lattice spacing are solved. Spacing is now confirmed
twice over by methods sharing almost no assumptions: a direct measurement via top-down
site extraction, and a **template competition against the deposited expanded (6DPU) and
compacted (6DPV) models, which returns expanded at t = 10**. The seam is not yet
located, but it is **less closed than previously recorded** — the statistic used to rule out
the density route had the wrong null.

**Protofilament number now looks like 13, not 14** — competing 12/13/14/15 models gives 13
at 7.6σ by a statistic validated against synthetic ground truth in *both* directions, and
two other routes agree. The dissenting route has a known confound. This overturns the
earlier working assumption; it is not yet settled, and it sits against the chemistry, since
GMPCPP favours 14. See "Protofilament number" below.

**Curved tubes now have a tested path.** A synthetic bent microtubule with known truth
recovers all four readouts — polarity to 0.6°, protofilament number exactly, rise to
0.01%, and the monomer register at 2.19x chance with a patch template. The straight-axis
code degrades predictably rather than failing: the rise bias follows −L²/(40R²), measured.
See "Curved microtubules" below.

What is left needs **different models rather than different analysis** — and that recipe
is now built and validated. `models/build_mt_templates.py` produces matched templates
from any pair of depositions, and the 2×2 design *measures* the model-quality confound
instead of assuming it away. Point it at 13- against 14-protofilament models next. For
the seam, the analysis now knows where the peaks are, which makes a very large model
affordable to search, so it is worth one more attempt.

---

## Methods

Written to be lifted into a methods section. Every parameter below is the one actually
used; the code is `src/leopard_em/analysis/{correlation_peaks,filament_lattice}.py`.

### 1. Search and detection recovery

A constrained `match_template` search is run with a filament sidecar that restricts which
(pixel, orientation) tuples may win the maximum-intensity projection: `psi` and `theta` to
a cone about the drawn filament direction, `phi` free over 0–360°. Statistics are taken
over the full orientation grid, so the normalisation is unaffected by the constraint.

The analysis reads the **sparse correlation table**, not the per-pixel maps. The maps keep
one hypothesis per pixel, so the near and far walls of a tube annihilate each other and
every sub-maximal orientation is discarded — but orientation *is* the measurement here.
The table stores every detection above a raw cross-correlation threshold (default 6.5),
and `detections_from_hdf5` streams and thresholds it during the read.

Each detection carries `(x, y, phi, theta, psi, defocus)` and a z-score

> z = (cc − correlation_mean) / correlation_variance

where `correlation_variance` holds a **standard deviation** despite its name; dividing by
it directly reproduces the stored `scaled_mip` exactly.

### 2. Template calibration

`TemplateLatticeGeometry.from_pdb` measures, from chain centroids: protofilament number,
subunit radius, monomer rise, lateral axial offset, and — critically — the vector from the
box centre to the tube axis. A ring template is axis-centred (0.2 Å); a patch template is
offset by 86.8 Å, and that offset rotates with `phi`.

### 3. Axis recovery

For each detection the axis offset `p` is rotated into the image and subtracted:

> image_offset = (Rᵀp)[:2],  R = R_ZYZ(phi, theta, psi)

The slice extractor rotates the *sampling grid*, so the object is rotated by Rᵀ; this was
verified numerically to better than 0.1 px. Every detection — any protofilament, either
wall — therefore becomes an independent prediction of the same axis line, and the scatter
of those predictions about one line is a free consistency check.

A score-weighted total-least-squares line is fitted through them. **The sign of the fitted
direction must be pinned** (PCA fixes an axis only up to sign, and the helical unwrap
depends on it), using the direction implied by the detections' own orientations.

### 4. Top-down site extraction

Global thresholding forces a bad trade: cut high and real sites are lost, cut low and
noise corrupts the axis fit, which biases everything measured along it. Instead:

1. **Bootstrap.** Orientation-aware peak finding at a *high* cut (z > 8 here, giving 160
   peaks on the full frame). Two detections suppress one another only if within **12 px
   *and* 15°** — so hits overlapping in position but disagreeing in angle both survive,
   which is what keeps the two walls of the tube apart. These 160 fit the axis.
2. **Unwrap.** Each detection's axis point is projected onto the axis to give `s`, then
   corrected for the helix:

   > s_corr = s + (lateral_offset / angular_spacing) · phi

   after which every protofilament falls on a common lattice of period `rise`.
3. **Assign.** `axial_index = round((s_corr − phase) / rise)`, where `phase` is the
   score-weighted circular mean of the bootstrap peaks. Each detection is binned to its
   nearest lattice site; the **highest-scoring detection in each bin wins**, however weak.
4. **Refit** the rise from the winners and repeat (3 iterations; convergence is immediate).

Note this is a **one-dimensional assignment**, not a five-dimensional search: the whole
`(x, y, phi, theta, psi)` tuple collapses to a single axial coordinate. The effective
window is therefore **±rise/2 = ±20.9 Å axially**, with **no transverse and no angular
gate** — see the caveat in "Things that will trip you up".

**Anti-circularity check.** The window is a full half-repeat, so positions are free to land
anywhere; `axial_deviation_angstrom` records where they actually landed. Observed **2.92 Å
rms against ±20.9 Å**, which says the data and not the model set the positions. Re-run from
different bootstrap thresholds and starting repeats; the answer should not move (spread
0.008 Å across configurations, against 0.083 Å for threshold-based picking).

### 5. Lattice spacing

**Direct.** Index the site positions against integer lattice indices and fit position
against index. Fitting a line through many indexed positions beats measuring gaps: the
slope's standard error falls as **M^−1.5** in repeats spanned, not M^−0.5. A global scan
over ±10% precedes local refinement — without it the estimator returns its own initial
guess, because aliases sit ~rise²/span apart and local refinement cannot escape one.

Out-of-plane tilt shortens the projected repeat by `sin(theta)`; pass per-peak `theta` to
divide it out (+0.09% here at mean theta 87.8°, but 1.5% at theta 80°).

Report **both** the absolute repeat and the ratio to the template's own repeat, the latter
measured by autocorrelation of the *rendered* template so estimator bias partly cancels and
no pixel-size assumption enters. A 2DTM spacing is fundamentally (true spacing) × (assumed
pixel size); the ratio is the form in which a later recalibration is one multiplication.

**By competition.** Build templates from the deposited expanded and compacted models and
ask which fits better. This needs no estimator at all — no aliasing, no threshold
sensitivity, no sub-harmonic degeneracy. Two depositions are never equally good, so build
the full **2 × 2 of {atom source} × {lattice}**, which splits into orthogonal contrasts:

> lattice effect = mean over atom source of (expanded − compacted)
> quality effect = mean over lattice of (model A − model B)

The quality term *measures* the confound instead of assuming it away. Compare **paired per
site** — the same physical sites across runs — which removes defocus, ice and near/far wall
variation. Result: lattice +0.446 ± 0.045 z (t = 9.96, 54/60 sites); quality +0.017 ± 0.036
(t = 0.47) — indistinguishable from zero.

### 6. Protofilament number

Templates of different N are built from **the same atoms**, so unlike the spacing case
there is no model-quality confound and no 2 × 2 is needed. Three quantities follow from N
once the lateral contact is held fixed:

> radius ∝ N  (8.45 Å per protofilament) · twist = 360/N · lateral rise = n_start · u / N

Radius is the lever, and it is large. The lateral-rise relation is validated, not assumed:
it reproduces the deposited helical operators of 6DPU and 6DPV to 0.06%.

**Use max z, not a paired mean.** On synthetic data that is definitively 13-PF the paired
contrast 13 − 14 is t = 0.80 (55/127 sites) — a coin flip — because averaging over ~130
mostly-marginal sites dilutes the signal. Max z reads the single best-registered site,
where a correct radius pays in full. Max z is normally unfair between templates because it
tracks mass; that objection does not apply when all templates share their atoms, and the
controls below confirm it empirically.

**Validate with synthetic ground truth, in both directions.** Build a micrograph containing
a microtubule of known N — tile the template's own projection along the axis at an integer
number of repeats, apply the micrograph's CTF and B-factor, and add noise made by phase
randomising the real image so the power spectrum, and hence the whitening filter, is
realistic. Place it on the real filament's line so the same sidecar applies. **Calibrate by
correlating against the real micrograph at a known peak** — sign and magnitude must match
(real +0.0275, synthetic +0.0280) — and *render the image and look at it* before running
any search.

Running the identical competition against a known 13-PF and a known 14-PF tube recovers 13
and 14 respectively, so the method is not biased toward whichever template built the truth.

---

## The data

| | |
|---|---|
| micrograph | `Frames/..._0005_X-1Y+1-0_sum_DW.mrc`, 5760 × 4092 |
| crop | `..._cropped_4.mrc`, 1440 × 1023, a genuine crop at offset **(y=1535, x=2160)** (verified, correlation 1.0000 — not a binning) |
| pixel size | **0.9194 Å, externally calibrated.** The MRC header's 0.9432 Å is the nominal value and should be ignored |
| CTF | defocus 8390.5 / 8035.0 Å, astigmatism −0.865° |
| templates | `6dpu_4_patches`, `6dpu_2rings`, `6dpu_4rings`, simulated at 0.9194 Å, `b_factor_scaling=0.5`, dose 0–50 e⁻/Å², k3 MTF |
| competition set | `{6dpu,6dpv}_atoms_{6dpu,6dpv}_lattice_2rings[_flatB]`, built by `models/build_mt_templates.py` from PDB 6DPU (GMPCPP, 3.1 Å, EMD-7973) and 6DPV (GDP, 3.3 Å, EMD-7974) |

### Template properties (all measured, not assumed)

| property | value |
|---|---|
| protofilaments | **14** (not 13 — worth knowing, it was assumed to be 13 early on) |
| subunit radius | 118.8 Å |
| dimer repeat | 83.9 Å; monomer 41.91 Å (by autocorrelation) |
| lateral offset | −9.0 Å per protofilament → −126 Å per turn = 3 monomers = **1.5 dimers**, which is what forces a seam |
| α vs β | 41.9% sequence identity (textbook), 28 chains each in the 2-ring model |
| axis offset | patch **86.8 Å** off the box centre; ring templates axis-centred |

The models were built from a 12-mer with helical symmetry applied. That explains why
the α→β step (40.99 ± 0.01 Å) and β→α step (42.97 ± 0.01 Å) are so precisely
reproducible: the alternation is real and measured, just replicated by symmetry.

---

## What works

Full micrograph, 2-ring template, via `extract_lattice_sites`:

| readout | result |
|---|---|
| **polarity** | ψ 279.4°, **2633 : 1** — decisive |
| **lattice spacing** | **41.86 Å** monomer = 0.9988 × the template's own repeat |
| sites | **82 of 83 occupied (99%)**, weakest site z = 6.12 |
| axis fit | residual 6–8 Å over a ~3200 Å track |
| protofilament number | 14, but margin only **1.01×** — *not* resolved |

**The lattice matches the 6DPU model to ~0.1%.** The two estimators (site-based
41.864 Å, peak-based 41.89 Å) differ by 0.03 Å; treat that gap, not either formal
error, as the honest systematic floor.

An earlier "−0.5 Å compaction at ~2σ", measured on the 387 Å crop, did **not** survive
the full 3255 Å track — it was a small-sample fluctuation. Do not trust spacing from
short segments *when fitting a rise*. See the next section for why competition is exempt.

### Lattice spacing, confirmed independently by template competition

The deposited expanded (6DPU/GMPCPP) and compacted (6DPV/GDP) models were built into
matched templates and competed against each other on the crop. **The answer is expanded,
and it agrees with the direct measurement by a route that fits nothing, uses no axis, and
has no estimator bias.**

| contrast | effect | t | sites |
|---|---|---|---|
| **lattice** expanded − compacted | **+0.446 ± 0.045 z** | **+9.96** | **54/60 positive** |
| quality 6dpu atoms − 6dpv atoms | +0.017 ± 0.036 z | +0.47 | 34/60 |
| interaction | +0.091 ± 0.030 z | +2.97 | 44/60 |

**The 2×2 design is the point.** Two depositions are never equally good, so competing
them naively measures model quality as much as structure. Building all four combinations
of {atom source} × {lattice} splits those into orthogonal contrasts, and the
model-quality term came out at +0.017 ± 0.036 z — indistinguishable from zero, 34/60
sites, a coin flip. The confound is *measured absent*, not argued away. The lattice
effect is 26× larger.

The small but real interaction (t = 3.0) says 6DPU's atoms discriminate the lattice
slightly better than 6DPV's, which is expected: 6DPU atoms on the 6DPU lattice is the one
fully self-consistent native combination.

**The crop is legitimate here even though it is too short for the estimator.** A wrong
lattice is stretched 2.72% axially, so it misregisters ±4.0 Å *within the template
itself*; the per-site signal does not depend on track length at all. Track length only
buys paired samples. That makes this a 9-minute-per-cell test instead of 52.

Equalisation that made it fair (all in `build_mt_templates.py`):

- trimmed to **common atoms**, not common residues — per-residue trimming still left 84
  protein atoms unequal, because both models build different numbers of side-chain atoms
  inside shared residues. Protein atom counts are now identical; the residual 140-atom
  difference is the GMPCPP+Mg vs GDP ligand, which is real biology and stays.
- centroids recomputed **after** trimming, else 6DPU's monomers sat 0.045 Å off their
  intended heights — a one-sided error, since 6DPV's residue set *is* the common set.
- `_flatB` variants set every B to 31.18. `ttsim3d` reads per-atom B and scales it, so the
  native 2.5 Å² gap is a real sharpness advantage to 6DPU.
- the **deposited** helical operators, which close the 14₃ lattice (14 × 8.99 = 3 monomers
  of 6DPU; 14 × 8.754 = 3 of 6DPV). The lattice measured directly from the 12-mer is
  9.47 Å and does **not** close — using it builds a broken tube.

Measured template geometry, for reference:

| | 6DPU (GMPCPP, expanded) | 6DPV (GDP, compacted) |
|---|---|---|
| dimer repeat | **83.958 ± 0.008 Å** | **81.671 ± 0.007 Å** |
| α→β / β→α | 42.97 / 40.99 (alternation **1.97 Å**) | 41.15 / 40.52 (alternation **0.64 Å**) |
| radius | 118.36 Å | 118.92 Å |

**Two cautions.**

1. **The result is degenerate with the pixel size.** The repeat ratio is 1.02800; the
   nominal/calibrated pixel ratio (0.9432 / 0.9194) is 1.02589. They agree to **0.21%**,
   so this test would have said "compacted" with equal confidence on the header pixel
   size. What it gives is a *consistency* check: the calibrated 0.9194 Å and the known
   GMPCPP state agree, which they would not if the calibration were off by that 2.6%.
2. **Do not use radius as a discriminator between these two.** 6DPU and 6DPV differ by
   0.56 Å, and rebuilding the same ring from each of the three deposited protofilaments
   spans 117.93 / 118.18 / 118.47 Å. Signal and construction noise are the same size.
   Only the repeat is safely above it.

The effect is smaller than intuition suggests — 0.45 z against a site mean of ~7.5, about
6% — because the search re-centres and re-picks defocus, recovering much of a wrong
template's loss, and because z normalises each template by its own noise. Significance
comes from consistency across sites, not from a large per-site drop. Peak counts move the
same way and were deliberately *not* used as the statistic: 225 and 198 for the expanded
templates, 155 and 149 for the compacted.

### Use `extract_lattice_sites`, not a global threshold

**This is the recommended route and supersedes threshold-based picking.** Fit a rough
lattice from the strongest peaks only, then take the best detection within half a
repeat of every site the model predicts, however weak. Noise away from the lattice is
never looked at, so purity and completeness stop competing:

| | global threshold (z > 8) | **top-down** |
|---|---|---|
| sites occupied | 71% | **99%** |
| sharpness \|R\| | 0.93 | 0.93 |
| weakest site kept | z 8.0 | **z 6.12** |
| spread across configurations | 0.083 Å | **0.008 Å** |

It is not circular, and the check is built in: the search window is a full half-repeat,
so positions are free to land anywhere, and `axial_deviation_angstrom` records where
they actually landed — 2.9 Å rms against a 20.9 Å window. The driver prints this and
warns when it fails. Always read that line; on the short cropped track it correctly
reports `CHECK: positions may be pinned to the model`.

### Why a global threshold fails: pick high and lose sites, pick low and bias the fit

The fitted rise drifts systematically with the score cut, converging only once the
lattice is well resolved:

| z cut | peaks | \|R\| | rise (Å) | formal ± | axis residual |
|---|---|---|---|---|---|
| 5.5 | 7956 | 0.17 | 41.814 | 0.005 | **47.6 Å** |
| 7.0 | 889 | 0.77 | 41.847 | 0.008 | 10.1 Å |
| 8.0 | 160 | 0.93 | 41.879 | 0.010 | 7.7 Å |
| 9.0 | 24 | 0.99 | 41.897 | 0.007 | 6.6 Å |

Weak peaks bias the rise **low**, because they corrupt the axis fit (47.6 Å residual at
z = 5.5) and everything downstream is measured along that axis. The ±0.02 Å quoted
above is the spread across well-resolved cuts, not the formal fit error.

**The formal error is actively misleading here**: ±0.005 Å at z = 5.5 while the value
is 0.08 Å wrong. Adding weak peaks shrinks the formal error while making the answer
worse. Judge convergence with `lattice_sharpness`, not with the error bar. This whole
trade is what the top-down method removes.

Tilt foreshortening is corrected (mean θ = 87.8°, a +0.09% effect here, but 1.5% at
θ = 80° so it is not always negligible).

---

## What does not work, and why

### Protofilament number — improved, but still not decisive

On *threshold-picked* peaks the azimuth harmonic gives 13, 14 or 15 depending on
template and cut, always with a margin near 1.0–1.5×. Restricting instead to detections
that sit **on the fitted lattice** helps a lot, and the answer becomes correct:

| z cut | n | 11 | 12 | 13 | **14** | 15 | 16 | best |
|---|---|---|---|---|---|---|---|---|
| 5.0 | 3.7 M | 0.022 | 0.051 | 0.083 | **0.289** | 0.036 | 0.040 | **14 (3.49×)** |
| 6.0 | 383 k | 0.051 | 0.086 | 0.178 | **0.301** | 0.016 | 0.037 | 14 (1.69×) |
| 7.0 | 32 k | 0.101 | 0.170 | **0.366** | 0.353 | 0.294 | 0.109 | 13 (1.04×) |

**This estimator wants the opposite of every other one here: weak detections, not
strong.** Rise is a position measurement, so it needs precision and weak peaks corrupt
the axis. Protofilament number is a *census* — it needs the whole circumference
sampled and does not care about per-peak precision, and a strict cut biases the sample
towards the near wall and the best-scoring protofilaments. The on-lattice filter
supplies the purity that the threshold otherwise would.

Still not something to rely on. The margin collapses to 1.04× by z > 7, so it holds
only in the low-z on-lattice regime; the patch template gives 13 (wrong), and at z > 5
its strongest harmonic is N = 2, which is the near/far wall asymmetry rather than a
protofilament count. And the ring template is itself 14-fold pseudo-symmetric, so some
of that 14-harmonic may be reading the template rather than the specimen — a confound
one dataset cannot rule out.

Also note the estimator must be given a **restricted candidate range**. Unrestricted over
orders 2–20 a low-order N = 3 harmonic dominates, and the reported margin is meaningless;
the table above is over 11–16.

#### Competing 13 / 14 / 15 protofilament models

`models/build_mt_templates.py --protofilaments 13 14 15` builds these from the **same
6DPU atoms**, so unlike the expanded/compacted case there is no model-quality confound at
all — the templates differ *only* in N. Three quantities follow from N once the lateral
contact is held fixed:

| N | radius | twist | lateral rise |
|---|---|---|---|
| 12 | 101.4 Å | −30.000° | 10.495 Å |
| 13 | 109.9 Å | −27.692° | 9.687 Å |
| 14 | 118.3 Å | −25.714° | 8.995 Å |
| 15 | 126.8 Å | −24.000° | 8.396 Å |

**Radius scales with N** — 8.45 Å per protofilament — because the lateral spacing is a
property of the tubulin interface, not of N: a 13-mer tube is narrower, not more loosely
packed. Keeping R at 118.8 Å and fitting 13 protofilaments would space them 57.4 Å apart
and they would not touch. The lateral rise follows from `n_start × monomer / N`, which is
**not assumed**: it reproduces the deposited helical operators of both 6DPU (8.9955
predicted vs 8.99) and 6DPV (8.7505 vs 8.754), to 0.06%.

The one residual confound is template mass, and it is self-diagnosing: mass rises with N,
so a **middle-peaked** result cannot be explained by it, while a monotonic one can.

**On the real crop the answer is not usable:**

| N | peaks | max z | mean z at sites | paired vs 13 |
|---|---|---|---|---|
| 12 | 194 | 9.28 | 7.039 | t = 0.39, 40/89 |
| **13** | 203 | **12.93** | **7.143** | — |
| 14 | 216 | 9.04 | 6.703 | t = 1.91, **42/89** |
| 15 | 105 | 8.25 | 6.269 | t = 4.65, 53/89 |

13 wins on the mean but **42 of 89 sites favour it, so 47 favour 14** — the per-site
majority points the other way and the mean is carried by a few large differences. 12 and
13 are a dead heat. Treat this as inconclusive.

#### The synthetic positive control — and it passes

`run_scripts/make_synthetic_microtubule.py` builds a micrograph containing a microtubule
of **known** protofilament number, at the same pixel size, defocus, SNR and position as
the real one, so the same config and constraint sidecar apply unchanged. Running the
identical competition against ground truth:

| N | peaks | max z | mean z | paired vs 14 |
|---|---|---|---|---|
| 12 | 104 | 7.68 | 6.067 | t = 3.47, 65/109 |
| 13 | 147 | 7.87 | 6.128 | t = 2.38, 54/109 |
| **14 (truth)** | **158** | **10.09** | **6.509** | — |
| 15 | 123 | 7.89 | 5.923 | t = 3.72, 60/109 |

**The method recovers the truth, and the curve is middle-peaked** (12 < 13 < 14 > 15),
which rules out both feared artefacts — not template mass, not a smaller-is-better bias.

A second control was then built from a **13-PF** template, to close the one confound the
first cannot: that the method favours whichever template made the ground truth.

| N | peaks | max z | mean z | paired vs 13 |
|---|---|---|---|---|
| 12 | 193 | 8.24 | 6.234 | t = 0.46, 56/127 |
| **13 (truth)** | **206** | **10.33** | **6.304** | — |
| 14 | 182 | 8.24 | 6.191 | t = 0.80, 55/127 |
| 15 | 89 | 7.42 | 5.711 | t = 5.68, 86/127 |

**The control is symmetric**: 13-PF truth returns 13, 14-PF truth returns 14. Template
competition is validated in both directions and is not biased toward its own construction.

#### Use max z, not the paired mean — the paired mean is nearly blind to ±1 protofilament

On data that is *definitively* 13-PF, the paired contrast 13 − 14 is **t = 0.80, 55/127
sites** — a coin flip. The paired mean only separates 15 (t = 5.68). It averages 127 sites,
most of them marginal, and dilutes the signal.

**Max z is the statistic that works here**, and both controls validate it:

| dataset | winner | max z | other three | margin | significance |
|---|---|---|---|---|---|
| control, truth = 13 | **13** ✓ | 10.33 | 7.97 ± 0.47 | +2.36 | 5.0σ |
| control, truth = 14 | **14** ✓ | 10.09 | 7.81 ± 0.12 | +2.28 | 19.1σ |
| **real crop** | **13** | **12.93** | 8.86 ± 0.54 | **+4.07** | **7.6σ** |

The signature is consistent — the true N stands about 2 z above every wrong answer — and on
the real data the margin is *larger* than in either control. Max z reads the single
best-registered site, where a correct radius pays off in full; averaging dilutes it.

Max z is normally a poor comparator between templates because it tracks template mass. That
objection does not apply here: all four templates come from the same 6DPU atoms, and the
two controls demonstrate empirically that it recovers the truth. **This was a real error on
my part** — the paired-mean machinery was built to avoid a bias that this particular
comparison does not have, and it cost a day of treating the answer as unresolved.

#### Verdict: probably 13, with a chemistry caveat

Re-read in this light, three independent routes agree and the dissenter has a known flaw:

| route | says | note |
|---|---|---|
| template competition, max z | **13** | validated in both directions, 7.6σ |
| patch-template harmonic | **13** | previously written off as "wrong" — on the assumption the answer was 14 |
| supertwist / moiré | **13** | total azimuth excursion 3–5° where 14 needs +20.5°; excludes 12/14/15 |
| helical unwrap sharpness | **13** | corroborates the above — same (s, φ) information, not a separate vote |
| model-predicted site occupancy | (13) | **does not count** — raw occupancy favours 13 but the null absorbs it, and the run contradicts its own control |
| ring-template harmonic | 14 (3.49×) | the ring is itself 14-fold pseudo-symmetric — the confound flagged from the start |

The "no supertwist signal" result was first recorded as a failed measurement. Measured
properly — on the clean lattice sites, with refined angles — it is a **bound that excludes
12, 14 and 15**, and 13 is the one lattice whose protofilaments run parallel to the axis.
See "The moiré / supertwist route" below for how far that can be pushed.

**Against this:** GMPCPP strongly favours 14-protofilament microtubules, and 6DPU itself is
a 14-PF GMPCPP structure. 13-PF occurs but would be the less common outcome here. Do not
treat this as settled on one filament in one micrograph.

**The clean test** is now sharp and falsifiable rather than exploratory: if this is 13-PF
there is **no supertwist at any length**. A longer microtubule from the same grid separates
the hypotheses outright — 14-PF gives 0.63°/100 Å of φ drift and 204 nm moiré fringes,
13-PF gives exactly zero of both.

#### Corroboration: which N's helical unwrap best stacks the lattice

A cheap check needing no new search. The unwrap `s_corr = s + (lateral/spacing)·φ` depends
on N, so ask which N stacks the detections onto the sharpest lattice.

**First, a trap.** Under the closure model `lateral = n_start·u/N` and `spacing = 360/N`,
the ratio is `n_start·u/360` — **the N cancels exactly** and the test has no power. Use the
physically correct model instead: the lateral contact δ₀ is a fixed property of the tubulin
interface, so `lateral/spacing = δ₀·N/360`, which does depend on N. (The two coincide at
N = 13, since δ₀ is *defined* by 13₃ closure.)

| N | ring: dev rms | ring \|R\| | patch: dev rms | patch \|R\| |
|---|---|---|---|---|
| 12 | 3.94 Å | 0.881 | **3.18 Å** | 0.907 |
| **13** | **2.91 Å** | **0.935** | 3.41 Å | **0.918** |
| 14 | 3.34 Å | 0.898 | 4.48 Å | 0.834 |
| 15 | 4.86 Å | 0.781 | 7.05 Å | 0.229 |
| 16 | 7.20 Å | 0.598 | 9.48 Å | 0.540 |

Ring data is **middle-peaked on both metrics**. Patch data peaks at 13 on sharpness, with 12
marginally tighter on deviation; both agree that ≥14 degrades badly.

**It passes its null.** If φ were noise, the unwrap would merely inject noise proportional
to `|δ₀N/360|`, which grows with N, so sharpness would fall *monotonically from N = 12*. It
does not — both datasets peak at 13. So φ carries real helical information.

**But do not count this as an independent vote.** It interrogates the same (s, φ) coupling
as the supertwist scan, shares the axis fit and detections with the site extraction, and
resolves "≤13" much better than it resolves 13 from 12 — 12 and 13 differ by only 8% in
unwrap coefficient. Treat it as corroboration. The one route with a genuinely independent
physical handle is template competition, because it measures **radius**.

#### Predicting sites per model and counting them — built, run, inconclusive

`score_lattice_model` (library) and `run_scripts/run_lattice_model_competition.py`. The
idea: hold the detections **completely fixed** and vary only the model, predicting where
every subunit should appear in (x, y) *and* azimuth and asking whether a detection is
there. No new search, so no template confound can enter. Sites predicted but empty are
counted, which makes occupancy a real statistic rather than the near tautology it is when
sites are defined by the detections themselves.

**Result on the full-frame patch table, 2.97 M detections at z > 7.5, 24 null draws:**

| model | occupancy | null | excess |
|---|---|---|---|
| extended N=12 | 47.2% | 46.7% ± 1.7% | +0.3σ |
| **extended N=13** | **61.1%** | 48.4% ± 11.5% | +1.1σ |
| extended N=14 | 44.1% | 44.3% ± 3.4% | −0.0σ |
| extended N=15 | 48.9% | 42.9% ± 9.2% | +0.7σ |
| compacted N=13 | 56.4% | 40.0% ± 9.9% | **+1.7σ** |

Raw occupancy favours **N = 13 emphatically** — 61.1% against 44–49%, a 12–17 point gap
pointing the same way as everything else. But **the null absorbs it**: nothing clears 2σ.

**Do not treat this as evidence**, for three reasons:

1. **The run contradicts itself.** The highest-excess model is *compacted* N = 13 while
   the repeat control says *extended*. The two headline statistics of one run disagree.
2. **The control passes only weakly.** Extended beats compacted 4 of 4 (p ≈ 0.06 on
   direction alone), but by 4.1 / 4.7 / **0.3** / 1.4 points — a tie at N = 14.
3. **An unexplained null anomaly.** N = 12 and 14 have tight nulls (±1.7%, ±3.4%) while
   13 and 15 have loose ones (±11.5%, ±9.2%). This persisted from 8 draws to 24, so it is
   real and not small-sample noise. **Nobody should read this test until that is
   understood** — it is exactly the term the ranking statistic divides by.

**Why it was always going to be weak**, established before running it: under closure the
unwrap coefficient is N-independent, so *every* model places its sites on the same
continuous helix and different N merely samples it at different azimuths. A site inside
the angular window is automatically inside the axial window (mismatch ≤ 3.4 Å), so the
axial test contributes nothing to N. The protofilament axis rests **entirely** on whether
the matched azimuths form a coset of a 360/N grid — a windowed, threshold-gated
re-derivation of the azimuth harmonic, not new physics. The repeat axis is the strong one,
and it is the control.

**Cost and traps, if anyone revisits it.** 2 h 20 min for the run above. Matching cost
scales with *detection density inside the window*, not detection count, so lowering the
threshold is far more expensive than it looks — nearer quadratic than linear. The
transverse band prefilter (cut to detections near the axis before building the tree) is
the optimisation that was designed and never implemented; it attacks the term that
actually dominates. Four bugs were found and fixed by the tests while building this, all
recorded in `tests/analysis/test_filament_lattice.py::TestScoreLatticeModel`:

- the closure formula used the *dimer* repeat where it needs the monomer (3 × 84/14 = 18,
  but the real lateral offset is 9) — now derived from the template's measured
  `axial_shift_per_turn`, which needs no assumption about subunits per repeat;
- **azimuth wrapped to ±180°**, flipping the helical term for every protofilament past
  halfway and displacing exactly half the sites. The offset must accumulate monotonically
  the whole way round, because going once round genuinely rises by the turn shift;
- sites predicted beyond the filament's ends, depressing occupancy by a model-dependent
  amount;
- the phase scan sat at the edge of a plateau, leaving a constant 9.0 Å in every
  deviation — the window is wider than the step, so many offsets tie.

### The moiré / supertwist route — now a real bound

A 13₃ lattice closes with protofilaments *parallel* to the axis, so it has no supertwist
and no moiré fringes; every other N must skew. From the measured monomer rise the natural
lateral offset is 3 × 41.979 / 13 = **9.687 Å**, and the skew follows:

| N | skew | moiré period | φ drift over the 3255 Å track |
|---|---|---|---|
| 12 | −0.871° | 175 nm | −28.0° |
| **13** | **0** | **none** | **0°** |
| 14 | +0.747° | 204 nm | +20.5° |
| 15 | +1.393° | 109 nm | +35.8° |

This is what an experienced eye reads: fringe spacing, and for 13 their absence. Fringes
are the beat between the near and far walls, whose protofilaments cross at twice the skew;
at zero skew the two gratings stay in register the whole length and there is nothing to
beat against.

**The 2DTM version needs no fringes.** φ is already measured per detection, so the same
information is available as a slope, dφ/ds. That makes it structurally different from
template competition: a slope cannot be biased by template size, and it is *signed*, so 12
and 14 differ in direction and not merely in magnitude.

#### Attempt 1 — harmonic scan over all detections (weak)

Scanning the drift freely on 1.5 M patch detections, the best drift for each harmonic
order is −0.36 / −0.22 / +0.12 / +0.14 °/100 Å for N = 12/13/14/15, none matching its
prediction, with gains of only 1.03–1.29× over zero drift. This yields no error bar and no
way to tell "the drift is zero" from "there is no drift information here".

#### Attempt 2 — the clean lattice sites

The occupied sites of the fitted lattice are a pure subset: **82 sites, 99% of the 83
predicted, rise 41.850 Å, 3.41 Å rms deviation, over a 3261 Å track**. Two different
measurements can be made from them, and they behave oppositely.

**The azimuth census gets worse, and this is structural.** Cleaning keeps the single best
detection per axial index, which is systematically the near wall and the best-scoring
protofilaments, so circumferential coverage collapses to **9 of 24 bins — an occupied arc
of 255° with a 105° hole**. A census resolves orders no finer than 360/arc ≈ 1.4, so 13
and 14 are formally inseparable. A(13) = 0.843 looks like a win until the rest of the
profile is read: A(2) = 0.583, A(11) = 0.526, A(27) = 0.663. That is clustering, not
quantisation. **Do not use a φ-permutation as the null** — the harmonic amplitude depends
only on the multiset of angles, so permuting leaves it unchanged and the "floor" it
returns is meaningless.

**The drift measurement gets better,** because it needs φ tracked *along* the axis, not a
census *around* it — restricted azimuth coverage is no obstacle at all. Cluster the clean
sites by azimuth into individual protofilaments and fit φ against the raw along-axis
coordinate within each. Use the **raw** coordinate, never the unwrapped one: the unwrap
folds φ into s by construction and would manufacture a slope.

#### The quantisation trap

On grid angles this gives 0.160 ± 0.033 °/100 Å, which would be 13 at 4.8σ and everything
else at ≥14σ. **The error bar is fake.** φ takes only 19 distinct values across the 82
sites and **1–3 within any one protofilament** (20 sites at 167.77° and one at 171.26°;
another cluster stepping once, 111.84° → 115.34°). The grid step is 3.495°, so the fit
reads out *where a single grid step falls*. Three symptoms give it away:

- both halves of the track return a slope of exactly 0.000;
- 3 of 4 clusters have a significant quadratic term — a staircase is not a line;
- cleaning at min_z 6 instead of 7 moves the answer to 0.081 ± 0.138.

An in-plane axis rotation of ±0.1° changes nothing, because φ is an absolute Euler angle
rather than one measured against the axis — so the drift is at least not an axis-fit
artefact. Note also that the ring and patch axis fits come out **antiparallel**, which
flips the sign of any measured slope; sign is a convention here until polarity is tied to
the axis direction.

#### Attempt 3 — refined angles, which is the measurement

The limit is angular sampling, not counts or purity, so the fix is refinement rather than
more data. `run_scripts/refine_clean_sites.py` builds a particle stack from the clean
sites and runs `refine_template` at a **0.4° fine φ step**. φ goes from 19 distinct values
to **59**, from 1–3 per protofilament to **9–12**, and mean z rises 13.26 → 15.51.

| ⟨φ⟩ | n | span | distinct φ | slope (°/100 Å) | residual |
|---|---|---|---|---|---|
| 113.6° | 16 | 2092 Å | 11 | +0.192 ± 0.025 | 0.45° |
| 140.2° | 15 | 3223 Å | 12 | +0.119 ± 0.022 | 0.69° |
| 167.9° | 21 | 1502 Å | 9 | +0.055 ± 0.044 | 0.69° |
| 336.2° | 10 | 3231 Å | 10 | +0.161 ± 0.070 | 2.15° |

Combined **0.140 ± 0.030 °/100 Å**.

**But it is not a supertwist, because it is not constant.** A supertwist must be a constant
slope. Splitting each protofilament in half gives first-half slopes of −0.190 to +0.173
against second-half −0.667 to +0.245, and the quadratic term is significant in 2 of 4
clusters. That is the signature of a gently bent tube, which is what a real microtubule on
a grid is, not of a twist.

#### What survives — the bound

Whatever the local wobble, the **total azimuth excursion along each protofilament is 3–5°
over the track**, where the alternatives require a monotonic +20.5° (N = 14), −28.0°
(N = 12) or +35.8° (N = 15). So the moiré route now **excludes 12, 14 and 15, and is
consistent with 13** — as a limit rather than as a detection. That is a genuine
strengthening of the earlier "no drift found", which had no error bar at all.

#### What it still cannot do

- **No ground-truth validation is available.** `run_scripts/make_synthetic_microtubule.py`
  tiles a single template projection at a fixed 2 × 83.958 Å repeat, so both the 13-truth
  and the 14-truth controls are exactly periodic and contain **zero supertwist by
  construction, whatever N**. Validating this estimator needs a synthetic built from a
  genuinely helical long model, not from a tiled projection.
- **Sign is not yet meaningful** (see above), so 12 and 14 are excluded by magnitude only.
- The bound rests on **four protofilaments of one filament in one micrograph**.

The clean test is unchanged and remains cheap: a 14-PF tube must show +0.63°/100 Å and
204 nm fringes at *any* length, a 13-PF tube shows zero at any length. **≥600 nm of
continuous tube settles it outright.**

### Measuring the diameter directly from the image — dead

17 Å per protofilament, 18.5 px, no template and no correlation, so no size bias is
possible. It does not work: on **synthetic data where the answer is known to be 236.6 Å it
returns 184 Å**. On the real crop the transverse profile is a single broad blob with no
two-wall structure and is badly asymmetric (half-maximum at −149.5 and +116.0 Å), so
something adjacent contaminates the background. Do not revive this without a proper local
background subtraction and neighbour masking — and re-check it against the synthetic,
which is what showed it to be broken rather than merely unlucky.

### The seam — four routes, all closed or tested

1. **By density — re-examined, and the first answer was wrong.** This was originally
   recorded as closed on the grounds that `subunit_contrast` returns **0.98–1.20** for
   every template and "1.0 means indistinguishable". **That null is wrong.** The
   statistic divides the axial autocorrelation at half a repeat by the value at a whole
   one, and the autocorrelation is of a *finite* object: a 42 Å shift keeps more overlap
   than an 84 Å shift whatever the subunits look like. A perfectly monomer-periodic
   template therefore scores well above 1, by an amount set by template length. That is
   the whole story of the spread — the 2-ring 1.20 against the 4-ring 1.05 is length, not
   one template being better at telling α from β. A monomer-symmetrised copy of each
   template scores **1.29–1.59**, so the null is never 1.0 and it moves by 0.30 with
   length alone. (That symmetrised value is a useful demonstration but not a usable
   null in its own right — it smears the template's ends and over-charges. See "Seam
   route 1, measured properly" for the measurement.)

2. **By monomer geometry.** The monomer lattice closes exactly: 14 × 9 Å = 126 Å = 3
   monomers, and 126 mod 42 = 0. There is no monomer-level discontinuity anywhere on
   the tube. The seam exists only modulo 84 Å.

3. **By searching φ with a seamed template.** A 14₃ dimer lattice must contain a seam
   (1.5 dimers per turn cannot close), and the models do. But it is a **pure labelling
   flip** — radius sd 0.355 Å, axial steps a uniform −9.0 Å — so rotating the template
   by one protofilament changes nothing detectable.

The fourth route is open in principle but not in practice: the 41/43 Å alternation is a
*positional* signature, independent of resolution. But the detectable quantity is the
0.5 Å alternating component of position (a quarter of the 1.98 Å step difference), and
axial scatter is 2.35 Å after refinement. That needs ~200 peaks per protofilament, or
~2.6 µm of tube. Refinement improved scatter only 16% (2.78 → 2.35 Å) and the limit is
outlier peaks, not curvature or angular sampling — a cubic axis fit buys 0.5 Å, and
the ring's 0.2 Å axis offset means φ error contributes nothing.

### Seam route 1, measured properly

`subunit_contrast` cannot settle this, for the reason above. Two measurements can.

#### Why the null is 1.5 and not 1.0 — checked on blobs

This is counter-intuitive enough to be worth a demonstration, because "1.0 means
indistinguishable" is the natural reading and it is wrong.

The intuition that gives 1.0 is the *infinite* filament: there the autocorrelation is
periodic, `ACF(u) = ACF(2u) = ACF(0)`, and the ratio is 1. A finite object of axial
extent `L` instead carries a triangular envelope, so

> ACF(u)/ACF(2u) = (1 − u/L) / (1 − 2u/L)

A **2-ring template is 4 monomers**, so `L ≈ 4u` and the null is
`(1 − ¼)/(1 − ½) = ` **1.500** — for any template, whatever its subunits look like.

Tested on a train of identical Gaussian blobs, where there is no α/β difference by
construction:

| monomers | predicted | measured |
|---|---|---|
| 4 | 1.488 | **1.500** |
| 6 | 1.247 | 1.250 |
| 8 | 1.165 | 1.167 |

The 6DPU 2-ring template's measured null is **1.500**, exactly as the 4-monomer
prediction says. (Trains longer than ~12 monomers overflow the 600 px box and the
circular autocorrelation wraps, flattening the measured value to ~1.014; that is the
wraparound floor, not the envelope.)

The statistic does respond to a real difference — with four blobs, a β amplitude of 1.0
scores 1.500, 0.7 scores 1.409, 0.5 scores 1.200 and 0.0 scores 0.001. So 6DPU's observed
1.255 against its null of 1.500 is a genuine α/β difference. It simply is not measured
against 1.0.

#### What a correlation would actually feel

The quantity that decides the route is the drop in correlation when the template is
placed **one monomer out of register**: `ACF(u)/ACF(0)`. For a template whose two
half-repeats are identical that drop is not zero — the template is finite, so a shifted
copy overlaps less — and that finite-length term is the **floor**. It has to be measured,
not modelled: a Fourier-symmetrised volume is *not* a valid null, because averaging a
finite object with a shifted copy smears its ends and charges a penalty of its own. (That
mistake gives the monomer-identical control a spurious 12% "α/β deficit".)

So `models/build_seam_controls.py` builds a 2×2 of real templates, identical in length,
atoms and B-factors, differing only in the two properties that can separate α from β:

| template | α/β density | 41/43 Å alternation |
|---|---|---|
| `periodic` | no | no | ← the floor |
| `samemonomer` | no | yes |
| `equalsteps` | yes | no |
| `observed` | yes | yes |

The floor is **0.7376–0.7548 across every low-pass cutoff from 40 Å to full** — flat, as a
pure finite-length term must be. Everything below it is real register information:

| low-pass | 41/43 only | α/β only | both |
|---|---|---|---|
| 40 Å | 0.70% | −0.25% | 0.50% |
| 20 Å | 0.74% | 0.62% | 1.29% |
| 12 Å | 0.88% | 1.75% | 2.37% |
| 8 Å | 1.34% | 3.56% | 4.05% |
| 6 Å | 1.95% | 5.52% | 5.86% |
| 5 Å | 2.87% | 7.25% | 7.44% |
| 4 Å | 5.22% | 10.32% | 10.15% |
| 3 Å | 7.93% | 13.47% | 13.20% |
| full | 12.71% | 17.76% | 17.24% |

**Read three things off this.**

- There is **no sharp 3–4 Å crossover**. The register signal grows smoothly and
  monotonically from ~1% at 20 Å to 17% at full resolution. The old "α and β only become
  distinguishable below 3–4 Å" was the right order of magnitude for where it gets *large*,
  but it was inferred from an invalid statistic and it is not a threshold.
- **α/β density dominates**, not the 41/43 Å alternation. That is the opposite of what the
  positional argument (route 4) would suggest: at 8 Å the density term is 3.56% against
  the alternation's 1.34%. The two do not add — observed (4.05%) is barely above density
  alone — so they are partly redundant, which also means route 4 is not an independent
  reservoir of signal on top of route 1.
- The 4-ring template's floor is **higher** (0.86 at 40 Å against 0.75), because a longer
  template loses less overlap. Longer templates raise the floor *and* the signal; the
  useful comparison is always against the matched floor.

#### The infinite-filament answer, and why it is length-independent

The finite 2×2 above still leaves a worry: is 17% a property of the specimen or of a
2-ring template? It is not. Rebuilt as a **single protofilament 12 monomers long** (the
register question is purely axial, so one protofilament carries it, and 12 × 42 Å = 504 Å
fits the 600 px box) the floor rises from 0.742 to 0.905 exactly as a finite-length term
must — but **the ratio to the floor does not move**:

| | 4 monomers (2-ring tube) | 12 monomers (1 protofilament) |
|---|---|---|
| periodic (floor) | 0.7419 | 0.9046 |
| samemonomer | 0.8756 | 0.8885 |
| equalsteps | 0.8262 | 0.8210 |
| **observed** | **0.8313** | **0.8339** |

Because the ratio is length-independent, it *is* the infinite-filament answer: **a
one-monomer register error costs about 17% of the correlation**, however long the
filament. Resolved by low-pass cutoff, on the 12-monomer filament:

| low-pass | 41/43 only | α/β only | both |
|---|---|---|---|
| 40 Å | 0.42% | −0.02% | 0.43% |
| 20 Å | 0.49% | 0.78% | 1.17% |
| 12 Å | 0.66% | 1.88% | 2.26% |
| 8 Å | 1.09% | 3.66% | 3.89% |
| 6 Å | 1.63% | 5.60% | 5.66% |
| 5 Å | 2.44% | 7.33% | 7.19% |
| 4 Å | 4.49% | 10.47% | 9.79% |
| 3 Å | 6.86% | 13.65% | 12.74% |
| full | 11.19% | 17.97% | 16.68% |

**Do not use an isolated α-against-β FSC for this.** It is the natural shortcut — for
disjoint monomers the register penalty *is* the α/β cross-correlation — and it gives a
much larger answer: FSC(α, β) on a common lattice site is 0.949 at 40–20 Å, 0.727 at
12–8 Å and 0.571 overall, implying a 43% penalty. The identity fails because monomers in
a filament are not disjoint: the density is continuous, isolated monomers carry hard
boundaries that decorrelate, and a good deal of the filament matches in either register.
The filament measurement is the one a search actually feels. (`analyse_alpha_beta_fsc.py`
keeps the FSC calculation, with an α-against-α control that returns exactly 1.0000.)

#### And the real data shows it — 2DTM can tell which monomer it is on

The top-down extraction gives this test for free. Sites are indexed at the **monomer**
spacing, so consecutive indices differ by exactly one monomer: alternate sites have the
dimer template one monomer out of register, β sitting on α. Laterally adjacent monomers
in a B-lattice are the same type, so a whole index is one type — except across the seam.

On the patch template, full micrograph, 82 sites at 99% occupancy:

- sites one monomer apart differ by **1.35 z on a mean of 13.3 — about 10% of the
  score**, over 40 pairs;
- the **period-2 amplitude is 0.72 against a shuffled chance level of 0.30**, and it
  holds at every detection threshold (min_z 6/7/8) and bootstrap cut (8/9/10) tried,
  spanning 0.72–0.79 against nulls of 0.30–0.34.

**The control that matters.** Index at the **dimer** spacing instead and every site is in
the same register, so any alternation must be an artefact of the extraction. It vanishes:
amplitude **0.018 against a null of 0.385**, paired difference +0.013 ± 0.190 (t = 0.07)
on 41 sites. The alternation is the specimen, not the method.

Which parity is "in register" is arbitrary — it follows the extraction's phase origin and
flips between runs, which is why the phase-free period-2 amplitude is the statistic
quoted rather than the signed difference.

The ring template on the same micrograph gives the same sign more weakly (paired t = 2.62
on 39 pairs, 0.26 z), and the 13-PF cropped search agrees at t = 3.2 but on only 12 sites.

**10% sits between the simulated 9.8% at 4 Å and 16.7% at full**, so the score is drawing
on information to at least ~4 Å — which also answers the resolution question left open
above, without needing a separate measurement.

#### What this does to the seam

Route 1 is no longer just "not closed": the discrimination it needs is **demonstrated on
this data**. That changes the seam from a resolution problem into a bookkeeping one — the
seam is the protofilament whose register phase is flipped relative to its neighbours, and
the phase is now a measurable per-site quantity. The next step is a
protofilament-resolved version of this same alternation test.

Still missing, and it should be done before the result is leaned on: the identical test
on a **synthetic of known register**, which would show the alternation appearing at the
right amplitude when the truth is known. The existing synthetics are built by tiling one
projection, so they carry a fixed register and would serve.

#### The layer lines, which say the same thing independently

A monomer repeat u puts layer lines at `kz = m/u`; **half-order** lines at
`kz = (2m+1)/2u` exist only if the two half-repeats differ. Their power relative to the
integer orders needs no reference volume at all. Against a floor of 0.002:

| | 40–20 Å | 12–8 Å | 6–5 Å | 4–3.5 Å | 3–2.5 Å |
|---|---|---|---|---|---|
| observed | 0.171 | 0.177 | 0.321 | 0.382 | 0.627 |
| 41/43 only | 0.055 | 0.033 | 0.166 | 0.257 | 0.556 |
| α/β only | 0.176 | 0.194 | 0.386 | 0.430 | 0.761 |

Same conclusion: substantial half-order power at every resolution, dominated by density.

#### What this does and does not change

It does **not** show that 2DTM can locate the seam here. It shows that the *template*
retains register information, and that the measurement previously used to rule that out
was invalid. The remaining question is whether the data support it.

Defocus is **8213 Å, i.e. 0.82 µm** — a low underfocus, not a high one. The first CTF
zero sits near **12.7 Å** (300 kV, Cs 2.7 mm), so the resolution range carrying most of
the register signal is transferred, and the CTF is not what limits this route. What does
limit it is dose weighting and the specimen B-factor, and neither has been measured on
this data. That is the thing to do before any more work here: establish what resolution
actually contributes to the score. A 4% effect at 8 Å is worth chasing (80 sites per
protofilament gives ~3σ); a 1% effect at 20 Å is not.

**13 versus 14 protofilaments makes no difference to any of this** (0.171 against 0.173 at
40–20 Å; floors and drops agree to the third decimal), which is expected — every quantity
here is axial, and N only changes the radius.

### The last hypothesis — tested, and negative

If the *real* seam were structurally anomalous where the symmetrised template is not, it
would show as a localised score deficit at one protofilament, needing no α/β
discrimination and no sub-Ångström precision. `extract_lattice_sites(...,
resolve_protofilaments=True)` on the patch template now measures exactly this:
**1122 of 1162 sites (97%) occupied**, with occupancy and mean score per protofilament.

The answer is not a seam. Occupancy is uniform (95–100%), and while mean score varies
a lot — 7.92 to 12.09 — it does so in **blocks of about three protofilaments**, not as
one outlier. A smooth first-harmonic model explains only 30% of the variance and the
residuals reach −26.7σ across many protofilaments, so neither "smooth near/far
modulation" nor "one anomalous protofilament" describes it.

Two confounders dominate, both larger than any plausible seam effect:

- the patch spans ~6 protofilaments per match, so per-protofilament attribution is
  smeared over roughly that many;
- 6 of 7 diametrically opposite pairs show the near wall scoring above the far.

**The machinery exists and is tested; the measurement is not yet interpretable.** To
make it so you would need a narrower template (fewer protofilaments per match, at a
cost in SNR), an explicit near/far wall model, and reproducibility across several
microtubules.

The remaining idea is not a new route but more statistical power on the first one —
a whole-microtubule model, which is affordable to search now that the lattice is
known. See "Next steps".

---

## Curved microtubules

Everything above rests on one straight filament, and the analysis bakes that in:
`FilamentAxis` is a point and a direction, and `filament_coordinates` projects onto that
line. `GDP_curved_mgraph.mrc` contains tubes that are not straight, so the assumption had
to be tested rather than argued about.

### A synthetic bent tube with known truth

`run_scripts/simulate_curved_microtubule.py` lays a microtubule along a circular arc
across a 2048² frame. The 2-ring template is tiled at its own repeat (two dimers) with
each copy rotated to the **local tangent**, so psi is a function of arc length rather
than a constant. Blocks are ~178 px against a bend radius of 8192 px, so within-block
straightness costs a sagitta of 0.2 px.

Protofilament number, lattice, monomer rise, polarity and the curve are all inputs, and
are written to a `.truth.json` beside the micrograph. The default is 13 PF on the
**6DPV/compacted** lattice, matching GDP: rise 40.943 Å, sagitta 100 px, radius 8192 px
(0.75 µm), total turn 17.9°, psi sweeping 308.7° → 291.3°.

It also writes a `.paths.json` in the drawing tool's own format, so the manual
segmentation step is skipped entirely on synthetic data — the curve is known, so the
constraint can be built from it directly.

**What it does not contain is bend strain.** Each tiled block is an internally unstrained
straight tube, so the real `u0[1 + R·kappa·cos(phi − phi0)]` modulation is absent by
construction. That is still a useful ground truth — a strain measurement here must return
zero, so it tests that curvature is not *invented* — but recovering a known non-zero
strain needs a bent atomic model, not tiled rigid blocks.

### All four readouts work on a bent tube

Against the known answers, on the high-SNR version (max z 14.2):

| readout | measured | truth | |
|---|---|---|---|
| polarity | psi 299.44° | 300.00° | 0.56° |
| protofilament number | 13, margin 23.5× | 13 | exact |
| lattice spacing (patch) | 40.937 Å | 40.943 Å | −0.01% |
| lattice spacing (ring) | 40.906 Å | 40.943 Å | −0.09% |
| monomer register (patch) | 2.19× chance, 13.9% | present | — |

Polarity and protofilament number were never at risk: `estimate_polarity` uses only psi
and `estimate_protofilament_number` is a pure azimuth harmonic, so neither touches the
axis at all.

### The curvature is visible exactly where predicted

The transverse coordinate — the one `extract_lattice_sites` discards — traces the arc:

| along (px) | −691 | −554 | −402 | −209 | −27 | 154 | 375 | 593 | 778 |
|---|---|---|---|---|---|---|---|---|---|
| across (px) | −17.8 | −7.1 | +2.5 | +10.4 | +14.4 | +13.7 | +7.4 | −4.1 | −19.1 |

33.5 px peak to peak, a clean inverted parabola. The site deviations bow by **+8.47 Å**,
a fifth of a monomer. Neither quantity is examined anywhere in the current analysis, which
is why curvature degrades it *silently*.

### The chord-vs-arc bias law, confirmed on data

Projecting a curved tube onto a straight axis measures chord rather than arc, which
biases the rise low by approximately **−L²/(40R²)**. Restricting the same track to
shorter spans changes L while holding R, the template and the SNR fixed:

| span (px) | sites | rise (Å) | measured | predicted |
|---|---|---|---|---|
| 951 | 25 | 40.926 | −0.043% | −0.034% |
| 1211 | 31 | 40.914 | −0.070% | −0.055% |
| 1470 | 37 | 40.906 | −0.092% | −0.081% |
| 1730 | 42 | 40.906 | −0.090% | −0.111% |

Measured tracks predicted within ~25% once the track is long enough for the slope to be
precise. (Shorter spans are statistical noise, not bias: at 432 px the rise reads +0.92%
on 14 sites, because the slope's standard error falls as M^−1.5.)

**So the straight code degrades but does not break at this curvature.** At R = 0.75 µm
over 1730 px the bias is 0.1%, well inside the 2.5% that separates expanded from
compacted. It scales as L², reaching 1% at roughly 0.48 µm of tube at the same radius —
which is where a curved axis stops being optional.

### Ring against patch, settled on ground truth

`models/build_patch_template.py` cuts a patch — a few adjacent protofilaments rather than
the closed tube — from the same generator as every other template, for any N and either
lattice. A 13-PF, 5-protofilament, 6DPV patch measures radius 109.87 Å, rise 40.835 Å and
an **axis offset of 85.8 Å**, matching the 86.8 Å of the existing 14-PF patch.

On the identical curved micrograph, the register test:

| template | sites | amplitude | chance | ratio | difference |
|---|---|---|---|---|---|
| **patch, 138°** | 42 | 0.812 | 0.371 | **2.19×** | 1.62 z = **13.9%** |
| ring, 360° | 42 | 0.238 | 0.347 | 0.69× | 0.48 z = 4.0% |
| patch, dimer control | 22 | 0.294 | 0.488 | 0.60× | — |

This is the first time the ring/patch difference has been shown against **known** truth
rather than inferred from real data, and 13.9% sits alongside the 10% measured on the real
GMPCPP micrograph.

**Curvature is not what limits the register.** Splitting the track gives the same effect
at every scale — 13.5% and 12.2% in halves, 9.0–16.6% in thirds, 9.7–12.4% in quarters.
The whole-track ratio is the strongest simply because more sites lower the chance level,
which falls as 1/sqrt(n). An 8.47 Å bow is a fifth of a monomer, nowhere near enough to
flip parity.

### Why the patch wins, and it is not the template

Patch and ring have **identical axial autocorrelation** (0.6193 against 0.6194) — the
patch is a subset of the ring's protofilaments, so per-column axial profiles are the same.
The difference is **projection**.

A 3-start helix staggers protofilament *p* axially by −9.69 Å. Summing 13 of them spreads
their alpha/beta phases over 13 × 9.69 = 126 Å = 1.5 dimer periods, so the dimer order
largely cancels; over 5 adjacent protofilaments the spread is only 48 Å. The surviving
fraction is `|sin(n·pi·phase)/sin(pi·phase)| / n`:

- ring, 13 PF: **21.7%** of the per-protofilament register signal
- patch, 5 PF: **54.8%**

A 2.5× advantage predicted, 3.5× observed. The same argument explains the projected tube's
layer lines: the dimer order at 83.6 Å is **42× stronger** than the monomer order at
41.8 Å, because it is the *monomer* order the helical phase spread cancels, not the dimer.

### Annotation and the continuous constraint

Leopard-EM already evaluates a per-pixel angular restraint on the GPU —
`_orient_ok_from_psi_center` in `backend/utils.py` expands the Euler box about a
`psi_center` field per pixel. What did not exist was any way to *produce* that field for a
filament: `rasterize_spatial_maps` cannot, and the only writer was the membrane exporter.

- `programs/constrained_search/napari_choose_filament_path.py` draws editable polylines
  (vertices addable, movable, deletable) with a **per-path width**, fits a spline, and
  writes `eligible`, `region_id` and `psi_center` from the local **tangent**.
- `programs/constrained_search/make_filament_constraint.py` converts the saved paths to
  the sidecar, so a napari-only environment without `h5py` can still draw.

⚠️ The membrane exporter stores the local **normal**; a filament needs the **tangent**.
Both are `atan2(-dy, dx)`, so copying it unchanged would put every allowed psi at 90° to
the tube.

Only psi is per-pixel: theta and phi remain one global Euler box, and
`cone_half_angle_deg` governs theta and psi inseparably. That covers a tube bending **in
the image plane**. Out-of-plane bending would need a per-pixel theta, which does not exist
anywhere in Leopard-EM.

### A 145,000x speedup that was not optional

`count_orientations_from_psi_center` was O(pixels × angles). On a 2046 × 2880 crop with
88,416 in-box angles that measured **581 minutes**, which blocked the workflow outright.

But a pixel's count depends only on its psi *value*, never on where the pixel is, and for
a cone under 90° the allowed angles form one contiguous arc of the sorted grid — which two
binary searches settle. Now **0.24 s**. Verified bit-identical against the original scan
across cones of 5–89° and every polarity combination, with the wide-cone case still taking
the old path. It speeds the membrane workflow up equally.

### What is still open

- **The curved axis itself is not built.** Everything above uses the existing straight
  fit, deliberately, so there is a measured baseline rather than an assertion. See the
  plan for `FilamentCurve` and arc-length projection.
- **Bend strain** needs a bent atomic model; the tiled synthetic has none.
- **A 13-PF patch on the real GDP micrograph** has not been run.
- The synthetic's noise is a phase-randomised real frame. Using the GDP frame buried the
  tube — its power spectrum is full of microtubule crosshatch — so the GMPCPP frame is
  used instead, being mostly ice.

## Things that will trip you up

- **Do not centre the patch template on the tube axis.** It was tried and is worse on
  every measure (axis residual 21.7 Å vs 10.3 Å, |R| 0.39 vs 0.59, 3× more detections).
  Keeping the mass off the box centre is what makes φ well determined: rolling φ then
  physically moves the patch, whereas a centred patch just sweeps it around the tube.
  The original off-centre design was deliberate and correct.
- **The three template shapes are complementary and cannot be merged.** Ring: φ
  undetermined (14-fold pseudo-symmetric) but position excellent — best for axis,
  spacing, PF number. Off-axis patch: φ well determined, position mediocre — the only
  one carrying per-protofilament information. Centred patch: worst of both.
- **The optimal z threshold is template-dependent, and differs by task.** At z = 7 the
  ring gives |R| = 0.75 but the patch collapses to 0.12. Check `lattice_sharpness`
  rather than assuming, and see the table above for which task wants which cut.
- **Short tracks lie.** See the spacing result above.
- **`lattice_sharpness` is degenerate under sub-harmonics.** An 84 Å lattice scores
  perfectly at 42 Å with alternate sites empty. The default ±10% search window cannot
  span a factor of two so it is safe, but a wider one is not. Low occupancy is the tell.
- **Do not load a full-frame correlation table with `CorrelationTable.from_hdf5`.** Use
  `detections_from_hdf5`, which streams and thresholds during the read.
- **`correlation_variance` holds a standard deviation**, not a variance. The z-score
  divides by it directly.
- **The correlation table threshold is on raw cross-correlation, not z.** They coincide
  here only because the correlation mean and variance are ≈ 0 and 1.

### Building a synthetic micrograph — five ways it went wrong

All five were found the hard way, and four of them produced the *same* symptom: a search
that finds nothing, invariant to template and to amplitude.

- **`calculate_ctf_2d` takes defocus in MICROMETRES.** Passing 8212.763 Å means 82 cm of
  defocus; the microtubule becomes a coarse grating across the whole field and the search
  anti-correlates with it. `leopard_em/utils/ctf_utils.py` converts with `* 1e-4`. The
  visual tell is fringe spacing: √(λ·Δf) is ~14 px here, and the broken version showed
  ~150 px. **This one masqueraded as a sign error** and led to a "fix" that flipped the
  contrast to compensate — with correct units the sign needs no flip, and the real
  microtubule correlates at **+0.0275, 16.5σ**.
- **`pos_x` is the correlation-map index, not the image position.** `pos_x_img = pos_x +
  300` is the particle centre. Proof: the map is (424, 841) = exactly
  (1023 − 600 + 1, 1440 − 600 + 1). Using the raw columns puts a synthetic object 276 Å
  from the real one, outside the constraint mask, where the search may not look.
- **The search damps its template by `ctf_B_factor` (60 Å² here)** before correlating.
  Omit it and the two differ at exactly the frequencies whitening weights most.
- **SNR calibration by `z ≈ k‖s‖/σ` is wrong**, because the search whitens first. Measured
  0.69× the requested z. Calibrate empirically instead.
- **Do not hand-roll a whitened correlation to check any of this.** That check disagreed
  with the raw one and wasted hours; the raw correlation against the real micrograph at a
  known peak is the reliable comparator.

**Render the image before running any search.** Every one of these was visible at a glance
in a rendered comparison, and none of them was obvious from the numbers. The acceptance
test that works, and costs seconds:

> the synthetic must correlate with the real micrograph's **sign and magnitude** at a
> known peak — real +0.0275, synthetic +0.0280.

Roughly an hour of GPU went on searching images that had no detectable microtubule in
them, because that test was applied after the searches instead of before.

### Threshold vs lattice completeness

Measured on the full-frame ring data, which is what set the new default of 6.5:

| z cut | peaks | \|R\| | on-lattice | sites occupied |
|---|---|---|---|---|
| 5.5 | 7956 | 0.17 | 58% | 100% |
| 6.5 | 1948 | 0.57 | 81% | 100% |
| **7.0** | 889 | 0.77 | 92% | **100%** |
| 8.0 | 160 | 0.93 | 99% | **71%** |
| 9.0 | 24 | 0.99 | 100% | 22% |

Purity rises and completeness collapses. Above z ≈ 7.5 you discard real lattice sites.

**These are three separate decisions — do not conflate them, as was done at first:**

| decision | wants | why |
|---|---|---|
| **table** threshold (written to disk, irreversible) | ~6.5 | keeps 100% of sites for any later analysis, while cutting the table ~100x |
| **completeness** analyses (the seam score-deficit test) | low, ~7 | needs every real site populated |
| **parameter fitting** (rise, axis) | high, >8 | noise peaks corrupt the axis and bias the rise low |

Choosing the table threshold does not commit you to analysing at that threshold.

---

## Bugs found and fixed (all committed)

| commit | what |
|---|---|
| `37841be` | Correlation table decoded orientations wrongly — `psi_angles` was stored as identical values in every table ever written. The grid is psi-outer, the code assumed psi-inner. Data was recoverable; format bumped to v2, which now stores the full grid |
| `68b3153` | `get_rfft_slices_from_volume` used `fftn` where `rfftn` was required, returning wrongly-sized, geometrically wrong projections. **Membrane polarity overlays made before this are stretched and should be regenerated** |
| `dcb9eb2` | `ttsim3d` imported eagerly, so a broken install took down every manager |
| `ab62f50` | Lattice repeat estimator returned its own initial guess on long tracks (38.5 Å from a guess of 39, 45.2 Å from a guess of 45). Aliases sit ~rise²/span apart and local refinement cannot escape one. Now scans globally first |
| `7f82a5e` | Correlation table threshold configurable, default raised 5.5 → 6.5 |
| `d4abf49` | `detections_from_hdf5`: streams a table off disk. The full-frame table needed ~40 GB as Python lists against 6 GB as arrays |
| `8f6bad6` | `extract_lattice_sites`: top-down site extraction, 1-D and protofilament-resolved |

Also fixed in the analysis layer: `_infer_rise` was biased by the alternating 41/43 Å
steps (median picked one alternate; a plain linear fit is biased for odd step counts)
and by non-lattice gaps between patches.

---

## Next steps, in order

Everything left needs **different models**, not different analysis. The analysis side is
now in reasonable shape; what limits all three remaining questions is having only one
template geometry to compare against.

The models came from a 12-mer with helical symmetry applied, so most of these are a
matter of re-running that symmetry step with different parameters — no new structure
determination needed.

### 1. Protofilament number — models built, competition run, still unresolved

12/13/14/15-PF templates exist and the competition has been run on both the real crop and
a synthetic control. See "Protofilament number" above for the numbers. Where it stands:
the method **provably works** (recovers 14 from a known 14-PF microtubule, middle-peaked),
but the real data gives a broad 12–13 maximum whose per-site majority favours 14. What is
left, cheapest first:

- **Synthetic control built from a 13-PF template** (~1 h). The one confound the current
  control cannot exclude: that the method favours whichever template made the truth. If a
  13-PF ground truth returns 13, the control is symmetric and trustworthy.
- **Full micrograph rather than the crop** (~3.5 h for four). More paired sites; the
  control's sign tests were only 54–65/109, so power is the limiting factor.
- **A longer microtubule.** This unlocks the moiré route as well — see below.

Use `models/build_mt_templates.py` — it already does the fetching, trimming, B-matching
and symmetry expansion, and takes any pair of PDB IDs. The 2×2 and its analysis are
written and validated; only the deposition IDs change.

### 2. Lattice spacing — ~~compete expanded against compacted~~ **DONE**

Expanded, t = 10, with the model-quality confound measured at zero. See "Lattice
spacing, confirmed independently by template competition" above. What remains optional:

- the **native-B control** (`--native-b` on both scripts, ~36 min) — the quality contrast
  is already null so this is belt-and-braces, but it is the honest check that
  B-flattening did not create the result;
- the **full micrograph** (~3.5 h) — would allow spacing variation *along* the tube,
  though at t = 10 the verdict will not change;
- the **patch arm** — the only one that could show whether compaction varies around the
  circumference, or between near and far wall, which the ring cannot see.

### 3. Seam — a whole-microtubule model, searched cheaply

Register discrimination scales as √M in the number of subunits, so the way to attack
the seam is a **much larger model** — a whole microtubule rather than two rings. That
would normally be prohibitive to search.

**It is affordable now because we already know where the peaks are.** With the lattice
fitted, a full orientation search is unnecessary: either refine from the known
positions and orientations, or run a very heavily restrained match over a small window
around each predicted site. The expensive part of 2DTM is the orientation search, and
the model removes almost all of it.

Worth knowing before investing: `subunit_contrast` says α and β are indistinguishable in
every template rendered so far, and that is a noise-free upper bound. A bigger model
raises the *statistical* power (√M) but does not change the per-subunit contrast, so
this is a bet that √M over a whole microtubule is enough to overcome a very small
per-subunit signal. Measure `subunit_contrast` on the big model first — it costs
seconds and will say whether the bet is worth making.

### 4. Other systematics, once there are more models

- **A longer microtubule — the highest-value thing to acquire.** Everything still open is
  limited by 325 nm of one filament, not by the analysis. At ≥600 nm the moiré period
  (204 nm for 14-PF) gives ≥3 fringes, making the supertwist measurable both by eye and
  via the dφ/ds fit — a *slope*, so immune to the template-size worry that dogs the
  competition, and signed, so 12 and 14 differ in direction not just magnitude. It also
  supplies more paired sites for the competition and enough averaging for a width
  measurement. One acquisition unlocks three independent routes.
- **Magnification anisotropy** — the largest uncorrected systematic on the spacing
  number. Biases the repeat depending on the filament's in-plane direction, and no
  `mag_matrix` is configured. Needs a second filament at a different in-plane angle.
- **More microtubules and more micrographs** — everything so far is one filament in one
  image. Polarity and spacing are single-object measurements; the biology is in their
  distribution.

### Practical notes for any competition

- Simulate every candidate **identically** (same box, `b_factor_scaling`, dose, MTF), or
  the comparison measures rendering differences rather than structure.
- Compare on a common footing. Templates of different mass have different intrinsic
  SNR, so raw maximum z is not a fair statistic on its own; prefer lattice occupancy,
  sharpness, and summed score over matched sites.
- Cost is ~9 min per template on the crop, ~52 min on the full frame, and far less than
  either for a restrained search from a known lattice.

---

## Reproducing

```bash
# Full pipeline on one correlation table (defaults to the full-frame ring table)
python ../programs/microtubule/run_microtubule_lattice.py

# Progress of long searches
./check_full_runs.sh
```

| file | what |
|---|---|
| `configs/match_tm_{crop,full}_{patch4,2rings}.yaml` | search configs |
| `configs/filament_constraint_{box,full}.yaml` | constraint sidecars; the full one extends the drawn line across the whole frame |
| `run_scripts/run_match_{cropped,full}_*.py` | search runners |
| `run_scripts/simulate_patch_centred.py` | template simulation (the centred experiment — kept as a record of a negative result) |
| `models/build_mt_templates.py` | **template competition builder** — fetches two depositions, equalises them, emits the 2×2 |
| `run_scripts/simulate_lattice_competition.py` | simulates the 2×2 (~29 s each, CPU) |
| `run_scripts/run_lattice_competition_cropped.py` | runs the 2×2 on the crop (~9 min per cell) |
| `run_scripts/analyse_lattice_competition.py` | paired per-site comparison and the three contrasts |
| `run_scripts/run_lattice_model_competition.py` | competes lattice *models* against fixed detections — built and run, inconclusive; read the null caveat before using it |
| `results_{cropped,full}/` | outputs, including correlation tables |

**Use the `leopard-em` conda env** (`/home/jdickerson/miniconda3/envs/leopard-em/bin/python`).
Neither `.venv` nor base miniconda has `h5py`, so `FilamentConstraint` will not import
there. GPU simulation also fails on this box (`nvrtc` cannot find
`libnvrtc-builtins.so.13.0`); simulate on CPU, which is what the existing templates used.

Correlation tables from the full frame are 5.9–7.7 GB (247–321 M detections at the old
5.5 threshold). Load them with `detections_from_hdf5`, which streams and thresholds
during the read in a few seconds; `CorrelationTable.from_hdf5` would need ~40 GB of
Python lists. The new 6.5 default should make future tables ~100× smaller.
