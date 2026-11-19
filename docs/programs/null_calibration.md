---
title: The Null Calibration Program
description: Detection thresholds from phase-randomised and analytic null searches
---

# Null calibration: thresholds from null searches

By default, match_template picks particles above a z-score threshold set for one false positive per micrograph ([`gaussian_noise_zscore_cutoff`](match_template.md#running-the-match-template-program)).
That threshold treats every correlation of the search (pixels × orientations × defocus planes) as an independent Gaussian trial.
They are not independent:

- neighbouring orientations, defocus planes and pixels are strongly correlated (adjacent defocus planes 200 Å apart correlate at about 0.93);
- the per-pixel z-score normalisation removes part of the tail.

So the number of false positives the nominal threshold allows is only roughly what it promises.
On a yeast lamella micrograph (xenon_131, 60S template, 4096² at 0.936 Å/px), the nominal threshold let through 0.4 to 0.6 false positives per micrograph with a sharp template (B 0), and far fewer with blurred ones (B 60 / 120: the calibrated threshold sits 0.1 to 0.3 z lower).
The nominal threshold is safe for a one-false-positive target but loses particles.
It cannot be used for a false discovery rate, or for z-scores after refinement.

The null calibration program measures the null instead, for the exact search configuration, in two ways:

| | Phase-randomised null | Analytic null |
|---|---|---|
| What it does | Searches phase-randomised copies of the micrograph and counts their distinct peaks | Predicts the null peak rate from the template, filters, grids and the micrograph's power spectrum |
| Cost | One full search per copy (5 or more copies for a 1-false-positive target) | Under a minute on a GPU, about 30 min on a CPU; no search |
| Precision | ±0.05 z with 5 copies, ±0.03 with 10–20 | Within 0.00–0.08 z of measured thresholds (validation below) |
| Covers | Anything the search does, including refinement of the null candidates | match_template searches (uniform orientation grid, one pixel size), and refine_template of their candidates |

Both work for any target, from a fraction of a false positive per micrograph to the rates a false discovery rate needs.

!!! note "Neither null sees structure"

    A phase-randomised copy keeps the micrograph's Fourier amplitudes (CTF, astigmatism, ice, detector) but no structure.
    Structured backgrounds that produce false peaks, such as carbon edges, flat featureless patches or dense objects at the frame edge, are not in either null.
    Exclude them with a mask (`exclude_mask_path`) from both the null and the real micrograph.

## Phase-randomised null

Each copy of the micrograph keeps its Fourier amplitudes exactly and gets random phases (`leopard_em.utils.phase_randomization.phase_randomize_image`).
It is searched with the unchanged match_template configuration, and its distinct peaks above a floor are collected with the same suppression radius, border and mask as the real micrograph.
Pooling the copies, the threshold for a target of *f* false positives per micrograph is the *m*-th highest null peak, with *m* = copies × *f*.
This is unbiased: it delivers the target on average.
For a guarantee instead (at most *f* with 90% probability), use the `"confidence"` rule; it needs copies × *f* ≥ 2.3.

How many copies:

- **One false positive per micrograph:** 5 copies give the threshold to about ±0.05 z, 10–20 to ±0.03. One copy is not enough (the realised rate then varies from 0.2 to 4 per micrograph).
- **10 or more false positives per micrograph, or a false discovery rate:** precision depends on copies × target, so 1–2 copies are enough.
- **Different micrographs** need different seeds.

**Refined z-scores.** Refinement raises null peaks too, so a threshold on refined z must come from null candidates refined the same way.
The analytic refined null below predicts it without refining any null candidates.
Alternatively, set `write_null_particles: true` to write a particle stack per copy (`null_seed<seed>_particles.csv`) of the same peaks it counted, refine those exactly like the real particles, and pass the refined null z-scores to [`empirical_zscore_cutoff`](../api/analysis.md). The stack holds every distinct peak above `peak_floor`, which is 85,711 peaks (124 MB) at z 5 on a 4k map, so set the floor near the threshold you intend to refine around.
Use copies of each micrograph itself: refine_template whitens each image from its own spectrum, so copies made from a different micrograph under-predict refined false positives.

## Analytic null

For a phase-randomised image, the correlation of the micrograph with the template projections is a Gaussian random field over position, orientation and defocus.
Its covariance is fixed by the projections and the image power spectrum alone, so everything a null search measures can be computed.
The prediction has three parts:

1. **Lattice local maxima:** the expected number of search points above *z* that are higher than both neighbours along every search axis (Worsley, 2005), from the field's correlations at one and two grid steps along x, y, ψ, θ, φ and defocus.
2. **Crowding:** peaks within the suppression radius of a higher one are dropped.
3. **Per-pixel normalisation:** the z-score divides by the standard deviation over the search at each pixel. A null peak raises its own pixel's standard deviation, and the standard deviation fluctuates between pixels; to first order this multiplies the null tail by exp(−κ(z⁴/4 − z²/2)), with κ the mean squared correlation of random search points.

The projections are built with the backend's own steps (CTF and whitening filters, normalisation), and the inner products are weighted by the preprocessed micrograph's power spectrum.
Sampling sizes (`n_mean_samples`, `n_kappa_pairs`, `n_lattice_samples`) set the precision. The spread of the cut between sampling seeds comes almost entirely from κ's sample, which is heavy-tailed; the number of lattice samples hardly matters (16 to 512 give the same cut). The defaults (3,000 / 256 / 32) give a spread of about ±0.003 z (SD over seeds) on the xenon_131 13-plane search, for about 45 s on a GPU.

The engine-agnostic parts (`leopard_em.analysis.analytic_null.field` and `.lattice`) take projections and grid neighbours from a callable, so other search engines can reuse them.

## Analytic refined null

refine_template re-scores each candidate on a finer local grid: orientation offsets composed with the candidate's orientation, defocus offsets about its plane, and a few pixels about its position.
The refined z-score is the best of those scores, so null candidates move up too, and a threshold on refined z has to come from the refined null.

Only the search's candidates are refined.
A null candidate of height *u* is a search point whose score is *u* and that beat every search point around it.
Refinement gives it a gain *G* = (best score in the refinement window) − *u*, and the expected number of refined null peaks at or above *z* per micrograph is

λ_ref(*z*) = ∫ ℓ(*u*) P(*G* ≥ *z* − *u* | *u*) d*u*, over *u* ≥ *t*_c,

where ℓ(*u*) = −dλ/d*u* is the density of the search's null peaks over height (the analytic null above) and *t*_c is the threshold at which candidates are refined.

The gain comes from the same Gaussian field, at a few sampled grid points:

1. **The window and what the search saw.** The joint covariance of every score refinement looks at (orientation offsets × defocus offsets, within ±1 px) and of the search's own grid points around the candidate (one step along ψ, θ and φ, and the adjacent defocus planes, within ±2 px).
2. **Condition on the candidate's score.** Given that it is *u*, each other score has mean ρ*u* and variance 1 − ρ², for its correlation ρ with the candidate.
3. **Condition on it being the search's peak.** Of 200,000 Gaussian draws of the window, only those where the candidate beat every search point are kept.
4. **The gain** of each kept draw is the window's maximum minus *u*.

The window is refine_template's own: its `orientation_refinement_config` offsets, composed as refine_template composes them, and its `defocus_refinement_config` range, modelled at a 40 Å step.
Counting every local maximum of refinement's fine grid over the whole search instead, as if refinement searched everywhere, overcounts by about 2× at z 7 on xenon_131: refinement never sees maxima away from a candidate, or at pixels whose best search score fell below *t*_c.

The engine-agnostic part (`leopard_em.analysis.analytic_null.refine`) takes the window's and the search points' projections from the engine, like the rest of the analytic null.

## Configuration

```yaml
match_template_config_path: /path/to/match-template-configuration.yaml
output_directory: /path/to/null-calibration-output

seeds: [1, 2, 3, 4, 5]     # one phase-randomised copy per seed; [] skips this null
false_positives: [1.0, 5.0]  # targets, per micrograph
peak_floor: 6.0            # null peaks are collected down to this z-score

mask_radius: 10.0          # peak calling: use the same values for the real micrograph
border: 0
exclude_mask_path: null    # MRC or .npy, nonzero = excluded, shape of the z-score map

write_null_particles: false

analytic:
  enabled: true
  n_mean_samples: 3000
  n_kappa_pairs: 256
  n_lattice_samples: 32
  batch_size: 4
  seed: 0
  device: cpu              # e.g. cuda:0
  refine:                  # used with refine_template_config_path
    n_candidates: 8        # sampled null candidates
    n_draws: 200000        # Gaussian draws per candidate
    shift_radius: 1        # window positions, +-px
    condition_radius: 2    # positions where the candidate beat the search's points
    defocus_step: 40.0     # modelled defocus step, A (refine's own if coarser)

# Optional: also predict the threshold on refined z-scores
refine_template_config_path: /path/to/refine-template-configuration.yaml
refine_candidate_threshold: 6.25  # match_template z at or above which candidates are refined
```

The micrograph, template and every search setting come from the match_template YAML, unchanged.
From the refine_template YAML only the orientation, defocus and pixel size refinement settings are read, so its particle stack need not exist yet.

## Running the null calibration program

Point `YAML_CONFIG_PATH` in [`programs/null_calibration/run_null_calibration.py`](https://github.com/Lucaslab-Berkeley/Leopard-EM/blob/main/programs/null_calibration/run_null_calibration.py) at the configuration and run it.
It writes `null_summary.json` to the output directory. For each target, the file holds:

- `nominal`: the independent-trials threshold;
- `phase_randomized`, `phase_randomized_confidence_90`: the thresholds from the null copies;
- `analytic`: the predicted threshold;
- `analytic_refined` (with a refine_template configuration): the predicted threshold on refined z-scores.

It also holds the null peak rate λ(z) (expected distinct null peaks ≥ z per micrograph) from the copies and from the prediction.

Pick the real micrograph's particles with the calibrated threshold:

```python
df = mt_manager.results_to_dataframe(locate_peaks_kwargs={"z_score_cutoff": 7.52})
```

The same tools are available in Python:

```python
from leopard_em.analysis import NullPeaks, distinct_peaks, empirical_zscore_cutoff
from leopard_em.pydantic_models.managers import NullCalibrationManager

manager = NullCalibrationManager.from_yaml("null_calibration.yaml")
null_peaks = manager.run_phase_randomized_null()  # NullPeaks
threshold = null_peaks.zscore_cutoff(false_positives=1.0)
rate = null_peaks.rate([7.0, 7.5, 8.0])  # expected null peaks >= z per micrograph

prediction = manager.predict_analytic_null()  # AnalyticNullPrediction
predicted = prediction.zscore_cutoff(1.0)

refined = manager.predict_analytic_refined_null(prediction)  # RefinedNullPrediction
refined_threshold = refined.zscore_cutoff(1.0)
refined.gain.summary()  # median, p90, p99 gain of a null candidate by its height
```

## Validation

**Analytic null against measured nulls.** The setting is a 4096² yeast lamella micrograph (xenon_131, 0.936 Å/px) with the 60S template at three B-factors. The search used a uniform grid of 1.5° in ψ and 2.5° in θ, with whitening. Peaks were called with a 10 px radius and a 16 px border.

The measured thresholds come from phase-randomised copies searched with the same configuration. *Counted* is the m-th pooled peak, as above. *Smoothed* replaces the counted tail with a fitted generalised Pareto tail. Counting error is ±0.06–0.07 z with 3–4 copies and ±0.035 with 14.

| Search | Analytic 1-FP threshold | Measured: counted / smoothed (copies) | Nominal |
|---|---|---|---|
| 13 planes 200 Å apart, B 0 | **7.693** | 7.700 / 7.678 (4) | 7.775 |
| 13 planes, B 60 | **7.581** | 7.661 / 7.554 (3) | 7.775 |
| 13 planes, B 120 | **7.496** | 7.542 / 7.482 (3) | 7.775 |
| 3 planes 400 Å apart, B 0 | **7.522** | 7.515 / 7.506 (14) | 7.587 |

- **Thresholds:** every prediction is within the measured thresholds' counting error.
- **Null rate:** wherever there are 100 or more null peaks to compare, the predicted rate is within about ±15% of the measured one.
- **Sampling:** the analytic thresholds use the default sample sizes (3,000 / 256 / 32) and seed 0. Seeds 1 and 2 move them by up to +0.01 z. On the 13-plane B 0 search, 16 to 512 lattice samples give the same threshold, and the spread between seeds comes from κ's sample: ±0.006 z (SD) with 64 κ batch pairs, ±0.003 z with 256.
- **Cost:** each prediction took 9 s (3 planes) to about 47 s (13 planes) on an RTX 6000 Ada, after the search set-up.
- **Why the nominal threshold is so far off for blurred templates:** at z = 7 the predicted null is 0.58, 0.24 and 0.13 of the formula at B 0, 60 and 120. A blurred template gives a smoother correlation field, with fewer effectively independent trials (adjacent ψ steps correlate at 0.69, 0.80 and 0.84). It also gives more correlation that every orientation shares, which the per-pixel normalisation removes (κ = 4.7, 7.8 and 10 × 10⁻⁴).

**Analytic refined null against refined null searches.** The 3-plane search above (B 0) with its candidates at z ≥ 6.25 refined by refine_template's light grid (ψ 1.5° / 0.5° and θ 2.5° / 0.5°, 77 orientation offsets; defocus ±200 Å at 20 Å).
The measured null is 14 phase-randomised copies whose candidates (93,915 in all) were refined the same way.
The calculated curve is scaled to the copies' counted area (a 30 px border and one excluded corner: × 0.908) and moved by refine_template's own z offset (its z runs 0.012 below match_template's at a fixed pose).

| | Calculated: refined from the candidates | Counted (14 copies) | Every fine-grid peak, as if refinement searched everywhere |
|---|---|---|---|
| Refined null peaks per micrograph at z 7.0 / 7.5 / 7.75 | 436 / 17.8 / 2.9 | 400 / 14.5 / 2.0 | 748 / 20.7 / 3.1 |
| 1-FP refined threshold | **7.89** | 7.84 counted, 7.90 smoothed | 7.89 |
| Gain of a null candidate at z ≈ 6.5: median / p90 / p99 | +0.165 / +0.45 / +0.77 | +0.149 / +0.43 / +0.74 | |

- **Thresholds:** the refined 1-FP threshold lies between the counted and smoothed measurements. The search's own is 7.52, so refinement raises it by 0.37 z.
- **The curve below the threshold:** 1.09× the counted null at z 7.0. Counting every fine-grid peak instead gives 1.9×.
- **Window and sampling:** 9 sampled candidates (3 per plane); seeds 0–2 give thresholds of 7.891–7.902, and rates at z 7.0 within ±1.5%. Scoring ±2 px instead of ±1 px, or conditioning at ±1, ±2 or ±3 px, moves the threshold by at most 0.005 z; an 80 Å defocus step instead of 40 Å, by 0.015 z.
- **Cost:** about 35 s per sampled candidate on an RTX 6000 Ada (7,900 scores per window), after the search prediction.

**Phase-randomised null.** The copies are those used for the measured thresholds above. The phase randomisation reproduces them bit for bit (same algorithm and seeds).

**Runner, end to end.** `NullCalibrationManager` re-ran one stored copy (seed 2) of the 3-plane search, taking 1 h 57 min on one GPU. Its z-score map is identical to the stored one (max |Δz| = 0): the same 85,711 distinct peaks ≥ 5, at the same positions. With `write_null_particles`, those peaks were exported as a particle stack in about 6 s.

**Small-scale check (test suite).** The test suite runs the predictor on a 128² coloured-noise image with a 16³ template and checks it against 24 phase-randomised copies searched on the CPU. At z 4.2–4.6, where null peaks are rare, the prediction is within 10% of the counts.

## Limitations

- **Validated on one micrograph and one template** (xenon_131, 60S at three B-factors). Other templates and datasets are untested.
- **The analytic null is first order in κ.** At z ≈ 7 it over-deflates the tail by 5–20% for sharp templates. Where κz⁴ approaches 1 (very blurred templates, or white noise), it under-predicts by up to about 40%.
- **The analytic null treats the search axes as conditionally independent** and models crowding as for Poisson-distributed peaks. It is accurate where null peaks are rare relative to the suppression disc, which covers any useful threshold.
- **The analytic null covers match_template** (a uniform orientation grid, one pixel size) **and refine_template of its candidates** (no pixel size refinement). The refined null is validated on one search and one refinement; refined positions are merged by refine_template's own suppression, which is not modelled (null candidates are already a suppression radius apart).
- **Neither null includes structured backgrounds.** Mask them.
