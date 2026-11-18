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
| Covers | Anything the search does, including refinement of the null candidates | match_template searches (uniform orientation grid, one pixel size) |

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
Set `write_null_particles: true` to write a particle stack per copy (`null_seed<seed>_particles.csv`) of the same peaks it counted, refine those exactly like the real particles, and pass the refined null z-scores to [`empirical_zscore_cutoff`](../api/analysis.md). The stack holds every distinct peak above `peak_floor`, which is 85,711 peaks (124 MB) at z 5 on a 4k map, so set the floor near the threshold you intend to refine around.
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
```

The micrograph, template and every search setting come from the match_template YAML, unchanged.

## Running the null calibration program

Point `YAML_CONFIG_PATH` in [`programs/null_calibration/run_null_calibration.py`](https://github.com/Lucaslab-Berkeley/Leopard-EM/blob/main/programs/null_calibration/run_null_calibration.py) at the configuration and run it.
It writes `null_summary.json` to the output directory. For each target, the file holds:

- `nominal`: the independent-trials threshold;
- `phase_randomized`, `phase_randomized_confidence_90`: the thresholds from the null copies;
- `analytic`: the predicted threshold.

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

**Phase-randomised null.** The copies are those used for the measured thresholds above. The phase randomisation reproduces them bit for bit (same algorithm and seeds).

**Runner, end to end.** `NullCalibrationManager` re-ran one stored copy (seed 2) of the 3-plane search, taking 1 h 57 min on one GPU. Its z-score map is identical to the stored one (max |Δz| = 0): the same 85,711 distinct peaks ≥ 5, at the same positions. With `write_null_particles`, those peaks were exported as a particle stack in about 6 s.

**Small-scale check (test suite).** The test suite runs the predictor on a 128² coloured-noise image with a 16³ template and checks it against 24 phase-randomised copies searched on the CPU. At z 4.2–4.6, where null peaks are rare, the prediction is within 10% of the counts.

## Limitations

- **Validated on one micrograph and one template** (xenon_131, 60S at three B-factors). Other templates and datasets are untested.
- **The analytic null is first order in κ.** At z ≈ 7 it over-deflates the tail by 5–20% for sharp templates. Where κz⁴ approaches 1 (very blurred templates, or white noise), it under-predicts by up to about 40%.
- **The analytic null treats the search axes as conditionally independent** and models crowding as for Poisson-distributed peaks. It is accurate where null peaks are rare relative to the suppression disc, which covers any useful threshold.
- **The analytic null covers match_template only:** a uniform orientation grid and one pixel size. Refinement is not modelled; use the phase-randomised null for refined z-scores.
- **Neither null includes structured backgrounds.** Mask them.
