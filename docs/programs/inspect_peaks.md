---
title: The Peak & Frame Inspection Programs
description: Inspecting the full local search grid around identified particles, optionally per movie frame
---

# Peak & frame inspection

`refine_template` searches a local grid of orientation, defocus, and pixel-size offsets around each particle and keeps only the single best-scoring hypothesis.
The `inspect_peaks` and `frame_inspection` programs run that same local search but **skip the best-peak reduction**, returning the full grid of local scores for every particle instead.
This is useful for diagnosing *why* a particle refined the way it did — for example, visualizing how sharply peaked the correlation is around the refined orientation, or checking a particle's stability across individual movie frames before trusting its refined pose.

!!! note "Same configuration schema as `refine_template`"

    Both `PeakInspectionManager` and `FrameInspectionManager` are subclasses of `RefineTemplateManager` and reuse its exact YAML schema — see the [refine template program details](refine_template.md) for the meaning of `particle_stack`, `defocus_refinement_config`, `orientation_refinement_config`, `pixel_size_refinement_config`, and `preprocessing_filters`.
    Whatever grid those refinement configs define is the grid that gets scored and returned in full, rather than reduced to a single best value.

!!! warning "Output tensors can be very large"

    The full local score grid is `(N, n_px, n_defocus, n_orient, H, W)` in `"cross_correlation"` mode — this can get large quickly for big particle stacks or fine-grained refinement configs.
    Use `"frc"` output mode (below) for a much more compact result, and/or subset the particle stack in your config for large runs.
    There is currently no automatic chunking of the particle stack to manage memory.

## Peak inspection (`PeakInspectionManager`)

`PeakInspectionManager` runs the refine-template backend without best-peak reduction, returning the score at every searched hypothesis for every particle.

A default config file is available [here on the GitHub page](https://raw.githubusercontent.com/Lucaslab-Berkeley/Leopard-EM/refs/heads/main/programs/inspect_peaks/inspect_peaks_example_config.yaml) — it is identical in structure to the [refine template example config](https://raw.githubusercontent.com/Lucaslab-Berkeley/Leopard-EM/refs/heads/main/programs/refine_template/refine_template_example_config.yaml).

### Running peak inspection

We provide an example script, [`Leopard-EM/programs/inspect_peaks/run_inspect_peaks.py`](https://github.com/Lucaslab-Berkeley/Leopard-EM/blob/main/programs/inspect_peaks/run_inspect_peaks.py), which loads a config, runs peak inspection, and saves the result to a self-describing `.npz` file.
Edit the constants near the top of the script:

```python
YAML_CONFIG_PATH = "/path/to/inspect_peaks_configuration.yaml"
OUTPUT_PATH = "/path/to/results_inspect_peaks.npz"
CORRELATION_BATCH_SIZE = 32  # lower if you run out of GPU memory
OUTPUT_MODE = "cross_correlation"  # or "frc"
```

Or drive it directly from Python:

```python
from leopard_em.pydantic_models.managers import PeakInspectionManager

manager = PeakInspectionManager.from_yaml("/path/to/inspect_peaks_configuration.yaml")
output_path = manager.run_and_save_peak_inspection(
    output_path="/path/to/results_inspect_peaks.npz",
    correlation_batch_size=32,
    prefer_refined_angles=True,
    output_mode="cross_correlation",  # or "frc"
)
```

`run_peak_inspection(...)` is also available if you'd rather get the raw tensor (and, in `"frc"` mode, the frequency bins) back in memory without writing a file.

### Output modes

- `"cross_correlation"` (default) — returns the full local cross-correlation map for every hypothesis, shape `(N, n_px, n_defocus, n_orient, H, W)`.
- `"frc"` — returns local Fourier ring correlation spectra instead of full 2-D maps, shape `(N, n_px, n_defocus, n_orient, n_freq)`, which is far more compact for large searches.

See [Data from peak & frame inspection](../data_formats.md#data-from-peak-frame-inspection) for the full `.npz` file layout and how to load results back with `leopard_em.analysis.load_inspection_result`.

## Per-frame peak inspection (`FrameInspectionManager`)

`FrameInspectionManager` extends `PeakInspectionManager` to score each movie frame **independently**, rather than the motion-corrected sum used by `match_template`/`refine_template`/`inspect_peaks`.
This is useful for checking how a particle's local correlation landscape evolves across a movie's exposure — for example, to spot particles that only correlate strongly in a subset of frames.

### Configuring the movie

Per-frame inspection requires `movie_config.enabled: true`, plus either a per-frame deformation field or an explicit per-particle shifts CSV to align each particle box to the correct position in every frame:

```yaml
movie_config:
  enabled: true
  movie_path: /some/path/to/aligned_or_unaligned_movie.mrc
  # Provide exactly one of the following two:
  deformation_field_path: /some/path/to/deformation_grid.csv
  # particle_shifts_path: /some/path/to/particle_shifts.csv
  pre_exposure: 0.0
  fluence_per_frame: 1.0
```

`particle_shifts_path`, if provided, takes precedence over `deformation_field_path`.
It should be a CSV with columns `particle_index`, `frame`, `y_shift`, `x_shift`.

A default config file is available [here on the GitHub page](https://raw.githubusercontent.com/Lucaslab-Berkeley/Leopard-EM/refs/heads/main/programs/inspect_peaks/frame_inspection_example_config.yaml).

### Optional per-frame template dose weighting

Setting `apply_template_dose_weighting=True` applies cumulative electron-dose filtering to the (non-dose-weighted) template separately for each frame's exposure interval, using `pre_exposure`/`fluence_per_frame` from `movie_config` — this accounts for radiation damage accumulating over the course of the movie when scoring later frames.
It is off by default.

### Running per-frame peak inspection

We provide an example script, [`Leopard-EM/programs/inspect_peaks/run_frame_inspection.py`](https://github.com/Lucaslab-Berkeley/Leopard-EM/blob/main/programs/inspect_peaks/run_frame_inspection.py), which mirrors `run_inspect_peaks.py`:

```python
from leopard_em.pydantic_models.managers import FrameInspectionManager

manager = FrameInspectionManager.from_yaml("/path/to/frame_inspection_configuration.yaml")
output_path = manager.run_and_save_peak_inspection_per_frame(
    output_path="/path/to/results_frame_inspection.npz",
    correlation_batch_size=32,
    prefer_refined_angles=True,
    apply_projection_normalization=True,
    output_mode="cross_correlation",  # or "frc"
    apply_template_dose_weighting=False,
)
```

The resulting `.npz` carries an extra `frame` axis immediately after the particle axis — `(N, T, n_px, n_defocus, n_orient, H, W)` in `"cross_correlation"` mode, or `(N, T, n_px, n_defocus, n_orient, n_freq)` in `"frc"` mode, for `T` movie frames — and includes a `frame_index` array mapping that axis to actual movie frame numbers.
See [Data from peak & frame inspection](../data_formats.md#data-from-peak-frame-inspection) for the complete layout.

## Loading and inspecting results

Both programs write the same self-describing `.npz` format, loadable with a single helper regardless of which program produced it:

```python
from leopard_em.analysis import load_inspection_result

result = load_inspection_result("results_inspect_peaks.npz")

result.scores                # main tensor; see result.axes for per-dimension labels
result.axes                  # e.g. ("particle", "pixel_size", "defocus", "orientation", "y", "x")
result.euler_angle_offsets   # (n_orient, 3) ZYZ offsets per orientation index
result.defocus_offsets       # (n_defocus,) relative defocus offsets (Angstroms)
result.pixel_size_offsets    # (n_px,) relative pixel-size offsets
result.particle_index        # (N,) maps tensor rows back to the particle stack, if available
result.frequency_bins        # (n_freq,) FRC frequencies, "frc" mode only
result.frame_index           # (T,) movie frame index, per-frame inspection only
```
