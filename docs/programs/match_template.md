---
title: The Match Template Program
description: Description of the match template program and its configuration
---

# The match template program

The match template program takes in a micrograph and a simulated reference structure along with search parameters to find locations in the micrograph which agree with expected 2D projections of this reference structure.
Notably, the match template program simultaneously finds the location and orientations of these particles as well as the depth of the particle within the sample.

!!! info "GPU usage with match template"

    The match template program is GPU-intensive, and there is the `ORIENTATION_BATCH_SIZE` parameter within the `run_match_template.py` script to help manage GPU resources.
    If you encounter a CUDA out of memory error try decreasing the `ORIENTATION_BATCH_SIZE` parameter.

## Configuration options

A default config file for the match template program is available [here on the GitHub page](https://raw.githubusercontent.com/Lucaslab-Berkeley/Leopard-EM/refs/heads/main/programs/match_template/match_template_example_config.yaml).
This file is separated into multiple "blocks" each configuring distinct portions of the program discussed briefly below.
<!-- Defining and exporting these configurations in terms of Python objects is detailed in [Match Template Configuration](../examples/basic_configuration.ipynb) -->

### Top-level micrograph and template paths

The first two fields in the configuration are define where the simulated 3D reference template and 2D micrograph files are located.
These are both saved in the mrc file format, and the paths need to be updated to match your system and experiment.
Configurations are made on a per-micrograph basis, that is, a new configuration file should be made for each micrograph in your dataset.

```yaml
template_volume_path: /some/path/to/template.mrc
micrograph_path:      /some/path/to/micrograph.mrc
```

!!! note

    The reference template should be simulated under the same conditions as the experimental micrograph (pixel size, cumulative electron exposure, etc.).
    A 3D volume can be simulated from a PDB structure using the [TeamTomo ttsim3d](https://github.com/teamtomo/ttsim3d) Python package.

### Output result files

The next block is the `match_template_result` configuration which defines where to save match template results to disk
Note that these paths need to be writable by the user executing the program, and the `match_template_result.allow_file_overwrite` field needs to be set to `true` if you are overwriting pre-existing result files.
Otherwise the match template program will not proceed if the files already exist.

```yaml
match_template_result:
  allow_file_overwrite: true
  mip_path:                   /some/path/to/output_mip.mrc
  scaled_mip_path:            /some/path/to/output_scaled_mip.mrc
  orientation_psi_path:       /some/path/to/output_orientation_psi.mrc
  orientation_theta_path:     /some/path/to/output_orientation_theta.mrc
  orientation_phi_path:       /some/path/to/output_orientation_phi.mrc
  relative_defocus_path:      /some/path/to/output_relative_defocus.mrc
  correlation_average_path:   /some/path/to/output_correlation_average.mrc
  correlation_variance_path:  /some/path/to/output_correlation_variance.mrc
```

Results are saved as MRC files with positions \( (x, y) \) corresponding to positions in the image.
See [Data Formats](../data_formats.md) for more information.

### Optics group for micrograph parameters

Constructing the appropriate projective filters requires the microscope parameters used to collect the micrograph, namely the defocus of the image.
These microscope parameters are collected under the `optics_group` block with the most commonly modified parameters listed below.

```yaml
optics_group:
  label: some_label
  voltage: 300.0
  pixel_size: 1.06   # in Angstroms
  defocus_u: 5200.0  # in Angstroms
  defocus_v: 4950.0  # in Angstroms
  astigmatism_angle: 25.0  # in degrees
  spherical_aberration: 2.7  # in millimeters
  amplitude_contrast_ratio: 0.07
  ctf_B_factor: 60.0  # in Angstroms^2
```

!!! Note

    The `label` field is currently unused and can be set to any string, but may be integrated into other workflows in the future to differentiate between multiple micrograph and/or refined optical parameters.

### Defocus search space configuration

Defining the defocus search space is configured using the `defocus_search_config` block which defines the exact relative defocus values searched over in units of Angstroms.
For example, to search defocus values ranging across -120 to +120 nm relative to the fitted CTF defocus values in `optics_group`, use the following configuration.

```yaml
defocus_search_config:
  defocus_max:  1200.0  # in Angstroms, relative to the defocus_u and defocus_v values
  defocus_min: -1200.0  # in Angstroms, relative to the defocus_u and defocus_v values
  defocus_step: 200.0   # in Angstroms
  enabled: true
```

!!! Note

    The defocus search can be turned off by changed the `enabled` field from `true` to `false`. This will run the 2DTM search at a single defocus plane corresponding to the fitted CTF defocus values.

### Orientation search space configuration

How points in orientation space (SO(3) space) are sampled is configured using the the `orientation_search_config` block.
At its most basic, only three parameters need considered: `base_grid_method`, `psi_step`, and `theta_step`.
The `uniform` grid sampling method uses the Hopf Fibration from the [torch-so3 package](https://github.com/teamtomo/torch-so3) to generate a uniformly distributed set of points across SO(3) space and should be the first method you try when running a 2DTM search.
There is also `healpix` option which uses a [HEALPix](https://healpix.jpl.nasa.gov) discretization of the sphere to sample orientation space.

The other two fields define how finely orientation space is sampled with lower angular values corresponding to a larger number of samples.
The `psi_step` field controls how many degrees are between two consecutive **in-plane** rotations while `theta_step` corresponds to the step size for **out-of-plane** rotations.

??? info "A note on default angular sampling parameters"

    We have empirically found an angular sampling of `psi_step: 1.5` and `theta_step: 2.5` works well for maximizing 60S ribosomal subunit detections *in situ*. These values do not come from some underlying theory. Increasing angular sampling from these parameters in the 60S case does not lead to more particle detections, but it does raise the computational cost and 2DTM noise floor. Decreasing angular sampling leads to particle "misses" which is undesirable.

    If you are searching for a structure other than the 60S ribosome, then an alternate set of angular sampling parameters may work better. Finding these parameters may require some trial and error, and the search parameters must also strike a balance between computational cost, minimizing the noise floor, and maximizing sensitivity.

Below, we show the `orientation_search_config` with the above parameters

```yaml
orientation_search_config:
  base_grid_method: uniform
  psi_step: 1.5    # in degrees
  theta_step: 2.5  # in degrees
```

### Configuring the pre-processing filters

Pre-processing filters are applied to both the image and template in Fourier space.
Below, we briefly discuss the parameter choices for the whitening and band-pass filters, the two most commonly used 2DTM filters. There are two additional pre-processing filters, phase-randomization & arbitrary curve, whose configuration is discussed [here](../examples/01_basic_configuration.ipynb).
In most cases, the default values should suffice, but nevertheless the knobs to tweak how calculations are performed are included for completeness' sake.

#### Whitening filter

The whitening filter, with parameters defined under `preprocessing_filters.whitening_filter`, flattens the 1D power spectrum of the image so each frequency component contributes equally to the cross-corelation; the same filter is applied to template projections.
The whitening filter is enabled by default and necessary to compensate for the the strong low-frequency components of *in situ* cryo-EM images,[^1] but the filter can be disabled by changing `enabled: true` to `enabled: false`.
Changing the `do_power_spectrum` to `false` will calculate the whitening filter based on the amplitude spectrum instead of the power spectrum, but we don't observe a major difference when swapping this parameter.

The `whitening_filter.max_freq` field defines the maximum spatial frequency considered (in terms of Nyquist) when calculating the whitening filter; the default of `0.5` should perform well in most cases.
Similarly, keeping the default `num_freq_bins: null` will choose the number of frequency bins automatically based on input image shape.
Values between 500-2,000 are generally good for typical cryo-EM images with ~4,000 pixels along each dimension.

#### Bandpass filter

Bandpass filtering is disabled by default but can be turned on by changing `enabled: false` to `enabled: true`.
Note that the `high_freq_cutoff` and `low_freq_cutoff` fields are both defined in terms of the Nyquist frequency with values of `null` corresponding to no cutoff.
For example, `high_freq_cutoff: 0.5` and `low_freq_cutoff: null` would correspond to a low-pass filter to the Nyquist frequency of the image.
Filtering to a specific resolution in terms of Angstroms means doing some math before populating the fields.

```yaml
preprocessing_filters:
  whitening_filter:
    enabled: true
    do_power_spectrum: true
    max_freq: 0.5  # In terms of Nyquist frequency
    num_freq_bins: null
  bandpass_filter:
    enabled: false
    falloff: null
    high_freq_cutoff: null  # Both high/low in terms of Nyquist frequency
    low_freq_cutoff: null   # e.g. low-pass to 3 Å @ 1.06 Å/px would be 1.06/3 = 0.353
```

### Configuring GPUs for a match template run

The final block of the match template configuration file is used to choose which GPUs will run on.
The `num_cpus` field is used to control how many concurrent streams the `match_template` program will run on for each device.
Values between 1 and 8 are generally advised, but mileage may vary on your system.
The `gpu_ids` field is a list of integers defining which GPU device index(s) the program will target.
A value of `gpu_ids: "all"` is a special case which will target all GPUs on the machine (note that only single-node execution is currently supported).
The example below will distribute work equally between the zeroth and first GPU device which need discoverable by PyTorch.

```yaml
computational_config:
  gpu_ids:
  - 0
  - 1
  num_cpus: 4  # 4 streams x 2 devices = 8 total streams
```

!!! warning "RuntimeError: CUDA error: invalid device ordinal"

    If you encounter the error `RuntimeError: CUDA error: invalid device ordinal`, then you've probably listed more GPU devices than are on your machine!
    Check how many GPUs you have (for example with `nvidia-smi`) and update the `gpu_ids` field accordingly.

## Running the match template program

Once you've configured a YAML file, running the match template program is fairly simple.
We have an example script, [`Leopard-EM/programs/run_match_template.py`](https://github.com/Lucaslab-Berkeley/Leopard-EM/blob/main/programs/match_template/run_match_template.py), which processes a single micrograph against a single reference template.
In addition to the YAML configuration path, there are the additional constant variables `DATAFRAME_OUTPUT_PATH` and `ORIENTATION_BATCH_SIZE` listed near the top of the Python script.
The latter variable should be adjusted based on available GPU memory while the former defines where an output csv file containing located particles should be saved.

We have not specified any arguments in the line 54 of this script: `df = mt_manager.results_to_dataframe()`.
You can have more control over the peak extraction by specifying a `locate_peak_kwargs` dictionary which can control the desired number of false positives in the search or pick particles given a pre-defined z-score cutoff.
For example,

```python
# Select peaks bases on a number of false positives
df = mt_manager.results_to_dataframe(locate_peaks_kwargs={"false_positives": 1.0})


# Uses a pre-defined z-score cutoff for peak calling
df = mt_manager.results_to_dataframe(locate_peaks_kwargs={"z_score_cutoff": 7.8})
```

Currently, Leopard-EM uses a false-positive rate of 1 per micrograph by default to differentiate between true particles and background.

### Match template output files

The provided program script will output the statistics maps over the image for the search as well as a Pandas DataFrame with compacted information on found particles.
These data can be passed onto downstream analysis, for example the refine template program.
More detail about these data and their formats is on the [Leopard-EM data formats page](../data_formats.md).


## Mathematical description

Described succinctly using mathematics, the match template constructs the orientational search space, \( \mathbf{R} = \{ R_1, R_2, \dots, R_n\} \), and relative defocus search space, \( \mathbf{Q} = \{ \Delta f_1, \Delta f_2, \dots, \Delta f_m\} \), to generate the CTF-convolved projections of a reference template:

$$
\mathbf{P} = \mathbf{Q} \times \mathbf{R} = \{ p_{11}, \dots, p_{n1}, \dots, p_{nm} \}
$$

The "best" projection, based on cross-correlation with the cryo-EM image \( I \) (which is the same shape as the projection), is found,

$$
\begin{align}
    \tilde{p}_{ij} = \text{argmax}_{p \in \mathbf{P}} \left(p \cdot I \right)
\end{align}
$$

where the indexes \( i \) and \( j \) correspond to the orientation and relative defocus which generated the "best" projection.
Orientations of the best projection are returned as Euler angles \( (\phi, \theta, \psi) \) in the ZYZ format.
Note that in practice, this search is done simultaneous for all positions, \( (x, y) \), within an image much larger than the projection using the convolution theorem, but here we focus on a single location in the image.

The match template program also returns the Maximum Intensity Projection (MIP) which is the cross-correlation score of the best projection as well as a z-score (also called "scaled mip") on a per-pixel level.

$$
\begin{align}
    \text{MIP} &= \tilde{p}_{ij} \cdot I \\[6pt]
    \text{z-score} &= \frac{
        \text{MIP} - \mu(\{ p \cdot I : \forall p \in \mathbf{P} \})}
        {\sigma(\{ p \cdot I : \forall p \in \mathbf{P} \})}
\end{align}
$$

There are other steps, namely Fourier filtering, which will be discussed in a later theory section of the documentation.


[^1]: J Peter Rickgauer, Nikolaus Grigorieff, Winfried Denk (2017) Single-protein detection in crowded molecular environments in cryo-EM images eLife 6:e25648, https://doi.org/10.7554/eLife.25648