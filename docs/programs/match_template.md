---
title: The Match Template Program
description: Description of the match template program
---

# The match template program

The match template program takes in a micrograph and a simulated reference structure along with search parameters to find locations in the image which agree with expected 2D projections of this reference structure.
Notably, the match template program simultaneously find the orientations of these particles as well as the depth of the particle within the sample.

## Configuration options

A default config file for the match template program is available [here on the GitHub page](https://raw.githubusercontent.com/Lucaslab-Berkeley/Leopard-EM/refs/heads/main/programs/match_template/match_template_example_config.yaml).
This file is separated into multiple "blocks" each configuring distinct portions of the program discussed briefly below.
Defining and exporting these configurations in terms of Python objects is detailed in [Match Template Configuration](../examples/basic_configuration.ipynb)

### Top-level micrograph and template paths

The first two fields in the configuration are paths to the 3D reference template and 2D micrograph saved as MRC files.
Update these paths based on your system/experiment.

```yaml
template_volume_path: /some/path/to/template.mrc
micrograph_path:      /some/path/to/micrograph.mrc
```

Note the reference template should be simulated under the same conditions as the experimental micrograph (e.g. total exposure).
A 3D volume can be simulated from a PDB structure using the [TeamTomo ttsim3d](https://github.com/teamtomo/ttsim3d) Python package.

### Output result files

The next block is the `match_template_result` configuration which defined the output paths for the match template program results.
Note that these paths need to be writable by the user executing the program, and the `match_template_result` needs to be set to `true` if you are overwriting pre-existing result files.

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

Constructing the projective filters requires knowledge of what microscope parameters were used to collect the micrograph, namely the defocus of the image.
These microscope parameters are collected under the `optics_group` block with the most common parameters listed below.

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

Note the `label` field is currently unused, but could be integrated in the future to differentiate between multiple micrographs and/or refined optical parameters.

### Defocus search space configuration

Defining the defocus search space is configured using the `defocus_search_config` block which defines the minimum, maximum, and step size of defocus values searched over in units of Angstroms.

```yaml
defocus_search_config:
  defocus_max:  1200.0  # in Angstroms, relative to the defocus_u and defocus_v values
  defocus_min: -1200.0  # in Angstroms, relative to the defocus_u and defocus_v values
  defocus_step: 200.0   # in Angstroms
  enabled: true
```

Note that these defocus values are relative to `optics_group.defocus_u` and `optics_group.defocus_v` values.
Also, the defocus search can be turned off by changing `enabled: true` to `enabled: false`.

### Orientation search space configuration

Defining the orientation search space is configured using the `orientation_search_config` block which defines the orientation sampling parameters.
We find that a psi step size of 1.5 degrees and theta step size of 2.5 degrees with a uniform base grid works well when searching for ribosomes, but other sized structures may need these parameters adjusted.
There is also the `"healpix"` option for the `base_grid_method` field which uses the [HEALPix](https://healpix.jpl.nasa.gov) discretization of the sphere to sample orientation space.
Also, the underlying [torch-so3 package](https://github.com/teamtomo/torch-so3) supports multiple particles symmetries, and the use if this in Leopard-EM is discussed [here](../examples/basic_configuration.ipynb).


```yaml
orientation_search_config:
  base_grid_method: uniform
  psi_step: 1.5      # in degrees
  theta_step: 2.5  # in degrees

```

### Configuring the pre-processing filters

We also include some configuration options for pre-processing Fourier filters applied to both the image and template.
Below, we briefly discuss the parameter choices for the whitening and band-pass filters -- the two most common filter types; there are two additional pre-processing filters (phase-randomization and arbitrary curve) discussed [here](../examples/basic_configuration.ipynb).
In most cases, the default values should suffice, but nevertheless the knobs to tweak how calculations are performed are included for completeness' sake.

The whitening filter, with parameters defined under `preprocessing_filters.whitening_filter`, flattens the 1D power spectrum of the image so each frequency component contributes equally to the cross-corelation; the same filter is applied to template projections.
The whitening filter is enabled by default and necessary to compensate for the the strong low-frequency components of *in situ* cryo-EM images,[^1] but the filter can be disabled by changing `enabled: true` to `enabled: false`.
Changing the `do_power_spectrum` to `false` will calculate the whitening filter based on the amplitude spectrum instead of the power spectrum.
The `whitening_filter.max_freq` field defines the maximum frequency considered (in terms of Nyquist) when calculating the whitening filter; the default of `0.5` should perform well in most cases.
Similarly, keeping the default `num_freq_bins: null` will choose the number of frequency bins automatically based on input image shape.
Values between 500-2,000 are generally good.

Bandpass filtering is disabled by default but can be turned on by changing `enabled: false` to `enabled: true`.
Note that the `high_freq_cutoff` and `low_freq_cutoff` fields are both defined in terms of the Nyquist frequency with values of `null` corresponding to no cutoff.
For example, `high_freq_cutoff: 0.5` and `low_freq_cutoff: null` would correspond to a low-pass filter to the Nyquist frequency of the image.
Filtering to a specific resolution in terms of Angstroms means doing some basic math to populate these fields.

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
The `num_cpus` field can currently be ignored and just set to `1`.
The `gpu_ids` field is a list of integers defining which GPU device index(s) the program will target.
Configuring the search to be run on the first two GPUs is shown below.

```yaml
computational_config:
  gpu_ids:
  - 0
  - 1
  num_cpus: 1
```

## Running the match template program

Once you've configured a YAML file, running the match template program is fairly simple.
We have an example script, [`src/programs/run_match_template.py`](https://github.com/Lucaslab-Berkeley/Leopard-EM/blob/main/src/programs/run_match_template.py), which processes a single micrograph against a single reference template.
Again, you will need to simulate a 3D electron scattering potential from a PDB file (for example with the [ttsim3d](https://github.com/teamtomo/ttsim3d) package) before running the script.

We have not specified any arguments in the line `df = mt_manager.results_to_dataframe()`.
You can have more control over the peak extraction by specifying a locate_peak_kwargs dictionary.

  `df = mt_manager.results_to_dataframe(locate_peaks_kwargs={"false_positives": 1.0})`

or 

  `df = mt_manager.results_to_dataframe(locate_peaks_kwargs={"z_score_cutoff": 7.8})`


### Match template output files

The above script will output the statistics maps over the image for the search as well as a Pandas DataFrame with compacted information on found particles.
These data can be passed onto downstream analysis, for example the refine template program.
More detail about these data is on the [data formats page](../data_formats.md).


## Mathematical description

Described succinctly using mathematics, the match template constructs the orientational search space, \( \mathbf{R} = \{ R_1, R_2, \dots, R_n\} \), and relative defocus search space, \( \mathbf{Q} = \{ \Delta f_1, \Delta f_2, \dots, \Delta f_m\} \), to generate the CTF-convolved projections searched over:

$$
\mathbf{P} = \mathbf{Q} \times \mathbf{R} = \{ p_{11}, \dots, p_{n1}, \dots, p_{nm} \}
$$

The "best" projection, based on cross-correlation with the cryo-EM image \( I \), is found,

$$
\begin{align}
    \tilde{p}_{ij} = \argmax_{p \in \mathbf{P}} \left(p \cdot I \right)
\end{align}
$$

where the indexes \( i \) and \( j \) correspond to the orientation and relative defocus which generated the "best" projection.
Orientations of the best projection are returned as Euler angles \( (\phi, \theta, \psi) \) in the ZYZ format.
Note that this search is done simultaneous for all positions,  \( (x, y) \) within the image using the convolution theorem, but here we focus on a single location in the image.
The match template program also returns the Maximum Intensity Projection (MIP) which is the cross-correlation score of the best projection as well as a z-score on a per-pixel level.

$$
\begin{align}
    \text{MIP} &= \tilde{p}_{ij} \cdot I \\[6pt]
    \text{z-score} &= \frac{
        \text{MIP} - \mu(\{ p \cdot I : \forall p \in \mathbf{P} \})}
        {\sigma(\{ p \cdot I : \forall p \in \mathbf{P} \})}
\end{align}
$$

There are other steps, namely Fourier filtering, which are discussed further in the theory portion of the documentation.


[^1]: J Peter Rickgauer, Nikolaus Grigorieff, Winfried Denk (2017) Single-protein detection in crowded molecular environments in cryo-EM images eLife 6:e25648, https://doi.org/10.7554/eLife.25648