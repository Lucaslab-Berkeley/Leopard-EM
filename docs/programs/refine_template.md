---
title: The Refine Template Program
description: Description of the refine template program and its configuration
---

# The refine template program

Although the match template program samples millions of points across orientation space, this can be considered relatively coarse search compared to the theoretical angular accuracy.
To achieve an even higher angular accuracy, we finely sample SO(3) space locally for each match template identified particle using the `refine_template` program.
A defocus refinement is also included in the `refine_template` program  whose relative sampling is another configurable parameter.

!!! note "Refine template does not find additional particles"

    Refine template uses pre-identified particle locations and orientations from the `match_template` program to refine particle parameters.
    This will increase the 2DTM SNR values of these already identified particles, but this will *not* find any additional peaks by bringing them above the cutoff threshold.

## Configuration options

A default config file for the match template program is available [here on the GitHub page](https://raw.githubusercontent.com/Lucaslab-Berkeley/Leopard-EM/refs/heads/main/programs/refine_template/refine_template_example_config.yaml).
This file is separated into multiple "blocks" each configuring distinct portions of the program briefly discussed below.

### Template volume path

The first field in the example configuration file is `template_volume_path` which is a path to the simulated 3D reference template.
If you are running `refine_template` directly after `match_template`, then this field should be copied directly from your match template configuration YAML.

```yaml
template_volume_path: /some/path/to/template.mrc
```

### Particle stack of particles to refine

The next portion of the example configuration is the `particle_stack` block which is used to extract information from the original micrograph and template matching results.
Within this block, we have the `df_path` field which is a csv file path which contains a complete set of information for particle locations, orientations, paths to 2DTM result files, and more.
This csv file is written when [running the match template program](match_template.md#running-the-match-template-program), and the output path can be directly copied into this configuration field.

!!! note "Refining results from `refine_template`"

    The refine template program writes an updated csv file with refined particle parameters to disk which itself can be fed back into the refinement step.
    That is, the csv file for `df_path` does not need to come from match template.
    Running multiple refinement can be useful to compare between similar reference structures using 2DTM.

The next two fields are `extracted_box_size` and `original_template_size` which together are used to extract regions in the image and statistics maps around a particle.
Set the `original_template_size` field to the same shape as the simulated volume, that is if the 3D mrc file for the reference template is of shape \( (512, 512, 512) \), then this filed should be `original_template_size: [512, 512]`.

We use `extracted_box_size` to allow for some padding around the particle which permits some flexibility in particle location during the refinement step.
Note that the extracted box shape *must* be larger than the original template size and an even integer.
Values around 4-24 pixels larger than the original template size are advised, and going larger can start to slow down computation without providing any sensitivity benefit.

The particle stack block should look something like the following.

```yaml
particle_stack:
  df_path: /some/path/to/particles.csv
  extracted_box_size: [528, 528]
  original_template_size: [512, 512]
```

### Configuring the defocus refinement search

2DTM is highly sensitive to particle defocus, and particle refinement can localize a particle to a higher accuracy than the initial full-orientation search.
The `defocus_refinement_config` block defines what defocus values are searched over relative to the previous best particle defocus.

!!! note "Accuracy of defocus refinement"

    Obtaining highly-accurate per-particle defocus values is dependent on accurate orientation estimations and the quality of experimental data.
    You may find different search parameters work better depending on the sample and reference template structure.

The following configuration will search 100 Angstroms above and below the best particle defocus value in 20 Angstrom increments.

```yaml
defocus_refinement_config:
  enabled: true
  defocus_max:  100.0  # in Angstroms, relative to "best" defocus value in particle stack dataframe
  defocus_min: -100.0  # in Angstroms, relative to "best" defocus value in particle stack dataframe
  defocus_step: 20.0   # in Angstroms
```

Defocus refinement can be turned off by setting `enabled: false`, but is enabled by default.

### Sampling orientation space locally

Orientation space is sampled in fine increments during the refinement step, and this sampling is configured with the `orientation_refinement_config` block.
Here, the in-plane rotation sampling increment is controlled by the `psi_step_fine` field, and the range of in-plane rotations is defined by `psi_step_coarse`.
In the configuration example below, the relative searched relative in-plane rotations would be \( [-1.5, -1.35, \dots, 1.35, 1.5] \) in units of degrees.
The same applies for the out-of-plane rotations controlled by the `theta_step_coarse` and `theta_step_fine` fields.

```yaml
orientation_refinement_config:
  enabled: true
  psi_step_coarse:   1.5   # in degrees
  psi_step_fine:     0.15  # in degrees
  theta_step_coarse: 2.5   # in degrees
  theta_step_fine:   0.25  # in degrees
```

A good way of choosing these parameters is setting the coarse angular step size to the step size used in `match_template` while the fine angular step size is a free parameter to choose based on desired accuracy.
Also, like the defocus refinement search, orientation refinement can be disabled by setting `enabled: false`, but it is enabled by default.

### Varying pixel size during refinement

Since 2DTM is sensitive to accurate pixel sizes, we include a final search space configuration block called `pixel_size_refinement_config`.
Like orientation and defocus refinement, this searches over a uniform grid of pixel sizes relative to the original pixel size (defined in the particle stack csv).
However, pixel size refinement is turned off by default and can be enabled by setting `enabled: true`.

!!! warning "Pixel size refinement vs the `optimize_template` program"

    Pixel size refinement happens on a per-particle basis in the `refine_template` program whereas the `optimize_template` program finds the "best" global pixel size for a reference structure across all particles.
    If you are doubtful of a deposited model's pixel size accuracy (or the relative pixel size of your micrograph), run the `optimize_template` program rather than using template refinement to identify the correct pixel size.

The following is the default pixel size refinement configuration.

```yaml
pixel_size_refinement_config:
  enabled: false
  pixel_size_min: -0.005
  pixel_size_max:  0.005
  pixel_size_step: 0.001
```

### Pre-processing filters applied before search

The `preprocessing_filters` block should be copied directly from the original `match_template` program configuration.
All these parameters are discussed in more detail on the [match template program page](match_template.md#configuring-the-pre-processing-filters).

### Configuring GPUs for a match template run

The refine template program parallelizes across multiple GPUs by splitting which particles are refined across the configured list of GPU devices.
The `num_cpus` field controls how many concurrent streams of work are being submitted to each GPUs; in most cases, a value of `1` or `2` will saturate the GPU and give the best performance, although your mileage may vary.
Like [configuring GPUs for a match template run](match_template.md#configuring-gpus-for-a-match-template-run), GPUs are targeted by their device index or the special string `"all"`
The following configuration will run `refine_template` on GPU zero.

```yaml
computational_config:
  gpu_ids: 0
  num_cpus: 1
```

The following configuration will run `refine_template` on all available GPUs with two streams per GPU.

```yaml
computational_config:
  gpu_ids: "all"
  num_cpus: 2
```

## Running the refine template program

Once you've configured a YAML file, running the refine template program is fairly simple.
We have an example script, [`Leopard-EM/programs/run_refine_template.py`](https://github.com/Lucaslab-Berkeley/Leopard-EM/blob/main/programs/refine_template/run_refine_template.py), which processes single particle stack against a single reference template.
In addition to the YAML configuration path, there are the additional variables `DATAFRAME_OUTPUT_PATH` and `PARTICLE_BATCH_SIZE` near the top of the Python script.
The latter variable is used to process multiple particles at once since we want to maximize hardware utilization.
But this parameter also needs to balance available GPU memory.

The former variable, `DATAFRAME_OUTPUT_PATH`, will write a new particle stack csv file with new columns corresponding to the refined position, orientation, defocus, and pixel size on a per-particle basis.
More details on the particle stack csv format can be found on the [Leopard-EM data formats page](../data_formats.md).
