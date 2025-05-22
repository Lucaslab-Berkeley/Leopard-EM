---
title: The Optimize Template Program
description: Description of the optimize template program
---

# The optimize template program

The optimize template program takes in a particle stack of known particle locations and orientations and adjusts the pixel size of the template to maximize the 2DTM SNR values.
The SNRs from 2DTM are extremely sensitive to the pixel size used to simulate the map.
As a result, we suggest running optimize template early on in data processing to maximize the SNRs.

## Configuration options

A default config file for the optimize template program is available [here on the GitHub page](https://raw.githubusercontent.com/Lucaslab-Berkeley/Leopard-EM/refs/heads/main/programs/optimize_template/optimize_template_example_config.yaml).
This file is separated into multiple "blocks" each configuring distinct portions of the program discussed briefly below.

### Particle stack

The first field we specify is the particle stack to optimize the pixel size with.
With this you must specify the path to the particle stack dataframe (as output from match or refine template), the original template box size in pixels, and the box size to extract the particle with.
This extracted box size must be even, but the absolute values are not important, although they are the same for [Refine Template](../programs/refine_template.md).

```yaml
particle_stack:
  df_path: results/results_goodModel.csv  # Needs to be readable by pandas
  extracted_box_size: [528, 528]
  original_template_size: [512, 512]
```

### Simulator configuration

The optimize template program uses the [TeamTomo ttsim3d](https://github.com/teamtomo/ttsim3d) Python package to simulate maps with different pixel sizes.
Detailed configuration information is described on the above page, but the following is the simulator configuration in brief.

!!! note

    The pixel size specified for the simulator used will be the starting pixel size used for the search.

```yaml
simulator:
  simulator_config:
    voltage: 300.0
    apply_dose_weighting: true
    dose_start: 0.0
    dose_end: 50.0
    dose_filter_modify_signal: "rel_diff"
    upsampling: -1  
    mtf_reference: "falcon4EC_300kv"
  pdb_filepath: "parsed_6Q8Y_whole_LSU_match3.pdb"
  volume_shape: [512, 512, 512]
  b_factor_scaling: 0.5
  additional_b_factor: 0
  pixel_spacing: 0.95
```

### Specifying pixel size search range and steps

The program works in two stages, a coarse and a fine pixel size search.
We first do a coarse search with a larger range and step size, and then use the best pixel size from that search to do localized, fine-grain search.
Note that the min/max values are relative to the pixel size in the simulator configuration.

```yaml
pixel_size_coarse_search:
  enabled: true
  pixel_size_min: -0.05
  pixel_size_max: 0.05
  pixel_size_step: 0.01
pixel_size_fine_search:
  enabled: true
  pixel_size_min: -0.008
  pixel_size_max: 0.008
  pixel_size_step: 0.001
```

This will first search between 0.90 and 1.00 in 0.01 Angstrom steps, before searching with finer steps around the best value.

### Pre-processing filters and computational config

These should be the same as for [Match Template](../programs/match_template.md) run to identify the particles.
