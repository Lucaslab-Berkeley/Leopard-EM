---
title: The Constrained Search Program
description: Description of the constrained search program
---

# The constrained search program

The constrained search program takes in a particle stack of known particle locations and orientations and searches for another particle based on these.
This allows us to perform for fewer cross correlations, reducing the noise and thus increasing our sensitivity so we can detect smaller proteins in situ.

## Configuration options

A default config file for the constrained search program is available [here on the GitHub page](https://raw.githubusercontent.com/Lucaslab-Berkeley/Leopard-EM/refs/heads/main/programs/constrained_search/constrained_search_example_config.yaml).
This file is separated into multiple "blocks" each configuring distinct portions of the program discussed briefly below.

### Top-level template paths

The first two fields in the configuration is the path to the 3D reference template  MRC file
The template should be of the particle we want to search for with the constrained search.

```yaml
template_volume_path: /some/path/to/template.mrc
```

### Top-level center vector

The center vector is the vector that points from the reference particle to the constrained particle when all euler angles are 0 (default orientation).
This can be calculated using the script [get_center_vector.py](https://raw.githubusercontent.com/Lucaslab-Berkeley/Leopard-EM/refs/heads/main/programs/constrained_search/utils/get_center_vector.py).

```yaml
center_vector: [53.658134, 82.582367, 47.170788]
```

### Particle stacks for the reference and constrained particle

You must provide particle stacks for the constrained particle as well as the reference particle.
The euler angles and locations are taken from the reference particle stack, and the mean and variance are taken from the constrained particle stack input, allowing us to accurately calculate a z-score.
The extracted box size determines how many pixels will be searched over and is calculated as (extracted_size - original_size +1).
Since we want as few cross-correlations as possible, this should be set as low as possible, ensuring that the particle can still be found within this range.

```yaml
particle_stack_reference: # This is from the reference particles
  df_path: /some/path/to/particles.csv  # Needs to be readable by pandas
  extracted_box_size: [520, 520]
  original_template_size: [512, 512]
particle_stack_constrained: # This is from the constrained particles
  df_path: /some/path/to/particles.csv  # Needs to be readable by pandas
  extracted_box_size: [520, 520]
  original_template_size: [512, 512]
```

### Orientation refinement configuration

We must specify what angular space to perform the orientation search over.
The first thing we must specify is the primary rotation axis, which we set as the Z axis.
If unknown, this can be calculated using two PDB models (one rotated and one unrotated) using the script [get_rot_axis.py](https://raw.githubusercontent.com/Lucaslab-Berkeley/Leopard-EM/refs/heads/main/programs/constrained_search/utils/get_center_vector.py).

As well as searching over one rotation axis, we can search over a second axis, which by default is the y axis.
This can be changed to any axis orthogonal to the primary axis by specifying a `roll_axis` and using the `base_grid_method: roll`.
If a second axis is not known, a roll axis search can be performed (`search_roll_axis: true`) with a specified `roll_step`.

The most important parameters to specify is the range and step size for the psi and theta searches.
The psi angles are rotations around the Z-axis, and theta rotations around the orthogonal axis.
Since psi and phi are redundant for small angular searches, we usually do not need to search over phi.

```yaml
orientation_refinement_config:
  enabled: true
  rotation_axis_euler_angles: [0.0, 0.0, 0.0] # This is the rotation axis
  base_grid_method: uniform
  psi_step: 1.0   # psi in degrees
  theta_step: 1.0   # theta and phi in degrees
  phi_min: -0.0
  phi_max: 0.0
  theta_min: -8.0
  theta_max: 2.0
  psi_min: -13.0
  psi_max: 2.0
  search_roll_axis: false
  roll_axis: [0,1] # [x,y] This defines the roll axis (orthogonal to the rotation axis).
  roll_step: 2.0 
```

### Pre-processing filters and computational config

These should be the same as for [Match Template](../programs/match_template.md).



