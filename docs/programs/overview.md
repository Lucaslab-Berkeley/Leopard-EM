---
title: Overview of Leopard-EM programs
description: A basic overview of each of the built-in programs in Leopard-EM
---

# Programs at a glance

The Leopard-EM package currently has five main programs which are easily user-configurable through YAML files and are runnable from Python scripts. These five programs encompass a variety of 2DTM data processing workflows. Here, we provide a brief overview of each program's functionality and the necessary input data for that program. Detailed information on configuring and running each program can be found on their respective pages, linked below.

!!! note

    The functionality of Leopard-EM extends beyond just these five programs. We encourage users to leverage the modular Pydantic models and backend functions to build more complex cryo-EM workflows.

## Match Template

The `match_template` program searches for regions in a 2D micrograph containing a macromolecule using a simulated reference structure.
The configuration parameters define how the search space (orientation & defocus) is sampled, what pre-processing filters to apply, and where to save the result files from the program.
The [match template program details](match_template.md) contains further information on configuring the program.

The required inputs (besides config fields) for `match_template` are:

* A processed 2D micrograph (aligned and summed),
* The estimated CTF defocus parameters for that micrograph, and
* A simulated 3D reference template (see package [ttsim3d](https://github.com/teamtomo/ttsim3d) for simulating reference templates).


## Refine Template

After the `match_template` program identifies particles from a "coarse" search, the `refine_template` program locally refines the orientation, location, and defocus parameters on a per-particle basis.
The desired degree of angular & defocus accuracy are configurable parameters, and many of the other fields should be copied directly from the `match_template` config.
The [refine template program details](refine_template.md) has further information on configuring the program.

The required inputs for `refine_template` are:

* A simulated 3D reference template (see package [ttsim3d](https://github.com/teamtomo/ttsim3d) for simulating reference templates), and
* The csv file of particle locations and orientations output from the `match_template` program.


## Optimize Template

The 2DTM SNR is extremely sensitive to incorrect pixel size in the reference template structure.
Since not all deposited structures have accurate pixel sizes, we include the `optimize_template` program to maximize 2DTM SNR values by varying the pixel size of the map.
A set of pre-identified particles are used (rather than doing a full-image search) for computational efficiency.
The [optimize template program details](optimize_template.md) contains further information on configuring the program.

The required inputs for `optimize_template` are:

* Simulation configuration for the [ttsim3d](https://github.com/teamtomo/ttsim3d) package including the reference structure, and
* The csv file of particle locations and orientations output from either the `match_template` or `refine_template` program.


## Constrained Search

The `constrained_search` program uses pre-identified locations and orientations of a reference particle (lets call it particle A) to constrain the search space for another particle (called particle B); constraining the search space increases the sensitivity of 2DTM.
This is useful when particle A is easier to identify with 2DTM and when prior knowledge about the spatial relationship between particles A and B is available.
<!-- TODO: Uncomment this when constrained search is finished -->
<!-- See the [constrained 40S ribosome example](../examples/constrained_search/constrained_search.ipynb) for a detailed demonstration of setting up and analyzing a constrained 2DTM search.
Along with the tutorial above, the [constrained search program details](constrained_search.md) contains further information on configuration the program. -->

The [constrained search program details](constrained_search.md) contains further information on configuration the program.

The required inputs for `constrained_search` are:

* A simulated 3D reference template (see package [ttsim3d](https://github.com/teamtomo/ttsim3d)) of the constrained particle to search for,
* The relative position, (x, y, z), and relative orientation between the reference and constrained particle,
* A particle stack csv (from `match_template` or `refine_template`) for the reference particle,
* A particle stack csv (from `match_template`) for the constrained particle, and
* Estimates on the relative position & orientation as well as flexibility of the constrained particle.


## Optimize B-Factor script

In addition to the four larger programs, we have a script, [optimize_b_factor.py](https://github.com/Lucaslab-Berkeley/Leopard-EM/blob/main/programs/optimize_b_factor.py), which varies the B-factor applied across a model to maximize one of four SNR metrics. This *ad hoc* B-factor can help account for increased sample structural heterogeneity which is not captured in the input model, or it can help select for particles which closely match the reference template model.

The script uses `match_template` under-the-hood to re-calculate the 2DTM SNR of particles over a variety of CTF B-factors. In addition to parameters modifiable within the script, the program inputs are the same as the [match template program](#match-template).