---
title: Programs Landing Page
description: What each of the programs in Leopard-EM do and how to configure inputs.
---

# Programs at a glance

The Leopard-EM package currently has 4 programs that are easily user-configurable and callable.
Each of these programs is built from modular components, namely Pydantic models for input/output parsing and Python functions for data processing; building upon these modular components for more complex workflows is both possible and encouraged.

I want to
---------

- find particles in cryo-EM images using a reference template --> `match_template`
- refine the location, orientations, and defocus values of particles --> `refine_template`
- search over pixel size *for the micrograph* using 2DTM z-scores as the metric --> `refine_template`
- search for similar but slightly different structures using already template matched results --> `refine_template`
- optimize the pixel size *for the reference template* using 2DTM z-scores as the metric --> `optimize_template`
- adjust b-factors added to the template to increase number of identified particles --> `optimize_B_factor`

More details on each of the programs can be found on their respective pages, linked below.

- [Match Template](match_template.md)
