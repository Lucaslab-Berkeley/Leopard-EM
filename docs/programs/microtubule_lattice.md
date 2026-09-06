---
title: Microtubule lattice analysis
description: Reading filament geometry out of a constrained 2DTM search
---

# Microtubule lattice analysis

A constrained `match_template` search tells you *where* a microtubule is. This analysis
reads its *structure* out of the same run: which way it points, how its lattice is
spaced, and how many protofilaments it has.

The route runs through the sparse [`CorrelationTable`](../data_formats/match_template_results.md#correlation-table-sparse-detections)
rather than the per-pixel statistics maps, because the maps keep only the single best
hypothesis at each pixel — so the near and far wall of a tube annihilate each other, and
every sub-maximal orientation is lost.

## Why orientation-aware peaks

A filament template is built with its axis along template z, which gives the matched
Euler angles direct geometric meaning:

| angle | meaning for a filament |
|---|---|
| `psi` | in-plane direction of the filament; its two poles are the two polarities |
| `phi` | roll about the tube axis — which protofilament, and near versus far wall |
| `theta` | tilt of the axis out of the image plane (90° = in-plane) |

`find_peaks_orientation_aware` picks peaks in position **and** orientation: one detection
suppresses another only when they are close in *both*. Detections that overlap in
`(x, y)` but disagree about angle therefore both survive, which is what keeps the two
walls of a tube apart.

Suppression radii are bracketed from below by the width of one particle's correlation
lobe and from above by the closest spacing you need to resolve. The lobe width is set by
the template's *resolution content*, not its box size — a 600-pixel microtubule template
still gives a lobe only a few pixels across. `TemplateLatticeGeometry.suggested_xy_radius_px`
and `suggested_angular_radius_deg` derive both from a lattice.

## Calibrating the template

`TemplateLatticeGeometry` carries the one thing the angles do not: where the filament
axis sits *inside* the template box. A template built from a complete ring is
axis-centred; one built from a patch of lattice is offset, and that offset rotates with
`phi`. Undoing it turns every peak — whichever protofilament and whichever wall — into an
independent prediction of the same axis, and the scatter of those predictions about one
line is a free consistency check on the whole chain.

For the reference repeat, prefer `rise_from_template_autocorrelation` over the deposited
model. It measures the template as actually rendered, uses the same estimator applied to
the data so estimator bias partly cancels, and returns pixels — so the resulting ratio
carries no pixel-size assumption.

## Choosing a template

The three template shapes are genuinely complementary and cannot be merged:

| template | `phi` | position | good for |
|---|---|---|---|
| complete ring | undetermined (N-fold pseudo-symmetric) | excellent | axis, lattice spacing, protofilament number |
| off-axis patch | well determined | mediocre | per-protofilament information |
| axis-centred patch | degenerate | mediocre | nothing — avoid |

Centring a patch on the axis is counter-productive: rolling `phi` then sweeps it around
the tube while the box centre stays put, so many orientations place mass in overlapping
positions. Keeping the patch off-centre is what makes its orientation well determined.

## What you get

Run [`programs/microtubule/run_microtubule_lattice.py`](https://github.com/Lucaslab-Berkeley/Leopard-EM/blob/main/programs/microtubule/run_microtubule_lattice.py).
Positions and orientations are read from `refined_*` columns when present, so run
`refine_template` on the peak table first.

- **Polarity** — reliable. The two `psi` poles separate by orders of magnitude whenever
  the template resolves the filament's direction.
- **Lattice spacing** — reliable, reported both absolutely and as a ratio against the
  template. A 2DTM spacing measurement is fundamentally the product (true spacing) ×
  (assumed pixel size), so an externally calibrated pixel size is a prerequisite for
  quoting an absolute number; without one, only the ratio is meaningful, and it is the
  form in which a later calibration becomes a single multiplication. Note the pixel size
  recorded in an MRC header is often the nominal value rather than a calibrated one.

  Two further systematics matter at the sub-percent level. **Out-of-plane tilt**
  shortens the projected repeat by `sin(theta)`; pass `theta_deg` to
  `estimate_lattice_rise` to divide it out (1.5% already at `theta` = 80°).
  **Magnification anisotropy** biases the repeat by an amount that depends on the
  filament's in-plane direction, and is not corrected unless a `mag_matrix` is
  configured. Measuring the same filament with two different templates is a useful
  internal check: they share the pixel size but little else.
- **Protofilament number** — indicative only. Check the margin over the runner-up; a
  narrow margin means inconclusive, and competing whole templates of different
  protofilament number is the more reliable route.

## The seam, and why it is not attempted

Locating a seam means telling the two halves of the repeat apart — for a microtubule,
alpha from beta. Three routes are closed:

- **By density.** Alpha and beta only separate below about 3–4 Å, which 2DTM does not
  reach on this kind of data. `subunit_contrast` measures how far away a given template
  is: a value near 1 means a half-repeat shift correlates as well as a whole one, and
  since that is the template against itself it is a noise-free upper bound. No amount of
  data improves it.
- **By monomer geometry.** The monomer lattice closes exactly — a microtubule rises
  three monomers per turn — so there is no monomer-level discontinuity anywhere on the
  tube. `turns_are_closed_in_repeats` shows this directly.
- **By searching `phi` with a seamed template.** A symmetrised model's seam is a pure
  labelling flip with no structural distortion, so rotating the template changes nothing
  detectable.

A fourth route remains open in principle: the two half-steps of a repeat are unevenly
spaced (about 41 and 43 Å in tubulin), which is a *positional* signature independent of
resolution. Exploiting it needs sub-Angstrom axial precision, roughly a factor of two
beyond what refinement currently delivers.
