---
title: Particle Stack Formats
description: CSV vs HDF5 back-ends for particle stacks used by refine template, constrained search, and inspect peaks
---

## Particle stack formats

A `ParticleStack` collects everything needed to re-extract and re-score individual particles identified by `match_template` — their locations, orientations, defocus values, and references to the source micrograph and statistics maps — for use by `refine_template`, `constrained_search`, and the peak/frame inspection programs.
As with match template results, particle stacks support two storage back-ends.

!!! note "Choosing a back-end"

    Both back-ends are subclasses of a shared (non-instantiable) base class and expose the same in-memory API (`get_euler_angles()`, `get_relative_defocus()`, `construct_image_stack(...)`, etc.) — only how the particle table (and optionally particle images) are read from and written to disk differs.
    `ParticleStack` remains available as a backward-compatible alias for `ParticleStackCSV`.

!!! warning "`refine_template` currently only supports the CSV back-end"

    Unlike `match_template`, the `refine_template` program (and `PeakInspectionManager`/`FrameInspectionManager`, which subclass it) is CSV-in, CSV-out: `RefineTemplateManager.particle_stack` is typed as plain `ParticleStack` (i.e. `ParticleStackCSV`), not a CSV/HDF5 union, and `run_refine_template(...)` always writes its refined DataFrame with `to_csv(...)`.
    There is currently no way to pass a `ParticleStackHDF5` into `refine_template` or to have it write an HDF5-backed result directly.
    If you want an HDF5-backed particle stack after refinement, convert the refined CSV manually with `ParticleStackCSV.to_hdf5(...)` as a separate post-processing step (see below).

### CSV back-end (`ParticleStackCSV`)

This is the only Leopard-EM behavior for versions ``<=v1.2``: the particle table is a CSV file (the same DataFrame [written by `match_template` or `refine_template`](../data_formats.md#match-template-dataframe)), and particle images are extracted on demand from the micrograph/statistics-map paths referenced in each row.

```yaml
particle_stack:
  df_path: /some/path/to/particles.csv
  extracted_box_size: [528, 528]
  original_template_size: [512, 512]
```

**Use the CSV back-end when...**

- You're feeding the direct output of `match_template` or `refine_template` into the next program in the pipeline — this is the default hand-off format documented on the [refine template](../programs/refine_template.md#particle-stack-of-particles-to-refine) and [constrained search](../programs/constrained_search.md) program pages.
- You want to inspect or edit particle metadata as a plain-text/CSV table (e.g. in a spreadsheet or with `pandas`) without unpacking an HDF5 file.
- Your source micrographs and MRC statistics maps are expected to stay available at their original paths — the CSV back-end re-reads them each time, so it stays in sync with those files rather than freezing a snapshot.

```python
from leopard_em.pydantic_models.data_structures import ParticleStackCSV

particle_stack = ParticleStackCSV(
    df_path="/some/path/to/particles.csv",
    extracted_box_size=(528, 528),
    original_template_size=(512, 512),
)

# Load the (deduplicated) referenced micrographs, then extract per-particle boxes
images, indices = particle_stack.load_images_grouped_by_column("micrograph_path")
image_stack = particle_stack.construct_image_stack(
    images=images,
    indices=indices,
    extraction_size=particle_stack.extracted_box_size,
)
```

### HDF5 back-end (`ParticleStackHDF5`)

`ParticleStackHDF5` stores the particle table in a single `.h5` file, and can optionally bundle the extracted particle images (`image_stack`) and/or per-particle local correlation statistics (`local_stats`) directly into that same file, so the stack no longer depends on the original micrograph/statistics-map files being available at their recorded paths.

```yaml
particle_stack:
  hdf5_path: /some/path/to/particles.h5
  extracted_box_size: [528, 528]
  original_template_size: [512, 512]
```

**Use the HDF5 back-end when...**

- You want a fully portable, self-contained particle stack — one file you can archive, share, or move to another machine without also shipping every referenced micrograph and statistics map.
- You want per-particle local correlation statistics (`local_stats_correlation_average`/`local_stats_correlation_variance`) stored alongside the particle table rather than recomputed.

#### Two loading modes

`ParticleStackHDF5` supports two mutually exclusive modes, controlled by the `image_stack_stored`/`local_stats_stored` flags (set automatically by `to_hdf5(...)`, and read back by `from_hdf5(...)`):

- **Load from referenced files** (`image_stack_stored=False`): the HDF5 file stores only the particle table; `image_stack`/`local_stats` are computed on demand from the micrograph/statistics-map paths in that table, same as the CSV back-end.
- **Load from HDF5 directly** (`image_stack_stored=True` and/or `local_stats_stored=True`): `image_stack`/`local_stats` are read from the HDF5 datasets, no access to the original micrograph files needed.

#### HDF5 file layout

```text
/ (root)
│  attrs: leopard_em_version, extracted_box_size, original_template_size,
│         image_stack_stored, local_stats_stored,
│         global_whitening_applied, local_whitening_applied,
│         global_normalization_applied, local_normalization_applied
├─ particles/
│      particle_id            (N,)   variable-length str  "{mic_stem}_{idx:05d}"
│      <column>               (N,)   float64 or variable-length str
│      ...
├─ image_stack                (N, box_h, box_w)             float32  [optional]
└─ local_stats/                                                      [optional]
       correlation_average    (N, valid_h, valid_w)         float32
       correlation_variance   (N, valid_h, valid_w)         float32
```

where `valid_h = extracted_box_size[0] - original_template_size[0] + 1` and `valid_w = extracted_box_size[1] - original_template_size[1] + 1` (see the [note on correlation modes](../data_formats.md#a-note-on-correlation-modes-and-output-shapes)).

### Converting a CSV-backed stack to HDF5

`ParticleStackCSV.to_hdf5(...)` is the recommended migration path.
It re-uses the CSV back-end's already-configured extraction settings, generates a `particle_id` for each row, and writes a new `ParticleStackHDF5`:

```python
from leopard_em.pydantic_models.data_structures import ParticleStackCSV

csv_stack = ParticleStackCSV(
    df_path="/some/path/to/particles.csv",
    extracted_box_size=(528, 528),
    original_template_size=(512, 512),
)

# Extract particle images before baking them into the HDF5 file
images, indices = csv_stack.load_images_grouped_by_column("micrograph_path")
csv_stack.image_stack = csv_stack.construct_image_stack(
    images=images,
    indices=indices,
    extraction_size=csv_stack.extracted_box_size,
)

hdf5_stack = csv_stack.to_hdf5(
    "/some/path/to/particles.h5",
    include_image_stack=True,  # requires image_stack to already be populated
    include_local_stats=False,
)
```

### Loading an existing HDF5 particle stack

```python
from leopard_em.pydantic_models.data_structures import ParticleStackHDF5

particle_stack = ParticleStackHDF5.from_hdf5("/some/path/to/particles.h5")

if particle_stack.image_stack_stored:
    image_stack = particle_stack.image_stack
else:
    images, indices = particle_stack.load_images_grouped_by_column("micrograph_path")
    image_stack = particle_stack.construct_image_stack(
        images=images,
        indices=indices,
        extraction_size=particle_stack.extracted_box_size,
    )
```
