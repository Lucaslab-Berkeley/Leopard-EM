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

!!! note "Particle table indexing"

    For both back-ends the in-memory particle table (`get_dataframe_copy()`) has a plain `0..N-1` index, so row labels always equal row positions and line up with the per-particle tensors (`image_stack`, `local_stats`).
    Each particle's unique, human-readable `particle_id` (e.g. `micrograph_00012`) is an ordinary column, but not currently used for indexing.
    Particle indexes passed to particle stack methods (e.g. the groups returned by `load_images_grouped_by_column`) are always integer row positions.

!!! note "`refine_template` and `constrained_search` inputs and outputs"

    `RefineTemplateManager.particle_stack`, `OptimizeTemplateManager.particle_stack`, and `ConstrainedSearchManager.particle_stack_reference`/`particle_stack_constrained` accept either back-end (`AnyParticleStack`), so a `ParticleStackHDF5` can be passed in directly wherever a particle stack input is required. In a YAML config the back-end is chosen by the key used: `df_path` for CSV, `hdf5_path` for HDF5.

    The output format is inferred from the extension of the output path (`.csv`, or `.h5`/`.hdf5`), and otherwise matches the input back-end. An explicit `output_format` that contradicts the extension raises an error.

    `ConstrainedSearchManager.export_results(...)` additionally writes two small CSV sibling tables — `<base>_parameters.csv` (search parameters and the false-positive threshold) and `<base>_above_threshold.csv` (rows above that threshold) — derived from the base of `output_dataframe_path` regardless of the main table's `output_format`.

    `OptimizeTemplateManager`'s optional per-pixel-size diagnostic dumps (`write_individual_csv`) are intermediate debug artifacts, not the program's primary output, and remain hardcoded to CSV.

    To convert an existing CSV-backed refined table to HDF5 after the fact, use `ParticleStackCSV.to_hdf5(...)` as a separate post-processing step (see below), or see [exporting results](#exporting-refined-results) for the general-purpose helper.

### Exporting refined results

`RefineTemplateManager.export_results(...)` and `ConstrainedSearchManager.export_results(...)` build the refined particle table and write it to disk, then **return the newly-written particle stack instance** (a `ParticleStackCSV` or `ParticleStackHDF5`, matching whichever `output_format` was used) — reuse it directly as input to the next program in a pipeline instead of re-reading it from disk:

```python
result_stack = refine_manager.export_results(
    output_dataframe_path="/some/path/to/refined.h5",
    result=refine_result,
    output_format="hdf5",  # omit to match the input particle_stack's back-end
)
# result_stack is a ready-to-use ParticleStackHDF5 — e.g. feed it straight into
# ConstrainedSearchManager(particle_stack_reference=result_stack, ...)
```
Both `ParticleStackCSV` and `ParticleStackHDF5` also expose a symmetric `export_results(...)` method for writing an existing instance's own table back to its own path (`df_path`/`hdf5_path`):

- `ParticleStackCSV.export_results()` — writes the particle table to `df_path`.
- `ParticleStackHDF5.export_results(include_image_stack=False, include_local_stats=False)` — alias for `to_hdf5(...)`.

Internally, the managers build their output through the shared `export_particle_stack(df, output_path, source_particle_stack=None, output_format=None, *, extracted_box_size=None, original_template_size=None)` helper (`leopard_em.pydantic_models.data_structures.export_particle_stack`). It picks the output back-end (explicit `output_format`, else the file extension, else the source's back-end), builds the matching `ParticleStackCSV`/`ParticleStackHDF5` around the table, writes it, and returns it.
Without a `source_particle_stack` (e.g. to turn a match template result table straight into an HDF5 stack), pass the box sizes directly; the pre-processing flags then default to `False`, which is correct for match template output.
Only the particle table is written — stored image stacks and local statistic maps are never carried over, since particle positions change between programs.

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

`ParticleStackHDF5` stores the particle table in a single `.h5` file, and can optionally bundle the extracted particle images (`image_stack`) and/or per-particle local statistic maps (`local_stats`) directly into that same file, so the stack no longer depends on the original micrograph/statistics-map files being available at their recorded paths.

The file records the box sizes and pre-processing state, so pointing a config at an existing file is enough:

```yaml
particle_stack:
  hdf5_path: /some/path/to/particles.h5
  # Optional: override the box sizes recorded in the file. Stored tensors that no
  # longer fit are then ignored (with a warning) and recomputed from the
  # referenced micrographs/statistics maps.
  # extracted_box_size: [528, 528]
  # original_template_size: [512, 512]
  # Optional: set to false to re-extract particles from the micrographs (with global
  # filtering) even if the file stores an image stack.
  # use_stored_image_stack: true
```

**Use the HDF5 back-end when...**

- You want a fully portable, self-contained particle stack — one file you can archive, share, or move to another machine without also shipping every referenced micrograph and statistics map.
- You want per-particle local statistic maps stored alongside the particle table rather than recomputed. `local_stats` is a `dict[str, torch.Tensor]` keyed by `*_path` column name — any subset of `mip_path`, `scaled_mip_path`, `psi_path`, `theta_path`, `phi_path`, `defocus_path`, `correlation_average_path`, `correlation_variance_path` (or all of them) can be stored, not just correlation average/variance.

#### Stored vs referenced data

What a data a particle stack needs is taken from the file if it was stored, and from referenced files otherwise. The `image_stack_stored`/`local_stats_stored` attributes are set from the file and report what it contains.

- **Only the particle table stored**: particle images and statistic maps are extracted on demand from the micrograph/statistics-map paths in the table, same as the CSV back-end.
- **Local statistic maps stored** (`local_stats_stored`): the per-particle correlation mean/standard-deviation crops used by refinement are read from the file instead of re-reading and cropping the full-size maps.
- **Image stack stored** (`image_stack_stored`): `refine_template`, `optimize_template`, and `constrained_search` use the stored particle images directly, so the micrographs are not needed. Whole-micrograph (global) filtering is impossible for pre-extracted particles, so the whitening/bandpass filters are then computed **per particle** (as with `apply_global_filtering: false`), and a warning is emitted if global filtering was requested. Set `use_stored_image_stack: false` to re-extract from the micrographs instead.

Stored tensors are tied to the particle positions they were extracted at. Changing a position column (or the referenced paths) with `set_column(...)` discards the affected stored tensors, and they are recomputed from the referenced files.

To populate `local_stats` before writing, use `get_local_stat_maps(...)` (extracts the valid cross-correlation region around each particle for any `*_path` column) and assign the result:

```python
particle_stack.local_stats.update(particle_stack.get_local_stat_maps())
# or a specific subset:
particle_stack.local_stats.update(
    particle_stack.get_local_stat_maps(columns=["mip_path", "correlation_average_path"])
)
particle_stack.to_hdf5(include_local_stats=True)
```

#### HDF5 file layout

```text
/ (root)
│  attrs: format_version, writer_version, leopard_em_version,
│         extracted_box_size, original_template_size,
│         image_stack_stored, local_stats_stored,
│         global_whitening_applied, local_whitening_applied,
│         global_normalization_applied, local_normalization_applied
├─ particles/
│      particle_id            (N,)   variable-length str  "{mic_stem}_{idx:05d}"
│      <column>               (N,)   native int/float/bool, or variable-length str
│      ...                           (dataset attr `encoding`: "numeric" | "str")
├─ image_stack                (N, box_h, box_w)             float32  [optional]
└─ local_stats/                                                      [optional]
       <column>               (N, valid_h, valid_w)         float32
       ...                    -- one dataset per entry in `local_stats` at write
                                  time, named after its column (e.g. `mip_path`,
                                  `correlation_average_path`)
```

where `valid_h = extracted_box_size[0] - original_template_size[0] + 1` and `valid_w = extracted_box_size[1] - original_template_size[1] + 1` (see the [note on correlation modes](../data_formats.md#a-note-on-correlation-modes-and-output-shapes)).

String columns store missing values as empty strings, and list/dict values (e.g. `mag_matrix`, Zernike coefficients) as JSON text; they are read back as strings, the same as from a CSV file.
`particle_id` values are `{mic_stem}_{idx:05d}`; when different micrographs share a file name, a short hash of the full path is added to the stem to keep IDs unique.
Files written by Leopard-EM v1.3 (no `format_version` attribute) are still read.

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

# Extract particle images before baking them into the HDF5 file. Reflect padding
# matches how refine_template extracts particles from micrographs.
images, indices = csv_stack.load_images_grouped_by_column("micrograph_path")
csv_stack.construct_image_stack(
    images=images,
    indices=indices,
    extraction_size=csv_stack.extracted_box_size,
    padding_mode="reflect",
)
# Optionally also store the correlation statistics needed for refinement
csv_stack.local_stats.update(
    csv_stack.get_local_stat_maps(
        columns=["correlation_average_path", "correlation_variance_path"]
    )
)

hdf5_stack = csv_stack.to_hdf5(
    "/some/path/to/particles.h5",
    include_image_stack=True,  # requires image_stack to already be populated
    include_local_stats=True,  # requires local_stats to already be populated
)
```

### Loading an existing HDF5 particle stack

```python
from leopard_em.pydantic_models.data_structures import ParticleStackHDF5

# Equivalent to ParticleStackHDF5.from_hdf5("/some/path/to/particles.h5")
particle_stack = ParticleStackHDF5(hdf5_path="/some/path/to/particles.h5")

image_stack = particle_stack.get_stored_image_stack()
if image_stack is None:
    images, indices = particle_stack.load_images_grouped_by_column("micrograph_path")
    image_stack = particle_stack.construct_image_stack(
        images=images,
        indices=indices,
        extraction_size=particle_stack.extracted_box_size,
    )
```
