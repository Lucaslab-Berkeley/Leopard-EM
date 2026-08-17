---
title: Match Template Result Formats
description: MRC vs HDF5 back-ends for match template statistics maps, and the CorrelationTable
---

## Match template result formats

The `match_template` program supports two storage back-ends for its output statistics maps, plus an optional sparse `CorrelationTable` of every detection which crossed a correlation threshold during the search.
This page describes both result back-ends, when to reach for each one, and the `CorrelationTable` format, with minimal code examples for exporting and loading each.

!!! note "Choosing a back-end"

    The back-end is selected by which class you instantiate for the `match_template_result` field: `MatchTemplateResultMRC` or `MatchTemplateResultHDF5`.
    Both are subclasses of a shared (non-instantiable) base class and expose the same in-memory tensor attributes (`mip`, `scaled_mip`, `correlation_average`, `correlation_variance`, `orientation_psi`, `orientation_theta`, `orientation_phi`, `relative_defocus`) — only how they're read from and written to disk differs.
    `MatchTemplateResult` remains available as a backward-compatible alias for `MatchTemplateResultMRC`.

### MRC back-end (`MatchTemplateResultMRC`)

This is the _only_ Leopard-EM behavior for versions ``<=v1.2``: each of the eight statistics maps is written as its own [MRC format](https://www.ccpem.ac.uk/mrc-format/mrc2014/) file.

```yaml
match_template_result:
  allow_file_overwrite: true
  mip_path:                  ./output_mip.mrc
  scaled_mip_path:           ./output_scaled_mip.mrc
  correlation_average_path:  ./output_correlation_average.mrc
  correlation_variance_path: ./output_correlation_variance.mrc
  orientation_psi_path:      ./output_orientation_psi.mrc
  orientation_theta_path:    ./output_orientation_theta.mrc
  orientation_phi_path:      ./output_orientation_phi.mrc
  relative_defocus_path:     ./output_relative_defocus.mrc
```

**Use the MRC back-end when...**

- You want to open individual statistics maps directly in other cryo-EM tools (IMOD, ChimeraX, RELION, etc.) without going through Leopard-EM or Python.
- You're following an existing pipeline or set of scripts built around per-statistic MRC paths (e.g. the columns documented in [Program Output Formats](../data_formats.md#match-template-dataframe)).
- You want to inspect or overwrite a single statistic (say, just the scaled MIP) without touching the others.

```python
from leopard_em.pydantic_models.results import MatchTemplateResultMRC

# Loading an existing result back into memory
result = MatchTemplateResultMRC(
    mip_path="./output_mip.mrc",
    scaled_mip_path="./output_scaled_mip.mrc",
    correlation_average_path="./output_correlation_average.mrc",
    correlation_variance_path="./output_correlation_variance.mrc",
    orientation_psi_path="./output_orientation_psi.mrc",
    orientation_theta_path="./output_orientation_theta.mrc",
    orientation_phi_path="./output_orientation_phi.mrc",
    relative_defocus_path="./output_relative_defocus.mrc",
)
result.load_tensors_from_paths()

# result.mip, result.scaled_mip, ... are now populated torch.Tensor attributes
peaks_df = result.peaks_to_dataframe()
```

### HDF5 back-end (`MatchTemplateResultHDF5`)

`MatchTemplateResultHDF5` bundles all eight statistics maps, plus run metadata (`leopard_em_version`, `total_projections`, `total_orientations`, `total_defocus`), into a single `.h5` file.

```yaml
match_template_result:
  allow_file_overwrite: true
  hdf5_path: ./match_template_output.h5
  compress: true   # gzip level 4; set false for faster writes at the cost of file size
```

**Use the HDF5 back-end when...**

- You want a single self-contained, portable result file per micrograph instead of eight — simpler to move, archive, or upload alongside a `CorrelationTable`.
- You're processing many micrographs and want to keep the number of output files per run manageable, especially if you're running in a HPC environment.
- You want run metadata (`leopard_em_version`, total search-space sizes) recorded alongside the statistics maps rather than tracked separately.

### HDF5 file layout

```text
/  (root)
│  attrs: leopard_em_version, total_projections,
│         total_orientations, total_defocus
└─ tensors/
       mip                  float32, shape (H-h+1, W-w+1), gzip-4 (if compress=True)
       scaled_mip           float32, shape (H-h+1, W-w+1), gzip-4
       correlation_average  float32, shape (H-h+1, W-w+1), gzip-4
       correlation_variance float32, shape (H-h+1, W-w+1), gzip-4
       orientation_psi      float32, shape (H-h+1, W-w+1), gzip-4
       orientation_theta    float32, shape (H-h+1, W-w+1), gzip-4
       orientation_phi      float32, shape (H-h+1, W-w+1), gzip-4
       relative_defocus     float32, shape (H-h+1, W-w+1), gzip-4
```

where `(H, W)` is the original micrograph size and `(h, w)` is the projected template size, both in units of pixels.

```python
from leopard_em.pydantic_models.results import MatchTemplateResultHDF5

# Loading an existing result back into memory
result = MatchTemplateResultHDF5.from_hdf5("./match_template_output.h5")

# result.mip, result.scaled_mip, ... are populated torch.Tensor attributes,
# same as the MRC back-end
peaks_df = result.peaks_to_dataframe()
```

#### Effect on the match template DataFrame

When using `MatchTemplateResultHDF5`, the `*_path` columns in the [match template DataFrame](../data_formats.md#match-template-dataframe) (`mip_path`, `scaled_mip_path`, `psi_path`, `theta_path`, `phi_path`, `defocus_path`, `correlation_average_path`, `correlation_variance_path`) all point to the **same** `hdf5_path` instead of eight distinct file paths.

## Correlation table (sparse detections)

Where the statistics maps above only retain the _best_ value at each `(x, y)` position, an instance of a `CorrelationTable` object records every search index (defocus offset x out-of-plane orientation x in-plane orientation) whose cross-correlation exceeded a configured threshold, anywhere in the search space.
This is useful for downstream analysis of near-threshold or secondary peaks that don't show up in the per-pixel best-statistic maps — for example, distinguishing a single strong detection from several correlated-but-weaker hypotheses at nearby search indices.

By default, `MatchTemplateManager.run_match_template(...)` computes a `CorrelationTable` for every run (`compute_correlation_table=True`) and stores it on `match_template_result.correlation_table`; pass `compute_correlation_table=False` to skip this and leave it empty.
Computing and storing the correlation table does incur a few percent overhead in total runtime.

### Correlation table HDF5 layout

`CorrelationTable` always uses its own HDF5 format for on-disk storage (independent of which `MatchTemplateResult` back-end you're using):

```text
/metadata              (attrs: correlation_threshold, num_observations)
/search_space/
    defocus_offsets         float32 1-D
    phi_theta_angles        float32 (n, 2)
    psi_angles              float32 1-D
/detections/
    search_index            int32 1-D
    x                       int32 1-D
    y                       int32 1-D
    correlation_value       float32 1-D
    correlation_mean        float32 1-D
    correlation_variance    float32 1-D
```

### Exporting and loading a correlation table

!!! warning "Correlation table export is currently MRC-specific"

    `export_correlation_table()` and `load_correlation_table_from_path()` — which read/write via the `correlation_table_path` field — are only implemented on `MatchTemplateResultMRC`.
    `MatchTemplateResultMRC.export_results()` calls `export_correlation_table()` automatically, so setting `correlation_table_path` in your MRC-backed config is enough.
    For `MatchTemplateResultHDF5`, `export_results()` does **not** currently export the correlation table automatically — call `.to_hdf5(...)` on the `CorrelationTable` directly, as shown below.

```python
# MRC back-end: correlation_table_path is exported/loaded automatically
mrc_result.correlation_table_path = "./output_correlation_table.h5"
mrc_result.export_results()  # writes the 8 mrc files AND the correlation table
mrc_result.load_correlation_table_from_path()

# HDF5 back-end: export the correlation table explicitly (not done by export_results)
hdf5_result.export_results()  # writes hdf5_path only
hdf5_result.correlation_table.to_hdf5("./output_correlation_table.h5")

# Loading a correlation table directly, regardless of back-end
from leopard_em.pydantic_models.results import CorrelationTable

table = CorrelationTable.from_hdf5("./output_correlation_table.h5")
table_df = table.to_dataframe()  # one row per detection
```

In both cases, `mrc_result`/`hdf5_result` are `MatchTemplateResult*` instances populated by `MatchTemplateManager.run_match_template(...)` (with `compute_correlation_table=True`, the default) — `match_template_result.correlation_table` is set automatically after the run.
