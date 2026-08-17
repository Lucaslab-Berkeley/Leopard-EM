# Match template program example scripts and configs

This directory contains example files for configuring the match template program, and example Python scripts for running the match template program.
See the online documentation for comprehensive information on configuring the match template program.

## Files

- `match_template_example_config.yaml` - An example configuration YAML file for constructing a `MatchTemplateManager` object.
- `run_match_template.py` - The default Python script for running the match template program. This supports muli-GPU systems (configure GPUs using the YAML file).
- `run_distributed_match_template.py` - A Python script for running match template on large-scale distributed systems (multi-node clusters). _Use the default script unless you're running on more than one machine_.
- `distributed_match_template.slurm` - An example SLURM script for running the distributed match template (_launching from a workload manager is required_).
- `run_sectored_match_template.py` - Sectored SO(3) search: splits orientation space into HEALPix sectors and runs an independent masked-MIP match template per sector, then merges results. Requires `torch-so3 >= 0.5.2`.

## Sectored match template

`run_sectored_match_template.py` uses `get_sectored_euler_angles` from `torch-so3` to
partition SO(3) into equal-area HEALPix sectors, run an independent
masked-MIP `core_match_template` per sector, and merge results with
`merge_runs_independent_zscore`.

**Key design points:**

- Each sector includes *all* fine directions (some outside the user's angular window) to ensure the per-sector mean and variance are computed over a complete local neighbourhood. This gives accurate z-score normalisation.
- Orientations outside the user-defined or symmetry-derived bounds are masked from the MIP via the `streamed_masked_mip` backend.
- The peak-significance multiplicity uses only the *eligible* orientation count (inside bounds, summed across sectors), not the full search count.

**Tunable constants** at the top of the file:

| Constant | Default | Meaning |
|---|---|---|
| `YAML_CONFIG_PATH` | `"..."` | Path to the YAML config file |
| `DATAFRAME_OUTPUT_PATH` | `"..."` | Output CSV path |
| `ORIENTATION_BATCH_SIZE` | `8` | Orientations per GPU batch |
| `NSIDE_COARSE` | `1` | Coarse HEALPix resolution (`n_sectors = 12 * NSIDE_COARSE**2`) |
| `NSIDE_FINE` | `None` | Fine HEALPix resolution (inferred from `theta_step` if `None`) |

**Installation requirement:** `torch-so3 >= 0.5.2` (same as the package dependency).

```bash
pip install "torch-so3>=0.5.2"
```
