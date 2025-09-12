# Match template program example scripts and configs

This directory contains example files for configuring the match template program, and example Python scripts for running the match template program.
See the online documentation for comprehensive information on configuring the match template program.

## Files

- `match_template_example_config.yaml` - An example configuration YAML file for constructing a `MatchTemplateManager` object.
- `run_match_template.py` - The default Python script for running the match template program. This supports muli-GPU systems (configure GPUs using the YAML file).
- `run_distributed_match_template.py` - A Python script for running match template on large-scale distributed systems (multi-node clusters). _Use the default script unless you're running on more than one machine_.
- `distributed_match_template.slurm` - An example SLURM script for running the distributed match template (_launching from a workload manager is required_).