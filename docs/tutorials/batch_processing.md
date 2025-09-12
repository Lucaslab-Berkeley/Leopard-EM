# Processing batches of micrographs with Leopard-EM

Often times cryo-EM users may want to process tens to hundreds of micrographs with 2DTM using the same search parameters, reference template, etc.
Using batch processing scripts to run through a large set of micrographs is extremely useful, especially when compared to running each 2DTM search manually.
In this tutorial, we walk through a basic example of doing batch processing with Leopard-EM on a cluster with SLURM scheduling.

The pre-requisites for this tutorial are:

1. Multiple aligned and summed micrographs as MRC files.
2. A single simulated reference template in MRC format.
3. CTF estimations for each micrograph (tutorial assumes CTFFIND5 was used).
4. A cluster with SLURM scheduling and GPU nodes available.
5. Leopard-EM installed in the current Python environment.

??? note "Why Leopard-EM doesn't handle batch processing internally"

    Leopard-EM is designed to be a flexible and modular Python package for 2DTM, and as such we focus on providing the core functionality for 2DTM workflows.
    Reproducibility is another key aspect of Leopard-EM, and we encourage users to have a one-to-one mapping between input configuration files and 2DTM results.
    This means that Leopard-EM does not inherently handle batch processing, but instead provides a simple interface for users to write their own 2DTM workflows specific to their computing environments and project needs.

## Pre-requisite data layout

This tutorial assumes you are working within a directory with already processed micrographs and associated CTF estimations as well as a simulated reference volume named `ref_template.mrc`.
Each micrograph has a unique name, and these micrographs are stored under the `micrographs/` directory.
The CTF estimations share the same prefix as the micrograph names, and are stored under the `ctf_estimations/` directory.
Below is an example of the directory structure where we have 100 micrographs:

```
/some/path/to/my_project/
├── ref_template.mrc
├── micrographs/
│   ├── micrograph_0.mrc
│   ├── micrograph_1.mrc
│   ├── micrograph_2.mrc
│   ├── ...
│   ├── micrograph_99.mrc
├── ctf_estimations/
│   ├── micrograph_0_diagnostic.txt
│   ├── micrograph_1_diagnostic.txt
│   ├── micrograph_2_diagnostic.txt
│   ├── ...
│   ├── micrograph_99_diagnostic.txt
```

An example of the CTF estimation diagnostic file is shown below:

```txt
# Output from CTFFind version 5.0.2, run on 2025-02-13 21:14:21
# Input file: /some/path/to/my_project/micrograph_1.mrc ; Number of micrographs: 1
# Pixel size: 0.930 Angstroms ; acceleration voltage: 300.0 keV ; spherical aberration: 2.70 mm ; amplitude contrast: 0.07
# Box size: 512 pixels ; min. res.: 30.0 Angstroms ; max. res.: 3.5 Angstroms ; min. def.: 0.0 um; max. def. 50000.0 um
# Columns: #1 - micrograph number; #2 - defocus 1 [Angstroms]; #3 - defocus 2; #4 - azimuth of astigmatism; #5 - additional phase shift [radians]; #6 - cross correlation; #7 - spacing (in Angstroms) up to which CTF rings were fit successfully; #8 - Estimated tilt axis angle; #9 - Estimated tilt angle ; #10 Estimated sample thickness (in Angstroms)
1.000000 8984.742188 8709.236328 45.948771 0.000000 0.446317 3.720000 221.912292 12.251187 1619.773438
```

We will later write a Python function which reads in this diagnostic data and extracts the relevant defocus parameters for each micrograph.

## Base YAML configuration file

Next, we need to create a base YAML configuration file.
Most of these fields will be the same for each micrograph, but other like the CTF defocus parameters need populated for each micrograph.
See the [Match Template program page](../programs/match_template.md) for information on each of the configuration fields which will be specific to your cryo-EM imaging setup.
We enclose the populated fields in double curly braces `{{...}}` which are used for string replacement in the later Python script.

This file should be saved as `base_match_template_config.yaml` in the project directory root.

!!! Note "Code copy button (top-left)"

    This tutorial includes _a lot_ of code snippets.
    There is a copy button in the top-left corner of each code block which will copy the entire code block to your clipboard.

```yaml
# base_match_template_config.yaml
template_volume_path: "/some/path/to/my_project/ref_template.mrc"
micrograph_path:      "/some/path/to/my_project/micrographs/{{ micrograph_path }}"
match_template_result:
  allow_file_overwrite: true
  mip_path:                   "/some/path/to/my_project/match_template_results/{{ micrograph_name }}_output_mip.mrc"
  scaled_mip_path:            "/some/path/to/my_project/match_template_results/{{ micrograph_name }}_output_scaled_mip.mrc"
  orientation_psi_path:       "/some/path/to/my_project/match_template_results/{{ micrograph_name }}_output_orientation_psi.mrc"
  orientation_theta_path:     "/some/path/to/my_project/match_template_results/{{ micrograph_name }}_output_orientation_theta.mrc"
  orientation_phi_path:       "/some/path/to/my_project/match_template_results/{{ micrograph_name }}_output_orientation_phi.mrc"
  relative_defocus_path:      "/some/path/to/my_project/match_template_results/{{ micrograph_name }}_output_relative_defocus.mrc"
  correlation_average_path:   "/some/path/to/my_project/match_template_results/{{ micrograph_name }}_output_correlation_average.mrc"
  correlation_variance_path:  "/some/path/to/my_project/match_template_results/{{ micrograph_name }}_output_correlation_variance.mrc"
optics_group:
  label: my_optics_group
  voltage: 300.0
  pixel_size: 0.936   # in Angstroms
  defocus_u: "{{ defocus_u_value }}"  # in Angstroms
  defocus_v: "{{ defocus_v_value }}"  # in Angstroms
  astigmatism_angle: "{{ astigmatism_angle_value }}"
  spherical_aberration: 2.7  # in millimeters
  amplitude_contrast_ratio: 0.07
  phase_shift: 0.0
  ctf_B_factor: 0.0
defocus_search_config:
  defocus_min: -1000.0  # in Angstroms, relative to defocus_{u,v}
  defocus_max: 1000.0   # in Angstroms, relative to defocus_{u,v}
  defocus_step: 200.0   # in Angstroms
orientation_search_config:
  base_grid_method: uniform
  psi_step: 1.5    # in degrees
  theta_step: 2.5  # in degrees
preprocessing_filters:
  whitening_filter:
    enabled: true
    do_power_spectrum: true
    max_freq: 0.5  # In terms of Nyquist frequency
    num_freq_bins: null
  bandpass_filter:
    enabled: false
computational_config:
  gpu_ids: "all"
  num_cpus: 8
```

## Populating the YAML configuration file

Now that we have a YAML configuration file to build off of, we next write a Python script to populate the necessary fields for each micrograph.
This script is basic expecting an exact correspondence between the micrograph names and the CTF estimation diagnostic files and that the diagnostics are from CTFFIND5.
However, this script is easily extensible to handle more complex cases.

After creating a new file with the following code and saving it as `populate_match_template_config.py`, our project directory structure should look like this:

```
/some/path/to/my_project/
├── ref_template.mrc
├── micrographs/
│   ├── micrograph_0.mrc
│   ├── micrograph_1.mrc
│   ├── micrograph_2.mrc
│   ├── ...
│   ├── micrograph_99.mrc
├── ctf_estimations/
│   ├── micrograph_0_diagnostic.txt
│   ├── micrograph_1_diagnostic.txt
│   ├── micrograph_2_diagnostic.txt
│   ├── ...
│   ├── micrograph_99_diagnostic.txt
│
├── base_match_template_config.yaml    <-- New (contents above)
└── populate_match_template_config.py  <-- New (contents below)
```

```python
"""Script to populate the base YAML configuration file for each micrograph."""

import os
import yaml
import glob
import re


# Path constants which are updatable
INPUT_MICROGRAPHS_DIR = "/some/path/to/my_project/micrographs/"
CTF_DIAGNOSTICS_DIR = "/some/path/to/my_project/ctf_estimations/"
BASE_CONFIG_PATH = "/some/path/to/my_project/base_match_template_config.yaml"
OUTPUT_DIR = "/some/path/to/my_project/match_template_results/"


def parse_ctffind5_result(diagnostic_path: str) -> tuple[float, float, float]:
    """Parse the CTFFIND5 diagnostic file to extract defocus parameters.

    Parameters
    ----------
    diagnostic_path : str
        Path to the CTFFIND5 diagnostic file.
    Returns
    -------
    tuple[float, float, float]
        A tuple containing defocus values
        (defocus_u, defocus_v, astigmatism_angle).
    """
    with open(diagnostic_path, "r") as f:
        lines = f.readlines()

    # Assuming first non-comment line contains all info
    for line in lines:
        if not line.startswith("#"):
            parts = line.split()
            defocus_u = float(parts[1])
            defocus_v = float(parts[2])
            astigmatism_angle = float(parts[3])
            return defocus_u, defocus_v, astigmatism_angle


def populate_single_config(
    base_config_dict: dict, micrograph_path: str, ctf_diagnostic_path: str
) -> dict:
    """Populates a single configuration dictionary.

    Parameters
    ----------
    base_config_dict : dict
        The base configuration dictionary to populate.
    micrograph_path : str
        The path to the micrograph file.

    Returns
    -------
    dict
        The populated configuration dictionary.
    """

    # Populate the micrograph path and results fields
    base_config_dict["micrograph_path"] = micrograph_path

    # Replace all "{{ micrograph_name }}" placeholders in match_template_result paths
    basename = os.path.basename(micrograph_path)
    basename = os.path.splitext(basename)[0]
    for result_key, result_path in base_config_dict["match_template_result"].items():
        if isinstance(result_path, str) and "{{ micrograph_name }}" in result_path:
            updated_path = result_path.replace("{{ micrograph_name }}", basename)
            base_config_dict["match_template_result"][result_key] = updated_path

    # Get the defocus parameters and populate the optics group
    defocus_u, defocus_v, astigmatism_angle = parse_ctffind5_result(ctf_diagnostic_path)
    base_config_dict["optics_group"]["defocus_u"] = defocus_u
    base_config_dict["optics_group"]["defocus_v"] = defocus_v
    base_config_dict["optics_group"]["astigmatism_angle"] = astigmatism_angle

    return base_config_dict


def create_micrograph_pairs(micrograph_paths, ctf_diagnostic_paths):
    """Create pairs of micrograph and CTF diagnostic files."""
    # Create dictionaries mapping index to file path
    micrographs = {}
    diagnostics = {}

    # Extract indices from micrograph files
    for path in micrograph_paths:
        match = re.search(r"micrograph_(\d+)\.mrc$", os.path.basename(path))
        if match:
            index = int(match.group(1))
            micrographs[index] = path

    # Extract indices from diagnostic files
    for path in ctf_diagnostic_paths:
        match = re.search(r"micrograph_(\d+)_diagnostic\.txt$", os.path.basename(path))
        if match:
            index = int(match.group(1))
            diagnostics[index] = path

    # Create pairs for matching indices
    pairs = []
    for index in sorted(micrographs.keys()):
        if index in diagnostics:
            pairs.append((micrographs[index], diagnostics[index]))

    return pairs


def main():
    """Main function to loop through all micrographs."""
    # Find all micrographs and CTF diagnostics
    micrograph_paths = glob.glob(os.path.join(INPUT_MICROGRAPHS_DIR, "*.mrc"))
    ctf_diagnostic_paths = glob.glob(
        os.path.join(CTF_DIAGNOSTICS_DIR, "*_diagnostic.txt")
    )

    # Create pairs based on filename indices
    pairs = create_micrograph_pairs(micrograph_paths, ctf_diagnostic_paths)

    # Load base configuration
    with open(BASE_CONFIG_PATH, "r") as f:
        base_config_dict = yaml.safe_load(f)

    # Process each pair
    for i, (micrograph_path, ctf_diagnostic_path) in enumerate(pairs):
        populated_config = populate_single_config(
            base_config_dict.copy(), micrograph_path, ctf_diagnostic_path
        )

        # Create output directory and save configuration
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_filename = os.path.join(
            OUTPUT_DIR,
            f"{os.path.basename(micrograph_path).replace('.mrc', '')}_match_template_config.yaml",
        )
        with open(output_filename, "w") as out_f:
            yaml.dump(populated_config, out_f)

        print(f"Finished processing {output_filename} ({i + 1}/{len(pairs)})")


if __name__ == "__main__":
    main()

```

Now run the above script which will generate all the necessary YAML configurations.

```bash
python populate_match_template_config.py
```

Our project directory structure should now look like this:

```
/some/path/to/my_project/
├── ref_template.mrc
├── micrographs/
│   ├── micrograph_0.mrc
│   ├── micrograph_1.mrc
│   ├── micrograph_2.mrc
│   ├── ...
│   ├── micrograph_99.mrc
├── ctf_estimations/
│   ├── micrograph_0_diagnostic.txt
│   ├── micrograph_1_diagnostic.txt
│   ├── micrograph_2_diagnostic.txt
│   ├── ...
│   ├── micrograph_99_diagnostic.txt
├── match_template_results/                       <-- New
│   ├── micrograph_0_match_template_config.yaml   <-- New
│   ├── micrograph_1_match_template_config.yaml   <-- New
│   ├── micrograph_2_match_template_config.yaml   <-- New
│   ├── ...
│   ├── micrograph_99_match_template_config.yaml  <-- New
├── base_match_template_config.yaml
└── populate_match_template_config.py
```

!!! note "Batch processing in Leopard-EM not depend on CTFFIND5"

    This tutorial assumes outputs from the CTFFIND5 program each in their own diagnostic file, but the above script can be adapted to any number of CTF estimation outputs.
    As long as you can uniquely map each micrograph to a set of defocus parameters (presumably in a Python function), the above population of the configuration will be straightforward.

## Setting up the Leopard-EM 2DTM script

The final step before job submission is to create a Python script for running the `match_template` program.
We adapt the included Python script from the `programs/` directory to accept a YAML file as an argument; the program is completely configured through the YAML file, so this is the only argument which changes between different 2DTM runs.
Copy the following code into a new file named `run_match_template.py` in the project directory root.

```python
"""Program for running whole-orientation search using 2D template matching."""

import sys

from leopard_em.pydantic_models.managers import MatchTemplateManager

# Change batch size based on available GPU memory
ORIENTATION_BATCH_SIZE = 8


def main() -> None:
    yaml_config_path = sys.argv[1]
    dataframe_output_path = yaml_config_path.replace("config.yaml", "results.csv")
    mt_manager = MatchTemplateManager.from_yaml(yaml_config_path)

    print("Loaded configuration.\nRunning match_template...")

    mt_manager.run_match_template(
        orientation_batch_size=ORIENTATION_BATCH_SIZE,
        do_result_export=True,  # Saves the statistics immediately upon completion
    )

    print("Finished core match_template call.\nExporting results...")

    df = mt_manager.results_to_dataframe()
    df.to_csv(dataframe_output_path, index=True)

    print("Done!")


# NOTE: Invoking  program under `if __name__ == "__main__"`
# necessary for multiprocesing
if __name__ == "__main__":
    main()

```

Thats it!
From the project directory root, we can now initiate a 2DTM run through the following
command for a particular configuration:

```bash
python run_match_template.py match_template_results/micrograph_0_match_template_config.yaml
```

## Wrapping the runs into a SLURM array job

Rather than running each 2DTM search manually, we can use an included SLURM job scheduler to process all of the data. Here, we choose to use a SLURM array job since we have a large number of micrographs to process and the computational needs don't change between runs.

!!! caution "Adapting to your SLURM environment"

    Different computing environments have different SLURM configurations including how to request GPU resources, constraints on job allocations, and how to record allocations to a computing account.
    We cannot possibly enumerate all possible configurations, but the following script is a good starting point.
    You will need to adapt the SLURM job script below to your specific computing environment, but the principles of running _N_ independent searches across _N_ micrographs remains the same.

Create a new file named `run_match_template_slurm.sh` in the project directory root with the following code:

```bash
#!/bin/bash
#SBATCH --job-name=batch_2dtm_example
#SBATCH --account=<<<YOUR_ACCOUNT>>>
#SBATCH --partition=<<<YOUR_PARTITION>>>
#SBATCH --qos=<<<YOUR_QOS>>>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8  # <------ Match with 'num_cpus' in YAML
#SBATCH --gres=gpu:L40:1  # <------- Adjust based on GPU/node configuration
#SBATCH --time=10:00:00
#SBATCH --array=0-99  # <----------- Adjust based on number of micrographs
#SBATCH --output=batch_2dtm_example_%A_task_%a.out
#SBATCH --error=batch_2dtm_example_%A_task_%a.err

#####################################
### Load modules and activate     ###
### Leopard-EM Python environment ###
#####################################
# NOTE: You will need to adjust these lines!
ml anaconda3
conda activate leopard-em

#######################################################
### Decode config file based on SLURM_ARRAY_TASK_ID ###
#######################################################
# NOTE: You will also need to adjust the CONFIG_DIRECTORY variable below
CONFIG_DIRECTORY="/some/path/to/my_project/match_template_results"

# Find all YAML files in the directory, sort them, and select the one corresponding to the current task ID.
# NOTE: Does not depend on naming scheme and should work for any set of YAML configurations.
CONFIG_FILE=$(ls "${CONFIG_DIRECTORY}"/*.yaml | sort | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

# Check if the config file exists
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file found for task ID $SLURM_ARRAY_TASK_ID."
    exit 1
fi

# Run the match_template script with the selected config file
# NOTE: You may need to wrap the `python run_match_template.py $CONFIG_FILE`
#       call within an `srun` command depending on your cluster configuration
#       to properly expose the GPU devices to the process.
echo "Running match_template with config file: $CONFIG_FILE"
python run_match_template.py $CONFIG_FILE

```

??? Caution "Making GPU devices discoverable within SLURM via srun"

    Depending on your cluster configuration, you may need to wrap the `python run_match_template.py $CONFIG_FILE` command within an srun command to properly expose the GPU devices to the command.
    For example, this might look like:

    ```bash
    srun --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:L40:1 python run_match_template.py $CONFIG_FILE
    ```

## Conclusion

In this tutorial, we walked through how to set up batch processing of micrographs with Leopard-EM using a SLURM array job.
Many components of this tutorial will need to be adapted to your specific use-case, like the naming convention for micrographs, how the CTF estimations are mapped to each micrograph, and your particular SLURM environment.
However, the principles of creating a single YAML configuration for each template matching run and then executing each run independently remain the same.

In short, batch processing with Leopard-EM follows these steps:
1. Placing micrographs and CTF estimations in a known directory structure,
2. Creating a base YAML configuration file with placeholders for micrograph-specific parameters,
3. Populating the YAML configuration file for each micrograph, and
4. Running the `match_template` program for each of the populated YAML configuration files.