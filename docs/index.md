---
title: Leopard-EM Homepage
description: Overview of the Leopard-EM package for 2DTM in Python
---

# Leopard-EM

Welcome to the **L**ocation & ori**E**ntati**O**n of **PAR**ticles found using two-**D**imensional t**E**mplate **M**atching (Leopard-EM) online documentation!
Leopard-EM is a Python implementation of Two-Dimensional Template Matching (2DTM) which itself is a data analysis method in cryo-EM.
This package reflects most of the functionality described in Lucas, *et al.* (2021)[^1] with additional user-friendly features for integrating into broader data science workflows.

If you use Leopard-EM in your research, please cite ((TODO: add manuscript link)).

## Installation

### Requirements

The general system requirements for Leopard-EM are

- Python version 3.10 or above
- PyTorch 2.4.0 or above
- Linux operating system

The package config contains a complete set of requirements which are automatically downloaded and checked during the installation process.
Leopard-EM should work on MacOS and Windows operating systems, but we only fully support Linux.
The package is also tested against modern Nvidia GPUs but should theoretically run on most PyTorch-supported GPUs.
Your mileage may vary depending on system architecture.

We also recommend using a virtual environment manager (such as [conda](https://docs.conda.io/en/latest/)) to avoid conflicts with other installed software on your system.
Please open a [open up a bug report](https://github.com/Lucaslab-Berkeley/Leopard-EM/issues/new) on the GitHub page if you experience major issues during the installation process.

### Pre-packaged releases

Pre-packaged versions of Leopard-EM are released on the Python Package Index (PyPI).
To install the latest pre-packaged release of Leopard-EM, run the following:

```bash
pip install leopard-em
```

### Installing from Source

If you want to install Leopard-EM from source, first clone the repository and install the package using pip:

```bash
git clone https://github.com/Lucaslab-Berkeley/Leopard-EM.git
cd Leopard-EM
pip install .
```

### For Developers

Developers who are interested in contributing to Leopard-EM should fork the repository into their own GitHub account.
Navigate to the [Leopard-EM GitHub landing page](https://github.com/Lucaslab-Berkeley/Leopard-EM) and click on fork in the top right-hand corner.
Then clone your fork and add the Lucaslab-Berkeley remote as an upstream:

```bash
git clone https://github.com/YOUR_USERNAME/Leopard-EM.git
cd Leopard-EM
git remote add upstream https://github.com/Lucaslab-Berkeley/Leopard-EM
```

Check that the remote has been properly added (`git remote -v`) then run the following to install the package along with the optional development dependencies.

```bash
pip install -e '.[dev,test,docs]'
```

See the [Contributing](#contributing) page for detailed guidelines on contributing to the package.

## Basic Usage

Leopard-EM is runnable through a set of pre-built Python scripts and easily modifiable YAML configurations.
There are currently five main programs (located under [`programs/` folder](https://github.com/Lucaslab-Berkeley/Leopard-EM/tree/main/programs)) each with their own configuration files.
Detailed documentation for each program can be found on the [Program Documentation Page](programs/programs_landing_page.md), but the five man programs are as follows:

1. `match_template` - Runs a whole orientation search for a given reference structure on a single micrograph.
2. `refine_template` - Takes particles identified from match_template and refines their location, orientation, and defocus parameters.
3. `optimize_template` - Optimizes the pixel size of the micrograph and template structure model using a set of identified particles.
4. `constrained_search` - Uses the location and orientation of identified particles to constrain the search parameters of a second particle.
5. `optimize_b_factor.py` - Script to optimize the b-factor added to a model (using 2DTM) for a set of metrics.

<!-- A minimally working Python script for running the match template program is shown below -->

### Match template example

The following Python script will run a basic match template program using only the built-in Pydantic models.
See the programs documentation page for more details on running each program.

```python
from leopard_em.pydantic_models.managers import MatchTemplateManager
from leopard_em.pydantic_models.config import MatchTemplateResult
from leopard_em.pydantic_models.config import OpticsGroup
from leopard_em.pydantic_models.config import DefocusSearchConfig
from leopard_em.pydantic_models.config import OrientationSearchConfig
from leopard_em.pydantic_models.config import ComputationalConfig

# Microscope imaging parameters
my_optics_group = OpticsGroup(
    label="my_optics_group",
    pixel_size=1.2,    # In Angstroms
    voltage=300,       # In kV
    defocus_u=5100.0,  # In Angstroms
    defocus_v=4900.0,  # In Angstroms
    astigmatism_angle=0.0,  # In degrees
)

# Relative defocus planes to search across
df_search_config = DefocusSearchConfig(
    label="defocus_search",
    min_defocus=1000,   # In Angstroms, relative
    max_defocus=-1000,  # In Angstroms, relative
    step_size=200.0,    # In Angstroms
)

# Orientation sampling of SO(3) space, using default
orientation_search_config = OrientationSearchConfig()

# Where to save the output results
mt_result = MatchTemplateResult(
    allow_file_overwrite=True,
    mip_path="/path/to/output_mip.mrc",
    scaled_mip_path="/path/to/output_scaled_mip.mrc",
    correlation_average_path="/path/to/output_correlation_average.mrc",
    correlation_variance_path="/path/to/output_correlation_variance.mrc",
    orientation_psi_path="/path/to/output_orientation_psi.mrc",
    orientation_theta_path="/path/to/output_orientation_theta.mrc",
    orientation_phi_path="/path/to/output_orientation_phi.mrc",
    relative_defocus_path="/path/to/output_relative_defocus.mrc",
    pixel_size_path="/path/to/output_pixel_size.mrc",
)

# Which GPUs to run template matching on (here the first 2 GPUs)
comp_config = ComputationalConfig(gpu_ids=[0, 1])

mt_manager = MatchTemplateManager(
    micrograph_path="/path/to/2D_image.mrc",
    template_volume_path="/path/to/template_volume.mrc",
    optics_group=my_optics_group,
    defocus_search_config=df_search_config,
    orientation_search_config=orientation_search_config,
    match_template_result=mt_result,
    computational_config=comp_config,
    # pixel_size_search_config
    # preprocessing_filters
)


def main():
    # Load and run the match template configuration
    mt_manager = MatchTemplateManager.from_yaml(YAML_CONFIG_PATH)
    mt_manager.run_match_template(
        orientation_batch_size=ORIENTATION_BATCH_SIZE,
        do_result_export=True,  # Saves the statistics immediately upon completion
    )

    # Construct and export the dataframe of picked peaks
    df = mt_manager.results_to_dataframe()
    df.to_csv(DATAFRAME_OUTPUT_PATH, index=True)


if __name__ == "__main__":
    main()
```

## Documentation and Examples

Under construction

## Theory

Under construction

## API

Under construction
<!-- TODO: Get some autodocs to parse the docstrings and generate API documentation. -->

## Contributing
We encourage contributions to this package from the broader cryo-EM/ET and structural biology communities.
Leopard-EM is configured with a set of development dependencies to help contributors maintain code quality and consistency.
See the [Installation -- For Developers](#for-developers) section for instructions on how to install these dependencies.

### Using `pre-commit`
The `pre-commit` package is used to run a set of code quality checks and auto-formatters on the codebase.
If this is your first time installing the package, you will need to install the pre-commit hooks:

```bash
pre-commit install--install-hooks
```

After staging changes, but before making a commit, you can run the pre-commit checks with:

```bash
pre-commit run
```

This will go through the staged files, check that all the changed code adheres to the style guidelines, and auto-format the code where necessary.
If all the tests pass, you can commit the changes.

### Running Tests
Leopard-EM uses the `pytest` package for running tests.
To run the tests, simply run the following command from the root directory of the repository:

```bash
pytest
```

Note that we are still working on expanding the unit tests to cover more of the package, but we ask that any new code contributions include tests where appropriate.


## License

The code in this repository is licensed under the **BSD 3-Clause License**. See the [LICENSE](LICENSE) file for full details.

## References

[^1]: Lucas BA, Himes BA, Xue L, Grant T, Mahamid J, Grigorieff N. Locating macromolecular assemblies in cells by 2D template matching with cisTEM. Elife. 2021 Jun 11;10:e68946. doi: 10.7554/eLife.68946. PMID: 34114559; PMCID: PMC8219381.