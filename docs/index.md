---
title: Leopard-EM Homepage
description: Overview of the Leopard-EM package for 2DTM in Python
---

# Leopard-EM

**L**ocation & ori**E**ntati**O**n of **PAR**ticles found using two-**D**imensional t**E**mplate **M**atching (Leopard-EM) is a Python implementation of Two-Dimensional Template Matching (2DTM) using PyTorch for GPU acceleration. This package reflects most of the functionality described in Lucas, *et al.* (2021)[^1] with additional user-friendly features for integrating into broader data science workflows.

## Installation

Pre-packaged versions of Leopard-EM are released on the Python Package Index (PyPI).
We target Linux operating systems on Python versions 3.9 - 3.12 for these releases, and the PyTorch GPU acceleration backend is only tested against NVIDIA GPUs.
With these caveats in mind, the package can be installed using pip:

```bash
pip install leopard-em
```

We also recommend you install the package in a virtual environment (such as [conda](https://docs.conda.io/en/latest/)) to avoid conflicts with other installed Python packages or software on your machine.
If there are persistent issues during installation, you can [open up a bug report](https://github.com/Lucaslab-Berkeley/Leopard-EM/issues/new) on the GitHub page.

### Installing from Source

If you want to install Leopard-EM from source, first clone the repository and install the package using pip:

```bash
git clone https://github.com/Lucaslab-Berkeley/Leopard-EM.git
cd Leopard-EM
pip install .
```

The `.` (period) here refers to the current working directory, and pip should parse the necessary configurations for installation.

### For Developers

Developers who are interested in contributing to Leopard-EM should install the package in an editable configuration with the necessary development dependencies.
After cloning the repository, navigate to the root directory of the repository and run the following command:

```bash
pip install -e '.[dev,test,docs]'
```

See the [Contributing](#contributing) section for more information on how to contribute to the package.

## Basic Usage

Leopard-EM is most easily used by editing configuration YAML files, loading these YAML files using Python object, then running the program through a python script.
There are currently 4 main programs under `/src/programs` which can be edited in-place or coped to new Python scripts on your machine:

- `match_template.py`: Runs the whole orientation search a given reference template on a single cryo-EM image.
- `refine_template.py`: Refines the orientation and defocus parameters for particles identified from the match template program.
- `optimize_template.py`: Optimizes the pixel size of the reference temple; necessary if the pixel size of the deposited PDB model is much different from the pixel size of the micrograph.
- `optimize_B_factor.py`: Optimizes the additional b-factor (blurring) applied to the template during the search.

A minimally working Python script for running the match template program is shown below; further information on running each program can be found here: [Programs](programs/programs_landing_page.md)

```python
from leopard_em.pydantic_models import MatchTemplateManager

# Editable parameters for the program
YAML_CONFIG_PATH = "/path/to/match-template-configuration.yaml"
DATAFRAME_OUTPUT_PATH = "/path/to/match-template-results.csv"
ORIENTATION_BATCH_SIZE = 32  # Tune based on GPU vram


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

Work in progress

## Theory

Work in progress

## API

Work in progress
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

### Building Documentation
The documentation for Leopard-EM is built using [MkDocs](https://www.mkdocs.org) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) for generating the documentation site.
If you've installed the package with the optional `docs` dependencies, you can build the documentation site with the following command:

```bash
mkdocs build
mkdocs serve
```

The first command will construct the HTML files for the documentation site, and the second command will start a local server (at `127.0.0.1:8000`) to view the site.

## License

The code in this repository is licensed under the **BSD 3-Clause License**. See the [LICENSE](LICENSE) file for full details.

## References

[^1]: Lucas BA, Himes BA, Xue L, Grant T, Mahamid J, Grigorieff N. Locating macromolecular assemblies in cells by 2D template matching with cisTEM. Elife. 2021 Jun 11;10:e68946. doi: 10.7554/eLife.68946. PMID: 34114559; PMCID: PMC8219381.