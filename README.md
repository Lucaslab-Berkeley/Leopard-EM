# Leopard-EM: Python based template matching

[![License](https://img.shields.io/pypi/l/Leopard-EM.svg?color=green)](https://github.com/Lucaslab-Berkeley/Leopard-EM/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/Leopard-EM.svg?color=green)](https://pypi.org/project/Leopard-EM)
[![Python Version](https://img.shields.io/pypi/pyversions/Leopard-EM.svg?color=green)](https://python.org)
[![CI](https://github.com/Lucaslab-Berkeley/Leopard-EM/actions/workflows/ci.yml/badge.svg)](https://github.com/Lucaslab-Berkeley/Leopard-EM/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Lucaslab-Berkeley/Leopard-EM/branch/main/graph/badge.svg)](https://github.com/Lucaslab-Berkeley/Leopard-EM)

Leopard-EM (**L**ocation & ori**E**ntati**O**n of **PAR**ticles found using two-**D**imensional t**E**mplate **M**atching) is a python package for running two-dimensional template matching (2DTM) on cryo-EM images.
Check out the [online documentation](https://lucaslab-berkeley.github.io/Leopard-EM/) for more details about using, configuring, and analyzing data from the Leopard-EM Python package.

> [!IMPORTANT]
> **If you use Leopard-EM in your research, please include these linked references:** [Citation](README.md#Citation) (bottom of this page).


## Basic Installation

The newest released version of the package can be installed from PyPI using pip:

```bash
pip install leopard-em
```

## Usage

### Template Matching

Inputs to the template matching programs can be configured through a YAML file or a set of Python objects.
The [example YAML configuration file](programs/match_template/match_template_example_config.yaml) acts as a template for configuring your own runs with more details on configuring and running 2DTM on the [programs section of the documentation](https://lucaslab-berkeley.github.io/Leopard-EM/programs/match_template/).
We also provide a default [`match_template.py`](programs/match_template/run_match_template.py) script with some additional detail, but a compacted version of this script is shown below.
Once configured with the proper paths, parameters, etc., the program can run as follows:

```python
from leopard_em.pydantic_models.managers import MatchTemplateManager

YAML_CONFIG_PATH = "/path/to/match-template-configuration.yaml"
DATAFRAME_OUTPUT_PATH = "/path/to/match-template-results.csv"
ORIENTATION_BATCH_SIZE = 8

def main():
    mt_manager = MatchTemplateManager.from_yaml(YAML_CONFIG_PATH)
    mt_manager.run_match_template(ORIENTATION_BATCH_SIZE)
    df = mt_manager.results_to_dataframe()
    df.to_csv("/path/to/results.csv")

# NOTE: invoking from `if __name__ == "__main__"` is necessary
# for proper multiprocessing/GPU-distribution behavior
if __name__ == "__main__":
    main()
```

### Template Refinement

Particle orientations and locations can be refined using the `RefineTemplateManager` objects after a template matching run.
The `RefineTemplateManager` is similarly a set of Pydantic models capable of configuration via YAML files.
The [example YAML configuration file](programs/refine_template/refine_template_example_config.yaml) acts as a template for configuring your own runs.
We also provide a default [`refine_template.py`](programs/refine_template/run_refine_template.py) script with some additional detail, but a compacted version of this script is shown below.
Once configured with the proper paths, parameters, etc., the program can run as follows:

```python
from leopard_em.pydantic_models.managers import RefineTemplateManager

YAML_CONFIG_PATH = "/path/to/refine-template-configuration.yaml"
DATAFRAME_OUTPUT_PATH = "/path/to/refine-template-results.csv"
PARTICLE_BATCH_SIZE = 80

def main():
    rt_manager = RefineTemplateManager.from_yaml(YAML_CONFIG_PATH)
    rt_manager.run_refine_template(DATAFRAME_OUTPUT_PATH, PARTICLE_BATCH_SIZE)

if __name__ == "__main__":
    main()
```

## Installation for Development

The package can be installed from source in editable mode with the optional development libraries via pip.

```bash
git clone https://github.com/Lucaslab-Berkeley/Leopard-EM.git
cd Leopard-EM
pip install -e '.[dev,test, docs]'
```

Further information on development and contributing to the repo can be found in our online documentation.

## Citation

If you use Leopard-EM in your research, please include the following citations in your biblography:

- [Leopard-EM pre-print](https://doi.org/10.1101/2025.08.26.672452)
> ```
> @article {Giammar2025.08.26.672452,
>	author = {Giammar, Matthew David and Dickerson, Joshua Luke and Hall, Laina Nicole and Lucas, Bronwyn Ayla},
>	title = {Leopard-EM: An extensible 2DTM package to accelerate in situ structural biology},
>	elocation-id = {2025.08.26.672452},
>	year = {2025},
>	doi = {10.1101/2025.08.26.672452},
>	publisher = {Cold Spring Harbor Laboratory},
>	URL = {https://www.biorxiv.org/content/early/2025/08/29/2025.08.26.672452},
>	eprint = {https://www.biorxiv.org/content/early/2025/08/29/2025.08.26.672452.full.pdf},
>	journal = {bioRxiv}
> }
> ```

