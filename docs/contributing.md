# Contributing to the Leopard-EM Package

Welcome to the contributing section of the documentation. There are multiple ways to contribute to the Leopard-EM codebase ranging from reporting & fixing a bug to adding new features into the package. While we strive to make 2DTM accessible and to promote open-source science, we are only a handful of researchers managing this package. To ensure a streamlined development cycle, we’ve put together a set of contribution guidelines for the Leopard-EM package.

The tl;dr of the contributing guidelines:

* Follow the bug report template so we can help fix bugs\!  
* Create feature requests *before* adding new features to the library; community suggestions and contributions are encouraged, but we want to keep the scope narrow for Leopard-EM.  
* Open pull requests to the proper branches, and pull requests must be reviewed by a maintainer before merging.

## Using GitHub Issues

### Opening a bug report

We’ve included bug report/issue templates in our GitHub page which detail the necessary information to include when opening a new bug or issue report. This helps us track down the bug/issue and find a fix. We reserve the right to close bug reports that don’t include this necessary information.

### Making feature requests

We encourage users to make applicable feature requests, but we also want to prevent [feature creep](https://en.wikipedia.org/wiki/Feature\_creep). If you are thinking about adding new functionality or features to the package, please open a feature request *before* opening a pull request so we can discuss if that feature fits within the scope of Leopard-EM. For example, the following would fall outside the scope:

* Generating 3D reconstructions from 2DTM results
* Adding a GUI for processing data using 2DTM (this could be a new package\!)

For major feature requests, we ask contributors to take initiative for implementing the feature.

## Making contributions to the Leopard-EM package

### Setting up Leopard-EM on your machine as a developer

Leopard-EM is installable as a PyPI package, but we include additional development and testing dependencies. Please follow the [developer installation instructions](index.md#for-developers) to properly configure these additional dependencies.

Also, make sure you’ve forked the repo when installing from source. Your local repository should be tracking the forked copy as “origin” and the Lucaslab-Berkekely/Leopard-EM repo as “upstream”. Commits should be pushed to your fork, and changes should only come into the main Leopard-EM repository through pull requests.

### Pull requests and branches

As mentioned before, new code should only be added to the main Leopard-EM repository through a pull request. This ensures all the necessary changes adhere to the style and testing standards defined in the package and that maintainers can review newly added code. The process for successfully creating and merging pull request is:

1. Committing and pushing the changes to your fork of the repository (“origin” upstream).
2. Creating a PR from a fork branch to one of the branches (listed below) depending on the kind of addition/change.
3. Ensuring the PR passes all the CI checks and tests.
4. Getting the PR reviewed and approved by one of the Leopard-EM maintainers.
5. Asking one of the maintainers to merge the PR.

We also have a pull request template in the repository listing all the components and steps for a successful pull request. Please follow these carefully\!

Pull requests should be made to one of the following branches to ensure the main branch stays stable:

* **dev** – Intended for testing out new features or other development changes; code may be unstable.
* **feature\_{abcd}** – Branch specifically for a larger feature named “abcd” which may require many changes over a significant period of time
* **docs\_main** – Updates to the online documentation site to be made immediately. For example, updating URLs of hosted data or correcting minor errors not specific to the source code.
* **docs\_dev** – Development branch for documentation which may be unstable or include pages/statements which don’t directly correspond to the current main branch.
* **prerelease\_v{x.y.z}** 	– Code for final cleanup and staging before releasing version “x.y.z” of  Leopard-EM.

### Building Documentation
The documentation for Leopard-EM is built using [MkDocs](https://www.mkdocs.org) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) for generating the documentation site.
If you've installed the package with the optional `docs` dependencies, you can build the documentation site with the following command:

```bash
mkdocs build
mkdocs serve
```

The first command will construct the HTML files for the documentation site, and the second command will start a local server (at `127.0.0.1:8000`) to view the site.

### Code style guidelines

The package is already configured with code formatting standards which aim to reduce complexity and increase readability. Please make sure your code changes are properly [type-hinted](https://docs.python.org/3/library/typing.html), include [docstrings](https://en.wikipedia.org/wiki/Docstring\#Python), and pass the formatting checks.

#### Configuring pre-commit hooks

The pre-commit package is used to run a set of code quality checks and auto-formatters on the codebase. The pre-commit hooks need installed and configured once after a fresh installation.

```bash
pre-commit install --install-hooks
```

Now, the pre-commit tool can be run on staged files before making a commit by running the following.

```bash
pre-commit run
```

#### Running pylint

Pylint is another tool to ensure consistent code quality. The pylint tool can be run on all files in the source directory by running:

```bash
pylint src/
```

Please make sure new code passes the pylint tests.

#### Running unit tests

Leopard-EM uses the pytest package for running tests. To run the tests, simply run the following command from the root directory of the repository:

```bash
pytest
```

Note that we are still working on expanding the unit tests to cover more of the package.