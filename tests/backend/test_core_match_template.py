"""Pytest for a consistent core_match_template output

NOTE: Future updates will make degenerate search positions (same orientation, defocus,
image (i, j), *and* same cross-correlation value) in the match_template
non-deterministic for the resulting statistics files. That is, if two sets of
orientations or defocus values produce the same cross-correlation value it is not
guaranteed which set of orientations or defocus values will be returned. This unit test
handles this by checking for equality in the MIP and only checking for equality in
the statistics files where the partial results are not the same.

NOTE: Floating point error accumulates over the search space in the correlation mean
and correlation variance, so these results are checked for closeness within a tolerance.

NOTE: This test can take up to 10 minutes given the moderate sized search space and
GPU requirements.
"""

import subprocess
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import torch

from leopard_em.pydantic_models.managers import MatchTemplateManager

YAML_PATH = (
    Path(__file__).parent
    / "../tmp/test_match_template_xenon_216_000_0.0_DWS_config.yaml"
).resolve()
ZENODO_URL = "https://zenodo.org/records/17069607"
ORIENTATION_BATCH_SIZE = 20


def download_comparison_data() -> None:
    """Downloads the example data from Zenodo."""
    subprocess.run(["zenodo_get", "--output-dir=tests/tmp", ZENODO_URL], check=True)


def setup_match_template_manager() -> MatchTemplateManager:
    """Instantiate the manager object and run the template matching program."""
    mt_manager = MatchTemplateManager.from_yaml(YAML_PATH)
    mt_manager.make_backend_core_function_kwargs()

    return mt_manager


def mrcfile_allclose(path_a: str, path_b: str, **kwargs) -> bool:
    """Wrapper for all close call for two mrcfiles"""
    data_a = mrcfile.read(path_a)
    data_b = mrcfile.read(path_b)

    return np.allclose(data_a, data_b, **kwargs)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for this test."
)
@pytest.mark.slow
def test_core_match_template():
    download_comparison_data()
    mt_manager = setup_match_template_manager()

    # Run the match template program
    mt_manager.run_match_template(
        orientation_batch_size=ORIENTATION_BATCH_SIZE,
        do_result_export=True,  # Saves the statistics immediately upon completion
    )

    # fmt: off
    reference_data = [
        "tests/tmp/test_match_template_xenon_216_000_0_output_correlation_average.mrc",
        "tests/tmp/test_match_template_xenon_216_000_0_output_correlation_variance.mrc",
        "tests/tmp/test_match_template_xenon_216_000_0_output_orientation_phi.mrc",
        "tests/tmp/test_match_template_xenon_216_000_0_output_orientation_psi.mrc",
        "tests/tmp/test_match_template_xenon_216_000_0_output_orientation_theta.mrc",
        "tests/tmp/test_match_template_xenon_216_000_0_output_relative_defocus.mrc",
        "tests/tmp/test_match_template_xenon_216_000_0_output_scaled_mip.mrc",
        "tests/tmp/test_match_template_xenon_216_000_0_output_mip.mrc",
    ]
    test_data = [
        "tests/tmp/output_match_template_xenon_216_000_0_output_correlation_average.mrc",
        "tests/tmp/output_match_template_xenon_216_000_0_output_correlation_variance.mrc",
        "tests/tmp/output_match_template_xenon_216_000_0_output_orientation_phi.mrc",
        "tests/tmp/output_match_template_xenon_216_000_0_output_orientation_psi.mrc",
        "tests/tmp/output_match_template_xenon_216_000_0_output_orientation_theta.mrc",
        "tests/tmp/output_match_template_xenon_216_000_0_output_relative_defocus.mrc",
        "tests/tmp/output_match_template_xenon_216_000_0_output_scaled_mip.mrc",
        "tests/tmp/output_match_template_xenon_216_000_0_output_mip.mrc",
    ]
    # fmt: on

    # Check the files for equality
    for a, b in zip(reference_data, test_data):
        assert mrcfile_allclose(a, b)


if __name__ == "__main__":
    test_core_match_template()
