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
ZENODO_URL = "https://zenodo.org/records/17069838"
ORIENTATION_BATCH_SIZE = 20


def download_comparison_data() -> None:
    """Downloads the example data from Zenodo."""
    subprocess.run(["zenodo_get", "--output-dir=tests/tmp", ZENODO_URL], check=True)


def setup_match_template_manager() -> MatchTemplateManager:
    """Instantiate the manager object and run the template matching program."""
    return MatchTemplateManager.from_yaml(YAML_PATH)


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

    # Ensure the MIPs are the same, if they are not then there's an issue...
    assert mrcfile_allclose(
        "tests/tmp/test_match_template_xenon_216_000_0_output_mip.mrc",
        "tests/tmp/output_match_template_xenon_216_000_0_output_mip.mrc",
    ), "MIPs are different between runs!"

    # Ensure the mean and variances are close to a slightly reduced tolerance.
    # Reduced tolerance because of accumulated floating point error
    assert mrcfile_allclose(
        "tests/tmp/test_match_template_xenon_216_000_0_output_correlation_variance.mrc",
        "tests/tmp/output_match_template_xenon_216_000_0_output_correlation_variance.mrc",
        rtol=1e-6,
        atol=1e-6,
    ), "Correlation variances diverge over the search space between runs!"

    assert mrcfile_allclose(
        "tests/tmp/test_match_template_xenon_216_000_0_output_correlation_average.mrc",
        "tests/tmp/output_match_template_xenon_216_000_0_output_correlation_average.mrc",
        rtol=1e-6,
        atol=1e-6,
    ), "Correlation averages diverge over the search space between runs!"

    # NOTE: Since shared work queue introduces non-determinism meaning we cannot
    # distinguish between search points with the same.
    # Here, we require the following to be true for the results to be considered the
    # same even if they're not exactly equal:
    # 1) Fewer than 0.2% of pixel can have different values (for each file)
    # 2) Overlap between the positions of the differences is at least 50%

    def get_diff_indices(path_a: str, path_b: str):
        """Get positions of differences between to mrcfiles"""
        a_data = mrcfile.read(path_a)
        b_data = mrcfile.read(path_b)

        return np.where(~np.isclose(a_data, b_data))

    diff_defocus = get_diff_indices(
        "tests/tmp/test_match_template_xenon_216_000_0_output_relative_defocus.mrc",
        "tests/tmp/output_match_template_xenon_216_000_0_output_relative_defocus.mrc",
    )
    diff_phi = get_diff_indices(
        "tests/tmp/test_match_template_xenon_216_000_0_output_orientation_phi.mrc",
        "tests/tmp/output_match_template_xenon_216_000_0_output_orientation_phi.mrc",
    )
    diff_theta = get_diff_indices(
        "tests/tmp/test_match_template_xenon_216_000_0_output_orientation_theta.mrc",
        "tests/tmp/output_match_template_xenon_216_000_0_output_orientation_theta.mrc",
    )
    diff_psi = get_diff_indices(
        "tests/tmp/test_match_template_xenon_216_000_0_output_orientation_psi.mrc",
        "tests/tmp/output_match_template_xenon_216_000_0_output_orientation_psi.mrc",
    )

    mip = mrcfile.read("tests/tmp/test_match_template_xenon_216_000_0_output_mip.mrc")

    # Check 1: Fewer than 0.2% pixels
    ACC_PCT = 0.002  # 0.2 percent
    assert len(diff_defocus[0]) < mip.size * ACC_PCT, ">0.2 pct defocus values differ"
    assert len(diff_phi[0]) < mip.size * ACC_PCT, ">0.2 pct phi values differ"
    assert len(diff_theta[0]) < mip.size * ACC_PCT, ">0.2 pct theta values differ"
    assert len(diff_psi[0]) < mip.size * ACC_PCT, ">0.2 pct psi values differ"

    # Check 2: Overlap at least 50%
    defocus_set = set(zip(diff_defocus[0], diff_defocus[1]))
    phi_set = set(zip(diff_phi[0], diff_phi[1]))
    theta_set = set(zip(diff_theta[0], diff_theta[1]))
    psi_set = set(zip(diff_psi[0], diff_psi[1]))

    assert len(defocus_set.intersection(phi_set)) / len(defocus_set)
    assert len(defocus_set.intersection(theta_set)) / len(defocus_set)
    assert len(defocus_set.intersection(psi_set)) / len(defocus_set)

    assert len(phi_set.intersection(defocus_set)) / len(phi_set)
    assert len(phi_set.intersection(theta_set)) / len(phi_set)
    assert len(phi_set.intersection(psi_set)) / len(phi_set)

    assert len(theta_set.intersection(defocus_set)) / len(theta_set)
    assert len(theta_set.intersection(phi_set)) / len(theta_set)
    assert len(theta_set.intersection(psi_set)) / len(theta_set)

    assert len(psi_set.intersection(defocus_set)) / len(psi_set)
    assert len(psi_set.intersection(phi_set)) / len(psi_set)
    assert len(psi_set.intersection(theta_set)) / len(psi_set)


if __name__ == "__main__":
    test_core_match_template()
