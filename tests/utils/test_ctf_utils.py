"""Tests for the ctf_utils module."""

import pandas as pd
import pytest
import torch

from leopard_em.utils.ctf_utils import _setup_ctf_kwargs_from_particle_stack


def test_setup_ctf_kwargs_from_particle_stack_with_string_index():
    """CTF kwargs are read from the first row via positional access.

    The lookups must use ``.iloc[0]`` (positional) rather than ``[0]``
    (label-based), which raises a ``KeyError`` on a string index (pandas >= 3).
    """
    df = pd.DataFrame(
        {
            "voltage": [300.0, 300.0],
            "spherical_aberration": [2.7, 2.7],
            "amplitude_contrast_ratio": [0.07, 0.07],
            "phase_shift": [0.0, 0.0],
            "ctf_B_factor": [0.0, 0.0],
            "refined_pixel_size": [1.06, 1.06],
            "mag_matrix": [None, None],
            "even_zernikes": [None, None],
            "odd_zernikes": [None, None],
        },
        index=pd.Index(["p_00000", "p_00001"], name="particle_id"),
    )

    kwargs = _setup_ctf_kwargs_from_particle_stack(df, template_shape=(64, 64))

    assert kwargs["voltage"] == 300.0
    assert kwargs["spherical_aberration"] == 2.7
    assert kwargs["amplitude_contrast_ratio"] == 0.07
    assert kwargs["ctf_B_factor"] == 0.0
    assert kwargs["phase_shift"] == 0.0
    assert kwargs["pixel_size"] == 1.06
    assert kwargs["template_shape"] == (64, 64)


def _ctf_df(index=None, **overrides):
    columns = {
        "voltage": [300.0, 300.0],
        "spherical_aberration": [2.7, 2.7],
        "amplitude_contrast_ratio": [0.07, 0.07],
        "phase_shift": [0.0, 0.0],
        "ctf_B_factor": [0.0, 0.0],
        "refined_pixel_size": [1.06, 1.06],
        "mag_matrix": [None, None],
        "even_zernikes": [None, None],
        "odd_zernikes": [None, None],
    }
    columns.update(overrides)
    return pd.DataFrame(columns, index=index)


def test_setup_ctf_kwargs_from_particle_stack_with_non_zero_based_index():
    """``[0]`` would raise KeyError on any pandas version for this integer index."""
    kwargs = _setup_ctf_kwargs_from_particle_stack(
        _ctf_df(index=[3, 7]), template_shape=(64, 64)
    )
    assert kwargs["voltage"] == 300.0


@pytest.mark.parametrize(
    ("mag_matrix", "even_zernikes"),
    [
        # In-memory values straight from match_template (lists/dicts)
        ([[1.01, 0.0, 0.0, 0.99]] * 2, [{"Z40": 0.1}] * 2),
        # The same values after a CSV / HDF5 round trip (strings)
        (["[1.01, 0.0, 0.0, 0.99]"] * 2, ['{"Z40": 0.1}'] * 2),
        # Mixed representations of equal values
        (
            [[1.01, 0, 0, 0.99], "[1.01, 0.0, 0.0, 0.99]"],
            [{"Z40": 0.1}, '{"Z40": 0.1}'],
        ),
    ],
)
def test_setup_ctf_kwargs_accepts_list_and_json_values(mag_matrix, even_zernikes):
    """List/dict valued columns are compared by content, whatever the representation."""
    df = _ctf_df(mag_matrix=mag_matrix, even_zernikes=even_zernikes)
    kwargs = _setup_ctf_kwargs_from_particle_stack(df, template_shape=(64, 64))

    torch.testing.assert_close(
        kwargs["mag_matrix"], torch.tensor([[1.01, 0.0], [0.0, 0.99]])
    )
    assert set(kwargs["even_zernikes"]) == {"Z40"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"voltage": [300.0, 200.0]},
        {"mag_matrix": [[1.0, 0.0, 0.0, 1.0], [1.1, 0.0, 0.0, 1.0]]},
        {"odd_zernikes": ['{"Z31": 0.1}', '{"Z31": 0.2}']},
    ],
)
def test_setup_ctf_kwargs_rejects_differing_values(overrides):
    """Parameters that must be shared across particles raise when they differ."""
    with pytest.raises(ValueError, match="must be the same across all particles"):
        _setup_ctf_kwargs_from_particle_stack(
            _ctf_df(**overrides), template_shape=(64, 64)
        )
