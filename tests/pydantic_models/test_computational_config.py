"""Tests for the ComputationalConfig model"""

import pytest
from pydantic import ValidationError

from leopard_em.pydantic_models.config import ComputationalConfigMatch


def test_default_values():
    """
    Test the default values of the ComputationalConfig.

    Verifies that the default values for gpu_ids and num_cpus are set correctly.
    """
    config = ComputationalConfigMatch()
    assert config.gpu_ids == [0]
    assert config.num_cpus == 1


def test_invalid_gpu_ids():
    """
    Test invalid gpu_ids values.

    Verifies that a ValidationError is raised when invalid gpu_ids are provided.
    """
    with pytest.raises(ValidationError):
        ComputationalConfigMatch(gpu_ids=[-1])  # Negative GPU ID is invalid

    with pytest.raises(ValidationError):
        ComputationalConfigMatch(gpu_ids=[])  # Empty list
