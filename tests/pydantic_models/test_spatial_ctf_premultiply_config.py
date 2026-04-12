"""Tests for spatial CTF config models (isolated imports)."""

from __future__ import annotations

import importlib

import pytest
from tests._spatial_ctf_import_hacks import install_minimal_pydantic_models_packages

install_minimal_pydantic_models_packages()
importlib.import_module("leopard_em.pydantic_models.config.spatial_ctf_premultiply")

from pydantic import ValidationError  # noqa: E402

from leopard_em.pydantic_models.config.spatial_ctf_premultiply import (  # noqa: E402
    SpatialPsfConfig,
)


def test_spatial_psf_kernel_must_be_odd():
    # Even sizes >= 101 hit the odd validator (smaller values fail Field(ge=101) first).
    with pytest.raises(ValidationError, match="odd"):
        SpatialPsfConfig(kernel_size=102, grid_nx=4, grid_ny=4)


def test_spatial_psf_construct_valid():
    p = SpatialPsfConfig(kernel_size=129, grid_nx=8, grid_ny=8)
    assert p.kernel_size == 129 and p.grid_nx == 8 and p.grid_ny == 8
