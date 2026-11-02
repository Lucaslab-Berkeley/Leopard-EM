"""Tests for BaseModel2DTM serialization helpers."""

import pandas as pd

from leopard_em.pydantic_models.custom_types import BaseModel2DTM
from leopard_em.pydantic_models.data_structures import ParticleStack
from leopard_em.pydantic_models.formats import MATCH_TEMPLATE_DF_COLUMN_ORDER


class _Inner(BaseModel2DTM):
    shape: tuple[int, int]


class _Outer(BaseModel2DTM):
    name: str
    angles: tuple[float, float, float]
    inner: _Inner
    optional_pair: tuple[int, int] | None = None


def test_yaml_round_trip_with_tuple_fields(tmp_path):
    """Tuple fields (also nested ones) are written as plain YAML sequences."""
    model = _Outer(
        name="x",
        angles=(1.0, 2.5, 3.0),
        inner=_Inner(shape=(4, 5)),
        optional_pair=(6, 7),
    )
    path = tmp_path / "model.yaml"
    model.to_yaml(path)

    assert "!!python" not in path.read_text()
    assert _Outer.from_yaml(path) == model


def test_particle_stack_yaml_round_trip(tmp_path):
    """A ParticleStack config written with to_yaml can be read back with from_yaml."""
    df_path = tmp_path / "particles.csv"
    pd.DataFrame(
        {col: [0] * 2 for col in MATCH_TEMPLATE_DF_COLUMN_ORDER}
        | {"micrograph_path": ["mic.mrc"] * 2}
    ).to_csv(df_path, index=False)
    stack = ParticleStack(
        df_path=str(df_path),
        extracted_box_size=(24, 22),
        original_template_size=(16, 15),
    )
    path = tmp_path / "stack.yaml"
    stack.to_yaml(path)

    loaded = ParticleStack.from_yaml(path)
    assert loaded.model_dump() == stack.model_dump()
    assert loaded.num_particles == 2
