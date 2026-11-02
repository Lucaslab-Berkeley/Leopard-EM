"""Tests for the stateless particle-table helpers behind the particle stacks."""

import numpy as np
import pandas as pd
import pytest
import torch

from leopard_em.pydantic_models.data_structures import ParticleStack
from leopard_em.pydantic_models.data_structures.particle_stack._table import (
    as_row_positions,
    choose_refined_columns,
    column_tensor,
    group_rows,
    pixel_positions,
    position_columns,
)


def _choose(df, prefer=True, **kwargs):
    return choose_refined_columns(
        df,
        ["phi", "psi"],
        prefer,
        refined_name="angles",
        original_name="angles",
        **kwargs,
    )


def test_choose_refined_columns_prefers_complete_refined_set():
    df = pd.DataFrame({"phi": [1.0], "psi": [2.0]})
    with pytest.warns(UserWarning, match="Refined angles not found"):
        assert _choose(df) == ["phi", "psi"]

    df["refined_phi"] = [3.0]
    with pytest.warns(UserWarning, match=r"incomplete .*missing \['refined_psi'\]"):
        assert _choose(df) == ["phi", "psi"]

    df["refined_psi"] = [4.0]
    assert _choose(df) == ["refined_phi", "refined_psi"]
    assert _choose(df, prefer=False) == ["phi", "psi"]


def test_choose_refined_columns_silent_when_absent_if_requested(recwarn):
    df = pd.DataFrame({"phi": [1.0], "psi": [2.0]})
    assert _choose(df, warn_if_absent=False) == ["phi", "psi"]
    assert not recwarn.list

    df["refined_phi"] = [3.0]  # a partial set is still warned about
    with pytest.warns(UserWarning, match="incomplete"):
        _choose(df, warn_if_absent=False)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_choose_refined_columns_non_finite(bad_value):
    df = pd.DataFrame(
        {"phi": [1.0], "psi": [2.0], "refined_phi": [bad_value], "refined_psi": [4.0]}
    )
    with pytest.warns(UserWarning, match="contain NaN or inf values"):
        assert _choose(df) == ["phi", "psi"]


def test_position_columns_use_both_refined_axes_or_neither(recwarn):
    df = pd.DataFrame({"pos_y": [1], "pos_x": [2]})
    assert position_columns(df) == ("pos_y", "pos_x")
    assert not recwarn.list  # freshly template-matched particles: no warning

    df["refined_pos_x"] = [3.0]
    with pytest.warns(UserWarning, match="Refined positions incomplete"):
        assert position_columns(df) == ("pos_y", "pos_x")

    df["refined_pos_y"] = [4.0]
    assert position_columns(df) == ("refined_pos_y", "refined_pos_x")

    df.loc[0, "refined_pos_y"] = np.nan
    with pytest.warns(UserWarning, match="Refined positions contain NaN"):
        assert position_columns(df) == ("pos_y", "pos_x")


def test_pixel_positions_round_float_coordinates():
    df = pd.DataFrame(
        {
            "pos_y": [1.4, 2.6],
            "pos_x": [0.0, 7.5],
            "refined_pos_y": [3.5, -0.6],
            "refined_pos_x": [0.4, 6.5],
        }
    )
    pos_y, pos_x = pixel_positions(df)
    assert pos_y.dtype == np.int64
    assert pos_y.tolist() == [4, -1]  # round half to even, refined preferred
    assert pos_x.tolist() == [0, 6]


def test_as_row_positions():
    assert as_row_positions(pd.Index([2, 0]), 3).tolist() == [2, 0]
    assert as_row_positions([], 3).dtype == np.int64
    with pytest.raises(TypeError, match="integer row positions"):
        as_row_positions(["mic_00000"], 3)
    with pytest.raises(IndexError, match="out of range"):
        as_row_positions([0, 3], 3)
    with pytest.raises(IndexError, match="out of range"):
        as_row_positions([-1], 3)


def test_group_rows_returns_positions_for_any_index():
    df = pd.DataFrame(
        {"micrograph_path": ["b.mrc", "a.mrc", "b.mrc"]},
        index=pd.Index(["x", "y", "z"], name="particle_id"),
    )
    groups = group_rows(df, "micrograph_path")
    assert list(groups) == ["a.mrc", "b.mrc"]
    assert groups["a.mrc"].tolist() == [1]
    assert groups["b.mrc"].tolist() == [0, 2]

    with pytest.raises(ValueError, match="not found"):
        group_rows(df, "mip_path")
    df.loc["y", "micrograph_path"] = None
    with pytest.raises(ValueError, match="empty for 1 particle"):
        group_rows(df, "micrograph_path")


def test_column_tensor_does_not_share_memory():
    df = pd.DataFrame({"defocus_u": [1.0, 2.0]})
    tensor = column_tensor(df, "defocus_u")
    tensor += 1
    assert df["defocus_u"].tolist() == [1.0, 2.0]
    assert torch.equal(tensor, torch.tensor([2.0, 3.0], dtype=torch.float64))


def test_get_euler_angles_falls_back_on_non_finite_refined_angles():
    stack = ParticleStack(
        df_path="",
        extracted_box_size=(4, 4),
        original_template_size=(2, 2),
        skip_df_load=True,
    )
    stack._df = pd.DataFrame(
        {
            "phi": [1.0, 2.0],
            "theta": [3.0, 4.0],
            "psi": [5.0, 6.0],
            "refined_phi": [10.0, 20.0],
            "refined_theta": [30.0, np.nan],
            "refined_psi": [50.0, 60.0],
        }
    )
    with pytest.warns(UserWarning, match="Refined angles contain NaN"):
        angles = stack.get_euler_angles()
    assert angles.tolist() == [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]

    stack.set_column("refined_theta", [30.0, 40.0])
    assert stack.get_euler_angles()[:, 1].tolist() == [30.0, 40.0]
