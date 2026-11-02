"""Stateless helpers for the DataFrame backing each particle stack."""

import hashlib
import warnings
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from leopard_em.pydantic_models.formats import (
    MATCH_TEMPLATE_DF_COLUMN_ORDER,
    PARTICLE_ID_COLUMN,
)

# Columns holding particle positions which if changed invalidate per-particle tensors
POSITION_COLUMNS = frozenset({"pos_x", "pos_y", "refined_pos_x", "refined_pos_y"})


def any_nan_or_inf(s: pd.Series) -> bool:
    """Helper function to check if any value in the Series is NaN or infinite."""
    return bool(s.isna().any() or s.isin([float("inf"), float("-inf")]).any())


def check_required_columns(df: pd.DataFrame, source: str) -> None:
    """Raise if ``df`` lacks any column required of a particle table."""
    missing_columns = [
        col for col in MATCH_TEMPLATE_DF_COLUMN_ORDER if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing the following columns in DataFrame from {source}: "
            f"{missing_columns}"
        )


def generate_particle_ids(df: pd.DataFrame) -> list[str]:
    """Generate unique particle IDs of the form ``{mic_stem}_{local_idx:05d}``.

    Raises
    ------
    ValueError
        If ``micrograph_path`` is missing or contains empty values.
    """
    if "micrograph_path" not in df.columns:
        raise ValueError("Cannot generate particle IDs without 'micrograph_path'.")
    if df["micrograph_path"].isna().any():
        raise ValueError(
            "Cannot generate particle IDs: 'micrograph_path' is empty for "
            f"{int(df['micrograph_path'].isna().sum())} particle(s)."
        )

    groups = df.groupby("micrograph_path", sort=False).indices
    stems = {path: Path(str(path)).stem for path in groups}
    stem_counts = Counter(stems.values())

    ids = np.empty(len(df), dtype=object)
    for path, positions in groups.items():
        stem = stems[path]
        if stem_counts[stem] > 1:
            digest = hashlib.blake2b(str(path).encode(), digest_size=4).hexdigest()
            stem = f"{stem}-{digest}"
        for local_idx, position in enumerate(positions):
            ids[position] = f"{stem}_{local_idx:05d}"

    res: list[str] = ids.tolist()
    return res


def normalize_particle_dataframe(
    df: pd.DataFrame,
    on_invalid_ids: Literal["raise", "regenerate"] = "raise",
) -> pd.DataFrame:
    """Return a copy of ``df`` in the canonical in-memory particle-table layout.

    Parameters
    ----------
    df : pd.DataFrame
        The particle table.
    on_invalid_ids : Literal["raise", "regenerate"]
        What to do when a ``particle_id`` column is present but contains duplicate or
        empty values. "regenerate" replaces it (with a warning) and is used for legacy
        inputs; "raise" is used everywhere else.

    Returns
    -------
    pd.DataFrame
    """
    if df.index.name == PARTICLE_ID_COLUMN and PARTICLE_ID_COLUMN not in df.columns:
        df = df.reset_index()
    else:
        df = df.reset_index(drop=True)

    if PARTICLE_ID_COLUMN not in df.columns:
        return df

    ids = df[PARTICLE_ID_COLUMN]
    num_empty = int(ids.isna().sum() + (ids.astype(str) == "").sum())
    duplicated = ids[ids.duplicated(keep=False)].unique().tolist()
    if not num_empty and not duplicated:
        return df

    problem = (
        f"{num_empty} empty value(s)"
        if not duplicated
        else f"duplicate value(s) (e.g. {duplicated[:5]})"
    )
    if on_invalid_ids == "raise":
        raise ValueError(f"Column '{PARTICLE_ID_COLUMN}' contains {problem}.")

    warnings.warn(
        f"Column '{PARTICLE_ID_COLUMN}' contains {problem}; regenerating particle IDs.",
        UserWarning,
        stacklevel=3,
    )
    df[PARTICLE_ID_COLUMN] = generate_particle_ids(df)
    return df


def to_pixel_positions(values: np.ndarray) -> np.ndarray:
    """Convert (possibly float-typed) pixel coordinates to int64."""
    return np.rint(np.asarray(values, dtype=np.float64)).astype(np.int64)


def position_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Return the y/x position column names to use (refined preferred, per axis)."""
    y_col = "refined_pos_y" if "refined_pos_y" in df.columns else "pos_y"
    x_col = "refined_pos_x" if "refined_pos_x" in df.columns else "pos_x"
    return y_col, x_col


def pixel_positions(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the int64 ``(y, x)`` pixel positions of every particle.

    Parameters
    ----------
    df : pd.DataFrame
        The particle table. Positions are read from :func:`position_columns`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The ``(pos_y, pos_x)`` arrays, one entry per row.
    """
    y_col, x_col = position_columns(df)
    return (
        to_pixel_positions(df[y_col].to_numpy()),
        to_pixel_positions(df[x_col].to_numpy()),
    )


def as_row_positions(indexes: Any, num_particles: int) -> np.ndarray:
    """Validate and convert particle indexes to 0-based row positions.

    Parameters
    ----------
    indexes : Any
        Integer row positions (e.g. a ``pd.Index`` from :func:`group_rows`, a list or
        an array).
    num_particles : int
        Number of rows in the particle table.

    Returns
    -------
    np.ndarray
        1-D int64 array of row positions.

    Raises
    ------
    TypeError
        If ``indexes`` are not integers (e.g. ``particle_id`` strings).
    IndexError
        If any position is outside ``[0, num_particles)``.
    """
    positions = np.asarray(indexes).reshape(-1)
    if positions.size == 0:
        return positions.astype(np.int64)
    if positions.dtype.kind not in "iu":
        raise TypeError(
            "Particle indexes must be integer row positions (0..N-1), got dtype "
            f"'{positions.dtype}'. To select by particle_id, look up the positions "
            f"in the '{PARTICLE_ID_COLUMN}' column first."
        )
    positions = positions.astype(np.int64)
    out_of_range = (positions < 0) | (positions >= num_particles)
    if out_of_range.any():
        examples = positions[out_of_range][:5].tolist()
        raise IndexError(
            f"{int(out_of_range.sum())} particle index(es) out of range for a "
            f"stack of {num_particles} particles (e.g. {examples})."
        )
    return positions


def column_tensor(df: pd.DataFrame, column: str) -> torch.Tensor:
    """Return a column as a tensor that does not share memory with ``df``."""
    return torch.tensor(df[column].to_numpy().copy())


# pylint: disable=too-many-arguments
def choose_refined_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    prefer_refined: bool,
    *,
    refined_name: str,
    original_name: str,
    check_finite: bool = True,
    stacklevel: int = 3,
) -> list[str]:
    """Pick the ``refined_*`` versions of ``columns`` when they are usable.

    Parameters
    ----------
    df : pd.DataFrame
        The particle table.
    columns : Sequence[str]
        The original (unrefined) column names, e.g. ``["phi", "theta", "psi"]``.
    prefer_refined : bool
        Whether to use the ``refined_<column>`` columns at all.
    refined_name : str
        Name of the refined quantity used in warnings, e.g. "defocus values".
    original_name : str
        Name of the original quantity used in warnings.
    check_finite : bool, optional
        Fall back to the original columns (with a warning) if any refined value is NaN
        or infinite, by default True.
    stacklevel : int, optional
        Passed to ``warnings.warn``, by default 3 (the caller of the caller).

    Returns
    -------
    list[str]
        Either every ``refined_<column>`` (all present and usable) or ``columns``.
    """
    if not prefer_refined:
        return list(columns)

    refined = [f"refined_{column}" for column in columns]
    if not all(column in df.columns for column in refined):
        warnings.warn(
            f"Refined {refined_name} not found in DataFrame, using original "
            f"{original_name}...",
            stacklevel=stacklevel,
        )
        return list(columns)

    if check_finite and any(any_nan_or_inf(df[column]) for column in refined):
        warnings.warn(
            f"Refined {refined_name} contain NaN or inf values, using original "
            f"{original_name}...",
            stacklevel=stacklevel,
        )
        return list(columns)

    return refined


def group_rows(df: pd.DataFrame, column: str) -> dict[Any, np.ndarray]:
    """Group particle row positions by the value of ``column`` (e.g. an image path).

    Parameters
    ----------
    df : pd.DataFrame
        The particle table.
    column : str
        Column to group by; must be set for every particle.

    Returns
    -------
    dict[Any, np.ndarray]
        Maps each (sorted) value of ``column`` to the 0-based row positions of the
        particles sharing it.

    Raises
    ------
    ValueError
        If the column is missing or is empty for any particle.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")

    num_missing = int(df[column].isna().sum())
    if num_missing:
        raise ValueError(
            f"Column '{column}' is empty for {num_missing} particle(s); every "
            "particle must reference an image file."
        )

    # ``.indices`` gives row *positions*, which is what the positional tensors built
    # from these groups need regardless of the DataFrame's index.
    groups: dict[Any, np.ndarray] = df.groupby(column).indices
    return groups
