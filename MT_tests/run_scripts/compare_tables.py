"""Compare two correlation tables detection by detection (e.g. Leopard-EM vs SVD Stage 2).

usage: python run_scripts/compare_tables.py --a <table A> --b <table B>
           [--a-half N] [--b-half N] [--min-z 6] [--radius 3] [--angle 6]

Positions are compared in the micrograph (particle-centre) frame: x + half, where half is
template_shape // 2 from each table's metadata (or --a-half / --b-half for tables that
predate it). A row of A is matched when B has a row within --radius px whose rotation is
within --angle degrees (the two searches sample orientations differently, so exact grid
points never coincide). Reported per z bin of A, and the same the other way round.
"""

from __future__ import annotations

import argparse
import pathlib

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from leopard_em.pydantic_models.results.correlation_table import detections_from_hdf5

BINS = [6, 7, 8, 9, 10, 12, 100]


def load(path: pathlib.Path, half: int | None, min_z: float) -> pd.DataFrame:
    if half is None:
        with h5py.File(path, "r") as f:
            ts = f["metadata"].attrs.get("template_shape")
        if ts is None:
            raise ValueError(f"{path} records no template_shape: pass its half box")
        half = int(np.asarray(ts)[-1]) // 2
    df = detections_from_hdf5(str(path), min_z_score=min_z)
    df["xi"], df["yi"] = df.x + half, df.y + half
    return df.reset_index(drop=True)


def matched(a: pd.DataFrame, b: pd.DataFrame, radius: float, angle: float) -> np.ndarray:
    """Per row of a: whether b has a row within radius px and angle degrees."""
    out = np.zeros(len(a), bool)
    if len(a) == 0 or len(b) == 0:
        return out
    tree = cKDTree(b[["yi", "xi"]].to_numpy(float))
    qa = Rotation.from_euler("ZYZ", a[["phi", "theta", "psi"]].to_numpy(), degrees=True).as_quat()
    qb = Rotation.from_euler("ZYZ", b[["phi", "theta", "psi"]].to_numpy(), degrees=True).as_quat()
    cos_half = np.cos(np.radians(angle) / 2)
    for i, cand in enumerate(tree.query_ball_point(a[["yi", "xi"]].to_numpy(float), radius)):
        if cand:
            out[i] = bool((np.abs(qb[cand] @ qa[i]) >= cos_half).any())
    return out


def report(name: str, df: pd.DataFrame, ok: np.ndarray) -> None:
    z = df.z_score.to_numpy()
    cells = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = (z >= lo) & (z < hi)
        if sel.any():
            cells.append(f"z {lo}-{hi if hi < 100 else '+'}: {ok[sel].mean():5.1%} of {sel.sum():>8,}")
    print(f"{name}: {ok.mean():.1%} of {len(df):,} rows matched")
    print("    " + " | ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=pathlib.Path, required=True)
    ap.add_argument("--b", type=pathlib.Path, required=True)
    ap.add_argument("--a-half", type=int, default=None)
    ap.add_argument("--b-half", type=int, default=None)
    ap.add_argument("--min-z", type=float, default=6.0)
    ap.add_argument("--radius", type=float, default=3.0)
    ap.add_argument("--angle", type=float, default=6.0)
    ap.add_argument("--max-rows", type=int, default=2_000_000,
                    help="randomly subsample larger tables (matching is per row)")
    a = ap.parse_args()
    A, B = load(a.a, a.a_half, a.min_z), load(a.b, a.b_half, a.min_z)
    rng = np.random.default_rng(0)
    As = A.iloc[rng.choice(len(A), a.max_rows, replace=False)] if len(A) > a.max_rows else A
    Bs = B.iloc[rng.choice(len(B), a.max_rows, replace=False)] if len(B) > a.max_rows else B
    print(f"A {a.a}: {len(A):,} rows with z >= {a.min_z}, max z {A.z_score.max():.2f}")
    print(f"B {a.b}: {len(B):,} rows with z >= {a.min_z}, max z {B.z_score.max():.2f}")
    print(f"match: within {a.radius} px and {a.angle} deg")
    report("A found by B", As, matched(As, B, a.radius, a.angle))
    report("B found by A", Bs, matched(Bs, A, a.radius, a.angle))


if __name__ == "__main__":
    main()
