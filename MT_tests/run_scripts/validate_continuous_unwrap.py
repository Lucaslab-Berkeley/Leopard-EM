"""Does the continuous arc-length unwrap recover a register that is known to be there?

The curved synthetic is the only frame where the answer is known: a full patch search,
indexed the SEGMENTED way over the whole track, reads the register at 2.19x chance with
the dimer control at 0.60x, and its true monomer rise is 40.943 A.

So run the continuous unwrap on it. If the register survives and the rise comes back, the
continuous index is sound and can be the default; if the register is lost, the arc length
is drifting and the segmented route stays.

The synthetic also carries its own truth curve (the arc the tube was tiled along), so the
fitted curve can be scored against the real one rather than against a hand drawing.

Usage:
    python run_scripts/validate_continuous_unwrap.py
"""

import json
import pathlib
import sys

import numpy as np

from leopard_em.analysis.filament_lattice import (
    TemplateLatticeGeometry,
    extract_lattice_sites,
)
from leopard_em.pydantic_models.results.correlation_table import detections_from_hdf5

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyse_gdp_curved as A  # noqa: E402
from continuous_unwrap_curved import continuous_sites  # noqa: E402
from frame_geometry import legacy_frame_geometry  # noqa: E402
from seam_from_patch_search import alternation  # noqa: E402

TABLE = "results_curved/output_correlation_table_clean_patch13.h5"
PATCH_STEM = "6dpu_atoms_6dpv_lattice_13pf_patch5_2rings_flatB"
PATHS = "Frames/synthetic_curved_13pf_6dpv_sag100_clean.paths.json"
TRUTH = "Frames/synthetic_curved_13pf_6dpv_sag100_clean.truth.json"
KNOWN_RATIO, KNOWN_CONTROL = 2.19, 0.60


def main() -> None:
    """Compare segmented and continuous indexing on a frame with a known register."""
    truth = json.loads((MT_ROOT / TRUTH).read_text())
    geometry = TemplateLatticeGeometry.from_pdb(
        str(MT_ROOT / "models" / f"{PATCH_STEM}.pdb"))
    drawn = json.loads((MT_ROOT / PATHS).read_text())["paths"][0]["points"]
    frame = legacy_frame_geometry(MT_ROOT / TABLE, PATCH_STEM)
    print(frame.describe())
    curve = A.DrawnCurve(np.asarray(drawn, dtype=float), pixel_size=frame.pixel_size)

    detections = detections_from_hdf5(
        str(MT_ROOT / TABLE), min_z_score=7.0).reset_index(drop=True)
    print(f"{len(detections):,} patch detections, max z "
          f"{detections['z_score'].max():.2f}")
    print(f"truth: {truth['n_protofilaments']} PF, rise "
          f"{truth['monomer_rise_angstrom']:.3f} Å, sagitta {truth['sagitta_px']:.0f} px")
    print(f"known answer, SEGMENTED whole-track: {KNOWN_RATIO}x "
          f"(dimer control {KNOWN_CONTROL}x)\n")

    rng = np.random.default_rng(0)
    print(f"{'method':<14} {'sites':>6} {'occ':>6} {'rise Å':>9} {'vs truth':>10} "
          f"{'dev rms':>9} {'amp':>7} {'chance':>8} {'ratio':>7} {'ctrl':>7}")

    # The established route, for reference on identical data.
    sites = extract_lattice_sites(
        detections, geometry, frame.pixel_size, bootstrap_score_threshold=8.0)
    index = np.asarray(sites.axial_index).astype(int)
    score = np.asarray(sites.score)
    amplitude, null = alternation(index, score, rng)
    c_amp, c_null = alternation(index // 2, score, rng)
    deviation = np.asarray(sites.axial_deviation_angstrom)
    print(f"{'straight axis':<14} {len(index):6d} {sites.occupancy:5.0%} "
          f"{sites.rise_angstrom:9.3f} "
          f"{100 * (sites.rise_angstrom - truth['monomer_rise_angstrom']) / truth['monomer_rise_angstrom']:+9.2f}% "
          f"{np.sqrt((deviation ** 2).mean()):8.2f}Å {amplitude:7.3f} {null:8.3f} "
          f"{amplitude / null:6.2f}x {c_amp / c_null:6.2f}x")

    # The continuous route.
    boot = A.pick_bootstrap(detections["z_score"].to_numpy(), want=20)
    result = continuous_sites(detections, geometry, boot, pixel_size=frame.pixel_size)
    index = result["axial_index"]
    score = result["score"]
    amplitude, null = alternation(index, score, rng)
    c_amp, c_null = alternation(index // 2, score, rng)
    occupancy = len(index) / max(result["n_predicted"], 1)
    print(f"{'continuous arc':<14} {len(index):6d} {occupancy:5.0%} "
          f"{result['rise']:9.3f} "
          f"{100 * (result['rise'] - truth['monomer_rise_angstrom']) / truth['monomer_rise_angstrom']:+9.2f}% "
          f"{np.sqrt((result['deviation'] ** 2).mean()):8.2f}Å {amplitude:7.3f} "
          f"{null:8.3f} {amplitude / null:6.2f}x {c_amp / c_null:6.2f}x")

    ratio = amplitude / null
    print()
    if ratio >= 0.7 * KNOWN_RATIO:
        print(f"PASS: the continuous index keeps the register ({ratio:.2f}x against the "
              f"segmented {KNOWN_RATIO}x).\n  It is sound, and can be the default.")
    else:
        print(f"FAIL: the continuous index loses the register ({ratio:.2f}x against "
              f"{KNOWN_RATIO}x).\n  The arc length is drifting; keep the segmented route.")


if __name__ == "__main__":
    main()
