"""On the real data, can 2DTM tell which monomer it is sitting on?

The simulation says a template one monomer out of register -- beta on alpha -- keeps
about 96% of the correlation at 8 A and 83% at full resolution. This asks whether that
shows up in the experiment.

The test falls out of the top-down extraction for free. Sites are indexed at the MONOMER
spacing, so consecutive axial indices differ by exactly one monomer: one parity class has
the dimer template in register, the other has it one monomer off. Laterally adjacent
monomers in a B-lattice are the same type, so a whole index is one type -- except across
the seam, which is precisely why the seam would show as a phase flip in this signal.

Reported three ways, because each fails differently:

  by parity   mean z at even against odd indices -- simple, but a slow trend along the
              filament leaks into it;
  paired      consecutive (even, odd) pairs differenced -- immune to any slow trend;
  harmonic    amplitude of the period-2 component of z against axial index, against a
              null from shuffling, which needs no choice of which parity is which.

A null result here is informative: it says the two registers are indistinguishable in
this data, and the seam cannot be located by score alone however many sites are used.

Usage:  python run_scripts/analyse_register_alternation.py
"""

import numpy as np

from leopard_em.analysis.filament_lattice import (
    TemplateLatticeGeometry,
    extract_lattice_sites,
)
from leopard_em.pydantic_models.results.correlation_table import detections_from_hdf5

PIXEL_SIZE = 0.9194

DATASETS = [
    (
        "13 PF, cropped",
        "results_cropped/output_correlation_table_6dpu_13pf_2rings_flatB_crop.h5",
        "models/6dpu_13pf_2rings_flatB.pdb",
    ),
    (
        "14 PF ring, full micrograph",
        "results_full/output_correlation_table_2rings_full.h5",
        "models/6dpu_2rings_aligned_zero.pdb",
    ),
    (
        "14 PF patch, full micrograph",
        "results_full/output_correlation_table_patch4_full.h5",
        "models/6dpu_4_patches_aligned_zero.pdb",
    ),
]


def alternation(index, score, rng, n_null=4000):
    """Period-2 amplitude of score against axial index, and its shuffled null."""
    weight = score - score.mean()
    phase = np.exp(1j * np.pi * index)  # period 2 in index
    amplitude = float(np.abs((weight * phase).sum()) / len(index))
    null = np.array([
        float(np.abs((rng.permutation(weight) * phase).sum()) / len(index))
        for _ in range(n_null)
    ])
    return amplitude, float(np.percentile(null, 95)), float((null >= amplitude).mean())


def report(label, table, geometry_path, rng):
    """Parity, paired and harmonic tests on one search."""
    geometry = TemplateLatticeGeometry.from_pdb(geometry_path)
    detections = detections_from_hdf5(table, min_z_score=7.0).reset_index(drop=True)
    sites = extract_lattice_sites(
        detections, geometry, PIXEL_SIZE, bootstrap_score_threshold=8.0
    )
    index = np.asarray(sites.axial_index).astype(int)
    score = np.asarray(sites.score)
    order = np.argsort(index)
    index, score = index[order], score[order]

    print(f"\n=== {label} ===")
    print(f"  {len(index)} occupied sites of {sites.n_predicted} "
          f"({sites.occupancy:.0%}), rise {sites.rise_angstrom:.3f} Å, "
          f"z {score.min():.1f}–{score.max():.1f}")

    even, odd = score[index % 2 == 0], score[index % 2 == 1]
    if len(even) > 1 and len(odd) > 1:
        difference = even.mean() - odd.mean()
        error = np.sqrt(even.var(ddof=1) / len(even) + odd.var(ddof=1) / len(odd))
        print(f"  by parity   even {even.mean():6.3f} (n={len(even)})  "
              f"odd {odd.mean():6.3f} (n={len(odd)})  "
              f"diff {difference:+.3f} ± {error:.3f}  t = {difference / error:+.2f}")

    # Consecutive pairs only, so any drift along the filament cancels.
    lookup = dict(zip(index, score))
    pairs = [(lookup[i], lookup[i + 1]) for i in index
             if i % 2 == 0 and i + 1 in lookup]
    if len(pairs) > 1:
        delta = np.array([a - b for a, b in pairs])
        sem = delta.std(ddof=1) / np.sqrt(len(delta))
        print(f"  paired      {len(delta)} pairs, mean {delta.mean():+.3f} ± {sem:.3f}, "
              f"t = {delta.mean() / sem:+.2f}, {int((delta > 0).sum())} positive")

    amplitude, null95, p_value = alternation(index, score, rng)
    print(f"  harmonic    period-2 amplitude {amplitude:.4f}, "
          f"null 95th {null95:.4f}, p = {p_value:.3f}")

    # What the simulation says to expect, if the data reach that resolution.
    for cutoff, loss in (("20 Å", 0.0117), ("8 Å", 0.0389), ("4 Å", 0.0979)):
        print(f"    expected if the data reach {cutoff:>5}: "
              f"{loss * score.mean():.3f} z difference")


def main():
    """Run the register-alternation test on every available search."""
    rng = np.random.default_rng(0)
    for label, table, geometry in DATASETS:
        try:
            report(label, table, geometry, rng)
        except FileNotFoundError as error:
            print(f"\n=== {label} ===\n  skipped: {error}")


if __name__ == "__main__":
    main()
