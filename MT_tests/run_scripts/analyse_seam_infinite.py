"""Register discrimination for a long filament, and its spatial-frequency dependence.

If a template sits one monomer out of register every alpha lands on a beta. What that
costs is ACF(u)/ACF(0) -- but for a FINITE template that is confounded by length, since a
one-monomer shift loses overlap whatever the subunits look like. The floor is measured
with a `periodic` control (identical monomers, equal steps), and the ratio to it is the
part that is really alpha/beta.

That ratio is length-independent, which is what makes it the infinite-filament answer:
measured on a 4-monomer template it is 0.8313, on a 12-monomer one 0.8339, while the
floor itself moves from 0.742 to 0.905. So a one-monomer register error costs ~17% of the
correlation however long the filament is.

Note this is NOT the same as FSC between the two monomers in isolation, which gives a
much larger 43%. Isolated monomers have hard boundaries and are treated as disjoint; in a
filament the density is continuous and a good deal of it matches in either register. The
filament measurement is the one a template search would feel.

Templates are single protofilaments 12 monomers long, which is what fits the 600 px box.
The register question is purely axial, so one protofilament carries it.

Usage:  python run_scripts/analyse_seam_infinite.py
"""

import mrcfile
import numpy as np

PIXEL_SIZE = 0.9194
MONOMER_A = 41.979
STEM = "maps/6dpu_seam_{}_1pf_6rings_flatB_0.9194_bscale0.5.mrc"

VARIANTS = [
    ("periodic    (null: neither)", "periodic"),
    ("samemonomer (41/43 Å only)", "samemonomer"),
    ("equalsteps  (α/β only)", "equalsteps"),
    ("observed    (both)", "observed"),
]
CUTOFFS = [40.0, 20.0, 12.0, 8.0, 6.0, 5.0, 4.0, 3.0, None]


def retained_by_cutoff(volume, monomer_px, cutoffs):
    """ACF(u)/ACF(0) at each low-pass cutoff, from a single transform."""
    nz, ny, nx = volume.shape
    spectrum = np.fft.fftn(np.fft.rfft(volume - volume.mean(), axis=0), axes=(1, 2))
    power = np.abs(spectrum) ** 2

    kz = np.fft.rfftfreq(nz)[:, None, None]
    ky = np.fft.fftfreq(ny)[None, :, None]
    kx = np.fft.fftfreq(nx)[None, None, :]
    radius = np.sqrt(kz**2 + ky**2 + kx**2) / PIXEL_SIZE

    index = int(round(monomer_px))
    out = []
    for cutoff in cutoffs:
        if cutoff is None:
            mask = 1.0
        else:
            edge = 1.0 / cutoff
            mask = 0.5 * (1.0 + np.cos(np.pi * np.clip(
                (radius - 0.9 * edge) / (0.2 * edge), 0.0, 1.0))) ** 2
        profile = np.fft.irfft((power * mask).sum(axis=(1, 2)), n=nz)
        out.append(float(profile[index] / profile[0]))
    return np.array(out)


def main():
    """Register retention against low-pass cutoff, relative to the measured floor."""
    monomer_px = MONOMER_A / PIXEL_SIZE
    print(f"monomer u = {MONOMER_A} Å = {monomer_px:.2f} px")
    print("Correlation kept with the template one monomer out of register, on a")
    print("12-monomer protofilament, divided by the periodic control's own value.")
    print("1.000 means α and β are indistinguishable at that resolution.\n")

    labels = [f"{c:.0f}" if c else "full" for c in CUTOFFS]
    rows = {}
    for label, key in VARIANTS:
        with mrcfile.open(STEM.format(key), permissive=True) as handle:
            volume = np.asarray(handle.data, dtype=np.float32)
        rows[label] = retained_by_cutoff(volume, monomer_px, CUTOFFS)

    floor = rows[VARIANTS[0][0]]
    print(f"{'low-pass (Å)':<28} " + "  ".join(f"{l:>8}" for l in labels))
    print("-" * (28 + 10 * len(labels)))
    print(f"{'periodic (raw floor)':<28} " + "  ".join(f"{v:8.4f}" for v in floor))
    for label, _ in VARIANTS[1:]:
        print(f"{label:<28} " + "  ".join(f"{v:8.4f}" for v in rows[label] / floor))

    print("\nRegister signal = 1 − that ratio: the fraction of the correlation lost.\n")
    print(f"{'low-pass (Å)':<28} " + "  ".join(f"{l:>8}" for l in labels))
    print("-" * (28 + 10 * len(labels)))
    for label, _ in VARIANTS[1:]:
        print(f"{label:<28} "
              + "  ".join(f"{100 * (1 - v):7.2f}%" for v in rows[label] / floor))


if __name__ == "__main__":
    main()
