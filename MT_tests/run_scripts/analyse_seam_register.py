"""Seam route 1, done as a 2x2 of real templates.

What a seam search would actually feel is the drop in correlation when the template is
placed one monomer out of register: ACF(u)/ACF(0). For a template whose two half-repeats
are identical that drop is not zero -- the template is finite, so a shifted copy overlaps
less. That finite-length term is the floor, and it must be MEASURED, not modelled: a
Fourier-symmetrised volume is not a valid null, because averaging a finite object with a
shifted copy smears its ends and charges a penalty of its own.

So four real templates, identical in length, atoms and B-factors, differing only in the
two things that can distinguish alpha from beta:

                      alpha/beta density     41/43 A alternation
    periodic                 no                     no            <- the floor
    samemonomer              no                    yes
    equalsteps              yes                     no
    observed                yes                    yes

Everything below each template's own floor is real register information, split by which
property supplies it, and resolved by low-pass cutoff.
"""

import mrcfile
import numpy as np

PIXEL_SIZE = 0.9194
MONOMER_A = 41.979
STEM = "6dpu_seam_{}_13pf_2rings_flatB_0.9194_bscale0.5.mrc"

VARIANTS = [
    ("periodic   (floor: neither)", "periodic"),
    ("samemonomer(41/43 only)", "samemonomer"),
    ("equalsteps (α/β only)", "equalsteps"),
    ("observed   (both)", "observed"),
]
CUTOFFS = [40.0, 20.0, 12.0, 8.0, 6.0, 5.0, 4.0, 3.0, None]


def retained(volume, monomer_px, cutoffs):
    """ACF(u)/ACF(0) at each low-pass cutoff, from one transform."""
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
    """Correlation retained one monomer out, and the drop below the floor."""
    monomer_px = MONOMER_A / PIXEL_SIZE
    labels = [f"{c:.0f}" if c else "full" for c in CUTOFFS]

    rows = {}
    for label, key in VARIANTS:
        with mrcfile.open(f"maps/{STEM.format(key)}", permissive=True) as handle:
            volume = np.asarray(handle.data, dtype=np.float32)
        rows[label] = retained(volume, monomer_px, CUTOFFS)

    print("Correlation retained with the template one monomer out of register\n")
    print(f"{'low-pass (Å)':<30} " + "  ".join(f"{l:>7}" for l in labels))
    print("-" * (30 + 9 * len(labels)))
    for label, _ in VARIANTS:
        print(f"{label:<30} " + "  ".join(f"{v:7.4f}" for v in rows[label]))

    floor = rows[VARIANTS[0][0]]
    print("\nDrop below the floor — the register information actually available\n")
    print(f"{'low-pass (Å)':<30} " + "  ".join(f"{l:>7}" for l in labels))
    print("-" * (30 + 9 * len(labels)))
    for label, _ in VARIANTS[1:]:
        drop = 100.0 * (floor - rows[label]) / floor
        print(f"{label:<30} " + "  ".join(f"{v:6.2f}%" for v in drop))


if __name__ == "__main__":
    main()
