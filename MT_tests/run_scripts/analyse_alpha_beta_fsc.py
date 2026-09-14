"""How different are α and β, as a function of spatial frequency?

This is the infinite-filament form of the register question, and it has no
template-length term to correct for. For a filament of alternating monomers A and B, the
correlation kept when the template sits one monomer out of register -- β on α -- is

    ACF(u)/ACF(0) = 2·Re⟨A,B⟩ / (⟨A,A⟩ + ⟨B,B⟩)

which is exactly the normalised cross-correlation between the two monomers. Per
resolution shell that is FSC(α, β). Its null is exactly 1: if α and β were the same
protein the FSC would be 1 everywhere and the register would carry no information at all.
Whatever it falls short of 1 by is the fraction of the correlation a search loses by
being one monomer out.

Both monomers were written on the same lattice site and centred by the same vector, so
they are compared in the orientation the lattice actually presents them -- which is what
a template search sees, and is not the same as their similarity after optimal
superposition.

Usage:  python run_scripts/analyse_alpha_beta_fsc.py
"""

import mrcfile
import numpy as np

PIXEL_SIZE = 0.9194
ALPHA = "maps/6dpu_seam_monomer_alpha_0.9194_bscale0.5.mrc"
BETA = "maps/6dpu_seam_monomer_beta_0.9194_bscale0.5.mrc"
SHELLS = [40.0, 20.0, 12.0, 8.0, 6.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.0]


def load(path):
    """Simulated volume, mean removed."""
    with mrcfile.open(path, permissive=True) as handle:
        volume = np.asarray(handle.data, dtype=np.float32)
    return volume - volume.mean()


def shell_correlation(a, b, shells):
    """FSC between two volumes, plus the all-frequency value."""
    fa = np.fft.rfftn(a)
    fb = np.fft.rfftn(b)
    nz, ny, nx = a.shape

    kz = np.fft.fftfreq(nz)[:, None, None]
    ky = np.fft.fftfreq(ny)[None, :, None]
    kx = np.fft.rfftfreq(nx)[None, None, :]
    with np.errstate(divide="ignore"):
        resolution = PIXEL_SIZE / np.sqrt(kz**2 + ky**2 + kx**2)

    def correlation(mask):
        cross = float(np.real(fa[mask] * np.conj(fb[mask])).sum())
        norm = np.sqrt(float((np.abs(fa[mask]) ** 2).sum())
                       * float((np.abs(fb[mask]) ** 2).sum()))
        return cross / norm if norm > 0 else np.nan

    everything = np.isfinite(resolution) & (resolution <= shells[0])
    out = [correlation(everything)]
    for low, high in zip(shells[:-1], shells[1:]):
        out.append(correlation((resolution <= low) & (resolution > high)))
    return np.array(out)


def main():
    """FSC(α, β), and what it means for a one-monomer register error."""
    alpha, beta = load(ALPHA), load(BETA)

    # A control that must return 1.000: alpha against itself. Anything less would mean
    # the shell machinery, not the proteins, is producing the difference.
    control = shell_correlation(alpha, alpha, SHELLS)
    measured = shell_correlation(alpha, beta, SHELLS)

    labels = ["all"] + [f"{a:.0f}–{b:.0f}" for a, b in zip(SHELLS[:-1], SHELLS[1:])]
    print("FSC between α and β on the same lattice site.")
    print("1.000 = indistinguishable, register carries no information.\n")
    print(f"{'shell (Å)':<14} " + "  ".join(f"{l:>8}" for l in labels))
    print("-" * (14 + 10 * len(labels)))
    print(f"{'α vs α (check)':<14} " + "  ".join(f"{v:8.4f}" for v in control))
    print(f"{'α vs β':<14} " + "  ".join(f"{v:8.4f}" for v in measured))
    print(f"\n{'register signal':<14} "
          + "  ".join(f"{100 * (1 - v):7.2f}%" for v in measured))
    print("\n(register signal = 1 − FSC: the fraction of the correlation lost by")
    print(" placing the template one monomer out, with β on α.)")


if __name__ == "__main__":
    main()
