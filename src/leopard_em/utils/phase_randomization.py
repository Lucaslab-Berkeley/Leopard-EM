"""Phase-randomised copies of an image, for null (noise-only) searches.

A phase-randomised copy keeps the exact Fourier amplitudes of the image, and so its
power spectrum (CTF, astigmatism, ice, detector), but replaces every phase with a
random one. Searching it with the same template and settings as the image gives the
null distribution of the search: what it finds when no particles are present.
"""

import numpy as np
import torch


def phase_randomize_image(image: torch.Tensor | np.ndarray, seed: int) -> torch.Tensor:
    """Return a copy of ``image`` with its Fourier amplitudes kept and phases random.

    The amplitudes ``|F|`` are kept exactly. Each phase is taken from the real FFT of
    white Gaussian noise drawn with ``numpy.random.default_rng(seed)``, which has the
    Hermitian symmetry of a real image, so the result is real and its amplitude
    spectrum equals the input's to rounding. The DC term (the image mean) is kept.
    The computation is done in float64; the same image and seed always give the same
    copy.

    Parameters
    ----------
    image : torch.Tensor | np.ndarray
        2-D real image.
    seed : int
        Seed of the random phases. Use one seed per copy, and different seeds for
        different micrographs.

    Returns
    -------
    torch.Tensor
        The phase-randomised image, on the CPU, in the input's floating dtype
        (float64 for integer input).
    """
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {array.shape}.")
    out_dtype = array.dtype if np.issubdtype(array.dtype, np.floating) else np.float64
    x = array.astype(np.float64)

    spectrum = np.fft.rfft2(x)
    noise = np.fft.rfft2(np.random.default_rng(seed).standard_normal(x.shape))
    magnitude = np.abs(noise)
    phase = np.ones_like(noise)
    np.divide(noise, magnitude, out=phase, where=magnitude > 0)

    randomized = np.abs(spectrum) * phase
    randomized[0, 0] = spectrum[0, 0]
    out = np.fft.irfft2(randomized, s=x.shape).astype(out_dtype)
    return torch.from_numpy(out)
