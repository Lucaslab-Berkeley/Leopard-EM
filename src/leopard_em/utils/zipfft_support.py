"""Support layer for the optional zipFFT library.

NOTE: Module is intentionally *not* re-exported from ``leopard_em.utils.__init__`` so
that ``import leopard_em.utils`` does not attempt to load the optional compiled
extension.
"""

from functools import cache

__all__ = [
    "ZIPFFT_AVAILABLE",
    "ZIPFFT_SUPPORTED_CONFIGS",
    "snap_image_shape_to_zipfft",
    "zipfft",
    "zipfft_supported_batch_sizes",
    "zipfft_supported_image_shapes",
]

try:
    import zipfft

    # pylint: disable=c-extension-no-member
    ZIPFFT_SUPPORTED_CONFIGS: tuple[tuple[int, ...], ...] = tuple(
        tuple(config) for config in zipfft.padded_rconv2d.get_supported_conv_configs()
    )
    ZIPFFT_AVAILABLE = True
except ImportError:
    zipfft = None
    ZIPFFT_SUPPORTED_CONFIGS = ()
    ZIPFFT_AVAILABLE = False


# Index layout of each entry returned by zipFFT's 'get_supported_conv_configs':
_SIGNAL_Y, _SIGNAL_X, _FFT_Y, _FFT_X, _BATCH, _IS_CROSS_CORR = range(6)


@cache
def zipfft_supported_image_shapes(
    template_shape: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """Image shapes compiled into zipFFT for a given template shape.

    Parameters
    ----------
    template_shape : tuple[int, int]
        Real-space template (kernel) shape ``(h, w)``.

    Returns
    -------
    tuple[tuple[int, int], ...]
        Supported image shapes, ordered smallest total area first. Empty when
        zipFFT is unavailable or was not compiled for this template shape.
    """
    shapes = {
        (int(config[_FFT_Y]), int(config[_FFT_X]))
        for config in ZIPFFT_SUPPORTED_CONFIGS
        if config[_SIGNAL_Y] == template_shape[0]
        and config[_SIGNAL_X] == template_shape[1]
        and config[_IS_CROSS_CORR]
    }
    return tuple(sorted(shapes, key=lambda shape: (shape[0] * shape[1], shape)))


@cache
def zipfft_supported_batch_sizes(
    template_shape: tuple[int, int], image_shape: tuple[int, int]
) -> tuple[int, ...]:
    """Orientation batch sizes compiled for a ``(template, image)`` shape pair.

    Parameters
    ----------
    template_shape : tuple[int, int]
        Real-space template (kernel) shape ``(h, w)``.
    image_shape : tuple[int, int]
        Image shape as zipFFT sees it. Note that the caller in
        ``leopard_em.backend.cross_correlation`` passes a *transposed* image shape,
        matching the pre-transposed layout handed to ``zipfft.padded_rconv2d.corr``.

    Returns
    -------
    tuple[int, ...]
        Supported batch sizes, largest first. Empty when zipFFT is unavailable or
        no compiled configuration matches, in which case callers should fall back
        to processing one orientation at a time.
    """
    batch_sizes = {
        int(config[_BATCH])
        for config in ZIPFFT_SUPPORTED_CONFIGS
        if config[_SIGNAL_Y] == template_shape[0]
        and config[_SIGNAL_X] == template_shape[1]
        and config[_FFT_Y] == image_shape[0]
        and config[_FFT_X] == image_shape[1]
        and config[_IS_CROSS_CORR]
    }
    return tuple(sorted(batch_sizes, reverse=True))


def snap_image_shape_to_zipfft(
    image_shape: tuple[int, int], template_shape: tuple[int, int]
) -> tuple[int, int] | None:
    """Smallest compiled zipFFT image shape that fits ``image_shape``.

    Parameters
    ----------
    image_shape : tuple[int, int]
        Real-space image shape ``(H, W)`` to be padded up.
    template_shape : tuple[int, int]
        Real-space template shape ``(h, w)``.

    Returns
    -------
    Optional[tuple[int, int]]
        Smallest square compiled shape greater than or equal to ``image_shape`` in
        both dimensions, or ``None`` if no compiled configuration fits.
    """
    for candidate in zipfft_supported_image_shapes(template_shape):
        if candidate[0] != candidate[1]:
            continue
        if candidate[0] >= image_shape[0] and candidate[1] >= image_shape[1]:
            return candidate
    return None
