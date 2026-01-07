"""File containing Fourier-slice based cross-correlation functions for 2DTM."""

import torch
from torch_fourier_slice import extract_central_slices_rfft_3d

from leopard_em.backend.utils import (
    normalize_template_projection,
    normalize_template_projection_compiled,
)


# pylint: disable=too-many-locals,E1102
def do_streamed_orientation_cross_correlate(
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    projective_filters: torch.Tensor,
    streams: list[torch.cuda.Stream],
) -> torch.Tensor:
    """Calculates a grid of 2D cross-correlations over multiple CUDA streams.

    NOTE: This function is more performant than a batched 2D cross-correlation with
    shape (N, H, W) when the kernel (template) is much smaller than the image (e.g.
    kernel is 512x512 and image is 4096x4096). Each cross-correlation is computed
    individually and stored in a batched tensor for the grid of orientations, defoci,
    and pixel size values.

    NOTE: this function returns a cross-correlogram with "same" mode (i.e. the
    same size as the input image). See numpy correlate docs for more information.

    Parameters
    ----------
    image_dft : torch.Tensor
        Real-fourier transform (RFFT) of the image with large image filters
        already applied. Has shape (H, W // 2 + 1).
    template_dft : torch.Tensor
        Real-fourier transform (RFFT) of the template volume to take Fourier
        slices from. Has shape (l, h, w // 2 + 1) where (l, h, w) is the original
        real-space shape of the template volume.
    rotation_matrices : torch.Tensor
        Rotation matrices to apply to the template volume. Has shape
        (num_orientations, 3, 3).
    projective_filters : torch.Tensor
        Multiplied 'ctf_filters' with 'whitening_filter_template'. Has shape
        (num_Cs, num_defocus, h, w // 2 + 1). Is RFFT and not fftshifted.
    streams : list[torch.cuda.Stream]
        List of CUDA streams to use for parallel computation. Each stream will
        handle a separate cross-correlation.

    Returns
    -------
    torch.Tensor
        Cross-correlation of the image with the template volume for each
        orientation and defocus value. Will have shape
        (num_Cs, num_defocus, num_orientations, H, W).
    """
    # Accounting for RFFT shape
    projection_shape_real = (template_dft.shape[1], template_dft.shape[2] * 2 - 2)
    image_shape_real = (image_dft.shape[0], image_dft.shape[1] * 2 - 2)

    num_orientations = rotation_matrices.shape[0]
    num_Cs = projective_filters.shape[0]  # pylint: disable=invalid-name
    num_defocus = projective_filters.shape[1]

    cross_correlation = torch.empty(
        size=(num_Cs, num_defocus, num_orientations, *image_shape_real),
        dtype=image_dft.real.dtype,  # Deduce the real dtype from complex DFT
        device=image_dft.device,
    )

    # Do a batched Fourier slice extraction for all the orientations at once.
    fourier_slices = extract_central_slices_rfft_3d(
        volume_rfft=template_dft,
        rotation_matrices=rotation_matrices,
    )
    fourier_slices = torch.fft.ifftshift(fourier_slices, dim=(-2,))
    fourier_slices[..., 0, 0] = 0 + 0j  # zero out the DC component (mean zero)
    fourier_slices *= -1  # flip contrast

    # Barrier to ensure Fourier slice computation on default stream is done before
    # continuing computation in parallel on non-default streams.
    default_stream = torch.cuda.default_stream(image_dft.device)
    for s in streams:
        s.wait_stream(default_stream)

    # Iterate over the orientations
    for i in range(num_orientations):
        fourier_slice = fourier_slices[i]

        # Iterate over the different pixel sizes (Cs) and defocus values for this
        # particular orientation
        for j in range(num_defocus):
            for k in range(num_Cs):
                # Use a round-robin scheduling for the streams
                job_idx = (i * num_defocus * num_Cs) + (j * num_Cs) + k
                stream_idx = job_idx % len(streams)
                stream = streams[stream_idx]

                with torch.cuda.stream(stream):
                    # Apply the projective filter and do template normalization
                    fourier_slice_filtered = fourier_slice * projective_filters[k, j]
                    projection = torch.fft.irfft2(fourier_slice_filtered)
                    projection = torch.fft.ifftshift(projection, dim=(-2, -1))
                    projection = normalize_template_projection_compiled(
                        projection,
                        projection_shape_real,
                        image_shape_real,
                    )

                    # NOTE: Decomposing 2D FFT into component 1D FFTs. Saves on first
                    # pass where many lines are zeros. Approx 6-8% speedup.
                    temp_fft = torch.fft.rfft(projection, n=image_shape_real[1], dim=-1)
                    projection_dft = torch.fft.fft(
                        temp_fft, n=image_shape_real[0], dim=-2
                    )

                    # # Padded forward Fourier transform for cross-correlation
                    # projection_dft = torch.fft.rfft2(projection, s=image_shape_real)

                    projection_dft[0, 0] = 0 + 0j

                    # Cross correlation step by element-wise multiplication
                    projection_dft = image_dft * projection_dft.conj()
                    torch.fft.irfft2(
                        projection_dft,
                        s=image_shape_real,
                        out=cross_correlation[k, j, i],
                    )

    # Record 'fourier_slices' on each stream to ensure it's not deallocated before all
    # streams are finished processing.
    for s in streams:
        fourier_slices.record_stream(s)

    # Wait for all streams to finish
    for stream in streams:
        stream.synchronize()

    # shape is (num_Cs, num_defocus, num_orientations, H, W)
    return cross_correlation


# pylint: disable=E1102
def do_batched_orientation_cross_correlate(
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    projective_filters: torch.Tensor,
) -> torch.Tensor:
    """Batched projection and cross-correlation with fixed (batched) filters.

    NOTE: This function is similar to `do_streamed_orientation_cross_correlate` but
    it computes cross-correlation batches over the orientation space. For example, if
    there are 32 orientations to process and 10 different defocus values, then there
    would be a total of 10 batched-32 cross-correlations computed.

    NOTE: that this function returns a cross-correlogram with "same" mode (i.e. the
    same size as the input image). See numpy correlate docs for more information.

    Parameters
    ----------
    image_dft : torch.Tensor
        Real-fourier transform (RFFT) of the image with large image filters
        already applied. Has shape (H, W // 2 + 1).
    template_dft : torch.Tensor
        Real-fourier transform (RFFT) of the template volume to take Fourier
        slices from. Has shape (l, h, w // 2 + 1) where (l, h, w) is the original
        real-space shape of the template volume.
    rotation_matrices : torch.Tensor
        Rotation matrices to apply to the template volume. Has shape
        (num_orientations, 3, 3).
    projective_filters : torch.Tensor
        Multiplied 'ctf_filters' with 'whitening_filter_template'. Has shape
        (num_Cs, num_defocus, h, w // 2 + 1). Is RFFT and not fftshifted.

    Returns
    -------
    torch.Tensor
        Cross-correlation of the image with the template volume for each
        orientation and defocus value. Will have shape
        (num_Cs, num_defocus, num_orientations, H, W).
    """
    # Accounting for RFFT shape
    projection_shape_real = (template_dft.shape[1], template_dft.shape[2] * 2 - 2)
    image_shape_real = (image_dft.shape[0], image_dft.shape[1] * 2 - 2)

    num_Cs = projective_filters.shape[0]  # pylint: disable=invalid-name
    num_defocus = projective_filters.shape[1]

    cross_correlation = torch.empty(
        size=(
            num_Cs,
            num_defocus,
            rotation_matrices.shape[0],
            *image_shape_real,
        ),
        dtype=image_dft.real.dtype,  # Deduce the real dtype from complex DFT
        device=image_dft.device,
    )

    # Extract central slice(s) from the template volume
    fourier_slice = extract_central_slices_rfft_3d(
        volume_rfft=template_dft,
        rotation_matrices=rotation_matrices,
    )
    fourier_slice = torch.fft.ifftshift(fourier_slice, dim=(-2,))
    fourier_slice[..., 0, 0] = 0 + 0j  # zero out the DC component (mean zero)
    fourier_slice *= -1  # flip contrast

    # Apply the projective filters on a new batch dimension
    fourier_slice = fourier_slice[None, None, ...] * projective_filters[:, :, None, ...]

    # Inverse Fourier transform into real space and normalize
    projections = torch.fft.irfftn(fourier_slice, dim=(-2, -1))
    projections = torch.fft.ifftshift(projections, dim=(-2, -1))
    projections = normalize_template_projection_compiled(
        projections,
        projection_shape_real,
        image_shape_real,
    )

    for j in range(num_defocus):
        for k in range(num_Cs):
            projections_dft = torch.fft.rfftn(
                projections[k, j, ...], dim=(-2, -1), s=image_shape_real
            )
            projections_dft[..., 0, 0] = 0 + 0j

            # Cross correlation step by element-wise multiplication
            projections_dft = image_dft[None, ...] * projections_dft.conj()
            torch.fft.irfftn(
                projections_dft, dim=(-2, -1), out=cross_correlation[k, j, ...]
            )

    return cross_correlation


# pylint: disable=E1102
def do_batched_orientation_cross_correlate_cpu(
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    projective_filters: torch.Tensor,
) -> torch.Tensor:
    """Same as `do_streamed_orientation_cross_correlate` but on the CPU.

    The only difference is that this function does not call into a compiled torch
    function for normalization.

    TODO: Figure out a better way to split up CPU/GPU functions while remaining
    performant and not duplicating code.

    Parameters
    ----------
    image_dft : torch.Tensor
        Real-fourier transform (RFFT) of the image with large image filters
        already applied. Has shape (H, W // 2 + 1).
    template_dft : torch.Tensor
        Real-fourier transform (RFFT) of the template volume to take Fourier
        slices from. Has shape (l, h, w // 2 + 1).
    rotation_matrices : torch.Tensor
        Rotation matrices to apply to the template volume. Has shape
        (orientations, 3, 3).
    projective_filters : torch.Tensor
        Multiplied 'ctf_filters' with 'whitening_filter_template'. Has shape
        (defocus_batch, h, w // 2 + 1). Is RFFT and not fftshifted.

    Returns
    -------
    torch.Tensor
        Cross-correlation for the batch of orientations and defocus values.s
    """
    # Accounting for RFFT shape
    projection_shape_real = (template_dft.shape[1], template_dft.shape[2] * 2 - 2)
    image_shape_real = (image_dft.shape[0], image_dft.shape[1] * 2 - 2)

    # Extract central slice(s) from the template volume
    fourier_slice = extract_central_slices_rfft_3d(
        volume_rfft=template_dft,
        rotation_matrices=rotation_matrices,
    )
    fourier_slice = torch.fft.ifftshift(fourier_slice, dim=(-2,))
    fourier_slice[..., 0, 0] = 0 + 0j  # zero out the DC component (mean zero)
    fourier_slice *= -1  # flip contrast

    # Apply the projective filters on a new batch dimension
    fourier_slice = fourier_slice[None, None, ...] * projective_filters[:, :, None, ...]

    # Inverse Fourier transform into real space and normalize
    projections = torch.fft.irfftn(fourier_slice, dim=(-2, -1))
    projections = torch.fft.ifftshift(projections, dim=(-2, -1))
    projections = normalize_template_projection(
        projections,
        projection_shape_real,
        image_shape_real,
    )

    # Padded forward Fourier transform for cross-correlation
    projections_dft = torch.fft.rfftn(projections, dim=(-2, -1), s=image_shape_real)
    projections_dft[..., 0, 0] = 0 + 0j  # zero out the DC component (mean zero)

    # Cross correlation step by element-wise multiplication
    projections_dft = image_dft[None, None, None, ...] * projections_dft.conj()
    cross_correlation = torch.fft.irfftn(projections_dft, dim=(-2, -1))

    return cross_correlation
