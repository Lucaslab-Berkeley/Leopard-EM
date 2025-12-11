"""File containing Fourier-slice based cross-correlation functions for 2DTM."""

import torch
from torch_fourier_slice import extract_central_slices_rfft_3d

from leopard_em.backend.utils import (
    normalize_template_projection,
    normalize_template_projection_compiled,
)


def _combine_ewald_rffts_to_full_fft(
    slice_p_rfft: torch.Tensor,  # Shape: (..., H, W//2+1)
    slice_q_rfft: torch.Tensor,  # Shape: (..., H, W//2+1)
) -> torch.Tensor:
    """Combine two rfft-format Ewald sphere slices into a full FFT.

    Output assignment:
        - kx > 0: from slice_P
        - kx < 0: from slice_Q (derived via 3D Hermitian symmetry)
        - kx = 0, ky >= 0: from slice_P
        - kx = 0, ky < 0: from slice_Q

    The 3D Hermitian symmetry of the source volume implies:
        slice_Q(kx, ky) = conj(slice_P(-kx, -ky))

    This is used to derive slice_Q values at kx < 0 from slice_P values at kx > 0.
    """
    height = slice_p_rfft.shape[-2]
    width = (slice_p_rfft.shape[-1] - 1) * 2  # Recover full width

    full_fft = torch.zeros(
        *slice_p_rfft.shape[:-2],
        height,
        width,
        dtype=slice_p_rfft.dtype,
        device=slice_p_rfft.device,
    )

    # === kx >= 0 region (columns 0 to width//2): from slice_P ===
    full_fft[..., :, : width // 2 + 1] = slice_p_rfft

    # === kx < 0 region (columns width//2+1 to width-1): from slice_Q ===
    # slice_Q(kx, ky) = conj(slice_P(-kx, -ky))
    # For kx = -1, -2, ..., -(width//2 - 1):
    #   full[ky_idx, width + kx] = conj(slice_P[-kx, -ky])
    #   where -ky index = (height - ky_idx) % height

    if width > 2:  # Only if there are kx < 0 columns (excluding Nyquist)
        # Get slice_P at positive kx (columns 1 to width//2-1, excluding DC and Nyquist)
        # shape (..., height, width//2-1)
        p_pos_kx = slice_p_rfft[..., :, 1 : width // 2]

        # Flip ky: index i -> index (height - i) % height
        # This maps: 0->0, 1->height-1, 2->height-2, ..., height-1->1
        ky_indices = torch.arange(height, device=slice_p_rfft.device)
        neg_ky_indices = (-ky_indices) % height
        p_neg_ky = p_pos_kx[..., neg_ky_indices, :]

        # Conjugate to get slice_Q values
        q_neg_kx = torch.conj(p_neg_ky)

        # Flip kx order: we have kx=1,2,...,width//2-1 but need to place at
        # width-1,width-2,...,width//2+1
        q_neg_kx = torch.flip(q_neg_kx, dims=[-1])

        # Place in output
        full_fft[..., :, width // 2 + 1 :] = q_neg_kx

    # === kx = 0, ky < 0: from slice_Q directly ===
    # Rows with ky < 0: indices height//2+1 to height-1 (for even height) or
    # (height+1)//2 to height-1 (for odd height)
    ky_neg_start = height // 2 + 1 if height % 2 == 0 else (height + 1) // 2
    if ky_neg_start < height:
        full_fft[..., ky_neg_start:, 0] = slice_q_rfft[..., ky_neg_start:, 0]

    return full_fft


# pylint: disable=E1102
def _extract_and_process_ewald_flipped_slice(
    fourier_slice: torch.Tensor,
    template_dft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    projection_shape_real: tuple[int, int],
    projective_filters: torch.Tensor,
    voltage: float | None,
    pixel_size: float | None,
    apply_ewald: bool,
) -> torch.Tensor:
    """Extract and process Ewald flipped slice, apply filters, and add to original.

    If Ewald correction is enabled, extracts a flipped slice, applies projective
    filters, and adds its complex conjugate to the original fourier_slice.

    Parameters
    ----------
    fourier_slice : torch.Tensor
        The original fourier slice (with projective filters already applied).
    template_dft : torch.Tensor
        Real-fourier transform (RFFT) of the template volume.
    rotation_matrices : torch.Tensor
        Rotation matrices to apply to the template volume.
    projection_shape_real : tuple[int, int]
        Real-space shape of the projection.
    projective_filters : torch.Tensor
        Projective filters to apply to the flipped slice. Shape depends on context:
        - For batched: (num_Cs, num_defocus, h, w // 2 + 1)
        - For streamed: (h, w // 2 + 1) for individual filter
    voltage : float | None
        The voltage value for the particle image, in kV.
    pixel_size : float | None
        The pixel size value for the particle image, in Angstroms.
    apply_ewald : bool
        Whether to apply Ewald sphere correction.

    Returns
    -------
    torch.Tensor
        If Ewald correction is enabled: full FFT (complex) combining both slices.
        Otherwise: the original fourier_slice in RFFT format.
    """
    if not (apply_ewald and voltage is not None and pixel_size is not None):
        return fourier_slice

    ewald_kwargs_flipped = {
        "apply_ewald_curvature": True,
        "ewald_voltage_kv": voltage,
        "ewald_px_size": pixel_size,
        "ewald_flip_sign": True,
    }
    fourier_slice_flipped = extract_central_slices_rfft_3d(
        volume_rfft=template_dft,
        image_shape=(projection_shape_real[0],) * 3,
        rotation_matrices=rotation_matrices,
        **ewald_kwargs_flipped,
    )
    fourier_slice_flipped = torch.fft.ifftshift(fourier_slice_flipped, dim=(-2,))
    fourier_slice_flipped[..., 0, 0] = 0 + 0j  # zero out the DC component
    fourier_slice_flipped *= -1  # flip contrast

    # Apply projective filters to flipped slice
    fourier_slice_flipped = fourier_slice_flipped * projective_filters

    # Combine RFFT slices into full FFT
    full_fft_slice = _combine_ewald_rffts_to_full_fft(
        fourier_slice,
        fourier_slice_flipped,
    )

    return full_fft_slice


# pylint: disable=too-many-locals,E1102
def do_streamed_orientation_cross_correlate(
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    projective_filters: torch.Tensor,
    streams: list[torch.cuda.Stream],
    voltage: float | None = None,
    pixel_size: float | None = None,
    apply_ewald: bool = False,
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
    voltage : float | None, optional
        The voltage value for the particle image, in kV. Defaults to None.
        Should be provided when available from the particle stack.
    pixel_size : float | None, optional
        The pixel size value for the particle image, in Angstroms. Defaults to None.
        Should be provided when available from the particle stack.
    apply_ewald : bool, optional
        Whether to apply Ewald sphere correction. Defaults to False.

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
    # Prepare Ewald parameters if needed
    ewald_kwargs = {}
    if apply_ewald and voltage is not None and pixel_size is not None:
        ewald_kwargs = {
            "apply_ewald_curvature": True,
            "ewald_voltage_kv": voltage,  # Voltage is already in kV
            "ewald_px_size": pixel_size,
            "ewald_flip_sign": False,
        }

    fourier_slices = extract_central_slices_rfft_3d(
        volume_rfft=template_dft,
        image_shape=(projection_shape_real[0],) * 3,
        rotation_matrices=rotation_matrices,
        **ewald_kwargs,
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

                    # Apply Ewald correction if enabled
                    fourier_slice_filtered = _extract_and_process_ewald_flipped_slice(
                        fourier_slice=fourier_slice_filtered,
                        template_dft=template_dft,
                        rotation_matrices=rotation_matrices[i : i + 1],
                        projection_shape_real=projection_shape_real,
                        projective_filters=projective_filters[k, j],
                        voltage=voltage,
                        pixel_size=pixel_size,
                        apply_ewald=apply_ewald,
                    )

                    if apply_ewald and voltage is not None and pixel_size is not None:
                        # Full FFT requires ifft2 and take real part
                        projection = torch.fft.ifft2(fourier_slice_filtered).real
                    else:
                        # RFFT can use irfft2
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
    voltage: float | None = None,
    pixel_size: float | None = None,
    apply_ewald: bool = False,
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
    voltage : float | None, optional
        The voltage value for the particle image, in kV. Defaults to None.
        Should be provided when available from the particle stack.
    pixel_size : float | None, optional
        The pixel size value for the particle image, in Angstroms. Defaults to None.
        Should be provided when available from the particle stack.
    apply_ewald : bool, optional
        Whether to apply Ewald sphere correction. Defaults to False.

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
    # Prepare Ewald parameters if needed
    ewald_kwargs = {}
    if apply_ewald and voltage is not None and pixel_size is not None:
        ewald_kwargs = {
            "apply_ewald_curvature": True,
            "ewald_voltage_kv": voltage,  # Voltage is already in kV
            "ewald_px_size": pixel_size,
            "ewald_flip_sign": False,
        }

    fourier_slice = extract_central_slices_rfft_3d(
        volume_rfft=template_dft,
        image_shape=(projection_shape_real[0],) * 3,  # NOTE: requires cubic template
        rotation_matrices=rotation_matrices,
        **ewald_kwargs,
    )
    fourier_slice = torch.fft.ifftshift(fourier_slice, dim=(-2,))
    fourier_slice[..., 0, 0] = 0 + 0j  # zero out the DC component (mean zero)
    fourier_slice *= -1  # flip contrast

    # Apply the projective filters on a new batch dimension
    fourier_slice = fourier_slice[None, None, ...] * projective_filters[:, :, None, ...]

    # Extract and process flipped slice if Ewald correction is enabled
    fourier_slice = _extract_and_process_ewald_flipped_slice(
        fourier_slice=fourier_slice,
        template_dft=template_dft,
        rotation_matrices=rotation_matrices,
        projection_shape_real=projection_shape_real,
        projective_filters=projective_filters[:, :, None, ...],
        voltage=voltage,
        pixel_size=pixel_size,
        apply_ewald=apply_ewald,
    )

    # Inverse Fourier transform into real space and normalize
    if apply_ewald and voltage is not None and pixel_size is not None:
        # Full FFT requires ifftn and take real part
        projections = torch.fft.ifftn(fourier_slice, dim=(-2, -1)).real
    else:
        # RFFT can use irfftn
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
    voltage: float | None = None,
    pixel_size: float | None = None,
    apply_ewald: bool = False,
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
    voltage : float | None, optional
        The voltage value for the particle image, in kV. Defaults to None.
        Should be provided when available from the particle stack.
    pixel_size : float | None, optional
        The pixel size value for the particle image, in Angstroms. Defaults to None.
        Should be provided when available from the particle stack.
    apply_ewald : bool, optional
        Whether to apply Ewald sphere correction. Defaults to False.

    Returns
    -------
    torch.Tensor
        Cross-correlation for the batch of orientations and defocus values.s
    """
    # Accounting for RFFT shape
    projection_shape_real = (template_dft.shape[1], template_dft.shape[2] * 2 - 2)
    image_shape_real = (image_dft.shape[0], image_dft.shape[1] * 2 - 2)

    # Extract central slice(s) from the template volume
    # Prepare Ewald parameters if needed
    ewald_kwargs = {}
    if apply_ewald and voltage is not None and pixel_size is not None:
        ewald_kwargs = {
            "apply_ewald_curvature": True,
            "ewald_voltage_kv": voltage,  # Voltage is already in kV
            "ewald_px_size": pixel_size,
            "ewald_flip_sign": False,
        }

    fourier_slice = extract_central_slices_rfft_3d(
        volume_rfft=template_dft,
        image_shape=(projection_shape_real[0],) * 3,  # NOTE: requires cubic template
        rotation_matrices=rotation_matrices,
        **ewald_kwargs,
    )
    fourier_slice = torch.fft.ifftshift(fourier_slice, dim=(-2,))
    fourier_slice[..., 0, 0] = 0 + 0j  # zero out the DC component (mean zero)
    fourier_slice *= -1  # flip contrast

    # Apply the projective filters on a new batch dimension
    fourier_slice = fourier_slice[None, None, ...] * projective_filters[:, :, None, ...]

    # Extract and process flipped slice if Ewald correction is enabled
    fourier_slice = _extract_and_process_ewald_flipped_slice(
        fourier_slice=fourier_slice,
        template_dft=template_dft,
        rotation_matrices=rotation_matrices,
        projection_shape_real=projection_shape_real,
        projective_filters=projective_filters[:, :, None, ...],
        voltage=voltage,
        pixel_size=pixel_size,
        apply_ewald=apply_ewald,
    )

    # Inverse Fourier transform into real space and normalize
    if apply_ewald and voltage is not None and pixel_size is not None:
        # Full FFT requires ifftn and take real part
        projections = torch.fft.ifftn(fourier_slice, dim=(-2, -1)).real
    else:
        # RFFT can use irfftn
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
