"""Program for computing averaged template cross-correlation.

Averages template projections over all defoci and orientations.
"""

import roma
import torch
import tqdm
from torch_fourier_slice import extract_central_slices_rfft_3d, transform_slice_2d

from leopard_em.backend.distributed import run_multiprocess_jobs
from leopard_em.backend.utils import normalize_template_projection_compiled
from leopard_em.pydantic_models.managers import MatchTemplateManager
from leopard_em.utils.data_io import write_mrc_from_tensor

#######################################
### Editable parameters for program ###
#######################################

# Edit your YAML file to configure the match template program.
# See online documentation for more information on editing this file.
YAML_CONFIG_PATH = "config/match_template_config_DW_Unbend.yaml"

# Path where the averaged cross-correlagram will be saved
OUTPUT_MRC_PATH = "match_tm_output_DW/averaged_cross_correlagram_Unbend2.mrc"

# Batch size for processing orientations
ORIENTATION_BATCH_SIZE = 64
##############################################################
### Main function called to run the average template program ###
##############################################################


def _process_orientations_single_device(
    result_dict: dict,
    rank: int,
    template_dft: torch.Tensor,
    projective_filters: torch.Tensor,
    euler_angles: torch.Tensor,
    mag_matrix: torch.Tensor | None,
    device: torch.device,
    orientation_batch_size: int,
    num_Cs: int,
    num_defocus: int,
) -> None:
    """Process orientations on a single device and accumulate results.

    Parameters
    ----------
    result_dict : dict
        Shared dictionary to store results in.
    rank : int
        Rank/device index for this process.
    template_dft : torch.Tensor
        Template volume in RFFT format.
    projective_filters : torch.Tensor
        Projective filters to apply.
    euler_angles : torch.Tensor
        Euler angles to process on this device.
    mag_matrix : torch.Tensor | None
        Anisotropic magnification matrix.
    device : torch.device
        Device to run computation on.
    orientation_batch_size : int
        Batch size for processing orientations.
    num_Cs : int
        Number of pixel size values.
    num_defocus : int
        Number of defocus values.
    """
    # Move data to this device
    device_template_dft = template_dft.to(device)
    device_projective_filters = projective_filters.to(device)
    device_mag_matrix = (
        mag_matrix.to(device) if mag_matrix is not None else None
    )
    device_euler_angles = euler_angles.to(device)

    device_num_orientations = device_euler_angles.shape[0]

    # Initialize accumulator for this device
    h = template_dft.shape[1]
    w = template_dft.shape[2]
    device_averaged_slice = torch.zeros(
        (h, w), dtype=torch.complex64, device=device
    )
    device_total_count = 0

    # Calculate total projections for this device
    total_projections = device_num_orientations * num_defocus * num_Cs

    # Process orientations in batches for this device
    pbar = tqdm.tqdm(
        total=total_projections,
        desc=f"Device {rank}",
        unit="projection",
        unit_scale=True,
        position=rank,
        leave=True,
    )

    for batch_start in range(0, device_num_orientations, orientation_batch_size):
        batch_end = min(
            batch_start + orientation_batch_size, device_num_orientations
        )
        batch_size = batch_end - batch_start
        batch_euler_angles = device_euler_angles[batch_start:batch_end]

        # Convert Euler angles to rotation matrices for this batch
        batch_rotation_matrices = roma.euler_to_rotmat(
            "ZYZ", batch_euler_angles, degrees=True, device=device
        ).to(torch.float32)

        # Extract Fourier slices for this batch
        fourier_slices = extract_central_slices_rfft_3d(
            volume_rfft=device_template_dft,
            rotation_matrices=batch_rotation_matrices,
        )

        # Apply anisotropic magnification transform if provided
        if device_mag_matrix is not None:
            rfft_shape = (
                device_template_dft.shape[1],
                device_template_dft.shape[2],
            )
            stack_shape = (batch_size,)
            fourier_slices = transform_slice_2d(
                projection_image_dfts=fourier_slices,
                rfft_shape=rfft_shape,
                stack_shape=stack_shape,
                transform_matrix=device_mag_matrix,
            )

        # Apply preprocessing: ifftshift, zero DC, flip contrast
        fourier_slices = torch.fft.ifftshift(fourier_slices, dim=(-2,))
        fourier_slices[..., 0, 0] = 0 + 0j  # zero out DC component
        fourier_slices *= -1  # flip contrast

        # Apply projective filters and accumulate (batched approach)
        fourier_slices_expanded = fourier_slices[None, None, ...]
        filtered_slices = (
            fourier_slices_expanded * device_projective_filters[:, :, None, ...]
        )

        # Accumulate all filtered slices for this batch
        batch_sum = filtered_slices.sum(dim=2)  # (num_Cs, num_defocus, h, w)
        device_averaged_slice += batch_sum.sum(
            dim=(0, 1)
        )  # Sum over Cs and defocus
        device_total_count += batch_size * num_defocus * num_Cs

        # Update progress bar by number of projections in this batch
        projections_in_batch = batch_size * num_defocus * num_Cs
        pbar.update(projections_in_batch)

    pbar.close()

    # Store results in shared dictionary (move to CPU and convert to numpy)
    result_dict[rank] = {
        "averaged_slice": device_averaged_slice.cpu().numpy(),
        "total_count": device_total_count,
    }


def main() -> None:
    """Main function for running the average template cross-correlate program."""
    # Load configuration
    mt_manager = MatchTemplateManager.from_yaml(YAML_CONFIG_PATH)
    print("Loaded configuration.")

    # Get all necessary parameters from the manager
    core_kwargs = mt_manager.make_backend_core_function_kwargs()
    image_dft = core_kwargs["image_dft"]
    template_dft = core_kwargs["template_dft"]
    ctf_filters = core_kwargs["ctf_filters"]
    whitening_filter_template = core_kwargs["whitening_filter_template"]
    euler_angles = core_kwargs["euler_angles"]
    mag_matrix = core_kwargs.get("mag_matrix", None)

    # Get devices from computational config (same as match_template)
    devices = mt_manager.computational_config.gpu_devices
    if isinstance(devices, torch.device):
        devices = [devices]

    print(f"Using {len(devices)} device(s): {[str(d) for d in devices]}")

    # Move everything to CPU first (like match_template does)
    # Each device will move data to its own device during processing
    image_dft = image_dft.cpu()
    template_dft = template_dft.cpu()
    ctf_filters = ctf_filters.cpu()
    whitening_filter_template = whitening_filter_template.cpu()
    euler_angles = euler_angles.cpu()
    if mag_matrix is not None:
        mag_matrix = mag_matrix.cpu()

    print("Computing projective filters...")
    # Combine CTF filters and whitening filter to get projective filters
    # Shape: (num_Cs, num_defocus, h, w // 2 + 1)
    projective_filters = ctf_filters * whitening_filter_template[None, None, ...]
    num_Cs = projective_filters.shape[0]
    num_defocus = projective_filters.shape[1]
    num_orientations = euler_angles.shape[0]

    print(f"Number of orientations: {num_orientations}")
    print(f"Number of defocus values: {num_defocus}")
    print(f"Number of pixel size values (Cs): {num_Cs}")
    print(f"Processing in batches of {ORIENTATION_BATCH_SIZE} orientations...")

    # Split orientations across devices
    num_devices = len(devices)
    orientations_per_device = [num_orientations // num_devices] * num_devices
    # Distribute remaining orientations
    for i in range(num_orientations % num_devices):
        orientations_per_device[i] += 1

    # Prepare kwargs for each device process
    kwargs_per_device = []
    for device_idx, device in enumerate(devices):
        start_idx = sum(orientations_per_device[:device_idx])
        end_idx = start_idx + orientations_per_device[device_idx]
        device_euler_angles = euler_angles[start_idx:end_idx]

        if device_euler_angles.shape[0] == 0:
            continue

        print(
            f"Device {device_idx} ({device}): will process "
            f"{device_euler_angles.shape[0]} orientations "
            f"({start_idx}-{end_idx-1})"
        )

        kwargs = {
            "template_dft": template_dft,
            "projective_filters": projective_filters,
            "euler_angles": device_euler_angles,
            "mag_matrix": mag_matrix,
            "device": device,
            "orientation_batch_size": ORIENTATION_BATCH_SIZE,
            "num_Cs": num_Cs,
            "num_defocus": num_defocus,
        }
        kwargs_per_device.append(kwargs)

    # Run multiprocess jobs in parallel
    print("Starting parallel processing on all devices...")
    result_dict = run_multiprocess_jobs(
        target=_process_orientations_single_device,
        kwargs_list=kwargs_per_device,
    )

    # Combine results from all devices
    print("Combining results from all devices...")
    device_accumulators = []
    device_counts = []
    for rank in range(len(kwargs_per_device)):
        result = result_dict[rank]
        device_accumulators.append(torch.from_numpy(result["averaged_slice"]))
        device_counts.append(result["total_count"])

    averaged_fourier_slice = sum(device_accumulators)
    total_count = sum(device_counts)

    # Move to first device for final processing
    device = devices[0]
    averaged_fourier_slice = averaged_fourier_slice.to(device)
    image_dft = image_dft.to(device)

    # Compute the average
    averaged_fourier_slice /= total_count
    print(
        f"Averaged over {total_count} combinations "
        f"(orientations x defocus x Cs)"
    )

    # Account for RFFT shape
    projection_shape_real = (template_dft.shape[1], template_dft.shape[2] * 2 - 2)
    image_shape_real = (image_dft.shape[0], image_dft.shape[1] * 2 - 2)

    # Move template_dft to device for shape reference
    template_dft = template_dft.to(device)

    # Inverse FFT to get the averaged projection in real space
    print("Computing inverse FFT to get averaged projection...")
    projection = torch.fft.irfft2(averaged_fourier_slice)
    projection = torch.fft.ifftshift(projection, dim=(-2, -1))

    # Normalize the projection
    print("Normalizing projection...")
    projection = normalize_template_projection_compiled(
        projection,
        projection_shape_real,
        image_shape_real,
    )

    # Forward FFT with padding for cross-correlation
    print("Computing forward FFT for cross-correlation...")
    # NOTE: Decomposing 2D FFT into component 1D FFTs (as in cross_correlation.py)
    temp_fft = torch.fft.rfft(projection, n=image_shape_real[1], dim=-1)
    projection_dft = torch.fft.fft(temp_fft, n=image_shape_real[0], dim=-2)

    # Zero out DC component
    projection_dft[0, 0] = 0 + 0j

    # Cross-correlate with image
    print("Computing cross-correlation...")
    projection_dft = image_dft * projection_dft.conj()

    # Inverse FFT to get cross-correlagram
    print("Computing inverse FFT to get cross-correlagram...")
    cross_correlagram = torch.fft.irfft2(projection_dft, s=image_shape_real)

    # Save the cross-correlagram as MRC file
    print(f"Saving cross-correlagram to {OUTPUT_MRC_PATH}...")
    # Move tensor to CPU before writing
    cross_correlagram_cpu = cross_correlagram.cpu()
    write_mrc_from_tensor(
        data=cross_correlagram_cpu,
        mrc_path=OUTPUT_MRC_PATH,
        mrc_header=None,
        overwrite=True,
    )

    print("Done!")


if __name__ == "__main__":
    main()

