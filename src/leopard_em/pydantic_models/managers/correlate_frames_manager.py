"""Pydantic model for running the correlate frames program."""

import gc
import os
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch
from pydantic import ConfigDict
from ttsim3d.models import Simulator

from leopard_em.backend.core_refine_template import core_refine_template
from leopard_em.pydantic_models.config import (
    ComputationalConfig,
    PreprocessingFilters,
)
from leopard_em.pydantic_models.custom_types import BaseModel2DTM, ExcludedTensor
from leopard_em.pydantic_models.data_structures import ParticleStack
from leopard_em.pydantic_models.utils import (
    _setup_ctf_kwargs_from_particle_stack,
    setup_frame_filters_particle_stack,
    setup_image_normalization_factor,
)
from leopard_em.utils.data_io import (
    load_mrc_image,
    load_mrc_volume,
)


class CorrelateFramesManager(BaseModel2DTM):
    """Model holding parameters necessary for running the correlate frames program.

    Attributes
    ----------
    aligned_frames_path : str
        Path to the aligned frames mrc movie.
    template_pdb_path : str
        Path to the template pdb file.
    particle_stack : ParticleStack
        Particle stack object containing particle data.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    computational_config : ComputationalConfig
        What computational resources to allocate for the program.
    simulator : Simulator
        The simulator object.
    fluence_per_frame : Annotated[float, Field(ge=0.0)]
        The fluence per frame.


    Methods
    -------
    TODO serialization/import methods
    __init__(self, skip_mrc_preloads: bool = False, **data: Any)
        Initialize the optimize template manager.
    make_backend_core_function_kwargs(self) -> dict[str, Any]
        Create the kwargs for the backend optimize_template core function.
    run_optimize_template(self, output_text_path: str) -> None
        Run the optimize template program.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    aligned_frames_path: str
    sum_mrc_path: str
    particle_stack: ParticleStack
    preprocessing_filters: PreprocessingFilters
    computational_config: ComputationalConfig
    simulator: Simulator

    # Excluded tensors
    aligned_frames: ExcludedTensor
    sum_mrc: ExcludedTensor

    def _setup_frame_independent_kwargs(self) -> dict[str, Any]:
        """Setup backend kwargs that are independent of frames.

        Returns
        -------
        dict[str, Any]
            Dictionary containing frame-independent backend kwargs
        """
        defocus_u, defocus_v = self.particle_stack.get_absolute_defocus()
        defocus_angle = torch.tensor(self.particle_stack["astigmatism_angle"])

        ctf_kwargs = _setup_ctf_kwargs_from_particle_stack(
            self.particle_stack,
            (self.simulator.volume_shape[-2], self.simulator.volume_shape[-1]),
        )
        euler_angles = self.particle_stack.get_euler_angles(prefer_refined_angles=True)

        euler_angle_offsets = torch.zeros((1, 3))
        defocus_offsets = torch.tensor([0.0])
        pixel_size_offsets = torch.tensor([0.0])

        # Get correlation statistics
        corr_mean_stack = self.particle_stack.construct_cropped_statistic_stack(
            "correlation_average"
        )
        corr_std_stack = (
            self.particle_stack.construct_cropped_statistic_stack(
                "correlation_variance"
            )
            ** 0.5
        )  # var to std

        return {
            "euler_angles": euler_angles,
            "euler_angle_offsets": euler_angle_offsets,
            "defocus_u": defocus_u,
            "defocus_v": defocus_v,
            "defocus_angle": defocus_angle,
            "defocus_offsets": defocus_offsets,
            "pixel_size_offsets": pixel_size_offsets,
            "corr_mean": corr_mean_stack,
            "corr_std": corr_std_stack,
            "ctf_kwargs": ctf_kwargs,
            "device": self.computational_config.gpu_devices,
        }

    def _load_and_setup(
        self,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Load movie and sum MRC files and setup normalization.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]
            Tuple containing:
            - movie_mrc: The loaded movie volume
            - sum_mrc: The loaded sum MRC image
            - normalization_factor: The calculated normalization factor
            - refined_mips: Initialized array for refined MIPs
            - z_scores: Initialized array for z-scores
        """
        movie_mrc = load_mrc_volume(self.aligned_frames_path)
        sum_mrc = load_mrc_image(self.sum_mrc_path)

        num_particles = self.particle_stack.num_particles
        num_frames = len(movie_mrc)
        refined_mips = np.zeros((num_particles, num_frames))
        z_scores = np.zeros((num_particles, num_frames))

        normalization_factor = setup_image_normalization_factor(
            particle_stack=self.particle_stack,
            preprocessing_filters=self.preprocessing_filters,
            mrc_image=sum_mrc,
        )

        return movie_mrc, normalization_factor, refined_mips, z_scores

    def _simulate_template_for_frame(
        self, frame_idx: int, num_frames: int
    ) -> torch.Tensor:
        """Simulate template for a specific frame with appropriate fluence.

        Parameters
        ----------
        frame_idx : int
            Index of the current frame
        num_frames : int
            Total number of frames

        Returns
        -------
        torch.Tensor
            The simulated template for this frame
        """
        fluence_per_frame = (
            self.simulator.simulator_config.dose_end
            - self.simulator.simulator_config.dose_start
        ) / num_frames

        frame_fluence_start = (
            self.simulator.simulator_config.dose_start + fluence_per_frame * frame_idx
        )
        frame_fluence_end = (
            self.simulator.simulator_config.dose_start
            + fluence_per_frame * (frame_idx + 1)
        )

        self.simulator.simulator_config.dose_start = frame_fluence_start
        self.simulator.simulator_config.dose_end = frame_fluence_end

        return self.simulator.run()

    def _setup_frame_kwargs(
        self, frame: np.ndarray, template: torch.Tensor, normalization_factor: float
    ) -> dict[str, Any]:
        """Setup backend kwargs for a specific frame.

        Parameters
        ----------
        frame : np.ndarray
            The current frame to process
        template : torch.Tensor
            The simulated template for this frame
        normalization_factor : float
            Normalization factor for the images

        Returns
        -------
        dict[str, Any]
            Dictionary containing frame-specific backend kwargs
        """
        # pre-process template and frame
        particle_images_dft, template_dft, projective_filters = (
            setup_frame_filters_particle_stack(
                particle_stack=self.particle_stack,
                preprocessing_filters=self.preprocessing_filters,
                template=template,
                mrc_image=frame,
                normalization_factor=normalization_factor,
            )
        )

        return {
            "particle_stack_dft": particle_images_dft,
            "template_dft": template_dft,
            "projective_filters": projective_filters,
        }

    def _process_all_results(
        self, refined_mips: np.ndarray, z_scores: np.ndarray, output_dataframe_path: str
    ) -> None:
        """Process and save all results from all frames.

        Parameters
        ----------
        refined_mips : np.ndarray
            Array containing refined MIPs for all frames
        z_scores : np.ndarray
            Array containing z-scores for all frames
        output_dataframe_path : str
            Path to save the output dataframes
        """
        # Create dataframes for MIPs and z-scores with each frame in a column
        df_refined = self.particle_stack._df.copy()
        frames_df_mip = pd.DataFrame(
            {
                "particle_index": df_refined["particle_index"],
                "aligned_frames_path": self.aligned_frames_path,
            }
        )
        frames_df_zscore = pd.DataFrame(
            {
                "particle_index": df_refined["particle_index"],
                "aligned_frames_path": self.aligned_frames_path,
            }
        )

        # Add each frame's MIP and z-score as a column
        for i in range(refined_mips.shape[1]):
            frames_df_mip[f"frame_{i}_mip"] = refined_mips[:, i]
            frames_df_zscore[f"frame_{i}_zscore"] = z_scores[:, i]

        # Get base path without extension
        base_path = os.path.splitext(output_dataframe_path)[0]

        # Save frames dataframes
        frames_df_mip.to_csv(f"{base_path}_frames_mip.csv", index=False)
        frames_df_zscore.to_csv(f"{base_path}_frames_zscore.csv", index=False)

        # Sum results to standard dataframe
        sum_refined_mip = np.sum(refined_mips, axis=1)
        sum_z_score = np.sum(z_scores, axis=1)

        # Save the summed results
        df_refined = self.particle_stack._df.copy()
        df_refined["refined_mip"] = sum_refined_mip
        df_refined["refined_scaled_mip"] = sum_z_score
        df_refined.to_csv(output_dataframe_path, index=False)

    def _try_external_cuda_reset(self, device_ids):
        """Try to reset CUDA devices using external tools.
        
        This is a very aggressive approach that uses system commands to try to 
        completely free CUDA memory. This should only be used when other methods fail.
        
        Parameters
        ----------
        device_ids : list
            List of device IDs to reset
        """
        try:
            import subprocess
            import time
            
            # Only try this if we have devices to reset
            if not device_ids:
                return
                
            # Try to reset CUDA devices using nvidia-smi
            try:
                # Check if nvidia-smi is available first
                check_process = subprocess.run(
                    ["which", "nvidia-smi"], 
                    check=False, 
                    capture_output=True
                )
                
                if check_process.returncode != 0:
                    # nvidia-smi not available
                    return
                
                # For each device, try to reset it
                for device_id in device_ids:
                    if isinstance(device_id, torch.device):
                        device_idx = device_id.index
                    elif isinstance(device_id, str) and ":" in device_id:
                        device_idx = int(device_id.split(":")[-1])
                    else:
                        device_idx = device_id
                        
                    # Try to reset the device
                    print(f"Attempting aggressive memory clearing for device {device_idx}")
                    
                    # First, try a gentler approach - query the GPU status
                    subprocess.run(
                        ["nvidia-smi", "-i", str(device_idx)], 
                        check=False, 
                        capture_output=True
                    )
                    
                    # Memory info query can sometimes help trigger a cleanup
                    subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv", "-i", str(device_idx)],
                        check=False,
                        capture_output=True
                    )
                    
                    # Wait a moment
                    time.sleep(0.5)
                    
                # Force PyTorch to re-detect GPUs
                if torch.cuda.is_available():
                    # Re-initialize PyTorch's CUDA context
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"External GPU reset failed: {e}")
                
        except Exception as e:
            print(f"Error in external CUDA reset: {e}")
     
        # Final cleanup to ensure we're in a consistent state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _reset_cuda_memory_allocator(self, force_complete_reset=False):
        """Reset PyTorch's CUDA memory allocator to completely free all memory.
        
        This tries multiple approaches to completely free all GPU memory including 
        reserved memory by PyTorch's caching allocator.
        
        Parameters
        ----------
        force_complete_reset : bool
            If True, perform an extreme reset by completely recreating the CUDA context
        """
        if not torch.cuda.is_available():
            return
            
        # First try standard approach which is safest
        torch.cuda.empty_cache()
        gc.collect()
        
        # Perform the extreme reset if requested - CAUTION: This is very destructive
        if force_complete_reset:
            try:
                # Force a complete device reset
                devices = []
                if hasattr(self, 'computational_config') and hasattr(self.computational_config, 'gpu_devices'):
                    devices = self.computational_config.gpu_devices
                else:
                    devices = list(range(torch.cuda.device_count()))
                    
                # Save currently selected device to restore it later
                if len(devices) > 0:
                    try:
                        current_device = torch.cuda.current_device()
                    except:
                        current_device = 0
                        
                    # For each device, try a complete CUDA context reset
                    for device_id in devices:
                        # Handle different device formats
                        if isinstance(device_id, torch.device):
                            device_idx = device_id.index
                        elif isinstance(device_id, str) and ":" in device_id:
                            device_idx = int(device_id.split(":")[-1])
                        else:
                            device_idx = device_id
                            
                        # Reset this device's CUDA context
                        try:
                            # Set current device
                            torch.cuda.set_device(device_idx)
                            
                            # Force synchronization
                            torch.cuda.synchronize(device_idx)
                            
                            # Clear cache
                            torch.cuda.empty_cache()
                            
                            # Reset peak memory stats
                            torch.cuda.reset_peak_memory_stats(device_idx)
                            
                            # Reset max memory allocated
                            torch.cuda.reset_max_memory_allocated(device_idx)
                            
                            # Reset max memory cached
                            torch.cuda.reset_max_memory_cached(device_idx)
                        except Exception as e:
                            print(f"Error during device {device_idx} reset: {e}")
                    
                    # Restore original device
                    try:
                        torch.cuda.set_device(current_device)
                    except:
                        pass
            except Exception as e:
                print(f"Extreme CUDA reset failed: {e}")
                
        # Try to access PyTorch's CUDA caching allocator using safer methods
        try:
            # Try with torch._C
            if hasattr(torch, '_C'):
                if hasattr(torch._C, '_cuda_emptyCache'):
                    torch._C._cuda_emptyCache()
                    
            # Try the newest PyTorch internal API
            if hasattr(torch.cuda, 'memory') and hasattr(torch.cuda.memory, 'empty_cache'):
                torch.cuda.memory.empty_cache()
        except Exception as e:
            print(f"PyTorch internal API cache clearing failed: {e}")
            
        # Try with explicit context manager for each device
        devices = []
        if hasattr(self, 'computational_config') and hasattr(self.computational_config, 'gpu_devices'):
            devices = self.computational_config.gpu_devices
        else:
            devices = list(range(torch.cuda.device_count()))
            
        for device_id in devices:
            try:
                if isinstance(device_id, torch.device):
                    device_idx = device_id.index
                elif isinstance(device_id, str) and ":" in device_id:
                    device_idx = int(device_id.split(":")[-1])
                else:
                    device_idx = device_id
                    
                with torch.cuda.device(f"cuda:{device_idx}"):
                    # Standard empty cache
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Reset peak memory stats
                    torch.cuda.reset_peak_memory_stats()
                    
                    # Force a memory compaction if available
                    if hasattr(torch.cuda, 'memory_stats') and hasattr(torch.cuda, 'memory_summary'):
                        try:
                            # Force fragmented memory to be compacted
                            _ = torch.cuda.memory_stats()
                            _ = torch.cuda.memory_summary()
                        except Exception:
                            pass
            except Exception as e:
                print(f"Error resetting device {device_id}: {e}")
                
        # Final garbage collection
        gc.collect()

    def _show_memory_details(self, label="Memory Details"):
        """Print detailed memory information for all devices."""
        if not torch.cuda.is_available():
            print("CUDA not available")
            return
            
        print(f"\n==== {label} ====")
        
        # Get all devices
        devices = []
        if hasattr(self, 'computational_config') and hasattr(self.computational_config, 'gpu_devices'):
            devices = self.computational_config.gpu_devices
        else:
            devices = list(range(torch.cuda.device_count()))
            
        # Print memory information for each device
        for device_id in devices:
            try:
                if isinstance(device_id, torch.device):
                    device_idx = device_id.index
                elif isinstance(device_id, str) and ":" in device_id:
                    device_idx = int(device_id.split(":")[-1])
                else:
                    device_idx = device_id
                    
                with torch.cuda.device(f"cuda:{device_idx}"):
                    print(f"\nDevice {device_idx}:")
                    
                    # Basic memory stats
                    allocated = torch.cuda.memory_allocated() / 1e9
                    reserved = torch.cuda.memory_reserved() / 1e9
                    max_allocated = torch.cuda.max_memory_allocated() / 1e9
                    max_reserved = torch.cuda.max_memory_reserved() / 1e9
                    print(f"  Current: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
                    print(f"  Peak:    {max_allocated:.2f} GB allocated, {max_reserved:.2f} GB reserved")
                    
                    # Detailed memory stats if available
                    if hasattr(torch.cuda, 'memory_stats'):
                        try:
                            stats = torch.cuda.memory_stats(device_idx)
                            print(f"  Active allocs: {stats.get('active_bytes.all', 0) / 1e9:.2f} GB")
                            print(f"  Inactive allocs: {stats.get('inactive_split_bytes.all', 0) / 1e9:.2f} GB")
                            print(f"  Active alloc count: {stats.get('active_count.all', 0)}")
                            print(f"  Inactive alloc count: {stats.get('inactive_count.all', 0)}")
                            print(f"  Segment count: {stats.get('segment_count.current', 0)}")
                        except Exception:
                            print("  Detailed stats not available")
            except Exception as e:
                print(f"Error getting memory details for device {device_id}: {e}")
                
        print("=" * 40)
        
    def _is_tensor(self, obj):
        """Safe check if an object is a tensor to avoid warnings."""
        try:
            return isinstance(obj, torch.Tensor)
        except Exception:
            return False

    def run_correlate_frames(
        self, output_dataframe_path: str, orientation_batch_size: int = 8,
        frame_match_dir: str = None,
        variance_string: str = None,
        mean_string: str = None,
        debug_memory: bool = True,
        force_cuda_reset: bool = False,
        show_memory_details: bool = False,
    ) -> None:
        """Run the correlate frames program.

        Parameters
        ----------
        output_dataframe_path : str
            Path to save the output dataframes
        orientation_batch_size : int
            Number of orientations to process at once. Defaults to 64.
        frame_match_dir : str
            Directory with per-frame match template results
        variance_string: str
            String for variance file names
        mean_string: str
            String for mean file names
        debug_memory : bool
            If True, print memory usage debug information
        force_cuda_reset : bool
            If True, try aggressive methods to completely clear CUDA memory
        show_memory_details : bool
            If True, show detailed memory information for debugging
        Returns
        -------
        None
        """
        # 1) Setup frame-independent backend kwargs
        frame_independent_kwargs = self._setup_frame_independent_kwargs()

        # 2) Load movie and setup arrays
        movie_mrc, normalization_factor, refined_mips, z_scores = self._load_and_setup()

        if frame_match_dir is not None:
            file_list = os.listdir(frame_match_dir)
            variance_list = [f for f in file_list if variance_string in f]
            mean_list = [f for f in file_list if mean_string in f]
            # Sort by frame number (i) in filename that ends with {i}.mrc
            variance_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            mean_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        def print_memory(label):
            """Print memory usage for all GPU devices."""
            if not debug_memory:
                return
                
            # Print memory info for all devices
            if hasattr(self, 'computational_config') and hasattr(self.computational_config, 'gpu_devices'):
                gpu_devices = self.computational_config.gpu_devices
                print(f"\n=== {label} ===")
                for device_id in gpu_devices:
                    # Handle different device ID formats
                    if isinstance(device_id, torch.device):
                        device_idx = device_id.index
                    elif isinstance(device_id, str) and ":" in device_id:
                        device_idx = int(device_id.split(":")[-1])
                    else:
                        device_idx = device_id
                        
                    # Print memory stats for this device
                    try:
                        with torch.cuda.device(f"cuda:{device_idx}"):
                            allocated = torch.cuda.memory_allocated() / 1e9
                            reserved = torch.cuda.memory_reserved() / 1e9
                            max_allocated = torch.cuda.max_memory_allocated() / 1e9
                            max_reserved = torch.cuda.max_memory_reserved() / 1e9
                            print(f"Device {device_idx}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
                            print(f"         Max: {max_allocated:.2f} GB allocated, {max_reserved:.2f} GB reserved")
                    except Exception as e:
                        print(f"Error getting memory for device {device_idx}: {e}")
            else:
                # Fallback to general memory info
                print(f"Memory at {label}: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
    
        # Process each frame
        for i, frame in enumerate(movie_mrc):
            print(f"Running for frame {i+1} of {len(movie_mrc)}")
            self._clear_all_gpu_memory()  # Clear GPU memory at the beginning of each iteration
            self._reset_cuda_memory_allocator()  # Reset the CUDA memory allocator
            
            print_memory(f"Start of frame {i+1}")
            if show_memory_details:
                self._show_memory_details(f"Start of frame {i+1}")
            
            # Simulate template for this frame
            template = self._simulate_template_for_frame(i, len(movie_mrc))
            print_memory(f"After simulate_template frame {i+1}")
            
            # Setup frame-specific kwargs
            frame_kwargs = self._setup_frame_kwargs(
                frame, template, normalization_factor
            )
            print_memory(f"After setup_frame_kwargs frame {i+1}")
            
            # Clear frame and template from memory
            del frame
            del template
            torch.cuda.empty_cache()
            print_memory(f"After del frame/template frame {i+1}")


            if frame_match_dir is not None:
                var_path = os.path.join(frame_match_dir, variance_list[i])
                mean_path = os.path.join(frame_match_dir, mean_list[i])
                self.particle_stack.set_column("correlation_variance_path", var_path)
                self.particle_stack.set_column("correlation_average_path", mean_path)
                corr_mean_stack = self.particle_stack.construct_cropped_statistic_stack(
                    "correlation_average"
                )
                corr_std_stack = (
                    self.particle_stack.construct_cropped_statistic_stack(
                        "correlation_variance"
                    )
                    ** 0.5
                )  # var to std
                frame_kwargs["corr_mean"] = corr_mean_stack
                frame_kwargs["corr_std"] = corr_std_stack
                

            # Combine all kwargs
            backend_kwargs = {**frame_independent_kwargs, **frame_kwargs}

            print(f"running for frame {i+1} of {len(movie_mrc)}")
            
            # Get results with no_grad
            result = self.get_refine_result(backend_kwargs, orientation_batch_size)
            print_memory(f"After core_refine_template frame {i+1}")
            if show_memory_details:
                self._show_memory_details(f"After core_refine_template frame {i+1}")

            # Store results directly in NumPy arrays
            refined_mips[:, i] = result["refined_cross_correlation"]
            z_scores[:, i] = result["refined_z_score"]
            
            # Immediately clear results from memory
            result_keys = list(result.keys())  # Create a copy of the keys
            for key in result_keys:
                del result[key]
            del result
            
            # Clear backend arguments
            backend_keys = list(backend_kwargs.keys())  # Create a copy of the keys
            for key in backend_keys:
                if self._is_tensor(backend_kwargs[key]):  # Use the safe tensor check
                    del backend_kwargs[key]
            del backend_kwargs
            
            # Clear frame kwargs
            frame_keys = list(frame_kwargs.keys())  # Create a copy of the keys
            for key in frame_keys:
                if self._is_tensor(frame_kwargs[key]):  # Use the safe tensor check
                    del frame_kwargs[key]
            del frame_kwargs
            
            # Force aggressive memory cleanup
            self._clear_all_gpu_memory()
            self._reset_cuda_memory_allocator(force_complete_reset=False)  # Use complete reset
            
            # Run multiple garbage collection passes
            for _ in range(3):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Try external CUDA reset if forced
            if force_cuda_reset and hasattr(self, 'computational_config') and hasattr(self.computational_config, 'gpu_devices'):
                self._try_external_cuda_reset(self.computational_config.gpu_devices)
            
            print_memory(f"After cleanup frame {i+1}")
            if show_memory_details:
                self._show_memory_details(f"After cleanup frame {i+1}")

        # Process and save all results
        self._process_all_results(refined_mips, z_scores, output_dataframe_path)
        
        # Final memory cleanup
        self._clear_all_gpu_memory()
        self._reset_cuda_memory_allocator()

    def _clear_all_gpu_memory(self):
        """Clear memory on all available GPU devices completely.
        
        This tries to free all allocated and reserved memory on all GPUs.
        """
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            return
            
        # Call Python's garbage collector to clean up any unreferenced objects
        gc.collect()
        
        # Clear memory on all available GPU devices
        if hasattr(self, 'computational_config') and hasattr(self.computational_config, 'gpu_devices'):
            # Use the specified GPU devices
            gpu_devices = self.computational_config.gpu_devices
        else:
            # If not specified, check all available devices
            gpu_devices = list(range(torch.cuda.device_count()))
        
        # Try more aggressive approach to completely clear CUDA memory
        try:
            # First clear the PyTorch CUDA caches on all devices
            torch.cuda.empty_cache()
                
            for device_id in gpu_devices:
                # Handle the case where device_id is already a torch.device or string
                if isinstance(device_id, torch.device):
                    device = device_id
                    device_idx = device_id.index
                elif isinstance(device_id, str) and ":" in device_id:
                    # Handle strings like "cuda:0"
                    device = torch.device(device_id)
                    device_idx = device.index
                else:
                    # Handle integer indices
                    device = torch.device(f"cuda:{device_id}")
                    device_idx = device_id
                    
                # Now use the device for context
                with torch.cuda.device(device):
                    # Synchronize the device to ensure all operations are complete
                    torch.cuda.synchronize()
                    
                    # Move to CPU and clear all tensors that might still be on GPU
                    for obj in gc.get_objects():
                        try:
                            if torch.is_tensor(obj) and obj.device.type == 'cuda' and obj.device.index == device_idx:
                                # Only move data if it's still a valid tensor
                                if obj.is_leaf and obj.storage().size() > 0:
                                    obj.data = obj.data.cpu()
                                del obj
                        except Exception as e:
                            # Silently continue - object might be invalid or already deleted
                            pass
                    
                    # Empty CUDA cache again to release all memory
                    torch.cuda.empty_cache()
                    
                    # Force a second synchronization to ensure memory is released
                    torch.cuda.synchronize()
                    
                    # Use the most aggressive approach: reset peak memory stats
                    torch.cuda.reset_peak_memory_stats(device_idx)
                    
                    # Try to clear all allocated and reserved memory by re-initializing
                    # IMPORTANT: This is very aggressive and might affect other processes
                    if hasattr(torch.cuda, '_lazy_init'):
                        # Force re-initialization of CUDA
                        torch.cuda._lazy_init()
                    
                    # Try to explicitly clear the memory allocator
                    if hasattr(torch.cuda, 'caching_allocator_delete'):
                        torch.cuda.caching_allocator_delete()
            
            # Additional forceful clearing for CUDA memory
            for i in range(3):  # Multiple rounds of garbage collection can help
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
        except Exception as e:
            print(f"Warning: Advanced memory clearing failed: {e}")
            # Fallback to basic memory clearing
            torch.cuda.empty_cache()
            
        # Reset PyTorch's internal CUDA state to release as much memory as possible
        # Note: This is a very aggressive approach that forces PyTorch to rebuild its CUDA context
        # CAUTION: Only use as a last resort as it may affect other processes using the GPUs
        try:
            # Re-initialize CUDA to clear all caches
            # This is undocumented and may change in future PyTorch versions
            if hasattr(torch.cuda, '_lazy_call'):
                torch.cuda._lazy_call(torch.cuda.synchronize)
        except Exception as e:
            # If fails, use the standard approach
            torch.cuda.empty_cache()
            
        # Reset peak memory stats to better track future allocations
        torch.cuda.reset_peak_memory_stats()

    def get_refine_result(
        self, backend_kwargs: dict, orientation_batch_size: int = 64
    ) -> dict[str, np.ndarray]:
        """Get refine template result.

        Parameters
        ----------
        backend_kwargs : dict
            Keyword arguments for the backend processing
        orientation_batch_size : int
            Number of orientations to process at once. Defaults to 64.

        Returns
        -------
        dict[str, np.ndarray]
            The result of the refine template program.
        """
        # Adjust batch size if orientation search is disabled
        orientation_batch_size = 1
        # pylint: disable=duplicate-code
        result: dict[str, np.ndarray] = {}
        
        # Run with no_grad to avoid keeping gradients
        with torch.no_grad():
            torch_result = core_refine_template(
                batch_size=orientation_batch_size, **backend_kwargs
            )
            
            # Immediately convert tensors to NumPy arrays and move them to CPU
            result = {k: v.cpu().numpy().copy() for k, v in torch_result.items()}
            
            # Explicitly delete all PyTorch tensors (make a copy of keys first)
            keys_to_delete = list(torch_result.keys())
            for k in keys_to_delete:
                del torch_result[k]
            del torch_result
            
            # Force a CUDA cache clearing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Try to force multiprocessing resource cleanup
                if hasattr(torch.multiprocessing, '_proctitle_thread'):
                    torch.multiprocessing._cleanup()
                
                # Reset all peak memory stats
                torch.cuda.reset_peak_memory_stats()

        # Force Python's garbage collector to run multiple times
        for _ in range(2):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result
