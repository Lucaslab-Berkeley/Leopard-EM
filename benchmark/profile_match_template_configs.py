"""Benchmarking script for core_match_template performance.

Benchmark is using a 4096 x 4096 pixel image with a 512 x 512 x 512 template. Smaller
size images will make a bigger performance impact than reducing the template volume.
This script can be modified to benchmark images of other sizes (e.g. K3 images).

NOTE: This benchmark can take up to 10 minutes given the moderate sized search space and
GPU requirements.
"""

import csv
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import click
import numpy as np
import psutil
import torch

from leopard_em.backend.core_match_template import core_match_template
from leopard_em.pydantic_models.managers import MatchTemplateManager

# Import wrapper around Python-based NVIDIA Management Library (NVML)
try:
    import pynvml
except ImportError as e:
    raise ImportError(
        "pynvml is required for GPU monitoring. "
        "Install via 'pip install nvidia-ml-py' (may differ on your setup)"
    ) from e

# Reference files to download for benchmarking
DOWNLOAD_DIR = (Path(__file__).parent / "tmp").resolve()
YAML_PATH = (
    Path(DOWNLOAD_DIR) / "test_match_template_xenon_216_000_0.0_DWS_config.yaml"
).resolve()
ZENODO_URL = "https://zenodo.org/records/17069838"


class GPUMonitor:
    """Monitor GPU and system metrics during benchmark execution.

    Supports monitoring multiple GPUs simultaneously when multi-GPU execution is
    enabled. Each GPU's metrics are written to separate columns in the CSV with
    device-specific prefixes.
    """

    def __init__(
        self,
        device_ids: list[int] | int = 0,
        sample_interval: float = 1.0,
        csv_path: str = "gpu_monitoring.csv",
        config_metadata: dict | None = None,
    ):
        """Initialize GPU monitor.

        Parameters
        ----------
        device_ids : list[int] | int
            List of CUDA device IDs to monitor, or single device ID as int
        sample_interval : float
            Sampling interval in seconds
        csv_path : str
            Path to output CSV file for monitoring data
        config_metadata : dict | None
            Optional metadata about the benchmark configuration to include in CSV
        """
        if csv_path is None:
            raise ValueError("csv_path must be provided for GPUMonitor")

        # Convert single device_id to list for uniform handling
        self.device_ids = [device_ids] if isinstance(device_ids, int) else device_ids
        self.sample_interval = sample_interval
        self.csv_path = csv_path
        self.config_metadata = config_metadata or {}
        self.monitoring = False
        self.monitor_thread = None
        self.samples = []
        self.current_phase = "idle"
        self.has_error = False
        self.error_type = None
        self.csv_file = None
        self.csv_writer = None

        # Initialize NVML
        pynvml.nvmlInit()

        # Create handles for all devices
        self.handles = {}
        self.max_sm_clocks = {}
        self.max_mem_clocks = {}

        for device_id in self.device_ids:
            self.handles[device_id] = pynvml.nvmlDeviceGetHandleByIndex(device_id)

            # Get max clock speeds for reference
            try:
                self.max_sm_clocks[device_id] = pynvml.nvmlDeviceGetMaxClockInfo(
                    self.handles[device_id], pynvml.NVML_CLOCK_SM
                )
                self.max_mem_clocks[device_id] = pynvml.nvmlDeviceGetMaxClockInfo(
                    self.handles[device_id], pynvml.NVML_CLOCK_MEM
                )
            except Exception:
                self.max_sm_clocks[device_id] = None
                self.max_mem_clocks[device_id] = None

        # Initialize psutil for system monitoring
        self.process = psutil.Process()

    def __del__(self) -> None:
        """Cleanup NVML and close CSV file."""
        if self.csv_file:
            self.csv_file.close()

        pynvml.nvmlShutdown()

    def set_phase(self, phase_name: str) -> None:
        """Set the current benchmark phase.

        Parameters
        ----------
        phase_name : str
            Name of the current benchmark phase (e.g., 'setup', 'execution')
        """
        self.current_phase = phase_name

    def mark_error(self, error_type: str) -> None:
        """Mark that an error occurred and stop monitoring.

        Parameters
        ----------
        error_type : str
            Type or description of the error encountered
        """
        self.has_error = True
        self.error_type = error_type
        self.monitoring = False

    def _get_pcie_throughput(self, handle) -> tuple[float | None, float | None]:
        """Get PCIe throughput in MB/s, if available."""
        try:
            # PCIe throughput in KB/s, convert to MB/s
            pcie_tx = (
                pynvml.nvmlDeviceGetPcieThroughput(
                    handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                )
                / 1024.0
            )
            pcie_rx = (
                pynvml.nvmlDeviceGetPcieThroughput(
                    handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                )
                / 1024.0
            )
        except Exception:
            pcie_tx = None
            pcie_rx = None

        return pcie_tx, pcie_rx

    def _get_device_temperature(self, handle) -> float | None:
        """Get GPU temperature in Celsius, if available."""
        try:
            temperature = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            temperature = None

        return temperature

    def _get_device_power_usage(self, handle) -> float | None:
        """Get GPU power usage in Watts, if available."""
        try:
            # mW to W
            power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            power_usage = None

        return power_usage

    def _get_device_clock_speeds(self, handle) -> tuple[float | None, float | None]:
        """Get GPU clock speeds (SM clock, memory clock) in MHz, if available.

        Parameters
        ----------
        handle : pynvml.nvmlDevice_t
            NVML device handle
        """
        try:
            sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        except Exception:
            sm_clock = None
            mem_clock = None

        return sm_clock, mem_clock

    def _get_gpu_metrics(self, device_id: int) -> dict[str, Any]:
        """Collect all metrics for a single GPU device.

        Parameters
        ----------
        device_id : int
            CUDA device ID to collect metrics for

        Returns
        -------
        Dictionary of metrics with device-specific keys (e.g., 'gpu0_util_percent')
        """
        handle = self.handles[device_id]
        prefix = f"gpu{device_id}_"

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

        # Get specialized metrics, if available setting to None if error
        temperature = self._get_device_temperature(handle)
        power_usage = self._get_device_power_usage(handle)
        sm_clock, mem_clock = self._get_device_clock_speeds(handle)
        pcie_tx, pcie_rx = self._get_pcie_throughput(handle)

        return {
            f"{prefix}util_percent": utilization.gpu,
            f"{prefix}memory_used_mb": mem_info.used / (1024**2),
            f"{prefix}memory_total_mb": mem_info.total / (1024**2),
            f"{prefix}memory_util_percent": (mem_info.used / mem_info.total) * 100,
            f"{prefix}memory_bandwidth_util_percent": utilization.memory,
            f"{prefix}temperature_c": temperature,
            f"{prefix}power_w": power_usage,
            f"{prefix}sm_clock_mhz": sm_clock,
            f"{prefix}memory_clock_mhz": mem_clock,
            f"{prefix}pcie_tx_throughput_MBps": pcie_tx,
            f"{prefix}pcie_rx_throughput_MBps": pcie_rx,
        }

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                timestamp = time.perf_counter()

                # Get system CPU and memory stats (once per sample)
                try:
                    system_cpu_percent = psutil.cpu_percent(interval=None)
                    system_memory = psutil.virtual_memory()
                    system_memory_percent = system_memory.percent
                    system_memory_used_gb = system_memory.used / (1024**3)
                    system_memory_available_gb = system_memory.available / (1024**3)
                except Exception:
                    system_cpu_percent = None
                    system_memory_percent = None
                    system_memory_used_gb = None
                    system_memory_available_gb = None

                # Start with base sample data
                sample = {
                    "timestamp": timestamp,
                    "phase": self.current_phase,
                    "config_id": self.config_metadata.get("config_id"),
                    "run_id": self.config_metadata.get("run_id"),
                    "orientation_batch_size": self.config_metadata.get(
                        "orientation_batch_size"
                    ),
                    "num_cuda_streams": self.config_metadata.get("num_cuda_streams"),
                    "num_gpus": len(self.device_ids),
                    "gpu_ids": ",".join(map(str, self.device_ids)),
                    "has_error": self.has_error,
                    "error_type": self.error_type,
                    # System metrics (shared across all GPUs)
                    "system_cpu_percent": system_cpu_percent,
                    "system_memory_percent": system_memory_percent,
                    "system_memory_used_gb": system_memory_used_gb,
                    "system_memory_available_gb": system_memory_available_gb,
                }

                # Collect metrics for each GPU
                for device_id in self.device_ids:
                    gpu_metrics = self._get_gpu_metrics(device_id)
                    sample.update(gpu_metrics)

                self.samples.append(sample)

                # Write to CSV if file is open
                if self.csv_writer:
                    self.csv_writer.writerow(sample)
                    self.csv_file.flush()  # Ensure data is written immediately

            except Exception as e:
                print(f"Warning: Monitoring error: {e}")

            time.sleep(self.sample_interval)

    def _generate_fieldnames(self) -> list[str]:
        """Generate CSV fieldnames dynamically based on number of GPUs.

        Returns
        -------
            List of column names for the CSV file
        """
        # Base fieldnames (common to all configs)
        base_fields = [
            "timestamp",
            "phase",
            "config_id",
            "run_id",
            "orientation_batch_size",
            "num_cuda_streams",
            "num_gpus",
            "gpu_ids",
            "has_error",
            "error_type",
        ]

        # GPU-specific metrics (repeated for each GPU)
        gpu_metric_names = [
            "util_percent",
            "memory_used_mb",
            "memory_total_mb",
            "memory_util_percent",
            "memory_bandwidth_util_percent",
            "temperature_c",
            "power_w",
            "sm_clock_mhz",
            "memory_clock_mhz",
            "pcie_tx_throughput_MBps",
            "pcie_rx_throughput_MBps",
        ]

        # Generate per-GPU fieldnames
        gpu_fields = []
        for device_id in self.device_ids:
            for metric in gpu_metric_names:
                gpu_fields.append(f"gpu{device_id}_{metric}")

        # System-level metrics (shared)
        system_fields = [
            "system_cpu_percent",
            "system_memory_percent",
            "system_memory_used_gb",
            "system_memory_available_gb",
        ]

        return base_fields + gpu_fields + system_fields

    def start(self):
        """Start monitoring in background thread."""
        self.samples = []
        self.monitoring = True
        self.has_error = False
        self.error_type = None

        # Open CSV file for writing
        self.csv_file = open(self.csv_path, "w", newline="")
        fieldnames = self._generate_fieldnames()
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        self.csv_file.flush()

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self) -> dict[str, Any]:
        """Stop monitoring and return statistics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

        # Close CSV file
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

        if not self.samples:
            return {}

        # Helper function to calculate statistics for a metric
        def calc_stats(values: np.ndarray) -> dict[str, float]:
            """Calculate min, max, mean, std for a numpy array."""
            return {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()),
            }

        # Helper function to extract non-None values
        def extract_valid(key: str) -> list:
            """Extract non-None values for a given key from samples."""
            return [s[key] for s in self.samples if s[key] is not None]

        # Build statistics dictionary
        stats = {
            "num_samples": len(self.samples),
            "duration": self.samples[-1]["timestamp"] - self.samples[0]["timestamp"],
            "csv_path": str(self.csv_path) if self.csv_path else None,
            "num_gpus": len(self.device_ids),
            "gpu_ids": self.device_ids,
        }

        # Collect statistics for each GPU
        for device_id in self.device_ids:
            prefix = f"gpu{device_id}_"
            gpu_stats = {}

            # Always-present GPU metrics
            required_metrics = [
                ("util_percent", "gpu_utilization_percent"),
                ("memory_used_mb", "memory_used_mb"),
                ("memory_util_percent", "memory_utilization_percent"),
                (
                    "memory_bandwidth_util_percent",
                    "memory_bandwidth_utilization_percent",
                ),
            ]

            for sample_suffix, stats_suffix in required_metrics:
                key = f"{prefix}{sample_suffix}"
                values = np.array([s[key] for s in self.samples])
                gpu_stats[stats_suffix] = calc_stats(values)

            # Optional GPU metrics
            optional_metrics = [
                ("temperature_c", "temperature_c"),
                ("power_w", "power_usage_w"),
                ("sm_clock_mhz", "sm_clock_mhz"),
                ("memory_clock_mhz", "memory_clock_mhz"),
                ("pcie_tx_throughput_MBps", "pcie_tx_throughput_mbps"),
                ("pcie_rx_throughput_MBps", "pcie_rx_throughput_mbps"),
            ]

            for sample_suffix, stats_suffix in optional_metrics:
                key = f"{prefix}{sample_suffix}"
                values = extract_valid(key)
                if values:
                    gpu_stats[stats_suffix] = calc_stats(np.array(values))

            # Add GPU-specific stats to main stats dictionary
            stats[f"gpu{device_id}"] = gpu_stats

        # Add system-level statistics (shared across all GPUs)
        system_metrics = [
            ("system_cpu_percent", "system_cpu_percent"),
            ("system_memory_percent", "system_memory_percent"),
        ]

        for sample_key, stats_key in system_metrics:
            values = extract_valid(sample_key)
            if values:
                stats[stats_key] = calc_stats(np.array(values))

        return stats


def download_comparison_data() -> None:
    """Downloads the example data from Zenodo."""
    subprocess.run(
        ["zenodo_get", f"--output-dir={DOWNLOAD_DIR}", ZENODO_URL], check=True
    )

    # Change the paths pointing to the tests/tmp directory to benchmark/tmp directory
    # in the downloaded YAML file
    with open(YAML_PATH) as f:
        yaml_text = f.read()

    yaml_text = yaml_text.replace("tests/tmp", "benchmark/tmp")

    with open(YAML_PATH, "w") as f:
        f.write(yaml_text)


def setup_match_template_manager(num_cuda_streams: int) -> MatchTemplateManager:
    """Instantiate the manager object and prepare for template matching."""
    manager = MatchTemplateManager.from_yaml(YAML_PATH)
    # Override the num_cpus from YAML with the grid search parameter
    manager.computational_config.num_cpus = num_cuda_streams
    return manager


def benchmark_match_template_single_run(
    mt_manager: MatchTemplateManager,
    orientation_batch_size: int,
    num_cuda_streams: int,
    config_id: int,
    run_id: int,
    device_info: dict,
    csv_path: str | None = None,
    monitor_interval: float = 1.0,
) -> dict[str, float]:
    """Run a single benchmark and return timing statistics."""
    monitor = None
    try:
        torch.cuda.synchronize()

        # Reset PyTorch memory stats
        torch.cuda.reset_peak_memory_stats()

        # Get GPU IDs from manager's computational config
        gpu_ids = mt_manager.computational_config.gpu_ids

        # Handle case for single GPU ID as int
        if isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]

        # Handle special case for "all" GPUs
        if isinstance(gpu_ids, str) and gpu_ids.lower() == "all":
            gpu_ids = list(range(torch.cuda.device_count()))

        # Prepare config metadata for CSV
        config_metadata = {
            "config_id": config_id,
            "run_id": run_id,
            "orientation_batch_size": orientation_batch_size,
            "num_cuda_streams": num_cuda_streams,
        }

        # Initialize GPU monitor with CSV output (supports multi-GPU)
        monitor = GPUMonitor(
            device_ids=gpu_ids,
            sample_interval=monitor_interval,
            csv_path=csv_path,
            config_metadata=config_metadata,
        )
        monitor.start()

        ####################################################
        ### 1. Profile the make core backend kwargs time ###
        ####################################################

        monitor.set_phase("setup")
        start_time = time.perf_counter()

        core_kwargs = mt_manager.make_backend_core_function_kwargs()

        setup_time = time.perf_counter() - start_time

        ############################################
        ### 2. Profile the backend function call ###
        ############################################

        monitor.set_phase("execution")
        start_time = time.perf_counter()

        result = core_match_template(
            **core_kwargs,
            orientation_batch_size=orientation_batch_size,
            num_cuda_streams=num_cuda_streams,
        )
        total_projections = result["total_projections"]  # number of CCGs calculated, N

        torch.cuda.synchronize()
        execution_time = time.perf_counter() - start_time

        ##################################################
        ### 3. Use extremely small search to estimate  ###
        ###    constant core_match_template setup cost ###
        ##################################################

        monitor.set_phase("adjustment")
        core_kwargs["euler_angles"] = torch.rand(size=(100, 3)) * 180
        start_time = time.perf_counter()

        result = core_match_template(
            **core_kwargs,
            orientation_batch_size=orientation_batch_size,
            num_cuda_streams=num_cuda_streams,
        )
        adjustment_projections = result[
            "total_projections"
        ]  # number of CCGs calculated, n

        adjustment_time = time.perf_counter() - start_time

        # Stop GPU monitoring and get stats
        gpu_stats = monitor.stop()

        # Get PyTorch memory stats
        torch_max_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        torch_current_memory_mb = torch.cuda.memory_allocated() / (1024**2)

        # Doing the adjustment computations
        N = total_projections
        n = adjustment_projections
        T = execution_time
        t = adjustment_time
        throughput = (N - n) / (T - t)
        core_deadtime = T - N * (T - t) / (N - n)

        return {
            "success": True,
            "setup_time": setup_time,
            "execution_time": execution_time,
            "total_projections": total_projections,
            "adjustment_time": adjustment_time,
            "adjustment_projections": adjustment_projections,
            "throughput": throughput,
            "core_deadtime": core_deadtime,
            "torch_max_memory_mb": torch_max_memory_mb,
            "torch_current_memory_mb": torch_current_memory_mb,
            "gpu_monitoring": gpu_stats,
            "error": None,
        }

    except torch.cuda.OutOfMemoryError as e:
        # Handle OOM specifically
        if monitor:
            monitor.mark_error("OutOfMemoryError")
            monitor.stop()
        torch.cuda.empty_cache()
        return {
            "success": False,
            "error": "OutOfMemoryError",
            "error_message": str(e),
            "setup_time": None,
            "execution_time": None,
            "total_projections": None,
            "adjustment_time": None,
            "adjustment_projections": None,
            "throughput": None,
            "core_deadtime": None,
            "torch_max_memory_mb": None,
            "torch_current_memory_mb": None,
            "gpu_monitoring": {},
        }

    except Exception as e:
        # Handle any other errors
        if monitor:
            monitor.mark_error(type(e).__name__)
            monitor.stop()
        torch.cuda.empty_cache()
        return {
            "success": False,
            "error": type(e).__name__,
            "error_message": str(e),
            "setup_time": None,
            "execution_time": None,
            "total_projections": None,
            "adjustment_time": None,
            "adjustment_projections": None,
            "throughput": None,
            "core_deadtime": None,
            "torch_max_memory_mb": None,
            "torch_current_memory_mb": None,
            "gpu_monitoring": {},
        }


def run_benchmark_for_config(
    orientation_batch_size: int,
    num_cuda_streams: int,
    config_id: int,
    num_runs: int,
    device_info: dict,
    csv_output_dir: Path,
    monitor_interval: float = 1.0,
) -> dict[str, Any]:
    """Run multiple benchmark iterations for a specific configuration."""
    results = []
    successful_runs = 0
    failed_runs = 0

    for run_idx in range(num_runs):
        print(
            f"    Run {run_idx + 1}/{num_runs} "
            f"(batch={orientation_batch_size}, streams={num_cuda_streams})..."
        )

        # Create CSV filename for this run
        csv_filename = f"config{config_id:03d}_run{run_idx + 1}.csv"
        csv_path = csv_output_dir / csv_filename

        mt_manager = setup_match_template_manager(num_cuda_streams)
        result = benchmark_match_template_single_run(
            mt_manager,
            orientation_batch_size,
            num_cuda_streams,
            config_id,
            run_idx + 1,
            device_info,
            str(csv_path),
            monitor_interval,
        )
        results.append(result)

        if result["success"]:
            successful_runs += 1
            print(f"      ✓ CSV saved to: {csv_filename}")
        else:
            failed_runs += 1
            msg = f"      ⚠️  Run failed: {result['error']}"
            msg += f" - {result['error_message']}"
            print(msg)

        torch.cuda.empty_cache()

    # Check if all runs failed
    if successful_runs == 0:
        print("    ❌ All runs failed for this configuration")
        return {
            "orientation_batch_size": orientation_batch_size,
            "num_cuda_streams": num_cuda_streams,
            "success": False,
            "successful_runs": 0,
            "failed_runs": failed_runs,
            "total_projections": None,
            "mean_setup_time": None,
            "std_setup_time": None,
            "mean_execution_time": None,
            "std_execution_time": None,
            "mean_throughput": None,
            "std_throughput": None,
            "mean_core_deadtime": None,
            "std_core_deadtime": None,
            "max_memory_used_mb": None,
            "mean_gpu_utilization_percent": None,
            "max_gpu_utilization_percent": None,
            "all_results": results,
        }

    # Calculate statistics only from successful runs
    successful_results = [r for r in results if r["success"]]

    execution_times = np.array([r["execution_time"] for r in successful_results])
    setup_times = np.array([r["setup_time"] for r in successful_results])
    throughputs = np.array([r["throughput"] for r in successful_results])
    core_deadtimes = np.array([r["core_deadtime"] for r in successful_results])
    total_projections_list = [r["total_projections"] for r in successful_results]

    # Aggregate GPU monitoring stats
    max_memory_used = max(
        r["gpu_monitoring"].get("memory_used_mb", {}).get("max", 0)
        for r in successful_results
        if r.get("gpu_monitoring")
    )
    mean_gpu_util = np.mean(
        [
            r["gpu_monitoring"].get("gpu_utilization_percent", {}).get("mean", 0)
            for r in successful_results
            if r.get("gpu_monitoring")
        ]
    )
    max_gpu_util = max(
        r["gpu_monitoring"].get("gpu_utilization_percent", {}).get("max", 0)
        for r in successful_results
        if r.get("gpu_monitoring")
    )

    mst, sst = setup_times.mean(), setup_times.std()
    mxt, sxt = execution_times.mean(), execution_times.std()
    mtt, stt = throughputs.mean(), throughputs.std()
    mct, sct = core_deadtimes.mean(), core_deadtimes.std()

    return {
        "orientation_batch_size": orientation_batch_size,
        "num_cuda_streams": num_cuda_streams,
        "success": True,
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "total_projections": total_projections_list,
        "mean_setup_time": mst,
        "std_setup_time": sst,
        "mean_execution_time": mxt,
        "std_execution_time": sxt,
        "mean_throughput": mtt,
        "std_throughput": stt,
        "mean_core_deadtime": mct,
        "std_core_deadtime": sct,
        "max_memory_used_mb": max_memory_used,
        "mean_gpu_utilization_percent": mean_gpu_util,
        "max_gpu_utilization_percent": max_gpu_util,
        "all_results": results,
    }


def save_benchmark_results(result: dict, output_file: str) -> None:
    """Save benchmark results to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nBenchmark results saved to: {output_file}")


def save_incremental_results(
    grid_results: list[dict],
    device_info: dict[str, Any],
    output_file: str,
    best_config: dict | None = None,
) -> None:
    """Save incremental benchmark results after each configuration."""
    result = {
        "device_name": device_info["device_name"],
        "device_sm_arch": device_info["device_sm_arch"],
        "device_memory_gb": device_info["device_memory_gb"],
        "grid_results": grid_results,
        "successful_configs": sum(1 for r in grid_results if r["success"]),
        "failed_configs": sum(1 for r in grid_results if not r["success"]),
        "optimal_config": best_config,
    }

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)


def run_grid_search(
    orientation_batch_sizes: list[int],
    num_cuda_streams_list: list[int],
    num_runs: int,
    monitor_interval: float = 1.0,
    output_file: str = "benchmark_grid_results.json",
    download_data: bool = True,
) -> dict[str, Any]:
    """Run benchmark across a grid of parameters.

    Parameters
    ----------
    orientation_batch_sizes : list[int]
        List of orientation batch sizes to test
    num_cuda_streams_list : list[int]
        List of CUDA stream counts to test
    num_runs : int
        Number of benchmark runs per configuration
    monitor_interval : float
        GPU monitoring sample interval in seconds
    output_file : str
        Output file for benchmark results
    download_data : bool
        Whether to download benchmark data from Zenodo. If False, assumes data
        has already been downloaded.
    """
    # Download example data to use for benchmarking
    if download_data:
        print("Downloading benchmarking data...")
        download_comparison_data()
        print("Done!")
    else:
        print("Skipping data download (using existing files)...")

    # Create CSV output directory
    output_path = Path(output_file)
    csv_output_dir = output_path.parent / f"{output_path.stem}_csv"
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"CSV files will be saved to: {csv_output_dir}")

    # Get CUDA device properties
    device = torch.cuda.get_device_properties(0)
    device_name = str(device.name)
    sm_architecture = device.major * 10 + device.minor
    device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU device SM architecture: {sm_architecture}")
    print("Running benchmark on device:", device_name)
    print(f"GPU device has {device_memory:.2f} GB of memory")
    print()

    device_info = {
        "device_name": device_name,
        "device_sm_arch": sm_architecture,
        "device_memory_gb": device_memory,
    }

    grid_results = []
    total_configs = len(orientation_batch_sizes) * len(num_cuda_streams_list)
    config_idx = 0
    successful_configs = 0
    failed_configs = 0

    for batch_size in orientation_batch_sizes:
        for num_streams in num_cuda_streams_list:
            config_idx += 1
            print(
                f"\nConfiguration {config_idx}/{total_configs}: "
                f"batch_size={batch_size}, num_streams={num_streams}"
            )
            print("-" * 70)

            config_result = run_benchmark_for_config(
                batch_size,
                num_streams,
                config_idx,
                num_runs,
                device_info,
                csv_output_dir,
                monitor_interval,
            )
            grid_results.append(config_result)

            if config_result["success"]:
                successful_configs += 1

                # Build success message
                msg = f"    Mean throughput: {config_result['mean_throughput']:.3f}"
                msg += f" ± {config_result['std_throughput']:.3f} corr/sec\n"
                msg += "    Mean execution time: "
                msg += f"{config_result['mean_execution_time']:.3f}"
                msg += f" ± {config_result['std_execution_time']:.3f} seconds\n"
                msg += "    Max memory used: "
                msg += f"{config_result['max_memory_used_mb']:.1f} MB\n"
                msg += "    Mean GPU utilization: "
                msg += f"{config_result['mean_gpu_utilization_percent']:.1f}%"

                if config_result["failed_runs"] > 0:
                    msg += f"\n    ⚠️  {config_result['failed_runs']}"
                    msg += f"/{num_runs} runs failed"

                print(msg)
            else:
                failed_configs += 1
                print("    ❌ Configuration failed - all runs unsuccessful")

            # Save incremental results after each configuration
            successful_grid_results = [r for r in grid_results if r["success"]]
            current_best = None
            if successful_grid_results:
                current_best_config = max(
                    successful_grid_results, key=lambda x: x["mean_throughput"]
                )
                current_best = {
                    "orientation_batch_size": current_best_config[
                        "orientation_batch_size"
                    ],
                    "num_cuda_streams": current_best_config["num_cuda_streams"],
                    "mean_throughput": current_best_config["mean_throughput"],
                    "mean_execution_time": current_best_config["mean_execution_time"],
                    "max_memory_used_mb": current_best_config["max_memory_used_mb"],
                    "mean_gpu_utilization_percent": current_best_config[
                        "mean_gpu_utilization_percent"
                    ],
                    "successful_runs": current_best_config["successful_runs"],
                }

            save_incremental_results(
                grid_results, device_info, output_file, current_best
            )
            print(f"    💾 Incremental results saved to {output_file}")

    # Find optimal configuration based on mean throughput from successful configs
    successful_grid_results = [r for r in grid_results if r["success"]]

    if not successful_grid_results:
        print("\n" + "=" * 70)
        print("❌ GRID SEARCH FAILED - No successful configurations")
        print("=" * 70)
        return {
            "device_name": device_name,
            "device_sm_arch": sm_architecture,
            "device_memory_gb": device_memory,
            "csv_output_dir": str(csv_output_dir),
            "grid_results": grid_results,
            "successful_configs": 0,
            "failed_configs": failed_configs,
            "optimal_config": None,
        }

    best_config = max(successful_grid_results, key=lambda x: x["mean_throughput"])

    print("\n" + "=" * 70)
    print("GRID SEARCH SUMMARY")
    print("=" * 70)
    print(f"\nSuccessful configurations: {successful_configs}/{total_configs}")
    print(f"Failed configurations: {failed_configs}/{total_configs}")
    print(f"\nCSV files saved to: {csv_output_dir}")
    print("\nOptimal Configuration:")
    print(f"  Orientation batch size: {best_config['orientation_batch_size']}")
    print(f"  Number of CUDA streams: {best_config['num_cuda_streams']}")
    print(
        f"  Mean throughput: {best_config['mean_throughput']:.3f} ± "
        f"{best_config['std_throughput']:.3f} corr/sec"
    )
    print(
        f"  Mean execution time: {best_config['mean_execution_time']:.3f} ± "
        f"{best_config['std_execution_time']:.3f} seconds"
    )
    print(f"  Max memory used: {best_config['max_memory_used_mb']:.1f} MB")
    print(f"  Mean GPU utilization: {best_config['mean_gpu_utilization_percent']:.1f}%")
    print(f"  Successful runs: {best_config['successful_runs']}/{num_runs}")
    print("\nAll Configurations:")
    print("-" * 90)
    print(
        f"{'Batch':<8} {'Streams':<8} {'Status':<10} {'Throughput':<20} "
        f"{'Exec Time (s)':<18} {'GPU Util %':<12} {'Max Mem (MB)':<12}"
    )
    print("-" * 90)

    # Sort with proper handling of None values
    def sort_key(x):
        """Sort key that handles None values by putting them at the end."""
        throughput = x.get("mean_throughput")
        if throughput is None:
            return -float("inf")  # Put failed configs at the end
        return throughput

    for result in sorted(grid_results, key=sort_key, reverse=True):
        if result["success"]:
            status = f"✓ ({result['successful_runs']}/"
            status += f"{result['successful_runs'] + result['failed_runs']})"
            print(
                f"{result['orientation_batch_size']:<8} "
                f"{result['num_cuda_streams']:<8} "
                f"{status:<10} "
                f"{result['mean_throughput']:>6.1f} ± {result['std_throughput']:<6.1f} "
                f"{result['mean_execution_time']:>6.2f} ± "
                f"{result['std_execution_time']:<6.2f} "
                f"{result['mean_gpu_utilization_percent']:>8.1f} "
                f"{result['max_memory_used_mb']:>10.1f}"
            )
        else:
            status = "✗ FAILED"
            print(
                f"{result['orientation_batch_size']:<8} "
                f"{result['num_cuda_streams']:<8} "
                f"{status:<10} "
                f"{'N/A':<27} {'N/A':<18} {'N/A':<12} {'N/A':<12}"
            )

    return {
        "device_name": device_name,
        "device_sm_arch": sm_architecture,
        "device_memory_gb": device_memory,
        "csv_output_dir": str(csv_output_dir),
        "grid_results": grid_results,
        "successful_configs": successful_configs,
        "failed_configs": failed_configs,
        "optimal_config": {
            "orientation_batch_size": best_config["orientation_batch_size"],
            "num_cuda_streams": best_config["num_cuda_streams"],
            "mean_throughput": best_config["mean_throughput"],
            "mean_execution_time": best_config["mean_execution_time"],
            "max_memory_used_mb": best_config["max_memory_used_mb"],
            "mean_gpu_utilization_percent": best_config["mean_gpu_utilization_percent"],
            "successful_runs": best_config["successful_runs"],
        },
    }


@click.command()
@click.option(
    "--batch-sizes",
    default="10,20,40",
    type=str,
    help="Comma-separated list of orientation batch sizes to test (default: 10,20,40)",
)
@click.option(
    "--cuda-streams",
    default="2,4,8",
    type=str,
    help="Comma-separated list of CUDA stream counts to test (default: 2,4,8)",
)
@click.option(
    "--num-runs",
    default=3,
    type=int,
    help="Number of benchmark runs per configuration (default: 3)",
)
@click.option(
    "--monitor-interval",
    default=1.0,
    type=float,
    help="GPU monitoring sample interval in seconds (default: 1.0)",
)
@click.option(
    "--output-file",
    default="benchmark_grid_results.json",
    type=str,
    help="Output file for benchmark results (default: benchmark_grid_results.json)",
)
@click.option(
    "--download-data/--no-download-data",
    default=True,
    help="Whether to download benchmark data from Zenodo (default: --download-data). "
    "Use --no-download-data if you want to use existing files on disk.",
)
def main(
    batch_sizes: str,
    cuda_streams: str,
    num_runs: int,
    monitor_interval: float,
    output_file: str,
    download_data: bool,
):
    """Main benchmarking function with grid search over parameters."""
    if not torch.cuda.is_available():
        print("CUDA not available, exiting...")
        return

    # Parse comma-separated values
    orientation_batch_sizes = [int(x.strip()) for x in batch_sizes.split(",")]
    num_cuda_streams_list = [int(x.strip()) for x in cuda_streams.split(",")]

    print("Grid Search Benchmark Configuration:")
    print(f"  Orientation batch sizes: {orientation_batch_sizes}")
    print(f"  CUDA stream counts: {num_cuda_streams_list}")
    print(f"  Number of runs per config: {num_runs}")
    print(f"  GPU monitoring interval: {monitor_interval}s")
    print(f"  Output file: {output_file}")
    print(f"  Download data: {download_data}")
    print(
        f"  Total configurations: "
        f"{len(orientation_batch_sizes) * len(num_cuda_streams_list)}"
    )
    print()

    result = run_grid_search(
        orientation_batch_sizes,
        num_cuda_streams_list,
        num_runs,
        monitor_interval,
        output_file,
        download_data,
    )
    save_benchmark_results(result, output_file)


if __name__ == "__main__":
    main()
