"""Benchmarking script for core_match_template performance.

Benchmark is using a 4096 x 4096 pixel image with a 512 x 512 x 512 template. Smaller
size images will make a bigger performance impact than reducing the template volume.
This script can be modified to benchmark images of other sizes (e.g. K3 images).

NOTE: This benchmark can take up to 10 minutes given the moderate sized search space and
GPU requirements.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch

from leopard_em.backend.core_match_template import core_match_template
from leopard_em.pydantic_models.managers import MatchTemplateManager

DOWNLOAD_DIR = (Path(__file__).parent / "tmp").resolve()
YAML_PATH = (
    Path(DOWNLOAD_DIR) / "test_match_template_xenon_216_000_0.0_DWS_config.yaml"
).resolve()
ZENODO_URL = "https://zenodo.org/records/17069838"


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


def setup_match_template_manager() -> MatchTemplateManager:
    """Instantiate the manager object and prepare for template matching."""
    return MatchTemplateManager.from_yaml(YAML_PATH)


def benchmark_match_template_single_run(
    mt_manager: MatchTemplateManager, orientation_batch_size: int
) -> dict[str, float]:
    """Run a single benchmark and return timing statistics."""
    torch.cuda.synchronize()

    ####################################################
    ### 1. Profile the make core backend kwargs time ###
    ####################################################

    start_time = time.perf_counter()

    core_kwargs = mt_manager.make_backend_core_function_kwargs()

    setup_time = time.perf_counter() - start_time

    ############################################
    ### 2. Profile the backend function call ###
    ############################################

    start_time = time.perf_counter()

    result = core_match_template(
        **core_kwargs,
        orientation_batch_size=orientation_batch_size,
        num_cuda_streams=mt_manager.computational_config.num_cpus,
    )
    total_projections = result["total_projections"]  # number of CCGs calculated, N

    execution_time = time.perf_counter() - start_time

    ##################################################
    ### 3. Use extremely smalls search to estimate ###
    ###    constantcore_match_template setup cost. ###
    ##################################################
    # This is using the timing model where the time -- T -- to compute N
    # cross-correlations is dependent on some device rate -- r -- and a constant setup
    # cost in terms of time. This setup time (or core-deadtime) is part of distributing
    # data to each device, compiling helper functions, and other overhead. What we can
    # measure is the total time:
    #
    # T_N = N/r + k
    #
    # Taking a large number of cross-correlations -- N -- (the performance profiled
    # above) and a smaller number of cross-correlations -- n -- we can back out
    # the constants along this curve
    #
    #             T_n = n/r + k
    # --> (T_N - T_n) = (N - n) / r
    # -->           r = (N - n) / (T_N - T_n)
    # -->           k = N * (T_N - T_n) / (N - n)

    core_kwargs["euler_angles"] = torch.rand(size=(100, 3)) * 180
    start_time = time.perf_counter()

    result = core_match_template(
        **core_kwargs,
        orientation_batch_size=orientation_batch_size,
        num_cuda_streams=mt_manager.computational_config.num_cpus,
    )
    adjustment_projections = result["total_projections"]  # number of CCGs calculated, n

    adjustment_time = time.perf_counter() - start_time

    # Doing the adjustment computations
    N = total_projections
    n = adjustment_projections
    T = execution_time
    t = adjustment_time
    throughput = (N - n) / (T - t)
    core_deadtime = T - N * (T - t) / (N - n)

    return {
        "setup_time": setup_time,
        "execution_time": execution_time,
        "total_projections": total_projections,
        "adjustment_time": adjustment_time,
        "adjustment_projections": adjustment_projections,
        "throughput": throughput,
        "core_deadtime": core_deadtime,
    }


def run_benchmark(orientation_batch_size: int, num_runs: int) -> dict[str, Any]:
    """Run multiple benchmark iterations and collect statistics."""
    # Download example data to use for benchmarking
    print("Downloading benchmarking data...")
    download_comparison_data()
    print("Done!")

    # Get CUDA device properties
    device = torch.cuda.get_device_properties(0)
    device_name = str(device.name)
    sm_architecture = device.major * 10 + device.minor
    device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU device SM architecture: {sm_architecture}")
    print("Running benchmark on device:", device_name)
    print(f"GPU device has {device_memory:.2f} GB  of memory")

    results = []

    for run_idx in range(num_runs):
        print(f"Running benchmark iteration {run_idx + 1}/{num_runs}...")

        mt_manager = setup_match_template_manager()
        result = benchmark_match_template_single_run(mt_manager, orientation_batch_size)
        results.append(result)

        print()
        print()
        print(f"    Setup time        : {result['setup_time']:.3f} seconds")
        print(f"    Execution time    : {result['execution_time']:.3f} seconds")
        print(f"    throughput (adj.) : {result['throughput']:.3f} corr/sec")
        print(f"    core dead-time    : {result['core_deadtime']:.3f} seconds")

        torch.cuda.empty_cache()

    execution_times = np.array([r["execution_time"] for r in results])
    setup_times = np.array([r["setup_time"] for r in results])
    throughputs = np.array([r["throughput"] for r in results])
    core_deadtimes = np.array([r["core_deadtime"] for r in results])
    total_projections_list = [r["total_projections"] for r in results]

    mst, sst = setup_times.mean(), setup_times.std()
    mxt, sxt = execution_times.mean(), execution_times.std()
    mtt, stt = throughputs.mean(), throughputs.std()
    mct, sct = core_deadtimes.mean(), core_deadtimes.std()

    print("\nSummary statistics over all runs (mean / std)")
    print("-------------------------------------------------------------")
    print(f"  Setup time        (seconds)  {mst:.3f} / {sst:.3f}")
    print(f"  Execution time    (seconds)  {mxt:.3f} / {sxt:.3f}")
    print(f"  Throughput (adj.) (corr/sec) {mtt:.3f} / {stt:.3f}")
    print(f"  Core dead-time    (seconds)  {mct:.3f} / {sct:.3f}")
    print("-------------------------------------------------------------")

    stats = {
        "total_projections": total_projections_list,
        "device_name": device_name,
        "device_sm_arch": sm_architecture,
        "device_memory_gb": device_memory,
        "mean_setup_time": mst,
        "mean_execution_time": mxt,
        "mean_throughput": mtt,
        "mean_core_deatime": mct,
        "all_results": results,
    }

    return stats


def save_benchmark_results(result: dict, output_file: str) -> None:
    """Save benchmark results to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nBenchmark results saved to: {output_file}")


@click.command()
@click.option(
    "--orientation-batch-size",
    default=20,
    type=int,
    help="Batch size for orientation processing (default: 20). Vary based on GPU specs",
)
@click.option(
    "--num-runs",
    default=3,
    type=int,
    help="Number of benchmark runs for statistical analysis (default: 3)",
)
@click.option(
    "--output-file",
    default="benchmark_results.json",
    type=str,
    help="Output file for benchmark results (default: benchmark_results.json)",
)
def main(orientation_batch_size: int, num_runs: int, output_file: str):
    """Main benchmarking function with Click CLI interface."""
    if not torch.cuda.is_available():
        print("CUDA not available exiting...")
        return

    print("Benchmark configuration:")
    print(f"  Orientation batch size: {orientation_batch_size}")
    print(f"  Number of runs: {num_runs}")
    print(f"  Output file: {output_file}")

    result = run_benchmark(orientation_batch_size, num_runs)
    # pprint(result)
    save_benchmark_results(result, output_file)


if __name__ == "__main__":
    main()
