# Distributed (multi-node) template matching with Leopard-EM

Processing data with 2DTM is a computationally intensive process, but the current 2DTM algorithm is an [embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel) problem.
Leopard-EM natively supports multi-GPU systems (see [An Introductory 2DTM tutorial](match_template_intro.md)) for parallelizing the 2DTM search.
In this tutorial, we discuss how Leopard-EM can further parallelize 2DTM by using multi-node clusters by running **distributed match template**.

This tutorial assumes the following:

- You have access to a multi-node cluster with a GPU partition,
- The cluster is using [SULRM](https://slurm.schedmd.com/documentation.html) as a job scheduler and workload manager,
- Familiarity with the default `match_template` program, and
- Base knowledge on scheduling jobs with SLURM.

!!! Warning "An imperative discussion on _when_ distributed match template is appropriate"

    For the sake of discussion, let's assume the cluster partition has the following resources:

    - 16 total nodes (with sufficient CPU and memory)
    - 4x GPUs per node, all same type
    - Total of 64 GPUs on the partition

    Let's also assume the objectives, in order, for the project are

    1. Maximize GPU utilization, and
    2. Process data as efficiently as possible.

    Using these assumptions, we will examine different scenarios and discuss if multi-node, distributed match template is appropriate.

    **Scenario 1: Large number of micrographs to process**

    If you have a large number of micrographs to process, enough to backlog the queue on your cluster with 2DTM jobs, then running distributed match template _does not make sense_.
    Assuming perfect strong scaling, running a single distributed match template across all 64 GPUs will finish 64x faster than a single-GPU job.
    However, you'd get the same micrograph processing throughput if each micrograph is processed on a single node (4x GPUs) or even on a single GPU since 16 or 64 jobs could be running concurrently, respectively.

    Requesting a job that takes _all_ the nodes in the partition is also likely to get backlogged in the job queue.
    The SLURM scheduler will try to queue up as many jobs that can run concurrently as possible to maximize cluster utilization, and jobs requesting fewer resources generally get queued faster.
    Your colleagues will also appreciate you not consuming the whole partition for your jobs.

    **Scenario 2: Few number of micrographs, and low priority**

    If you have only a few micrographs, say three, to process and it takes 16h/micrograph/gpu to run `match_template`, then the following distribution for time-spans for complete processing are possible:

    1. Single-GPU, sequential - Each micrograph is processed sequentially on a single-GPU: 16 x 3 = 48 hours.
    2. Single-GPU, parallel - Three jobs, each single-GPU, running independently and in parallel: 16 x (3 / 3) = 16 hours.
    <!-- 3. Single-node (4x GPUs), sequential - Entire node allocated for one job that processes all three micrographs: (16 / 4) * 3 = 12 hours. -->
    3. Single-node (4x GPUs), parallel - Entire node allocated, three different jobs that each process one micrograph: (16 / 4) * (3 / 3) = 4 hours.
    4. 16-node (64x GPUs), sequential - Whole partition allocated to one job which processed all three micrographs: (16 / 64) * 3 = 0.75 hours.

    Assuming we're happy to get the template matching results at some point it makes sense to request more jobs each using fewer resources.
    Running distributed match template _does not make sense_ under this scenario.
    Again, job queue time must be taken into consideration, and the 16-node job may be sitting in the queue longer than it would have taken for even case (1) to be allocated and completed.

    **Scenario 3: Few number of micrographs, and high priority**

    Using the same setup as **Scenario 2**, if we _really_ need to get template matching results right now (and there's some mechanism for placing our resource-hungry job at the tippy-top of the queue), then it may make sense to run distributed match template.

    **Scenario 4: Obtaining near real-time 2DTM results**

    Let's say you're lucky enough to have exclusive access to this partition, and the processing time is 4h/micrograph/gpu (fair assumption for some of the newest GPUs).
    If you really need real-time 2DTM results for your project, then allocating the entire partition would obtain a throughput of 4 / 64 = 1/16 hour = 3.75 minutes to process each micrograph.
    Under this scenario, it makes sense to run the distributed match template program.

## Data pre-requisites

Here, we have the same pre-requisites as the [intro tutorial pre-requisites](match_template_intro.md#data-and-computation-pre-requisites).
Note that this includes a fully formed `MatchTemplateManager` configuration file.

Distributed computation should work out-of-the-box for PyTorch, so there are no other packages to download/install.

## Setting up Leopard-EM script for distributed computation.

Distributed match template requires a few extra steps to setup inter-node communication before launching the backend program.
These are handled automatically in the Python script below which is also included on the [Leopard-EM github programs page](https://github.com/Lucaslab-Berkeley/Leopard-EM/blob/main/programs/match_template/run_distributed_match_template.py).

The script is intended to be run as `python --config FILE --output FILE --batch_size INTEGER` where the three command line arguments are used to define the match template configuration YAML file, the output csv file, and the orientation batch size, respectively.

??? info "`run_distributed_match_template.py`"

    ```python
    """Run the match_template program in a distributed, multi-node environment.

    NOTE: This script needs to be launched using `torchrun` and within a distributed
    environment where multiple nodes can communicate with each other. See the online
    documentation and example scripts for more information on running distributed multi
    node match_template.

    NOTE: The 'gpu_ids' field in the YAML config is ignored when running in distributed
    mode. Each process is assigned to a single GPU based on its local rank.
    """

    import os
    import time

    import click
    import torch.distributed as dist

    from leopard_em.pydantic_models.managers import MatchTemplateManager

    #######################################
    ### Editable parameters for program ###
    #######################################

    # NOTE: You can also use `click` to pass argument to this script from command line
    YAML_CONFIG_PATH = "/path/to/config.yaml"
    DATAFRAME_OUTPUT_PATH = "out.csv"
    ORIENTATION_BATCH_SIZE = 20


    def initialize_distributed() -> tuple[int, int, int]:
        """Initialize the distributed environment.

        Returns
        -------
            (world_size, global_rank, local_rank)
        """
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = os.environ.get("LOCAL_RANK", None)

        # Raise error if LOCAL_RANK is not set. This *should* be handled by torchrun, but...
        # It is up to the user to rectify this issue on their system.
        if local_rank is None:
            raise RuntimeError("LOCAL_RANK environment variable unset!.")

        local_rank = int(local_rank)

        return world_size, rank, local_rank


    @click.command()
    @click.option(
        "--config",
        "-c",
        type=click.Path(exists=True, dir_okay=False, path_type=str),
        default=YAML_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    @click.option(
        "--output",
        "-o",
        type=click.Path(dir_okay=False, path_type=str),
        default=DATAFRAME_OUTPUT_PATH,
        help="Path to save the output dataframe CSV.",
    )
    @click.option(
        "--batch_size",
        "-b",
        type=int,
        default=ORIENTATION_BATCH_SIZE,
        help="Number of orientations to process in a single batch.",
    )
    def main(config: str, output: str, batch_size: int) -> None:
        """Main function for the distributed match_template program.

        Each process is associated with a single GPU, and we front-load the distributed
        initialization and GPU assignment in this script. This allows both the manager
        object and the backend match_template code to remain relatively simple.
        """
        world_size, rank, local_rank = initialize_distributed()
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        print(
            f"{time_str} RANK={rank}: Initialized {world_size} processes "
            f"(local_rank={local_rank})."
        )

        # Do not pre-load mrc files, unless zeroth rank. Data will be broadcast later.
        mt_manager = MatchTemplateManager.from_yaml(
            config, preload_mrc_files=bool(rank == 0)
        )
        mt_manager.run_match_template_distributed(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            orientation_batch_size=batch_size,
            do_result_export=(rank == 0),  # Only save results from rank 0
        )

        # Only do the df export on rank 0
        if rank == 0:
            df = mt_manager.results_to_dataframe()
            df.to_csv(output, index=True)

        # Close the distributed process group
        dist.destroy_process_group()

        print("Done!")


    if __name__ == "__main__":
        start_time = time.time()
        main()
        print(f"Total time: {time.time() - start_time:.1f} seconds.")

    ```

## SLURM script for launching distributed match template.

The SLURM batch script for distributed match template is relatively simple.
The example script can be found on the [Leopard-EM github programs page](https://github.com/Lucaslab-Berkeley/Leopard-EM/blob/main/programs/match_template/distributed_match_template.slurm), but it's contents are also listed below.
It does the following

1. Requests the number of nodes, node configurations, and other SLURM options,
2. Defines a setup command to load the necessary modules / environments to run Leopard-EM,
3. Defines the program to run (distributed match template python script), and
4. Wraps this into a `surn` / `torchrun` launch which actually runs the program across all the nodes.

**There are a portions of the script which need adapted to your specific computing environment.**

### SLURM header

The header defining the allocations for the job _must_ be edited to match how you would launch a job on your cluster

```bash
#SBATCH --job-name=distributed-match-template-%j
#SBATCH --nodes=4               # EDIT: how many nodes allocated
#SBATCH --ntasks-per-node=1     # crucial! - only 1 task per node
#SBATCH --cpus-per-task=8       # EDIT: match number of GPUs per node
#SBATCH --gres=gpu:8            # EDIT: number & type of GPUs per node
#SBATCH --time=2:00:00          # EDIT: desired runtime (hh:mm:ss)
#SBATCH --partition=<part>      # EDIT: your partition
#SBATCH --qos=<qos>             # EDIT: your qos
#SBATCH --account=<acct>        # EDIT: your account name
#SBATCH --output=%x-%j.out
```

### Setup, number of GPUs, and program variables

The other portions which you will need to modify are the setup command, the number of GPUs per node, and the path to the Python program to run.

```bash
# EDIT: Necessary commands to set up your environment *before*
#       running the program (e.g. loading modules, conda envs, etc.)
SETUP="ml anaconda3 && \
    source ~/.bashrc && \
    conda activate leopard-em-dev && \
"

# EDIT: How many GPUs per node (should match what was requested in --gres)
GPUS_PER_NODE=8

# EDIT: Define your program an its argument
PROGRAM="programs/match_template/run_distributed_match_template.py"
# OR if CLI arguments are required:
# PROGRAM="programs/match_template/run_distributed_match_template.py --arg1 val1 --arg2 val2"
```

The rest of the script should work as-is, but there might be particular constraints on your cluster.
If you're getting errors on the launch, check with your SysAdmin.

### Entire script

??? "`distributed_match_template.slurm`"

    ```bash
    #!/bin/bash

    # ***
    # *** This is an example SLURM job script for launching a distributed
    # *** match_template job using torchrun over multiple nodes in a cluster.
    # *** There are many points at which you will need to modify the script
    # *** to fit onto your specific cluster environment.
    # ***
    # *** NOTE: If you are just trying to saturate GPU resources and have
    # ***       enough micrographs to process (and no time pressure for
    # ***       results), then it's advisable to just launch multiple
    # ***       single-node jobs instead of distributed jobs. 
    # ***

    #SBATCH --job-name=distributed-match-template-%j
    #SBATCH --nodes=4               # EDIT: how many nodes allocated
    #SBATCH --ntasks-per-node=1     # crucial! - only 1 task per node
    #SBATCH --cpus-per-task=8       # EDIT: match number of GPUs per node
    #SBATCH --gres=gpu:8            # EDIT: number & type of GPUs per node
    #SBATCH --time=2:00:00          # EDIT: desired runtime (hh:mm:ss)
    #SBATCH --partition=<part>      # EDIT: your partition
    #SBATCH --qos=<qos>             # EDIT: your qos
    #SBATCH --account=<acct>        # EDIT: your account name
    #SBATCH --output=%x-%j.out


    echo "START TIME: $(date)"


    # EDIT: Necessary commands to set up your environment *before*
    #       running the program (e.g. loading modules, conda envs, etc.)
    SETUP="ml anaconda3 && \
        source ~/.bashrc && \
        conda activate leopard-em-dev && \
    "

    # EDIT: How many GPUs per node (should match what was requested in --gres)
    GPUS_PER_NODE=8

    # EDIT: Define your program an its argument
    PROGRAM="programs/match_template/run_distributed_match_template.py"
    # OR if CLI arguments are required:
    # PROGRAM="programs/match_template/run_distributed_match_template.py --arg1 val1 --arg2 val2"



    # Verbose output for debugging purposes (can comment out if not needed)
    set -x
    srun hostname  # each allocated node prints the hostname

    # Some parameters to extract necessary information from SLURM
    allocated_nodes=$(scontrol show hostname $SLURM_JOB_NODELIST)
    nodes=${allocated_nodes//$'\n'/ } # replace newlines with spaces
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    echo Head Node: $head_node
    echo Node List: $nodes
    export LOGLEVEL=INFO

    # The command for torchrun to launch the distributed job
    # NOTE: --rdzv_id requires an open port, so using a random number.
    #       But there may be restrictions on allowed ports on your cluster...
    LAUNCHER="torchrun \
        --nproc_per_node=$GPUS_PER_NODE \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --rdzv_id=$RANDOM \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$head_node:29500 \
        "
    CMD="$SETUP $LAUNCHER $PROGRAM"


    echo "Running command:"
    echo $CMD
    echo "-------------------"
    srun /bin/bash -c "$CMD"

    echo "END TIME: $(date)"
    ```

## Queuing the distributed match template job

Placing the job into the queue is as easy as running `sbatch distributed_match_template.slurm` or whatever name(s) you may have assigned to the scripts.
