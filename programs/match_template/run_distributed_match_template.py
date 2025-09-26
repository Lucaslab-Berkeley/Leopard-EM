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
