"""Run the match_template program in a distributed, multi-node environment.

NOTE: This script needs to be launched using `torchrun` and within a distributed
environment where multiple nodes can communicate with eachother. See the online
documentation for more information and example scripts for running distributed multi
node match_template.

NOTE: Leopard-EM currently assumes a homogeneous cluster where all nodes have the same
number of GPUs, GPU type, etc.
"""

# ntasks-per-node should be set to the number of GPUs per node

import time

import torch.distributed as dist

from leopard_em.pydantic_models.managers import MatchTemplateManager

#######################################
### Editable parameters for program ###
#######################################

# NOTE: You can also use `click` to pass argument to this script from command line
YAML_CONFIG_PATH = "some/path/to/config.yaml"
DATAFRAME_OUTPUT_PATH = "some/path/to/match_template_results.csv"
ORIENTATION_BATCH_SIZE = 8


def main() -> None:
    """Main function for the distributed match_template program."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print(f"[{start_time_str}] Initialized, rank {rank} of {world_size}.")

    mt_manager = MatchTemplateManager.from_yaml(
        YAML_CONFIG_PATH, preload_mrc_files=True
    )
    mt_manager.run_match_template_distributed(
        orientation_batch_size=ORIENTATION_BATCH_SIZE,
        do_result_export=(rank == 0),  # Only save results from rank 0
    )
