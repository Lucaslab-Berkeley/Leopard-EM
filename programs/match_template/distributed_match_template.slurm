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