#!/bin/bash

#SBATCH --job-name=asp_paper
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --output=pretraining/jobs/slurm/%x-%j.out
#SBATCH --error=pretraining/jobs/slurm/%x-%j.err

# Submit from the repo root: sbatch pretraining/jobs/asp_paper.sh [config]
# Adjust the conda.sh path below for your own cluster -- this one is a placeholder.
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate ai4brainmri

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Still DDP-based (untouched -- see pretraining/README.md), so still needs torchrun even for
# a single GPU.
torchrun --nproc_per_node=1 --master_port=$((12000 + RANDOM % 1000)) -m pretraining.seg_mae_combined_train \
    --cfg "${1:-pretraining/configs/asp_paper.yaml}" --local_rank 0
