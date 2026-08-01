#!/bin/bash

#SBATCH --job-name=mae_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu
#SBATCH --time=240:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --output=pretraining/jobs/slurm/%x-%j.out
#SBATCH --error=pretraining/jobs/slurm/%x-%j.err

# Submit from the repo root: sbatch pretraining/jobs/mae_train.sh
# Adjust the conda.sh path below for your own cluster -- this one is a placeholder.
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate ai4brainmri

# debugging flags (optional)
export PYTHONFAULTHANDLER=1

# Single-process, no DDP/torchrun -- unlike seg_mae_combined_train.py, there's no original
# multi-GPU implementation this needs to stay faithful to (see pretraining/README.md).
python -m pretraining.mae_train --cfg pretraining/configs/mae_train.yaml
