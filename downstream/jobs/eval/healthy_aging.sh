#!/bin/bash

#SBATCH --job-name=healthy_aging
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=100G
#SBATCH --gres=gpu
#SBATCH --time=3-00:00:0
#SBATCH --output=downstream/jobs/eval/slurm/%x-%j.out
#SBATCH --error=downstream/jobs/eval/slurm/%x-%j.err

# Submit from the repo root: sbatch downstream/jobs/eval/healthy_aging.sh
# Adjust the conda.sh path below for your own cluster -- this one is a placeholder.
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh
conda activate ai4brainmri

python -m downstream.evaluate --cfg downstream/configs/eval/healthy_aging.yaml
