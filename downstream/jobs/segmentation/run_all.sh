#!/bin/bash
#SBATCH --job-name=segmentation
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu
#SBATCH --time=3-00:00:00
#SBATCH --output=downstream/jobs/segmentation/slurm/%x-%j.out
# Submit from the repo root: sbatch downstream/jobs/segmentation/run_all.sh brats_asp
# (any file name under downstream/configs/segmentation/ without .yaml)
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate ai4brainmri
python -m downstream.segmentation.train_segmentation --cfg "downstream/configs/segmentation/${1:?config name}.yaml"
