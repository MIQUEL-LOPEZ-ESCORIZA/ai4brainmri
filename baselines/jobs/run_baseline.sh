#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu
#SBATCH --time=2-00:00:00
#SBATCH --output=baselines/jobs/slurm/%x-%j.out
# Submit from the repo root, e.g.:
#   sbatch baselines/jobs/run_baseline.sh cnn_ad         (ResNet-18; needs a GPU)
#   sbatch baselines/jobs/run_baseline.sh anatomical_ad  (logistic regression; CPU is enough)
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate ai4brainmri
CFG="baselines/configs/${1:?config name}.yaml"
case "$1" in
  cnn_*) python -m baselines.cnn.train_resnet --cfg "$CFG" ;;
  anatomical_*) python -m baselines.anatomical.train_linear --cfg "$CFG" ;;
esac
