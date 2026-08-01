#!/bin/bash

#SBATCH --job-name=val_mae
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --output=pretraining/linear_probing/jobs/slurm/%x-%j.out
#SBATCH --error=pretraining/linear_probing/jobs/slurm/%x-%j.err

# Submit from the repo root: sbatch pretraining/linear_probing/jobs/val_job_mae.sh
# Adjust the conda.sh path below for your own cluster -- this one is a placeholder.
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate ai4brainmri

# debugging flags (optional)
export PYTHONFAULTHANDLER=1

# ---- CHANGED: was launched via `torchrun --nproc_per_node=1 ...` against validation.py, but
# that script never touched torch.distributed -- plain python is equivalent and simpler. Also
# now merged with validation_mae_seg_combined.py into linear_probe_sweep.py, selected here via
# --checkpoint_source.
python -m pretraining.linear_probing.linear_probe_sweep \
    --checkpoint_source mae \
    --train_dataset_csv "/path/to/data/mae_UKB_ADNI_HCP_CamCAN_IXI_train_with_stat.csv" \
    --test_dataset_csv "/path/to/data/mae_UKB_ADNI_HCP_CamCAN_IXI_val_with_stat.csv" \
    --features_to_validate "age" "gender" "CerebralWhiteMatterVol" "Brain-Stem_Volume_mm3" "CortexVol"  "BrainSegVol" "Left-Hippocampus_Volume_mm3" "Right-Hippocampus_Volume_mm3" \
    --cache_directory /path/to/cache/validation \
    --batch_size 256 \
    --checkpoints_dir "/path/to/checkpoints/mae_pretrained_model" \
    --vit_params '{"in_chans": 1, "img_size": [128, 128, 128], "patch_size": [16, 16, 16], "patch_embed": "conv", "pos_embed": "sincos", "use_flash_attn": false, "pooling": "cls", "qkv_bias": false}' \
    --output_dir "pretraining/linear_probing/mae_seg" \
    --cut_off_epoch 0
