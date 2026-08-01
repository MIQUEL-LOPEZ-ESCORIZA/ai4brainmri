#!/bin/bash

#SBATCH --job-name=val_mae_seg_combined_all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --output=pretraining/linear_probing/jobs/slurm/%x-%j.out
#SBATCH --error=pretraining/linear_probing/jobs/slurm/%x-%j.err

# Submit from the repo root: sbatch pretraining/linear_probing/jobs/val_job_mae_seg_combined_all.sh
# Adjust the conda.sh path below for your own cluster -- this one is a placeholder.
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate ai4brainmri

# debugging flags (optional)
export PYTHONFAULTHANDLER=1

# --- Put this near the top of your script ---
FEATURES=(
"age" "gender"
"Left-Lateral-Ventricle_Volume_mm3" "Left-Inf-Lat-Vent_Volume_mm3"
"Left-Cerebellum-White-Matter_Volume_mm3" "Left-Cerebellum-Cortex_Volume_mm3"
"Left-Thalamus_Volume_mm3" "Left-Caudate_Volume_mm3" "Left-Putamen_Volume_mm3" "Left-Pallidum_Volume_mm3"
"3rd-Ventricle_Volume_mm3" "4th-Ventricle_Volume_mm3" "Brain-Stem_Volume_mm3"
"Left-Hippocampus_Volume_mm3" "Left-Amygdala_Volume_mm3" "CSF_Volume_mm3"
"Left-Accumbens-area_Volume_mm3" "Left-VentralDC_Volume_mm3"
"Left-vessel_Volume_mm3" "Left-choroid-plexus_Volume_mm3"
"Right-Lateral-Ventricle_Volume_mm3" "Right-Inf-Lat-Vent_Volume_mm3"
"Right-Cerebellum-White-Matter_Volume_mm3" "Right-Cerebellum-Cortex_Volume_mm3"
"Right-Thalamus_Volume_mm3" "Right-Caudate_Volume_mm3" "Right-Putamen_Volume_mm3" "Right-Pallidum_Volume_mm3"
"Right-Hippocampus_Volume_mm3" "Right-Amygdala_Volume_mm3"
"Right-Accumbens-area_Volume_mm3" "Right-VentralDC_Volume_mm3"
"Right-vessel_Volume_mm3" "Right-choroid-plexus_Volume_mm3"
"5th-Ventricle_Volume_mm3" "WM-hypointensities_Volume_mm3"
"Left-WM-hypointensities_Volume_mm3" "Right-WM-hypointensities_Volume_mm3"
"non-WM-hypointensities_Volume_mm3"
"Left-non-WM-hypointensities_Volume_mm3" "Right-non-WM-hypointensities_Volume_mm3"
"Optic-Chiasm_Volume_mm3"
"CC_Posterior_Volume_mm3" "CC_Mid_Posterior_Volume_mm3" "CC_Central_Volume_mm3"
"CC_Mid_Anterior_Volume_mm3" "CC_Anterior_Volume_mm3"

"bankssts_SurfArea" "bankssts_GrayVol" "bankssts_ThickAvg"
"caudalanteriorcingulate_SurfArea" "caudalanteriorcingulate_GrayVol" "caudalanteriorcingulate_ThickAvg"
"caudalmiddlefrontal_SurfArea" "caudalmiddlefrontal_GrayVol" "caudalmiddlefrontal_ThickAvg"
"cuneus_SurfArea" "cuneus_GrayVol" "cuneus_ThickAvg"
"entorhinal_SurfArea" "entorhinal_GrayVol" "entorhinal_ThickAvg"
"fusiform_SurfArea" "fusiform_GrayVol" "fusiform_ThickAvg"
"inferiorparietal_SurfArea" "inferiorparietal_GrayVol" "inferiorparietal_ThickAvg"
"inferiortemporal_SurfArea" "inferiortemporal_GrayVol" "inferiortemporal_ThickAvg"
"isthmuscingulate_SurfArea" "isthmuscingulate_GrayVol" "isthmuscingulate_ThickAvg"
"lateraloccipital_SurfArea" "lateraloccipital_GrayVol" "lateraloccipital_ThickAvg"
"lateralorbitofrontal_SurfArea" "lateralorbitofrontal_GrayVol" "lateralorbitofrontal_ThickAvg"
"lingual_SurfArea" "lingual_GrayVol" "lingual_ThickAvg"
"medialorbitofrontal_SurfArea" "medialorbitofrontal_GrayVol" "medialorbitofrontal_ThickAvg"
"middletemporal_SurfArea" "middletemporal_GrayVol" "middletemporal_ThickAvg"
"parahippocampal_SurfArea" "parahippocampal_GrayVol" "parahippocampal_ThickAvg"
"paracentral_SurfArea" "paracentral_GrayVol" "paracentral_ThickAvg"
"parsopercularis_SurfArea" "parsopercularis_GrayVol" "parsopercularis_ThickAvg"
"parsorbitalis_SurfArea" "parsorbitalis_GrayVol" "parsorbitalis_ThickAvg"
"parstriangularis_SurfArea" "parstriangularis_GrayVol" "parstriangularis_ThickAvg"
"pericalcarine_SurfArea" "pericalcarine_GrayVol" "pericalcarine_ThickAvg"
"postcentral_SurfArea" "postcentral_GrayVol" "postcentral_ThickAvg"
"posteriorcingulate_SurfArea" "posteriorcingulate_GrayVol" "posteriorcingulate_ThickAvg"
"precentral_SurfArea" "precentral_GrayVol" "precentral_ThickAvg"
"precuneus_SurfArea" "precuneus_GrayVol" "precuneus_ThickAvg"
"rostralanteriorcingulate_SurfArea" "rostralanteriorcingulate_GrayVol" "rostralanteriorcingulate_ThickAvg"
"rostralmiddlefrontal_SurfArea" "rostralmiddlefrontal_GrayVol" "rostralmiddlefrontal_ThickAvg"
"superiorfrontal_SurfArea" "superiorfrontal_GrayVol" "superiorfrontal_ThickAvg"
"superiorparietal_SurfArea" "superiorparietal_GrayVol" "superiorparietal_ThickAvg"
"superiortemporal_SurfArea" "superiortemporal_GrayVol" "superiortemporal_ThickAvg"
"supramarginal_SurfArea" "supramarginal_GrayVol" "supramarginal_ThickAvg"
"frontalpole_SurfArea" "frontalpole_GrayVol" "frontalpole_ThickAvg"
"temporalpole_SurfArea" "temporalpole_GrayVol" "temporalpole_ThickAvg"
"transversetemporal_SurfArea" "transversetemporal_GrayVol" "transversetemporal_ThickAvg"
"insula_SurfArea" "insula_GrayVol" "insula_ThickAvg"

"BrainSegVol" "VentricleChoroidVol" "rhCortexVol" "lhCerebralWhiteMatterVol"
"CerebralWhiteMatterVol" "TotalGrayVol" "SupraTentorialVolNotVent"
"BrainSegVol-to-eTIV" "lhSurfaceHoles" "SurfaceHoles" "NumVert"
"MeanThickness" "BrainSegVolNotVent" "CortexVol"
)

# ---- CHANGED: was launched via `torchrun --nproc_per_node=1 ...` against
# validation_mae_seg_combined.py, but that script never touched torch.distributed -- plain
# python is equivalent and simpler. Also now merged with validation.py into
# linear_probe_sweep.py, selected here via --checkpoint_source.
python -m pretraining.linear_probing.linear_probe_sweep \
    --checkpoint_source asp \
    --train_dataset_csv "/path/to/data/mae_UKB_ADNI_HCP_CamCAN_IXI_train_with_stat.csv" \
    --test_dataset_csv "/path/to/data/mae_UKB_ADNI_HCP_CamCAN_IXI_val_with_stat.csv" \
    --features_to_validate "${FEATURES[@]}" \
    --cache_directory /path/to/cache/validation \
    --batch_size 256 \
    --checkpoints_dir "/path/to/checkpoints/seg_mae_combined_checkpoints" \
    --vit_params '{"in_chans": 1, "img_size": [128, 128, 128], "patch_size": [16, 16, 16], "patch_embed": "conv", "pos_embed": "sincos", "use_flash_attn": false, "qkv_bias": true}' \
    --output_dir "pretraining/linear_probing/mae_seg" \
    --cut_off_epoch 450
