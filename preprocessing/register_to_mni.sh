#!/usr/bin/env bash
# Step 2: bring the conformed T1 and its FreeSurfer segmentation into MNI152 2009a nonlinear
# symmetric space with mri_vol2vol (paper Methods, "Data preprocessing"). Output volumes are
# 197 x 233 x 189 voxels at 1 mm isotropic. The image is resampled with trilinear interpolation,
# label volumes with nearest neighbour so label IDs are preserved.
#
# BraTS and ATLAS R2.0 are distributed already standardized and are NOT passed through this
# script (their manual masks are never interpolated).
#
# Usage: preprocessing/register_to_mni.sh <subjects_dir> <subject_id> <mni_template.nii.gz> <out_dir>
#   <mni_template.nii.gz>: mni_icbm152_t1_tal_nlin_sym_09a.nii (MNI152 2009a nonlinear symmetric)
set -euo pipefail
export SUBJECTS_DIR=${1:?subjects dir}
SUBJ=${2:?subject id}
TEMPLATE=${3:?MNI template}
OUT=${4:?output dir}
MRI="$SUBJECTS_DIR/$SUBJ/mri"
mkdir -p "$OUT"

# Talairach/MNI transform estimated by recon-all (-autorecon1)
XFM="$MRI/transforms/talairach.xfm"

mri_vol2vol --mov "$MRI/brain.mgz" --targ "$TEMPLATE" --xfm "$XFM" \
  --o "$OUT/${SUBJ}_T1_mni152.nii.gz" --trilin
mri_vol2vol --mov "$MRI/aparc+aseg.mgz" --targ "$TEMPLATE" --xfm "$XFM" \
  --o "$OUT/${SUBJ}_aparc+aseg_mni152.nii.gz" --nearest
