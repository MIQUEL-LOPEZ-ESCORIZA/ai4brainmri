#!/usr/bin/env bash
# Step 1: FreeSurfer v7.4.1 cortical reconstruction for one scan (paper Methods, "Data
# preprocessing"; Supplementary "Imaging data harmonization").
#
# recon-all runs three phases: -autorecon1 (motion correction, conform, intensity normalization,
# skull strip; ~0.4 h), -autorecon2 (subcortical segmentation and surface reconstruction; ~3.8 h)
# and -autorecon3 (spherical registration, cortical parcellation, statistics; ~2.0 h). Its outputs
# are used three ways:
#   * quality control: scans whose -autorecon1 fails are excluded from the study;
#   * anatomical features: stats/aseg.stats, stats/{lh,rh}.aparc.stats
#     -> baselines/anatomical/extract_freesurfer_features.py;
#   * ASP labels: mri/aparc+aseg.mgz -> preprocessing/register_to_mni.sh.
#
# Usage (one scan; parallelize with a job array, see preprocessing/README.md):
#   preprocessing/run_recon_all.sh <t1.nii.gz> <subject_id> <subjects_dir>
set -euo pipefail
T1=${1:?T1 image}
SUBJ=${2:?subject id}
export SUBJECTS_DIR=${3:?subjects dir}
mkdir -p "$SUBJECTS_DIR"

if ! recon-all -i "$T1" -s "$SUBJ" -autorecon1 -openmp "${OMP_NUM_THREADS:-1}"; then
  echo "$SUBJ,autorecon1_failed" >> "$SUBJECTS_DIR/qc_failures.csv"   # excluded (QC)
  exit 0
fi
recon-all -s "$SUBJ" -autorecon2 -autorecon3 -openmp "${OMP_NUM_THREADS:-1}" \
  || echo "$SUBJ,autorecon23_failed" >> "$SUBJECTS_DIR/qc_failures.csv"
