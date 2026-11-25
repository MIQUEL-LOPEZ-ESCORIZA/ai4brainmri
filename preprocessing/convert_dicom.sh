#!/usr/bin/env bash
# Step 0 (only for cohorts distributed as DICOM): convert each DICOM series to NIfTI.
# The paper used the dcm2nii tool; dcm2niix is its maintained successor with the same output.
#
# Usage: preprocessing/convert_dicom.sh <dicom_root> <nifti_out_dir>
#   Every leaf directory under <dicom_root> is treated as one series; output files are named
#   <series_dir_name>.nii.gz.
set -euo pipefail
DICOM_ROOT=${1:?dicom root}
OUT=${2:?output dir}
mkdir -p "$OUT"
find "$DICOM_ROOT" -type d -links 2 | while read -r series; do
  name=$(basename "$series")
  dcm2niix -z y -b n -f "$name" -o "$OUT" "$series" >/dev/null || echo "FAILED: $series" >&2
done
