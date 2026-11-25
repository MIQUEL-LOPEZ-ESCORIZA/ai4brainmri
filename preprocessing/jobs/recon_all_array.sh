#!/bin/bash
# SLURM array over a list of scans: one recon-all + MNI registration per task.
# scans.txt: one line per scan, "<t1.nii.gz> <scan_id>". recon-all takes ~6 h per scan.
#SBATCH --job-name=recon_all
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=preprocessing/jobs/slurm/%x-%A_%a.out
# Submit from the repo root:
#   sbatch --array=1-$(wc -l < scans.txt) preprocessing/jobs/recon_all_array.sh scans.txt <subjects_dir> <mni_template> <mni_out_dir>
set -euo pipefail
LIST=${1:?scan list}; SUBJECTS_DIR=${2:?}; TEMPLATE=${3:?}; OUT=${4:?}
read -r T1 SCAN_ID < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
preprocessing/run_recon_all.sh "$T1" "$SCAN_ID" "$SUBJECTS_DIR"
[ -f "$SUBJECTS_DIR/$SCAN_ID/mri/aparc+aseg.mgz" ] && preprocessing/register_to_mni.sh "$SUBJECTS_DIR" "$SCAN_ID" "$TEMPLATE" "$OUT"
