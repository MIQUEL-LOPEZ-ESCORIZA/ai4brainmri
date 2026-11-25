# Preprocessing

This implements the paper's "Data preprocessing" section and the Supplementary "Imaging data
harmonization". Every cohort except BraTS and ATLAS R2.0 (which are distributed already
standardized) goes through the same steps:

| Step | Script | Output |
|---|---|---|
| 0. DICOM → NIfTI (DICOM cohorts only) | `convert_dicom.sh` (dcm2niix) | `<series>.nii.gz` |
| 1. FreeSurfer 7.4.1 `recon-all` | `run_recon_all.sh` | `$SUBJECTS_DIR/<scan>/` with `stats/` and `mri/aparc+aseg.mgz`; failures logged to `qc_failures.csv` |
| 2. Registration to MNI152 2009a nonlinear symmetric | `register_to_mni.sh` (`mri_vol2vol`) | `<scan>_T1_mni152.nii.gz`, `<scan>_aparc+aseg_mni152.nii.gz` (197×233×189, 1 mm) |
| 3. Manifest | `build_manifest.py` | one CSV per cohort, schema in `data/README.md` |
| 4. Splits | `make_splits.py` | participant-level (ATLAS: site-level) train/val/test CSVs |

`jobs/recon_all_array.sh` runs steps 1 and 2 as a SLURM array, one scan per task. Each scan takes
about 6 hours on one CPU core: 0.36 h autorecon1, 3.78 h autorecon2 and 1.97 h autorecon3,
measured over 1,500 runs.

## What each output is used for

- **Quality control.** A scan whose `-autorecon1` stage fails is excluded from the study.
- **Anatomical features.** `stats/aseg.stats` and `stats/{lh,rh}.aparc.stats` feed
  `baselines/anatomical/extract_freesurfer_features.py`, which extracts the 178 features.
- **ASP labels.** The MNI-space `aparc+aseg` provides the 112 FreeSurfer labels, which are grouped
  into 39 regions (`pretraining/files/selected_brain_regions.csv`, `src/data/dataloaders.py::grouping_total`).
  Left/right pairs and the five corpus-callosum segments are merged. Regions that have no strong
  disease association and cover <1% of brain volume are dropped (5.2% of brain volume on average).
- **Model input.** The MNI-space T1, scaled to [0, 1] and padded or cropped to 128³ by the dataloaders.

## Things to check against your own installation

- `register_to_mni.sh` resamples FreeSurfer's skull-stripped `brain.mgz` with the `recon-all`
  Talairach transform. If you need bit-for-bit agreement with the paper's preprocessed volumes,
  first check which input volume and transform your copy of the original pipeline used.
