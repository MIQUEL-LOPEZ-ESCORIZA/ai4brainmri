# Data

No imaging data, clinical tables or per-subject derived files are distributed with this
repository. Every dataset is public, but access requires a separate application to its consortium
(data-use agreements forbid redistribution). This file documents what the paper used and how each
cohort enters the pipeline, so the experiments can be rebuilt from the original sources.

## Datasets

Counts are after FreeSurfer quality control (scans whose `recon-all -autorecon1` failed are excluded).

| Dataset | Role in the paper | Scans | Participants | Access |
|---|---|---:|---:|---|
| UK Biobank | Pretraining; brain-age training; internal brain-age test (holdout) | 67,294 | 62,665 | https://www.ukbiobank.ac.uk (application 166231) |
| HCP (Aging + Development) | Pretraining; brain-age training | 1,350 | 1,350 | https://www.humanconnectome.org (NDA) |
| CamCAN | Pretraining; brain-age training | 653 | 653 | https://cam-can.org |
| IXI | Pretraining; brain-age training | 581 | 581 | https://brain-development.org/ixi-dataset/ |
| ADNI | AD training + internal test; external brain-age test (healthy) | 6,471 | 1,817 | https://adni.loni.usc.edu |
| NACC | External AD test; external brain-age test (healthy) | 6,956 | 4,926 | https://naccdata.org |
| AIBL | External AD test | 1,167 | 664 | https://aibl.org.au (via LONI) |
| OASIS-1 / 2 / 3 / 4 | External AD tests (each analysed separately) | 436 / 373 / 2,275 / 181 | 416 / 150 / 1,044 / 180 | https://sites.wustl.edu/oasisbrains/ |
| MIRIAD | External AD test | 705 | 69 | https://www.ucl.ac.uk/drc/research/research-methods/miriad-scan-database |
| PPMI | Parkinson's disease | 2,032 | 1,306 | https://www.ppmi-info.org |
| OpenNeuro ds004199 | Epilepsy (FCD II) vs. controls | 163 | 163 | https://openneuro.org/datasets/ds004199 |
| EPISURG | External epilepsy test (patients only) | 268 | 268 | https://doi.org/10.5522/04/9996158 |
| UCSF-PDGM | IDH mutation (T1 and T1-CE) | 1,980 | 501 | https://www.cancerimagingarchive.net (UCSF-PDGM) |
| BraTS 2023 Adult Glioma | Whole-tumor segmentation (T1 and T1-CE) | 2,502 | 1,251 | https://www.synapse.org (BraTS 2023) |
| ATLAS R2.0 | Stroke-lesion segmentation | 655 | 655 (33 sites) | https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html |

Pretraining cohort: UKBB + HCP + CamCAN + IXI = 69,878 scans from 65,249 participants. The UKBB
holdout (10% of UKBB participants, 6,950 scans) is excluded from MAE and ASP optimization. It was
used to track linear-probe metrics during pretraining (not for checkpoint or model selection) and
is the internal brain-age test cohort. None of the downstream cohorts are in the pretraining set.

## Train/test protocol per task

All splits are at the **participant level**: every scan of a participant is in exactly one split.
ATLAS is split at the stricter **site level**. Use `preprocessing/make_splits.py`, which asserts
that no group appears in two splits.

| Task | Training | Internal test | External test | Labels |
|---|---|---|---|---|
| Alzheimer's diagnosis | ADNI, 1,453 participants (10% of them for validation) | ADNI, 364 participants | NACC, AIBL, OASIS-1/2/3/4, MIRIAD (all participants) | CN = 0, MCI = 0.5, AD = 1; metric: macro-AUC of AD-vs-CN, AD-vs-MCI, MCI-vs-CN |
| Parkinson's disease | PPMI, 1,572 scans | PPMI, 460 scans | — | PD diagnosis (PDDXDATE) vs. control |
| Epilepsy | OpenNeuro, 82 participants | OpenNeuro, 81 participants | EPISURG patients (n=268) + OpenNeuro test controls as negatives | FCD/epilepsy = 1, control = 0 |
| IDH mutation | UCSF-PDGM, 80% of participants | UCSF-PDGM, 20% of participants | — | any IDH mutation vs. wild type; T1 and T1-CE are separate samples, predictions averaged per participant |
| Whole-tumor segmentation | BraTS, 1,000 participants (also a 10% subset) | BraTS, 251 participants | — | labels 1+2+3; T1 and T1-CE used for training, T1-CE for testing |
| Stroke-lesion segmentation | ATLAS, 21 sites (355 scans) | ATLAS, 12 sites (300 scans) | — | manual lesion mask |
| Brain age | Healthy UKBB (excluding the holdout and 274 participants with CNS or neurocognitive ICD-10 codes) + HCP + CamCAN + IXI | UKBB holdout, 6,950 scans | Healthy ADNI (n=2,791) and NACC (n=1,136) participants under 90 | chronological age at scan |

BrainIAC is not evaluated on Alzheimer's diagnosis because its pretraining data include several of
our external AD cohorts.

## CSV schema

Every script reads plain CSV manifests (built by `preprocessing/build_manifest.py` and split by
`preprocessing/make_splits.py`). Column names are configurable in each YAML config. The defaults are:

| Column | Used by | Meaning |
|---|---|---|
| `img_path_T1_mni152` | all image models | MNI-registered T1 (`.nii.gz`/`.mgz`, 197×233×189, 1 mm) |
| `img_seg_path_T1` | ASP pretraining | FreeSurfer `aparc+aseg` in the same space (39 grouped regions are derived on the fly) |
| `is_img_path_T1_mni152`, `is_aparc+aseg` | pretraining | validity flags |
| `subject_id` | splits, bootstrap | participant identifier |
| `source_dataset` | evaluation | cohort name, for per-cohort results |
| `pred_label` / task label | training, evaluation | see the table above |
| `fs_subject_dir` | anatomical baseline | FreeSurfer subject folder (`stats/*.stats`) |
| `img_path_T1`, `mask_path` | segmentation | native-space image and mask (BraTS, ATLAS) |

`pretraining/files/selected_brain_regions.csv` (39 ASP regions, FreeSurfer label IDs) is the only
reference table shipped with the repository.
