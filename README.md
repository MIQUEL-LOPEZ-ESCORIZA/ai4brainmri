# AI4BrainMRI

Official code for:

> **Comparative Study of Anatomical and Learned Features in AI Models for Structural Brain MRI**
> Boyang Yu\*, Miquel Lopez Escoriza\*, Long Chen, Arjun V. Masurkar, Narges Razavian, Carlos Fernandez-Granda
> (\*equal contribution)
> Center for Data Science & Courant Institute of Mathematical Sciences, NYU; NYU Grossman School of Medicine

## Overview

We compare the three main feature-extraction paradigms for AI models of 3D structural
T1-weighted brain MRI:

1. **Anatomical features**: linear models on 178 FreeSurfer volumes, surface areas and cortical
   thicknesses;
2. **Supervised CNNs**: a 3D ResNet-18 for classification and regression, and a 3D U-Net for segmentation;
3. **Pretrained vision transformers**: a 3D ViT-B pretrained by masked autoencoding (MAE) on
   ~70,000 scans, adapted to each task with LoRA.

We compare them across seven clinical tasks and 18 public datasets (~96,000 scans from ~80,000
participants). The tasks are Alzheimer's disease diagnosis (internal plus seven external cohorts),
Parkinson's disease, epilepsy, tumor IDH mutation, whole-tumor segmentation, stroke-lesion
segmentation, and brain age.

We also introduce **ASP (Anatomy Segmentation Pretraining)**, a stage that continues pretraining
from the converged MAE encoder. A lightweight, skip-free decoder, conditioned on a region prompt,
segments 39 FreeSurfer brain regions directly from the ViT embedding. The decoder is first trained
alone on the frozen encoder; then segmentation epochs alternate with MAE epochs so that MAE
features are not forgotten.

**This repository covers the whole pipeline**: preprocessing, the three model families, ASP, every
downstream task, and the paired statistical comparison. Each component maps to a section of the
paper, as listed in the table below.

## Paper → code map

| Paper section | What | Code | Configs |
|---|---|---|---|
| Methods: Data preprocessing; Suppl. Imaging data harmonization | DICOM→NIfTI, FreeSurfer 7.4.1 `recon-all` (QC + features + labels), MNI152 2009a registration, manifests, participant/site-level splits | [`preprocessing/`](preprocessing/README.md) | — |
| Methods: Masked Autoencoder Pretraining | Stage 1: MAE, ViT-B/16, 75% masking, 1000 epochs | `pretraining/mae_train.py` | `pretraining/configs/mae_train.yaml` |
| Methods: Anatomy Segmentation Pretraining | Stage 2: ASP (frozen-encoder phase, then alternating seg/MAE epochs), 39 regions | `pretraining/seg_mae_combined_train.py`, `src/models/model.py` | `pretraining/configs/asp_paper.yaml` |
| Suppl. Ablation study on ASP training | no alternation / no freezing / from scratch | same script | `pretraining/configs/asp_ablation_*.yaml` |
| Fig. ASP(b): linear-probe monitoring | Linear probes of anatomy/demographics across pretraining checkpoints | `pretraining/linear_probing/` | — |
| Methods: Downstream application, LoRA | LoRA on the attention layers + task head (reported results) | `downstream/train.py`, `downstream/lora.py` | `downstream/configs/finetune/*_train.yaml` |
| Methods: Downstream application, full finetuning | Full finetuning (alternative strategy, Suppl.) | `downstream/train.py` | `downstream/configs/finetune/*_full_finetune.yaml` |
| Methods: Downstream application, linear probing | Parkinson's, IDH mutation | `downstream/linear_probe.py` | `downstream/configs/linear_probe/` |
| Results: segmentation tasks | UNETR decoder on the LoRA-adapted ViT; UNETR and U-Net baselines | `downstream/segmentation/` | `downstream/configs/segmentation/` |
| Methods: Linear models with anatomical features | Logistic/linear regression on 178 features (soft AD labels); random-forest comparison | [`baselines/anatomical/`](baselines/README.md) | `baselines/configs/anatomical_*.yaml` |
| Methods: Convolutional neural networks | ResNet-18 (SGD, lr 0.01, ≤60 epochs, early stopping) | [`baselines/cnn/`](baselines/README.md) | `baselines/configs/cnn_*.yaml` |
| Methods: Statistical analysis | Participant-level bootstrap CIs; paired bootstrap differences between models with Holm correction | `downstream/metrics.py`, `analysis/compare_models.py` | — |

## Installation

```bash
conda env create -f environment.yml
conda activate ai4brainmri
```

FreeSurfer 7.4.1 and `dcm2niix` are needed only for preprocessing.

## Data

All 18 datasets are public, but each is available only to qualified researchers who apply to its
consortium. No data or derived per-subject files are distributed here. See
[`data/README.md`](data/README.md) for each dataset's role and access link, the exact train/test
protocol per task, and the CSV schema every script expects.

## Reproducing the paper

Every config contains placeholder paths (`/path/to/...`). Point them at your preprocessed data
and checkpoints first.

```bash
# 0. Preprocess every cohort (FreeSurfer + MNI registration), build manifests and splits
sbatch --array=1-N preprocessing/jobs/recon_all_array.sh scans.txt $SUBJECTS_DIR $MNI_TEMPLATE data/mni/<cohort>
python -m preprocessing.build_manifest --mni-dir data/mni/<cohort> --subjects-dir $SUBJECTS_DIR --dataset <cohort> --out data/manifests/<cohort>.csv
python -m preprocessing.make_splits --manifest data/manifests/adni.csv --test-frac 0.2 --val-frac 0.1 --stratify-col pred_label --out-prefix data/splits/ADNI

# 1. Stage 1: MAE pretraining
python -m pretraining.mae_train --cfg pretraining/configs/mae_train.yaml

# 2. Stage 2: ASP
sbatch pretraining/jobs/asp_paper.sh pretraining/configs/asp_paper.yaml

# 3. Downstream: LoRA (reported) for the ASP and MAE ("Pretrained ViT") encoders
python -m downstream.train    --cfg downstream/configs/finetune/ad_diagnosis_train.yaml
python -m downstream.evaluate --cfg downstream/configs/eval/ad_diagnosis.yaml
python -m downstream.linear_probe --cfg downstream/configs/linear_probe/parkinsons.yaml
python -m downstream.segmentation.train_segmentation --cfg downstream/configs/segmentation/brats_asp.yaml

# 4. Baselines
python -m baselines.anatomical.train_linear --cfg baselines/configs/anatomical_ad.yaml
python -m baselines.cnn.train_resnet        --cfg baselines/configs/cnn_ad.yaml

# 5. Paired comparison of all models on the same test scans
python -m analysis.compare_models --metric ad_macro_auc --subject-col subject_id --cohort-col source_dataset \
    --reference asp --predictions asp=results/predictions/ad_diagnosis_asp.csv \
    mae=results/predictions/ad_diagnosis_mae.csv anat=results/predictions/ad_anatomical.csv cnn=results/predictions/ad_cnn.csv
```

Every evaluation script writes per-scan predictions in one shared schema
(`scan, label, prediction, subject_id[, source_dataset]`). As a result, any two models can be
compared on exactly the same participants.

## Compute

| Stage | Budget reported in the paper |
|---|---|
| FreeSurfer `recon-all` | ~6.1 h per scan, CPU |
| MAE pretraining | 1000 epochs (converged ~900), ~10 days on 2× A100 80 GB |
| ASP | 400 epochs (converged ~200), ~1 month on 1× RTX 8000 48 GB |
| Inference | ASP and MAE encoders are both a plain ViT-B/16 (the ASP decoder is discarded), so inference cost is identical |

## Pretrained checkpoints

The MAE and ASP encoder weights will be released with the publication. Links will be added here.

## Citation

```bibtex
@article{yu2026comparative,
  title   = {Comparative Study of Anatomical and Learned Features in AI Models for Structural Brain MRI},
  author  = {Yu, Boyang and Lopez Escoriza, Miquel and Chen, Long and Masurkar, Arjun V. and Razavian, Narges and Fernandez-Granda, Carlos},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).

## Contact

For questions about the paper, contact the corresponding author, Carlos Fernandez-Granda
(cfgranda@cims.nyu.edu). For issues with this codebase, please open a GitHub issue.
