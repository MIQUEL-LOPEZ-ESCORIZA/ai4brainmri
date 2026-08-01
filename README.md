# AI4BrainMRI

Official code for:

> **Comparative Study of Anatomical and Learned Features in AI Models for Structural Brain MRI**
> Boyang Yu\*, Miquel Lopez Escoriza\*, Long Chen, Arjun V. Masurkar, Narges Razavian, Carlos Fernandez-Granda
> (\*equal contribution)
> Center for Data Science & Courant Institute of Mathematical Sciences, NYU; NYU Grossman School of Medicine
> Preprint — venue TBD

## Overview

We comprehensively compare three feature-extraction paradigms for AI-based neuroimaging — anatomical
surface/volume features, supervised CNNs, and self-supervised ViT foundation models — across seven
clinical tasks and 19 public datasets covering over 80,000 subjects. A linear model on 161 FreeSurfer
anatomical features matches the diagnostic performance of CNNs and pretrained ViTs, indicating that
deep models on structural MRI tend to rediscover anatomical information rather than surpass it.

Building on this observation, we propose **ASP (Anatomy Segmentation Pretraining)**: after standard
masked-autoencoder (MAE) pretraining, a lightweight bottleneck decoder is added and trained to segment
39 brain regions in epochs alternating with continued MAE reconstruction, explicitly injecting
anatomical structure into the pretrained embedding. ASP matches or beats plain MAE pretraining across
diagnostic tasks and clearly outperforms it on biological age estimation. See the paper for full
results.

This repository contains the ASP pipeline: pretraining (stage 1 MAE, stage 2 ASP) and its downstream
application (fine-tuning, linear probing, evaluation) to clinical tasks.

## Repository structure

```
ai4brainmri/
├── src/            # shared library: ViT/MAE/segmentation-decoder models, dataloaders,
│                   # optimizer/scheduler utilities
├── pretraining/    # stage 1 (MAE) and stage 2 (ASP) pretraining — see pretraining/README.md
└── downstream/     # fine-tuning, linear probing, and evaluation on clinical tasks
                    # — see downstream/README.md
```

This mirrors the paper's Methods section: pretraining first, then downstream application, with the
three adaptation strategies (full fine-tuning, partial fine-tuning, linear probing) applied per task
according to how much labeled data is available.

## Installation

```bash
conda env create -f environment.yml
conda activate ai4brainmri
```

## Data

All experiments use 3D structural T1-weighted MRI, preprocessed with FreeSurfer's `recon-all` and
co-registered to the MNI152 2009a Nonlinear Symmetric template (197×233×189, 1mm isotropic). Data used
in the paper come from 19 publicly available repositories (UK Biobank, ADNI, NACC, AIBL, OASIS 1-4,
MIRIAD, HCP, CamCAN, IXI, PPMI, OpenNeuro, EPISURG, UCSF-PDGM, BraTS, ATLAS), available to qualified
researchers through application to the respective consortia. No new data was collected for this study.

Preprocessed data is not distributed with this repository. See [`downstream/README.md`](downstream/README.md)
for the expected CSV/manifest format.

## Usage

```bash
# Pretraining — stage 1: masked-autoencoder pretraining
python -m pretraining.mae_train --cfg pretraining/configs/mae_train.yaml

# Pretraining — stage 2: Anatomy Segmentation Pretraining (alternating segmentation/MAE epochs)
torchrun --nproc_per_node=1 --master_port=12001 -m pretraining.seg_mae_combined_train \
    --cfg pretraining/configs/mae_seg_combined_train.yaml --local_rank 0

# Downstream — fine-tune an ASP-pretrained encoder on a clinical task
python -m downstream.train --cfg downstream/configs/finetune/ad_diagnosis_train.yaml

# Downstream — final, paper-comparable evaluation (bootstrap CIs, per-cohort breakdown)
python -m downstream.evaluate --cfg downstream/configs/eval/ad_diagnosis.yaml

# Downstream — linear probing (frozen encoder + sklearn head)
python -m downstream.linear_probe --cfg downstream/configs/linear_probe/glioma_registered.yaml
```

See [`pretraining/README.md`](pretraining/README.md) and [`downstream/README.md`](downstream/README.md)
for the full task list, config reference, and SLURM job scripts.

## Pretrained checkpoints

Not yet released. This section will be updated with download links upon publication.

## Citation

```bibtex
@article{yu2026comparative,
  title   = {Comparative Study of Anatomical and Learned Features in AI Models for Structural Brain MRI},
  author  = {Yu, Boyang and Lopez Escoriza, Miquel and Chen, Long and Masurkar, Arjun V. and Razavian, Narges and Fernandez-Granda, Carlos},
  journal = {Preprint},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).

## Contact

For questions about the paper, contact the corresponding author: Carlos Fernandez-Granda
(cfgranda@cims.nyu.edu). For issues with this codebase, please open a GitHub issue.
