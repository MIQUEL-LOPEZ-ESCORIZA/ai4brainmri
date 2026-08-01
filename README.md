# AI4BrainMRI

Clean rebuild of the codebase behind:

> **Comparative Study of Anatomical and Learned Features in AI Models for Structural Brain MRI**
> Boyang Yu†, Miquel Lopez Escoriza†, Long Chen, Arjun V. Masurkar, Narges Razavian,
> Carlos Fernandez-Granda (†equal contribution)
> Center for Data Science / Courant Institute, NYU; NYU Grossman School of Medicine.
> *Venue/year: TBD — this repo tracks a preprint draft (`AI_for_Structural_MRI (4).pdf`).*

The paper compares three feature-extraction paradigms for 3D structural T1-weighted MRI
(anatomical features, CNNs, pretrained ViTs) across seven clinical tasks and ~80,000 subjects,
and proposes **ASP (Anatomy Segmentation Pretraining)**: pretrain a ViT with masked autoencoding,
then add a lightweight bottleneck decoder and train it to segment 39 brain regions in alternating
epochs with continued MAE reconstruction — injecting explicit anatomical structure into the
pretrained embedding. ASP matches or beats plain MAE pretraining across diagnostic tasks and
clearly outperforms it on biological age estimation.

## Why this repo exists

The original implementation (`AI_FOR_STRUCTURAL_BRAIN_MRI/`, plus a separate `SEGMENTATION`
pretraining repo) grew organically during experimentation: the same fine-tuning logic was
duplicated across four locations, task behavior was switched by matching substrings inside
checkpoint filenames, and ~1GB of SLURM logs ended up committed to git. This repo restructures
just the ASP pipeline — the paper's proposed method and (per the results) its best-performing
model — into one place organized so the code mirrors the paper's own structure instead of a pile
of cluster job scripts. `downstream/` is a from-scratch rebuild; `pretraining/` is a structural
+ self-containment port of the original training code (see "Structure" below for which is which,
and why).

## Structure

```
ai4brainmri/
├── src/                    # shared library (ViT/MAE/segmentation-decoder models, dataloaders,
│                            # DDP + optimizer/scheduler utils, utils_seg.py) -- used by
│                            # pretraining/, hoisted to the top level rather than nested, so it
│                            # isn't tied to one pipeline
├── pretraining/             # stage 1 (mae_train.py, new code) + stage 2 (seg_mae_combined_train.py,
│                            # ported as-is from the original SEGMENTATION repo: the alternating
│                            # segmentation/MAE training loop) + linear-probing-during-pretraining
│                            # monitoring (paper Figure 4b) -- see pretraining/README.md for what
│                            # was copied vs. written from scratch vs. dropped
└── downstream/              # downstream application of a pretrained ViT to clinical tasks:
                             # train.py (fine-tune), linear_probe.py (frozen encoder + sklearn),
                             # evaluate.py (final, bootstrap-CI'd, per-cohort metrics) -- see
                             # downstream/README.md for the task list
```

This mirrors the paper's Methods section ordering: pretraining stage first, then downstream
application, with the three adaptation strategies the paper names (full fine-tuning, partial
fine-tuning, linear probing) applied per task based on how much labeled data is available.

`downstream/` is a from-scratch, single-GPU rewrite (see its README for what's simplified).
`pretraining/` is the opposite: a structural port of the original multi-GPU/DDP code with the
training logic untouched, since it needs to match what actually produced the published
checkpoints. See `pretraining/README.md` for the stage-1 (plain MAE) gap that's still unfilled.

## Installation

```bash
conda env create -f environment.yml
conda activate ai4brainmri
```

Requires a preprocessed dataset (T1 MNI152 space, FreeSurfer-registered) and an ASP-pretrained
checkpoint — neither is distributed with this repo. See `downstream/README.md` for the CSV
format and `pretraining/README.md` for what produces a checkpoint.

## Status

- [x] Downstream ASP pipeline, fine-tuning (`train.py`): AD diagnosis, healthy aging, epilepsy
- [x] Downstream ASP pipeline, linear probing (`linear_probe.py`): glioma/IDH mutation (raw +
      registered), Parkinson's, epilepsy
- [x] Downstream ASP pipeline, final evaluation (`evaluate.py`): bootstrap-CI'd metrics, the
      3-way pairwise AD-diagnosis macro-AUC, per-cohort stratified breakdown
- [x] Pretraining, ASP stage (`seg_mae_combined_train.py`) — ported and self-contained, see
      `pretraining/README.md`
- [x] Pretraining, plain MAE stage (`mae_train.py`) — new code (no original to port, its own
      external repo is inaccessible), see `pretraining/README.md`
- [ ] LoRA adaptation (mentioned in the paper for some tasks; no implementation exists in the
      original codebase either)
- [ ] Segmentation downstream tasks (BraTS, ATLAS) — out of scope for this pass
- [ ] MCI progression, NIFD (frontotemporal dementia) — deliberately dropped, see
      `downstream/README.md`
