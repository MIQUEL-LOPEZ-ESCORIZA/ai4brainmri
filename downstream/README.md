# Downstream application of pretrained ViTs

This follows the paper's "Downstream Application of Pretrained ViTs" section. A pretrained ViT-B
encoder, either ASP (`pretraining/configs/asp_paper.yaml`) or plain MAE, the paper's "Pretrained
ViT" (`pretraining/configs/mae_train.yaml`), gets a small task head and is adapted to each clinical
task. Both encoders go through exactly the same downstream configs; only `model.pretrained`
changes.

| Strategy | What is trained | Used for in the paper | Config |
|---|---|---|---|
| **LoRA** (reported results) | Low-rank adapters in the attention layers (`attn.qkv`) + task head; pretrained weights frozen | AD diagnosis, epilepsy, brain age, and the ViT encoders of both segmentation tasks | `model.adaptation: lora` (`configs/finetune/*_train.yaml`, `configs/segmentation/*_{asp,mae}.yaml`) |
| **Full finetuning** (alternative strategy, Supplementary) | Every encoder parameter + head, same optimization settings as LoRA | Supplementary comparison with LoRA (AD, epilepsy, brain age) | `model.adaptation: full` (`configs/finetune/*_full_finetune.yaml`) |
| **Linear probing** | `sklearn` logistic/linear regression on frozen CLS embeddings | Parkinson's disease, IDH mutation (small training sets); pretraining monitoring | `linear_probe.py`, `configs/linear_probe/` |

Task heads:
- **Classification:** `LayerNorm -> Linear` on the CLS embedding. Alzheimer's diagnosis uses the soft
  2-way target CN = 0, MCI = 0.5, AD = 1 (`train.smooth_mci_label: true`), which is equivalent to
  binary cross-entropy on soft labels.
- **Regression (brain age):** the same head with `num_classes: 1` and `MSELoss`.
- **Segmentation:** a UNETR decoder fed by the hidden states of blocks 3, 6, 9 and 12
  (`segmentation/model.py`).

Augmentation: random 128³ crop and intensity shift during training. At test time, a center 128³
crop for classification/regression, or sliding-window inference over the full volume for segmentation.

Evaluation (`evaluate.py`, `linear_probe.py`, `segmentation/train_segmentation.py`) reports
bootstrap 95% CIs. When `data.subject_col` is set, participants are resampled (all scans of a
participant together), and per-scan predictions are written to `predictions_csv` for the paired
model comparison in `analysis/compare_models.py`.

---

# ASP downstream pipeline — implementation details

Downstream application of an **ASP** (Anatomy Segmentation Pretraining)-pretrained ViT encoder
to clinical tasks. See `pretraining/README.md` for what ASP pretraining is; this pipeline only
*consumes* a checkpoint it produces.

Three entry points, cleanly separated by responsibility — everything else (which task, which
data, which strategy variant) is a config file, not a new script:

| Script | Responsibility | Configs live in |
|---|---|---|
| `train.py` | Adapt the encoder (LoRA, full or partial finetuning), pick the best checkpoint by validation loss | `configs/finetune/` |
| `linear_probe.py` | Frozen encoder + `sklearn` linear model, for small-data tasks | `configs/linear_probe/` |
| `evaluate.py` | The *final*, paper-comparable numbers for a fine-tuned checkpoint: bootstrap CIs, per-cohort breakdown, the 3-way pairwise AD-diagnosis AUC | `configs/eval/` |

`train.py` and `evaluate.py` are deliberately separate. `train.py`'s own end-of-epoch validation
metrics exist only to pick a checkpoint — they are not what the original repo reported. The real
numbers came from a second step: the original repo re-ran `main_finetune_final.py` in eval-only
mode to save raw predictions to `.npy`, then a notebook loaded those and computed bootstrap-CI'd,
per-cohort metrics by hand. `evaluate.py` does both of those steps as one script.
`linear_probe.py` reports its final number the same way (via `metrics.py`), since the original
repo applied bootstrap CIs uniformly regardless of adaptation strategy.

## Files

| File | Role |
|---|---|
| `config.py` | Dataclass config + YAML loader (`--set section.field=value` CLI overrides) |
| `model.py` | ViT encoder, classification/regression head, checkpoint loading, adaptation (`prepare_encoder`) |
| `lora.py` | LoRA adapters for the attention layers |
| `segmentation/` | UNETR (pretrained ViT + LoRA, or from scratch) and U-Net, BraTS/ATLAS training and Dice evaluation |
| `data.py` | CSV dataset + train/val/test transforms, shared by all three scripts |
| `engine.py` | Train/validate loop, loss, optimizer, LR schedule — used by `train.py` only |
| `metrics.py` | Bootstrap CI, the 3-way pairwise AD-diagnosis macro-AUC, regression metrics — used by `evaluate.py` and `linear_probe.py` |
| `train.py` | CLI: fine-tune, save the best checkpoint |
| `linear_probe.py` | CLI: frozen-encoder embeddings + `sklearn` linear model, bootstrap-CI'd |
| `evaluate.py` | CLI: final evaluation of a fine-tuned checkpoint |

## Data format

**Fine-tuning / eval configs** (`data.train_csv`/`val_csv`/`test_csv`) need a CSV with:
- `img_path_T1_mni152` — absolute path to a FreeSurfer-registered T1 volume in MNI152 space
  (`.nii.gz` or `.mgz`); configurable via `data.img_path_col`.
- `pred_label` — 0/1 for classification (AD diagnosis uses 0/0.5/1, see below), a continuous
  value for regression (age in years); configurable via `data.label_col`.
- For per-cohort breakdown in `evaluate.py`: a column like `source_dataset` naming which cohort
  each row came from, set as `data.cohort_col`.

**Linear-probing configs** work one of two ways (see `linear_probe.py`'s module docstring for
the full explanation):
- **Self-contained** (glioma): same `img_path_col`/`label_col` CSV shape as above, plus an
  optional `data.group_col` to average predictions per group (e.g. patient ID) before scoring.
- **Shared pool** (Parkinson's, epilepsy): `data.pool_csv` is a large pool of scans embedded
  once (`img_path_col`); each task's train/test CSV instead carries `data.join_col` (e.g.
  `subject_session`) + a label, matched against the pool by that key. `data.label_map`
  optionally collapses a categorical label column to 0/1.

## Tasks

### Fine-tuning (`configs/finetune/`) → final evaluation (`configs/eval/`)

| Task | Strategy | Train config | Eval config(s) |
|---|---|---|---|
| AD diagnosis | LoRA (FF variant: `ad_diagnosis_train_full_finetune.yaml`) | `ad_diagnosis_train.yaml` | `ad_diagnosis.yaml` — broad multi-cohort test set (ADNI internal + NACC/AIBL/OASIS1-4/MIRIAD external), per-cohort breakdown via `data.cohort_col: source_dataset` |
| Brain age | LoRA, regression (FF variant: `healthy_aging_train_full_finetune.yaml`) | `healthy_aging_train.yaml` | `healthy_aging.yaml` (internal UK Biobank), `healthy_aging_external.yaml` (external NACC, healthy subjects only) |
| Epilepsy detection | LoRA (FF variant: `epilepsy_train_full_finetune.yaml`) | `epilepsy_train.yaml` | `epilepsy_internal.yaml` (OpenNeuro), `epilepsy_episurg.yaml` (external EPISURG) |

Each eval config's `model.pretrained` points at the checkpoint its matching train config saved
(`checkpoints/<task>.pt`) and is loaded with `model.py::load_finetuned_checkpoint` — `evaluate.py`
never touches the raw ASP checkpoint directly.

**Not implemented: MCI progression.** The original repo evaluated the AD-diagnosis checkpoint a
second way, scored against a `time_CN_to_AD` survival/concordance-index analysis on a separate
progression cohort (OASIS 2-3). Dropped from this rebuild entirely (not just deprioritized) —
no progression configs, no concordance-index code.

### Linear probing (`configs/linear_probe/`)

| Task | Embeddings from | Config(s) |
|---|---|---|
| Glioma / IDH mutation (registered) | Self-contained (UCSF-PDGM, MNI152) | `glioma_registered.yaml` |
| Glioma / IDH mutation (raw) | Self-contained (UCSF-PDGM, unregistered) | `glioma_raw.yaml` |
| Parkinson's diagnosis | Shared pool (PPMI) | `parkinsons.yaml` |
| Epilepsy detection (internal) | Shared pool (OpenNeuro) | `epilepsy.yaml` |
| Epilepsy detection (external) | Shared pool (OpenNeuro → EPISURG) | `epilepsy_episurg.yaml` |

Epilepsy has both a fine-tuning and a linear-probing path — the original repo did both and
compared them. `parkinsons.yaml` and `epilepsy.yaml`/`epilepsy_episurg.yaml` all point
`data.pool_csv` at the same shared scan pool, so running one after another only re-embeds it if
you change which checkpoint (`model.pretrained`) is loaded — per task, only the (much cheaper)
`sklearn` fit changes.

**Not implemented: NIFD (frontotemporal dementia).** Existed in the original repo, but its own
notebooks evaluated it two different, inconsistent ways (a true 4-class multiclass one-vs-rest
in one notebook vs. a collapsed CON-vs-rest binary task in another) with no way to tell which was
actually used for the paper's reported numbers. Dropped rather than guessing.

## Running

Before running anything, edit each config's placeholder paths
(`/path/to/data/...`, `/path/to/checkpoints/asp_pretrained.pth`) to point at your own
preprocessed data and ASP checkpoint — none of this is distributed with the repo (see
`pretraining/README.md`).

```bash
# 1. Fine-tune
python -m downstream.train --cfg downstream/configs/finetune/ad_diagnosis_train.yaml

# 2. Evaluate that checkpoint -- the numbers that actually matter
python -m downstream.evaluate --cfg downstream/configs/eval/ad_diagnosis.yaml

# Linear probing (self-contained or shared-pool, same command either way)
python -m downstream.linear_probe --cfg downstream/configs/linear_probe/glioma_registered.yaml
python -m downstream.linear_probe --cfg downstream/configs/linear_probe/parkinsons.yaml

# Override any config field from the CLI without editing the YAML
python -m downstream.train --cfg downstream/configs/finetune/healthy_aging_train.yaml \
    --set train.max_epochs=5 --set data.batch_size=4
```

Runs single-GPU (`cuda` if available, else `cpu`) — no `torchrun`/multi-GPU support in this
rewrite; see the top-level README for why.

On a SLURM cluster, `jobs/` has a ready-to-submit script per config above (`sbatch
downstream/jobs/finetune/ad_diagnosis_train.sh`) — see `jobs/README.md`.

## Training details that matter for reproducing the original's numbers

- **`train.base_lr` is a reference value for batch size 256, not the LR actually used.**
  `train.py` calls `engine.py::scale_lr_for_batch_size` right after loading the config, which
  rescales it to `base_lr * data.batch_size / 256` and recomputes `train.min_lr` as
  `scaled_base_lr * 1e-3` (the YAML's own `min_lr` value is discarded) — this is exactly what
  `main_finetune_final.py` does right before building the optimizer. With the default
  `batch_size: 64`, the LR actually used is a quarter of the YAML's stated `base_lr`. If you
  change `data.batch_size`, the effective LR changes with it automatically; `train.py` prints the
  post-scaling values so you can see what's actually being used.
- **`batch_size: 64`** is the default for every fine-tuning/eval config, matching the original
  (linear-probe configs use `8`, also matching — those just extract embeddings, not train).
- **Gradient clipping only touches the encoder's gradients**, not the head's — matches
  `engine_finetune.py`'s `clip_grad_norm_(model.parameters(), ...)` call, which never included
  the classifier.
- **Epilepsy has no held-out validation split** — `configs/finetune/epilepsy_train.yaml` points
  `val_csv` at the same file as `train_csv`, because the original repo does too (the training
  cohort, n=82, was too small for a real split). Not best practice, but faithful to what the
  original numbers were actually produced from.

## Evaluation methodology

`metrics.py` ports three things from the original repo's analysis notebooks, applied uniformly
by `evaluate.py` and `linear_probe.py`:

- **Bootstrap confidence intervals** (`bootstrap_ci`) — 1000 resamples, 95% CI, percentile
  method. Matches `notebooks/confidence_interval.ipynb`'s `add_bootstrap_CI`, which is what backs
  every "difference is/isn't statistically significant" claim in the paper.
- **The 3-way pairwise AD-diagnosis macro-AUC** (`pairwise_ad_aucs`) — AD-vs-CN, AD-vs-MCI,
  MCI-vs-CN, each its own bootstrap-CI'd AUC, macro-averaged. Matches
  `notebooks/eval_vit_AD_results (1).ipynb` / `example_get_metrics_AD.ipynb`'s `calc_aucs`,
  **with one correctness fix**: the original filters the smoothed 2-column target on
  inconsistent thresholds (e.g. the "AD_CN" pair excludes `col0 == 0.5`, which are the *AD* rows,
  not the *MCI* rows the naming implies it should exclude) — working through the arithmetic, that
  leaves `y_true` taking values `{0, 0.25}` for a `roc_curve` call, which isn't a valid binary
  label and would raise `ValueError: Data is not binary` on any real evaluation that actually
  contains MCI cases. `pairwise_ad_aucs` filters directly on the raw 0/0.5/1 label per pair
  instead — unambiguous, and matches what the naming actually intends. Worth checking against
  your own reported numbers if they were produced by the original notebook.
- **Per-cohort stratified breakdown** — `evaluate.py` reports the overall metric plus one more
  per unique value of `data.cohort_col`, matching how `eval_vit_AD_results (1).ipynb` iterates
  `diagnosis.source_dataset.unique()` rather than scoring the pooled multi-cohort CSV as one set.

`regression_metrics` (Pearson correlation + MAE + MSE, each bootstrap-CI'd) matches
`notebooks/healthy_aging_evaluation.ipynb`.

## What's simplified vs. the original repo

This pipeline intentionally does not reproduce a few things from the original
`AI_FOR_STRUCTURAL_BRAIN_MRI/main_finetune_final.py` + `engine_finetune.py` + its notebooks:
- **Single-GPU only** — no DDP/`torchrun`/`DistributedSampler`/`SyncBatchNorm`.
- **No multi-patch aggregation** — every ASP downstream config used a single 128³ crop
  (`NUM_PATCHES: 1`), so the old patch-reshape/mean-or-concat-aggregation logic is dropped.
- **Segmentation lives in `segmentation/`**, separate from the classification/regression
  scripts, and LoRA is implemented in `lora.py` (`model.adaptation: lora`), the strategy behind
  the paper's reported results.
- **No MCI progression, no NIFD** — see the "Not implemented" notes above.
- **The classifier head's LR floor** scales with its 100x-higher base LR (one shared cosine
  schedule across both param groups) rather than sharing the encoder's absolute LR floor — see
  the comment in `engine.py::build_optimizer`.
