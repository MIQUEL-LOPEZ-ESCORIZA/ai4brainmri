# Pretraining

`downstream/` only *consumes* a checkpoint produced by the stages described below. Point
`MODEL.PRETRAINED` in a downstream config at your own checkpoint to use one.

**Both stages are implemented.** Stage 2 (ASP, `seg_mae_combined_train.py`) is ported as-is from
the original `SEGMENTATION` cluster repo — see "ASP pretraining — implementation details" below
for what was copied, what was dropped, and how it maps to the paper. Stage 1 (plain MAE,
`mae_train.py`) has no original implementation to port from: that code runs in a separate cluster
codebase (internally referred to as `BrainATLAS/mae`) this project doesn't have access to, so it
was written from scratch, mirroring the shape of `seg_mae_combined_train.py`'s MAE half
(`build_mae`, `train_one_epoch_mae`, `val_one_epoch_mae`) but simplified — see "MAE pretraining
(stage 1) — implementation details" below for exactly what that means.

## Stage 1 — Masked Autoencoder (MAE) pretraining

A ViT-Base encoder is pretrained via masked autoencoding on a large general-population cohort
(69,878 scans / 65,249 subjects, primarily UK Biobank).

- 75% of 3D patches randomly masked; model reconstructs the full volume from the rest.
- Input: single-channel, intensity-scaled to [0, 1], padded to 256³, randomly cropped to 128³,
  random flip + intensity shift.
- Loss: MSE reconstruction.
- 1000 epochs, batch size 32/GPU, AdamW (β1=0.9, β2=0.95, weight decay=0.005), cosine LR
  schedule, base LR 1.5e-3, linear warmup over the first 2.5% of steps.
- Convergence around epoch 900 (~10 days on two 80GB A100s in the original run).

**Implemented** — see `mae_train.py` and "MAE pretraining (stage 1) — implementation details"
below for exactly how the config maps to these numbers and where this repo's version had to make
a judgment call the paper doesn't specify.

## Stage 2 — Anatomy Segmentation Pretraining (ASP)

**Implemented** — see below. Starting from the MAE-pretrained encoder, a lightweight
**bottleneck segmentation decoder** is added (no skip connections, deliberately — the goal is to
force anatomical information into the ViT embedding itself, not into decoder skip connections)
and trained to segment 39 brain regions, selected for disease relevance and consolidated across
symmetric/related structures (`files/selected_brain_regions.csv`).

Because predicting all 39 regions as one softmax volume would be memory-prohibitive
(39×128×128×128 logits), ASP uses a **prompt-based** decoder instead (inspired by Segment
Anything Model-style prompting): a random brain-region prompt is summed into the ViT embedding
before decoding, and the decoder outputs a binary segmentation mask for that one region.

Two models sharing one encoder, trained in **alternating epochs**: each epoch, a
`PromptSegmentationModel` (ViT + `ConvDecoder`) trains for one full epoch on the segmentation
loss, its encoder weights are copied into a separate `MaskedAutoencoderViT`, that trains for one
full epoch on MAE reconstruction, and its (further-updated) encoder is copied back — so the
segmentation and MAE objectives never touch the same forward pass, but continually re-synchronize
through the shared encoder. See "How this actually works" below for the exact mechanism and
file-by-file map.

- Loss: binary cross-entropy (segmentation), MSE (MAE reconstruction).
- Paper Methods states: 400 epochs, batch size 16/GPU, AdamW (β1=0.9, β2=0.95, weight decay=0.04),
  base LR 5e-4, cosine schedule with 5-epoch warmup, ~1 month on a single 42GB RTX8000. The
  config actually checked into the ported code (`configs/mae_seg_combined_train.yaml`)
  doesn't match this (`BASE_LR: 1.5e-8`, `MAX_EPOCHS: 1000`, `WARMUP_EPOCHS: 0`) — see the
  hyperparameter discrepancy note below; it's most likely a stale config, not what actually
  produced the published checkpoint.
- Mask ratio for the MAE half of this stage is **0.4** in the ported config, not the 0.75 used
  for stage-1 MAE pretraining above.

The checkpoint this stage produces is what `downstream/` calls the "ASP checkpoint" — it
contains encoder weights (`model_mae` sub-key convention, see `downstream/model.py`) shaped
by both objectives.

---

# MAE pretraining (stage 1) — implementation details

`mae_train.py` is new code, not a port — see the top of this file for why. It's written to reuse
as much of the ASP stage's own infrastructure as possible (`src/data/config.py`'s config system,
`src/utils/utils_seg.py`'s checkpoint resume / cosine schedule / seed helpers, `src/models/mae.py`'s
`MaskedAutoencoderViT`, `src/utils/optimizers.py`'s `get_optimizer`) and to mirror the *shape* of
`seg_mae_combined_train.py`'s MAE half (`build_mae`, `train_one_epoch_mae`, `val_one_epoch_mae`),
just without the alternating segmentation half or the DDP wrapping.

## Deliberate simplifications vs. `seg_mae_combined_train.py`

- **Single-process, no DDP/`torchrun`.** `seg_mae_combined_train.py` stays DDP-based because it's
  a faithful port of code that has to match what actually produced the published checkpoints.
  There's no equivalent original for stage 1 to be faithful to, so this uses the simpler of the
  two styles in this repo (matching `downstream/`'s single-GPU rewrite) rather than reproducing
  DDP boilerplate that has nothing to stay consistent with. `src/utils/utils_seg.py`'s
  `get_world_size()`/`get_rank()`/`save_on_master()`/`restart_from_checkpoint()`/
  `cosine_scheduler()` all already degrade to single-process behavior when
  `torch.distributed` was never initialized, so no DDP-specific fork of those was needed.
- **No runtime batch-size LR rescaling.** `seg_mae_combined_train.py::main()` treats its YAML's
  `BASE_LR` as a reference value for batch size 256 and rescales it by `× eff_batch_size / 256` at
  runtime (see the ASP section below) — one of the things that made that config's actual
  effective LR hard to pin down. `mae_train.py` skips that: `configs/mae_train.yaml`'s
  `BASE_LR`/`MIN_LR` are used exactly as written, and the default `BATCH_SIZE: 32` already matches
  the paper's stated "batch size 32/GPU" directly, so no rescaling is needed for the default
  config to reproduce the paper's stated LR.
- **`get_mae_loaders()` is called once**, before the epoch loop, not once per epoch. (
  `seg_mae_combined_train.py::train_loop()` calls `get_loaders(config)` again at the top of every
  epoch, which — since `ImageDataset.__init__` clears and rebuilds the `PersistentDataset` cache
  directory on construction — re-wipes the on-disk cache every epoch. Left as-is there per "don't
  change the algorithm"; not reproduced here since this script has no such constraint.)
- **CPU fallback for local smoke-testing** (`device = "cuda" if available else "cpu"`, matching
  `downstream/`'s convention) — real runs need a GPU regardless; this only matters for e.g.
  `TRAIN.MAX_EPOCHS: 1`-style dry runs. `USE_AMP` should be set `false` in the config if actually
  testing on CPU, since `torch.cuda.amp` isn't meaningful there.

## Data

No segmentation labels needed — `extract_image_datalist()` / `image_only_transform()` /
`MAEImageDataset` / `get_mae_loaders()` (`src/data/dataloaders.py`) are a separate, simpler path
alongside the segmentation-oriented `get_loaders()` the ASP stage uses, rather than that same
path with the mask/group machinery just discarded. `configs/mae_train.yaml`'s `DATA.TRAIN_CSV_PATH`
points at the same shipped `pretraining/files/sample_segmentation.csv` the ASP config uses, since
it already has the one column this stage actually needs (`img_path_T1_mni152`) — the segmentation
columns in that file are simply ignored here. The one augmentation this stage's transform pipeline
has that the ASP stage's `image_label_transform_total()` doesn't is `RandShiftIntensityd`, since
the paper's Methods section specifically calls out "random flip + intensity shift" for stage 1;
the exact shift magnitude isn't stated, so `offsets=0.1` is a modest, commonly-used default, not
a value taken from the paper.

## Checkpoint format

`save_checkpoint()` writes two different files with two different purposes, deliberately not
sharing one format the way `seg_mae_combined_train.py` does (it can get away with one format
because its models are DDP-wrapped, so their own `state_dict()` already carries the `module.`
prefix everything downstream expects):

- **`checkpoint.pth`** (resume) — plain, unprefixed `model`/`optimizer` keys, read back by
  `restart_from_checkpoint()` directly into this same non-DDP model/optimizer on the next launch.
- **`checkpoint_epoch_XXXX.pt`** (periodic, one per `TRAIN.SAVECKP_FREQ` epochs) — a `state_dict`
  key holding the encoder's weights with a `module.` prefix added by hand, purely so this stays
  loadable, unmodified, by two existing consumers that expect that prefix: `build_mae()`/
  `build_seg()`'s `MODEL.MAE_PRETRAINED`/`SSL_PRETRAINED` loading in `seg_mae_combined_train.py`,
  and `--checkpoint_source mae` in `linear_probing/linear_probe_sweep.py` (the Figure 4(b) sweep
  — see below). The `.pt` extension and `epoch_XXXX` naming both match what that sweep script's
  `mae` checkpoint source already expects to find.

## Hyperparameters not stated by the paper

`configs/mae_train.yaml` is annotated inline at each field, summarized here: `VAL_EVERY` /
`SAVECKP_FREQ: 10` (every epoch, as the ASP config does, would mean 1000 full checkpoints over
this run; every 10th still gives 100 points for the linear-probing sweep) and `GRAD_CLIP: 3.0`
(inherited from the ASP config's default) are both this repo's choices, not values taken from the
paper. `WARMUP_EPOCHS: 25` is derived (2.5% of `MAX_EPOCHS: 1000`), not stated directly.

## Linear probing during this stage

`linear_probing/linear_probe_sweep.py --checkpoint_source mae` sweeps `checkpoint_epoch_XXXX.pt`
snapshots from this script exactly the same way it sweeps ASP's `checkpoint_seg_XXXX.pth`
snapshots with `--checkpoint_source asp` — see "linear_probing/ — the Figure 4(b) mechanism"
below. `jobs/val_job_mae.sh` still has a placeholder `--checkpoints_dir`; point it at
`configs/mae_train.yaml`'s `MODEL.CHECKPOINT_DIR` (`checkpoints/mae` by default) once you have a
real run.

---

# ASP pretraining — implementation details

Ported from the original `SEGMENTATION` cluster repo in three passes: a pure structural move
(files relocated, only the imports that broke because of the move touched), a pass that made it
actually self-contained and runnable (hardcoded cluster paths replaced with config fields or
portable relative paths, one real bug fixed, dead code removed), and a cleanup pass that dropped
unused evaluation/analysis scripts, merged two near-duplicate ones, and removed a GPU busy-loop
warm-up script. **The ASP algorithm itself is untouched throughout**: the alternating
segmentation/MAE training loop, `update_model_weights`, the model architecture (`ConvDecoder`,
`PromptSegmentationModel`), and the region-prompt sampling are exactly what the original repo
does. If you're looking for the "rebuilt from scratch, simplified, single-GPU" style used
elsewhere in this repo, this folder is the deliberate exception — `downstream/` is that style,
this is a faithfully-behaving port, not a rewrite.

## Where things live now vs. the original repo

| This repo | Original `SEGMENTATION` repo | Why |
|---|---|---|
| `src/` (repo root) | `SEGMENTATION/src/` | Shared library (models, data, utils) — hoisted to the top level, not nested inside `pretraining/`, so it's available to any future pipeline that wants it |
| `src/utils/utils_seg.py` | `SEGMENTATION/utils_seg.py` | Same reasoning — `src/data/dataloaders.py` imports this directly (`from src.utils import utils_seg as utils`), so it lives alongside the rest of the shared library rather than at the repo root on its own |
| `pretraining/seg_mae_combined_train.py` | `SEGMENTATION/seg_mae_combined_train.py` | The actual ASP training script |
| `pretraining/configs/`, `jobs/`, `files/` | same names, `SEGMENTATION/{configs,jobs,files}/` | |
| `pretraining/linear_probing/linear_probe_sweep.py` | `SEGMENTATION/linear_probing/{validation.py,validation_mae_seg_combined.py}` | Merged — see below |

**No longer here at all** (present after the initial port, removed since): `iou_finetuned_model.py`
and its config/job script, `linear_probing/validation_finetuned_model.py` and its job script,
`linear_probing/read_and_plot.ipynb`, `mat_mul.py` (both copies). See "What was removed" below.

## What was changed, and why

**Pass 1 (structural move only)**:
1. `iou_finetuned_model.py` (since removed): `from seg_mae_combined_train import
   val_one_epoch_seg` → an absolute import, required once it moved into this repo's package
   layout.
2. Added `__init__.py` to `pretraining/` and `pretraining/linear_probing/` so
   `python -m pretraining....` resolves as a package.

**Pass 2 (make it actually self-contained and runnable)**, each marked inline with
`# ---- CHANGED ----`:
3. **Real bug** (in the now-removed `iou_finetuned_model.py`): `wandb_run` was only assigned
   inside `if config.WANDB.WANDB_ENABLE and dist.get_rank() == 0:`, then used unconditionally
   right after — `NameError` whenever wandb was disabled or rank != 0.
4. **Hardcoded paths that duplicated or ignored existing config/CLI plumbing** — the now-removed
   `iou_finetuned_model.py::build_model()` loaded the encoder from a hardcoded path instead of
   `config.MODEL.SSL_PRETRAINED`; the now-removed `validation_finetuned_model.py::main()`
   hardcoded a checkpoint path even though `--checkpoints_dir` was already a parsed CLI argument.
5. **Four dead `sys.path.append("/scratch/ml9715/...")` lines removed** (`src/data/dataloaders.py`
   and the linear-probing scripts — one pointed at `/scratch/ml9715/CLASSIFIER/mae`, a *different*
   repo entirely). Unnecessary once everything runs via `python -m pretraining....` from the
   repo root, which already puts the root on `sys.path`.
6. **`seg_mae_combined_train.py::create_region_bar_plot()`** resolved the region CSV via a
   hardcoded cluster path; now resolves it relative to the file's own location
   (`Path(__file__).resolve().parent / "files" / "selected_brain_regions.csv"`), which is a real,
   portable path since that CSV ships with the repo.
7. **Configs**: absolute cluster paths replaced with the `/path/to/...` placeholder convention
   used everywhere else in this repo. `DATA.TRAIN_CSV_PATH` is the exception — it points at the
   real shipped `pretraining/files/sample_segmentation.csv` by default, with a comment that
   the manifest's own image-path columns still need regenerating for your data. `MODEL.SEG_PRETRAINED`
   and `OUTPUT` are annotated in place as **not read by any code path** (see "How this actually
   works" below) rather than deleted, since they reflect real original intent even if unwired.
8. **Job scripts rewritten** to the same convention as `downstream/jobs/`: submit from the
   repo root, `#SBATCH --output`/`--error` under `.../jobs/slurm/`, `conda activate ai4brainmri`.
   `seg_mae_combined_train.py` still goes through `torchrun` (unlike `downstream/`'s job
   scripts — see below for why), just now via `-m pretraining.<module>` instead of an
   absolute script path. `linear_probing/jobs/val_job_*.sh` drop `torchrun` entirely and call
   `python -m` directly, since the linear-probing scripts never touch `torch.distributed` at all.

**Pass 3 (remove unused, merge duplicates)** — see "What was removed" and "linear_probing" below.

**Deliberately not changed** (confirmed with the user before Pass 2): the `TRAIN.*`
hyperparameter *values* in `configs/mae_seg_combined_train.yaml` — see the hyperparameter
discrepancy note below and the inline YAML comment. `freeze_encoder`/`unfreeze_encoder`
(defined, still unused) — not confident enough it's dead-by-mistake vs. dead-by-designed future
use to touch it.

## What was removed

Present after the initial structural port, removed once confirmed unused/non-functional:

- **`iou_finetuned_model.py`** (+ `configs/iou_finetuned.yaml`, `jobs/iou_finetuned.sh`) — eval-only
  script producing the paper's Figure 4(c) per-region IoU numbers: loaded a pure-MAE encoder + one
  hardcoded decoder-checkpoint epoch, ran one validation pass. Nothing else imported it (it only
  imported *from* `seg_mae_combined_train.py`), so removing it doesn't affect training.
- **`linear_probing/validation_finetuned_model.py`** (+ `linear_probing/jobs/val_job_finetuned.sh`) —
  probed one specific, already downstream-fine-tuned checkpoint (from the `CLASSIFIER`/
  `downstream` pipeline, not a raw pretraining snapshot) as a single reference point against the
  pretraining-epoch sweep the other two scripts produced.
- **`linear_probing/read_and_plot.ipynb`** — was already non-functional in this repo before
  removal: it doesn't read any structured results file, it regexes a raw SLURM `.out` stdout log
  for `"Validating epoch: N"` markers and printed metric dicts (hardcoded `LOG_PATH` pointing at
  a specific old log file on the original cluster). Those `.out` files are exactly the kind of
  runtime artifact this repo's `.gitignore` deliberately excludes, so the notebook had nothing to
  read even immediately after the port.
- **`mat_mul.py`** (both the `pretraining/` and `linear_probing/` copies, byte-identical) — an
  infinite `torch.bmm` busy-loop launched in the background (`&`) at the top of every job script
  purely to force the SLURM allocation's GPU to spin up / stay warm before the real training
  process started. Cluster-specific workaround, not part of the ASP algorithm; dropped along with
  the `python .../mat_mul.py &` line in each job script.
- **`SEGMENTATION/engine_pretrain.py`** (never copied in the first place) — dead code in the
  original repo. Nothing imports it, and it can't run if something did:
  `trainer()`/`tester()` call `train_one_epoch(...)`/`val_one_epoch(...)`, but the only functions
  actually defined in that file are `train_one_epoch_mae`/`val_one_epoch_mae` — a `NameError`
  waiting to happen.
- **`linear_probing/results/`, `wandb/`, `logs/`, `cache_dataset/`, `jobs/slurm/*.out`/`.err`** —
  run outputs and runtime artifacts, never copied.

## How this actually works (verified by reading the code, not just the paper)

`seg_mae_combined_train.py` builds **two separate DDP-wrapped models that share an encoder by
copying weights back and forth**, not one model with two loss heads:
- `seg_net = PromptSegmentationModel(ViT encoder, ConvDecoder)` — `src/models/model.py`.
  `ConvDecoder` sums the ViT's 768-dim CLS embedding with a learned per-region embedding (the
  "prompt"), reshapes to a 1×1×1×768 bottleneck, then 6 transposed-conv layers straight up to
  128³×1 — no skip connections.
- `mae_net = MaskedAutoencoderViT` — plain MAE reconstruction.
- `train_loop()`, per epoch: train `seg_net` one full epoch → `update_model_weights(models,
  "mae", ...)` copies the encoder from `seg_net` into `mae_net` → train `mae_net` one full epoch
  → validate both, checkpoint → `update_model_weights(models, "seg", ...)` copies the encoder
  back into `seg_net` for the next epoch.
- Still fully DDP-based (`torch.nn.parallel.DistributedDataParallel`, `utils.init_distributed_mode`)
  — this is why its job script still needs `torchrun` even for a single GPU, unlike
  `downstream/`, which was rewritten to drop DDP entirely.
- `src/data/dataloaders.py::grouping_total()` hardcodes the 39 region groups (FreeSurfer label
  IDs, symmetric L/R pairs merged — matches `files/selected_brain_regions.csv` exactly).
  `TargetMaskTransform_grouped` picks one of the 39 groups **at random per sample** at
  data-loading time — this is the region-prompt sampling.
- Auto-resumes from `MODEL.CHECKPOINT_DIR/checkpoint.pth` if present (DINO-style
  `restart_from_checkpoint`, from `src/utils/utils_seg.py`).
- Checkpoints every `TRAIN.SAVECKP_FREQ` epochs to `MODEL.CHECKPOINT_DIR/checkpoint_seg_XXXX.pth`
  — this is what `linear_probing/linear_probe_sweep.py` later sweeps over.

**Phases and configs.**

- **Frozen-encoder phase.** `TRAIN.FREEZE_ENCODER_EPOCHS` runs that many decoder-only epochs on
  the frozen MAE encoder before joint training starts; no MAE epochs run during this phase.
  `TRAIN.ALTERNATE_MAE: False` turns off the MAE epochs afterwards (an ablation). The defaults
  (`0` / `True`) leave the original loop unchanged, so `configs/mae_seg_combined_train.yaml` still
  behaves as before. `MODEL.SEG_PRETRAINED` can instead start from the checkpoint of a separate
  decoder-only run.
- **`configs/asp_paper.yaml`** carries the hyperparameters stated in the paper: 400 epochs, batch
  size 16, AdamW (0.9, 0.95), weight decay 0.04, base LR 5e-4 with cosine annealing and 5 warm-up
  epochs, and a 100-epoch frozen phase. `configs/asp_ablation_*.yaml` are the Supplementary
  ablations: no MAE alternation; no alternation and no freezing; ASP from a randomly initialized
  encoder. The "MAE only" variant is the stage-1 checkpoint itself. `jobs/asp_paper.sh <config>`
  runs any of them.
- **`configs/mae_seg_combined_train.yaml`** is the config of the original repository's last run,
  kept for reference. Its learning rate (1.5e-8) and epoch count do not match the paper's run;
  use `asp_paper.yaml` to reproduce the paper.
- **Mask ratio.** The MAE epochs inside ASP use a 0.4 mask ratio (`MAE.MASK_RATIO`); stage-1 MAE
  pretraining uses 0.75.

## `linear_probing/` — the Figure 4(b) mechanism

Separate from `downstream/linear_probe.py` (which linear-probes a *finished* checkpoint on
clinical task labels). This one monitors representation quality *during* pretraining: given a
directory of checkpoint snapshots `seg_mae_combined_train.py` already saved, it sweeps over every
qualifying epoch, extracts CLS embeddings for a train/test split, fits a fresh
`sklearn` `LogisticRegression` (for `gender`) / `Ridge` (for continuous features — age, cortex
volume, etc.) per feature at each epoch, and plots the trend — this is the mechanism behind the
paper's claim that embeddings "gradually capture anatomical information" over training. One run
of the script produces the whole curve, not one point.

**`linear_probe_sweep.py`** is a merge of the original repo's `validation.py` and
`validation_mae_seg_combined.py`, which were ~500-line near-duplicates differing only in which
checkpoint files they'd scan for and which key of the checkpoint dict held the encoder weights.
Everything else — data loading, transforms, feature extraction, the sklearn probes, the output
plot — was already byte-identical between the two and is unchanged here. The one real difference
is now a `--checkpoint_source {mae,asp}` flag, backed by a small `CHECKPOINT_SOURCES` table in
the script:

| `--checkpoint_source` | Was | Matches files | State dict key | Epoch filter |
|---|---|---|---|---|
| `mae` | `validation.py` | `*.pt` | `state_dict` | every epoch past `--cut_off_epoch` |
| `asp` | `validation_mae_seg_combined.py` | `*seg*.pth` | `model_mae` | every 5th epoch past `--cut_off_epoch` |

`validation_mae_seg_combined.py` also had a second, unused loader
(`load_model_intermidiate`, using `PromptSegmentationModelIntermidiateHippo` to probe an
intermediate decoder activation instead of the CLS embedding) — defined but never called from its
own `main()`, so it was dead code in the original too and wasn't carried over. If you want that
capability back, it would slot in as a third `CHECKPOINT_SOURCES` entry.

Not training — this is entirely decoupled from `seg_mae_combined_train.py`, run as its own
(non-distributed) job pointed at whatever checkpoints already exist on disk.

## Running

Same placeholder-path convention as the rest of this repo — edit every `/path/to/...` in the
configs and job scripts to point at your own data/checkpoints before running anything.
`DATA.TRAIN_CSV_PATH` already points at the real, shipped
`pretraining/files/sample_segmentation.csv` — regenerate its image-path columns for your
environment rather than replacing the file wholesale.

```bash
# Stage 1: MAE pretraining -- single process, no torchrun
python -m pretraining.mae_train --cfg pretraining/configs/mae_train.yaml

# Stage 2: ASP training (alternating seg/MAE epochs) -- still DDP, hence torchrun even for one GPU
torchrun --nproc_per_node=1 --master_port=12001 -m pretraining.seg_mae_combined_train \
    --cfg pretraining/configs/mae_seg_combined_train.yaml --local_rank 0

# Linear-probing sweep over already-saved checkpoints (see --help for the full argument list) --
# plain python, no torchrun needed, same script for either stage's checkpoints
python -m pretraining.linear_probing.linear_probe_sweep \
    --checkpoint_source mae \
    --train_dataset_csv ... --test_dataset_csv ... --features_to_validate age gender CortexVol \
    --checkpoints_dir ... --vit_params '{...}' --output_dir ... --cut_off_epoch 0
```

On a SLURM cluster: `sbatch pretraining/jobs/mae_train.sh`, `sbatch
pretraining/jobs/mae_seg_combined_train.sh` (and the `linear_probing/jobs/val_job_*.sh`
siblings), submitted from the repo root — see each script's own header comment for the "adjust
for your cluster" caveats (conda path, resource requests carried over unchanged from the
original, except `mae_train.sh`'s which are this repo's own estimate).
