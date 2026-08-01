# SLURM job scripts

One `sbatch` script per config in `../configs/`, same naming (`<config-stem>.sh`), organized into
the same three folders (`finetune/`, `eval/`, `linear_probe/`) so the folder alone tells you which
of the three entry-point scripts a job runs. Mirrors the layout of the original repo's
`mae/configs/SEGMENTATION_MAE_SEG_COMBINED/jobs/`, adjusted for this repo's single-GPU, three-script
structure — the original had one shape of job (`torchrun ... main_finetune_final.py --cfg ...`)
for every task, since fine-tuning, eval, and linear probing were all the same script there.

| Folder | Runs | Example |
|---|---|---|
| `finetune/` | `python -m downstream.train` | `finetune/ad_diagnosis_train.sh` |
| `eval/` | `python -m downstream.evaluate` | `eval/ad_diagnosis.sh` |
| `linear_probe/` | `python -m downstream.linear_probe` | `linear_probe/glioma_registered.sh` |

## Submitting

Submit from the repo root, so the module path (`downstream.*`) and config path resolve
correctly:

```bash
sbatch downstream/jobs/finetune/ad_diagnosis_train.sh
```

Each script's `#SBATCH --output`/`--error` paths are repo-root-relative (`downstream/jobs/
<category>/slurm/%x-%j.out`), so this only works correctly if `sbatch` is invoked from the repo
root — not from inside the `jobs/` folder itself (unlike the original repo's scripts, which
expected the opposite).

## Before running

- Edit the referenced config's placeholder paths first (data CSVs, checkpoint) — same as running
  the script directly, see `../README.md`.
- The `conda activate` block assumes an env named `ai4brainmri` (`conda env create -f environment.yml`
  from the repo root) and a specific `conda.sh` path that's almost certainly wrong for your
  cluster — both are placeholders copied from the original repo's convention, not something this
  rebuild can generalize.
- Resource requests (`--cpus-per-task=4 --mem-per-cpu=100G --gres=gpu --time=3-00:00:0`) are
  copied as-is from the original repo's jobs, which used the same values uniformly across every
  task. Adjust for your own cluster/queue policy.

## Not carried over

- `--use_wandb` — the original always passed this; no wandb integration exists in this rebuild's
  scripts (see the top-level README's "What's simplified" notes elsewhere).
- `torchrun`/multi-GPU launch — single process, plain `python -m ...`, consistent with the rest
  of this repo being single-GPU only.
