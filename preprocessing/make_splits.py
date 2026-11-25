"""Group-level train/validation/test splits (no participant -- or site -- in two splits).

Every split in the paper is made at the participant level: all scans of one participant go to
the same split, which matters for the longitudinal cohorts (ADNI, NACC, OASIS-2/3, MIRIAD, PPMI,
UCSF-PDGM with T1 + T1-CE, BraTS with T1 + T1-CE). ATLAS R2.0 is split by acquisition site
(`--group-col site`), which is stricter. Optionally stratifies on a label column so class
proportions are similar across splits.

Paper splits (see data/README.md for the full table):
    ADNI (AD)        80% / 20% participants, 10% of training participants held out for validation
    PPMI             80% / 20% participants
    UCSF-PDGM        80% / 20% participants
    OpenNeuro        50% / 50% participants
    BraTS 2023       1,000 / 251 participants (fixed counts: --n-test 251)
    ATLAS R2.0       21 / 12 sites (--group-col site --n-test 12)
    Brain age        UKBB holdout = 10% of UKBB participants (also excluded from MAE/ASP training)

Usage:
    python -m preprocessing.make_splits --manifest data/manifests/adni.csv --group-col subject_id \
        --test-frac 0.2 --val-frac 0.1 --stratify-col pred_label --seed 42 --out-prefix data/splits/ADNI
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def split_groups(groups: pd.Series, labels: pd.Series | None, frac: float | None, n: int | None,
                 rng: np.random.RandomState) -> set:
    """Pick test groups, stratified by each group's most frequent label if `labels` is given."""
    unique = groups.drop_duplicates()
    if labels is None:
        strata = {0: unique.tolist()}
    else:
        per_group = labels.groupby(groups).agg(lambda s: s.mode().iloc[0])
        strata = {k: v.index.tolist() for k, v in per_group.groupby(per_group)}
    total = len(unique)
    picked = set()
    for members in strata.values():
        members = list(members)
        rng.shuffle(members)
        k = round(len(members) * (frac if frac is not None else n / total))
        picked.update(members[:k])
    return picked


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--group-col", default="subject_id")
    p.add_argument("--stratify-col", default="")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--test-frac", type=float)
    g.add_argument("--n-test", type=int, help="number of test groups (e.g. 251 BraTS participants, 12 ATLAS sites)")
    p.add_argument("--val-frac", type=float, default=0.0, help="fraction of *training* groups held out for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-prefix", required=True)
    a = p.parse_args()

    df = pd.read_csv(a.manifest)
    rng = np.random.RandomState(a.seed)
    labels = df[a.stratify_col] if a.stratify_col else None
    test_groups = split_groups(df[a.group_col], labels, a.test_frac, a.n_test, rng)
    test = df[df[a.group_col].isin(test_groups)]
    train = df[~df[a.group_col].isin(test_groups)]
    splits = {"train": train, "test": test}
    if a.val_frac > 0:
        val_groups = split_groups(train[a.group_col], train[a.stratify_col] if a.stratify_col else None,
                                  a.val_frac, None, rng)
        splits["val"] = train[train[a.group_col].isin(val_groups)]
        splits["train"] = train[~train[a.group_col].isin(val_groups)]

    names = list(splits)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            overlap = set(splits[x][a.group_col]) & set(splits[y][a.group_col])
            assert not overlap, f"{a.group_col} overlap between {x} and {y}: {sorted(overlap)[:5]}"

    Path(a.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(f"{a.out_prefix}_{name}.csv", index=False)
        print(f"{name}: {len(part)} scans, {part[a.group_col].nunique()} {a.group_col} values")


if __name__ == "__main__":
    main()
