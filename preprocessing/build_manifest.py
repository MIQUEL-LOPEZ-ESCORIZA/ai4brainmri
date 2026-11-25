"""Step 3: build the scan manifest consumed by pretraining and downstream scripts.

Scans the output folder of `register_to_mni.sh` and writes one row per scan with the columns
the pipelines expect:

    subject_id, scan_id, source_dataset, img_path_T1_mni152, img_seg_path_T1,
    is_img_path_T1_mni152, is_aparc+aseg, fs_subject_dir

`pretraining/` (MAE and ASP) reads `img_path_T1_mni152` / `img_seg_path_T1` and the two `is_*`
validity flags (this replaces the shipped template `pretraining/files/sample_segmentation.csv`).
Clinical/demographic columns (age, sex, diagnosis labels) are merged in from a per-cohort table
with `--clinical`, joined on `scan_id`.

Usage:
    python -m preprocessing.build_manifest --mni-dir data/mni/UKBB --subjects-dir data/freesurfer/UKBB \
        --dataset UKBB --subject-regex '^(\\d+)_' --clinical data/clinical/ukbb.csv --out data/manifests/ukbb.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mni-dir", required=True)
    p.add_argument("--subjects-dir", required=True, help="FreeSurfer SUBJECTS_DIR of this cohort")
    p.add_argument("--dataset", required=True, help="cohort name written to source_dataset")
    p.add_argument("--subject-regex", default=r"^(.+?)_", help="extracts the participant ID from the scan ID")
    p.add_argument("--clinical", default="", help="optional CSV with a scan_id column")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    rows = []
    for img in sorted(Path(a.mni_dir).glob("*_T1_mni152.nii.gz")):
        scan_id = img.name.replace("_T1_mni152.nii.gz", "")
        seg = img.with_name(f"{scan_id}_aparc+aseg_mni152.nii.gz")
        m = re.match(a.subject_regex, scan_id)
        rows.append({
            "subject_id": m.group(1) if m else scan_id,
            "scan_id": scan_id,
            "source_dataset": a.dataset,
            "img_path_T1_mni152": str(img.resolve()),
            "img_seg_path_T1": str(seg.resolve()),
            "is_img_path_T1_mni152": img.exists(),
            "is_aparc+aseg": seg.exists(),
            "fs_subject_dir": str(Path(a.subjects_dir, scan_id).resolve()),
        })
    df = pd.DataFrame(rows)
    if a.clinical:
        df = df.merge(pd.read_csv(a.clinical), on="scan_id", how="left")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} scans from {df['subject_id'].nunique()} participants -> {a.out}")


if __name__ == "__main__":
    main()
