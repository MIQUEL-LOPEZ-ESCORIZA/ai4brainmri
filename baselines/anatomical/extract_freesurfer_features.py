"""Collect FreeSurfer `recon-all` statistics into one feature table (one row per scan).

Input: a manifest CSV with one row per scan and a column pointing at that scan's FreeSurfer
subject directory (the folder containing `stats/aseg.stats`, `stats/lh.aparc.stats`,
`stats/rh.aparc.stats`), as produced by `preprocessing/run_recon_all.sh`.

Output: the manifest with one extra column per anatomical feature. Feature names follow the
paper's Supplementary Table (see `features_paper.txt`):

- `<Structure>_Volume_mm3`        aseg.stats structure volumes (e.g. Left-Hippocampus_Volume_mm3)
- global measures                  aseg.stats `# Measure` lines (BrainSegVol, CortexVol, TotalGrayVol,
                                   lhCortexVol, rhCortexVol, CerebralWhiteMatterVol, ...)
- `<region>_GrayVol`               aparc GrayVol, summed over hemispheres
- `<region>_SurfArea`              aparc SurfArea, summed over hemispheres
- `<region>_ThickAvg`              aparc ThickAvg, surface-area-weighted mean over hemispheres
- `MeanThickness`, `lh/rhCortexThickness`, `CortexThickness`, `NumVert`, `lh/rhCortexArea`,
  `SurfaceHoles` (from aparc `# Measure` lines and aseg `SurfaceHoles`)

Run with `--features baselines/anatomical/features_paper.txt` to keep exactly the paper's feature
set; features that cannot be derived are reported (and the script fails unless
`--allow-missing`).

Usage:
    python -m baselines.anatomical.extract_freesurfer_features \
        --manifest data/manifests/adni.csv --subject-dir-col fs_subject_dir \
        --features baselines/anatomical/features_paper.txt --out data/features/adni_anatomical.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

_MEASURE = re.compile(r"^# Measure (\S+), (\S+), .*?, ([-\d.eE+]+), \S+")


def _read_measures(path: Path) -> dict[str, float]:
    """`# Measure <structure>, <name>, ...` lines, keyed both as `<name>` (e.g. BrainSegVol) and
    as `<structure>_<name>` (e.g. Cortex_MeanThickness in aparc.stats)."""
    out = {}
    for line in path.read_text().splitlines():
        m = _MEASURE.match(line)
        if m:
            structure, name, value = m.groups()
            out[name] = float(value)
            out[f"{structure}_{name}"] = float(value)
    return out


def _read_table(path: Path) -> pd.DataFrame:
    """Parse the whitespace table at the end of a FreeSurfer .stats file using its ColHeaders line."""
    lines = path.read_text().splitlines()
    header = next(l for l in lines if l.startswith("# ColHeaders")).split()[2:]
    rows = [l.split() for l in lines if l and not l.startswith("#")]
    return pd.DataFrame(rows, columns=header)


def features_for_subject(subject_dir: str) -> dict[str, float]:
    stats = Path(subject_dir) / "stats"
    feats: dict[str, float] = {}

    aseg = stats / "aseg.stats"
    feats.update({k: v for k, v in _read_measures(aseg).items() if "_" not in k})
    table = _read_table(aseg)
    for _, row in table.iterrows():
        feats[f"{row['StructName']}_Volume_mm3"] = float(row["Volume_mm3"])

    per_hemi = {}
    for hemi in ("lh", "rh"):
        path = stats / f"{hemi}.aparc.stats"
        measures = _read_measures(path)
        feats[f"{hemi}CortexThickness"] = measures.get("Cortex_MeanThickness", float("nan"))
        feats[f"{hemi}CortexArea"] = measures.get("Cortex_WhiteSurfArea", float("nan"))
        feats[f"{hemi}NumVert"] = measures.get("Cortex_NumVert", float("nan"))
        t = _read_table(path).set_index("StructName")
        per_hemi[hemi] = t[["GrayVol", "SurfArea", "ThickAvg"]].astype(float)

    lh, rh = per_hemi["lh"], per_hemi["rh"]
    for region in lh.index.intersection(rh.index):
        area_l, area_r = lh.at[region, "SurfArea"], rh.at[region, "SurfArea"]
        feats[f"{region}_GrayVol"] = lh.at[region, "GrayVol"] + rh.at[region, "GrayVol"]
        feats[f"{region}_SurfArea"] = area_l + area_r
        feats[f"{region}_ThickAvg"] = (
            (lh.at[region, "ThickAvg"] * area_l + rh.at[region, "ThickAvg"] * area_r) / (area_l + area_r)
        )

    area_l, area_r = feats["lhCortexArea"], feats["rhCortexArea"]
    feats["CortexThickness"] = (feats["lhCortexThickness"] * area_l + feats["rhCortexThickness"] * area_r) / (area_l + area_r)
    feats["MeanThickness"] = feats["CortexThickness"]
    feats["NumVert"] = feats["lhNumVert"] + feats["rhNumVert"]
    # TODO(paper authors): the paper's table also lists `<region>_CortexThickness` next to
    # `<region>_ThickAvg` for 15 regions. Their exact definition in the original feature
    # extraction is not recoverable from the paper; they are left unmapped (reported as missing)
    # until the original definition is filled in here.
    return feats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--subject-dir-col", default="fs_subject_dir")
    parser.add_argument("--features", default="", help="text file with one feature name per line ('#' = comment)")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    rows, failed = [], []
    for subject_dir in manifest[args.subject_dir_col]:
        try:
            rows.append(features_for_subject(subject_dir))
        except (FileNotFoundError, StopIteration, KeyError) as err:
            # recon-all failures are excluded from the study (paper Methods, "Data preprocessing")
            failed.append((subject_dir, repr(err)))
            rows.append({})
    feats = pd.DataFrame(rows)

    if args.features:
        wanted = [l.strip() for l in Path(args.features).read_text().splitlines() if l.strip() and not l.startswith("#")]
        missing = [f for f in wanted if f not in feats.columns]
        if missing:
            print(f"{len(missing)} requested features could not be derived: {missing}")
            if not args.allow_missing:
                raise SystemExit("re-run with --allow-missing to write the table without them")
        feats = feats.reindex(columns=[f for f in wanted if f in feats.columns])

    out = pd.concat([manifest.reset_index(drop=True), feats], axis=1)
    out["freesurfer_ok"] = [bool(r) for r in rows]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows x {feats.shape[1]} features to {args.out} ({len(failed)} failed FreeSurfer runs)")


if __name__ == "__main__":
    main()
