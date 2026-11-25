# Baselines

These are the non-ViT models of the paper's comparison. Every script writes per-scan predictions
in the shared schema, so its results can be compared with the ViTs using
`analysis/compare_models.py`.

## Anatomical features (`anatomical/`)

1. `extract_freesurfer_features.py` turns `recon-all` statistics into one feature table per
   cohort. `features_paper.txt` lists the 178 features of the paper's Supplementary table (regional
   volumes, cortical surface areas and cortical thicknesses; bilateral regions are combined across
   hemispheres).
2. `train_linear.py` trains the models:
   - **Diagnosis:** logistic regression on standardized features. Unregularized vs. L2 (C = 1) is
     chosen by 5-fold participant-grouped cross-validation on the training set.
   - **Alzheimer's:** soft labels (CN 0, MCI 0.5, AD 1) with log loss. Each MCI sample is
     duplicated into a weighted positive and a weighted negative, which is exactly the soft-label
     cross-entropy.
   - **Brain age:** ordinary least squares.
   - **`model: random_forest`:** the Supplementary linear-vs-nonlinear comparison, with the grid
     `n_estimators ∈ {10, 50, 100}`, `max_depth ∈ {5, 10, 20}`, `criterion ∈ {gini, entropy}`.
     `train_fraction` reproduces the 10% / 30% / 100% training-size experiment.

Note: `supramarginal_SurfArea` appears twice in the paper's feature table, so the file lists it
once. For 15 regions the table also lists a `<region>_CortexThickness` feature next to
`<region>_ThickAvg`. These features are kept in the list but are not derived until their original
definition is filled in (`extract_freesurfer_features.py` reports them as missing; run with
`--allow-missing` until then).

## CNN (`cnn/`)

`train_resnet.py` trains a 3D ResNet-18 from scratch (the paper's choice over ResNet-50 and 160³
inputs; see the Supplementary preliminary analysis). The setup is 128³ crops, batch size 16, SGD
with learning rate 0.01, up to 60 epochs, and early stopping on validation loss. Alzheimer's
diagnosis uses binary cross-entropy on the soft labels, and brain age uses mean squared error.

The U-Net segmentation baseline is in `downstream/segmentation/` (`kind: unet`), next to the
UNETR models.

## Running

```bash
python -m baselines.anatomical.extract_freesurfer_features --manifest data/manifests/adni.csv \
    --features baselines/anatomical/features_paper.txt --allow-missing --out data/features/adni_anatomical.csv
python -m baselines.anatomical.train_linear --cfg baselines/configs/anatomical_ad.yaml
python -m baselines.cnn.train_resnet --cfg baselines/configs/cnn_ad.yaml
# or on SLURM: sbatch baselines/jobs/run_baseline.sh cnn_ad
```
