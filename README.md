# Crop-type classification on a PASTIS Sentinel-2 subset

Pixel-level crop-type classification from multi-temporal Sentinel-2 imagery (102 patches of the
[PASTIS benchmark](https://github.com/VSainteuf/pastis-benchmark), tile T31TFM, France). The project covers data
loading, exploratory analysis, a leakage-aware patch split, training and comparison of six classical classifiers,
test-set evaluation and prediction maps. The written analysis is in [`report.md`](report.md).

**Headline result (test split, 18 unseen patches):** tree ensembles and logistic regression reach 82–84 % overall
accuracy and 0.43–0.45 mIoU. XGBoost is best (OA 0.844, mIoU 0.446, kappa 0.805). The seven major classes reach
F1 0.78–0.95, while classes with only a handful of training fields are not learned.

## Repository structure

```
├── README.md                 this file
├── report.md                 analysis and interpretation (Task G)
├── requirements.txt          pinned Python dependencies
├── configs/config.yaml       every setting: paths, split folds, features, sampling, model hyperparameters
├── notebooks/
│   └── exploration_and_training.ipynb   EDA, split, training results, evaluation and maps (Tasks B–F)
├── src/
│   ├── config.py             loads config.yaml, resolves paths
│   ├── data_loading.py       PastisSubset: reads S2 cubes, targets, metadata and acquisition dates
│   ├── preprocessing.py      reflectance scaling, NDVI/NDMI, per-pixel features, class-capped sampling, caching
│   ├── splits.py             train/val/test split from the official PASTIS folds (CLI)
│   ├── train.py              fits and validates the six models (CLI)
│   ├── evaluate.py           metrics, confusion matrices, val/test evaluation of saved models (CLI)
│   └── visualization.py      Plotly theme, PASTIS class colours, image composites
├── splits/                   train_ids.txt, val_ids.txt, test_ids.txt (patch IDs, one per line)
└── outputs/
    ├── figures/              every figure used in the notebook and report (PNG)
    ├── metrics/              CSV/JSON/NPZ metrics: model comparison, per-class scores, confusion matrices,
    │                         training_log.json (selected settings and fit / predict times per model)
    ├── predictions/          <split>_<model>.npy prediction maps (uint8 PASTIS class ids)
    └── models/               label_classes.json, feature_names.json (+ model binaries, not committed)
```

## Data

The dataset is not included. Place it in the project root as:

```
PASTIS_subset/
├── DATA_S2/S2_<patch_id>.npy          (46, 10, 128, 128) int16 — dates × bands × H × W, reflectance × 10,000
├── ANNOTATIONS/TARGET_<patch_id>.npy  (1, 128, 128) uint8 — PASTIS semantic labels (0 background … 19 void)
└── metadata.geojson                   patch footprints (EPSG:2154), acquisition dates, official folds
```

Another location can be set via `paths.data_root` in `configs/config.yaml`.

## Environment

Developed with Python 3.13 on Linux, 8 CPU cores, no GPU and ~1.3 GB of free RAM. At least 4 GB of RAM is
recommended: training peaks at ~3.3 GB resident, part of it memory-mapped feature files.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/plotly_get_chrome        # only if no Chrome/Chromium is installed (needed to export PNG figures)
```

Select the `.venv` interpreter as the notebook kernel.

## How to run / reproduce the results

Run from the project root. Each step reads `configs/config.yaml` and is deterministic (seed 42).

| Step | Command | Output | Time (8 cores) |
|---|---|---|---|
| 1. Split | `.venv/bin/python -m src.splits` | `splits/*_ids.txt` | seconds |
| 2. Train + validate | `.venv/bin/python -m src.train` | `outputs/models/`, `outputs/metrics/val_*`, `outputs/predictions/val_*` | ≈ 36 min |
| 3. Evaluate val + test | `.venv/bin/python -m src.evaluate` | `outputs/metrics/evaluation_*`, `outputs/predictions/test_*` | ≈ 13 min |
| 4. Notebook | open `notebooks/exploration_and_training.ipynb` and run all | all figures in `outputs/figures/` | ≈ 2 min |

- `python -m src.train --models xgboost lightgbm` retrains a subset; `python -m src.evaluate --splits test
  --models random_forest` evaluates a subset.
- Features are cached in `cache/` (~1.5 GB, git-ignored) on first use; later runs reuse them.
- Most evaluation time is SVM prediction (~6 min per split). All other models predict a split in seconds.
- **Trained models are not committed**: the six `.joblib` files total ~56 MB, almost all of it the random forest
  (47 MB). Step 2 recreates them, and step 3 confirms that the reloaded models reproduce the training-time validation
  scores exactly.

## Approach in brief

1. **Features:** each pixel's full season: 10 bands × 46 dates of reflectance (DN / 10,000, clipped to [0, 1]) plus
   per-date NDVI and NDMI, i.e. 552 features. This works without temporal resampling because all patches share the
   same 46 acquisition dates.
2. **Split:** whole patches, using the official PASTIS folds: train = folds 1–3 (66 patches), validation = fold 4,
   test = fold 5 (18 each). The val/test folds were chosen by an explicit class-coverage rule (`src/splits.py`).
3. **Training data:** a class-capped random sample of ≤ 15,000 pixels per class from the training patches (141,831
   pixels). Class imbalance is handled with that cap plus inverse-square-root-frequency sample weights.
4. **Models:** majority baseline, logistic regression, RBF SVM, random forest, XGBoost and LightGBM. Small grid
   searches and GBDT early stopping use a class-capped validation subsample only.
5. **Evaluation:** OA, per-class precision / recall / F1 / IoU, macro-F1, mIoU and kappa (void excluded), confusion
   matrices, patch-bootstrap intervals, error analysis by distance to parcel boundaries, and prediction maps.

## Key assumptions

- DN / 10,000 is surface reflectance. Values below 0 or above 1 are atmospheric-correction artefacts or clouds and
  are clipped.
- The shared 46-date calendar makes feature *t* comparable across patches. Data from several tiles would need
  resampling to a common temporal grid.
- Void (19) marks unreliable labels inside parcels and is excluded from training and scoring. Background (0) is
  treated as a real class.
- No cloud mask is available. Cloudy observations are kept, and models are expected to learn to ignore them.

## Limitations and known issues

- Pixels are classified independently. Parcel-edge pixels (25 % of labelled pixels) cause 45 % of test errors, and
  there is no spatial context or parcel-level aggregation.
- Classes present in only 1–3 training patches (spring barley, durum wheat, fruits/vegetables, sorghum, mixed cereal,
  grapevine, potatoes, orchard) are not learned. Beet is absent from the subset.
- Validation and test have 18 patches each, so rankings between the top models are not statistically decisive (see
  the bootstrap in the notebook and report).
- The patches come from one ~33 × 36 km area and one season. The scores measure within-region generalisation only.
- The notebook's figures are interactive Plotly outputs, which GitHub does not render. Static copies of every figure
  are in `outputs/figures/`, and `report.md` embeds the key ones.
