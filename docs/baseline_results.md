# Full-Feature XGBoost Baseline

This is a reproducible full-feature baseline for the operational two-date temporal representation. EARLY, TRANSITION, and LATE are representation names, not verified biological growth stages. This is not a CAMO-FS feature-selection result.

## Dataset and split

- Dataset: `data\processed\camo_temporal_261.parquet`
- Dataset SHA-256: `fff1716b40eb1f5b3600c7868643fdc236d955e73f89837442a0b06f949ccb9b`
- Shape: 325,834 rows × 261 features plus integer target `label`
- Representation counts: EARLY=87, TRANSITION=87, LATE=87
- Sample IDs: zero-based original Parquet row positions; no ID column was added to the dataset.
- Split: stratified 80% train / 20% test, seed 42; exact row positions are saved under `data/processed/` and reused on later runs.
- Sizes: 260,667 train and 65,167 test.
- Test observations were not supplied to `fit`; they were used only for final evaluation.

### Class distributions

| Label | Full count | Full share | Train count | Train share | Test count | Test share |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39,162 | 12.0190% | 31,330 | 12.0192% | 7,832 | 12.0184% |
| 2 | 3,598 | 1.1042% | 2,878 | 1.1041% | 720 | 1.1049% |
| 3 | 75,673 | 23.2244% | 60,538 | 23.2243% | 15,135 | 23.2249% |
| 4 | 74,067 | 22.7315% | 59,254 | 22.7317% | 14,813 | 22.7308% |
| 5 | 47,117 | 14.4604% | 37,694 | 14.4606% | 9,423 | 14.4598% |
| 6 | 85,074 | 26.1096% | 68,059 | 26.1096% | 17,015 | 26.1098% |
| 7 | 1,143 | 0.3508% | 914 | 0.3506% | 229 | 0.3514% |

## Model configuration

```json
{
  "objective": "multi:softprob",
  "eval_metric": "mlogloss",
  "n_estimators": 300,
  "max_depth": 6,
  "learning_rate": 0.05,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "random_state": 42,
  "n_jobs": -1,
  "tree_method": "hist",
  "verbosity": 1,
  "num_class": 7
}
```

## Test metrics

- Accuracy: **0.997560**
- Macro-F1: **0.994113**
- Weighted-F1: **0.997562**
- Cohen's Kappa: **0.996914**
- Training runtime: 131.83 seconds
- Total experiment runtime: 135.43 seconds

### Class-wise test performance

| Class label | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 1 | 0.998339 | 0.997446 | 0.997892 | 7832 |
| 2 | 1.000000 | 0.998611 | 0.999305 | 720 |
| 3 | 0.999075 | 0.999207 | 0.999141 | 15135 |
| 4 | 0.998176 | 0.997435 | 0.997805 | 14813 |
| 5 | 0.995643 | 0.994375 | 0.995009 | 9423 |
| 6 | 0.996889 | 0.998061 | 0.997474 | 17015 |
| 7 | 0.953782 | 0.991266 | 0.972163 | 229 |

The dataset is imbalanced, so Macro-F1 and the per-class results should be interpreted alongside accuracy. No hyperparameter search or feature selection was performed.

## Software versions

- python: `3.11.9`
- numpy: `1.26.4`
- pandas: `2.3.0`
- scikit-learn: `1.7.0`
- xgboost: `3.2.0`
- pyarrow: `20.0.0`

## Reproduction artifacts

- Split indices: `data/processed/train_indices.npy`, `data/processed/test_indices.npy`
- Split metadata: `data/processed/baseline_split.json`
- Configuration and data/split fingerprints: `results/baseline/configuration.json`
- Metrics: `results/baseline/metrics.json`
- Per-class metrics: `results/baseline/classification_report.csv`
- Confusion matrix: `results/baseline/confusion_matrix.csv`
- Trained XGBoost model: `results/baseline/xgboost_model.json`

Re-run with `python src/camo_fs/experiments/baseline_xgboost.py`. Existing saved indices are reused; the script refuses to silently replace them if they are incomplete or do not match the data fingerprint.
