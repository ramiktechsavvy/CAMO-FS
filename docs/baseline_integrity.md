# Baseline Integrity Investigation

## Scope and controls

This audit uses the Step 1 Parquet dataset and the exact row-position indices saved in `data/processed/train_indices.npy` and `test_indices.npy`. It does not modify the prior baseline outputs, raw UCI ZIP, or derived dataset. The representation ablation uses the same XGBoost parameters, seed, and train/test rows as the baseline; it is not tuning and is not CAMO-FS.

## Exact duplicate feature vectors and train/test overlap

- Dataset rows / feature dimensions: 325,834 / 261.
- Unique exact feature vectors: 325,802.
- Duplicate feature-vector groups: 32.
- Rows belonging to duplicate groups: 64.
- Duplicate groups with conflicting labels: 32.
- Unique vectors occurring in both train and test: 13.
- Of those overlapping vectors, label-agreeing / label-conflicting groups: 0 / 13.

The audit used row hashes to find repeated-vector candidates, then confirmed every candidate by exact equality of the full feature vector. Detailed counts are in `results/integrity/duplicate_leakage.json`.

## Near-duplicate diagnostic

Features were standardized using full-dataset means and population standard deviations, quantized to 0.01 standard-deviation increments, and hashed in chunks. Repeated rounded-vector groups: 32; affected rows: 64; rounded vectors shared across train/test: 13.

This is a scalable diagnostic, not an all-pairs distance calculation. It can miss close vectors that land in adjacent bins, and its threshold is not a domain-specific duplicate definition. Details and caveats are in `results/integrity/near_duplicate.json`.

## Representation ranges and temporal equation check

The derived feature matrix contains no NaN or infinity values. Summary statistics below pool all values across all 87 channels and all rows within each representation; standard deviation is population standard deviation.

| Representation | Min | Max | Mean | Std | All finite |
|---|---:|---:|---:|---:|:---:|
| EARLY | -34.308 | 556 | -0.293888 | 9.24887 | True |
| TRANSITION | -45285.4 | 1.10585e+06 | 8366.19 | 61205.2 | True |
| LATE | -45280 | 1.1059e+06 | 8365.89 | 61208 | True |

Using the saved metadata mappings, 4,352 sampled scalar checks across 256 rows and 17 channels were compared with the raw ZIP. Exact matches to raw July 14 minus raw July 5: **4,352/4,352**. Exact matches when subtracting the stored float32 LATE and EARLY snapshots: **240/4,352**; maximum absolute residual: 1.87683105e-06. The transition was constructed from full-precision parsed raw data, while snapshots are float32, so small differences after subtracting the stored snapshots are expected rounding effects.

## Original row-order inspection

The following table reports each of 10 contiguous, nearly equal-sized row segments. Class cells are within-segment proportions.

| Row segment | Rows | Train | Test | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | Class 6 | Class 7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32,583 | 26,011 | 6,572 | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 2 | 32,583 | 26,084 | 6,499 | 20.19% | 11.04% | 68.77% | 0.00% | 0.00% | 0.00% | 0.00% |
| 3 | 32,584 | 26,101 | 6,483 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 4 | 32,583 | 26,073 | 6,510 | 0.00% | 0.00% | 63.48% | 36.52% | 0.00% | 0.00% | 0.00% |
| 5 | 32,584 | 26,061 | 6,523 | 0.00% | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 0.00% |
| 6 | 32,583 | 26,043 | 6,540 | 0.00% | 0.00% | 0.00% | 90.79% | 9.21% | 0.00% | 0.00% |
| 7 | 32,583 | 26,096 | 6,487 | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% |
| 8 | 32,584 | 26,149 | 6,435 | 0.00% | 0.00% | 0.00% | 0.00% | 35.40% | 64.60% | 0.00% |
| 9 | 32,583 | 26,058 | 6,525 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% | 0.00% |
| 10 | 32,584 | 25,991 | 6,593 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 96.49% | 3.51% |

- Cramér's V for association between row segment and class: 0.748762.
- Adjacent equal-label rate: 99.9982%; expected under random ordering conditional on class counts: 20.9269%; ratio: 4.778.
- Longest contiguous same-label run: 85074 rows.

These diagnostics assess class order only. No spatial identifier or field/group key is available here, so they cannot establish or rule out spatial leakage. Detailed per-class runs and segment distributions are in `results/integrity/row_order_analysis.json` and `row_order_decile_class_distribution.csv`.

## Assessment

The full-feature baseline's accuracy of 99.756% is unusually high and warrants careful interpretation. The representation ablation in [`representation_ablation.md`](representation_ablation.md) shows how predictive performance changes when each 87-channel representation is used separately; this comparison is descriptive, not a causal attribution.

Exact train/test feature-vector overlap: **13**. Each overlapping vector has one row in each split, and all 13 pairs have conflicting labels; no label-consistent exact duplicate crosses the split. This is still direct feature-vector overlap, but it does not explain a high score through repeated same-label examples. The check does not rule out near duplicates, shared fields, spatial autocorrelation, or source-level grouping. The random stratified split is reproducible and suitable as an IID benchmark; its suitability for spatial generalization cannot be established from the available columns. Report this limitation and avoid claims of spatially independent validation.

The deterministic row-order split itself has no explicit test-time tuning. However, if source rows have field or spatial adjacency, a random split could place related observations in both partitions. The data contain no spatial IDs to test that possibility. State that the result is a random holdout baseline on this dataset, not an independent-region or independent-field estimate.

## Software versions

- python: `3.11.9`
- numpy: `1.26.4`
- pandas: `2.3.0`
- scikit-learn: `1.7.0`
- xgboost: `3.2.0`
- pyarrow: `20.0.0`

## Files

- Representation ablation: `results/integrity/representation_ablation.csv`
- Exact duplicates/leakage: `results/integrity/duplicate_leakage.json`
- Near-duplicate diagnostic: `results/integrity/near_duplicate.json`
- Temporal sanity: `results/integrity/temporal_sanity.json`
- Row-order analysis: `results/integrity/row_order_analysis.json`
