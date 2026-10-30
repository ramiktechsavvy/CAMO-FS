# CAMO-FS Final Results

## 1. Experimental setup

The experiment uses UCI Crop Mapping using Fused Optical-Radar Dataset #525, with 325,834 rows, 261 derived predictors, and seven crop classes. The canonical representations are EARLY (July 5), TRANSITION (July 14 minus July 5), and LATE (July 14). These are operational temporal representations, not verified biological stages. The existing seed-42 stratified split was reused: 260,667 training rows and 65,167 test rows. XGBoost used the saved baseline configuration (300 estimators, depth 6, learning rate 0.05, subsample and column sample 0.8, seed 42, histogram tree method). Each new method selected features using training data only, then was evaluated once on the saved test rows.

## 2. Baseline

The existing all-feature baseline achieved Accuracy 0.997560, Macro-F1 0.994113, Weighted-F1 0.997562, and Kappa 0.996914. Its artifacts were reused; the baseline was not retrained.

## 3. Representation analysis

| Representation | Features | Accuracy | Macro-F1 | Weighted-F1 | Kappa |
|---|---:|---:|---:|---:|---:|
| EARLY | 87 | 0.990486 | 0.986158 | 0.990485 | 0.987968 |
| TRANSITION | 87 | 0.996087 | 0.989275 | 0.996090 | 0.995051 |
| LATE | 87 | 0.989565 | 0.987537 | 0.989565 | 0.986803 |
| ALL_261 | 261 | 0.997560 | 0.994113 | 0.997562 | 0.996914 |

These are the already completed ablation results; they were not rerun.

## 4. SHAP analysis

The saved training-only TreeSHAP analysis found representation contributions of:
- EARLY: 24.252% of total mean absolute SHAP
- TRANSITION: 37.059% of total mean absolute SHAP
- LATE: 38.689% of total mean absolute SHAP

The ten global leaders include TRANSITION_CH080_MeanPC2, LATE_CH051_G, TRANSITION_CH051_G, LATE_CH046_yamodd, and TRANSITION_CH055_NDVI. Values are model attributions, not causal effects. The saved analysis used a documented SHAP/XGBoost base-score compatibility shim and native TreeSHAP parity check; see `docs/shap_analysis.md`.

## 5. ECI

ECI from the existing five SHAP fold rankings was 0.996387 overall. Representation ECI was EARLY 0.995295, TRANSITION 0.996632, and LATE 0.996044. Subset ECI values use the same rankings restricted to each method's selected features.

## 6. CAMO-FS optimization

CAMO-FS used SHAP-guided initialization and mutation, NSGA-II with objectives 1−Macro-F1, subset ratio, and 1−ECI. The saved optimizer ran 60 individuals for 40 generations, with a 10,000-row training-only surrogate objective (8,000 fit / 2,000 validation, 30 boosting rounds), caching 2,169 unique candidates. The selected subset received full saved-training 5-fold CV at 300 rounds. Final archive selection was weighted distance-to-ideal with weights (0.5, 0.2, 0.3); this is not geometric knee detection. Only one deterministic optimizer run exists.

## 7. Feature selection results

CAMO-FS selected 5 of 261 features (98.084% reduction; subset ratio 0.019157): `TRANSITION_CH051_G`, `TRANSITION_CH055_NDVI`, `LATE_CH003_sigVV`, `LATE_CH015_Rvv`, `LATE_CH027_l3`. Its full-training CV Macro-F1 was 0.969073. Its final saved-test Accuracy was 0.978931, Macro-F1 0.968552, Weighted-F1 0.978920, and Kappa 0.973351.

## 8. Baseline comparison

| Method | Features | Accuracy | Macro-F1 | Weighted-F1 | Kappa | ECI | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| All_261 | 261 | 0.997560 | 0.994113 | 0.997562 | 0.996914 | 0.996387 | 135.4 |
| Mutual_Information_5 | 5 | 0.924241 | 0.893217 | 0.924215 | 0.904171 | 1.000000 | 25.7 |
| RFE_5 | 5 | 0.942778 | 0.935409 | 0.942495 | 0.927567 | 1.000000 | 14.0 |
| RandomForest_5 | 5 | 0.916814 | 0.864098 | 0.916398 | 0.894658 | 1.000000 | 24.0 |
| SingleObjectiveGA_5 | 5 | 0.973422 | 0.968859 | 0.973414 | 0.966389 | 1.000000 | 35.9 |
| ContextFreeNSGAII_5 | 5 | 0.956742 | 0.948950 | 0.956475 | 0.945229 | 1.000000 | 44.2 |
| CAMO_FS_5 | 5 | 0.978931 | 0.968552 | 0.978920 | 0.973351 | 1.000000 | 550.1 |

Runtime includes selection, final-model training, and one test evaluation for newly run selectors. CAMO-FS includes its prior optimization and full training CV. The all-feature runtime is the existing baseline runtime. Selector runtimes therefore reflect their stated computational procedures and are not equal-cost implementations. MI and Random Forest used a deterministic 60,000-row stratified training sample; RFE used 20,000 rows. The GA and context-free NSGA-II used the same saved 10,000-row training-only search sample and 30-round search model. All final classifiers used the complete training partition and baseline 300-round classifier.

### CAMO-FS ablation interpretation

`results/final/camo_fs_ablation.csv` compares CAMO-FS with All_261, a single-objective GA, and a context-free NSGA-II. These deterministic comparisons describe the tested procedures; they do not isolate causal effects. The context-free NSGA-II retains ECI as an objective but uses no SHAP prior for initialization or mutation. Because all methods fix K=5, subset-size is constant; observed five-feature subset ECI was 1.0, and the context-free Pareto archive collapsed to one solution. Its second and third objectives were therefore degenerate in this comparison. Its search budget is also smaller than CAMO-FS's, so performance differences do not establish causal superiority of SHAP guidance.

## 9. Class-wise performance

The class-wise table includes all-feature, CAMO-FS, and the strongest non-CAMO selector (SingleObjectiveGA_5), selected post hoc by its single test Macro-F1 for descriptive reporting only. This did not tune any selector.

| Method | Class | F1 | Test support |
|---|---|---:|---:|
| All_261 | Peas | 0.999305 | 720 |
| All_261 | Broadleaf | 0.972163 | 229 |
| CAMO_FS_5 | Peas | 0.984807 | 720 |
| CAMO_FS_5 | Broadleaf | 0.908297 | 229 |
| SingleObjectiveGA_5 | Peas | 0.997222 | 720 |
| SingleObjectiveGA_5 | Broadleaf | 0.929032 | 229 |

## 10. Limitations

- UCI provides two acquisition dates, not three verified phenological stages. EARLY/TRANSITION/LATE are operational temporal representations.
- The split is random stratified and IID-style, not spatially independent. Original rows are class-block ordered; field/spatial IDs are unavailable.
- Exact feature-vector overlaps with conflicting labels were found in the integrity audit; the current split cannot establish field-level generalization.
- Evolutionary search used a 10,000-row, one-holdout, 30-round surrogate for candidate fitness. The rare Broadleaf class had 35 rows in that sample, including 7 in its validation partition. The selected feature set is conditional on this approximation.
- The context-free NSGA-II archive contained one solution after each generation. With K fixed at five, subset-size is constant, and the observed subset ECI was 1.0, leaving effectively only classification error to distinguish candidates. This is a degenerate and limited multi-objective comparator.
- Only one deterministic optimization and one final test evaluation were performed. No Wilcoxon tests or claims of statistically significant superiority are made.
- The SHAP 0.48.0/XGBoost 3.2.0 compatibility shim substitutes a scalar only in SHAP's expected-value parser; feature contributions were checked against native TreeSHAP on a fixed sample. Fold ECI measures consistency of the saved rankings, not refit-model stability.
- A bootstrap interval below quantifies row-resampling uncertainty for this one test prediction vector; it is not uncertainty across independent experiments.
- The test set is one fixed holdout and was used for the final comparisons; no further method tuning should use these results.

## 11. Conclusion

On this fixed random holdout, CAMO-FS reduced the input from 261 to five features and achieved test Macro-F1 0.968552. The selected subset was dominated by TRANSITION and LATE operational representations. The deterministic comparison describes this dataset split and these implementation choices only; it does not establish universal superiority, biological-stage discovery, spatial generalization, or statistically significant differences.

### Bootstrap interval for CAMO-FS test predictions

Method: nonparametric row bootstrap of the one saved test prediction vector; not independent experimental runs; 1000 replicates, seed 42.

- Accuracy 95% percentile CI: [0.977811, 0.980082]
- Macro-F1 95% percentile CI: [0.964284, 0.972630]

## Artifacts

Machine-readable metrics, predictions, per-class reports, confusion matrices, selector configurations, and plots are under `results/final/`. The fixed split and raw dataset were not modified.
