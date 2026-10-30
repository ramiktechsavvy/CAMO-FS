# Step 4 — SHAP analysis

This analysis explains the saved full-feature baseline model. EARLY, TRANSITION, and LATE are operational temporal representations, not verified biological growth stages.

## Method and sampling

- Model: saved XGBoost baseline at `results\baseline\xgboost_model.json`; no fitting or tuning was performed.
- Main sample: 10,000 rows selected by stratified sampling from saved training indices only, using `train_test_split(test_size=10000, stratify=y, random_state=42)`. The saved test indices were never passed to the explainer.
- Fold summaries: five stratified folds are formed within the saved training data. For each fold, 2,000 rows are sampled stratified from that fold's training partition. The same saved baseline model is explained; these are sampling-sensitivity rankings, not rankings from five refitted fold models.
- SHAP: `shap.TreeExplainer` (TreeSHAP), raw multiclass tree outputs. Attributions are pooled as the arithmetic mean of absolute SHAP values over sampled rows and all seven output classes. Thus rows retain their natural stratified-sample class proportions and each class output is equally represented per row.
- Batching: explanations are evaluated in batches of 250 rows to bound memory use.
- Feature prior: `normalized_shap_prior = mean_abs_shap / max(mean_abs_shap)`.

## XGBoost 3.x compatibility note

SHAP 0.48.0's XGBoost loader expects a scalar `base_score`, while this saved multiclass XGBoost 3.2.0 model stores per-class base scores. The script applies an in-memory parser shim solely to allow TreeExplainer to read the tree structure; it does not change the saved model. Additivity checking is disabled because the shim substitutes one scalar expected-value offset. Feature attributions were cross-checked against XGBoost's native `pred_contribs` TreeSHAP output and matched exactly on a deterministic validation sample. The analysis uses feature contributions, not SHAP expected values.

## Representation-level importance

| Representation | Total SHAP | Contribution (%) | Mean feature importance |
|---|---:|---:|---:|
| EARLY | 1.8235292 | 24.252% | 0.020960106 |
| TRANSITION | 2.7864354 | 37.059% | 0.032027993 |
| LATE | 2.9090139 | 38.689% | 0.033436941 |

## Global top 10 features

| Rank | Feature | Representation | Channel | Sensor | Measurement | Mean absolute SHAP |
|---:|---|---|---|---|---|---:|
| 1 | TRANSITION_CH080_MeanPC2 | TRANSITION | CH080 | optical | MeanPC2 | 0.29855005 |
| 2 | LATE_CH051_G | LATE | CH051 | optical | G | 0.26099844 |
| 3 | TRANSITION_CH051_G | TRANSITION | CH051 | optical | G | 0.24324353 |
| 4 | LATE_CH046_yamodd | LATE | CH046 | radar | yamodd | 0.20596568 |
| 5 | TRANSITION_CH055_NDVI | TRANSITION | CH055 | optical | NDVI | 0.20114752 |
| 6 | TRANSITION_CH065_SRre | TRANSITION | CH065 | optical | SRre | 0.198668 |
| 7 | LATE_CH045_freevol | LATE | CH045 | radar | freevol | 0.19286732 |
| 8 | TRANSITION_CH053_Redge | TRANSITION | CH053 | optical | Redge | 0.16152884 |
| 9 | TRANSITION_CH062_gNDVI | TRANSITION | CH062 | optical | gNDVI | 0.16144307 |
| 10 | LATE_CH053_Redge | LATE | CH053 | optical | Redge | 0.15930042 |

## Top 10 EARLY

| Global rank | Feature | Channel | Sensor | Measurement | Mean absolute SHAP |
|---:|---|---|---|---|---:|
| 12 | EARLY_CH008_Rhvhh | CH008 | radar | Rhvhh | 0.13615208 |
| 15 | EARLY_CH080_MeanPC2 | CH080 | optical | MeanPC2 | 0.12351894 |
| 16 | EARLY_CH040_krogks | CH040 | radar | krogks | 0.11513993 |
| 20 | EARLY_CH072_MeanPC1 | CH072 | optical | MeanPC1 | 0.1007726 |
| 22 | EARLY_CH046_yamodd | CH046 | radar | yamodd | 0.079987084 |
| 23 | EARLY_CH049_yamvol | CH049 | radar | yamvol | 0.077382729 |
| 31 | EARLY_CH053_Redge | CH053 | optical | Redge | 0.064117653 |
| 33 | EARLY_CH043_freeodd | CH043 | radar | freeodd | 0.059877057 |
| 35 | EARLY_CH021_Ro23 | CH021 | radar | Ro23 | 0.056875901 |
| 36 | EARLY_CH069_TCARI | CH069 | optical | TCARI | 0.056383856 |

## Top 10 TRANSITION

| Global rank | Feature | Channel | Sensor | Measurement | Mean absolute SHAP |
|---:|---|---|---|---|---:|
| 1 | TRANSITION_CH080_MeanPC2 | CH080 | optical | MeanPC2 | 0.29855005 |
| 3 | TRANSITION_CH051_G | CH051 | optical | G | 0.24324353 |
| 5 | TRANSITION_CH055_NDVI | CH055 | optical | NDVI | 0.20114752 |
| 6 | TRANSITION_CH065_SRre | CH065 | optical | SRre | 0.198668 |
| 8 | TRANSITION_CH053_Redge | CH053 | optical | Redge | 0.16152884 |
| 9 | TRANSITION_CH062_gNDVI | CH062 | optical | gNDVI | 0.16144307 |
| 18 | TRANSITION_CH064_NDVIre | CH064 | optical | NDVIre | 0.10869155 |
| 21 | TRANSITION_CH060_SAVI | CH060 | optical | SAVI | 0.087689271 |
| 28 | TRANSITION_CH001_sigHH | CH001 | radar | sigHH | 0.065776226 |
| 29 | TRANSITION_CH002_sigHV | CH002 | radar | sigHV | 0.064438181 |

## Top 10 LATE

| Global rank | Feature | Channel | Sensor | Measurement | Mean absolute SHAP |
|---:|---|---|---|---|---:|
| 2 | LATE_CH051_G | CH051 | optical | G | 0.26099844 |
| 4 | LATE_CH046_yamodd | CH046 | radar | yamodd | 0.20596568 |
| 7 | LATE_CH045_freevol | CH045 | radar | freevol | 0.19286732 |
| 10 | LATE_CH053_Redge | CH053 | optical | Redge | 0.15930042 |
| 11 | LATE_CH003_sigVV | CH003 | radar | sigVV | 0.15184548 |
| 13 | LATE_CH049_yamvol | CH049 | radar | yamvol | 0.13435944 |
| 14 | LATE_CH080_MeanPC2 | CH080 | optical | MeanPC2 | 0.13237264 |
| 17 | LATE_CH015_Rvv | CH015 | radar | Rvv | 0.11419578 |
| 19 | LATE_CH072_MeanPC1 | CH072 | optical | MeanPC1 | 0.10181118 |
| 24 | LATE_CH043_freeodd | CH043 | radar | freeodd | 0.075786959 |

## Interpretation and limitations

Representation percentages summarize model reliance under this fitted model and this training-only sample; they are not causal effects, and correlated channels can share or redistribute attribution. Global importance averages across the seven class outputs and the observed class mix. Fold files reuse one fitted model, so they prepare rank comparisons for later analysis but do not measure full model-refit variability. The operational representation labels should not be read as validated phenological stages. No test observations were used for SHAP or priors.

Total explanation runtime: 34.1 seconds (0.6 minutes).

## Artifacts

Importance and sampling records are under `results/shap/`; figures are under `results/shap/figures/`. `configuration.json`, `shap_sample_indices.npy`, and `fold_sample_indices.npz` preserve key reproducibility details.
