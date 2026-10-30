# Representation Ablation

## Protocol

Four independent XGBoost models were fit using the exact saved seed-42 stratified split. The stored baseline configuration was reused without tuning: `{"colsample_bytree": 0.8, "eval_metric": "mlogloss", "learning_rate": 0.05, "max_depth": 6, "n_estimators": 300, "n_jobs": -1, "num_class": 7, "objective": "multi:softprob", "random_state": 42, "subsample": 0.8, "tree_method": "hist", "verbosity": 1}`. Each model was fit only on training rows; test data were used only for final scoring. No feature selection, SHAP, or NSGA-II was performed. EARLY, TRANSITION, and LATE are operational date/change representations, not verified biological growth stages.

Split sizes: 260,667 train and 65,167 test. Features: 87 per individual representation and 261 combined.

## Results

| Representation | Features | Accuracy | Macro-F1 | Weighted-F1 | Kappa | Fit seconds |
|---|---:|---:|---:|---:|---:|---:|
| EARLY | 87 | 0.990486 | 0.986158 | 0.990485 | 0.987968 | 56.41 |
| TRANSITION | 87 | 0.996087 | 0.989275 | 0.996090 | 0.995051 | 59.82 |
| LATE | 87 | 0.989565 | 0.987537 | 0.989565 | 0.986803 | 56.72 |
| ALL_261 | 261 | 0.997560 | 0.994113 | 0.997562 | 0.996914 | 128.52 |

### Per-class F1

| Experiment | Class 1 F1 | Class 2 F1 | Class 3 F1 | Class 4 F1 | Class 5 F1 | Class 6 F1 | Class 7 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| EARLY | 0.993044 | 0.996530 | 0.998084 | 0.992912 | 0.978210 | 0.987427 | 0.956897 |
| TRANSITION | 0.997380 | 0.998609 | 0.999009 | 0.996759 | 0.991510 | 0.995418 | 0.946237 |
| LATE | 0.994897 | 0.997214 | 0.998149 | 0.987865 | 0.980899 | 0.985720 | 0.968017 |
| ALL_261 | 0.997892 | 0.999305 | 0.999141 | 0.997805 | 0.995009 | 0.997474 | 0.972163 |

## Interpretation

These results compare predictive signal in the representations under one fixed classifier and split. They do not establish causal importance, biological stage meaning, or performance on a spatially independent test set. Use Macro-F1 and class-wise scores because the target is imbalanced.

The full results, including per-class precision and recall, are in `results/integrity/representation_ablation.csv`.

## Software

- python: `3.11.9`
- numpy: `1.26.4`
- pandas: `2.3.0`
- scikit-learn: `1.7.0`
- xgboost: `3.2.0`
- pyarrow: `20.0.0`
