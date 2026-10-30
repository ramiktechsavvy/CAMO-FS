# CAMO-FS

**Context-Aware Multi-Objective Feature Selection for Smart Agriculture**

Research project for seven-class crop classification using the UCI Crop Mapping using Fused Optical-Radar Dataset (dataset #525). The planned experimental subset contains 60,000 stratified samples from the full dataset (325,834 samples, 175 features, 7 crop classes). The downstream classifier is planned to be an XGBoost/gradient-boosted classifier.

The planned method is SHAP-guided NSGA-II with three objectives: classification error (`1 - Macro-F1`), relative subset size, and explanation instability (`1 - ECI`). The final system is intended to support growth stages (Early, Mid, Late), stage-wise Pareto archives, and knee-point selection. Stage-to-feature mappings are intentionally unspecified until established from authoritative dataset metadata.

Evaluation is planned to cover baselines, repeated runs, bootstrap validation, statistical tests, explainability, and ablation studies. No data, results, or algorithm implementations are included in this initial scaffold.

## Repository layout

```text
CAMO-FS/
├── configs/                 # Experiment and dataset configuration
├── data/                    # Raw, interim, and processed data locations
├── docs/                    # Research and project documentation
├── experiments/             # Reproducible experiment entry points
├── src/camo_fs/             # Python package
│   ├── baselines/            # Baseline feature-selection methods
│   ├── camo_fs/              # CAMO-FS orchestration and method components
│   ├── evaluation/           # Metrics and statistical evaluation
│   ├── explainability/       # SHAP and explanation-stability components
│   ├── feature_selection/    # Feature-selection interfaces and components
│   ├── models/               # Classifier definitions and adapters
│   ├── optimization/         # Multi-objective optimization components
│   ├── preprocessing/        # Data preparation components
│   ├── utils/                # Shared utilities
│   └── visualization/        # Plotting and reporting components
└── tests/                    # Automated checks
```

## Setup

Create an environment with Python 3.10 or newer, then install the declared dependencies:

```bash
python -m pip install -r requirements.txt
```

Dataset access, preprocessing, experiment execution, and implementation details will be added in later work.
