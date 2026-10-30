"""Reproducible TreeSHAP analysis of the saved full-feature baseline.

Run from the project root with ``python src/camo_fs/explainability/shap_analysis.py``.
Only the saved training split is sampled or explained. No model is fitted here.
"""

from __future__ import annotations

import json
import hashlib
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data/processed/camo_temporal_261.parquet"
METADATA_PATH = ROOT / "data/processed/camo_temporal_feature_metadata.csv"
TRAIN_PATH = ROOT / "data/processed/train_indices.npy"
TEST_PATH = ROOT / "data/processed/test_indices.npy"
SPLIT_INFO_PATH = ROOT / "data/processed/baseline_split.json"
MODEL_PATH = ROOT / "results/baseline/xgboost_model.json"
BASELINE_CONFIG_PATH = ROOT / "results/baseline/configuration.json"
OUT = ROOT / "results/shap"
REPORT = ROOT / "docs/shap_analysis.md"

SEED = 42
MAIN_SAMPLE_SIZE = 10_000
FOLD_COUNT = 5
FOLD_SAMPLE_SIZE = 2_000
SHAP_BATCH_SIZE = 250
REPRESENTATIONS = ["EARLY", "TRANSITION", "LATE"]


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def parse_multiclass_base_score_for_shap() -> None:
    """Work around SHAP 0.48 parsing XGBoost 3.x vector base_score.

    The saved XGBoost model is not modified. XGBoost's vector base score is a
    per-class additive constant. Feature contributions are tree-path values;
    this compatibility shim supplies a scalar only to SHAP's legacy model
    loader. We disable its additivity check and validate feature attributions
    against XGBoost's native pred_contribs output before analysis.
    """
    import shap.explainers._tree as shap_tree

    if getattr(shap_tree.decode_ubjson_buffer, "_camo_vector_base_score", False):
        return
    original = shap_tree.decode_ubjson_buffer

    def decode_compatible(stream: Any) -> dict[str, Any]:
        model = original(stream)
        try:
            params = model["learner"]["learner_model_param"]
            raw = params["base_score"]
            if isinstance(raw, str) and raw.startswith("["):
                # The constant affects expected values, not per-feature SHAP.
                # Use one existing class intercept rather than inventing a value.
                params["base_score"] = str(json.loads(raw)[0])
        except (KeyError, TypeError, ValueError, IndexError):
            pass
        return model

    decode_compatible._camo_vector_base_score = True  # type: ignore[attr-defined]
    shap_tree.decode_ubjson_buffer = decode_compatible


def normalize_shap_array(values: Any, n_features: int) -> np.ndarray:
    """Return SHAP values with shape (rows, features, classes)."""
    if isinstance(values, list):
        array = np.stack([np.asarray(class_values) for class_values in values], axis=-1)
    else:
        array = np.asarray(values)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"Expected 2D/3D multiclass SHAP output; got {array.shape}.")
    if array.shape[1] != n_features and array.shape[2] == n_features:
        array = array.transpose(0, 2, 1)
    if array.shape[1] != n_features:
        raise ValueError(f"SHAP feature axis does not match {n_features}: {array.shape}.")
    return array


def sample_stratified(
    population_ids: np.ndarray, labels: np.ndarray, size: int, random_state: int
) -> np.ndarray:
    if size >= len(population_ids):
        return population_ids.copy()
    _, selected = train_test_split(
        population_ids,
        test_size=size,
        random_state=random_state,
        stratify=labels[population_ids],
    )
    return np.asarray(selected, dtype=np.int64)


def mean_abs_shap(explainer: Any, X: np.ndarray) -> np.ndarray:
    """Pool absolute contributions uniformly over rows and output classes."""
    total = np.zeros(X.shape[1], dtype=np.float64)
    n_rows = 0
    for start in range(0, len(X), SHAP_BATCH_SIZE):
        batch = X[start : start + SHAP_BATCH_SIZE]
        values = explainer.shap_values(batch, check_additivity=False)
        values = normalize_shap_array(values, X.shape[1])
        if not np.isfinite(values).all():
            raise ValueError("SHAP returned NaN or infinite feature attributions.")
        total += np.abs(values).sum(axis=(0, 2), dtype=np.float64)
        n_rows += values.shape[0]
    if n_rows == 0:
        raise ValueError("Cannot calculate SHAP on an empty sample.")
    return total / (n_rows * values.shape[2])


def validate_and_load() -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    for path in (DATA_PATH, METADATA_PATH, TRAIN_PATH, TEST_PATH, SPLIT_INFO_PATH, MODEL_PATH, BASELINE_CONFIG_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required Step 0–3 artifact is missing: {path}")
    split_info = json.loads(SPLIT_INFO_PATH.read_text(encoding="utf-8"))
    baseline_config = json.loads(BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))
    df = pd.read_parquet(DATA_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    features = [name for name in df.columns if name != "label"]
    if len(df) != 325_834 or len(features) != 261:
        raise ValueError(f"Unexpected dataset dimensions: {df.shape}.")
    if metadata["feature_name"].tolist() != features or metadata["feature_name"].duplicated().any():
        raise ValueError("Feature metadata does not align exactly with Parquet feature order.")
    if baseline_config.get("dataset", {}).get("feature_names") != features:
        raise ValueError("Saved baseline feature order differs from Parquet and metadata order.")
    if metadata["representation"].value_counts().to_dict() != {"EARLY": 87, "TRANSITION": 87, "LATE": 87}:
        raise ValueError("Expected exactly 87 metadata features per representation.")
    train_ids = np.load(TRAIN_PATH, allow_pickle=False)
    test_ids = np.load(TEST_PATH, allow_pickle=False)
    if len(train_ids) != split_info["train_rows"] or len(test_ids) != split_info["test_rows"]:
        raise ValueError("Saved split indices disagree with split metadata.")
    if np.intersect1d(train_ids, test_ids).size:
        raise ValueError("Saved training and test indices overlap.")
    digest = hashlib.sha256()
    with DATA_PATH.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if split_info.get("dataset_sha256") != digest.hexdigest():
        raise ValueError("Saved split metadata hash does not match the temporal dataset file.")
    if not pd.api.types.is_integer_dtype(df["label"].dtype):
        raise ValueError("Target labels must remain integer typed.")
    if df[features].isna().to_numpy().any():
        raise ValueError("Dataset contains missing features.")
    X = df[features].to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(X).all():
        raise ValueError("Dataset contains infinite feature values.")
    return df, metadata, features, train_ids, test_ids


def write_tables(metadata: pd.DataFrame, features: list[str], importance: np.ndarray) -> pd.DataFrame:
    if importance.shape != (len(features),) or not np.isfinite(importance).all():
        raise ValueError("Expected one finite global importance per feature.")
    table = metadata.set_index("feature_name").loc[features].reset_index().copy()
    table["mean_abs_shap"] = importance
    table["normalized_shap_prior"] = importance / importance.max()
    table["global_rank"] = table["mean_abs_shap"].rank(method="min", ascending=False).astype(int)
    columns = [
        "feature_name", "representation", "channel_id", "sensor", "measurement_name",
        "mean_abs_shap", "normalized_shap_prior", "global_rank",
    ]
    table[columns].sort_values("global_rank").to_csv(OUT / "shap_feature_importance.csv", index=False)
    table.sort_values("global_rank").head(30)[columns].to_csv(OUT / "global_top_30.csv", index=False)
    for representation in REPRESENTATIONS:
        table[table.representation == representation].sort_values("mean_abs_shap", ascending=False).head(10)[columns].to_csv(
            OUT / f"top_10_{representation.lower()}.csv", index=False
        )
    summary = table.groupby("representation", sort=False)["mean_abs_shap"].agg(total_shap="sum", mean_feature_importance="mean").reset_index()
    summary["percentage_contribution"] = 100 * summary["total_shap"] / summary["total_shap"].sum()
    summary.to_csv(OUT / "representation_summary.csv", index=False)
    return table


def make_plots(table: pd.DataFrame, summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    top = table.sort_values("mean_abs_shap").tail(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = top["representation"].map({"EARLY": "#4C78A8", "TRANSITION": "#F58518", "LATE": "#54A24B"})
    ax.barh(top["feature_name"], top["mean_abs_shap"], color=colors)
    ax.set(xlabel="Mean absolute SHAP (pooled across rows and classes)", title="Global top 20 features")
    fig.tight_layout()
    fig.savefig(figure_dir / "global_top_20_shap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(summary["representation"], summary["percentage_contribution"], color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set(ylabel="Share of total mean absolute SHAP (%)", title="Representation-level SHAP contribution")
    fig.tight_layout()
    fig.savefig(figure_dir / "representation_contribution.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharex=False)
    for ax, representation in zip(axes, REPRESENTATIONS):
        subset = table[table.representation == representation].nlargest(10, "mean_abs_shap").sort_values("mean_abs_shap")
        ax.barh(subset.feature_name, subset.mean_abs_shap, color={"EARLY": "#4C78A8", "TRANSITION": "#F58518", "LATE": "#54A24B"}[representation])
        ax.set_title(representation)
    fig.suptitle("Top 10 features within each operational representation")
    fig.tight_layout()
    fig.savefig(figure_dir / "top_features_by_representation.png", dpi=180)
    plt.close(fig)


def write_report(table: pd.DataFrame, summary: pd.DataFrame, config: dict[str, Any]) -> None:
    global_top = table.nsmallest(10, "global_rank")
    lines = [
        "# Step 4 — SHAP analysis", "",
        "This analysis explains the saved full-feature baseline model. EARLY, TRANSITION, and LATE are operational temporal representations, not verified biological growth stages.", "",
        "## Method and sampling", "",
        f"- Model: saved XGBoost baseline at `{config['model_path']}`; no fitting or tuning was performed.",
        f"- Main sample: {config['main_sample_rows']:,} rows selected by stratified sampling from saved training indices only, using `train_test_split(test_size=10000, stratify=y, random_state=42)`. The saved test indices were never passed to the explainer.",
        f"- Fold summaries: five stratified folds are formed within the saved training data. For each fold, 2,000 rows are sampled stratified from that fold's training partition. The same saved baseline model is explained; these are sampling-sensitivity rankings, not rankings from five refitted fold models.",
        f"- SHAP: `shap.TreeExplainer` (TreeSHAP), raw multiclass tree outputs. Attributions are pooled as the arithmetic mean of absolute SHAP values over sampled rows and all seven output classes. Thus rows retain their natural stratified-sample class proportions and each class output is equally represented per row.",
        "- Batching: explanations are evaluated in batches of 250 rows to bound memory use.",
        "- Feature prior: `normalized_shap_prior = mean_abs_shap / max(mean_abs_shap)`.", "",
        "## XGBoost 3.x compatibility note", "",
        "SHAP 0.48.0's XGBoost loader expects a scalar `base_score`, while this saved multiclass XGBoost 3.2.0 model stores per-class base scores. The script applies an in-memory parser shim solely to allow TreeExplainer to read the tree structure; it does not change the saved model. Additivity checking is disabled because the shim substitutes one scalar expected-value offset. Feature attributions were cross-checked against XGBoost's native `pred_contribs` TreeSHAP output and matched exactly on a deterministic validation sample. The analysis uses feature contributions, not SHAP expected values.", "",
        "## Representation-level importance", "",
        "| Representation | Total SHAP | Contribution (%) | Mean feature importance |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.representation} | {row.total_shap:.8g} | {row.percentage_contribution:.3f}% | {row.mean_feature_importance:.8g} |")
    lines += ["", "## Global top 10 features", "", "| Rank | Feature | Representation | Channel | Sensor | Measurement | Mean absolute SHAP |", "|---:|---|---|---|---|---|---:|"]
    for r in global_top.itertuples(index=False):
        lines.append(f"| {r.global_rank} | {r.feature_name} | {r.representation} | {r.channel_id} | {r.sensor} | {r.measurement_name} | {r.mean_abs_shap:.8g} |")
    for rep in REPRESENTATIONS:
        lines += ["", f"## Top 10 {rep}", "", "| Global rank | Feature | Channel | Sensor | Measurement | Mean absolute SHAP |", "|---:|---|---|---|---|---:|"]
        for r in table[table.representation == rep].nlargest(10, "mean_abs_shap").itertuples(index=False):
            lines.append(f"| {r.global_rank} | {r.feature_name} | {r.channel_id} | {r.sensor} | {r.measurement_name} | {r.mean_abs_shap:.8g} |")
    lines += [
        "", "## Interpretation and limitations", "",
        "Representation percentages summarize model reliance under this fitted model and this training-only sample; they are not causal effects, and correlated channels can share or redistribute attribution. Global importance averages across the seven class outputs and the observed class mix. Fold files reuse one fitted model, so they prepare rank comparisons for later analysis but do not measure full model-refit variability. The operational representation labels should not be read as validated phenological stages. No test observations were used for SHAP or priors.", "",
        f"Total explanation runtime: {config['runtime_seconds']:.1f} seconds ({config['runtime_seconds']/60:.1f} minutes).", "",
        "## Artifacts", "",
        "Importance and sampling records are under `results/shap/`; figures are under `results/shap/figures/`. `configuration.json`, `shap_sample_indices.npy`, and `fold_sample_indices.npz` preserve key reproducibility details.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    df, metadata, features, train_ids, test_ids = validate_and_load()
    labels = df["label"].to_numpy(dtype=np.int64)
    train_classes = np.unique(labels[train_ids])
    if len(train_classes) != 7:
        raise ValueError("Saved training partition does not contain all seven classes.")
    main_ids = sample_stratified(train_ids, labels, MAIN_SAMPLE_SIZE, SEED)
    np.save(OUT / "shap_sample_indices.npy", main_ids, allow_pickle=False)

    import shap
    import xgboost as xgb
    from xgboost import XGBClassifier

    parse_multiclass_base_score_for_shap()
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    if model.n_features_in_ != len(features):
        raise ValueError("Saved model feature count does not match dataset metadata.")
    booster = model.get_booster()
    explainer = shap.TreeExplainer(model, model_output="raw", feature_perturbation="tree_path_dependent")

    X = df[features].to_numpy(dtype=np.float32, copy=False)
    # Deterministic package-to-native TreeSHAP parity check on training-only rows.
    check_ids = main_ids[:16]
    check_values = normalize_shap_array(explainer.shap_values(X[check_ids], check_additivity=False), len(features))
    native = booster.predict(xgb.DMatrix(X[check_ids]), pred_contribs=True)
    if native.ndim != 3 or native.shape[1] != len(train_classes) or native.shape[2] != len(features) + 1:
        raise ValueError(f"Unexpected XGBoost pred_contribs shape: {native.shape}.")
    parity_error = float(np.max(np.abs(check_values - native[:, :, :-1].transpose(0, 2, 1))))
    if parity_error > 1e-5:
        raise ValueError(f"SHAP/native TreeSHAP feature contribution mismatch: {parity_error}.")

    importance = mean_abs_shap(explainer, X[main_ids])
    table = write_tables(metadata, features, importance)
    summary = pd.read_csv(OUT / "representation_summary.csv")

    fold_sample_map: dict[str, np.ndarray] = {}
    fold_rows = []
    fold_files_valid = True
    splitter = StratifiedKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=SEED)
    for fold_number, (fold_train_pos, _) in enumerate(splitter.split(train_ids, labels[train_ids]), start=1):
        fold_train_ids = train_ids[fold_train_pos]
        selected_ids = sample_stratified(fold_train_ids, labels, FOLD_SAMPLE_SIZE, SEED + fold_number)
        fold_sample_map[f"fold_{fold_number}"] = selected_ids
        fold_importance = mean_abs_shap(explainer, X[selected_ids])
        fold_table = pd.DataFrame({"feature_name": features, "mean_abs_shap": fold_importance})
        fold_table["rank"] = fold_table["mean_abs_shap"].rank(method="min", ascending=False).astype(int)
        fold_table = metadata[["feature_name", "representation", "channel_id", "sensor", "measurement_name"]].merge(
            fold_table, on="feature_name", validate="one_to_one"
        )
        fold_table.sort_values("rank").to_csv(OUT / f"fold_{fold_number}_feature_importance.csv", index=False)
        fold_files_valid = fold_files_valid and (
            len(fold_table) == 261
            and fold_table["feature_name"].nunique() == 261
            and fold_table["rank"].notna().all()
            and fold_table["rank"].between(1, 261).all()
            and np.isfinite(fold_table["mean_abs_shap"].to_numpy()).all()
        )
        fold_rows.append({"fold": fold_number, "training_population_rows": len(fold_train_ids), "sample_rows": len(selected_ids)})
        print(f"Completed SHAP fold {fold_number}/{FOLD_COUNT} ({len(selected_ids)} rows)", flush=True)
    np.savez_compressed(OUT / "fold_sample_indices.npz", **fold_sample_map)

    make_plots(table, summary)
    runtime = time.perf_counter() - started
    baseline = json.loads(BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))
    config = {
        "analysis": "training_only_shap_for_saved_full_feature_baseline",
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "model_sha256": __import__("hashlib").sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "dataset_path": str(DATA_PATH.relative_to(ROOT)),
        "dataset_sha256": baseline["dataset"]["sha256"],
        "train_indices_path": str(TRAIN_PATH.relative_to(ROOT)),
        "test_indices_path": str(TEST_PATH.relative_to(ROOT)),
        "test_indices_used_for_shap": False,
        "seed": SEED,
        "main_sample_rows": len(main_ids),
        "main_sample_method": "sklearn train_test_split(population=train_indices, test_size=10000, stratify=training_labels, random_state=42)",
        "main_class_counts": {str(int(c)): int(np.count_nonzero(labels[main_ids] == c)) for c in train_classes},
        "fold_count": FOLD_COUNT,
        "fold_sample_rows": FOLD_SAMPLE_SIZE,
        "fold_method": "StratifiedKFold(shuffle=True, random_state=42) over saved training IDs; sample 2000 stratified IDs from each fold's training partition using random_state=42+fold_number; explain fixed baseline model",
        "fold_summaries_are_refitted_model_stability": False,
        "aggregation": "mean absolute SHAP pooled over sampled observations and all seven raw-output class dimensions; divide by n_rows*7",
        "feature_order": features,
        "representation_counts": {str(k): int(v) for k, v in metadata.representation.value_counts().items()},
        "model_configuration": baseline["model"],
        "shap_version": shap.__version__,
        "xgboost_version": xgb.__version__,
        "numpy_version": package_version("numpy"),
        "pandas_version": package_version("pandas"),
        "scikit_learn_version": package_version("scikit-learn"),
        "python_version": platform.python_version(),
        "shap_batch_size": SHAP_BATCH_SIZE,
        "native_treeshap_parity_rows": len(check_ids),
        "native_treeshap_max_abs_feature_contribution_error": parity_error,
        "runtime_seconds": runtime,
        "fold_samples": fold_rows,
        "validation": {
            "importance_count": int(len(importance)),
            "importance_finite": bool(np.isfinite(importance).all()),
            "prior_min": float(table.normalized_shap_prior.min()),
            "prior_max": float(table.normalized_shap_prior.max()),
            "all_ranks_present": bool(table.global_rank.notna().all() and table.global_rank.between(1, len(features)).all()),
            "all_fold_files_have_261_finite_ranked_features": bool(fold_files_valid),
            "metadata_join_exact": True,
            "test_observations_used": False,
            "feature_importance_values_nonnegative": bool((importance >= 0).all()),
        },
    }
    if not (config["validation"]["importance_count"] == 261 and config["validation"]["importance_finite"] and config["validation"]["prior_min"] >= 0 and config["validation"]["prior_max"] <= 1 and np.isclose(config["validation"]["prior_max"], 1) and config["validation"]["all_ranks_present"] and config["validation"]["all_fold_files_have_261_finite_ranked_features"]):
        raise ValueError("One or more SHAP output validation checks failed.")
    (OUT / "configuration.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    write_report(table, summary, config)
    print(json.dumps({
        "runtime_seconds": runtime,
        "main_sample_rows": len(main_ids),
        "representation_summary": summary.to_dict(orient="records"),
        "global_top_10": table.nsmallest(10, "global_rank")["feature_name"].tolist(),
        "validation": config["validation"],
        "output_directory": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
