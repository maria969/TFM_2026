from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import load_config, project_path


RF_THRESHOLD = 0.25


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        mask = (y_prob >= lower) & (y_prob <= upper) if i == 0 else (y_prob > lower) & (y_prob <= upper)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(float(np.mean(y_true[mask])) - float(np.mean(y_prob[mask])))
    return float(ece)


def make_preprocessor(categorical_features: list[str], numeric_features: list[str]) -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_features),
            ("num", numeric_pipeline, numeric_features),
        ]
    )


def evaluate_once(
    df: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    target_col = config["target"]["target_column"]
    categorical_features = [c for c in config["model"]["categorical_features"] if c in df.columns]
    numeric_features = [c for c in config["model"]["numeric_features"] if c in df.columns]
    features = categorical_features + numeric_features

    model_df = df.dropna(subset=[target_col]).copy()
    X = model_df[features]
    y = model_df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(config["model"]["test_size"]),
        random_state=seed,
        stratify=y,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(categorical_features, numeric_features)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=seed,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= RF_THRESHOLD).astype(int)
    return {
        "seed": seed,
        "threshold": RF_THRESHOLD,
        "rows": len(y_test),
        "positive_rate_true": float(np.mean(y_test)),
        "positive_rate_predicted": float(np.mean(y_pred)),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_test, y_prob),
        "brier_score": brier_score_loss(y_test, y_prob),
        "ece": expected_calibration_error(y_test.to_numpy(), y_prob),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    config = load_config(project_path(args.config))
    reports_dir = project_path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(project_path(config["paths"]["prepared_csv"]))
    base_seed = int(config["model"]["random_state"])
    seeds = [base_seed + i for i in range(args.seeds)]

    metrics = pd.DataFrame([evaluate_once(df, config, seed) for seed in seeds])
    metrics.to_csv(reports_dir / "stability_seed_metrics.csv", index=False)

    metric_cols = ["precision", "recall", "f1", "pr_auc", "brier_score", "ece", "roc_auc"]
    summary = (
        metrics[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .transpose()
        .reset_index()
        .rename(columns={"index": "metric"})
    )
    summary["range"] = summary["max"] - summary["min"]
    summary["coefficient_of_variation"] = summary["std"] / summary["mean"].replace(0, np.nan)
    summary.to_csv(reports_dir / "stability_seed_summary.csv", index=False)

    print(f"Experimento de estabilidad guardado en {reports_dir}")
    print("Archivos generados:")
    print("- stability_seed_metrics.csv")
    print("- stability_seed_summary.csv")


if __name__ == "__main__":
    main()
