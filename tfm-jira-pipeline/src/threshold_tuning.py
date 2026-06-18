from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import load_config, project_path


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        mask = (y_prob > lower) & (y_prob <= upper)
        if not np.any(mask):
            continue
        bin_accuracy = np.mean(y_true[mask])
        bin_confidence = np.mean(y_prob[mask])
        ece += np.mean(mask) * abs(bin_accuracy - bin_confidence)
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


def evaluate_thresholds(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                "rows": len(y_true),
                "positive_rate_true": float(np.mean(y_true)),
                "positive_rate_predicted": float(np.mean(y_pred)),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "pr_auc": average_precision_score(y_true, y_prob),
                "brier_score": brier_score_loss(y_true, y_prob),
                "ece": expected_calibration_error(y_true, y_prob),
                "roc_auc": roc_auc_score(y_true, y_prob),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(project_path(args.config))
    reports_dir = project_path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(project_path(config["paths"]["prepared_csv"]))
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
        random_state=int(config["model"]["random_state"]),
        stratify=y,
    )

    preprocessor = make_preprocessor(categorical_features, numeric_features)
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=int(config["model"]["random_state"]),
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
    }
    thresholds = [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]

    rows = []
    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        rows.extend(evaluate_thresholds(model_name, y_test.to_numpy(), y_prob, thresholds))

    results = pd.DataFrame(rows)
    results.to_csv(reports_dir / "threshold_metrics.csv", index=False)

    best_f1 = results.loc[results.groupby("model")["f1"].idxmax()].copy()
    best_recall_f1_tradeoff = (
        results[results["recall"] >= 0.60]
        .sort_values(["model", "f1"], ascending=[True, False])
        .groupby("model")
        .head(1)
    )

    best_f1.to_csv(reports_dir / "threshold_best_f1.csv", index=False)
    best_recall_f1_tradeoff.to_csv(reports_dir / "threshold_recall_60_minimum.csv", index=False)

    print(f"Análisis de umbrales guardado en {reports_dir}")
    print("Archivos generados:")
    print("- threshold_metrics.csv")
    print("- threshold_best_f1.csv")
    print("- threshold_recall_60_minimum.csv")


if __name__ == "__main__":
    main()
