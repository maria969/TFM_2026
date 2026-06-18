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


def evaluate_model(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    metrics = {
        "model": model_name,
        "rows": len(y_true),
        "positive_rate": float(np.mean(y_true)),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = np.nan
    return metrics


def subgroup_metrics(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    subgroup_features: list[str],
    model_name: str,
) -> pd.DataFrame:
    rows = []
    eval_df = df_test.copy()
    eval_df["_y_true"] = y_true
    eval_df["_y_pred"] = y_pred
    eval_df["_y_prob"] = y_prob

    for feature in subgroup_features:
        if feature not in eval_df.columns:
            continue
        for value, group in eval_df.groupby(feature, dropna=False):
            if len(group) < 30:
                continue
            rows.append(
                {
                    "model": model_name,
                    "subgroup_feature": feature,
                    "subgroup_value": value,
                    "rows": len(group),
                    "positive_rate": float(group["_y_true"].mean()),
                    "precision": precision_score(group["_y_true"], group["_y_pred"], zero_division=0),
                    "recall": recall_score(group["_y_true"], group["_y_pred"], zero_division=0),
                    "f1": f1_score(group["_y_true"], group["_y_pred"], zero_division=0),
                    "pr_auc": average_precision_score(group["_y_true"], group["_y_prob"]),
                    "brier_score": brier_score_loss(group["_y_true"], group["_y_prob"]),
                    "ece": expected_calibration_error(
                        group["_y_true"].to_numpy(),
                        group["_y_prob"].to_numpy(),
                    ),
                }
            )
    return pd.DataFrame(rows)


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

    metrics_rows = []
    subgroup_frames = []

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics_rows.append(evaluate_model(model_name, y_test.to_numpy(), y_pred, y_prob))

        df_test = model_df.loc[X_test.index].copy()
        subgroup_frames.append(
            subgroup_metrics(
                df_test,
                y_test.to_numpy(),
                y_pred,
                y_prob,
                config["model"]["subgroup_features"],
                model_name,
            )
        )

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(reports_dir / "model_metrics.csv", index=False)

    subgroups = pd.concat(subgroup_frames, ignore_index=True) if subgroup_frames else pd.DataFrame()
    subgroups.to_csv(reports_dir / "subgroup_metrics.csv", index=False)

    calibration = metrics[["model", "brier_score", "ece"]].copy()
    calibration.to_csv(reports_dir / "calibration_summary.csv", index=False)

    print(f"Modelos entrenados. Métricas guardadas en {reports_dir}")


if __name__ == "__main__":
    main()

