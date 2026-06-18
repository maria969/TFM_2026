from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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


RANDOM_STATE = 42
RF_THRESHOLD = 0.25


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


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("placeholder", "passthrough"),
        ]
    )


def train_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: list[str],
    numeric_features: list[str],
) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(categorical_features, numeric_features)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
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


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty:
        return np.nan
    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return np.nan
    expected_counts, _ = np.histogram(expected, bins=quantiles)
    actual_counts, _ = np.histogram(actual, bins=quantiles)
    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, None)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def categorical_distribution_shift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in train.columns or col not in test.columns:
            continue
        train_dist = train[col].fillna("__NULL__").value_counts(normalize=True)
        test_dist = test[col].fillna("__NULL__").value_counts(normalize=True)
        values = set(train_dist.head(top_n).index).union(set(test_dist.head(top_n).index))
        for value in sorted(values, key=lambda x: str(x)):
            train_rate = float(train_dist.get(value, 0.0))
            test_rate = float(test_dist.get(value, 0.0))
            rows.append(
                {
                    "column": col,
                    "value": value,
                    "train_rate": train_rate,
                    "test_rate": test_rate,
                    "delta_rate": test_rate - train_rate,
                    "abs_delta_rate": abs(test_rate - train_rate),
                }
            )
    return pd.DataFrame(rows).sort_values(["column", "abs_delta_rate"], ascending=[True, False])


def split_temporal(model_df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_df = model_df.sort_values("created").copy()
    split_idx = int(len(sorted_df) * (1 - test_fraction))
    train_df = sorted_df.iloc[:split_idx].copy()
    test_df = sorted_df.iloc[split_idx:].copy()
    return train_df, test_df


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

    if "created" not in df.columns:
        raise ValueError("El dataset preparado debe contener la columna 'created'.")

    df["created"] = pd.to_datetime(df["created"], errors="coerce", utc=True)
    model_df = df.dropna(subset=[target_col, "created"]).copy()
    model_df[target_col] = model_df[target_col].astype(int)

    X = model_df[features]
    y = model_df[target_col]
    X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
        X,
        y,
        test_size=float(config["model"]["test_size"]),
        random_state=int(config["model"]["random_state"]),
        stratify=y,
    )

    random_model = train_rf(X_train_random, y_train_random, categorical_features, numeric_features)
    random_prob = random_model.predict_proba(X_test_random)[:, 1]
    random_metrics = evaluate(y_test_random.to_numpy(), random_prob, RF_THRESHOLD)
    random_metrics.update(
        {
            "split_type": "random_split",
            "train_rows": len(X_train_random),
            "test_rows": len(X_test_random),
            "train_start": str(model_df.loc[X_train_random.index, "created"].min()),
            "train_end": str(model_df.loc[X_train_random.index, "created"].max()),
            "test_start": str(model_df.loc[X_test_random.index, "created"].min()),
            "test_end": str(model_df.loc[X_test_random.index, "created"].max()),
        }
    )

    train_temporal, test_temporal = split_temporal(model_df, float(config["model"]["test_size"]))
    X_train_temporal = train_temporal[features]
    y_train_temporal = train_temporal[target_col]
    X_test_temporal = test_temporal[features]
    y_test_temporal = test_temporal[target_col]

    temporal_model = train_rf(X_train_temporal, y_train_temporal, categorical_features, numeric_features)
    temporal_prob = temporal_model.predict_proba(X_test_temporal)[:, 1]
    temporal_metrics = evaluate(y_test_temporal.to_numpy(), temporal_prob, RF_THRESHOLD)
    temporal_metrics.update(
        {
            "split_type": "temporal_split",
            "train_rows": len(X_train_temporal),
            "test_rows": len(X_test_temporal),
            "train_start": str(train_temporal["created"].min()),
            "train_end": str(train_temporal["created"].max()),
            "test_start": str(test_temporal["created"].min()),
            "test_end": str(test_temporal["created"].max()),
        }
    )

    metrics = pd.DataFrame([random_metrics, temporal_metrics])
    random_row = metrics.loc[metrics["split_type"] == "random_split"].iloc[0]
    for metric in ["precision", "recall", "f1", "pr_auc", "brier_score", "ece", "roc_auc"]:
        metrics[f"delta_vs_random_{metric}"] = metrics[metric] - random_row[metric]
    metrics.to_csv(reports_dir / "temporal_drift_metrics.csv", index=False)

    numeric_shift_rows = []
    for col in numeric_features + [target_col]:
        if col in train_temporal.columns and col in test_temporal.columns:
            numeric_shift_rows.append(
                {
                    "column": col,
                    "train_mean": pd.to_numeric(train_temporal[col], errors="coerce").mean(),
                    "test_mean": pd.to_numeric(test_temporal[col], errors="coerce").mean(),
                    "delta_mean": pd.to_numeric(test_temporal[col], errors="coerce").mean()
                    - pd.to_numeric(train_temporal[col], errors="coerce").mean(),
                    "psi": psi(train_temporal[col], test_temporal[col]),
                }
            )
    pd.DataFrame(numeric_shift_rows).to_csv(reports_dir / "temporal_numeric_shift.csv", index=False)

    cat_shift = categorical_distribution_shift(train_temporal, test_temporal, categorical_features)
    cat_shift.to_csv(reports_dir / "temporal_categorical_shift.csv", index=False)

    period_summary = pd.DataFrame(
        [
            {
                "split": "temporal_train",
                "rows": len(train_temporal),
                "start": train_temporal["created"].min(),
                "end": train_temporal["created"].max(),
                "positive_rate": train_temporal[target_col].mean(),
            },
            {
                "split": "temporal_test",
                "rows": len(test_temporal),
                "start": test_temporal["created"].min(),
                "end": test_temporal["created"].max(),
                "positive_rate": test_temporal[target_col].mean(),
            },
        ]
    )
    period_summary.to_csv(reports_dir / "temporal_period_summary.csv", index=False)

    print(f"Análisis de drift temporal guardado en {reports_dir}")
    print("Archivos generados:")
    print("- temporal_drift_metrics.csv")
    print("- temporal_numeric_shift.csv")
    print("- temporal_categorical_shift.csv")
    print("- temporal_period_summary.csv")


if __name__ == "__main__":
    main()
