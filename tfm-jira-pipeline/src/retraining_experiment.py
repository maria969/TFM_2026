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


def train_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: list[str],
    numeric_features: list[str],
    random_state: int,
) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(categorical_features, numeric_features)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=random_state,
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
    result = {
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
    }
    try:
        result["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        result["roc_auc"] = np.nan
    return result


def add_time_bins(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    sorted_df = df.sort_values("created").copy()
    sorted_df["time_bin"] = pd.qcut(
        np.arange(len(sorted_df)),
        q=n_bins,
        labels=[f"period_{i+1}" for i in range(n_bins)],
    )
    return sorted_df


def period_summary(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    return (
        df.groupby("time_bin", observed=True)
        .agg(
            rows=(target_col, "size"),
            start=("created", "min"),
            end=("created", "max"),
            positive_rate=(target_col, "mean"),
        )
        .reset_index()
    )


def evaluate_strategy(
    strategy: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    target_col: str,
    categorical_features: list[str],
    numeric_features: list[str],
    test_period: str,
    random_state: int,
) -> dict[str, Any]:
    X_train = train_df[features]
    y_train = train_df[target_col].astype(int)
    X_test = test_df[features]
    y_test = test_df[target_col].astype(int)

    model = train_rf(X_train, y_train, categorical_features, numeric_features, random_state=random_state)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test.to_numpy(), y_prob, RF_THRESHOLD)
    metrics.update(
        {
            "strategy": strategy,
            "test_period": test_period,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_start": train_df["created"].min(),
            "train_end": train_df["created"].max(),
            "test_start": test_df["created"].min(),
            "test_end": test_df["created"].max(),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(project_path(args.config))
    reports_dir = project_path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(project_path(config["paths"]["prepared_csv"]))
    target_col = config["target"]["target_column"]
    if "created" not in df.columns:
        raise ValueError("El dataset preparado debe contener la columna 'created'.")

    df["created"] = pd.to_datetime(df["created"], errors="coerce", utc=True)
    model_df = df.dropna(subset=[target_col, "created"]).copy()
    model_df[target_col] = model_df[target_col].astype(int)

    categorical_features = [c for c in config["model"]["categorical_features"] if c in model_df.columns]
    numeric_features = [c for c in config["model"]["numeric_features"] if c in model_df.columns]
    features = categorical_features + numeric_features

    binned = add_time_bins(model_df, n_bins=5)
    periods = list(binned["time_bin"].cat.categories)
    period_summary(binned, target_col).to_csv(reports_dir / "retraining_period_summary.csv", index=False)

    initial_periods = periods[:3]
    future_periods = periods[3:]
    rows = []
    static_train = binned[binned["time_bin"].isin(initial_periods)].copy()
    for test_period in future_periods:
        test_df = binned[binned["time_bin"] == test_period].copy()
        period_seed = RANDOM_STATE + periods.index(test_period)
        rows.append(
            evaluate_strategy(
                "static_initial_train",
                static_train,
                test_df,
                features,
                target_col,
                categorical_features,
                numeric_features,
                test_period,
                period_seed,
            )
        )

        prior_periods = [p for p in periods if periods.index(p) < periods.index(test_period)]
        cumulative_train = binned[binned["time_bin"].isin(prior_periods)].copy()
        rows.append(
            evaluate_strategy(
                "cumulative_retraining",
                cumulative_train,
                test_df,
                features,
                target_col,
                categorical_features,
                numeric_features,
                test_period,
                period_seed,
            )
        )

        sliding_periods = prior_periods[-2:]
        sliding_train = binned[binned["time_bin"].isin(sliding_periods)].copy()
        rows.append(
            evaluate_strategy(
                "sliding_window_2_periods",
                sliding_train,
                test_df,
                features,
                target_col,
                categorical_features,
                numeric_features,
                test_period,
                period_seed,
            )
        )

    metrics = pd.DataFrame(rows)

    static_by_period = metrics[metrics["strategy"] == "static_initial_train"][
        [
            "test_period",
            "precision",
            "recall",
            "f1",
            "pr_auc",
            "brier_score",
            "ece",
            "roc_auc",
        ]
    ].rename(
        columns={
            "precision": "static_precision",
            "recall": "static_recall",
            "f1": "static_f1",
            "pr_auc": "static_pr_auc",
            "brier_score": "static_brier_score",
            "ece": "static_ece",
            "roc_auc": "static_roc_auc",
        }
    )
    metrics = metrics.merge(static_by_period, on="test_period", how="left")
    for metric in ["precision", "recall", "f1", "pr_auc", "brier_score", "ece", "roc_auc"]:
        metrics[f"delta_vs_static_{metric}"] = metrics[metric] - metrics[f"static_{metric}"]

    metrics.to_csv(reports_dir / "retraining_metrics.csv", index=False)

    summary = (
        metrics.groupby("strategy")
        .agg(
            periods=("test_period", "nunique"),
            avg_precision=("precision", "mean"),
            avg_recall=("recall", "mean"),
            avg_f1=("f1", "mean"),
            avg_pr_auc=("pr_auc", "mean"),
            avg_brier_score=("brier_score", "mean"),
            avg_ece=("ece", "mean"),
            avg_roc_auc=("roc_auc", "mean"),
        )
        .reset_index()
    )
    static_summary = summary[summary["strategy"] == "static_initial_train"].iloc[0]
    for metric in ["precision", "recall", "f1", "pr_auc", "brier_score", "ece", "roc_auc"]:
        summary[f"delta_avg_{metric}_vs_static"] = summary[f"avg_{metric}"] - static_summary[f"avg_{metric}"]

    summary.to_csv(reports_dir / "retraining_summary.csv", index=False)

    print(f"Experimento de reentrenamiento guardado en {reports_dir}")
    print("Archivos generados:")
    print("- retraining_period_summary.csv")
    print("- retraining_metrics.csv")
    print("- retraining_summary.csv")


if __name__ == "__main__":
    main()
