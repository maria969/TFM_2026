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
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import load_config, project_path


DEFAULT_THRESHOLDS = {
    "logistic_regression": 0.50,
    "random_forest": 0.25,
}


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        if i == 0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob > lower) & (y_prob <= upper)
        if not np.any(mask):
            continue
        bin_accuracy = float(np.mean(y_true[mask]))
        bin_confidence = float(np.mean(y_prob[mask]))
        ece += float(np.mean(mask)) * abs(bin_accuracy - bin_confidence)
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


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def classification_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "model": model_name,
        "threshold": threshold,
        "rows": int(len(y_true)),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_predicted": float(np.mean(y_pred)),
        "accuracy": accuracy_score(y_true, y_pred),
        "error_rate": 1.0 - accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": safe_divide(tn, tn + fp),
        "false_positive_rate": safe_divide(fp, fp + tn),
        "false_negative_rate": safe_divide(fn, fn + tp),
        "negative_predictive_value": safe_divide(tn, tn + fn),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
    }


def calibration_bins(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        if i == 0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob > lower) & (y_prob <= upper)
        bin_size = int(mask.sum())
        if bin_size == 0:
            rows.append(
                {
                    "model": model_name,
                    "bin": i + 1,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "rows": 0,
                    "mean_predicted_probability": np.nan,
                    "observed_positive_rate": np.nan,
                    "calibration_gap": np.nan,
                    "bin_weight": 0.0,
                }
            )
            continue
        mean_prob = float(np.mean(y_prob[mask]))
        observed_rate = float(np.mean(y_true[mask]))
        rows.append(
            {
                "model": model_name,
                "bin": i + 1,
                "bin_lower": lower,
                "bin_upper": upper,
                "rows": bin_size,
                "mean_predicted_probability": mean_prob,
                "observed_positive_rate": observed_rate,
                "calibration_gap": observed_rate - mean_prob,
                "bin_weight": float(bin_size / len(y_true)),
            }
        )
    return pd.DataFrame(rows)


def load_thresholds(reports_dir) -> dict[str, float]:
    threshold_file = reports_dir / "threshold_best_f1.csv"
    thresholds = DEFAULT_THRESHOLDS.copy()
    if threshold_file.exists():
        threshold_df = pd.read_csv(threshold_file)
        for _, row in threshold_df.iterrows():
            thresholds[str(row["model"])] = float(row["threshold"])
    return thresholds


def build_models(random_state: int) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--calibration-bins", type=int, default=10)
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

    thresholds = load_thresholds(reports_dir)
    metrics_rows = []
    calibration_frames = []
    prediction_frames = []

    for model_name, estimator in build_models(int(config["model"]["random_state"])).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(categorical_features, numeric_features)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        threshold = thresholds.get(model_name, 0.5)
        y_pred = (y_prob >= threshold).astype(int)

        metrics_rows.append(classification_metrics(model_name, y_test.to_numpy(), y_prob, threshold))
        calibration_frames.append(calibration_bins(model_name, y_test.to_numpy(), y_prob, args.calibration_bins))

        pred_df = model_df.loc[X_test.index, ["key", "created"] if "key" in model_df.columns and "created" in model_df.columns else []].copy()
        pred_df["model"] = model_name
        pred_df["y_true"] = y_test.to_numpy()
        pred_df["y_probability"] = y_prob
        pred_df["threshold"] = threshold
        pred_df["y_predicted"] = y_pred
        pred_df["absolute_probability_error"] = np.abs(pred_df["y_true"] - pred_df["y_probability"])
        prediction_frames.append(pred_df)

    pd.DataFrame(metrics_rows).to_csv(reports_dir / "advanced_model_metrics.csv", index=False)
    pd.concat(calibration_frames, ignore_index=True).to_csv(reports_dir / "calibration_bins.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(reports_dir / "prediction_audit.csv", index=False)

    print(f"Métricas avanzadas guardadas en {reports_dir}")
    print("Archivos generados:")
    print("- advanced_model_metrics.csv")
    print("- calibration_bins.csv")
    print("- prediction_audit.csv")


if __name__ == "__main__":
    main()
