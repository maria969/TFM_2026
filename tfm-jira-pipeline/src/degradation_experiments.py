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


def build_model(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )


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


def set_random_nulls(df: pd.DataFrame, column: str, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    degraded = df.copy()
    mask = rng.random(len(degraded)) < rate
    degraded.loc[mask, column] = np.nan
    return degraded


def degrade_text_features(df: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    degraded = df.copy()
    mask = rng.random(len(degraded)) < rate
    if "description_length" in degraded.columns:
        degraded.loc[mask, "description_length"] = 0
    if "has_description" in degraded.columns:
        degraded.loc[mask, "has_description"] = 0
    return degraded


def add_priority_noise(df: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    degraded = df.copy()
    if "priority" not in degraded.columns:
        return degraded
    valid_priorities = degraded["priority"].dropna().unique()
    if len(valid_priorities) == 0:
        return degraded
    mask = rng.random(len(degraded)) < rate
    degraded.loc[mask, "priority"] = rng.choice(valid_priorities, size=int(mask.sum()), replace=True)
    return degraded


def remove_positive_training_examples(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    positive_keep_rate: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.Series]:
    positive_mask = y_train == 1
    keep_positive = rng.random(len(y_train)) < positive_keep_rate
    keep_mask = (~positive_mask) | keep_positive
    return X_train.loc[keep_mask].copy(), y_train.loc[keep_mask].copy()


def fit_predict(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: list[str],
    numeric_features: list[str],
    random_state: int,
) -> np.ndarray:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(categorical_features, numeric_features)),
            ("model", build_model(random_state)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline.predict_proba(X_test)[:, 1]


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

    scenarios: list[dict[str, Any]] = [
        {
            "scenario": "baseline",
            "description": "Datos originales preparados, sin degradación artificial.",
            "degradation_type": "none",
            "rate": 0.0,
        },
        {
            "scenario": "priority_null_10",
            "description": "Se elimina priority en el 10% de train y test.",
            "degradation_type": "priority_null",
            "rate": 0.10,
        },
        {
            "scenario": "priority_null_20",
            "description": "Se elimina priority en el 20% de train y test.",
            "degradation_type": "priority_null",
            "rate": 0.20,
        },
        {
            "scenario": "priority_null_30",
            "description": "Se elimina priority en el 30% de train y test.",
            "degradation_type": "priority_null",
            "rate": 0.30,
        },
        {
            "scenario": "description_loss_20",
            "description": "Se simula pérdida textual en el 20% de train y test poniendo description_length=0 y has_description=0.",
            "degradation_type": "description_loss",
            "rate": 0.20,
        },
        {
            "scenario": "description_loss_40",
            "description": "Se simula pérdida textual en el 40% de train y test poniendo description_length=0 y has_description=0.",
            "degradation_type": "description_loss",
            "rate": 0.40,
        },
        {
            "scenario": "priority_noise_20",
            "description": "Se cambia aleatoriamente priority en el 20% de train y test.",
            "degradation_type": "priority_noise",
            "rate": 0.20,
        },
        {
            "scenario": "priority_noise_40",
            "description": "Se cambia aleatoriamente priority en el 40% de train y test.",
            "degradation_type": "priority_noise",
            "rate": 0.40,
        },
        {
            "scenario": "positive_class_undersample_50",
            "description": "Se conserva solo el 50% de ejemplos positivos en entrenamiento; el test no se altera.",
            "degradation_type": "positive_undersample_train",
            "rate": 0.50,
        },
    ]

    rows = []
    rng_master = np.random.default_rng(RANDOM_STATE)

    for idx, scenario in enumerate(scenarios):
        rng = np.random.default_rng(int(rng_master.integers(0, 1_000_000)))
        X_train_s = X_train.copy()
        X_test_s = X_test.copy()
        y_train_s = y_train.copy()

        degradation_type = scenario["degradation_type"]
        rate = float(scenario["rate"])

        if degradation_type == "priority_null":
            X_train_s = set_random_nulls(X_train_s, "priority", rate, rng)
            X_test_s = set_random_nulls(X_test_s, "priority", rate, rng)
        elif degradation_type == "description_loss":
            X_train_s = degrade_text_features(X_train_s, rate, rng)
            X_test_s = degrade_text_features(X_test_s, rate, rng)
        elif degradation_type == "priority_noise":
            X_train_s = add_priority_noise(X_train_s, rate, rng)
            X_test_s = add_priority_noise(X_test_s, rate, rng)
        elif degradation_type == "positive_undersample_train":
            X_train_s, y_train_s = remove_positive_training_examples(
                X_train_s,
                y_train_s,
                positive_keep_rate=rate,
                rng=rng,
            )

        y_prob = fit_predict(
            X_train_s,
            X_test_s,
            y_train_s,
            categorical_features,
            numeric_features,
            random_state=RANDOM_STATE + idx,
        )
        metrics = evaluate(y_test.to_numpy(), y_prob, RF_THRESHOLD)
        metrics.update(
            {
                "scenario": scenario["scenario"],
                "description": scenario["description"],
                "degradation_type": degradation_type,
                "degradation_rate": rate,
                "train_rows": len(X_train_s),
                "test_rows": len(X_test_s),
                "train_positive_rate": float(y_train_s.mean()),
                "test_positive_rate": float(y_test.mean()),
            }
        )
        rows.append(metrics)

    results = pd.DataFrame(rows)
    ordered_cols = [
        "scenario",
        "description",
        "degradation_type",
        "degradation_rate",
        "threshold",
        "train_rows",
        "test_rows",
        "train_positive_rate",
        "test_positive_rate",
        "positive_rate_predicted",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "brier_score",
        "ece",
        "roc_auc",
    ]
    results = results[ordered_cols]

    baseline = results.loc[results["scenario"] == "baseline"].iloc[0]
    for metric in ["precision", "recall", "f1", "pr_auc", "brier_score", "ece", "roc_auc"]:
        results[f"delta_{metric}"] = results[metric] - baseline[metric]

    results.to_csv(reports_dir / "degradation_experiments.csv", index=False)

    summary_cols = [
        "scenario",
        "degradation_type",
        "degradation_rate",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "brier_score",
        "ece",
        "roc_auc",
        "delta_f1",
        "delta_recall",
        "delta_ece",
    ]
    results[summary_cols].to_csv(reports_dir / "degradation_summary.csv", index=False)

    print(f"Experimentos de degradación guardados en {reports_dir}")
    print("Archivos generados:")
    print("- degradation_experiments.csv")
    print("- degradation_summary.csv")


if __name__ == "__main__":
    main()
