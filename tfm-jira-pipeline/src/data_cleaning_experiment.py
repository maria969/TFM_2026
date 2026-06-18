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
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import load_config, normalize_string, project_path


RANDOM_STATE = 42
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


def quality_snapshot(df: pd.DataFrame, scenario: str, key_col: str | None, target_col: str) -> dict[str, Any]:
    duplicate_count = int(df.duplicated(subset=[key_col]).sum()) if key_col and key_col in df.columns else int(df.duplicated().sum())
    return {
        "scenario": scenario,
        "rows": len(df),
        "columns": len(df.columns),
        "total_nulls": int(df.isna().sum().sum()),
        "mean_null_rate": float(df.isna().mean().mean()),
        "target_null_rate": float(df[target_col].isna().mean()) if target_col in df.columns else np.nan,
        "duplicate_count": duplicate_count,
    }


def normalize_categorical_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cleaned = df.copy()
    for col in columns:
        if col not in cleaned.columns:
            continue
        cleaned[col] = cleaned[col].map(normalize_string)
        cleaned[col] = cleaned[col].fillna("__MISSING__")
    return cleaned


def cap_numeric_outliers(df: pd.DataFrame, columns: list[str], upper_quantile: float = 0.99) -> pd.DataFrame:
    cleaned = df.copy()
    for col in columns:
        if col not in cleaned.columns:
            continue
        numeric = pd.to_numeric(cleaned[col], errors="coerce")
        upper = numeric.quantile(upper_quantile)
        if pd.notna(upper):
            cleaned[col] = numeric.clip(lower=0, upper=upper)
    return cleaned


def remove_duplicate_keys(df: pd.DataFrame, key_col: str | None, created_col: str | None) -> pd.DataFrame:
    if not key_col or key_col not in df.columns:
        return df.drop_duplicates().copy()
    sort_cols = [key_col]
    if created_col and created_col in df.columns:
        sort_cols.append(created_col)
    return df.sort_values(sort_cols).drop_duplicates(subset=[key_col], keep="last").copy()


def build_clean_dataset(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = config["columns"]
    target_col = config["target"]["target_column"]
    categorical_features = [c for c in config["model"]["categorical_features"] if c in df.columns]
    numeric_features = [c for c in config["model"]["numeric_features"] if c in df.columns]

    actions = []
    before_rows = len(df)
    cleaned = normalize_categorical_columns(df, categorical_features)
    actions.append({"step": "normalize_categorical_missing_values", "detail": "Normaliza strings vacíos y valores nulos categóricos como __MISSING__."})

    cleaned = cap_numeric_outliers(cleaned, numeric_features, upper_quantile=0.99)
    actions.append({"step": "cap_numeric_outliers", "detail": "Limita variables numéricas al percentil 99 para reducir el efecto de outliers extremos."})

    cleaned = remove_duplicate_keys(cleaned, cols.get("key"), cols.get("created"))
    actions.append(
        {
            "step": "remove_duplicate_keys",
            "detail": f"Elimina duplicados por clave si existe la columna {cols.get('key')}.",
            "rows_before": before_rows,
            "rows_after": len(cleaned),
            "rows_removed": before_rows - len(cleaned),
        }
    )

    if target_col in cleaned.columns:
        unresolved = int(cleaned[target_col].isna().sum())
        actions.append(
            {
                "step": "preserve_unresolved_target_nulls",
                "detail": "Mantiene filas sin target para análisis de calidad, aunque se excluyen del entrenamiento.",
                "affected_rows": unresolved,
            }
        )

    return cleaned, pd.DataFrame(actions)


def train_and_evaluate(
    df: pd.DataFrame,
    scenario: str,
    config: dict[str, Any],
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
        random_state=int(config["model"]["random_state"]),
        stratify=y,
    )
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
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= RF_THRESHOLD).astype(int)
    return {
        "scenario": scenario,
        "threshold": RF_THRESHOLD,
        "rows": len(y_test),
        "positive_rate_true": float(np.mean(y_test)),
        "positive_rate_predicted": float(np.mean(y_pred)),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "pr_auc": average_precision_score(y_test, y_prob),
        "brier_score": brier_score_loss(y_test, y_prob),
        "ece": expected_calibration_error(y_test.to_numpy(), y_prob),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(project_path(args.config))
    reports_dir = project_path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    prepared_path = project_path(config["paths"]["prepared_csv"])
    cleaned_path = project_path("data/processed/jira_issues_cleaned.csv")
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(prepared_path)
    target_col = config["target"]["target_column"]
    key_col = config["columns"].get("key")

    cleaned, actions = build_clean_dataset(df, config)
    cleaned.to_csv(cleaned_path, index=False)

    quality = pd.DataFrame(
        [
            quality_snapshot(df, "prepared_original", key_col, target_col),
            quality_snapshot(cleaned, "cleaned_rules", key_col, target_col),
        ]
    )
    quality.to_csv(reports_dir / "data_cleaning_quality_comparison.csv", index=False)
    actions.to_csv(reports_dir / "data_cleaning_actions.csv", index=False)

    metrics = pd.DataFrame(
        [
            train_and_evaluate(df, "prepared_original", config),
            train_and_evaluate(cleaned, "cleaned_rules", config),
        ]
    )
    baseline = metrics.loc[metrics["scenario"] == "prepared_original"].iloc[0]
    for metric in ["precision", "recall", "f1", "balanced_accuracy", "pr_auc", "brier_score", "ece", "roc_auc"]:
        metrics[f"delta_{metric}_vs_original"] = metrics[metric] - baseline[metric]
    metrics.to_csv(reports_dir / "data_cleaning_metrics.csv", index=False)

    print(f"Experimento de limpieza guardado en {reports_dir}")
    print("Archivos generados:")
    print("- data/processed/jira_issues_cleaned.csv")
    print("- data_cleaning_quality_comparison.csv")
    print("- data_cleaning_actions.csv")
    print("- data_cleaning_metrics.csv")


if __name__ == "__main__":
    main()
