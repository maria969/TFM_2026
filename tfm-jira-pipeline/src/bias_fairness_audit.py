#!/usr/bin/env python3
"""
Auditoría de fairness operacional por subgrupos.

Calcula métricas globales, métricas por grupo y brechas entre grupos.
Está diseñado para el TFM sobre calidad de datos, sesgos y fiabilidad de IA.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


TARGET = "late_resolution"
DEFAULT_GROUP_COLUMNS = [
    "project_key",
    "issuetype",
    "priority",
    "status",
    "has_description",
    "created_year",
    "created_month",
]


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_dataset(config: dict, root: Path) -> Path:
    configured = (
        config.get("data", {}).get("modeling_dataset")
        or config.get("paths", {}).get("modeling_dataset")
        or config.get("dataset_path")
    )
    candidates = []
    if configured:
        candidates.append(root / configured)
        candidates.append(Path(configured))
    candidates.extend(
        [
            root / "data/processed/modeling_dataset.csv",
            root / "data/modeling_dataset.csv",
            root / "data/processed/jira_modeling_dataset.csv",
            root / "outputs/modeling_dataset.csv",
            root / "data/processed/features.csv",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No se encontró el dataset tabular. Añade data.modeling_dataset en config/config.yaml."
    )


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def metric_row(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    fnr = fn / (fn + tp) if (fn + tp) else np.nan
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = np.nan
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = np.nan
    try:
        brier = brier_score_loss(y_true, y_prob)
    except Exception:
        brier = np.nan
    return {
        "rows": len(y_true),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_predicted": float(np.mean(y_pred)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "ece": expected_calibration_error(y_true, y_prob),
    }


def build_preprocessor(df: pd.DataFrame, target: str, groups: list[str]) -> tuple[ColumnTransformer, list[str]]:
    excluded = {
        target,
        "_split",
        # Variables derivadas del resultado o de resolución: se excluyen para evitar fuga de información.
        "resolutiondate",
        "time_to_resolution_days",
    }
    feature_cols = [c for c in df.columns if c not in excluded]
    X = df[feature_cols]

    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]

    # Solo se codifican categóricas de baja/mediana cardinalidad.
    # Identificadores, URLs, textos largos y columnas con valores casi únicos se descartan.
    categorical = []
    for c in feature_cols:
        if c in numeric:
            continue
        nunique = X[c].nunique(dropna=True)
        c_lower = c.lower()
        looks_like_identifier_or_text = any(
            token in c_lower
            for token in [
                "id",
                "url",
                "self",
                "avatar",
                "email",
                "account",
                "display",
                "description",
                "summary",
                "body",
                "comment",
            ]
        )
        if nunique <= 50 and not looks_like_identifier_or_text:
            categorical.append(c)

    feature_cols = numeric + categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ],
        remainder="drop",
    )
    return preprocessor, feature_cols


def model_zoo(random_state: int) -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight=None, random_state=random_state),
        "logistic_regression_balanced": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=random_state, n_jobs=-1),
        "random_forest_balanced": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=random_state, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, random_state=random_state, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=random_state),
        "mlp_classifier": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=random_state, early_stopping=True),
    }


def disparity_summary(group_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["precision", "recall", "f1", "fpr", "fnr", "positive_rate_predicted", "brier_score", "ece", "roc_auc", "pr_auc"]
    for (model, group_col), sub in group_metrics.groupby(["model", "group_feature"]):
        eligible = sub[sub["rows"] >= 30].copy()
        if eligible.empty:
            continue
        for metric in metrics:
            values = eligible[metric].dropna()
            if values.empty:
                continue
            max_value = values.max()
            min_value = values.min()
            ratio = min_value / max_value if max_value not in [0, np.nan] and max_value != 0 else np.nan
            worst_idx = eligible[metric].idxmin()
            best_idx = eligible[metric].idxmax()
            rows.append(
                {
                    "model": model,
                    "group_feature": group_col,
                    "metric": metric,
                    "groups_considered": len(eligible),
                    "min_value": min_value,
                    "max_value": max_value,
                    "absolute_gap": max_value - min_value,
                    "min_max_ratio": ratio,
                    "worst_group": eligible.loc[worst_idx, "group_value"],
                    "best_group": eligible.loc[best_idx, "group_value"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--sample", type=int, default=0, help="Opcional: limitar filas para pruebas rápidas.")
    args = parser.parse_args()

    root = Path.cwd()
    config = load_config(args.config)
    reports_dir = root / "reports"
    reports_dir.mkdir(exist_ok=True)
    dataset_path = find_dataset(config, root)

    df = pd.read_csv(dataset_path)
    if TARGET not in df.columns:
        raise ValueError(f"No existe la variable objetivo {TARGET} en {dataset_path}")
    df = df[df[TARGET].notna()].copy()
    df[TARGET] = df[TARGET].astype(int)
    if args.sample and len(df) > args.sample:
        df = df.sample(args.sample, random_state=42)

    group_cols = [c for c in DEFAULT_GROUP_COLUMNS if c in df.columns]
    if not group_cols:
        raise ValueError("No se encontró ninguna columna de grupo para auditoría.")

    preprocessor, feature_cols = build_preprocessor(df, TARGET, group_cols)
    X = df[feature_cols]
    y = df[TARGET].astype(int)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx].to_numpy(), y.iloc[test_idx].to_numpy()
    test_df = df.iloc[test_idx].reset_index(drop=True)

    global_rows = []
    group_rows = []
    for model_name, estimator in model_zoo(42).items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        global_metric = metric_row(y_test, y_prob, threshold=args.threshold)
        global_metric.update({"model": model_name, "threshold": args.threshold})
        global_rows.append(global_metric)

        for group_col in group_cols:
            values = test_df[group_col].fillna("__NULL__").astype(str)
            for group_value, idx in values.groupby(values).groups.items():
                if len(idx) < 10:
                    continue
                row = metric_row(y_test[list(idx)], y_prob[list(idx)], threshold=args.threshold)
                row.update(
                    {
                        "model": model_name,
                        "threshold": args.threshold,
                        "group_feature": group_col,
                        "group_value": group_value,
                    }
                )
                group_rows.append(row)

    global_df = pd.DataFrame(global_rows)
    group_df = pd.DataFrame(group_rows)
    disparity_df = disparity_summary(group_df)

    global_df.to_csv(reports_dir / "bias_fairness_global_metrics.csv", index=False)
    group_df.to_csv(reports_dir / "bias_fairness_group_metrics.csv", index=False)
    disparity_df.to_csv(reports_dir / "bias_fairness_disparity_summary.csv", index=False)

    print(f"Auditoría de fairness operacional guardada en {reports_dir}")
    print("- bias_fairness_global_metrics.csv")
    print("- bias_fairness_group_metrics.csv")
    print("- bias_fairness_disparity_summary.csv")


if __name__ == "__main__":
    main()
