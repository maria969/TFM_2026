#!/usr/bin/env python3
"""
Escenarios controlados de sesgo:
- sesgo muestral por subrepresentación,
- sesgo de medición por pérdida selectiva de información,
- sesgo histórico/de etiqueta por ruido sistemático.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

TARGET = "late_resolution"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_dataset(config, root):
    configured = config.get("data", {}).get("modeling_dataset") or config.get("paths", {}).get("modeling_dataset")
    candidates = []
    if configured:
        candidates += [root / configured, Path(configured)]
    candidates += [
        root / "data/processed/modeling_dataset.csv",
        root / "data/modeling_dataset.csv",
        root / "data/processed/jira_modeling_dataset.csv",
        root / "outputs/modeling_dataset.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("No se encontró el dataset tabular.")


def ece(y_true, y_prob, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for i in range(bins):
        mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1] if i < bins - 1 else y_prob <= edges[i + 1])
        if mask.sum():
            out += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(out)


def metrics(y_true, y_prob, threshold=0.25):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "rows": len(y_true),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_predicted": float(np.mean(y_pred)),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
        "fnr": fn / (fn + tp) if (fn + tp) else np.nan,
        "pr_auc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "brier_score": brier_score_loss(y_true, y_prob),
        "ece": ece(y_true, y_prob),
    }


def build_pipeline(df):
    excluded = {
        TARGET,
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
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ]
    )
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)
    return Pipeline([("preprocess", pre), ("model", model)]), feature_cols


def choose_group(df):
    for col in ["project_key", "issuetype", "priority", "status"]:
        if col in df.columns and df[col].nunique(dropna=True) >= 3:
            counts = df[col].value_counts()
            return col, counts.index[-1], counts.index[0]
    raise ValueError("No hay columnas de grupo suficientes para simular sesgos.")


def apply_scenario(train_df, scenario, group_col, minority_group, majority_group):
    out = train_df.copy()
    rng = np.random.default_rng(42)
    if scenario == "baseline":
        return out
    if scenario == "sample_bias_underrepresent_minority":
        minority_idx = out[out[group_col] == minority_group].index
        keep_minority = rng.choice(minority_idx, size=max(1, int(len(minority_idx) * 0.35)), replace=False)
        keep_idx = out.index.difference(minority_idx).union(keep_minority)
        return out.loc[keep_idx].copy()
    if scenario == "measurement_bias_description_loss":
        if "description_length" in out.columns:
            mask = out[group_col] == minority_group
            out.loc[mask, "description_length"] = 0
        if "has_description" in out.columns:
            mask = out[group_col] == minority_group
            out.loc[mask, "has_description"] = 0
        return out
    if scenario == "measurement_bias_priority_missing":
        if "priority" in out.columns:
            mask = out[group_col] == minority_group
            out.loc[mask, "priority"] = np.nan
        return out
    if scenario == "historical_label_bias_minority":
        mask = (out[group_col] == minority_group) & (out[TARGET] == 1)
        idx = out[mask].index
        flip = rng.choice(idx, size=int(len(idx) * 0.30), replace=False) if len(idx) else []
        out.loc[flip, TARGET] = 0
        return out
    if scenario == "historical_label_bias_majority":
        mask = (out[group_col] == majority_group) & (out[TARGET] == 0)
        idx = out[mask].index
        flip = rng.choice(idx, size=int(len(idx) * 0.15), replace=False) if len(idx) else []
        out.loc[flip, TARGET] = 1
        return out
    raise ValueError(f"Escenario no reconocido: {scenario}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()
    root = Path.cwd()
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    config = load_config(args.config)
    df = pd.read_csv(find_dataset(config, root))
    df = df[df[TARGET].notna()].copy()
    df[TARGET] = df[TARGET].astype(int)
    group_col, minority_group, majority_group = choose_group(df)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET])
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    scenarios = [
        "baseline",
        "sample_bias_underrepresent_minority",
        "measurement_bias_description_loss",
        "measurement_bias_priority_missing",
        "historical_label_bias_minority",
        "historical_label_bias_majority",
    ]
    metric_rows = []
    group_rows = []
    for scenario in scenarios:
        scenario_train = apply_scenario(train_df, scenario, group_col, minority_group, majority_group)
        pipe, feature_cols = build_pipeline(scenario_train)
        pipe.fit(scenario_train[feature_cols], scenario_train[TARGET])
        y_true = test_df[TARGET].to_numpy()
        y_prob = pipe.predict_proba(test_df[feature_cols])[:, 1]
        row = metrics(y_true, y_prob, threshold=args.threshold)
        row.update({"scenario": scenario, "group_feature": group_col, "minority_group": minority_group, "majority_group": majority_group})
        metric_rows.append(row)
        eval_groups = test_df[group_col].fillna("__NULL__").astype(str).reset_index(drop=True)
        for gv, idx in eval_groups.groupby(eval_groups).groups.items():
            positions = np.asarray(list(idx), dtype=int)
            if len(positions) < 20:
                continue
            g = metrics(y_true[positions], y_prob[positions], threshold=args.threshold)
            g.update({"scenario": scenario, "group_feature": group_col, "group_value": gv})
            group_rows.append(g)

    metrics_df = pd.DataFrame(metric_rows)
    groups_df = pd.DataFrame(group_rows)
    disparity_rows = []
    for scenario, sub in groups_df.groupby("scenario"):
        for m in ["precision", "recall", "f1", "fpr", "fnr", "ece", "brier_score"]:
            vals = sub[m].dropna()
            if vals.empty:
                continue
            disparity_rows.append({"scenario": scenario, "metric": m, "absolute_gap": vals.max() - vals.min(), "min_value": vals.min(), "max_value": vals.max()})
    pd.DataFrame(disparity_rows).to_csv(reports / "bias_scenario_disparities.csv", index=False)
    metrics_df.to_csv(reports / "bias_scenario_metrics.csv", index=False)
    groups_df.to_csv(reports / "bias_scenario_group_metrics.csv", index=False)
    print(f"Experimentos de sesgo guardados en {reports}")


if __name__ == "__main__":
    main()
