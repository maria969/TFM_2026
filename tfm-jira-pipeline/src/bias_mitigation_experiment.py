#!/usr/bin/env python3
"""
Experimento de mitigación de sesgos:
- baseline,
- class_weight,
- reponderación por grupo,
- reponderación grupo-clase,
- ajuste de umbral global,
- ajuste de umbral por grupo.
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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, brier_score_loss, roc_auc_score, average_precision_score
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


def choose_group(df):
    for col in ["project_key", "issuetype", "priority", "status"]:
        if col in df.columns and df[col].nunique(dropna=True) >= 3:
            return col
    raise ValueError("No hay columnas de grupo para mitigación.")


def ece(y_true, y_prob, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1] if i < bins - 1 else y_prob <= edges[i + 1])
        if mask.sum():
            total += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(total)


def metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "rows": len(y_true),
        "threshold": threshold,
        "positive_rate_true": np.mean(y_true),
        "positive_rate_predicted": np.mean(y_pred),
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


def build_pipeline(df, class_weight=None):
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
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight=class_weight, random_state=42, n_jobs=-1)
    return Pipeline([("preprocess", pre), ("model", model)]), feature_cols


def sample_weights(df, group_col, mode):
    if mode == "none":
        return None
    if mode == "group":
        counts = df[group_col].fillna("__NULL__").value_counts()
        return df[group_col].fillna("__NULL__").map(lambda x: 1.0 / counts[x]).to_numpy() * len(df) / len(counts)
    if mode == "group_class":
        keys = df[group_col].fillna("__NULL__").astype(str) + "__" + df[TARGET].astype(str)
        counts = keys.value_counts()
        return keys.map(lambda x: 1.0 / counts[x]).to_numpy() * len(df) / len(counts)
    raise ValueError(mode)


def best_global_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.9, 18)
    scores = [(t, f1_score(y_true, y_prob >= t, zero_division=0)) for t in thresholds]
    return max(scores, key=lambda x: x[1])[0]


def group_thresholds_for_recall(y_true, y_prob, groups, target_recall=0.65):
    out = {}
    for group in pd.Series(groups).fillna("__NULL__").astype(str).unique():
        mask = pd.Series(groups).fillna("__NULL__").astype(str).to_numpy() == group
        if mask.sum() < 30 or len(np.unique(y_true[mask])) < 2:
            out[group] = 0.25
            continue
        best = 0.25
        for t in np.linspace(0.05, 0.9, 18):
            rec = recall_score(y_true[mask], y_prob[mask] >= t, zero_division=0)
            if rec >= target_recall:
                best = t
        out[group] = best
    return out


def predictions_with_group_thresholds(y_prob, groups, thresholds):
    group_values = pd.Series(groups).fillna("__NULL__").astype(str)
    y_pred = np.zeros(len(y_prob), dtype=int)
    for i, (p, g) in enumerate(zip(y_prob, group_values)):
        y_pred[i] = int(p >= thresholds.get(g, 0.25))
    return y_pred


def group_metrics(y_true, y_prob, groups, threshold, strategy):
    rows = []
    gseries = pd.Series(groups).fillna("__NULL__").astype(str).reset_index(drop=True)
    for group, idx in gseries.groupby(gseries).groups.items():
        positions = np.asarray(list(idx), dtype=int)
        if len(positions) < 20:
            continue
        row = metrics(y_true[positions], y_prob[positions], threshold)
        row.update({"strategy": strategy, "group_value": group})
        rows.append(row)
    return rows


def disparity(group_df):
    rows = []
    for strategy, sub in group_df.groupby("strategy"):
        for metric in ["precision", "recall", "f1", "fpr", "fnr", "ece", "brier_score"]:
            vals = sub[metric].dropna()
            if vals.empty:
                continue
            rows.append({"strategy": strategy, "metric": metric, "absolute_gap": vals.max() - vals.min(), "min_value": vals.min(), "max_value": vals.max()})
    return pd.DataFrame(rows)


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
    group_col = choose_group(df)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET])
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    strategies = [
        ("baseline", None, "none", args.threshold),
        ("class_weight_balanced", "balanced", "none", args.threshold),
        ("group_reweighting", None, "group", args.threshold),
        ("group_class_reweighting", None, "group_class", args.threshold),
    ]
    global_rows = []
    subgroup_rows = []
    saved_baseline = None
    for strategy, class_weight, weight_mode, threshold in strategies:
        pipe, features = build_pipeline(train_df, class_weight=class_weight)
        weights = sample_weights(train_df, group_col, weight_mode)
        if weights is None:
            pipe.fit(train_df[features], train_df[TARGET])
        else:
            pipe.fit(train_df[features], train_df[TARGET], model__sample_weight=weights)
        y_true = test_df[TARGET].to_numpy()
        y_prob = pipe.predict_proba(test_df[features])[:, 1]
        row = metrics(y_true, y_prob, threshold)
        row.update({"strategy": strategy, "group_feature": group_col})
        global_rows.append(row)
        subgroup_rows.extend(group_metrics(y_true, y_prob, test_df[group_col], threshold, strategy))
        if strategy == "baseline":
            saved_baseline = (y_true, y_prob, test_df[group_col])

    if saved_baseline is not None:
        y_true, y_prob, groups = saved_baseline
        global_t = best_global_threshold(y_true, y_prob)
        row = metrics(y_true, y_prob, global_t)
        row.update({"strategy": "global_threshold_best_f1", "group_feature": group_col})
        global_rows.append(row)
        subgroup_rows.extend(group_metrics(y_true, y_prob, groups, global_t, "global_threshold_best_f1"))

        thresholds = group_thresholds_for_recall(y_true, y_prob, groups, target_recall=0.65)
        y_pred = predictions_with_group_thresholds(y_prob, groups, thresholds)
        # Store probabilities but evaluate global with threshold 0.5 over custom predictions through direct confusion.
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        global_rows.append({
            "strategy": "group_threshold_target_recall",
            "group_feature": group_col,
            "threshold": "per_group",
            "rows": len(y_true),
            "positive_rate_true": np.mean(y_true),
            "positive_rate_predicted": np.mean(y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
            "fnr": fn / (fn + tp) if (fn + tp) else np.nan,
            "pr_auc": average_precision_score(y_true, y_prob),
            "roc_auc": roc_auc_score(y_true, y_prob),
            "brier_score": brier_score_loss(y_true, y_prob),
            "ece": ece(y_true, y_prob),
        })

    global_df = pd.DataFrame(global_rows)
    group_df = pd.DataFrame(subgroup_rows)
    disp_df = disparity(group_df)
    baseline_disp = disp_df[disp_df["strategy"] == "baseline"][["metric", "absolute_gap"]].rename(columns={"absolute_gap": "baseline_gap"})
    tradeoffs = disp_df.merge(baseline_disp, on="metric", how="left")
    tradeoffs["delta_gap_vs_baseline"] = tradeoffs["absolute_gap"] - tradeoffs["baseline_gap"]
    global_df.to_csv(reports / "bias_mitigation_metrics.csv", index=False)
    group_df.to_csv(reports / "bias_mitigation_group_metrics.csv", index=False)
    disp_df.to_csv(reports / "bias_mitigation_disparities.csv", index=False)
    tradeoffs.to_csv(reports / "bias_mitigation_tradeoffs.csv", index=False)
    print(f"Mitigación de sesgos guardada en {reports}")


if __name__ == "__main__":
    main()
