#!/usr/bin/env python3
"""
Exporta proxies de riesgo de gestión de proyectos:
- plazo,
- coste/esfuerzo,
- alcance/complejidad.

No sustituye a métricas reales de coste y alcance. Documenta proxies disponibles
en datos Jira anonimizados y resume su distribución.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

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


def numeric_col(df, names):
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0)
    return pd.Series(np.zeros(len(df)), index=df.index)


def build_proxies(df):
    out = df.copy()
    out["risk_time_proxy"] = pd.to_numeric(out[TARGET], errors="coerce") if TARGET in out.columns else np.nan
    effort_parts = [
        numeric_col(out, ["time_to_resolution_days"]),
        numeric_col(out, ["comments_count", "comment_count", "num_comments"]),
        numeric_col(out, ["description_length"]),
        numeric_col(out, ["changelog_count", "history_count", "num_status_changes"]),
    ]
    scope_parts = [
        numeric_col(out, ["issuelinks_count", "issue_links_count", "num_links"]),
        numeric_col(out, ["subtasks_count", "subtask_count", "num_subtasks"]),
        numeric_col(out, ["components_count", "component_count", "num_components"]),
        numeric_col(out, ["versions_count", "fix_versions_count", "affected_versions_count"]),
        numeric_col(out, ["labels_count", "label_count", "num_labels"]),
    ]
    out["risk_effort_proxy_raw"] = sum(effort_parts)
    out["risk_scope_proxy_raw"] = sum(scope_parts)
    for col in ["risk_effort_proxy_raw", "risk_scope_proxy_raw"]:
        q75 = out[col].quantile(0.75)
        q90 = out[col].quantile(0.90)
        out[col.replace("_raw", "_level")] = np.select(
            [out[col] >= q90, out[col] >= q75],
            ["high", "medium"],
            default="low",
        )
    return out


def summarize(df, group_cols):
    rows = []
    for col in ["risk_time_proxy", "risk_effort_proxy_raw", "risk_scope_proxy_raw"]:
        if col not in df.columns:
            continue
        rows.append({
            "scope": "global",
            "group_feature": "__GLOBAL__",
            "group_value": "__GLOBAL__",
            "metric": col,
            "rows": len(df),
            "mean": df[col].mean(),
            "median": df[col].median(),
            "p75": df[col].quantile(0.75),
            "p90": df[col].quantile(0.90),
        })
    for group_col in group_cols:
        if group_col not in df.columns:
            continue
        for value, sub in df.groupby(df[group_col].fillna("__NULL__")):
            if len(sub) < 20:
                continue
            for col in ["risk_time_proxy", "risk_effort_proxy_raw", "risk_scope_proxy_raw"]:
                rows.append({
                    "scope": "group",
                    "group_feature": group_col,
                    "group_value": value,
                    "metric": col,
                    "rows": len(sub),
                    "mean": sub[col].mean(),
                    "median": sub[col].median(),
                    "p75": sub[col].quantile(0.75),
                    "p90": sub[col].quantile(0.90),
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    root = Path.cwd()
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    config = load_config(args.config)
    df = pd.read_csv(find_dataset(config, root))
    enriched = build_proxies(df)
    group_cols = [c for c in ["project_key", "issuetype", "priority", "status", "created_year"] if c in enriched.columns]
    summary = summarize(enriched, group_cols)
    enriched.to_csv(reports / "project_risk_proxy_dataset.csv", index=False)
    summary[summary["scope"] == "global"].to_csv(reports / "project_risk_proxy_summary.csv", index=False)
    summary[summary["scope"] == "group"].to_csv(reports / "project_risk_proxy_by_group.csv", index=False)
    print(f"Proxies de riesgo de proyecto guardados en {reports}")


if __name__ == "__main__":
    main()

