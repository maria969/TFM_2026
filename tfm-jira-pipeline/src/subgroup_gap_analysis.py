from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from utils import load_config, project_path


METRICS = ["positive_rate", "precision", "recall", "f1", "pr_auc", "brier_score", "ece"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--min-rows", type=int, default=30)
    args = parser.parse_args()

    config = load_config(project_path(args.config))
    reports_dir = project_path(config["paths"]["reports_dir"])
    subgroup_path = reports_dir / "subgroup_metrics.csv"
    model_path = reports_dir / "model_metrics.csv"

    if not subgroup_path.exists():
        raise FileNotFoundError("No existe reports/subgroup_metrics.csv. Ejecuta primero src/train_baseline.py.")
    if not model_path.exists():
        raise FileNotFoundError("No existe reports/model_metrics.csv. Ejecuta primero src/train_baseline.py.")

    subgroups = pd.read_csv(subgroup_path)
    overall = pd.read_csv(model_path)
    subgroups = subgroups[subgroups["rows"] >= args.min_rows].copy()

    rows = []
    for _, group_row in subgroups.iterrows():
        model = group_row["model"]
        model_overall = overall[overall["model"] == model]
        if model_overall.empty:
            continue
        model_overall = model_overall.iloc[0]
        row = {
            "model": model,
            "subgroup_feature": group_row["subgroup_feature"],
            "subgroup_value": group_row["subgroup_value"],
            "rows": group_row["rows"],
        }
        for metric in METRICS:
            if metric not in group_row or metric not in model_overall:
                continue
            row[f"{metric}_subgroup"] = group_row[metric]
            row[f"{metric}_overall"] = model_overall[metric]
            row[f"{metric}_gap"] = group_row[metric] - model_overall[metric]
            row[f"{metric}_abs_gap"] = abs(group_row[metric] - model_overall[metric])
        rows.append(row)

    gaps = pd.DataFrame(rows)
    gaps.to_csv(reports_dir / "subgroup_gap_metrics.csv", index=False)

    summary_rows = []
    for model, model_df in gaps.groupby("model"):
        for feature, feature_df in model_df.groupby("subgroup_feature"):
            summary = {"model": model, "subgroup_feature": feature, "subgroups": len(feature_df)}
            for metric in METRICS:
                gap_col = f"{metric}_abs_gap"
                if gap_col in feature_df.columns:
                    summary[f"max_abs_{metric}_gap"] = feature_df[gap_col].max()
                    summary[f"mean_abs_{metric}_gap"] = feature_df[gap_col].mean()
            summary_rows.append(summary)

    pd.DataFrame(summary_rows).replace([np.inf, -np.inf], np.nan).to_csv(
        reports_dir / "subgroup_gap_summary.csv",
        index=False,
    )

    print(f"Análisis de gaps por subgrupo guardado en {reports_dir}")
    print("Archivos generados:")
    print("- subgroup_gap_metrics.csv")
    print("- subgroup_gap_summary.csv")


if __name__ == "__main__":
    main()
