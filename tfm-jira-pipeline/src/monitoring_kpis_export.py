from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils import load_config, project_path


LOWER_IS_BETTER = {"brier_score", "ece", "error_rate", "false_positive_rate", "false_negative_rate"}


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def status_for_metric(metric: str, value: float) -> str:
    if pd.isna(value):
        return "missing"
    if metric in LOWER_IS_BETTER:
        if value <= 0.05:
            return "green"
        if value <= 0.15:
            return "amber"
        return "red"
    if value >= 0.75:
        return "green"
    if value >= 0.50:
        return "amber"
    return "red"


def melt_metrics(df: pd.DataFrame, id_vars: list[str], source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metric_cols = [
        c
        for c in df.columns
        if c not in id_vars and pd.api.types.is_numeric_dtype(df[c])
    ]
    # Use "kpi" instead of "metric" as the melted column name to avoid collisions
    # with source tables that already contain a column called "metric"
    # (for example, stability_seed_summary.csv).
    long_df = df.melt(
        id_vars=[c for c in id_vars if c in df.columns],
        value_vars=metric_cols,
        var_name="kpi",
        value_name="value",
    )
    long_df["source_table"] = source
    long_df["status"] = long_df.apply(lambda r: status_for_metric(str(r["kpi"]), float(r["value"])), axis=1)
    return long_df


def build_kpi_catalog() -> pd.DataFrame:
    rows = [
        ("precision", "Rendimiento", "Proporción de predicciones positivas que eran realmente positivas.", "Mayor es mejor"),
        ("recall", "Rendimiento", "Proporción de positivos reales detectados por el modelo.", "Mayor es mejor"),
        ("f1", "Rendimiento", "Media armónica entre precision y recall.", "Mayor es mejor"),
        ("pr_auc", "Rendimiento", "Área bajo la curva precision-recall; útil con clases desbalanceadas.", "Mayor es mejor"),
        ("roc_auc", "Rendimiento", "Capacidad de ranking entre positivos y negativos.", "Mayor es mejor"),
        ("brier_score", "Calibración", "Error cuadrático medio de las probabilidades predichas.", "Menor es mejor"),
        ("ece", "Calibración", "Expected Calibration Error; distancia entre probabilidad predicha y frecuencia observada.", "Menor es mejor"),
        ("positive_rate_true", "Datos", "Proporción real de clase positiva.", "Depende del contexto"),
        ("positive_rate_predicted", "Modelo", "Proporción de casos marcados como positivos por el modelo.", "Depende del contexto"),
        ("delta_f1", "Robustez", "Cambio de F1 frente al escenario de referencia.", "Cerca de cero o positivo"),
        ("delta_ece", "Robustez", "Cambio de ECE frente al escenario de referencia.", "Cerca de cero o negativo"),
    ]
    return pd.DataFrame(rows, columns=["kpi", "family", "definition", "interpretation"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(project_path(args.config))
    reports_dir = project_path(config["paths"]["reports_dir"])
    powerbi_dir = project_path("data/powerbi")
    powerbi_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, tuple[pd.DataFrame, list[str]]] = {
        "model_metrics": (read_csv_if_exists(reports_dir / "model_metrics.csv"), ["model"]),
        "advanced_model_metrics": (read_csv_if_exists(reports_dir / "advanced_model_metrics.csv"), ["model", "threshold"]),
        "threshold_metrics": (read_csv_if_exists(reports_dir / "threshold_metrics.csv"), ["model", "threshold"]),
        "degradation_summary": (read_csv_if_exists(reports_dir / "degradation_summary.csv"), ["scenario", "degradation_type", "degradation_rate"]),
        "temporal_drift_metrics": (read_csv_if_exists(reports_dir / "temporal_drift_metrics.csv"), ["split_type"]),
        "retraining_summary": (read_csv_if_exists(reports_dir / "retraining_summary.csv"), ["strategy"]),
        "stability_seed_summary": (read_csv_if_exists(reports_dir / "stability_seed_summary.csv"), ["metric"]),
        "subgroup_gap_summary": (read_csv_if_exists(reports_dir / "subgroup_gap_summary.csv"), ["model", "subgroup_feature"]),
        "data_cleaning_metrics": (read_csv_if_exists(reports_dir / "data_cleaning_metrics.csv"), ["scenario"]),
    }

    long_frames = [
        melt_metrics(df, id_vars, source)
        for source, (df, id_vars) in tables.items()
        if not df.empty
    ]
    performance_long = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()
    if not performance_long.empty:
        performance_long["value"] = pd.to_numeric(performance_long["value"], errors="coerce")
    performance_long.to_csv(powerbi_dir / "performance_kpis_long.csv", index=False)

    quality = read_csv_if_exists(reports_dir / "data_quality_summary.csv")
    if not quality.empty:
        quality.to_csv(powerbi_dir / "data_quality_summary.csv", index=False)

    category = read_csv_if_exists(reports_dir / "category_distributions.csv")
    if not category.empty:
        category.to_csv(powerbi_dir / "category_distributions.csv", index=False)

    calibration = read_csv_if_exists(reports_dir / "calibration_bins.csv")
    if not calibration.empty:
        calibration.to_csv(powerbi_dir / "calibration_bins.csv", index=False)

    kpi_catalog = build_kpi_catalog()
    kpi_catalog.to_csv(powerbi_dir / "kpi_catalog.csv", index=False)

    snapshot_rows: list[dict[str, Any]] = []
    if not performance_long.empty:
        for metric, metric_df in performance_long.groupby("kpi"):
            numeric_values = pd.to_numeric(metric_df["value"], errors="coerce").dropna()
            if numeric_values.empty:
                continue
            latest_value = float(numeric_values.iloc[-1])
            snapshot_rows.append(
                {
                    "metric": metric,
                    "latest_value": latest_value,
                    "min_value": float(numeric_values.min()),
                    "max_value": float(numeric_values.max()),
                    "mean_value": float(numeric_values.mean()),
                    "status": status_for_metric(metric, latest_value),
                }
            )
    pd.DataFrame(snapshot_rows).replace([np.inf, -np.inf], np.nan).to_csv(powerbi_dir / "kpi_snapshot.csv", index=False)

    print(f"Exportación para Power BI guardada en {powerbi_dir}")
    print("Archivos generados:")
    print("- performance_kpis_long.csv")
    print("- kpi_catalog.csv")
    print("- kpi_snapshot.csv")
    print("- data_quality_summary.csv, category_distributions.csv y calibration_bins.csv si existían")


if __name__ == "__main__":
    main()
