from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from utils import ensure_parent_dir, load_config, project_path


def add_date_features(df: pd.DataFrame, created_col: str, resolution_col: str) -> pd.DataFrame:
    df[created_col] = pd.to_datetime(df[created_col], errors="coerce", utc=True)
    df[resolution_col] = pd.to_datetime(df[resolution_col], errors="coerce", utc=True)

    df["created_year"] = df[created_col].dt.year
    df["created_month"] = df[created_col].dt.month
    df["is_resolved"] = df[resolution_col].notna().astype(int)
    df["resolution_valid"] = (
        df[created_col].notna()
        & df[resolution_col].notna()
        & (df[resolution_col] >= df[created_col])
    ).astype(int)

    df["time_to_resolution_days"] = (
        df[resolution_col] - df[created_col]
    ).dt.total_seconds() / 86400

    df.loc[df["resolution_valid"] == 0, "time_to_resolution_days"] = np.nan
    return df


def add_text_quality_features(df: pd.DataFrame, summary_col: str, description_col: str) -> pd.DataFrame:
    for col in [summary_col, description_col]:
        if col not in df.columns:
            df[col] = np.nan

    df["summary_length"] = df[summary_col].fillna("").astype(str).str.len()
    df["description_length"] = df[description_col].fillna("").astype(str).str.len()
    df["has_description"] = (df["description_length"] > 0).astype(int)
    return df


def add_target(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    target_cfg = config["target"]
    resolution_col = target_cfg["resolution_time_column"]
    target_col = target_cfg["target_column"]

    valid = df[resolution_col].dropna()
    if valid.empty:
        raise ValueError("No hay valores válidos para calcular la variable objetivo.")

    if target_cfg["threshold_method"] == "percentile":
        threshold = np.percentile(valid, float(target_cfg["threshold_percentile"]))
    elif target_cfg["threshold_method"] == "fixed_days":
        threshold = float(target_cfg["threshold_days"])
    else:
        raise ValueError("threshold_method debe ser 'percentile' o 'fixed_days'.")

    df["late_threshold_days"] = threshold
    df[target_col] = np.where(df[resolution_col] > threshold, 1, 0)
    df.loc[df[resolution_col].isna(), target_col] = np.nan
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(project_path(args.config))
    paths = config["paths"]
    cols = config["columns"]

    input_path = project_path(paths["input_csv"])
    output_path = project_path(paths["prepared_csv"])

    df = pd.read_csv(input_path)
    df = add_date_features(df, cols["created"], cols["resolutiondate"])
    df = add_text_quality_features(df, cols["summary"], cols["description"])
    df = add_target(df, config)

    ensure_parent_dir(output_path)
    df.to_csv(output_path, index=False)
    print(f"Dataset preparado con {len(df):,} filas y {len(df.columns):,} columnas: {output_path}")


if __name__ == "__main__":
    main()

