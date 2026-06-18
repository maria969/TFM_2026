from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import load_config, project_path


def build_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "rows": len(df),
                "null_count": int(df[col].isna().sum()),
                "null_rate": float(df[col].isna().mean()),
                "distinct_count": int(df[col].nunique(dropna=True)),
                "distinct_rate": float(df[col].nunique(dropna=True) / max(len(df), 1)),
            }
        )
    return pd.DataFrame(rows)


def build_category_distributions(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frames = []
    for col in columns:
        if col not in df.columns:
            continue
        counts = df[col].fillna("__NULL__").value_counts(dropna=False).head(50)
        part = counts.rename_axis("value").reset_index(name="count")
        part["column"] = col
        part["rate"] = part["count"] / len(df)
        frames.append(part[["column", "value", "count", "rate"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_time_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if "created_year" not in df.columns or "created_month" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby(["created_year", "created_month"], dropna=False)
        .size()
        .reset_index(name="issues_count")
        .sort_values(["created_year", "created_month"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(project_path(args.config))
    prepared_path = project_path(config["paths"]["prepared_csv"])
    reports_dir = project_path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(prepared_path)

    quality = build_quality_summary(df)
    quality.to_csv(reports_dir / "data_quality_summary.csv", index=False)

    category_cols = [
        config["columns"]["project_key"],
        config["columns"]["issuetype"],
        config["columns"]["priority"],
        config["columns"]["status"],
        config["target"]["target_column"],
    ]
    categories = build_category_distributions(df, category_cols)
    categories.to_csv(reports_dir / "category_distributions.csv", index=False)

    time_distribution = build_time_distribution(df)
    time_distribution.to_csv(reports_dir / "time_distribution.csv", index=False)

    print(f"Informes EDA generados en {reports_dir}")


if __name__ == "__main__":
    main()

