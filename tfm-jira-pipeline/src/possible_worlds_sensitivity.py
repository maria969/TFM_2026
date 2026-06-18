from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import load_config, project_path


RANDOM_STATE = 42


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


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: list[str],
    numeric_features: list[str],
) -> Pipeline:
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
    return pipeline


def build_worlds(
    X_sample: pd.DataFrame,
    train_df: pd.DataFrame,
    categorical_features: list[str],
    numeric_features: list[str],
    n_worlds: int,
    rng: np.random.Generator,
) -> list[pd.DataFrame]:
    worlds = []
    category_values = {
        col: train_df[col].dropna().value_counts(normalize=True)
        for col in categorical_features
        if col in train_df.columns and train_df[col].dropna().nunique() > 0
    }
    numeric_values = {
        col: pd.to_numeric(train_df[col], errors="coerce").dropna()
        for col in numeric_features
        if col in train_df.columns and pd.to_numeric(train_df[col], errors="coerce").dropna().size > 0
    }

    for world_id in range(n_worlds):
        world = X_sample.copy()
        for col, distribution in category_values.items():
            mask = world[col].isna()
            if mask.any():
                world.loc[mask, col] = rng.choice(distribution.index.to_numpy(), size=int(mask.sum()), p=distribution.to_numpy())
        for col, values in numeric_values.items():
            mask = world[col].isna()
            if mask.any():
                world.loc[mask, col] = rng.choice(values.to_numpy(), size=int(mask.sum()), replace=True)
        world["_world_id"] = world_id + 1
        worlds.append(world)
    return worlds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--worlds", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=1000)
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
    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=float(config["model"]["test_size"]),
        random_state=int(config["model"]["random_state"]),
        stratify=y,
    )

    rng = np.random.default_rng(RANDOM_STATE)
    sample_n = min(args.sample_size, len(X_test))
    sample_index = rng.choice(X_test.index.to_numpy(), size=sample_n, replace=False)
    X_sample = X_test.loc[sample_index].copy()

    model = train_model(X_train, y_train, categorical_features, numeric_features)
    baseline_prob = model.predict_proba(X_sample)[:, 1]

    worlds = build_worlds(X_sample, X_train, categorical_features, numeric_features, args.worlds, rng)
    world_predictions = []
    for world in worlds:
        world_id = int(world["_world_id"].iloc[0])
        world_features = world.drop(columns=["_world_id"])
        probs = model.predict_proba(world_features)[:, 1]
        world_predictions.append(
            pd.DataFrame(
                {
                    "row_index": world_features.index,
                    "world_id": world_id,
                    "predicted_probability": probs,
                }
            )
        )

    long_df = pd.concat(world_predictions, ignore_index=True)
    long_df.to_csv(reports_dir / "possible_worlds_predictions.csv", index=False)

    summary = (
        long_df.groupby("row_index")
        .agg(
            worlds=("world_id", "nunique"),
            mean_probability=("predicted_probability", "mean"),
            std_probability=("predicted_probability", "std"),
            min_probability=("predicted_probability", "min"),
            max_probability=("predicted_probability", "max"),
        )
        .reset_index()
    )
    summary["probability_range"] = summary["max_probability"] - summary["min_probability"]
    baseline = pd.DataFrame({"row_index": X_sample.index, "baseline_probability": baseline_prob})
    summary = summary.merge(baseline, on="row_index", how="left")
    summary.to_csv(reports_dir / "possible_worlds_sensitivity_summary.csv", index=False)

    aggregate = pd.DataFrame(
        [
            {
                "sample_rows": sample_n,
                "worlds": args.worlds,
                "mean_probability_std": summary["std_probability"].mean(),
                "p95_probability_range": summary["probability_range"].quantile(0.95),
                "max_probability_range": summary["probability_range"].max(),
            }
        ]
    )
    aggregate.to_csv(reports_dir / "possible_worlds_sensitivity_aggregate.csv", index=False)

    print(f"Análisis de sensibilidad por posibles mundos guardado en {reports_dir}")
    print("Archivos generados:")
    print("- possible_worlds_predictions.csv")
    print("- possible_worlds_sensitivity_summary.csv")
    print("- possible_worlds_sensitivity_aggregate.csv")


if __name__ == "__main__":
    main()
