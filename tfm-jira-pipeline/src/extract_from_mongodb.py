from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pymongo import MongoClient

from utils import ensure_parent_dir, load_config, normalize_string, project_path


def get_nested(document: dict, path: str, default=None):
    current = document
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def issue_to_row(issue: dict) -> dict:
    fields = issue.get("fields", {}) if isinstance(issue.get("fields", {}), dict) else {}

    project = fields.get("project") or {}
    issue_type = fields.get("issuetype") or {}
    priority = fields.get("priority") or {}
    status = fields.get("status") or {}

    return {
        "_id": str(issue.get("_id")),
        "key": normalize_string(issue.get("key")),
        "created": fields.get("created"),
        "resolutiondate": fields.get("resolutiondate"),
        "project_key": normalize_string(project.get("key")),
        "project_name": normalize_string(project.get("name")),
        "issuetype": normalize_string(issue_type.get("name")),
        "priority": normalize_string(priority.get("name")),
        "status": normalize_string(status.get("name")),
        "summary": normalize_string(fields.get("summary")),
        "description": normalize_string(fields.get("description")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(project_path(args.config))
    mongo_cfg = config["mongodb"]
    output_path = project_path(config["paths"]["input_csv"])

    client = MongoClient(mongo_cfg["uri"])
    collection = client[mongo_cfg["database"]][mongo_cfg["collection"]]

    limit = int(mongo_cfg.get("limit", 0) or 0)
    cursor = collection.find({})
    if limit > 0:
        cursor = cursor.limit(limit)

    rows = [issue_to_row(issue) for issue in cursor]
    df = pd.DataFrame(rows)

    ensure_parent_dir(output_path)
    df.to_csv(output_path, index=False)
    print(f"Exportadas {len(df):,} issues a {output_path}")


if __name__ == "__main__":
    main()

