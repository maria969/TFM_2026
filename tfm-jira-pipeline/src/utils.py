from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative_path: str | Path) -> Path:
    base_dir = Path(__file__).resolve().parents[1]
    return base_dir / relative_path


def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def normalize_string(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.lower() in {"nan", "none", "null", "na", "n/a"}:
        return None
    return value

