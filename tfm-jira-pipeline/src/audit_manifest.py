from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from utils import load_config, project_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, project_root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "relative_path": str(path.relative_to(project_root)),
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": sha256_file(path),
    }


def pip_freeze() -> list[str]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return sorted(result.stdout.splitlines())
    except Exception:
        return []
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    project_root = project_path(".").resolve()
    config_path = project_path(args.config)
    config = load_config(config_path)
    reports_dir = project_path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    script_files = sorted((project_root / "src").glob("*.py"))
    report_files = sorted(reports_dir.glob("*.csv"))
    config_files = sorted((project_root / "config").glob("*.yaml"))

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "python_version": sys.version,
        "platform": platform.platform(),
        "config": config,
        "scripts": [file_record(path, project_root) for path in script_files],
        "configs": [file_record(path, project_root) for path in config_files],
        "reports": [file_record(path, project_root) for path in report_files],
        "dependencies": pip_freeze(),
        "recommended_execution_order": [
            "python3 src/extract_from_mongodb.py --config config/config.yaml",
            "python3 src/prepare_dataset.py --config config/config.yaml",
            "python3 src/eda_quality.py --config config/config.yaml",
            "python3 src/train_baseline.py --config config/config.yaml",
            "python3 src/threshold_tuning.py --config config/config.yaml",
            "python3 src/advanced_metrics.py --config config/config.yaml",
            "python3 src/degradation_experiments.py --config config/config.yaml",
            "python3 src/data_cleaning_experiment.py --config config/config.yaml",
            "python3 src/temporal_drift_experiment.py --config config/config.yaml",
            "python3 src/retraining_experiment.py --config config/config.yaml",
            "python3 src/stability_experiment.py --config config/config.yaml",
            "python3 src/subgroup_gap_analysis.py --config config/config.yaml",
            "python3 src/possible_worlds_sensitivity.py --config config/config.yaml",
            "python3 src/monitoring_kpis_export.py --config config/config.yaml",
            "python3 src/audit_manifest.py --config config/config.yaml",
        ],
    }

    with open(reports_dir / "audit_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    inventory = pd.DataFrame(manifest["scripts"] + manifest["configs"] + manifest["reports"])
    inventory.to_csv(reports_dir / "audit_file_inventory.csv", index=False)

    runbook = [
        "# Runbook de auditoría y trazabilidad del pipeline",
        "",
        "## Objetivo",
        "",
        "Este documento registra el orden recomendado de ejecución, los artefactos generados y las evidencias necesarias para reproducir el pipeline experimental del TFM.",
        "",
        "## Orden recomendado de ejecución",
        "",
    ]
    for command in manifest["recommended_execution_order"]:
        runbook.append(f"- `{command}`")
    runbook.extend(
        [
            "",
            "## Evidencias de trazabilidad",
            "",
            "- `audit_manifest.json`: manifiesto completo con configuración, entorno, scripts, informes y dependencias.",
            "- `audit_file_inventory.csv`: inventario tabular con tamaño, fecha de modificación y hash SHA-256 de cada archivo.",
            "- Los hashes SHA-256 permiten verificar si un script o informe ha cambiado entre ejecuciones.",
        ]
    )
    with open(reports_dir / "audit_runbook.md", "w", encoding="utf-8") as f:
        f.write("\n".join(runbook))

    print(f"Manifiesto de auditoría guardado en {reports_dir}")
    print("Archivos generados:")
    print("- audit_manifest.json")
    print("- audit_file_inventory.csv")
    print("- audit_runbook.md")


if __name__ == "__main__":
    main()
