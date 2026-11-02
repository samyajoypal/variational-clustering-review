"""Write-once run creation for new Paper 1 evidence; no Git dependency."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def start_run(directory: Path, *, stage: str, config: dict, sources: list[Path]) -> dict:
    """Refuse existing directories and snapshot every explicitly declared source."""
    if stage not in {"deterministic_simulation", "development", "validation", "final_benchmark"}:
        raise ValueError("Unknown evidence stage")
    source_paths = sorted(set([Path(__file__).resolve(), *(p.resolve() for p in sources)]))
    for path in source_paths:
        path.relative_to(ROOT)
        if not path.is_file():
            raise FileNotFoundError(path)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", config)
    source_hashes = {}
    for path in source_paths:
        relative = path.relative_to(ROOT)
        target = directory / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source_hashes[str(relative)] = sha256(target)
    versions = {}
    for name in ["numpy", "scipy", "scikit-learn", "torch", "matplotlib"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    metadata = {
        "schema_version": "paper1-review-run-v2",
        "protocol_version": config.get("protocol_version", "paper1-review-20261027-v1"),
        "run_id": directory.name,
        "fresh_run": True,
        "stage": stage,
        "started_utc": timestamp(),
        "status": "running",
        "config_sha256": sha256(directory / "config.json"),
        "source_file_sha256": source_hashes,
        "source_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
        "environment": {"python": sys.version, "platform": platform.platform(), "versions": versions},
        "legacy_outputs_used": False,
    }
    write_json(directory / "run.json", metadata)
    return metadata


def finish_run(directory: Path, metadata: dict, *, status: str = "completed", **extra: object) -> None:
    if status not in {"completed", "failed"}:
        raise ValueError("Terminal status required")
    value = dict(metadata, status=status, finished_utc=timestamp(), **extra)
    value["artifact_sha256"] = {
        str(p.relative_to(directory)): sha256(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "run.json"
    }
    write_json(directory / "run.json", value)
