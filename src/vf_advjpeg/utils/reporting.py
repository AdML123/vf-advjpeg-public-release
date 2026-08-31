from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .fs import ensure_dir, write_json
from .runtime import environment_snapshot, git_metadata


def write_run_manifest(
    output_dir: str | Path,
    config: dict[str, Any],
    device: str,
    git_root: str | Path | None = None,
) -> Path:
    target_dir = ensure_dir(output_dir)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "environment": environment_snapshot(device),
        "git": git_metadata(git_root),
    }
    manifest_path = target_dir / "run_manifest.json"
    write_json(manifest_path, payload)
    return manifest_path
