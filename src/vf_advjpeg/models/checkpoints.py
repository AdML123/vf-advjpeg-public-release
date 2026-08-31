from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from vf_advjpeg.utils.fs import ensure_dir, resolve_project_path


def checkpoint_path(config: dict[str, Any], model_name: str) -> Path:
    root = resolve_project_path(config["paths"]["checkpoint_root"])
    ensure_dir(root)
    return root / f"{model_name}_pet37.pt"


def save_checkpoint(
    config: dict[str, Any],
    model_name: str,
    model_state: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    path = checkpoint_path(config, model_name)
    torch.save({"model_state": model_state, "metadata": metadata}, path)
    return path


def load_checkpoint(config: dict[str, Any], model_name: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = checkpoint_path(config, model_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {model_name}: {path}")
    return torch.load(path, map_location=map_location)

