from __future__ import annotations

import importlib.metadata
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PACKAGE_VERSION_NAMES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pillow": "Pillow",
    "PyYAML": "PyYAML",
    "matplotlib": "matplotlib",
    "scikit-image": "scikit-image",
    "scikit-rf": "scikit-rf",
    "scipy": "scipy",
    "torch": "torch",
    "torchvision": "torchvision",
    "lpips": "lpips",
    "tqdm": "tqdm",
}


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def git_metadata(cwd: str | Path | None = None) -> dict[str, Any]:
    root = Path(cwd or Path.cwd())
    commands = {
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        "commit": ["git", "rev-parse", "HEAD"],
        "status": ["git", "status", "--short"],
    }
    payload: dict[str, Any] = {}
    for key, command in commands.items():
        try:
            payload[key] = (
                subprocess.check_output(command, cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
            )
        except Exception:
            payload[key] = None
    return payload


def installed_package_versions() -> dict[str, str | None]:
    payload: dict[str, str | None] = {}
    for display_name, package_name in PACKAGE_VERSION_NAMES.items():
        try:
            payload[display_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            payload[display_name] = None
    return payload


def environment_snapshot(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "packages": installed_package_versions(),
    }
