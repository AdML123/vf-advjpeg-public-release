from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import extract_archive


class DictDataset(Dataset):
    def __init__(self, dataset: Dataset, sample_prefix: str) -> None:
        self.dataset = dataset
        self.sample_prefix = sample_prefix

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, label = self.dataset[index]
        return {
            "image": image,
            "label": int(label),
            "sample_id": f"{self.sample_prefix}_{index:05d}",
            "dataset_index": index,
        }


def _local_cifar10_archives(root_path: Path) -> list[Path]:
    candidates = {
        root_path / "cifar-10-python.tar.gz",
        root_path.parent.parent / "cifar-10-python.tar.gz",
    }
    return sorted(
        (path for path in candidates if path.exists()),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )


def _has_valid_cifar10(root_path: Path) -> bool:
    try:
        CIFAR10(root=str(root_path), train=False, transform=None, download=False)
    except RuntimeError:
        return False
    return True


def _ensure_cifar10_available(root_path: Path, download: bool) -> None:
    if _has_valid_cifar10(root_path):
        return
    if download:
        return

    extracted = root_path / "cifar-10-batches-py"
    last_error: Exception | None = None
    for archive in _local_cifar10_archives(root_path):
        if extracted.exists():
            shutil.rmtree(extracted)
        try:
            extract_archive(str(archive), str(root_path))
        except (EOFError, RuntimeError, OSError) as exc:
            last_error = exc
            continue
        if _has_valid_cifar10(root_path):
            return

    message = f"Could not prepare a valid CIFAR-10 test set under {root_path}"
    if last_error is None:
        raise RuntimeError(f"{message}; no local cifar-10-python.tar.gz archive was found")
    raise RuntimeError(message) from last_error


def build_cifar10_eval_dataset(root: str | Path, image_size: int = 32, download: bool = False) -> Dataset:
    root_path = Path(root)
    _ensure_cifar10_available(root_path, download=download)
    transform_ops: list[Any] = []
    if image_size != 32:
        transform_ops.append(transforms.Resize((image_size, image_size)))
    transform_ops.append(transforms.ToTensor())
    dataset = CIFAR10(
        root=str(root_path),
        train=False,
        transform=transforms.Compose(transform_ops),
        download=download,
    )
    return DictDataset(dataset, "cifar10_test")
