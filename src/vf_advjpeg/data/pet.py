from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from vf_advjpeg.utils.fs import ensure_dir, read_json, resolve_project_path, write_json


class IndexedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int], sample_ids: list[str] | None = None) -> None:
        self.dataset = dataset
        self.indices = indices
        self.sample_ids = sample_ids or [str(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset_index = self.indices[index]
        image, label = self.dataset[dataset_index]
        return {
            "image": image,
            "label": int(label),
            "sample_id": self.sample_ids[index],
            "dataset_index": dataset_index,
        }


@dataclass(slots=True)
class PetDatasetBundle:
    train: Dataset
    val: Dataset
    calibration: Dataset
    evaluation: Dataset


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    ops: list[Any] = [
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    if train:
        ops.insert(0, transforms.RandomHorizontalFlip())
    return transforms.Compose(ops)


def _extract_labels(dataset: OxfordIIITPet) -> np.ndarray:
    if hasattr(dataset, "_labels"):
        labels = np.asarray(dataset._labels, dtype=np.int64)
        if labels.min() == 1:
            labels = labels - 1
        return labels
    labels = []
    for _, label in dataset:
        labels.append(int(label))
    return np.asarray(labels, dtype=np.int64)


def _extract_sample_ids(dataset: OxfordIIITPet) -> list[str]:
    images = getattr(dataset, "_images", None)
    if images is None:
        return [str(index) for index in range(len(dataset))]
    return [Path(path).stem for path in images]


def stratified_partition(
    labels: np.ndarray,
    sizes: dict[str, int],
    seed: int,
) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    total_requested = sum(sizes.values())
    if total_requested > len(labels):
        raise ValueError("Requested split sizes exceed available examples.")

    all_indices = np.arange(len(labels))
    class_indices = {label: all_indices[labels == label].tolist() for label in sorted(set(labels.tolist()))}
    for indices in class_indices.values():
        rng.shuffle(indices)

    counts_per_split = {name: {label: 0 for label in class_indices} for name in sizes}
    class_counts = {label: len(indices) for label, indices in class_indices.items()}

    for name, target_size in sizes.items():
        quotas = {}
        fractions = []
        assigned = 0
        for label, count in class_counts.items():
            raw = count * target_size / len(labels)
            quota = int(np.floor(raw))
            quotas[label] = quota
            assigned += quota
            fractions.append((raw - quota, label))
        for _, label in sorted(fractions, reverse=True)[: target_size - assigned]:
            quotas[label] += 1
        counts_per_split[name] = quotas

    assignments = {name: [] for name in sizes}
    remaining = {label: list(indices) for label, indices in class_indices.items()}

    for name, quotas in counts_per_split.items():
        for label, quota in quotas.items():
            take = remaining[label][:quota]
            assignments[name].extend(take)
            remaining[label] = remaining[label][quota:]

    for name in assignments:
        rng.shuffle(assignments[name])
        assignments[name] = assignments[name][: sizes[name]]

    return assignments


def maybe_subsample_indices(indices: list[int], size: int | None, seed: int) -> list[int]:
    if size is None or size <= 0 or size >= len(indices):
        return list(indices)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(indices), size=size, replace=False)
    return chosen.tolist()


def repair_incomplete_downloads(data_root: Path) -> None:
    dataset_root = data_root / "oxford-iiit-pet"
    checks = [
        (dataset_root / "images.tar.gz", dataset_root / "images"),
        (dataset_root / "annotations.tar.gz", dataset_root / "annotations"),
    ]
    for archive_path, extracted_dir in checks:
        if archive_path.exists() and archive_path.stat().st_size == 0:
            archive_path.unlink()
            continue
        if archive_path.exists() and not extracted_dir.exists():
            archive_path.unlink()


def create_split_manifest(config: dict[str, Any]) -> dict[str, Any]:
    data_root = resolve_project_path(config["paths"]["data_root"])
    repair_incomplete_downloads(data_root)
    image_size = config["data"]["image_size"]
    download = bool(config["data"]["download"])
    seed = int(config["runtime"]["seed"])
    validation_fraction = float(config["data"]["validation_fraction"])
    calibration_size = int(config["data"]["calibration_size"])
    evaluation_size = int(config["data"]["evaluation_size"])

    trainval = OxfordIIITPet(
        root=data_root,
        split="trainval",
        target_types="category",
        transform=build_transforms(image_size, train=True),
        download=download,
    )
    test = OxfordIIITPet(
        root=data_root,
        split="test",
        target_types="category",
        transform=build_transforms(image_size, train=False),
        download=download,
    )

    generator = torch.Generator().manual_seed(seed)
    val_size = max(1, int(round(len(trainval) * validation_fraction)))
    train_size = len(trainval) - val_size
    train_subset, val_subset = random_split(trainval, [train_size, val_size], generator=generator)

    test_labels = _extract_labels(test)
    test_ids = _extract_sample_ids(test)
    assignments = stratified_partition(
        test_labels,
        sizes={"calibration": calibration_size, "evaluation": evaluation_size},
        seed=seed,
    )

    calibration_ids = [test_ids[index] for index in assignments["calibration"]]
    evaluation_ids = [test_ids[index] for index in assignments["evaluation"]]

    manifest = {
        "train_indices": train_subset.indices,
        "val_indices": val_subset.indices,
        "calibration_indices": assignments["calibration"],
        "evaluation_indices": assignments["evaluation"],
        "calibration_ids": calibration_ids,
        "evaluation_ids": evaluation_ids,
    }
    manifest_path = resolve_project_path(config["paths"]["split_manifest"])
    write_json(manifest_path, manifest)
    return manifest


def load_split_manifest(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = resolve_project_path(config["paths"]["split_manifest"])
    if not manifest_path.exists():
        return create_split_manifest(config)
    return read_json(manifest_path)


def build_datasets(config: dict[str, Any]) -> PetDatasetBundle:
    data_root = resolve_project_path(config["paths"]["data_root"])
    repair_incomplete_downloads(data_root)
    image_size = config["data"]["image_size"]
    download = bool(config["data"]["download"])
    manifest = load_split_manifest(config)
    seed = int(config["runtime"]["seed"])
    train_indices = maybe_subsample_indices(list(manifest["train_indices"]), config["data"].get("train_subset_size"), seed)
    val_indices = maybe_subsample_indices(list(manifest["val_indices"]), config["data"].get("val_subset_size"), seed + 1)

    trainval = OxfordIIITPet(
        root=data_root,
        split="trainval",
        target_types="category",
        transform=build_transforms(image_size, train=True),
        download=download,
    )
    trainval_eval = OxfordIIITPet(
        root=data_root,
        split="trainval",
        target_types="category",
        transform=build_transforms(image_size, train=False),
        download=download,
    )
    test = OxfordIIITPet(
        root=data_root,
        split="test",
        target_types="category",
        transform=build_transforms(image_size, train=False),
        download=download,
    )

    test_ids = _extract_sample_ids(test)
    calibration_ids = [test_ids[index] for index in manifest["calibration_indices"]]
    evaluation_ids = [test_ids[index] for index in manifest["evaluation_indices"]]

    return PetDatasetBundle(
        train=IndexedSubset(trainval, train_indices),
        val=IndexedSubset(trainval_eval, val_indices),
        calibration=IndexedSubset(test, list(manifest["calibration_indices"]), calibration_ids),
        evaluation=IndexedSubset(test, list(manifest["evaluation_indices"]), evaluation_ids),
    )


def collate_examples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    sample_ids = [item["sample_id"] for item in batch]
    dataset_indices = torch.tensor([item["dataset_index"] for item in batch], dtype=torch.long)
    return {
        "images": images,
        "labels": labels,
        "sample_ids": sample_ids,
        "dataset_indices": dataset_indices,
    }


def build_dataloaders(config: dict[str, Any], bundle: PetDatasetBundle) -> dict[str, DataLoader]:
    num_workers = int(config["runtime"]["num_workers"])
    return {
        "train": DataLoader(
            bundle.train,
            batch_size=int(config["data"]["train_batch_size"]),
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_examples,
        ),
        "val": DataLoader(
            bundle.val,
            batch_size=int(config["data"]["eval_batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_examples,
        ),
        "calibration": DataLoader(
            bundle.calibration,
            batch_size=int(config["data"]["eval_batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_examples,
        ),
        "evaluation": DataLoader(
            bundle.evaluation,
            batch_size=int(config["data"]["eval_batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_examples,
        ),
    }


def ensure_data_dirs(config: dict[str, Any]) -> None:
    ensure_dir(resolve_project_path(config["paths"]["data_root"]))
    ensure_dir(resolve_project_path(Path(config["paths"]["split_manifest"]).parent))
