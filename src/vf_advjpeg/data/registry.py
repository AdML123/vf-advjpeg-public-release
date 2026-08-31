from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.utils.data import DataLoader, Dataset, Subset

from vf_advjpeg.data.pet import build_dataloaders as build_pet_dataloaders
from vf_advjpeg.data.pet import build_datasets as build_pet_datasets
from vf_advjpeg.data.pet import collate_examples
from vf_advjpeg.data.cifar import build_cifar10_eval_dataset
from vf_advjpeg.data.imagenet import build_imagenet1k_subset
from vf_advjpeg.utils.fs import resolve_project_path


@dataclass(slots=True)
class EvaluationDatasetBundle:
    calibration: Dataset
    evaluation: Dataset


def _limited_subset(dataset: Dataset, size: int | None) -> Dataset:
    if size is None or size <= 0 or size >= len(dataset):
        return dataset
    return Subset(dataset, list(range(size)))


def _build_loader(config: dict[str, Any], dataset: Dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config["data"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(config["runtime"]["num_workers"]),
        collate_fn=collate_examples,
    )


def build_evaluation_datasets(config: dict[str, Any]) -> EvaluationDatasetBundle:
    dataset_name = str(config.get("data", {}).get("dataset", "pet37"))
    if dataset_name in {"pet37", "oxford_iiit_pet"}:
        bundle = build_pet_datasets(config)
        return EvaluationDatasetBundle(calibration=bundle.calibration, evaluation=bundle.evaluation)

    data_root = resolve_project_path(config["paths"]["data_root"])
    image_size = int(config["data"]["image_size"])
    if dataset_name == "cifar10":
        dataset = build_cifar10_eval_dataset(
            data_root,
            image_size=image_size,
            download=bool(config["data"].get("download", False)),
        )
    elif dataset_name == "imagenet1k_subset":
        dataset = build_imagenet1k_subset(data_root, image_size=image_size)
    else:
        raise KeyError(f"Unsupported dataset: {dataset_name}")

    calibration_size = config["data"].get("calibration_size")
    evaluation_size = config["data"].get("evaluation_size")
    return EvaluationDatasetBundle(
        calibration=_limited_subset(dataset, int(calibration_size) if calibration_size is not None else None),
        evaluation=_limited_subset(dataset, int(evaluation_size) if evaluation_size is not None else None),
    )


def build_evaluation_loaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    dataset_name = str(config.get("data", {}).get("dataset", "pet37"))
    if dataset_name in {"pet37", "oxford_iiit_pet"}:
        return build_pet_dataloaders(config, build_pet_datasets(config))

    bundle = build_evaluation_datasets(config)
    return {
        "calibration": _build_loader(config, bundle.calibration),
        "evaluation": _build_loader(config, bundle.evaluation),
    }
