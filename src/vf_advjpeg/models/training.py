from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from vf_advjpeg.models.checkpoints import save_checkpoint
from vf_advjpeg.models.factory import build_model
from vf_advjpeg.models.wrapper import NormalizedModel


@dataclass(slots=True)
class EpochMetrics:
    loss: float
    accuracy: float


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())


def evaluate(model: nn.Module, loader, device: torch.device) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_examples += int(labels.size(0))
    return EpochMetrics(
        loss=total_loss / max(total_examples, 1),
        accuracy=total_correct / max(total_examples, 1),
    )


def _trainable_parameters(model: NormalizedModel, model_name: str, freeze_backbone: bool):
    if not freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return model.parameters()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if model_name == "resnet18":
        params = list(model.model.fc.parameters())
    elif model_name in {"vgg16", "mobilenet_v2"}:
        params = list(model.model.classifier[-1].parameters())
    elif model_name == "densenet121":
        params = list(model.model.classifier.parameters())
    else:
        raise KeyError(f"Unsupported model: {model_name}")

    for parameter in params:
        parameter.requires_grad_(True)
    return params


def train_classifier(
    config: dict[str, Any],
    model_name: str,
    dataloaders: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    num_classes = int(config["models"]["num_classes"])
    mean = list(config["data"]["image_mean"])
    std = list(config["data"]["image_std"])

    model = NormalizedModel(build_model(model_name, num_classes, pretrained=True), mean, std).to(device)
    trainable_parameters = _trainable_parameters(
        model,
        model_name=model_name,
        freeze_backbone=bool(config["training"].get("freeze_backbone", False)),
    )
    optimizer = AdamW(
        trainable_parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["training"]["label_smoothing"]))
    epochs = int(config["training"]["epochs"])
    patience = int(config["training"]["early_stopping_patience"])
    grad_clip = float(config["training"]["grad_clip_norm"])

    best_state = None
    best_accuracy = -1.0
    patience_left = patience
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_examples = 0
        progress = tqdm(dataloaders["train"], desc=f"{model_name} epoch {epoch + 1}/{epochs}", leave=False)
        for batch in progress:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            running_correct += int((logits.argmax(dim=1) == labels).sum().item())
            running_examples += int(labels.size(0))

        train_metrics = EpochMetrics(
            loss=running_loss / max(running_examples, 1),
            accuracy=running_correct / max(running_examples, 1),
        )
        val_metrics = evaluate(model, dataloaders["val"], device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "val_loss": val_metrics.loss,
                "val_accuracy": val_metrics.accuracy,
            }
        )

        if val_metrics.accuracy > best_accuracy:
            best_accuracy = val_metrics.accuracy
            best_state = {
                "model_state": model.state_dict(),
                "history": history,
                "best_accuracy": best_accuracy,
            }
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left < 0:
                break

    if best_state is None:
        raise RuntimeError("Training finished without a best checkpoint.")

    save_checkpoint(
        config,
        model_name,
        best_state["model_state"],
        {"history": best_state["history"], "best_accuracy": best_state["best_accuracy"]},
    )
    return best_state


def load_frozen_model(config: dict[str, Any], model_name: str, device: torch.device) -> nn.Module:
    from vf_advjpeg.models.checkpoints import load_checkpoint
    from vf_advjpeg.models.factory import build_cifar10_model, build_imagenet_model, build_robustbench_model
    from vf_advjpeg.models.wrapper import NormalizedModel

    mean = list(config["data"]["image_mean"])
    std = list(config["data"]["image_std"])
    checkpoint_root = config["paths"]["checkpoint_root"]
    dataset_name = str(config.get("data", {}).get("dataset", "pet37"))

    if model_name.startswith("robustbench:"):
        _, dataset, threat_model, robust_model_name = model_name.split(":", maxsplit=3)
        model = build_robustbench_model(
            robust_model_name,
            dataset=dataset,
            threat_model=threat_model,
            model_dir=checkpoint_root,
        )
    elif dataset_name == "imagenet1k_subset":
        model = NormalizedModel(
            build_imagenet_model(model_name, checkpoint_root=checkpoint_root, pretrained=True),
            mean,
            std,
        )
    elif dataset_name == "cifar10" and model_name == "cifar10_resnet18":
        model = NormalizedModel(build_cifar10_model(model_name, checkpoint_root=checkpoint_root), mean, std)
    else:
        num_classes = int(config["models"]["num_classes"])
        payload = load_checkpoint(config, model_name, map_location=device)
        model = NormalizedModel(build_model(model_name, num_classes, pretrained=False), mean, std)
        model.load_state_dict(payload["model_state"])

    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
