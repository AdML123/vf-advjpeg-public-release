from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Callable
import importlib.util

import torch.nn as nn
from torchvision import models


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _optional_local_path(env_name: str, relative_default: str) -> Path:
    override = os.environ.get(env_name)
    return Path(override) if override else _repo_root() / relative_default


def _replace_classifier(model: nn.Module, model_name: str, num_classes: int) -> nn.Module:
    if model_name == "resnet18":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "vgg16":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == "mobilenet_v2":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == "densenet121":
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise KeyError(f"Unsupported model: {model_name}")
    return model


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    builders: dict[str, Callable[[], nn.Module]] = {
        "resnet18": lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None),
        "vgg16": lambda: models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None),
        "mobilenet_v2": lambda: models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        ),
        "densenet121": lambda: models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT if pretrained else None
        ),
    }
    if model_name not in builders:
        raise KeyError(f"Unsupported model: {model_name}")
    model = builders[model_name]()
    return _replace_classifier(model, model_name, num_classes)


def build_imagenet_model(model_name: str, checkpoint_root: str | Path, pretrained: bool = True) -> nn.Module:
    checkpoint_root = Path(checkpoint_root)
    os.environ["TORCH_HOME"] = str(checkpoint_root / "torch_home")

    if model_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        return models.vit_b_16(weights=weights)
    if model_name == "swin_t":
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        return models.swin_t(weights=weights)
    if model_name == "swin_v2_t":
        weights = models.Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None
        return models.swin_v2_t(weights=weights)
    if model_name in {"deit_tiny_patch16_224.fb_in1k", "deit_small_patch16_224.fb_in1k"}:
        import timm

        checkpoint_path = checkpoint_root / "imagenet" / model_name / "model.safetensors"
        if pretrained and not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing local DeiT checkpoint: {checkpoint_path}")
        return timm.create_model(
            model_name,
            pretrained=False,
            checkpoint_path=str(checkpoint_path) if pretrained else "",
        )
    raise KeyError(f"Unsupported ImageNet model: {model_name}")


def build_cifar10_model(model_name: str, checkpoint_root: str | Path) -> nn.Module:
    if model_name != "cifar10_resnet18":
        raise KeyError(f"Unsupported CIFAR-10 model: {model_name}")

    deeprobust_resnet = _optional_local_path(
        "VF_ADVJPEG_DEEPROBUST_RESNET",
        "DeepRobust-master/deeprobust/image/netmodels/resnet.py",
    )
    if not deeprobust_resnet.exists():
        raise FileNotFoundError(f"Missing local DeepRobust ResNet definition: {deeprobust_resnet}")
    spec = importlib.util.spec_from_file_location("_vf_advjpeg_deeprobust_resnet", deeprobust_resnet)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load DeepRobust ResNet definition: {deeprobust_resnet}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ResNet18 = module.ResNet18

    checkpoint_path = Path(checkpoint_root) / "cifar" / "CIFAR10_ResNet18_epoch_20.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing local CIFAR-10 checkpoint: {checkpoint_path}")
    import torch

    model = ResNet18()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return model


def _ensure_local_robustbench_importable() -> None:
    try:
        import robustbench  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    repo_candidates = [
        _optional_local_path("VF_ADVJPEG_ROBUSTBENCH_ROOT", "robustbench-master"),
        _optional_local_path("VF_ADVJPEG_AUTOATTACK_ROOT", "auto-attack-master"),
    ]
    for candidate in repo_candidates:
        if candidate.exists():
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)


def build_robustbench_model(
    model_name: str,
    dataset: str,
    threat_model: str,
    model_dir: str | Path,
) -> nn.Module:
    _ensure_local_robustbench_importable()
    from robustbench import load_model

    return load_model(
        model_name=model_name,
        dataset=dataset,
        threat_model=threat_model,
        model_dir=str(Path(model_dir)),
    )
