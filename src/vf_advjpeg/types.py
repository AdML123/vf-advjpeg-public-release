from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


JsonDict = dict[str, Any]


@dataclass(slots=True)
class AttackBatch:
    images: torch.Tensor
    labels: torch.Tensor
    sample_ids: list[str]


@dataclass(slots=True)
class AttackOutputs:
    clean_images: torch.Tensor
    adversarial_images: torch.Tensor
    clean_labels: torch.Tensor
    target_predictions: torch.Tensor
    adversarial_predictions: torch.Tensor
    metrics: JsonDict


@dataclass(slots=True)
class EstimationResult:
    pixel_grad: torch.Tensor
    loss_value: float
    qualities: list[int] = field(default_factory=list)
    diagnostics: JsonDict = field(default_factory=dict)

