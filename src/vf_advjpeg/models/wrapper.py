from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from vf_advjpeg.utils.tensor import normalize_batch


class NormalizedModel(nn.Module):
    def __init__(self, model: nn.Module, mean: list[float], std: list[float]) -> None:
        super().__init__()
        self.model = model
        self.mean = mean
        self.std = std

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(normalize_batch(images, self.mean, self.std))


class WeightedEnsemble(nn.Module):
    def __init__(self, models_with_weights: Iterable[tuple[nn.Module, float]]) -> None:
        super().__init__()
        modules = []
        weights = []
        for model, weight in models_with_weights:
            modules.append(model)
            weights.append(weight)
        self.models = nn.ModuleList(modules)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = [weight * model(images) for model, weight in zip(self.models, self.weights)]
        return torch.stack(logits, dim=0).sum(dim=0)

