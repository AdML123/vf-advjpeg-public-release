from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from vf_advjpeg.jpeg.backends import diffjpeg_surrogate, jpeg_identity_bpda
from vf_advjpeg.types import EstimationResult

from .quality import neighbor_qualities


def _compute_grad(loss: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
    return grad.detach()


@dataclass(slots=True)
class GradientEstimator:
    name: str

    def estimate(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor, quality: int) -> EstimationResult:
        raise NotImplementedError


class BPDAEstimator(GradientEstimator):
    def __init__(self) -> None:
        super().__init__(name="bpda")

    def estimate(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor, quality: int) -> EstimationResult:
        images = images.clone().detach().requires_grad_(True)
        logits = model(jpeg_identity_bpda(images, quality))
        loss = nn.functional.cross_entropy(logits, labels)
        return EstimationResult(pixel_grad=_compute_grad(loss, images), loss_value=float(loss.item()), qualities=[quality])


class DiffJPEGEstimator(GradientEstimator):
    def __init__(self) -> None:
        super().__init__(name="diffjpeg")

    def estimate(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor, quality: int) -> EstimationResult:
        images = images.clone().detach().requires_grad_(True)
        logits = model(diffjpeg_surrogate(images, quality, smooth_round=True))
        loss = nn.functional.cross_entropy(logits, labels)
        return EstimationResult(pixel_grad=_compute_grad(loss, images), loss_value=float(loss.item()), qualities=[quality])


class EOTEstimator(GradientEstimator):
    def __init__(self, neighbor_span: int, samples: int) -> None:
        super().__init__(name="eot")
        self.neighbor_span = neighbor_span
        self.samples = samples

    def estimate(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor, quality: int) -> EstimationResult:
        images = images.clone().detach().requires_grad_(True)
        qualities = neighbor_qualities(quality, self.neighbor_span, self.samples)
        losses = []
        for sample_quality in qualities:
            logits = model(jpeg_identity_bpda(images, sample_quality))
            losses.append(nn.functional.cross_entropy(logits, labels))
        loss = torch.stack(losses).mean()
        return EstimationResult(pixel_grad=_compute_grad(loss, images), loss_value=float(loss.item()), qualities=qualities)

