from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from contextlib import contextmanager

import numpy as np
import torch
from skimage.metrics import structural_similarity

try:
    import lpips
except Exception:  # pragma: no cover
    lpips = None


@dataclass(slots=True)
class MetricBundle:
    clean_accuracy: float
    jpeg_clean_accuracy: float
    clean_drop: float
    asr: float
    lpips: float
    ssim: float
    linf: float

    def to_dict(self) -> dict[str, float]:
        return {
            "clean_accuracy": self.clean_accuracy,
            "jpeg_clean_accuracy": self.jpeg_clean_accuracy,
            "clean_drop": self.clean_drop,
            "asr": self.asr,
            "lpips": self.lpips,
            "ssim": self.ssim,
            "linf": self.linf,
        }


@contextmanager
def _patched_lpips_alexnet(weights_path: Path) -> Iterator[None]:
    import lpips.pretrained_networks as pretrained_networks

    original_alexnet = pretrained_networks.tv.alexnet

    def _local_alexnet(*args, pretrained: bool = False, **kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("pretrained", None)
        kwargs.pop("weights", None)
        model = original_alexnet(*args, weights=None, **kwargs)
        if pretrained:
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        return model

    pretrained_networks.tv.alexnet = _local_alexnet
    try:
        yield
    finally:
        pretrained_networks.tv.alexnet = original_alexnet


class PerceptualMetrics:
    def __init__(
        self,
        device: torch.device,
        alexnet_weights_path: str | Path | None = None,
        allow_weight_download: bool = True,
    ) -> None:
        self.lpips_model = None
        weights_path = Path(alexnet_weights_path) if alexnet_weights_path is not None else None
        if lpips is not None:
            try:
                if weights_path is not None:
                    if not weights_path.exists():
                        raise FileNotFoundError(f"Missing local AlexNet weights: {weights_path}")
                    with _patched_lpips_alexnet(weights_path):
                        self.lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device)
                elif not allow_weight_download:
                    raise RuntimeError("LPIPS initialization requires a local AlexNet weights path when downloads are disabled.")
                else:
                    self.lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device)
                self.lpips_model.eval()
            except Exception:
                if not allow_weight_download:
                    raise
                self.lpips_model = None

    def compute_lpips(self, clean: torch.Tensor, adversarial: torch.Tensor) -> float:
        if self.lpips_model is None:
            return float("nan")
        with torch.no_grad():
            value = self.lpips_model(clean * 2.0 - 1.0, adversarial * 2.0 - 1.0).mean().item()
        return float(value)

    def compute_ssim(self, clean: torch.Tensor, adversarial: torch.Tensor) -> float:
        clean_np = clean.detach().cpu().permute(0, 2, 3, 1).numpy()
        adv_np = adversarial.detach().cpu().permute(0, 2, 3, 1).numpy()
        scores = [
            structural_similarity(clean_image, adv_image, channel_axis=2, data_range=1.0)
            for clean_image, adv_image in zip(clean_np, adv_np)
        ]
        return float(np.mean(scores))


def summarize_attack_metrics(
    clean_labels: torch.Tensor,
    clean_predictions: torch.Tensor,
    jpeg_clean_predictions: torch.Tensor,
    adv_predictions: torch.Tensor,
    clean_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    metric_backend: PerceptualMetrics,
) -> MetricBundle:
    clean_correct = clean_predictions.eq(clean_labels)
    clean_accuracy = float(clean_correct.float().mean().item())
    jpeg_clean_accuracy = float(jpeg_clean_predictions.eq(clean_labels).float().mean().item())
    eligible = clean_correct
    if eligible.any():
        asr = float(adv_predictions[eligible].ne(clean_labels[eligible]).float().mean().item())
    else:
        asr = 0.0
    clean_drop = clean_accuracy - jpeg_clean_accuracy
    linf = float((adversarial_images - clean_images).abs().amax().item())
    return MetricBundle(
        clean_accuracy=clean_accuracy,
        jpeg_clean_accuracy=jpeg_clean_accuracy,
        clean_drop=clean_drop,
        asr=asr,
        lpips=metric_backend.compute_lpips(clean_images, adversarial_images),
        ssim=metric_backend.compute_ssim(clean_images, adversarial_images),
        linf=linf,
    )
