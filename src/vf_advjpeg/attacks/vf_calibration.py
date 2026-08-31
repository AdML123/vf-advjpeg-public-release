from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from vf_advjpeg.attacks.dct_ops import (
    ac_frequency_indices,
    ac_mask_like,
    reconstruct_rgb_from_y_dct,
    rgb_to_ycbcr,
    y_channel_dct,
    y_pixel_grad_to_dct,
)
from vf_advjpeg.attacks.estimators import BPDAEstimator, EOTEstimator
from vf_advjpeg.attacks.transforms import input_diversity
from vf_advjpeg.attacks.vf import VFArtifact, build_vf_artifact
from vf_advjpeg.jpeg.tables import scaled_quant_table


@dataclass(slots=True)
class VFCalibrationResult:
    artifact: VFArtifact
    aggregated_responses: np.ndarray
    counts: np.ndarray


def _bucket_index(normalized_magnitude: torch.Tensor, bucket_edges: list[float]) -> torch.Tensor:
    bucket = torch.zeros_like(normalized_magnitude, dtype=torch.long)
    if bucket_edges:
        bucket = bucket + (normalized_magnitude >= bucket_edges[0]).long()
    if len(bucket_edges) > 1:
        bucket = bucket + (normalized_magnitude >= bucket_edges[1]).long()
    return bucket.clamp(min=0, max=len(bucket_edges))


def calibrate_vf(
    config: dict[str, Any],
    source_model: nn.Module,
    calibration_loader,
    device: torch.device,
    max_batches: int | None = None,
) -> VFCalibrationResult:
    attack_cfg = config["attack"]
    vf_cfg = config["vf"]
    qualities = list(range(int(vf_cfg["quality_min"]), int(vf_cfg["quality_max"]) + 1))
    bucket_edges = [float(edge) for edge in vf_cfg["amplitude_buckets"]]
    frequency_indices = ac_frequency_indices()
    num_bins = len(frequency_indices)
    num_buckets = len(bucket_edges) + 1
    sums = np.zeros((num_bins, num_buckets, len(qualities)), dtype=np.float64)
    counts = np.zeros((num_bins, num_buckets, len(qualities)), dtype=np.float64)
    rng = np.random.default_rng(int(config["runtime"]["seed"]))

    proxy_estimator = BPDAEstimator()
    teacher_estimator = EOTEstimator(
        neighbor_span=int(attack_cfg["eot_neighbor_span"]),
        samples=int(attack_cfg["teacher_eot_samples"]),
    )
    ratio_clip = float(vf_cfg["ratio_clip"])
    noise_scale = float(vf_cfg["calibration_noise_scale"])
    samples_per_image = int(vf_cfg["calibration_samples_per_image"])

    source_model.eval()
    progress = tqdm(calibration_loader, desc="vf calibration", leave=False)
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        with torch.no_grad():
            clean_ycbcr = rgb_to_ycbcr(images)
            clean_y_dct = y_channel_dct(images)
            mask = ac_mask_like(clean_y_dct)

        for _ in range(samples_per_image):
            noise = torch.randn_like(clean_y_dct)
            for quality_index, quality in enumerate(qualities):
                quant = scaled_quant_table(
                    quality,
                    "luma",
                    device=images.device,
                    dtype=images.dtype,
                ).view(1, 1, 1, 1, 8, 8)
                delta_dct = noise * quant * noise_scale * mask
                adversarial = reconstruct_rgb_from_y_dct(clean_ycbcr, clean_y_dct + delta_dct)
                adversarial = adversarial.clamp(0.0, 1.0)
                attack_input = input_diversity(
                    adversarial,
                    rng=rng,
                    probability=float(attack_cfg["di_prob"]),
                    scale_min=float(attack_cfg["di_scale_min"]),
                    scale_max=float(attack_cfg["di_scale_max"]),
                )

                proxy = proxy_estimator.estimate(source_model, attack_input, labels, quality)
                teacher = teacher_estimator.estimate(source_model, attack_input, labels, quality)
                proxy_dct = y_pixel_grad_to_dct(proxy.pixel_grad) * mask
                teacher_dct = y_pixel_grad_to_dct(teacher.pixel_grad) * mask
                bucket_tensor = _bucket_index(delta_dct.abs() / quant, bucket_edges)

                for bin_index, (u, v) in enumerate(frequency_indices):
                    ratio = torch.where(
                        proxy_dct[..., u, v].abs() > 1e-6,
                        teacher_dct[..., u, v] / proxy_dct[..., u, v],
                        torch.ones_like(proxy_dct[..., u, v]),
                    )
                    ratio = ratio.clamp(min=-ratio_clip, max=ratio_clip)
                    buckets = bucket_tensor[..., u, v]
                    for bucket_index in range(num_buckets):
                        values = ratio[buckets == bucket_index]
                        if values.numel() == 0:
                            continue
                        sums[bin_index, bucket_index, quality_index] += float(values.sum().item())
                        counts[bin_index, bucket_index, quality_index] += float(values.numel())

    aggregated = np.divide(sums, counts, out=np.ones_like(sums), where=counts > 0)
    artifact = build_vf_artifact(
        quality_grid=qualities,
        aggregated_responses=aggregated,
        amplitude_buckets=bucket_edges,
        max_model_order=int(vf_cfg["max_model_order"]),
        fit_error_threshold=float(vf_cfg["fit_error_threshold"]),
    )
    artifact.fit_diagnostics["counts"] = counts.tolist()
    return VFCalibrationResult(artifact=artifact, aggregated_responses=aggregated, counts=counts)

