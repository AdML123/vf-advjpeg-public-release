from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vf_advjpeg.attacks.dct_ops import (
    ac_mask_like,
    reconstruct_rgb_from_y_dct,
    rgb_to_ycbcr,
    y_channel_dct,
    y_pixel_grad_to_dct,
)
from vf_advjpeg.attacks.estimators import BPDAEstimator, DiffJPEGEstimator, EOTEstimator, GradientEstimator
from vf_advjpeg.attacks.metrics import PerceptualMetrics, summarize_attack_metrics
from vf_advjpeg.attacks.quality import QualityPolicy
from vf_advjpeg.attacks.transforms import input_diversity, translation_invariant_smoothing
from vf_advjpeg.attacks.vf import VFGradientCorrector
from vf_advjpeg.jpeg.backends import apply_real_jpeg


@dataclass(slots=True)
class AttackMethod:
    name: str
    estimator: GradientEstimator
    use_momentum: bool
    use_ti: bool
    use_di: bool
    use_vf: bool = False


def attack_method_from_name(config: dict[str, Any], method_name: str) -> AttackMethod:
    attack_cfg = config["attack"]
    if method_name == "baseline_weak":
        return AttackMethod(method_name, BPDAEstimator(), use_momentum=False, use_ti=False, use_di=False)
    if method_name == "baseline_strong":
        return AttackMethod(
            method_name,
            EOTEstimator(int(attack_cfg["eot_neighbor_span"]), int(attack_cfg["eot_samples"])),
            use_momentum=True,
            use_ti=True,
            use_di=True,
        )
    if method_name == "baseline_diffjpeg":
        return AttackMethod(method_name, DiffJPEGEstimator(), use_momentum=True, use_ti=True, use_di=True)
    if method_name == "vf_advjpeg":
        return AttackMethod(method_name, BPDAEstimator(), use_momentum=True, use_ti=True, use_di=True, use_vf=True)
    raise KeyError(f"Unsupported attack method: {method_name}")


def _project_to_linf(clean: torch.Tensor, current: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.max(torch.min(current, clean + epsilon), clean - epsilon).clamp(0.0, 1.0)


def generate_adversarial_examples(
    config: dict[str, Any],
    source_model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    method_name: str,
    attack_policy: QualityPolicy,
    rng: np.random.Generator,
    vf_corrector: VFGradientCorrector | None = None,
) -> dict[str, Any]:
    attack_cfg = config["attack"]
    method = attack_method_from_name(config, method_name)
    epsilon = float(attack_cfg["epsilon"])
    alpha = float(attack_cfg["alpha"])
    steps = int(attack_cfg["steps"])
    momentum_factor = float(attack_cfg["momentum"])

    clean = images.detach()
    with torch.no_grad():
        clean_ycbcr = rgb_to_ycbcr(clean)
        clean_y_dct = y_channel_dct(clean)
        mask = ac_mask_like(clean_y_dct)

    delta_dct = torch.zeros_like(clean_y_dct)
    momentum = torch.zeros_like(clean_y_dct)
    estimator_times = []

    for step in range(steps):
        current_quality = attack_policy.sample(rng, step=step)
        adv = reconstruct_rgb_from_y_dct(clean_ycbcr, clean_y_dct + delta_dct * mask)
        adv = _project_to_linf(clean, adv, epsilon)
        delta_dct = (y_channel_dct(adv) - clean_y_dct) * mask

        attack_input = adv
        if method.use_di:
            attack_input = input_diversity(
                attack_input,
                rng=rng,
                probability=float(attack_cfg["di_prob"]),
                scale_min=float(attack_cfg["di_scale_min"]),
                scale_max=float(attack_cfg["di_scale_max"]),
            )

        start = time.perf_counter()
        estimation = method.estimator.estimate(source_model, attack_input, labels, current_quality)
        estimator_times.append(time.perf_counter() - start)
        pixel_grad = estimation.pixel_grad
        if method.use_ti:
            pixel_grad = translation_invariant_smoothing(
                pixel_grad,
                kernel_size=int(attack_cfg["ti_kernel_size"]),
                sigma=float(attack_cfg["ti_sigma"]),
            )
        dct_grad = y_pixel_grad_to_dct(pixel_grad) * mask
        if method.use_vf:
            if vf_corrector is None:
                raise ValueError("VF method requested without a VF corrector.")
            dct_grad = vf_corrector.correct(dct_grad, delta_dct, current_quality)

        denom = dct_grad.abs().mean(dim=(-1, -2, -3, -4, -5), keepdim=True).clamp_min(1e-6)
        normalized_grad = dct_grad / denom
        if method.use_momentum:
            momentum = momentum_factor * momentum + normalized_grad
            update = momentum
        else:
            update = normalized_grad

        delta_dct = delta_dct + alpha * update.sign() * mask

    with torch.no_grad():
        adversarial = reconstruct_rgb_from_y_dct(clean_ycbcr, clean_y_dct + delta_dct * mask)
        adversarial = _project_to_linf(clean, adversarial, epsilon)
    return {
        "adversarial_images": adversarial.detach(),
        "mean_estimator_time": float(np.mean(estimator_times)) if estimator_times else 0.0,
        "steps": steps,
    }


def evaluate_adversarial_examples(
    target_model: nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    adversarial: torch.Tensor,
    final_quality: int,
    metric_backend: PerceptualMetrics,
) -> dict[str, Any]:
    with torch.no_grad():
        clean_logits = target_model(clean)
        jpeg_clean_logits = target_model(apply_real_jpeg(clean, final_quality))
        adv_logits = target_model(apply_real_jpeg(adversarial, final_quality))

    metrics = summarize_attack_metrics(
        clean_labels=labels,
        clean_predictions=clean_logits.argmax(dim=1),
        jpeg_clean_predictions=jpeg_clean_logits.argmax(dim=1),
        adv_predictions=adv_logits.argmax(dim=1),
        clean_images=clean,
        adversarial_images=adversarial,
        metric_backend=metric_backend,
    ).to_dict()
    metrics["attack_quality"] = int(final_quality)
    return {
        "adversarial_images": adversarial.detach(),
        "clean_logits": clean_logits.detach(),
        "jpeg_clean_logits": jpeg_clean_logits.detach(),
        "adv_logits": adv_logits.detach(),
        "metrics": metrics,
    }


def run_attack(
    config: dict[str, Any],
    source_model: nn.Module,
    target_model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    method_name: str,
    attack_policy: QualityPolicy,
    eval_policy: QualityPolicy,
    rng: np.random.Generator,
    vf_corrector: VFGradientCorrector | None = None,
    metric_backend: PerceptualMetrics | None = None,
) -> dict[str, Any]:
    generated = generate_adversarial_examples(
        config=config,
        source_model=source_model,
        images=images,
        labels=labels,
        method_name=method_name,
        attack_policy=attack_policy,
        rng=rng,
        vf_corrector=vf_corrector,
    )
    final_quality = eval_policy.sample(rng)
    backend = metric_backend or PerceptualMetrics(images.device)
    results = evaluate_adversarial_examples(
        target_model=target_model,
        clean=images.detach(),
        labels=labels,
        adversarial=generated["adversarial_images"],
        final_quality=int(final_quality),
        metric_backend=backend,
    )
    results["metrics"]["mean_estimator_time"] = float(generated["mean_estimator_time"])
    results["metrics"]["steps"] = int(generated["steps"])
    return results
