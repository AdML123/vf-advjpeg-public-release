from __future__ import annotations

import time
import re
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm

from vf_advjpeg.results import collect_result_frame as _collect_result_frame
from vf_advjpeg.study import study_metadata
from vf_advjpeg.utils.fs import ensure_dir, read_json, resolve_project_path, write_json


PATH_UNSAFE_PATTERN = re.compile(r'[<>:"/\\|?*]+')


@dataclass(slots=True)
class RunAccumulator:
    total_examples: int = 0
    clean_correct: int = 0
    jpeg_clean_correct: int = 0
    eligible_examples: int = 0
    adversarial_successes: int = 0
    lpips_sum: float = 0.0
    ssim_sum: float = 0.0
    linf_sum: float = 0.0
    estimator_time_sum: float = 0.0
    batches: int = 0

    def update(self, labels: torch.Tensor, results: dict[str, Any]) -> None:
        clean_predictions = results["clean_logits"].argmax(dim=1).cpu()
        jpeg_clean_predictions = results["jpeg_clean_logits"].argmax(dim=1).cpu()
        adv_predictions = results["adv_logits"].argmax(dim=1).cpu()
        labels_cpu = labels.cpu()

        clean_correct_mask = clean_predictions.eq(labels_cpu)
        self.total_examples += int(labels_cpu.numel())
        self.clean_correct += int(clean_correct_mask.sum().item())
        self.jpeg_clean_correct += int(jpeg_clean_predictions.eq(labels_cpu).sum().item())
        self.eligible_examples += int(clean_correct_mask.sum().item())
        self.adversarial_successes += int(adv_predictions[clean_correct_mask].ne(labels_cpu[clean_correct_mask]).sum().item())
        self.lpips_sum += float(results["metrics"]["lpips"]) * labels_cpu.numel()
        self.ssim_sum += float(results["metrics"]["ssim"]) * labels_cpu.numel()
        self.linf_sum += float(results["metrics"]["linf"]) * labels_cpu.numel()
        self.estimator_time_sum += float(results["metrics"]["mean_estimator_time"])
        self.batches += 1

    def finalize(self) -> dict[str, float]:
        clean_accuracy = self.clean_correct / max(self.total_examples, 1)
        jpeg_clean_accuracy = self.jpeg_clean_correct / max(self.total_examples, 1)
        asr = self.adversarial_successes / max(self.eligible_examples, 1)
        return {
            "clean_accuracy": clean_accuracy,
            "jpeg_clean_accuracy": jpeg_clean_accuracy,
            "clean_drop": clean_accuracy - jpeg_clean_accuracy,
            "asr": asr,
            "lpips": self.lpips_sum / max(self.total_examples, 1),
            "ssim": self.ssim_sum / max(self.total_examples, 1),
            "linf": self.linf_sum / max(self.total_examples, 1),
            "mean_estimator_time": self.estimator_time_sum / max(self.batches, 1),
            "examples": self.total_examples,
            "eligible_examples": self.eligible_examples,
            "adversarial_successes": self.adversarial_successes,
        }


def build_source_ensemble(config: dict[str, Any], device: torch.device) -> WeightedEnsemble:
    from vf_advjpeg.models.training import load_frozen_model
    from vf_advjpeg.models.wrapper import WeightedEnsemble

    models_with_weights = []
    for item in config["models"]["source"]:
        model = load_frozen_model(config, item["name"], device)
        models_with_weights.append((model, float(item["weight"])))
    ensemble = WeightedEnsemble(models_with_weights).to(device)
    ensemble.eval()
    return ensemble


def build_target_model(config: dict[str, Any], model_name: str, device: torch.device) -> torch.nn.Module:
    from vf_advjpeg.models.training import load_frozen_model

    return load_frozen_model(config, model_name, device)


def result_path(config: dict[str, Any], method: str, target_model: str, suite_name: str, seed: int) -> Path:
    root = resolve_project_path(config["paths"]["run_root"])
    safe_target = PATH_UNSAFE_PATTERN.sub("_", target_model).strip(" .") or "target"
    path = root / method / safe_target / suite_name
    ensure_dir(path)
    return path / f"seed_{seed}.json"


def iter_run_specs(
    config: dict[str, Any],
    methods: Iterable[str] | None = None,
    targets: Iterable[str] | None = None,
    suites: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
):
    method_names = list(methods or config["experiments"]["methods"])
    target_names = list(targets or config["models"]["targets"])
    suite_names = list(suites or config["experiments"]["quality_suites"].keys())
    run_seeds = list(seeds or config["experiments"]["seeds"])
    for method in method_names:
        for target in target_names:
            for suite in suite_names:
                for seed in run_seeds:
                    yield method, target, suite, int(seed)


def build_metric_backend(config: dict[str, Any], device: torch.device):
    from vf_advjpeg.attacks.metrics import PerceptualMetrics

    perceptual_cfg = config.get("perceptual", {})
    alexnet_weights_path = perceptual_cfg.get("alexnet_weights_path")
    resolved_weights = resolve_project_path(alexnet_weights_path) if alexnet_weights_path else None
    allow_weight_download = bool(perceptual_cfg.get("allow_weight_download", True))
    return PerceptualMetrics(
        device,
        alexnet_weights_path=resolved_weights,
        allow_weight_download=allow_weight_download,
    )


def run_single_experiment(
    config: dict[str, Any],
    evaluation_loader,
    source_model: torch.nn.Module,
    target_model_name: str,
    method_name: str,
    suite_name: str,
    seed: int,
    device: torch.device,
    overwrite: bool = False,
    max_batches: int | None = None,
) -> Path:
    from vf_advjpeg.attacks.core import run_attack
    from vf_advjpeg.attacks.quality import quality_policy_from_config
    from vf_advjpeg.attacks.vf import VFGradientCorrector, load_vf_artifact

    output_path = result_path(config, method_name, target_model_name, suite_name, seed)
    if output_path.exists() and not overwrite:
        return output_path

    target_model = build_target_model(config, target_model_name, device)
    suite_cfg = config["experiments"]["quality_suites"][suite_name]
    attack_policy = quality_policy_from_config(suite_cfg["attack_policy"])
    eval_policy = quality_policy_from_config(suite_cfg["eval_policy"])
    rng = np.random.default_rng(seed)
    vf_corrector = None
    if method_name == "vf_advjpeg":
        vf_corrector = VFGradientCorrector(
            load_vf_artifact(config["paths"]["vf_artifact"]),
            structure_mode=str(config.get("vf", {}).get("structure_mode", "full")),
        )

    accumulator = RunAccumulator()
    metric_backend = build_metric_backend(config, device)
    start = time.perf_counter()
    progress = tqdm(evaluation_loader, desc=f"{method_name}/{target_model_name}/{suite_name}/seed{seed}", leave=False)
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        results = run_attack(
            config=config,
            source_model=source_model,
            target_model=target_model,
            images=images,
            labels=labels,
            method_name=method_name,
            attack_policy=attack_policy,
            eval_policy=eval_policy,
            rng=rng,
            vf_corrector=vf_corrector,
            metric_backend=metric_backend,
        )
        accumulator.update(labels, results)

    payload = {
        "method": method_name,
        "target_model": target_model_name,
        "suite": suite_name,
        "seed": seed,
        "metrics": accumulator.finalize(),
        "duration_sec": time.perf_counter() - start,
        "device": str(device),
        "hardware": {
            "device": str(device),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "thread_env": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            },
        },
        "attack_policy": suite_cfg["attack_policy"],
        "eval_policy": suite_cfg["eval_policy"],
    }
    payload.update(study_metadata(config))
    write_json(output_path, payload)
    return output_path


def run_transfer_matrix_experiment(
    config: dict[str, Any],
    evaluation_loader,
    source_model: torch.nn.Module,
    target_model_names: Iterable[str],
    method_name: str,
    suite_name: str,
    seed: int,
    device: torch.device,
    overwrite: bool = False,
    max_batches: int | None = None,
) -> list[Path]:
    from vf_advjpeg.attacks.core import generate_adversarial_examples
    from vf_advjpeg.attacks.quality import quality_policy_from_config
    from vf_advjpeg.attacks.vf import VFGradientCorrector, load_vf_artifact
    from vf_advjpeg.jpeg.backends import apply_real_jpeg

    targets = list(target_model_names)
    output_paths = [result_path(config, method_name, target, suite_name, seed) for target in targets]
    if output_paths and all(path.exists() for path in output_paths) and not overwrite:
        return output_paths

    target_models = {target: build_target_model(config, target, device) for target in targets}
    suite_cfg = config["experiments"]["quality_suites"][suite_name]
    attack_policy = quality_policy_from_config(suite_cfg["attack_policy"])
    eval_policy = quality_policy_from_config(suite_cfg["eval_policy"])
    rng = np.random.default_rng(seed)
    vf_corrector = None
    if method_name == "vf_advjpeg":
        vf_corrector = VFGradientCorrector(
            load_vf_artifact(config["paths"]["vf_artifact"]),
            structure_mode=str(config.get("vf", {}).get("structure_mode", "full")),
        )

    accumulators = {target: RunAccumulator() for target in targets}
    target_eval_seconds = {target: 0.0 for target in targets}
    attack_seconds = 0.0
    metric_backend = build_metric_backend(config, device)
    progress = tqdm(evaluation_loader, desc=f"{method_name}/transfer/{suite_name}/seed{seed}", leave=False)
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)

        attack_start = time.perf_counter()
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
        attack_seconds += time.perf_counter() - attack_start
        final_quality = eval_policy.sample(rng)
        adversarial = generated["adversarial_images"]
        with torch.no_grad():
            jpeg_clean = apply_real_jpeg(images, int(final_quality))
            jpeg_adversarial = apply_real_jpeg(adversarial, int(final_quality))
        image_metrics = {
            "lpips": metric_backend.compute_lpips(images, adversarial),
            "ssim": metric_backend.compute_ssim(images, adversarial),
            "linf": float((adversarial - images).abs().amax().item()),
            "attack_quality": int(final_quality),
        }

        for target, target_model in target_models.items():
            eval_start = time.perf_counter()
            with torch.no_grad():
                clean_logits = target_model(images)
                jpeg_clean_logits = target_model(jpeg_clean)
                adv_logits = target_model(jpeg_adversarial)
            results = {
                "adversarial_images": adversarial.detach(),
                "clean_logits": clean_logits.detach(),
                "jpeg_clean_logits": jpeg_clean_logits.detach(),
                "adv_logits": adv_logits.detach(),
                "metrics": dict(image_metrics),
            }
            results["metrics"]["mean_estimator_time"] = float(generated["mean_estimator_time"])
            results["metrics"]["steps"] = int(generated.get("steps", config.get("attack", {}).get("steps", 0)))
            target_eval_seconds[target] += time.perf_counter() - eval_start
            accumulators[target].update(labels, results)

    written_paths: list[Path] = []
    for target, output_path in zip(targets, output_paths):
        payload = {
            "method": method_name,
            "target_model": target,
            "suite": suite_name,
            "seed": seed,
            "metrics": accumulators[target].finalize(),
            "duration_sec": attack_seconds + target_eval_seconds[target],
            "device": str(device),
            "hardware": {
                "device": str(device),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "thread_env": {
                    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
                    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                },
            },
            "attack_policy": suite_cfg["attack_policy"],
            "eval_policy": suite_cfg["eval_policy"],
        }
        payload.update(study_metadata(config))
        write_json(output_path, payload)
        written_paths.append(output_path)
    return written_paths


def collect_result_frame(config: dict[str, Any]):
    return _collect_result_frame(config)
