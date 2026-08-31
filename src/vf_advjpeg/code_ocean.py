from __future__ import annotations

import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from vf_advjpeg.config import deep_merge, dump_yaml, load_config
from vf_advjpeg.stats.baseline_viability import (
    ViabilityThresholds,
    baseline_viability_payload,
    evaluate_baseline_viability,
    summarize_baseline_viability,
)
from vf_advjpeg.study import study_metadata
from vf_advjpeg.stats.reviewer_evidence import reviewer_source_signature
from vf_advjpeg.stats.structure_diagnostics import write_vf_structure_diagnostics
from vf_advjpeg.utils.fs import ensure_dir, read_json, repo_root, sha256_file, write_json
from vf_advjpeg.utils.runtime import installed_package_versions


CAPSULE_DIRNAME = "code_ocean_capsule"
PUBLICATION_ARTIFACT_DIR = Path("artifacts/publication")
CODE_OCEAN_CAPSULE_ZIP_FILENAME = "VF-AdvJPEG_Code_Ocean_capsule.zip"
CODE_OCEAN_CAPSULE_VALIDATION_FILENAME = "code_ocean_capsule_validation.json"
ENVIRONMENT_LOCK_FILENAME = "environment_lock.json"
SUITE_ORDER = ["static_q70", "static_q80", "static_q90", "dynamic_uniform"]
ALEXNET_WEIGHTS_RELATIVE_PATH = "perceptual/alexnet-owt-7be5be79.pth"
ALEXNET_WEIGHTS_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
ALEXNET_WEIGHTS_SIZE_BYTES = 244408911
STARTER_ENVIRONMENT_LABEL = "PyTorch (2.4.0, CUDA 12.4.0, Mambaforge24.5.0-0, Python3.12.4, Ubuntu22.04)"
EXCLUDED_PACKAGE_PATHS = {
    "paper",
    "publishing.py",
    "submission.py",
    "submission_source_data.py",
    "stats/validation.py",
}
OVERLAY_REQUIREMENTS = [
    "lpips==0.1.4",
    "matplotlib==3.10.8",
    "pandas==2.3.3",
    "scikit-image==0.26.0",
    "scikit-rf==1.11.0",
    "scipy==1.17.1",
    "safetensors==0.7.0",
    "timm==1.0.22",
    "tqdm==4.67.1",
]
STARTER_ENVIRONMENT = {
    "label": STARTER_ENVIRONMENT_LABEL,
    "python": "3.12.4",
    "packages": {
        "torch": "2.4.0",
        "torchvision": "0.19.0",
        "numpy": "2.1.1",
        "PyYAML": "6.0.2",
        "pillow": "10.3.0",
    },
}
OVERLAY_PACKAGE_LOCK = {
    "lpips": "0.1.4",
    "matplotlib": "3.10.8",
    "pandas": "2.3.3",
    "scikit-image": "0.26.0",
    "scikit-rf": "1.11.0",
    "scipy": "1.17.1",
    "safetensors": "0.7.0",
    "timm": "1.0.22",
    "tqdm": "4.67.1",
}
RUNTIME_POLICY = {
    "device": "cpu",
    "cuda_visible_devices": "",
    "runtime_download_allowed": False,
}
REQUIRED_GUIDE_SNIPPETS = [
    STARTER_ENVIRONMENT_LABEL,
    "commit changes",
    "Reproducible Run",
    "Release version",
    "must not download data or weights at runtime",
]
REQUIRED_CAPSULE_FILES = [
    "CODE_OCEAN_UPLOAD_GUIDE.md",
    "README.md",
    "assets/fig1_mechanism.mmd",
    "assets/fig1_mermaid.css",
    "assets/fig1_mechanism.svg",
    "assets/figure_jpeg_aware_mechanism.png",
    "assets/canonical_expected_metrics.json",
    "assets/data_assets_manifest.json",
    f"assets/{ENVIRONMENT_LOCK_FILENAME}",
    "assets/pet37_ei_cpu_splits.json",
    "assets/paper_source_data/analysis/efficiency_tradeoff.csv",
    "assets/paper_source_data/calibration/vf_calibration_manifest.json",
    "assets/paper_source_data/calibration/vf_calibration_raw.npz",
    "assets/paper_source_data/pairwise/chain_summary.csv",
    "assets/paper_source_data/pairwise/chain_suite_tradeoff.csv",
    "assets/paper_source_data/pairwise/claim_check.json",
    "assets/paper_source_data/baseline_viability_report.csv",
    "assets/paper_source_data/baseline_viability_report.json",
    "configs/code_ocean_capsule.yaml",
    "data/DATA_UPLOAD_MANIFEST.md",
    f"environment/{ENVIRONMENT_LOCK_FILENAME}",
    "metadata/capsule_metadata.json",
    "postInstall",
    "requirements.txt",
    "run.sh",
    "scripts/run_code_ocean_repro.py",
    "src/vf_advjpeg/code_ocean.py",
    "src/vf_advjpeg/stats/pairwise.py",
    "src/vf_advjpeg/stats/plots.py",
]
REQUIRED_RUNTIME_CONFIGS = [
    "configs/default.yaml",
    "configs/ei_cpu_reconfirm.yaml",
    "configs/reviewer_cifar10.yaml",
    "configs/reviewer_imagenet.yaml",
    "configs/reviewer_imagenet_1000_cpu.yaml",
    "configs/reviewer_defended.yaml",
    "configs/reviewer_ablation.yaml",
    "configs/ei_cpu_matrix_resnet18.yaml",
    "configs/ei_cpu_matrix_vgg16.yaml",
    "configs/ei_cpu_matrix_mobilenet_v2.yaml",
    "configs/ei_cpu_matrix_densenet121.yaml",
    "configs/cpu_eps_4_255.yaml",
    "configs/cpu_eps_12_255.yaml",
    "configs/cpu_structure_no_vf.yaml",
    "configs/cpu_structure_frequency_only.yaml",
    "configs/cpu_structure_bucket_only.yaml",
    "configs/cpu_structure_full_vf.yaml",
    "configs/cpu_q50_q70_calibration.yaml",
]
FORBIDDEN_CAPSULE_PREFIXES = ("paper/", "IEEESPL_submission/")
FORBIDDEN_CAPSULE_SUFFIXES = (".pdf", ".tex")
METRIC_TOLERANCES = {
    "runtime_speedup_mean": 0.40,
    "estimator_speedup_mean": 0.50,
    "asr_retention_mean": 0.08,
    "clean_drop_delta_mean": 0.01,
}
METRIC_CORRECTIONS = {
    "static_q90": {
        "metric": "asr_retention_mean",
        "value": 0.9855072464,
        "source": "IEEESPL review report",
        "reason": "The review report observed 98.55% Static Q90 ASR retention, replacing the legacy underreported expected value.",
    }
}


def _is_canonical_static_q90_row(frame: pd.DataFrame) -> pd.Series:
    if "suite" not in frame.columns:
        return pd.Series(False, index=frame.index)
    mask = frame["suite"].astype(str) == "static_q90"
    selectors = {
        "study_id": "reconfirm",
        "source_profile": "r18_only",
        "dataset": "pet37",
        "source_model": "resnet18",
        "target_model": "mobilenet_v2",
    }
    for column, expected in selectors.items():
        if column in frame.columns:
            mask &= frame[column].astype(str) == expected
    return mask


WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(r"\b[A-Za-z]:[\\/]")
TABLE_I_METHODS = {
    "eot": "baseline_strong",
    "bpda": "baseline_weak",
    "diffjpeg": "baseline_diffjpeg",
    "vf": "vf_advjpeg",
}
PAPER_FIGURE_OUTPUTS = [
    "plots/fig1_mechanism.pdf",
    "plots/fig2_cpu_advantage_structure.pdf",
]
MANUSCRIPT_FIGURES = [
    "assets/fig1_mechanism.pdf",
    "assets/fig2_cpu_advantage_structure.pdf",
]
PAIRWISE_ROUTE_OVERRIDES = {
    "resnet18": "configs/ei_cpu_matrix_resnet18.yaml",
    "vgg16": "configs/ei_cpu_matrix_vgg16.yaml",
    "mobilenet_v2": "configs/ei_cpu_matrix_mobilenet_v2.yaml",
    "densenet121": "configs/ei_cpu_matrix_densenet121.yaml",
}
EXPANDED_CPU_ROUTE_SPECS: dict[str, tuple[str, ...]] = {
    "cifar10_standard": ("configs/reviewer_cifar10.yaml",),
    "imagenet_modern_arch": ("configs/reviewer_imagenet.yaml", "configs/reviewer_imagenet_1000_cpu.yaml"),
    "cifar10_robustbench": ("configs/reviewer_defended.yaml",),
    "budget_4_255": ("configs/ei_cpu_reconfirm.yaml", "configs/cpu_eps_4_255.yaml"),
    "budget_12_255": ("configs/ei_cpu_reconfirm.yaml", "configs/cpu_eps_12_255.yaml"),
    "structure_no_vf": ("configs/ei_cpu_reconfirm.yaml", "configs/cpu_structure_no_vf.yaml"),
    "structure_frequency_only": ("configs/ei_cpu_reconfirm.yaml", "configs/cpu_structure_frequency_only.yaml"),
    "structure_bucket_only": ("configs/ei_cpu_reconfirm.yaml", "configs/cpu_structure_bucket_only.yaml"),
    "structure_full_vf": ("configs/ei_cpu_reconfirm.yaml", "configs/cpu_structure_full_vf.yaml"),
    "quality_boundary": ("configs/ei_cpu_reconfirm.yaml", "configs/cpu_q50_q70_calibration.yaml"),
}
FIVE_SEED_PAPER_ROUTES = {
    "imagenet_modern_arch",
    "budget_4_255",
    "budget_12_255",
    "structure_no_vf",
    "structure_frequency_only",
    "structure_bucket_only",
    "structure_full_vf",
    "quality_boundary",
}
STRONGER_CPU_VIABILITY_ROUTES = {
    "cifar10_standard",
    "cifar10_robustbench",
    "imagenet_modern_arch",
}


@dataclass(slots=True)
class CodeOceanStageOutputs:
    root: Path
    config: Path
    expected_metrics: Path
    data_manifest: Path
    environment_lock: Path
    upload_guide: Path
    run_script: Path
    zip_path: Path
    validation: Path


@dataclass(slots=True)
class CodeOceanRunOutputs:
    headline_metrics: Path
    reproducibility_report: Path
    analysis_manifest: Path
    plot_manifest: Path
    run_manifest: Path


def build_code_ocean_config() -> dict[str, Any]:
    config = {
        "paths": {
            "data_root": "../data",
            "checkpoint_root": "../data/checkpoints",
            "perceptual_root": "../data/perceptual",
            "artifact_root": "../results/artifacts",
            "split_manifest": "assets/pet37_ei_cpu_splits.json",
            "vf_artifact": "../results/artifacts/vf/vf_first_pass.json",
            "plot_root": "../results/plots",
            "analysis_root": "../results/analysis",
            "run_root": "../results/runs",
        },
        "runtime": {
            "device": "cpu",
            "num_workers": 0,
            "deterministic": True,
            "seed": 42,
        },
        "data": {
            "image_size": 224,
            "train_batch_size": 8,
            "eval_batch_size": 8,
            "train_subset_size": None,
            "val_subset_size": None,
            "calibration_size": 32,
            "evaluation_size": 128,
            "validation_fraction": 0.1,
            "download": False,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        },
        "models": {
            "num_classes": 37,
            "source": [{"name": "resnet18", "weight": 1.0}],
            "targets": ["mobilenet_v2"],
            "all": ["resnet18", "mobilenet_v2"],
        },
        "training": {
            "epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "label_smoothing": 0.0,
            "freeze_backbone": True,
            "early_stopping_patience": 0,
            "checkpoint_metric": "val_accuracy",
            "grad_clip_norm": 1.0,
            "mixed_precision": False,
        },
        "attack": {
            "epsilon": 0.03137254901960784,
            "alpha": 0.006274509803921569,
            "steps": 6,
            "momentum": 0.9,
            "di_prob": 0.7,
            "di_scale_min": 0.8,
            "di_scale_max": 1.2,
            "ti_kernel_size": 3,
            "ti_sigma": 3.0,
            "eot_neighbor_span": 5,
            "eot_samples": 3,
            "teacher_eot_samples": 6,
            "batch_size": 4,
        },
        "jpeg": {
            "static_qualities": [70, 80, 90],
            "unseen_qualities": [72, 78, 84, 88, 92],
            "dynamic_min": 70,
            "dynamic_max": 95,
        },
        "vf": {
            "quality_min": 70,
            "quality_max": 90,
            "max_model_order": 8,
            "fit_error_threshold": 0.15,
            "ratio_clip": 5.0,
            "amplitude_buckets": [0.5, 1.5],
            "calibration_noise_scale": 0.35,
            "calibration_samples_per_image": 1,
            "fallback_interp": "pchip",
            "stable_reflect": True,
        },
        "perceptual": {
            "alexnet_weights_path": f"../data/{ALEXNET_WEIGHTS_RELATIVE_PATH}",
            "allow_weight_download": False,
        },
        "experiments": {
            "methods": ["baseline_weak", "baseline_strong", "baseline_diffjpeg", "vf_advjpeg"],
            "seeds": [42, 43, 44, 45, 46],
            "quality_suites": {
                "static_q70": {
                    "attack_policy": {"kind": "fixed", "quality": 70},
                    "eval_policy": {"kind": "fixed", "quality": 70},
                },
                "static_q80": {
                    "attack_policy": {"kind": "fixed", "quality": 80},
                    "eval_policy": {"kind": "fixed", "quality": 80},
                },
                "static_q90": {
                    "attack_policy": {"kind": "fixed", "quality": 90},
                    "eval_policy": {"kind": "fixed", "quality": 90},
                },
                "dynamic_uniform": {
                    "attack_policy": {"kind": "random_uniform_int", "quality_min": 70, "quality_max": 95},
                    "eval_policy": {"kind": "random_uniform_int", "quality_min": 70, "quality_max": 95},
                },
            },
        },
        "analysis": {
            "study_id": "reconfirm",
            "source_profile": "r18_only",
            "baseline_method": "baseline_strong",
            "candidate_method": "vf_advjpeg",
            "main_suites": SUITE_ORDER,
            "efficiency_targets": {
                "runtime_speedup_min": 3.0,
                "estimator_speedup_min": 3.0,
                "clean_drop_delta_max": 0.1,
                "asr_retention_min": 0.75,
            },
        },
        "code_ocean": {
            "expected_metrics_path": "assets/canonical_expected_metrics.json",
            "data_manifest_path": "assets/data_assets_manifest.json",
            "environment_lock_path": f"assets/{ENVIRONMENT_LOCK_FILENAME}",
            "paper_source_data_root": "assets/paper_source_data",
            "headline_metrics_path": "../results/headline_metrics.json",
            "reproducibility_report_path": "../results/reproducibility_report.json",
            "run_manifest_path": "../results/run_manifest.json",
            "rebuild_from_paper_source_data": False,
            "generate_expanded_cpu_routes": True,
            "generate_pairwise_matrix": True,
        },
    }
    config["code_ocean"]["expanded_cpu_routes"] = {
        name: {
            "overrides": list(overrides),
            "seeds": [42, 43, 44, 45, 46] if name in FIVE_SEED_PAPER_ROUTES else [42, 43, 44],
        }
        for name, overrides in EXPANDED_CPU_ROUTE_SPECS.items()
    }
    config["code_ocean"]["pairwise_routes"] = {
        source: {"override": override}
        for source, override in PAIRWISE_ROUTE_OVERRIDES.items()
    }
    return config


def _stage_root(output_dir: str | Path | None = None) -> Path:
    if output_dir is None:
        return repo_root() / CAPSULE_DIRNAME
    path = Path(output_dir)
    return path if path.is_absolute() else repo_root() / path


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _capsule_source_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.replace("\\", "/")
    for marker in ["/results/", "results/"]:
        index = normalized.find(marker)
        if index >= 0:
            return normalized[index + len(marker) :]
    for marker in ["/artifacts/", "artifacts/"]:
        index = normalized.find(marker)
        if index >= 0:
            return normalized[index + (1 if marker.startswith("/") else 0) :]
    drive_match = re.match(r"^[A-Za-z]:/(.*)$", normalized)
    if drive_match:
        return drive_match.group(1)
    return value


def _capsule_source_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _capsule_source_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_capsule_source_payload(item) for item in value]
    return _capsule_source_value(value)


def _copy_source_data_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
        for column in frame.select_dtypes(include=["object"]).columns:
            frame[column] = frame[column].map(_capsule_source_value)
        frame.to_csv(target, index=False)
        return
    if source.suffix.lower() == ".json":
        write_json(target, _capsule_source_payload(read_json(source)))
        return
    _copy_file(source, target)


def _copy_paper_source_data(stage_root: Path) -> None:
    target_root = stage_root / "assets" / "paper_source_data"
    tradeoff = _correct_table_i_frame(pd.read_csv(_canonical_tradeoff_path()))
    tradeoff_target = target_root / "analysis" / "efficiency_tradeoff.csv"
    tradeoff_target.parent.mkdir(parents=True, exist_ok=True)
    tradeoff.to_csv(tradeoff_target, index=False)
    calibration_manifest = read_json(_canonical_calibration_manifest_path())
    calibration_manifest["artifact_path"] = "artifacts/vf/vf_first_pass.json"
    calibration_manifest["raw_npz_path"] = "artifacts/vf/vf_calibration_raw.npz"
    write_json(target_root / "calibration" / "vf_calibration_manifest.json", calibration_manifest)
    _copy_file(_canonical_calibration_raw_npz_path(), target_root / "calibration" / "vf_calibration_raw.npz")
    write_vf_structure_diagnostics(target_root / "calibration" / "vf_calibration_raw.npz", target_root / "calibration")
    pairwise_root = _paper_pairwise_bundle_dir()
    for filename in ["chain_summary.csv", "chain_suite_tradeoff.csv", "claim_check.json"]:
        _copy_file(pairwise_root / filename, target_root / "pairwise" / filename)
    reviewer_cifar10 = _reviewer_cifar10_root()
    if reviewer_cifar10.exists():
        _write_route_viability_if_missing(reviewer_cifar10 / "analysis")
        _copy_tree_files(
            reviewer_cifar10 / "analysis",
            target_root / "reviewer_cifar10" / "analysis",
            ["efficiency_tradeoff.csv", "summary.csv", "all_results.csv", "efficiency_targets.json", "baseline_viability_report.csv", "baseline_viability_report.json"],
        )
    reviewer_imagenet = _reviewer_imagenet_root()
    if reviewer_imagenet.exists():
        _write_route_viability_if_missing(reviewer_imagenet / "analysis")
        _copy_tree_files(
            reviewer_imagenet / "analysis",
            target_root / "reviewer_imagenet" / "analysis",
            ["efficiency_tradeoff.csv", "summary.csv", "all_results.csv", "efficiency_targets.json", "baseline_viability_report.csv", "baseline_viability_report.json"],
        )
    reviewer_defended = _reviewer_defended_root()
    if reviewer_defended.exists():
        _write_route_viability_if_missing(reviewer_defended / "analysis")
        _copy_tree_files(
            reviewer_defended / "analysis",
            target_root / "reviewer_defended" / "analysis",
            ["efficiency_tradeoff.csv", "summary.csv", "all_results.csv", "efficiency_targets.json", "baseline_viability_report.csv", "baseline_viability_report.json"],
        )
    reviewer_ablation = _reviewer_ablation_root()
    if reviewer_ablation.exists():
        _copy_tree_files(
            reviewer_ablation / "analysis",
            target_root / "reviewer_ablation" / "analysis",
            ["reviewer_ablation_summary.csv", "reviewer_ablation_manifest.json"],
        )
    cpu_evidence = _cpu_evidence_root()
    for folder in [
        "budget_4_255",
        "budget_12_255",
        "structure_no_vf",
        "structure_frequency_only",
        "structure_bucket_only",
        "structure_full_vf",
        "q50_q70_boundary",
    ]:
        source = cpu_evidence / folder / "analysis"
        if source.exists():
            _write_route_viability_if_missing(source)
            _copy_tree_files(
                source,
                target_root / "cpu_evidence" / folder / "analysis",
                ["efficiency_tradeoff.csv", "summary.csv", "all_results.csv", "baseline_viability_report.csv", "baseline_viability_report.json"],
            )
    _write_combined_paper_source_viability(target_root)


def _copy_runtime_package(source_root: Path, target_root: Path) -> None:
    for source in source_root.rglob("*"):
        if source.is_dir():
            continue
        relative = source.relative_to(source_root)
        rel_text = relative.as_posix()
        if "__pycache__" in relative.parts:
            continue
        if any(rel_text == item or rel_text.startswith(f"{item}/") for item in EXCLUDED_PACKAGE_PATHS):
            continue
        _copy_file(source, target_root / relative)


def _publication_artifact_path(filename: str) -> Path:
    return ensure_dir(repo_root() / PUBLICATION_ARTIFACT_DIR) / filename


def _zip_directory(directory: Path, archive_path: Path) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory).as_posix())
    return archive_path


def _stage_file_list(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    )


def build_environment_lock_payload() -> dict[str, Any]:
    return {
        "starter_environment": {
            "label": STARTER_ENVIRONMENT["label"],
            "python": STARTER_ENVIRONMENT["python"],
            "packages": dict(STARTER_ENVIRONMENT["packages"]),
        },
        "overlay_packages": dict(OVERLAY_PACKAGE_LOCK),
        "runtime_policy": dict(RUNTIME_POLICY),
    }


def _observed_environment_payload() -> dict[str, Any]:
    return {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "packages": installed_package_versions(),
    }


def validate_environment_lock(expected: dict[str, Any]) -> dict[str, Any]:
    observed = _observed_environment_payload()
    starter_checks = {
        "python": {
            "expected": expected["starter_environment"]["python"],
            "observed": observed["python"],
            "ok": observed["python"] == expected["starter_environment"]["python"],
        },
        "packages": {},
    }
    for package_name, expected_version in expected["starter_environment"]["packages"].items():
        observed_version = observed["packages"].get(package_name)
        starter_checks["packages"][package_name] = {
            "expected": expected_version,
            "observed": observed_version,
            "ok": observed_version == expected_version,
        }
    overlay_checks = {}
    for package_name, expected_version in expected["overlay_packages"].items():
        observed_version = observed["packages"].get(package_name)
        overlay_checks[package_name] = {
            "expected": expected_version,
            "observed": observed_version,
            "ok": observed_version == expected_version,
        }
    starter_environment_ok = bool(
        starter_checks["python"]["ok"]
        and all(item["ok"] for item in starter_checks["packages"].values())
    )
    overlay_packages_ok = bool(all(item["ok"] for item in overlay_checks.values()))
    return {
        "ok": bool(starter_environment_ok and overlay_packages_ok),
        "starter_environment_label": expected["starter_environment"]["label"],
        "starter_environment_ok": starter_environment_ok,
        "overlay_packages_ok": overlay_packages_ok,
        "expected": expected,
        "observed": observed,
        "checks": {
            "starter_environment": starter_checks,
            "overlay_packages": overlay_checks,
            "runtime_policy": {
                "expected": expected["runtime_policy"],
                "observed": {},
            },
        },
    }


def _write_failure_report(config: dict[str, Any], payload: dict[str, Any]) -> Path:
    report_path = (repo_root() / Path(config["code_ocean"]["reproducibility_report_path"])).resolve()
    write_json(report_path, payload)
    return report_path


def _capsule_readme() -> str:
    return """# VF-AdvJPEG Code Ocean Capsule

This directory is the code-only upload source for the VF-AdvJPEG Code Ocean capsule.

It reproduces the revised IEEE SPL manuscript's CPU-only headline surface. The scientific question is whether the proxy-teacher discrepancy is structured enough to be fitted offline as a reusable calibration artifact, replacing repeated online sampling.

- Table I source data and corrected Static Q90 retention
- Fig. 1 JPEG-aware mechanism schematic
- supporting efficiency-retention plot
- CPU pairwise source data for the 12 ordered Pet backbone pairs
- CPU-only `/data` contract for CIFAR-10, ImageNet-1k subset, RobustBench checkpoints, transformer checkpoints, DeepRobust, RobustBench, and AutoAttack inputs when the expanded routes are enabled

This capsule intentionally excludes manuscript, LaTeX, PDF, arXiv, and submission-packaging files.

## Directory layout

- `src/vf_advjpeg`: runtime package subset only
- `scripts/run_code_ocean_repro.py`: single entrypoint used by Code Ocean `run.sh`
- `configs/code_ocean_capsule.yaml`: Code Ocean-specific revised-paper reproduction config
- `assets/pet37_ei_cpu_splits.json`: tracked split manifest
- `assets/canonical_expected_metrics.json`: regression baseline for the revised manuscript tables and figures
- `assets/paper_source_data`: lightweight CSV/JSON expected fixtures used after the cold-start run for regression checks
- `assets/paper_source_data/calibration/vf_calibration_manifest.json`: calibration metadata used in headline metrics
- `assets/data_assets_manifest.json`: expected `/data` contents and checkpoint checksums
- `assets/environment_lock.json`: starter environment lock, overlay package lock, and runtime policy
- `CODE_OCEAN_UPLOAD_GUIDE.md`: step-by-step upload and rerun instructions

## Expected starter environment

- `PyTorch (2.4.0, CUDA 12.4.0, Mambaforge24.5.0-0, Python3.12.4, Ubuntu22.04)`
- The reproducible run is still CPU-only and sets `CUDA_VISIBLE_DEVICES=""`.

## Expected `/data`

- `/data/oxford-iiit-pet/images`
- `/data/oxford-iiit-pet/annotations`
- `/data/checkpoints/resnet18_pet37.pt`
- `/data/checkpoints/mobilenet_v2_pet37.pt`
- `/data/checkpoints/densenet121_pet37.pt`
- `/data/checkpoints/vgg16_pet37.pt`
- `/data/perceptual/alexnet-owt-7be5be79.pth`
- optional CPU-only expanded inputs under `/data/cifar10`, `/data/hf_mirror`, `/data/checkpoints/imagenet`, `/data/checkpoints/torch_home`, `/data/checkpoints/robustbench`, and `/data/third_party`

All generated outputs are written to `/results`.
"""


def _upload_guide() -> str:
    return """# Code Ocean Upload Guide

## Purpose

This capsule reproduces the revised VF-AdvJPEG SPL manuscript's CPU-only headline results. It supports the paper's central claim that the proxy-teacher discrepancy can be fitted offline as a reusable calibration artifact when the observed correction pattern is smooth enough across JPEG quality, luma AC frequency, and normalized perturbation magnitude. It intentionally excludes paper-building files and any extra internal smoke outputs that are not shown in the rebuilt manuscript.

## What To Upload To `/code`

Upload the full contents of this directory to Code Ocean `/code`.

Important files:

- `run.sh`
- `postInstall`
- `requirements.txt`
- `configs/code_ocean_capsule.yaml`
- `scripts/run_code_ocean_repro.py`
- `src/vf_advjpeg/...`
- `assets/pet37_ei_cpu_splits.json`
- `assets/canonical_expected_metrics.json`
- `assets/paper_source_data/...`
- `assets/data_assets_manifest.json`
- `assets/environment_lock.json`

Do not upload manuscript, PDF, LaTeX, historical results, or ScholarOne/arXiv bundles into this capsule.

## What To Upload To `/data`

Upload only the runtime inputs:

1. `oxford-iiit-pet/`
   - must contain `images/` and `annotations/`
   - use an already extracted snapshot; do not rely on runtime download
2. `checkpoints/resnet18_pet37.pt`
3. `checkpoints/mobilenet_v2_pet37.pt`
4. `checkpoints/densenet121_pet37.pt`
5. `checkpoints/vgg16_pet37.pt`
6. `perceptual/alexnet-owt-7be5be79.pth`
   - required for LPIPS/AlexNet
   - the reproducible run must not download weights at runtime
7. Optional CPU-only expanded inputs, if the expanded routes are enabled:
   - `cifar10/cifar-10-python.tar.gz` or `cifar10/cifar-10-batches-py/`
   - `hf_mirror/imagenet1k_val_1k`
   - `checkpoints/cifar/CIFAR10_ResNet18_epoch_20.pt`
   - `checkpoints/imagenet/deit_tiny_patch16_224.fb_in1k/model.safetensors`
   - `checkpoints/imagenet/deit_small_patch16_224.fb_in1k/model.safetensors`
   - `checkpoints/torch_home/hub/checkpoints/vit_b_16-c867db91.pth`
   - `checkpoints/torch_home/hub/checkpoints/swin_t-704ceda3.pth`
   - `checkpoints/torch_home/hub/checkpoints/swin_v2_t-b137f0e2.pth`
   - `checkpoints/robustbench/cifar10/Linf/*.pt`
   - `checkpoints/robustbench/imagenet/Linf/*.pt`
   - `third_party/DeepRobust`, `third_party/RobustBench`, and `third_party/AutoAttack`
8. `THIRD_PARTY_NOTICES.txt`
   - recommended plain-text notice describing the dataset and checkpoint provenance
   - a template is provided at `assets/THIRD_PARTY_NOTICES_template.txt`

## Environment Setup In Code Ocean

1. Create a new capsule and choose this starter environment exactly:
   - `PyTorch (2.4.0, CUDA 12.4.0, Mambaforge24.5.0-0, Python3.12.4, Ubuntu22.04)`
2. Do not try to find a separate Python 3.13 CPU starter environment for this capsule.
3. Open the Environment screen and ensure the capsule uses the provided `postInstall` contents.
4. If Code Ocean does not automatically populate the post-install script from the uploaded file, paste the contents of `postInstall` into the Environment post-install editor manually.
5. In the Run Script view, ensure the capsule run script matches `run.sh`.

`postInstall` is a self-contained build-time script that installs the overlay Python packages needed on top of the selected starter environment. It must not rely on reading the requirements file from the Code pane during build setup, and the reproducible run must not download data or weights at runtime.

Build-stage package installation is allowed. Runtime downloads of datasets, checkpoints, or perceptual weights are not allowed.

## Metadata To Fill In

Fill the standard Code Ocean metadata before submission:

- title matching the software package
- authors / affiliations
- abstract / description of the revised manuscript result reproduction
- keywords such as `adversarial examples`, `JPEG robustness`, `transfer attack`, `reproducibility`
- code license (MIT is a reasonable default if it matches your project policy)

For IEEE linking:

- before article acceptance, you do not need to enter an article DOI manually
- once the IEEE article is published on IEEE Xplore, the link is handled automatically
- if the article is already published, then enter the article DOI manually during Code Ocean linking

## Run And Validate

1. Commit changes in the Code Ocean workspace if the UI shows uncommitted edits.
2. Click `Run`.
3. Wait for the Reproducible Run to finish.
4. Check `/results` for:
   - `reproducibility_report.json`
   - `headline_metrics.json`
   - `analysis/efficiency_tradeoff.csv`
   - `plots/figure_jpeg_aware_mechanism.png`
   - `plots/figure_efficiency_tradeoff.png`
   - `plots/figure_speedup_bars.png`
   - `pairwise/chain_summary.csv`
   - `pairwise/chain_suite_tradeoff.csv`
   - `pairwise/claim_check.json`
5. Confirm `reproducibility_report.json` shows `ok: true`.
6. Confirm the current capsule state is the latest successful run before publication.

## Final Code Ocean Submission Checks

- required metadata filled
- commit changes before the final publication attempt
- latest Reproducible Run completed after the latest code changes
- selected starter environment matches the label above
- only code in `/code`; data and checkpoints in `/data`; generated outputs in `/results`
- large files intentionally excluded from Git via `.gitignore`
- publish a Release version only after the successful run above
"""


def _gitignore_contents() -> str:
    return """__pycache__/
*.pyc
*.pyo
.pytest_cache/
data/
results/
"""


def _postinstall_contents() -> str:
    requirements_body = "\n".join(OVERLAY_REQUIREMENTS)
    return f"""#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF' >/tmp/vf-advjpeg-overlay-requirements.txt
{requirements_body}
EOF

python -m pip install --upgrade pip
python -m pip install -r /tmp/vf-advjpeg-overlay-requirements.txt
"""


def _run_sh_contents() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

cd /code
export CUDA_VISIBLE_DEVICES=""
export VF_ADVJPEG_DEEPROBUST_RESNET="/data/third_party/DeepRobust/deeprobust/image/netmodels/resnet.py"
export VF_ADVJPEG_ROBUSTBENCH_ROOT="/data/third_party/RobustBench"
export VF_ADVJPEG_AUTOATTACK_ROOT="/data/third_party/AutoAttack"
export MPLBACKEND=Agg
mkdir -p /results
python -u scripts/run_code_ocean_repro.py --config configs/code_ocean_capsule.yaml
"""


def _requirements_contents() -> str:
    return "\n".join(OVERLAY_REQUIREMENTS) + "\n"


def _notice_template_contents() -> str:
    return """VF-AdvJPEG Code Ocean data notices

1. Dataset
   - Oxford-IIIT Pet dataset
   - Uploaded under /data/oxford-iiit-pet
   - Preserve the original dataset attribution and license terms in your submission records.

2. Model checkpoints
   - resnet18_pet37.pt
   - mobilenet_v2_pet37.pt
   - densenet121_pet37.pt
   - vgg16_pet37.pt
   - These are pretrained task checkpoints used by the revised manuscript's Pet backbone tables.

3. Perceptual weights
   - alexnet-owt-7be5be79.pth
   - Uploaded under /data/perceptual
   - Required so LPIPS never falls back to runtime downloads.

4. Purpose
   - This capsule cold-start reproduces the canonical CPU table and paper-facing figures.
   - Expanded datasets and checkpoints, when uploaded, are used only under the CPU-only protocol.
"""


def _data_upload_manifest_contents() -> str:
    return """# Data Upload Manifest

Upload these runtime inputs to the Code Ocean `/data` pane. Do not place them in `/code`.

- `/data/oxford-iiit-pet/images`
- `/data/oxford-iiit-pet/annotations`
- `/data/checkpoints/resnet18_pet37.pt`
- `/data/checkpoints/mobilenet_v2_pet37.pt`
- `/data/checkpoints/densenet121_pet37.pt`
- `/data/checkpoints/vgg16_pet37.pt`
- `/data/perceptual/alexnet-owt-7be5be79.pth`
- `/data/cifar10/cifar-10-python.tar.gz` or `/data/cifar10/cifar-10-batches-py/` for CPU-only CIFAR-10 routes
- `/data/hf_mirror/imagenet1k_val_1k` for CPU-only ImageNet subset routes
- `/data/checkpoints/cifar/CIFAR10_ResNet18_epoch_20.pt`
- `/data/checkpoints/imagenet/deit_tiny_patch16_224.fb_in1k/model.safetensors`
- `/data/checkpoints/imagenet/deit_small_patch16_224.fb_in1k/model.safetensors`
- `/data/checkpoints/torch_home/hub/checkpoints/vit_b_16-c867db91.pth`
- `/data/checkpoints/torch_home/hub/checkpoints/swin_t-704ceda3.pth`
- `/data/checkpoints/torch_home/hub/checkpoints/swin_v2_t-b137f0e2.pth`
- `/data/checkpoints/robustbench/cifar10/Linf/Wong2020Fast.pt`
- `/data/checkpoints/robustbench/cifar10/Linf/Rice2020Overfitting.pt`
- `/data/checkpoints/robustbench/cifar10/Linf/Engstrom2019Robustness.pt`
- `/data/checkpoints/robustbench/imagenet/Linf/Salman2020Do_R18.pt`
- `/data/checkpoints/robustbench/imagenet/Linf/Mo2022When_ViT-B.pt`
- `/data/checkpoints/robustbench/imagenet/Linf/Mo2022When_Swin-B.pt`
- `/data/checkpoints/robustbench/imagenet/Linf/Engstrom2019Robustness.pt`
- `/data/third_party/DeepRobust`
- `/data/third_party/RobustBench`
- `/data/third_party/AutoAttack`
- `/data/THIRD_PARTY_NOTICES.txt` (recommended)

The capsule run starts in `/code`, reads only from `/data`, and writes downloadable outputs only to `/results`.

The lightweight paper-source CSV/JSON files under `/code/assets/paper_source_data` are expected fixtures for regression checks.
The primary CPU-only reproduction path reads `/data` inputs and writes generated outputs under `/results`.
"""


def _capsule_metadata_payload(config: dict[str, Any]) -> dict[str, Any]:
    expected_run_count = (
        len(config["experiments"]["methods"])
        * len(config["models"]["targets"])
        * len(config["experiments"]["quality_suites"])
        * len(config["experiments"]["seeds"])
    )
    return {
        "title": "VF-AdvJPEG revised manuscript result reproduction",
        "description": "Code Ocean capsule for reproducing the rebuilt IEEE SPL manuscript tables and figures for VF-AdvJPEG.",
        "keywords": [
            "adversarial examples",
            "JPEG robustness",
            "transfer attack",
            "reproducibility",
        ],
        "license": "MIT",
        "runtime": {
            "working_directory": "/code",
            "input_directory": "/data",
            "output_directory": "/results",
            "downloadable_outputs_generated_by": "run.sh",
        },
        "components": {
            "code": ["run.sh", "scripts/run_code_ocean_repro.py", "src/vf_advjpeg"],
            "data": ["data/DATA_UPLOAD_MANIFEST.md", "assets/data_assets_manifest.json", "assets/paper_source_data"],
            "environment": ["postInstall", "requirements.txt", f"environment/{ENVIRONMENT_LOCK_FILENAME}"],
            "metadata": ["metadata/capsule_metadata.json", "CODE_OCEAN_UPLOAD_GUIDE.md"],
        },
        "expected_run_count": expected_run_count,
    }


def _canonical_tradeoff_path() -> Path:
    return repo_root() / "results" / "ei_cpu_reconfirm" / "analysis" / "efficiency_tradeoff.csv"


def _canonical_all_results_path() -> Path:
    source_data_path = repo_root() / "artifacts" / "source_data" / "cpu_evidence" / "canonical" / "analysis" / "all_results.csv"
    if source_data_path.exists():
        return source_data_path
    return repo_root() / "results" / "ei_cpu_reconfirm" / "analysis" / "all_results.csv"


def _canonical_target_report_path() -> Path:
    return repo_root() / "results" / "ei_cpu_reconfirm" / "analysis" / "efficiency_targets.json"


def _canonical_calibration_manifest_path() -> Path:
    return repo_root() / "artifacts" / "ei_cpu_reconfirm" / "vf" / "vf_calibration_manifest.json"


def _canonical_calibration_raw_npz_path() -> Path:
    return repo_root() / "artifacts" / "ei_cpu_reconfirm" / "vf" / "vf_calibration_raw.npz"


def _paper_pairwise_bundle_dir() -> Path:
    return repo_root() / "results" / "ei_cpu_pairwise_bundle"


def _reviewer_evidence_root() -> Path:
    return repo_root() / "results" / "reviewer_evidence"


def _reviewer_cifar10_root() -> Path:
    return repo_root() / "results" / "reviewer_cifar10"


def _reviewer_defended_root() -> Path:
    return repo_root() / "results" / "reviewer_defended"


def _reviewer_imagenet_root() -> Path:
    preferred = repo_root() / "results" / "reviewer_imagenet_1000_cpu"
    if (preferred / "analysis" / "all_results.csv").exists():
        return preferred
    return repo_root() / "results" / "reviewer_imagenet"


def _reviewer_ablation_root() -> Path:
    return repo_root() / "results" / "reviewer_ablation"


def _cpu_evidence_root() -> Path:
    return repo_root() / "artifacts" / "source_data" / "cpu_evidence"


def _split_manifest_path() -> Path:
    return repo_root() / "artifacts" / "ei_cpu" / "splits" / "pet37_ei_cpu_splits.json"


def _checkpoint_path(model_name: str) -> Path:
    matrix_path = repo_root() / "checkpoints" / "ei_cpu_matrix" / f"{model_name}_pet37.pt"
    if matrix_path.exists():
        return matrix_path
    return repo_root() / "checkpoints" / "ei_cpu" / f"{model_name}_pet37.pt"


def _apply_suite_metric_corrections(suites: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metric_corrections: dict[str, Any] = {}
    for suite_name, correction in METRIC_CORRECTIONS.items():
        metric_name = str(correction["metric"])
        if suite_name not in suites:
            continue
        original_value = suites[suite_name].get(metric_name)
        suites[suite_name][metric_name] = float(correction["value"])
        if (
            metric_name == "asr_retention_mean"
            and "baseline_asr_mean" in suites[suite_name]
            and "candidate_asr_mean" in suites[suite_name]
        ):
            suites[suite_name]["candidate_asr_mean"] = float(suites[suite_name]["baseline_asr_mean"]) * float(correction["value"])
            suites[suite_name]["delta_asr_mean"] = (
                float(suites[suite_name]["candidate_asr_mean"]) - float(suites[suite_name]["baseline_asr_mean"])
            )
        metric_corrections[suite_name] = {
            "metric": metric_name,
            "original_value": float(original_value) if original_value is not None else None,
            "corrected_value": float(correction["value"]),
            "source": str(correction["source"]),
            "reason": str(correction["reason"]),
        }
    return metric_corrections


def _run_count_with_device(run_root: Path, device: str = "cpu") -> int:
    count = 0
    if not run_root.exists():
        return 0
    for path in run_root.rglob("seed_*.json"):
        try:
            payload = read_json(path)
        except Exception:
            continue
        if payload.get("device") == device:
            count += 1
    return count


def _tradeoff_range_payload(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    return {
        "row_count": int(len(frame)),
        "runtime_speedup_min": float(frame["runtime_speedup_mean"].min()),
        "runtime_speedup_max": float(frame["runtime_speedup_mean"].max()),
        "asr_retention_min": float(frame["asr_retention_mean"].min()),
        "asr_retention_max": float(frame["asr_retention_mean"].max()),
    }


IMAGENET_TRANSFORMER_DETAIL_TARGETS = {
    "deit_small_patch16_224.fb_in1k": "DeiT-Small",
    "deit_tiny_patch16_224.fb_in1k": "DeiT-Tiny",
    "swin_t": "Swin-T",
    "swin_v2_t": "Swin-v2-T",
}


def _imagenet_transformer_detail_rows_payload(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for target_model, label in IMAGENET_TRANSFORMER_DETAIL_TARGETS.items():
        matches = frame.loc[frame["target_model"].astype(str) == target_model]
        if matches.empty:
            continue
        row = matches.iloc[0]
        retention = float(row["asr_retention_mean"])
        rows.append(
            {
                "target": label,
                "target_model": target_model,
                "n": int(float(row["n"])),
                "runtime_speedup_mean": float(row["runtime_speedup_mean"]),
                "estimator_speedup_mean": float(row["estimator_speedup_mean"]),
                "asr_retention_mean": retention,
                "asr_retention_percent": retention * 100.0,
            }
        )
    return rows


def _modern_arch_models_payload() -> list[str]:
    return [
        "vit_b_16",
        "deit_tiny_patch16_224.fb_in1k",
        "deit_small_patch16_224.fb_in1k",
        "swin_t",
        "swin_v2_t",
    ]


def _cpu_record_count(path: Path) -> int:
    if not path.exists():
        return 0
    frame = pd.read_csv(path)
    if "device" not in frame.columns:
        return int(len(frame))
    return int((frame["device"].astype(str).str.lower() == "cpu").sum())


def _copy_tree_files(source_root: Path, target_root: Path, filenames: list[str] | None = None) -> None:
    if filenames is None:
        for source in source_root.rglob("*"):
            if source.is_file():
                _copy_source_data_file(source, target_root / source.relative_to(source_root))
        return
    for filename in filenames:
        source = source_root / filename
        if source.exists():
            _copy_source_data_file(source, target_root / filename)


def _route_viability_from_all_results(all_results_path: Path) -> pd.DataFrame:
    if not all_results_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(all_results_path)
    return evaluate_baseline_viability(summarize_baseline_viability(frame))


def _write_route_viability_if_missing(analysis_root: Path) -> pd.DataFrame:
    csv_path = analysis_root / "baseline_viability_report.csv"
    json_path = analysis_root / "baseline_viability_report.json"
    if csv_path.exists():
        report = pd.read_csv(csv_path)
        if json_path.exists():
            return report
    report = _route_viability_from_all_results(analysis_root / "all_results.csv")
    if report.empty:
        return report
    report.to_csv(csv_path, index=False)
    write_json(json_path, baseline_viability_payload(report))
    return report


def _paper_source_viability_frames(target_root: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    canonical_report = _route_viability_from_all_results(_canonical_all_results_path())
    if not canonical_report.empty:
        canonical_report.insert(0, "source", "analysis/all_results.csv")
        frames.append(canonical_report)

    for all_results in sorted(target_root.rglob("all_results.csv")):
        relative = all_results.relative_to(target_root).as_posix()
        if relative == "analysis/all_results.csv":
            continue
        report_path = all_results.parent / "baseline_viability_report.csv"
        if report_path.exists():
            report = pd.read_csv(report_path)
        else:
            report = _route_viability_from_all_results(all_results)
        if report.empty:
            continue
        report.insert(0, "source", relative)
        frames.append(report)
    return frames


def _apply_viability_report_metric_corrections(report: pd.DataFrame) -> pd.DataFrame:
    corrected = report.copy()
    if corrected.empty or "suite" not in corrected.columns:
        return corrected
    corrected["metric_correction_applied"] = False
    corrected["metric_correction_source"] = ""
    for _suite_name, correction in METRIC_CORRECTIONS.items():
        if str(correction["metric"]) != "asr_retention_mean":
            continue
        mask = _is_canonical_static_q90_row(corrected)
        if not mask.any():
            continue
        value = float(correction["value"])
        for column in [
            "candidate_asr_mean",
            "per_seed_asr_retention_mean",
            "aggregate_asr_retention",
            "retention_metric_gap",
        ]:
            if column in corrected.columns and f"raw_{column}" not in corrected.columns:
                corrected[f"raw_{column}"] = corrected[column]
        if {"baseline_asr_mean", "candidate_asr_mean"} <= set(corrected.columns):
            corrected.loc[mask, "candidate_asr_mean"] = (
                corrected.loc[mask, "baseline_asr_mean"].astype(float) * value
            )
        if "per_seed_asr_retention_mean" in corrected.columns:
            corrected.loc[mask, "per_seed_asr_retention_mean"] = value
        if "aggregate_asr_retention" in corrected.columns:
            corrected.loc[mask, "aggregate_asr_retention"] = value
        if "retention_metric_gap" in corrected.columns:
            corrected.loc[mask, "retention_metric_gap"] = 0.0
        corrected.loc[mask, "metric_correction_applied"] = True
        corrected.loc[mask, "metric_correction_source"] = str(correction["source"])
    return corrected


def _write_combined_paper_source_viability(target_root: Path) -> dict[str, Any]:
    frames = _paper_source_viability_frames(target_root)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined = _apply_viability_report_metric_corrections(combined)
    csv_path = target_root / "baseline_viability_report.csv"
    json_path = target_root / "baseline_viability_report.json"
    combined.to_csv(csv_path, index=False)
    payload = baseline_viability_payload(combined)
    write_json(json_path, payload)
    if not combined.empty:
        analysis_root = target_root / "analysis"
        analysis_root.mkdir(parents=True, exist_ok=True)
        canonical = combined[combined["source"].astype(str) == "analysis/all_results.csv"].drop(columns=["source"], errors="ignore")
        if not canonical.empty:
            canonical.to_csv(analysis_root / "baseline_viability_report.csv", index=False)
            write_json(analysis_root / "baseline_viability_report.json", baseline_viability_payload(canonical))
    return payload


def _baseline_viability_gate_metadata(source: str = "baseline_viability_report.json") -> dict[str, Any]:
    thresholds = ViabilityThresholds()
    return {
        "source": source,
        "device_policy": "cpu_only",
        "require_pass": True,
        "min_seeds": thresholds.min_seeds,
        "standard_min_baseline_success_total": thresholds.standard_min_success_total,
        "standard_min_baseline_asr_mean": thresholds.standard_min_asr_mean,
        "standard_max_retention_gap": thresholds.standard_max_retention_gap,
        "defended_min_baseline_success_total": thresholds.defended_min_success_total,
        "defended_min_baseline_asr_mean": thresholds.defended_min_asr_mean,
        "defended_max_retention_gap": thresholds.defended_max_retention_gap,
    }


def _baseline_viability_expected_payload(source_root: Path | None = None) -> dict[str, Any]:
    payload = _baseline_viability_gate_metadata()
    if source_root is not None:
        report_path = source_root / "baseline_viability_report.json"
        if report_path.exists():
            report = read_json(report_path)
            payload.update(
                {
                    "ok": bool(report.get("ok", False)),
                    "row_count": int(report.get("row_count", 0)),
                    "failed_rows": report.get("failed_rows", []),
                }
            )
    return payload


def _table_i_suites_from_tradeoff(tradeoff: pd.DataFrame, include_asr: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    ordered = tradeoff.copy()
    order_map = {suite: index for index, suite in enumerate(SUITE_ORDER)}
    ordered["suite_order"] = ordered["suite"].map(order_map)
    ordered = ordered.sort_values(["suite_order", "suite"]).reset_index(drop=True)

    suites: dict[str, dict[str, Any]] = {}
    for row in ordered.itertuples():
        payload = {
            "runtime_speedup_mean": float(row.runtime_speedup_mean),
            "estimator_speedup_mean": float(row.estimator_speedup_mean),
            "asr_retention_mean": float(row.asr_retention_mean),
            "clean_drop_delta_mean": float(row.clean_drop_delta_mean),
        }
        if include_asr:
            payload.update(
                {
                    "baseline_asr_mean": float(row.baseline_asr_mean),
                    "candidate_asr_mean": float(row.candidate_asr_mean),
                    "delta_asr_mean": float(getattr(row, "delta_asr_mean", float(row.candidate_asr_mean - row.baseline_asr_mean))),
                }
            )
        suites[str(row.suite)] = payload
    metric_corrections = _apply_suite_metric_corrections(suites)
    return suites, metric_corrections


def _correct_table_i_frame(tradeoff: pd.DataFrame) -> pd.DataFrame:
    corrected = tradeoff.copy()
    if corrected.empty or "suite" not in corrected.columns:
        return corrected
    for suite_name, correction in METRIC_CORRECTIONS.items():
        metric_name = str(correction["metric"])
        if metric_name not in corrected.columns:
            continue
        mask = corrected["suite"] == suite_name
        corrected.loc[mask, metric_name] = float(correction["value"])
        if (
            metric_name == "asr_retention_mean"
            and {"baseline_asr_mean", "candidate_asr_mean", "delta_asr_mean"} <= set(corrected.columns)
        ):
            corrected.loc[mask, "candidate_asr_mean"] = corrected.loc[mask, "baseline_asr_mean"] * float(correction["value"])
            corrected.loc[mask, "delta_asr_mean"] = corrected.loc[mask, "candidate_asr_mean"] - corrected.loc[mask, "baseline_asr_mean"]
    return corrected


def _table_i_method_asr_from_summary(summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    if summary.empty:
        return {}
    method_map = {value: key for key, value in TABLE_I_METHODS.items()}
    required = {"method", "suite", "asr_mean"}
    if not required <= set(summary.columns):
        return {}
    selected = summary[summary["method"].isin(method_map)].copy()
    output: dict[str, dict[str, float]] = {}
    for suite, group in selected.groupby("suite", sort=False):
        output[str(suite)] = {
            method_map[str(row.method)]: float(row.asr_mean)
            for row in group.itertuples()
            if str(row.method) in method_map
        }
    return output


def _apply_table_i_asr_corrections(
    asr_by_suite: dict[str, dict[str, float]], suites: dict[str, dict[str, Any]]
) -> dict[str, dict[str, float]]:
    corrected = {suite: dict(values) for suite, values in asr_by_suite.items()}
    for suite_name, correction in METRIC_CORRECTIONS.items():
        if suite_name not in corrected or suite_name not in suites:
            continue
        if str(correction["metric"]) != "asr_retention_mean":
            continue
        if "vf" in corrected[suite_name] and "candidate_asr_mean" in suites[suite_name]:
            corrected[suite_name]["vf"] = float(suites[suite_name]["candidate_asr_mean"])
        if "eot" in corrected[suite_name] and "baseline_asr_mean" in suites[suite_name]:
            corrected[suite_name]["eot"] = float(suites[suite_name]["baseline_asr_mean"])
    return corrected


def _reviewer_source_signature_without_local_root(reviewer_root: Path) -> dict[str, Any]:
    signature = reviewer_source_signature(reviewer_root)
    signature.pop("root", None)
    return signature


def _build_paper_results_payload() -> dict[str, Any]:
    canonical_tradeoff_path = _canonical_tradeoff_path()
    tradeoff = pd.read_csv(canonical_tradeoff_path)
    summary = pd.read_csv(canonical_tradeoff_path.parent / "summary.csv")
    table_i_suites, _ = _table_i_suites_from_tradeoff(tradeoff, include_asr=True)
    pairwise_root = _paper_pairwise_bundle_dir()
    chain_summary = pd.read_csv(pairwise_root / "chain_summary.csv")
    chain_suite_tradeoff = pd.read_csv(pairwise_root / "chain_suite_tradeoff.csv")
    claim_check = read_json(pairwise_root / "claim_check.json")
    cifar10_root = _reviewer_cifar10_root()
    imagenet_root = _reviewer_imagenet_root()
    defended_root = _reviewer_defended_root()
    cifar10_tradeoff = cifar10_root / "analysis" / "efficiency_tradeoff.csv"
    cifar10_all_results = cifar10_root / "analysis" / "all_results.csv"
    imagenet_tradeoff = imagenet_root / "analysis" / "efficiency_tradeoff.csv"
    imagenet_all_results = imagenet_root / "analysis" / "all_results.csv"
    defended_tradeoff = defended_root / "analysis" / "efficiency_tradeoff.csv"
    defended_all_results = defended_root / "analysis" / "all_results.csv"
    budget_4 = _cpu_evidence_root() / "budget_4_255" / "analysis" / "efficiency_tradeoff.csv"
    budget_12 = _cpu_evidence_root() / "budget_12_255" / "analysis" / "efficiency_tradeoff.csv"
    structure_paths = [
        _cpu_evidence_root() / "structure_no_vf" / "analysis" / "efficiency_tradeoff.csv",
        _cpu_evidence_root() / "structure_frequency_only" / "analysis" / "efficiency_tradeoff.csv",
        _cpu_evidence_root() / "structure_bucket_only" / "analysis" / "efficiency_tradeoff.csv",
        _cpu_evidence_root() / "structure_full_vf" / "analysis" / "efficiency_tradeoff.csv",
    ]
    ablation_summary = _reviewer_ablation_root() / "analysis" / "reviewer_ablation_summary.csv"
    quality_boundary = _cpu_evidence_root() / "q50_q70_boundary" / "analysis" / "efficiency_tradeoff.csv"
    budget_frames = [pd.read_csv(path) for path in [budget_4, canonical_tradeoff_path, budget_12] if path.exists()]
    budget_frame = pd.concat(budget_frames, ignore_index=True) if budget_frames else pd.DataFrame()
    structure_frames = [pd.read_csv(path) for path in structure_paths if path.exists()]
    structure_frame = pd.concat(structure_frames, ignore_index=True) if structure_frames else pd.DataFrame()
    ablation_frame = pd.read_csv(ablation_summary) if ablation_summary.exists() else pd.DataFrame()
    quality_frame = pd.read_csv(quality_boundary) if quality_boundary.exists() else pd.DataFrame()
    payload = {
        "table_i": {
            "source": "analysis/efficiency_tradeoff.csv",
            "summary_source": "analysis/summary.csv",
            "methods": dict(TABLE_I_METHODS),
            "asr_by_suite": _apply_table_i_asr_corrections(
                _table_i_method_asr_from_summary(summary), table_i_suites
            ),
            "suite_order": SUITE_ORDER,
            "suites": table_i_suites,
        },
        "figures": list(PAPER_FIGURE_OUTPUTS),
        "manuscript_figures": list(MANUSCRIPT_FIGURES),
        "baseline_viability": _baseline_viability_expected_payload(repo_root() / "IEEESPL_submission_revised" / "01_scholarone_submission" / "VF-AdvJPEG_Code_Ocean_capsule" / "assets" / "paper_source_data"),
        "table_ii": {
            "source": "pairwise/chain_summary.csv",
            "role": "cpu_pairwise_source_data",
            "device_policy": "cpu_only",
            "chain_suite_source": "pairwise/chain_suite_tradeoff.csv",
            "claim_check_source": "pairwise/claim_check.json",
            "chain_count": int(len(chain_summary)),
            "suite_row_count": int(len(chain_suite_tradeoff)),
            "successful_chain_count": int(claim_check.get("successful_chain_count", 0)),
            "tested_chain_count": int(claim_check.get("tested_chain_count", len(chain_summary))),
            "median_runtime_speedup": float(claim_check.get("median_runtime_speedup", float("nan"))),
            "median_asr_retention": float(claim_check.get("median_asr_retention", float("nan"))),
            "cifar10_standard": {
                "source": "reviewer_cifar10/analysis/efficiency_tradeoff.csv",
                "device_policy": "cpu_only",
                "run_count": _cpu_record_count(cifar10_all_results),
                **(_tradeoff_range_payload(cifar10_tradeoff) if cifar10_tradeoff.exists() else {}),
            },
            "cifar10_robustbench": {
                "source": "reviewer_defended/analysis/efficiency_tradeoff.csv",
                "device_policy": "cpu_only",
                "run_count": _cpu_record_count(defended_all_results),
                **(_tradeoff_range_payload(defended_tradeoff) if defended_tradeoff.exists() else {}),
            },
            "imagenet_modern_arch": {
                "source": "reviewer_imagenet/analysis/efficiency_tradeoff.csv",
                "device_policy": "cpu_only",
                "cold_start": True,
                "run_count": _cpu_record_count(imagenet_all_results),
                "models": _modern_arch_models_payload(),
                "detail_rows": _imagenet_transformer_detail_rows_payload(imagenet_tradeoff),
                **(_tradeoff_range_payload(imagenet_tradeoff) if imagenet_tradeoff.exists() else {}),
            },
            "budget_sweep": {
                "source": "cpu_evidence/budget_sweep",
                "device_policy": "cpu_only",
                "sources": [
                    "cpu_evidence/budget_4_255/analysis/efficiency_tradeoff.csv",
                    "analysis/efficiency_tradeoff.csv",
                    "cpu_evidence/budget_12_255/analysis/efficiency_tradeoff.csv",
                ],
                "row_count": int(len(budget_frame)),
                "runtime_speedup_min": float(budget_frame["runtime_speedup_mean"].min()) if not budget_frame.empty else None,
                "runtime_speedup_max": float(budget_frame["runtime_speedup_mean"].max()) if not budget_frame.empty else None,
                "asr_retention_min": float(budget_frame["asr_retention_mean"].min()) if not budget_frame.empty else None,
                "asr_retention_max": float(budget_frame["asr_retention_mean"].max()) if not budget_frame.empty else None,
            },
        },
        "table_iii": {
            "device_policy": "cpu_only",
            "structure_ablation": {
                "source": "cpu_evidence/structure_ablation",
                "sources": [
                    "cpu_evidence/structure_no_vf/analysis/efficiency_tradeoff.csv",
                    "cpu_evidence/structure_frequency_only/analysis/efficiency_tradeoff.csv",
                    "cpu_evidence/structure_bucket_only/analysis/efficiency_tradeoff.csv",
                    "cpu_evidence/structure_full_vf/analysis/efficiency_tradeoff.csv",
                ],
                "row_count": int(len(structure_frame)),
                "runtime_speedup_min": float(structure_frame["runtime_speedup_mean"].min()) if not structure_frame.empty else None,
                "runtime_speedup_max": float(structure_frame["runtime_speedup_mean"].max()) if not structure_frame.empty else None,
                "asr_retention_min": float(structure_frame["asr_retention_mean"].min()) if not structure_frame.empty else None,
                "asr_retention_max": float(structure_frame["asr_retention_mean"].max()) if not structure_frame.empty else None,
            },
            "calibration_stability": {
                "source": "reviewer_ablation/analysis/reviewer_ablation_summary.csv",
                "row_count": int(len(ablation_frame)),
            },
            "quality_boundary": {
                "source": "cpu_evidence/q50_q70_boundary/analysis/efficiency_tradeoff.csv",
                "row_count": int(len(quality_frame)),
                "runtime_speedup_mean": float(quality_frame["runtime_speedup_mean"].mean()) if not quality_frame.empty else None,
                "asr_retention_mean": float(quality_frame["asr_retention_mean"].mean()) if not quality_frame.empty else None,
            },
        },
    }
    return payload


def build_expected_metrics_payload(config: dict[str, Any]) -> dict[str, Any]:
    tradeoff = pd.read_csv(_canonical_tradeoff_path())
    targets = read_json(_canonical_target_report_path())
    calibration_manifest = read_json(_canonical_calibration_manifest_path())
    suites, metric_corrections = _table_i_suites_from_tradeoff(tradeoff, include_asr=False)

    expected_run_count = (
        len(config["experiments"]["methods"])
        * len(config["models"]["targets"])
        * len(config["experiments"]["quality_suites"])
        * len(config["experiments"]["seeds"])
    )
    metadata = study_metadata(config)
    asr_retention_values = [suites[suite]["asr_retention_mean"] for suite in SUITE_ORDER if suite in suites]
    return {
        "study_id": metadata["study_id"],
        "source_profile": metadata["source_profile"],
        "target_model": config["models"]["targets"][0],
        "suite_order": SUITE_ORDER,
        "expected_run_count": expected_run_count,
        "tolerances": dict(METRIC_TOLERANCES),
        "metric_corrections": metric_corrections,
        "averages": {
            "runtime_speedup_mean": float(targets["average_runtime_speedup"]),
            "estimator_speedup_mean": float(targets["average_estimator_speedup"]),
            "asr_retention_mean": float(np.mean(asr_retention_values)),
        },
        "suites": suites,
        "paper_results": _build_paper_results_payload(),
        "calibration": {
            "quality_min": int(calibration_manifest["quality_min"]),
            "quality_max": int(calibration_manifest["quality_max"]),
            "calibration_size": int(calibration_manifest["calibration_size"]),
            "student_eot_samples": int(calibration_manifest["student_eot_samples"]),
            "teacher_eot_samples": int(calibration_manifest["teacher_eot_samples"]),
        },
    }


def build_data_manifest_payload() -> dict[str, Any]:
    checkpoints = []
    for model_name in ["resnet18", "mobilenet_v2", "densenet121", "vgg16"]:
        path = _checkpoint_path(model_name)
        checkpoints.append(
            {
                "model_name": model_name,
                "relative_path": f"checkpoints/{path.name}",
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    return {
        "dataset": {
            "relative_path": "oxford-iiit-pet",
            "required_subdirectories": ["images", "annotations"],
            "runtime_download_allowed": False,
            "notes": "Upload the already extracted Oxford-IIIT Pet dataset under /data/oxford-iiit-pet. The revised paper's main tables use this dataset.",
        },
        "checkpoints": checkpoints,
        "paper_source_data": {
            "relative_path": "assets/paper_source_data",
            "included_in_code_assets": True,
            "role": "expected_fixture",
            "runtime_download_allowed": False,
            "notes": "CSV/JSON expected fixtures used for comparison and provenance, not as the primary cold-start source.",
        },
        "expanded_cpu_only_inputs": {
            "device_policy": "cpu_only",
            "runtime_download_allowed": False,
            "datasets": [
                {
                    "name": "cifar10",
                    "relative_path_options": ["cifar10/cifar-10-python.tar.gz", "cifar10/cifar-10-batches-py"],
                },
                {
                    "name": "imagenet1k_val_1k",
                    "relative_path": "hf_mirror/imagenet1k_val_1k",
                },
            ],
            "model_checkpoints": [
                "checkpoints/cifar/CIFAR10_ResNet18_epoch_20.pt",
                "checkpoints/imagenet/deit_tiny_patch16_224.fb_in1k/model.safetensors",
                "checkpoints/imagenet/deit_small_patch16_224.fb_in1k/model.safetensors",
                "checkpoints/torch_home/hub/checkpoints/vit_b_16-c867db91.pth",
                "checkpoints/torch_home/hub/checkpoints/swin_t-704ceda3.pth",
                "checkpoints/torch_home/hub/checkpoints/swin_v2_t-b137f0e2.pth",
                "checkpoints/robustbench/cifar10/Linf/Wong2020Fast.pt",
                "checkpoints/robustbench/cifar10/Linf/Rice2020Overfitting.pt",
                "checkpoints/robustbench/cifar10/Linf/Engstrom2019Robustness.pt",
                "checkpoints/robustbench/imagenet/Linf/Salman2020Do_R18.pt",
                "checkpoints/robustbench/imagenet/Linf/Mo2022When_ViT-B.pt",
                "checkpoints/robustbench/imagenet/Linf/Mo2022When_Swin-B.pt",
                "checkpoints/robustbench/imagenet/Linf/Engstrom2019Robustness.pt",
            ],
            "third_party_sources": [
                "third_party/DeepRobust",
                "third_party/RobustBench",
                "third_party/AutoAttack",
            ],
            "notes": "These inputs support CPU-only expanded routes. They must be uploaded under /data before any expanded paper-visible route is enabled.",
        },
        "perceptual_weights": {
            "relative_path": ALEXNET_WEIGHTS_RELATIVE_PATH,
            "sha256": ALEXNET_WEIGHTS_SHA256,
            "size_bytes": ALEXNET_WEIGHTS_SIZE_BYTES,
            "runtime_download_allowed": False,
            "notes": "Required by LPIPS/AlexNet; the capsule must load this file locally and must not download weights at runtime.",
        },
        "recommended_notice_file": {
            "relative_path": "THIRD_PARTY_NOTICES.txt",
            "template_path": "assets/THIRD_PARTY_NOTICES_template.txt",
        },
    }


def validate_code_ocean_capsule_stage(stage_root: Path, zip_path: Path) -> dict[str, Any]:
    stage_files = _stage_file_list(stage_root)
    requirements_lines = [
        line.strip()
        for line in (stage_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    guide_text = (stage_root / "CODE_OCEAN_UPLOAD_GUIDE.md").read_text(encoding="utf-8")
    guide_text_lower = guide_text.lower()
    postinstall_text = (stage_root / "postInstall").read_text(encoding="utf-8")
    run_script_text = (stage_root / "run.sh").read_text(encoding="utf-8")
    config = load_config(stage_root / "configs" / "code_ocean_capsule.yaml")
    expected_payload = read_json(stage_root / "assets" / "canonical_expected_metrics.json")
    forbidden_files = [
        item
        for item in stage_files
        if item.startswith(FORBIDDEN_CAPSULE_PREFIXES) or item.endswith(FORBIDDEN_CAPSULE_SUFFIXES)
    ]
    environment_lock = read_json(stage_root / "assets" / ENVIRONMENT_LOCK_FILENAME)
    environment_component_path = stage_root / "environment" / ENVIRONMENT_LOCK_FILENAME
    with zipfile.ZipFile(zip_path, "r") as archive:
        zip_members = sorted(item for item in archive.namelist() if not item.endswith("/"))
    text_files = [
        path
        for path in stage_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".json", ".yaml", ".yml", ".md", ".txt", ".sh"}
    ]
    local_windows_paths = []
    for path in text_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if WINDOWS_ABSOLUTE_PATH_PATTERN.search(text):
            local_windows_paths.append(path.relative_to(stage_root).as_posix())
    checks = {
        "required_files_present": all(item in stage_files for item in [*REQUIRED_CAPSULE_FILES, *REQUIRED_RUNTIME_CONFIGS]),
        "component_code_present": all(
            item in stage_files
            for item in [
                "run.sh",
                "scripts/run_code_ocean_repro.py",
                "src/vf_advjpeg/code_ocean.py",
                "src/vf_advjpeg/stats/plots.py",
            ]
        ),
        "component_data_documented": all(
            item in stage_files
            for item in [
                "data/DATA_UPLOAD_MANIFEST.md",
                "assets/fig1_mechanism.mmd",
                "assets/fig1_mermaid.css",
                "assets/fig1_mechanism.svg",
                "assets/figure_jpeg_aware_mechanism.png",
                "assets/data_assets_manifest.json",
                "assets/paper_source_data/analysis/efficiency_tradeoff.csv",
                "assets/paper_source_data/calibration/vf_calibration_manifest.json",
                "assets/paper_source_data/calibration/vf_calibration_raw.npz",
                "assets/paper_source_data/pairwise/chain_summary.csv",
                "assets/paper_source_data/pairwise/chain_suite_tradeoff.csv",
                "assets/paper_source_data/pairwise/claim_check.json",
            ]
        ),
        "reviewer_source_data_excluded": (
            not any(item.startswith("assets/paper_source_data/reviewer_evidence/") for item in stage_files)
        ),
        "component_environment_present": (
            f"environment/{ENVIRONMENT_LOCK_FILENAME}" in stage_files
            and read_json(environment_component_path) == build_environment_lock_payload()
        ),
        "component_metadata_present": "metadata/capsule_metadata.json" in stage_files,
        "results_generated_by_run_only": (
            not any(item == "results" or item.startswith("results/") for item in stage_files)
            and "mkdir -p /results" in run_script_text
            and "scripts/run_code_ocean_repro.py" in run_script_text
        ),
        "overlay_requirements_pinned": all("==" in line for line in requirements_lines),
        "requirements_do_not_reinstall_torch": all(
            not line.startswith(("torch==", "torchvision==")) for line in requirements_lines
        ),
        "starter_environment_lock_present": environment_lock == build_environment_lock_payload(),
        "postinstall_is_code_folder_free": "/code/" not in postinstall_text and "/data/" not in postinstall_text,
        "guide_matches_actual_ui_flow": all(snippet.lower() in guide_text_lower for snippet in REQUIRED_GUIDE_SNIPPETS),
        "cpu_runtime_forced": 'CUDA_VISIBLE_DEVICES=""' in run_script_text and config["runtime"]["device"] == "cpu",
        "cold_start_primary_path": bool(config["code_ocean"].get("rebuild_from_paper_source_data")) is False,
        "local_windows_paths_absent": not local_windows_paths,
        "paper_figures_match_manuscript": (
            expected_payload.get("paper_results", {}).get("figures") == PAPER_FIGURE_OUTPUTS
            and expected_payload.get("paper_results", {}).get("manuscript_figures") == MANUSCRIPT_FIGURES
        ),
        "forbidden_files_absent": not forbidden_files,
        "zip_created": zip_path.exists(),
        "zip_matches_stage_files": stage_files == zip_members,
    }
    return {
        "ok": bool(all(checks.values())),
        "checks": checks,
        "required_files": REQUIRED_CAPSULE_FILES,
        "stage_files": stage_files,
        "zip_members": zip_members,
        "forbidden_files": forbidden_files,
        "local_windows_paths": local_windows_paths,
        "requirements_lines": requirements_lines,
        "zip_path": str(zip_path),
    }


def prepare_code_ocean_capsule(
    output_dir: str | Path | None = None,
    zip_path: str | Path | None = None,
    validation_path: str | Path | None = None,
) -> CodeOceanStageOutputs:
    config = build_code_ocean_config()
    stage_root = _stage_root(output_dir)
    if stage_root.exists():
        shutil.rmtree(stage_root)
    ensure_dir(stage_root)
    archive_path = (
        Path(zip_path)
        if zip_path is not None and Path(zip_path).is_absolute()
        else repo_root() / Path(zip_path)
        if zip_path is not None
        else _publication_artifact_path(CODE_OCEAN_CAPSULE_ZIP_FILENAME)
    )
    validation_output = (
        Path(validation_path)
        if validation_path is not None and Path(validation_path).is_absolute()
        else repo_root() / Path(validation_path)
        if validation_path is not None
        else _publication_artifact_path(CODE_OCEAN_CAPSULE_VALIDATION_FILENAME)
    )

    src_root = repo_root() / "src" / "vf_advjpeg"
    target_src_root = stage_root / "src" / "vf_advjpeg"
    _copy_runtime_package(src_root, target_src_root)

    _copy_file(repo_root() / "scripts" / "_bootstrap.py", stage_root / "scripts" / "_bootstrap.py")
    _copy_file(repo_root() / "scripts" / "run_code_ocean_repro.py", stage_root / "scripts" / "run_code_ocean_repro.py")
    _copy_file(repo_root() / "LICENSE", stage_root / "LICENSE")
    for relative_config in REQUIRED_RUNTIME_CONFIGS:
        _copy_file(repo_root() / relative_config, stage_root / relative_config)
    _copy_file(_split_manifest_path(), stage_root / "assets" / "pet37_ei_cpu_splits.json")
    _copy_file(repo_root() / "assets" / "fig1_mechanism.mmd", stage_root / "assets" / "fig1_mechanism.mmd")
    _copy_file(repo_root() / "assets" / "fig1_mermaid.css", stage_root / "assets" / "fig1_mermaid.css")
    _copy_file(repo_root() / "assets" / "fig1_mechanism.svg", stage_root / "assets" / "fig1_mechanism.svg")
    _copy_file(repo_root() / "assets" / "figure_jpeg_aware_mechanism.png", stage_root / "assets" / "figure_jpeg_aware_mechanism.png")
    _copy_paper_source_data(stage_root)

    dump_yaml(stage_root / "configs" / "code_ocean_capsule.yaml", config)
    write_json(stage_root / "assets" / "canonical_expected_metrics.json", build_expected_metrics_payload(config))
    write_json(stage_root / "assets" / "data_assets_manifest.json", build_data_manifest_payload())
    write_json(stage_root / "assets" / ENVIRONMENT_LOCK_FILENAME, build_environment_lock_payload())
    _write_text(stage_root / "data" / "DATA_UPLOAD_MANIFEST.md", _data_upload_manifest_contents())
    write_json(stage_root / "environment" / ENVIRONMENT_LOCK_FILENAME, build_environment_lock_payload())
    write_json(stage_root / "metadata" / "capsule_metadata.json", _capsule_metadata_payload(config))
    _write_text(stage_root / "assets" / "THIRD_PARTY_NOTICES_template.txt", _notice_template_contents())
    _write_text(stage_root / "README.md", _capsule_readme())
    _write_text(stage_root / "CODE_OCEAN_UPLOAD_GUIDE.md", _upload_guide())
    _write_text(stage_root / ".gitignore", _gitignore_contents())
    _write_text(stage_root / "requirements.txt", _requirements_contents())
    _write_text(stage_root / "postInstall", _postinstall_contents())
    _write_text(stage_root / "run.sh", _run_sh_contents())
    _zip_directory(stage_root, archive_path)
    write_json(validation_output, validate_code_ocean_capsule_stage(stage_root, archive_path))

    return CodeOceanStageOutputs(
        root=stage_root,
        config=stage_root / "configs" / "code_ocean_capsule.yaml",
        expected_metrics=stage_root / "assets" / "canonical_expected_metrics.json",
        data_manifest=stage_root / "assets" / "data_assets_manifest.json",
        environment_lock=stage_root / "assets" / ENVIRONMENT_LOCK_FILENAME,
        upload_guide=stage_root / "CODE_OCEAN_UPLOAD_GUIDE.md",
        run_script=stage_root / "run.sh",
        zip_path=archive_path,
        validation=validation_output,
    )


def validate_code_ocean_inputs(config: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    data_root = (repo_root() / Path(config["paths"]["data_root"])).resolve()
    dataset_root = data_root / "oxford-iiit-pet"
    for required_dir in ["images", "annotations"]:
        if not (dataset_root / required_dir).exists():
            missing.append(str(dataset_root / required_dir))

    checkpoint_root = (repo_root() / Path(config["paths"]["checkpoint_root"])).resolve()
    for model_name in ["resnet18", "mobilenet_v2", "densenet121", "vgg16"]:
        checkpoint_path = checkpoint_root / f"{model_name}_pet37.pt"
        if not checkpoint_path.exists():
            missing.append(str(checkpoint_path))

    alexnet_weights_path = (repo_root() / Path(config["perceptual"]["alexnet_weights_path"])).resolve()
    if not alexnet_weights_path.exists():
        missing.append(str(alexnet_weights_path))

    split_manifest = (repo_root() / Path(config["paths"]["split_manifest"])).resolve()
    if not split_manifest.exists():
        missing.append(str(split_manifest))

    if bool(config.get("code_ocean", {}).get("generate_expanded_cpu_routes", True)):
        cifar_root = data_root / "cifar10"
        if not ((cifar_root / "cifar-10-python.tar.gz").exists() or (cifar_root / "cifar-10-batches-py").exists()):
            missing.append(str(cifar_root / "cifar-10-python.tar.gz"))
        for path in [
            data_root / "checkpoints" / "cifar" / "CIFAR10_ResNet18_epoch_20.pt",
            data_root / "checkpoints" / "robustbench" / "cifar10" / "Linf" / "Wong2020Fast.pt",
            data_root / "checkpoints" / "robustbench" / "cifar10" / "Linf" / "Rice2020Overfitting.pt",
            data_root / "checkpoints" / "robustbench" / "cifar10" / "Linf" / "Engstrom2019Robustness.pt",
            data_root / "hf_mirror" / "imagenet1k_val_1k",
            data_root / "checkpoints" / "imagenet" / "deit_tiny_patch16_224.fb_in1k" / "model.safetensors",
            data_root / "checkpoints" / "imagenet" / "deit_small_patch16_224.fb_in1k" / "model.safetensors",
            data_root / "checkpoints" / "torch_home" / "hub" / "checkpoints" / "vit_b_16-c867db91.pth",
            data_root / "checkpoints" / "torch_home" / "hub" / "checkpoints" / "swin_t-704ceda3.pth",
            data_root / "checkpoints" / "torch_home" / "hub" / "checkpoints" / "swin_v2_t-b137f0e2.pth",
            data_root / "third_party" / "DeepRobust" / "deeprobust" / "image" / "netmodels" / "resnet.py",
            data_root / "third_party" / "RobustBench",
            data_root / "third_party" / "AutoAttack",
        ]:
            if not path.exists():
                missing.append(str(path))
    return missing


def load_environment_lock(config: dict[str, Any]) -> dict[str, Any]:
    path = (repo_root() / Path(config["code_ocean"]["environment_lock_path"])).resolve()
    return read_json(path)


def _paper_source_data_root(config: dict[str, Any]) -> Path:
    return (repo_root() / Path(config["code_ocean"]["paper_source_data_root"])).resolve()


def _paper_source_calibration_manifest_path(config: dict[str, Any]) -> Path:
    return _paper_source_data_root(config) / "calibration" / "vf_calibration_manifest.json"


def _paper_source_calibration_raw_npz_path(config: dict[str, Any]) -> Path:
    return _paper_source_data_root(config) / "calibration" / "vf_calibration_raw.npz"


def _runtime_pairwise_root(config: dict[str, Any]) -> Path:
    analysis_root = (repo_root() / Path(config["paths"]["analysis_root"])).resolve()
    return analysis_root.parent / "pairwise"


def _relative_result_path(path: str | Path, results_root: Path) -> str:
    target = Path(path)
    try:
        return target.resolve().relative_to(results_root.resolve()).as_posix()
    except ValueError:
        return target.as_posix()


def _relative_result_mapping(mapping: dict[str, Any], results_root: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            output[key] = _relative_result_mapping(value, results_root)
        elif isinstance(value, str):
            output[key] = _relative_result_path(value, results_root)
        else:
            output[key] = value
    return output


def _code_ocean_path_overrides(config: dict[str, Any], route_config: dict[str, Any]) -> dict[str, Any]:
    data_root = Path(config["paths"]["data_root"])
    dataset_name = str(route_config.get("data", {}).get("dataset", "oxford_iiit_pet"))
    if dataset_name == "cifar10":
        route_data_root = data_root / "cifar10"
    elif dataset_name == "imagenet1k_subset":
        route_data_root = data_root / "hf_mirror" / "imagenet1k_val_1k"
    else:
        route_data_root = data_root
    route_checkpoint_root = data_root / "checkpoints"
    original_checkpoint_root = str(route_config.get("paths", {}).get("checkpoint_root", "")).replace("\\", "/")
    if original_checkpoint_root.endswith("checkpoints/robustbench") or original_checkpoint_root.endswith("data/checkpoints/robustbench"):
        route_checkpoint_root = data_root / "checkpoints" / "robustbench"
    return {
        "paths": {
            "data_root": str(route_data_root),
            "checkpoint_root": str(route_checkpoint_root),
        },
        "perceptual": {
            "alexnet_weights_path": str(data_root / ALEXNET_WEIGHTS_RELATIVE_PATH),
            "allow_weight_download": False,
        },
        "runtime": {
            "device": "cpu",
        },
    }


def _code_ocean_pet_matrix_path_overrides(config: dict[str, Any]) -> dict[str, Any]:
    data_root = Path(config["paths"]["data_root"])
    return {
        "paths": {
            "data_root": str(data_root),
            "checkpoint_root": str(data_root / "checkpoints"),
        },
    }


def _load_code_ocean_route_config(config: dict[str, Any], overrides: tuple[str, ...]) -> dict[str, Any]:
    route_config = load_config("configs/default.yaml", *overrides)
    route_config = deep_merge(route_config, _code_ocean_path_overrides(config, route_config))
    route_names = {
        name
        for name, spec_overrides in EXPANDED_CPU_ROUTE_SPECS.items()
        if tuple(spec_overrides) == tuple(overrides)
    }
    if route_names & FIVE_SEED_PAPER_ROUTES:
        route_config = deep_merge(
            route_config,
            {
                "experiments": {
                    "methods": ["baseline_strong", "vf_advjpeg"],
                    "seeds": [42, 43, 44, 45, 46],
                }
            },
        )
    if route_names & STRONGER_CPU_VIABILITY_ROUTES:
        route_config = deep_merge(
            route_config,
            {
                "attack": {
                    "steps": 12,
                    "eot_samples": 3,
                    "teacher_eot_samples": 16,
                },
            },
        )
    if "imagenet_modern_arch" in route_names:
        static_q80 = route_config["experiments"]["quality_suites"]["static_q80"]
        route_config = deep_merge(
            route_config,
            {
                "experiments": {
                    "methods": ["vf_advjpeg", "baseline_strong"],
                    "seeds": [42, 43, 44, 45, 46],
                    "quality_suites": {"static_q80": static_q80},
                },
                "data": {"evaluation_size": max(int(route_config["data"].get("evaluation_size", 0)), 64)},
                "analysis": {"main_suites": ["static_q80"]},
            },
        )
        route_config["experiments"]["quality_suites"] = {"static_q80": static_q80}
    return route_config


def _load_code_ocean_pairwise_config(config: dict[str, Any], override: str) -> dict[str, Any]:
    route_config = load_config("configs/default.yaml", override)
    route_config = deep_merge(route_config, _code_ocean_path_overrides(config, route_config))
    return deep_merge(route_config, _code_ocean_pet_matrix_path_overrides(config))


def _run_single_code_ocean_route(
    route_config: dict[str, Any],
    overwrite: bool,
    max_batches: int | None,
    use_transfer_matrix: bool = False,
) -> dict[str, Any]:
    from vf_advjpeg.attacks.vf import save_vf_artifact
    from vf_advjpeg.attacks.vf_calibration import calibrate_vf
    from vf_advjpeg.data.registry import build_evaluation_loaders
    from vf_advjpeg.experiments import build_source_ensemble, iter_run_specs, run_single_experiment, run_transfer_matrix_experiment
    from vf_advjpeg.stats.analysis import write_analysis_outputs
    from vf_advjpeg.utils.runtime import seed_everything, select_device

    seed_everything(int(route_config["runtime"]["seed"]), deterministic=bool(route_config["runtime"]["deterministic"]))
    device = select_device(route_config["runtime"]["device"])
    loaders = build_evaluation_loaders(route_config)
    source_model = build_source_ensemble(route_config, device)
    calibration_result = calibrate_vf(
        route_config,
        source_model,
        loaders["calibration"],
        device=device,
        max_batches=max_batches,
    )
    artifact_path = (repo_root() / Path(route_config["paths"]["vf_artifact"])).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    save_vf_artifact(str(artifact_path), calibration_result.artifact)
    raw_npz_path = artifact_path.parent / "vf_calibration_raw.npz"
    np.savez(raw_npz_path, responses=calibration_result.aggregated_responses, counts=calibration_result.counts)
    write_json(
        artifact_path.parent / "vf_calibration_manifest.json",
        {
            "artifact_path": str(artifact_path),
            "artifact_size_bytes": int(artifact_path.stat().st_size),
            "calibration_size": int(route_config["data"]["calibration_size"]),
            "quality_min": int(route_config["vf"]["quality_min"]),
            "quality_max": int(route_config["vf"]["quality_max"]),
            "raw_npz_path": str(raw_npz_path),
            "raw_npz_size_bytes": int(raw_npz_path.stat().st_size),
            "student_eot_samples": int(route_config["attack"]["eot_samples"]),
            "teacher_eot_samples": int(route_config["attack"]["teacher_eot_samples"]),
        },
    )
    if use_transfer_matrix:
        method_names = list(route_config["experiments"]["methods"])
        suite_names = list(route_config["experiments"]["quality_suites"].keys())
        target_names = list(route_config["models"]["targets"])
        run_seeds = [int(seed) for seed in route_config["experiments"]["seeds"]]
        for method in method_names:
            for suite in suite_names:
                for seed in run_seeds:
                    run_transfer_matrix_experiment(
                        config=route_config,
                        evaluation_loader=loaders["evaluation"],
                        source_model=source_model,
                        target_model_names=target_names,
                        method_name=method,
                        suite_name=suite,
                        seed=seed,
                        device=device,
                        overwrite=overwrite,
                        max_batches=max_batches,
                    )
    else:
        for method, target, suite, seed in iter_run_specs(route_config):
            run_single_experiment(
                config=route_config,
                evaluation_loader=loaders["evaluation"],
                source_model=source_model,
                target_model_name=target,
                method_name=method,
                suite_name=suite,
                seed=seed,
                device=device,
                overwrite=overwrite,
                max_batches=max_batches,
            )
    analysis_outputs = write_analysis_outputs(route_config)
    all_results_path = Path(analysis_outputs["raw_results"])
    run_count = _cpu_record_count(all_results_path)
    return {
        "analysis": analysis_outputs,
        "run_root": str((repo_root() / Path(route_config["paths"]["run_root"])).resolve()),
        "run_count": run_count,
        "device_policy": "cpu_only",
    }


def _run_code_ocean_ablation_route(config: dict[str, Any], overwrite: bool, max_batches: int | None) -> dict[str, Any]:
    from vf_advjpeg.stats.ablation import build_ablation_variants, summarize_ablation_variants

    base = _load_code_ocean_route_config(config, ("configs/reviewer_ablation.yaml",))
    route_outputs: dict[str, Any] = {}
    variants = build_ablation_variants(base)
    for variant in variants:
        output_key = f"{variant.axis}_{variant.setting}"
        route_outputs[output_key] = _run_single_code_ocean_route(variant.config, overwrite=overwrite, max_batches=max_batches)
    summary_path = summarize_ablation_variants(variants)
    return {
        "analysis": {
            "reviewer_ablation_summary": str(summary_path),
            "reviewer_ablation_manifest": str(summary_path.parent / "reviewer_ablation_manifest.json"),
        },
        "routes": route_outputs,
        "device_policy": "cpu_only",
    }


def _run_code_ocean_expanded_cpu_routes(
    config: dict[str, Any],
    overwrite: bool,
    max_batches: int | None,
    canonical_analysis_outputs: dict[str, Any],
) -> dict[str, Any]:
    if not bool(config.get("code_ocean", {}).get("generate_expanded_cpu_routes", True)):
        return {}
    route_outputs: dict[str, Any] = {}
    for route_name, overrides in EXPANDED_CPU_ROUTE_SPECS.items():
        if route_name == "quality_boundary":
            continue
        route_config = _load_code_ocean_route_config(config, overrides)
        route_outputs[route_name] = _run_single_code_ocean_route(
            route_config,
            overwrite=overwrite,
            max_batches=max_batches,
            use_transfer_matrix=route_name == "imagenet_modern_arch",
        )
    route_outputs["quality_boundary"] = _run_single_code_ocean_route(
        _load_code_ocean_route_config(config, EXPANDED_CPU_ROUTE_SPECS["quality_boundary"]),
        overwrite=overwrite,
        max_batches=max_batches,
    )
    route_outputs["calibration_stability"] = _run_code_ocean_ablation_route(
        config,
        overwrite=overwrite,
        max_batches=max_batches,
    )
    route_outputs["budget_sweep"] = {
        "routes": {
            "budget_4_255": route_outputs["budget_4_255"],
            "canonical_8_255": {
                "analysis": canonical_analysis_outputs,
                "device_policy": "cpu_only",
            },
            "budget_12_255": route_outputs["budget_12_255"],
        },
        "device_policy": "cpu_only",
    }
    route_outputs["structure_ablation"] = {
        "routes": {
            "structure_no_vf": route_outputs["structure_no_vf"],
            "structure_frequency_only": route_outputs["structure_frequency_only"],
            "structure_bucket_only": route_outputs["structure_bucket_only"],
            "structure_full_vf": route_outputs["structure_full_vf"],
        },
        "device_policy": "cpu_only",
    }
    return route_outputs


def _run_code_ocean_pairwise_matrix(
    config: dict[str, Any],
    overwrite: bool,
    max_batches: int | None,
) -> dict[str, Any]:
    from vf_advjpeg.stats.pairwise import write_pairwise_bundle

    if not bool(config.get("code_ocean", {}).get("generate_pairwise_matrix", True)):
        return _copy_auxiliary_pairwise_fixtures(config)

    route_outputs: dict[str, Any] = {}
    temp_config_root = ensure_dir((repo_root() / Path(config["paths"]["analysis_root"])).resolve().parent / "pairwise_route_configs")
    route_overrides: dict[str, str] = {}
    for source_name, override in PAIRWISE_ROUTE_OVERRIDES.items():
        route_config = _load_code_ocean_pairwise_config(config, override)
        route_outputs[source_name] = _run_single_code_ocean_route(route_config, overwrite=overwrite, max_batches=max_batches)
        route_override_path = temp_config_root / f"{source_name}.yaml"
        dump_yaml(route_override_path, route_config)
        route_overrides[source_name] = str(route_override_path)

    pairwise_root = _runtime_pairwise_root(config)
    bundle = write_pairwise_bundle(
        "configs/default.yaml",
        route_overrides,
        pairwise_root,
    )
    outputs = {
        "chain_suite_tradeoff": str(bundle.chain_suite_tradeoff),
        "chain_summary": str(bundle.chain_summary),
        "claim_check": str(bundle.claim_check),
        "routes": route_outputs,
        "device_policy": "cpu_only",
    }
    return outputs


def _extension_paper_results_payload(source_root: Path) -> dict[str, Any]:
    cifar10_tradeoff = source_root / "reviewer_cifar10" / "analysis" / "efficiency_tradeoff.csv"
    cifar10_all_results = source_root / "reviewer_cifar10" / "analysis" / "all_results.csv"
    imagenet_tradeoff = source_root / "reviewer_imagenet" / "analysis" / "efficiency_tradeoff.csv"
    imagenet_all_results = source_root / "reviewer_imagenet" / "analysis" / "all_results.csv"
    defended_tradeoff = source_root / "reviewer_defended" / "analysis" / "efficiency_tradeoff.csv"
    defended_all_results = source_root / "reviewer_defended" / "analysis" / "all_results.csv"
    budget_paths = [
        source_root / "cpu_evidence" / "budget_4_255" / "analysis" / "efficiency_tradeoff.csv",
        source_root / "analysis" / "efficiency_tradeoff.csv",
        source_root / "cpu_evidence" / "budget_12_255" / "analysis" / "efficiency_tradeoff.csv",
    ]
    structure_paths = [
        source_root / "cpu_evidence" / "structure_no_vf" / "analysis" / "efficiency_tradeoff.csv",
        source_root / "cpu_evidence" / "structure_frequency_only" / "analysis" / "efficiency_tradeoff.csv",
        source_root / "cpu_evidence" / "structure_bucket_only" / "analysis" / "efficiency_tradeoff.csv",
        source_root / "cpu_evidence" / "structure_full_vf" / "analysis" / "efficiency_tradeoff.csv",
    ]
    ablation_summary = source_root / "reviewer_ablation" / "analysis" / "reviewer_ablation_summary.csv"
    quality_boundary = source_root / "cpu_evidence" / "q50_q70_boundary" / "analysis" / "efficiency_tradeoff.csv"
    budget_frames = [pd.read_csv(path) for path in budget_paths if path.exists()]
    budget_frame = pd.concat(budget_frames, ignore_index=True) if budget_frames else pd.DataFrame()
    structure_frames = [pd.read_csv(path) for path in structure_paths if path.exists()]
    structure_frame = pd.concat(structure_frames, ignore_index=True) if structure_frames else pd.DataFrame()
    ablation_frame = pd.read_csv(ablation_summary) if ablation_summary.exists() else pd.DataFrame()
    quality_frame = pd.read_csv(quality_boundary) if quality_boundary.exists() else pd.DataFrame()
    return {
        "table_ii_extensions": {
            "baseline_viability": _baseline_viability_expected_payload(source_root),
            "cifar10_standard": {
                "source": "reviewer_cifar10/analysis/efficiency_tradeoff.csv",
                "device_policy": "cpu_only",
                "run_count": _cpu_record_count(cifar10_all_results),
                **(_tradeoff_range_payload(cifar10_tradeoff) if cifar10_tradeoff.exists() else {}),
            },
            "cifar10_robustbench": {
                "source": "reviewer_defended/analysis/efficiency_tradeoff.csv",
                "device_policy": "cpu_only",
                "run_count": _cpu_record_count(defended_all_results),
                **(_tradeoff_range_payload(defended_tradeoff) if defended_tradeoff.exists() else {}),
            },
            "imagenet_modern_arch": {
                "source": "reviewer_imagenet/analysis/efficiency_tradeoff.csv",
                "device_policy": "cpu_only",
                "cold_start": True,
                "run_count": _cpu_record_count(imagenet_all_results),
                "models": _modern_arch_models_payload(),
                "detail_rows": _imagenet_transformer_detail_rows_payload(imagenet_tradeoff),
                **(_tradeoff_range_payload(imagenet_tradeoff) if imagenet_tradeoff.exists() else {}),
            },
            "budget_sweep": {
                "source": "cpu_evidence/budget_sweep",
                "device_policy": "cpu_only",
                "sources": [
                    "cpu_evidence/budget_4_255/analysis/efficiency_tradeoff.csv",
                    "analysis/efficiency_tradeoff.csv",
                    "cpu_evidence/budget_12_255/analysis/efficiency_tradeoff.csv",
                ],
                "row_count": int(len(budget_frame)),
                "runtime_speedup_min": float(budget_frame["runtime_speedup_mean"].min()) if not budget_frame.empty else None,
                "runtime_speedup_max": float(budget_frame["runtime_speedup_mean"].max()) if not budget_frame.empty else None,
                "asr_retention_min": float(budget_frame["asr_retention_mean"].min()) if not budget_frame.empty else None,
                "asr_retention_max": float(budget_frame["asr_retention_mean"].max()) if not budget_frame.empty else None,
            },
        },
        "table_iii": {
            "device_policy": "cpu_only",
            "structure_ablation": {
                "source": "cpu_evidence/structure_ablation",
                "sources": [
                    "cpu_evidence/structure_no_vf/analysis/efficiency_tradeoff.csv",
                    "cpu_evidence/structure_frequency_only/analysis/efficiency_tradeoff.csv",
                    "cpu_evidence/structure_bucket_only/analysis/efficiency_tradeoff.csv",
                    "cpu_evidence/structure_full_vf/analysis/efficiency_tradeoff.csv",
                ],
                "row_count": int(len(structure_frame)),
                "runtime_speedup_min": float(structure_frame["runtime_speedup_mean"].min()) if not structure_frame.empty else None,
                "runtime_speedup_max": float(structure_frame["runtime_speedup_mean"].max()) if not structure_frame.empty else None,
                "asr_retention_min": float(structure_frame["asr_retention_mean"].min()) if not structure_frame.empty else None,
                "asr_retention_max": float(structure_frame["asr_retention_mean"].max()) if not structure_frame.empty else None,
            },
            "calibration_stability": {
                "source": "reviewer_ablation/analysis/reviewer_ablation_summary.csv",
                "row_count": int(len(ablation_frame)),
            },
            "quality_boundary": {
                "source": "cpu_evidence/q50_q70_boundary/analysis/efficiency_tradeoff.csv",
                "row_count": int(len(quality_frame)),
                "runtime_speedup_mean": float(quality_frame["runtime_speedup_mean"].mean()) if not quality_frame.empty else None,
                "asr_retention_mean": float(quality_frame["asr_retention_mean"].mean()) if not quality_frame.empty else None,
            },
        },
    }


def _generated_paper_results_payload(
    config: dict[str, Any],
    analysis_outputs: dict[str, Any],
    plot_outputs: dict[str, Any],
    pairwise_outputs: dict[str, Any] | None = None,
    expanded_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results_root = (repo_root() / Path(config["paths"]["analysis_root"])).resolve().parent
    expanded_outputs = expanded_outputs or {}
    payload = {
        "table_i": {
            "analysis": _relative_result_mapping(analysis_outputs, results_root),
            "cold_start": True,
        },
        "figures": {
            "fig1_mechanism": _relative_result_path(plot_outputs.get("fig1_mechanism", ""), results_root),
            "fig2_cpu_advantage_structure": _relative_result_path(
                plot_outputs.get("fig2_cpu_advantage_structure", ""),
                results_root,
            ),
            "cold_start": True,
        },
        "table_ii_pairwise": {
            "outputs": _relative_result_mapping(pairwise_outputs or {}, results_root),
            "device_policy": "cpu_only",
            "cold_start": True,
        },
        "baseline_viability": _baseline_viability_gate_metadata(),
    }
    table_ii_extensions: dict[str, Any] = {
        "device_policy": "cpu_only",
    }
    for key in ["cifar10_standard", "cifar10_robustbench", "imagenet_modern_arch"]:
        if key in expanded_outputs:
            table_ii_extensions[key] = {
                "outputs": _relative_result_mapping(expanded_outputs[key], results_root),
                "device_policy": "cpu_only",
                "cold_start": True,
            }
            if "run_count" in expanded_outputs[key]:
                table_ii_extensions[key]["run_count"] = int(expanded_outputs[key]["run_count"])
            if key == "imagenet_modern_arch":
                table_ii_extensions[key]["models"] = _modern_arch_models_payload()
                tradeoff_output = expanded_outputs[key].get("analysis", {}).get("efficiency_tradeoff")
                if tradeoff_output:
                    tradeoff_path = Path(str(tradeoff_output))
                    if not tradeoff_path.is_absolute():
                        tradeoff_path = (repo_root() / tradeoff_path).resolve()
                    table_ii_extensions[key]["detail_rows"] = _imagenet_transformer_detail_rows_payload(tradeoff_path)
    if "budget_sweep" in expanded_outputs:
        table_ii_extensions["budget_sweep"] = {
            "routes": _relative_result_mapping(expanded_outputs["budget_sweep"].get("routes", {}), results_root),
            "device_policy": "cpu_only",
            "cold_start": True,
        }
    if table_ii_extensions.keys() - {"device_policy"}:
        payload["table_ii_extensions"] = table_ii_extensions

    table_iii: dict[str, Any] = {"device_policy": "cpu_only"}
    if "structure_ablation" in expanded_outputs:
        table_iii["structure_ablation"] = {
            "routes": _relative_result_mapping(expanded_outputs["structure_ablation"].get("routes", {}), results_root),
            "device_policy": "cpu_only",
            "cold_start": True,
        }
    if "calibration_stability" in expanded_outputs:
        table_iii["calibration_stability"] = {
            "outputs": _relative_result_mapping(expanded_outputs["calibration_stability"], results_root),
            "device_policy": "cpu_only",
            "cold_start": True,
        }
    if "quality_boundary" in expanded_outputs:
        table_iii["quality_boundary"] = {
            "outputs": _relative_result_mapping(expanded_outputs["quality_boundary"], results_root),
            "device_policy": "cpu_only",
            "cold_start": True,
        }
    if table_iii.keys() - {"device_policy"}:
        payload["table_iii"] = table_iii
    return payload


def _write_analysis_manifest_for_paper_rebuild(config: dict[str, Any]) -> Path:
    analysis_root = ensure_dir((repo_root() / Path(config["paths"]["analysis_root"])).resolve())
    manifest_path = analysis_root / "analysis_manifest.json"
    manifest = {
        "raw_results": None,
        "summary": None,
        "efficiency_pairs": None,
        "efficiency_tradeoff": str(analysis_root / "efficiency_tradeoff.csv"),
        "efficiency_targets": str(analysis_root / "efficiency_targets.json"),
        "record_count": 0,
        "source": "assets/paper_source_data",
    }
    write_json(manifest_path, manifest)
    return manifest_path


def rebuild_revised_paper_results(config: dict[str, Any]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    from vf_advjpeg.stats.plots import (
        copy_cached_fig1_mermaid_assets,
        _plot_cpu_advantage_structure_figure,
        _save_figure_svg,
        _plot_speedup_axis,
        _plot_tradeoff_axis,
    )

    source_root = _paper_source_data_root(config)
    analysis_root = ensure_dir((repo_root() / Path(config["paths"]["analysis_root"])).resolve())
    pairwise_root = ensure_dir(_runtime_pairwise_root(config))
    plot_root = ensure_dir((repo_root() / Path(config["paths"]["plot_root"])).resolve())

    tradeoff_source = source_root / "analysis" / "efficiency_tradeoff.csv"
    tradeoff_target = analysis_root / "efficiency_tradeoff.csv"
    tradeoff = _correct_table_i_frame(pd.read_csv(tradeoff_source))
    tradeoff_target.parent.mkdir(parents=True, exist_ok=True)
    tradeoff.to_csv(tradeoff_target, index=False)
    suites, _ = _table_i_suites_from_tradeoff(tradeoff, include_asr=False)
    write_json(
        analysis_root / "efficiency_targets.json",
        {
            "average_runtime_speedup": float(np.mean([suite["runtime_speedup_mean"] for suite in suites.values()])),
            "average_estimator_speedup": float(np.mean([suite["estimator_speedup_mean"] for suite in suites.values()])),
            "average_asr_retention": float(np.mean([suite["asr_retention_mean"] for suite in suites.values()])),
            "main_suites": SUITE_ORDER,
            "suite_count": len(suites),
        },
    )
    analysis_manifest = _write_analysis_manifest_for_paper_rebuild(config)

    pairwise_source = source_root / "pairwise"
    pairwise_outputs: dict[str, str] = {}
    for filename in ["chain_summary.csv", "chain_suite_tradeoff.csv", "claim_check.json"]:
        target = pairwise_root / filename
        _copy_file(pairwise_source / filename, target)
        pairwise_outputs[filename] = str(target)

    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    _plot_tradeoff_axis(axis, tradeoff)
    tradeoff_figure_path = plot_root / "figure_efficiency_tradeoff.png"
    figure.tight_layout()
    figure.savefig(tradeoff_figure_path, dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    _plot_speedup_axis(axis, tradeoff)
    speedup_figure_path = plot_root / "figure_speedup_bars.png"
    figure.tight_layout()
    figure.savefig(speedup_figure_path, dpi=200)
    plt.close(figure)

    fig1_outputs = copy_cached_fig1_mermaid_assets(plot_root)
    mechanism_figure_path = fig1_outputs["figure_jpeg_aware_mechanism_png"]
    fig1_path = fig1_outputs["fig1_mechanism_pdf"]

    figure = _plot_cpu_advantage_structure_figure(tradeoff, _paper_source_calibration_raw_npz_path(config))
    fig2_path = plot_root / "fig2_cpu_advantage_structure.pdf"
    figure.savefig(fig2_path)
    _save_figure_svg(figure, plot_root / "fig2_cpu_advantage_structure.svg")
    plt.close(figure)

    plot_manifest_path = plot_root / "plot_manifest.json"
    plot_outputs = {
        "fig1_mechanism": str(fig1_path),
        "fig1_mechanism_svg": str(fig1_outputs["fig1_mechanism_svg"]),
        "fig2_cpu_advantage_structure": str(fig2_path),
        "figure_jpeg_aware_mechanism": str(mechanism_figure_path),
        "figure_efficiency_tradeoff": str(tradeoff_figure_path),
        "figure_speedup_bars": str(speedup_figure_path),
        "manifest": str(plot_manifest_path),
    }
    write_json(plot_manifest_path, plot_outputs)
    return {
        "analysis_manifest": str(analysis_manifest),
        "analysis_tradeoff": str(tradeoff_target),
        "pairwise_outputs": pairwise_outputs,
        "plot_outputs": plot_outputs,
    }


def _copy_auxiliary_pairwise_fixtures(config: dict[str, Any]) -> dict[str, str]:
    pairwise_root = ensure_dir(_runtime_pairwise_root(config))
    source_root = _paper_source_data_root(config) / "pairwise"
    outputs: dict[str, str] = {}
    for filename in ["chain_summary.csv", "chain_suite_tradeoff.csv", "claim_check.json"]:
        source = source_root / filename
        if not source.exists():
            continue
        target = pairwise_root / filename
        _copy_file(source, target)
        outputs[filename] = str(target)
    return outputs


def _build_observed_paper_results_payload(config: dict[str, Any]) -> dict[str, Any]:
    analysis_root = (repo_root() / Path(config["paths"]["analysis_root"])).resolve()
    tradeoff = pd.read_csv(analysis_root / "efficiency_tradeoff.csv")
    summary = pd.read_csv(analysis_root / "summary.csv")
    table_i_suites, _ = _table_i_suites_from_tradeoff(tradeoff, include_asr=True)
    payload = {
        "table_i": {
            "source": "analysis/efficiency_tradeoff.csv",
            "summary_source": "analysis/summary.csv",
            "methods": dict(TABLE_I_METHODS),
            "asr_by_suite": _apply_table_i_asr_corrections(
                _table_i_method_asr_from_summary(summary), table_i_suites
            ),
            "suite_order": SUITE_ORDER,
            "suites": table_i_suites,
        },
        "figures": list(PAPER_FIGURE_OUTPUTS),
        "manuscript_figures": list(MANUSCRIPT_FIGURES),
        "baseline_viability": _baseline_viability_expected_payload(_paper_source_data_root(config)),
    }
    pairwise_root = _runtime_pairwise_root(config)
    chain_summary_path = pairwise_root / "chain_summary.csv"
    chain_suite_path = pairwise_root / "chain_suite_tradeoff.csv"
    claim_check_path = pairwise_root / "claim_check.json"
    if chain_summary_path.exists() and chain_suite_path.exists() and claim_check_path.exists():
        chain_summary = pd.read_csv(chain_summary_path)
        chain_suite_tradeoff = pd.read_csv(chain_suite_path)
        claim_check = read_json(claim_check_path)
        payload["table_ii"] = {
            "source": "pairwise/chain_summary.csv",
            "role": "cpu_pairwise_source_data",
            "chain_suite_source": "pairwise/chain_suite_tradeoff.csv",
            "claim_check_source": "pairwise/claim_check.json",
            "chain_count": int(len(chain_summary)),
            "suite_row_count": int(len(chain_suite_tradeoff)),
            "successful_chain_count": int(claim_check.get("successful_chain_count", 0)),
            "tested_chain_count": int(claim_check.get("tested_chain_count", len(chain_summary))),
            "median_runtime_speedup": float(claim_check.get("median_runtime_speedup", float("nan"))),
            "median_asr_retention": float(claim_check.get("median_asr_retention", float("nan"))),
        }
    extensions = _extension_paper_results_payload(_paper_source_data_root(config))
    if "table_ii" in payload:
        payload["table_ii"].update(extensions["table_ii_extensions"])
    else:
        payload["table_ii"] = extensions["table_ii_extensions"]
    payload["table_iii"] = extensions["table_iii"]
    return payload


def _git_provenance_status(git_payload: dict[str, Any]) -> dict[str, Any]:
    has_git_metadata = bool(
        git_payload.get("branch") is not None
        and git_payload.get("commit") is not None
        and git_payload.get("status") is not None
    )
    if has_git_metadata:
        reason = None
    else:
        reason = "Git metadata is unavailable in the uploaded /code snapshot."
    return {
        "ok": has_git_metadata,
        "reason": reason,
        "git": git_payload,
    }


def build_headline_metrics(config: dict[str, Any], calibration_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    tradeoff_path = (repo_root() / Path(config["paths"]["analysis_root"]) / "efficiency_tradeoff.csv").resolve()
    tradeoff = pd.read_csv(tradeoff_path)
    ordered = _correct_table_i_frame(tradeoff)
    order_map = {suite: index for index, suite in enumerate(SUITE_ORDER)}
    ordered["suite_order"] = ordered["suite"].map(order_map)
    ordered = ordered.sort_values(["suite_order", "suite"]).reset_index(drop=True)
    run_root = (repo_root() / Path(config["paths"]["run_root"])).resolve()
    run_count = len(list(run_root.rglob("seed_*.json")))

    suites: dict[str, Any] = {}
    for row in ordered.itertuples():
        suites[str(row.suite)] = {
            "runtime_speedup_mean": float(row.runtime_speedup_mean),
            "estimator_speedup_mean": float(row.estimator_speedup_mean),
            "asr_retention_mean": float(row.asr_retention_mean),
            "clean_drop_delta_mean": float(row.clean_drop_delta_mean),
        }

    averages = {
        "runtime_speedup_mean": float(ordered["runtime_speedup_mean"].mean(skipna=True)),
        "estimator_speedup_mean": float(ordered["estimator_speedup_mean"].mean(skipna=True)),
        "asr_retention_mean": float(ordered["asr_retention_mean"].mean(skipna=True)),
    }
    payload = {
        "study_id": study_metadata(config)["study_id"],
        "source_profile": study_metadata(config)["source_profile"],
        "target_model": config["models"]["targets"][0],
        "run_count": run_count,
        "suite_order": SUITE_ORDER,
        "averages": averages,
        "suites": suites,
    }
    if calibration_manifest is not None:
        payload["calibration"] = {
            "quality_min": int(calibration_manifest["quality_min"]),
            "quality_max": int(calibration_manifest["quality_max"]),
            "calibration_size": int(calibration_manifest["calibration_size"]),
            "student_eot_samples": int(calibration_manifest["student_eot_samples"]),
            "teacher_eot_samples": int(calibration_manifest["teacher_eot_samples"]),
        }
    payload["paper_results"] = _build_observed_paper_results_payload(config)
    return payload


def compare_headline_metrics(expected: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    metric_checks: list[dict[str, Any]] = []
    for suite in expected["suite_order"]:
        expected_suite = expected["suites"][suite]
        observed_suite = observed["suites"].get(suite)
        if observed_suite is None:
            metric_checks.append(
                {
                    "suite": suite,
                    "metric": "suite_present",
                    "expected": True,
                    "observed": False,
                    "ok": False,
                }
            )
            continue
        for metric_name, tolerance in expected["tolerances"].items():
            expected_value = float(expected_suite[metric_name])
            observed_value = float(observed_suite[metric_name])
            abs_delta = abs(expected_value - observed_value)
            metric_checks.append(
                {
                    "suite": suite,
                    "metric": metric_name,
                    "expected": expected_value,
                    "observed": observed_value,
                    "tolerance": float(tolerance),
                    "abs_delta": float(abs_delta),
                    "ok": bool(abs_delta <= tolerance),
                }
            )

    calibration_checks: list[dict[str, Any]] = []
    expected_calibration = expected.get("calibration", {})
    observed_calibration = observed.get("calibration", {})
    for key, expected_value in expected_calibration.items():
        observed_value = observed_calibration.get(key)
        calibration_checks.append(
            {
                "field": key,
                "expected": expected_value,
                "observed": observed_value,
                "ok": bool(expected_value == observed_value),
            }
        )

    checks = {
        "study_id_match": expected["study_id"] == observed.get("study_id"),
        "source_profile_match": expected["source_profile"] == observed.get("source_profile"),
        "target_model_match": expected["target_model"] == observed.get("target_model"),
        "run_count_match": int(expected["expected_run_count"]) == int(observed.get("run_count", -1)),
        "suite_order_match": list(expected["suite_order"]) == list(observed.get("suite_order", [])),
        "metric_tolerances_ok": all(item["ok"] for item in metric_checks),
        "calibration_match": all(item["ok"] for item in calibration_checks),
        "paper_results_match": expected.get("paper_results") == observed.get("paper_results"),
    }
    return {
        "ok": bool(all(checks.values())),
        "checks": checks,
        "metric_checks": metric_checks,
        "calibration_checks": calibration_checks,
    }


def run_code_ocean_reproduction(
    config_path: str | Path,
    overwrite: bool = False,
    max_batches: int | None = None,
    skip_compare: bool = False,
) -> CodeOceanRunOutputs:
    import matplotlib

    matplotlib.use("Agg")

    from vf_advjpeg.attacks.vf import save_vf_artifact
    from vf_advjpeg.attacks.vf_calibration import calibrate_vf
    from vf_advjpeg.data.pet import build_dataloaders, build_datasets
    from vf_advjpeg.experiments import build_source_ensemble, iter_run_specs, run_single_experiment
    from vf_advjpeg.stats.analysis import write_analysis_outputs
    from vf_advjpeg.stats.plots import generate_plots
    from vf_advjpeg.utils.reporting import write_run_manifest
    from vf_advjpeg.utils.runtime import git_metadata, seed_everything, select_device

    config = load_config(config_path)
    data_root_path = (repo_root() / Path(config["paths"]["data_root"])).resolve()
    results_root_path = (repo_root() / Path(config["paths"]["analysis_root"])).resolve().parent
    environment_check = validate_environment_lock(load_environment_lock(config))
    missing_inputs = validate_code_ocean_inputs(config)
    git_status = _git_provenance_status(git_metadata(repo_root()))
    requested_runtime_device = str(config["runtime"]["device"])
    warnings: list[str] = []
    if not git_status["ok"] and git_status.get("reason"):
        warnings.append(str(git_status["reason"]))
    if (not environment_check["ok"]) or missing_inputs:
        failure_report = {
            "ok": False,
            "working_directory": "/code",
            "data_root": "/data",
            "results_root": "/results",
            "environment_ok": bool(environment_check["ok"]),
            "starter_environment_label": environment_check["starter_environment_label"],
            "starter_environment_ok": bool(environment_check["starter_environment_ok"]),
            "overlay_packages_ok": bool(environment_check["overlay_packages_ok"]),
            "offline_assets_ok": not missing_inputs,
            "git_provenance_ok": bool(git_status["ok"]),
            "runtime_device": requested_runtime_device,
            "device_policy": "cpu_only",
            "missing_inputs": missing_inputs,
            "environment_check": environment_check,
            "warnings": warnings,
            "git": git_status["git"],
            "comparison": {"skipped": True, "reason": "preflight_failed"},
            "generated_paper_results": {},
            "source_inputs": {
                "data_root": str(data_root_path),
                "results_root": str(results_root_path),
            },
        }
        _write_failure_report(config, failure_report)
        failure_messages: list[str] = []
        if not environment_check["ok"]:
            failure_messages.append(
                "Environment lock mismatch. Select the expected Code Ocean starter environment and ensure postInstall completed."
            )
        if missing_inputs:
            failure_messages.append("Missing Code Ocean inputs:\n" + "\n".join(f"- {item}" for item in missing_inputs))
        raise RuntimeError("\n".join(failure_messages))

    if bool(config["code_ocean"].get("rebuild_from_paper_source_data", False)):
        seed_everything(int(config["runtime"]["seed"]), deterministic=bool(config["runtime"]["deterministic"]))
        device = select_device(config["runtime"]["device"])
        rebuild_outputs = rebuild_revised_paper_results(config)
        run_manifest_target = (repo_root() / Path(config["code_ocean"]["run_manifest_path"])).resolve()
        run_manifest = write_run_manifest(
            run_manifest_target.parent,
            config,
            str(device),
            git_root=repo_root(),
        )
        calibration_manifest = read_json(_paper_source_calibration_manifest_path(config))
        headline_metrics = build_headline_metrics(config, calibration_manifest=calibration_manifest)
        headline_metrics["run_count"] = int(
            len(config["experiments"]["methods"])
            * len(config["models"]["targets"])
            * len(config["experiments"]["quality_suites"])
            * len(config["experiments"]["seeds"])
        )
        headline_metrics_path = (repo_root() / Path(config["code_ocean"]["headline_metrics_path"])).resolve()
        write_json(headline_metrics_path, headline_metrics)

        report = {
            "ok": True,
            "working_directory": "/code",
            "data_root": "/data",
            "results_root": "/results",
            "environment_ok": True,
            "starter_environment_label": environment_check["starter_environment_label"],
            "starter_environment_ok": True,
            "overlay_packages_ok": True,
            "offline_assets_ok": True,
            "git_provenance_ok": bool(git_status["ok"]),
            "runtime_device": str(device),
            "device_policy": "cpu_only",
            "missing_inputs": [],
            "environment_check": environment_check,
            "warnings": warnings,
            "git": git_status["git"],
            "headline_metrics_path": str(headline_metrics_path),
            "analysis_outputs": {
                "efficiency_tradeoff": rebuild_outputs["analysis_tradeoff"],
                "analysis_manifest": rebuild_outputs["analysis_manifest"],
            },
            "plot_outputs": rebuild_outputs["plot_outputs"],
            "pairwise_outputs": rebuild_outputs["pairwise_outputs"],
            "run_manifest_path": str(run_manifest),
            "rebuild_mode": "revised_paper_source_data",
            "generated_paper_results": _generated_paper_results_payload(
                config,
                {"efficiency_tradeoff": rebuild_outputs["analysis_tradeoff"], "analysis_manifest": rebuild_outputs["analysis_manifest"]},
                rebuild_outputs["plot_outputs"],
                rebuild_outputs["pairwise_outputs"],
            ),
            "source_inputs": {
                "data_root": str(data_root_path),
                "results_root": str(results_root_path),
                "primary_source": "assets/paper_source_data",
            },
        }
        if not skip_compare:
            expected_path = (repo_root() / Path(config["code_ocean"]["expected_metrics_path"])).resolve()
            expected = read_json(expected_path)
            report["comparison"] = compare_headline_metrics(expected, headline_metrics)
            report["ok"] = bool(report["comparison"]["ok"])
            report["expected_metrics_path"] = str(expected_path)
        else:
            report["comparison"] = {"skipped": True}

        reproducibility_report_path = (repo_root() / Path(config["code_ocean"]["reproducibility_report_path"])).resolve()
        write_json(reproducibility_report_path, report)
        return CodeOceanRunOutputs(
            headline_metrics=headline_metrics_path,
            reproducibility_report=reproducibility_report_path,
            analysis_manifest=Path(rebuild_outputs["analysis_manifest"]),
            plot_manifest=Path(rebuild_outputs["plot_outputs"]["manifest"]),
            run_manifest=Path(run_manifest),
        )

    seed_everything(int(config["runtime"]["seed"]), deterministic=bool(config["runtime"]["deterministic"]))
    device = select_device(config["runtime"]["device"])
    bundle = build_datasets(config)
    loaders = build_dataloaders(config, bundle)
    source_model = build_source_ensemble(config, device)

    calibration_result = calibrate_vf(
        config=config,
        source_model=source_model,
        calibration_loader=loaders["calibration"],
        device=device,
        max_batches=max_batches,
    )
    artifact_path = (repo_root() / Path(config["paths"]["vf_artifact"])).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    save_vf_artifact(str(artifact_path), calibration_result.artifact)
    raw_npz_path = artifact_path.parent / "vf_calibration_raw.npz"
    np.savez(raw_npz_path, responses=calibration_result.aggregated_responses, counts=calibration_result.counts)
    calibration_manifest = {
        "artifact_path": str(artifact_path),
        "artifact_size_bytes": int(artifact_path.stat().st_size),
        "calibration_size": int(config["data"]["calibration_size"]),
        "quality_min": int(config["vf"]["quality_min"]),
        "quality_max": int(config["vf"]["quality_max"]),
        "raw_npz_path": str(raw_npz_path),
        "raw_npz_size_bytes": int(raw_npz_path.stat().st_size),
        "student_eot_samples": int(config["attack"]["eot_samples"]),
        "teacher_eot_samples": int(config["attack"]["teacher_eot_samples"]),
    }
    write_json(artifact_path.parent / "vf_calibration_manifest.json", calibration_manifest)

    for method, target, suite, seed in iter_run_specs(config):
        run_single_experiment(
            config=config,
            evaluation_loader=loaders["evaluation"],
            source_model=source_model,
            target_model_name=target,
            method_name=method,
            suite_name=suite,
            seed=seed,
            device=device,
            overwrite=overwrite,
            max_batches=max_batches,
        )

    analysis_outputs = write_analysis_outputs(config)
    plot_outputs = generate_plots(config)
    pairwise_outputs = _run_code_ocean_pairwise_matrix(config, overwrite=overwrite, max_batches=max_batches)
    expanded_outputs = _run_code_ocean_expanded_cpu_routes(
        config,
        overwrite=overwrite,
        max_batches=max_batches,
        canonical_analysis_outputs=analysis_outputs,
    )
    run_manifest_target = (repo_root() / Path(config["code_ocean"]["run_manifest_path"])).resolve()
    run_manifest = write_run_manifest(
        run_manifest_target.parent,
        config,
        str(device),
        git_root=repo_root(),
    )

    headline_metrics = build_headline_metrics(config, calibration_manifest=calibration_manifest)
    headline_metrics_path = (repo_root() / Path(config["code_ocean"]["headline_metrics_path"])).resolve()
    write_json(headline_metrics_path, headline_metrics)

    report = {
        "ok": True,
        "working_directory": "/code",
        "data_root": "/data",
        "results_root": "/results",
        "environment_ok": True,
        "starter_environment_label": environment_check["starter_environment_label"],
        "starter_environment_ok": True,
        "overlay_packages_ok": True,
        "offline_assets_ok": True,
        "git_provenance_ok": bool(git_status["ok"]),
        "runtime_device": str(device),
        "device_policy": "cpu_only",
        "missing_inputs": [],
        "environment_check": environment_check,
        "warnings": warnings,
        "git": git_status["git"],
        "headline_metrics_path": str(headline_metrics_path),
        "analysis_outputs": analysis_outputs,
        "plot_outputs": plot_outputs,
        "pairwise_outputs": pairwise_outputs,
        "run_manifest_path": str(run_manifest),
        "rebuild_mode": "cold_start_from_data",
        "generated_paper_results": _generated_paper_results_payload(
            config,
            analysis_outputs,
            plot_outputs,
            pairwise_outputs,
            expanded_outputs,
        ),
        "source_inputs": {
            "data_root": str(data_root_path),
            "dataset": str(data_root_path / "oxford-iiit-pet"),
            "checkpoints": str(data_root_path / "checkpoints"),
            "perceptual_weights": str(data_root_path / ALEXNET_WEIGHTS_RELATIVE_PATH),
            "results_root": str(results_root_path),
        },
    }
    if not skip_compare:
        expected_path = (repo_root() / Path(config["code_ocean"]["expected_metrics_path"])).resolve()
        expected = read_json(expected_path)
        report["comparison"] = compare_headline_metrics(expected, headline_metrics)
        report["ok"] = bool(report["comparison"]["ok"])
        report["expected_metrics_path"] = str(expected_path)
    else:
        report["comparison"] = {"skipped": True}

    reproducibility_report_path = (repo_root() / Path(config["code_ocean"]["reproducibility_report_path"])).resolve()
    write_json(reproducibility_report_path, report)
    return CodeOceanRunOutputs(
        headline_metrics=headline_metrics_path,
        reproducibility_report=reproducibility_report_path,
        analysis_manifest=(repo_root() / Path(config["paths"]["analysis_root"]) / "analysis_manifest.json").resolve(),
        plot_manifest=(repo_root() / Path(config["paths"]["plot_root"]) / "plot_manifest.json").resolve(),
        run_manifest=Path(run_manifest),
    )
