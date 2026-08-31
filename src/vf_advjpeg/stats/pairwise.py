from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from vf_advjpeg.config import load_config
from vf_advjpeg.utils.fs import ensure_dir, read_json, resolve_project_path, write_json


DEFAULT_PAIRWISE_OVERRIDES = {
    "resnet18": "configs/ei_cpu_matrix_resnet18.yaml",
    "vgg16": "configs/ei_cpu_matrix_vgg16.yaml",
    "mobilenet_v2": "configs/ei_cpu_matrix_mobilenet_v2.yaml",
    "densenet121": "configs/ei_cpu_matrix_densenet121.yaml",
}
MAIN_SUITES = ["static_q70", "static_q80", "static_q90", "dynamic_uniform"]
PAIRWISE_CHAIN_RUNTIME_SPEEDUP_MIN = 1.0
PAIRWISE_CHAIN_ASR_RETENTION_MIN = 0.75
PAIRWISE_CHAIN_SUITE_SUCCESS_MIN = 3


@dataclass(slots=True)
class PairwiseBundleOutputs:
    chain_suite_tradeoff: Path
    chain_summary: Path
    claim_check: Path


def _load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _display_model(name: str) -> str:
    mapping = {
        "resnet18": "ResNet18",
        "vgg16": "VGG16",
        "mobilenet_v2": "MobileNetV2",
        "densenet121": "DenseNet121",
    }
    return mapping.get(name, name)


def _single_source_name(config: dict[str, Any]) -> str:
    source_items = config["models"]["source"]
    if len(source_items) != 1:
        raise ValueError("Pairwise matrix aggregation expects exactly one source model per config.")
    return str(source_items[0]["name"])


def load_pairwise_tradeoff(base_config: str, override_path: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    config = load_config(base_config, override_path)
    source_model = _single_source_name(config)
    tradeoff_path = resolve_project_path(config["paths"]["analysis_root"]) / "efficiency_tradeoff.csv"
    if not tradeoff_path.exists():
        raise FileNotFoundError(f"Missing tradeoff table: {tradeoff_path}")
    tradeoff = pd.read_csv(tradeoff_path)
    tradeoff["source_model"] = source_model
    tradeoff["pair_label"] = tradeoff["target_model"].map(lambda target: f"{_display_model(source_model)}->{_display_model(str(target))}")
    return config, tradeoff


def combine_chain_suite_tradeoff(studies: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in studies.values() if not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    ordered_columns = [
        "source_model",
        "target_model",
        "pair_label",
        "suite",
        "n",
        "baseline_asr_mean",
        "candidate_asr_mean",
        "asr_retention_mean",
        "runtime_speedup_mean",
        "estimator_speedup_mean",
        "clean_drop_delta_mean",
        "runtime_faster_significant",
        "efficiency_win",
    ]
    available = [column for column in ordered_columns if column in combined.columns]
    combined = combined[available + [column for column in combined.columns if column not in available]]
    return combined.sort_values(["source_model", "target_model", "suite"]).reset_index(drop=True)


def summarize_chain_level(chain_suite_tradeoff: pd.DataFrame) -> pd.DataFrame:
    if chain_suite_tradeoff.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for (source_model, target_model, pair_label), group in chain_suite_tradeoff.groupby(
        ["source_model", "target_model", "pair_label"],
        sort=True,
    ):
        suite_lookup = {str(row.suite): float(row.runtime_speedup_mean) for row in group.itertuples()}
        q70_is_fastest = False
        dynamic_is_slowest = False
        if {"static_q70", "dynamic_uniform"} <= set(suite_lookup):
            max_speedup = max(suite_lookup.values())
            min_speedup = min(suite_lookup.values())
            q70_is_fastest = suite_lookup["static_q70"] == max_speedup
            dynamic_is_slowest = suite_lookup["dynamic_uniform"] == min_speedup

        suite_success_mask = (
            group["runtime_speedup_mean"].fillna(0.0).gt(PAIRWISE_CHAIN_RUNTIME_SPEEDUP_MIN)
            & group["asr_retention_mean"].fillna(0.0).ge(PAIRWISE_CHAIN_ASR_RETENTION_MIN)
        )
        suite_success_count = int(suite_success_mask.sum())
        average_runtime_speedup = float(group["runtime_speedup_mean"].mean(skipna=True))
        average_estimator_speedup = float(group["estimator_speedup_mean"].mean(skipna=True))
        average_asr_retention = float(group["asr_retention_mean"].mean(skipna=True))
        record = {
            "source_model": source_model,
            "target_model": target_model,
            "pair_label": pair_label,
            "suite_count": int(len(group)),
            "suite_success_count": suite_success_count,
            "average_runtime_speedup": average_runtime_speedup,
            "average_estimator_speedup": average_estimator_speedup,
            "average_asr_retention": average_asr_retention,
            "q70_is_fastest": bool(q70_is_fastest),
            "dynamic_is_slowest": bool(dynamic_is_slowest),
            "holm_significant_suite_count": int(group["runtime_faster_significant"].fillna(False).sum()),
        }
        record["successful_chain"] = bool(
            record["suite_count"] == len(MAIN_SUITES)
            and average_runtime_speedup > PAIRWISE_CHAIN_RUNTIME_SPEEDUP_MIN
            and average_asr_retention >= PAIRWISE_CHAIN_ASR_RETENTION_MIN
            and suite_success_count >= PAIRWISE_CHAIN_SUITE_SUCCESS_MIN
        )
        records.append(record)

    return pd.DataFrame.from_records(records).sort_values(["source_model", "target_model"]).reset_index(drop=True)


def _wilson_ci(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z_value = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    proportion = successes / total
    denominator = 1.0 + (z_value**2 / total)
    centre = proportion + z_value**2 / (2.0 * total)
    margin = z_value * math.sqrt((proportion * (1.0 - proportion) / total) + (z_value**2 / (4.0 * total**2)))
    return (centre - margin) / denominator, (centre + margin) / denominator


def _bootstrap_median_ci(values: np.ndarray, seed: int = 42, samples: int = 2000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    medians = []
    for _ in range(samples):
        sample = rng.choice(values, size=values.size, replace=True)
        medians.append(float(np.median(sample)))
    low, high = np.quantile(np.asarray(medians), [0.025, 0.975])
    return float(low), float(high)


def build_pairwise_claim_check(chain_summary: pd.DataFrame) -> dict[str, Any]:
    if chain_summary.empty:
        return {
            "tested_chain_count": 0,
            "successful_chain_count": 0,
            "successful_chain_fraction": float("nan"),
            "successful_chain_fraction_wilson_ci": [float("nan"), float("nan")],
            "median_runtime_speedup": float("nan"),
            "median_runtime_speedup_bootstrap_ci": [float("nan"), float("nan")],
            "median_asr_retention": float("nan"),
            "median_asr_retention_bootstrap_ci": [float("nan"), float("nan")],
            "eligible_for_tested_backbones_claim": False,
            "paper_positioning": "single_chain_headline_only",
        }

    tested_chain_count = int(len(chain_summary))
    successful_chain_count = int(chain_summary["successful_chain"].fillna(False).sum())
    successful_chain_fraction = successful_chain_count / tested_chain_count
    successful_wilson = _wilson_ci(successful_chain_count, tested_chain_count)
    runtime_values = chain_summary["average_runtime_speedup"].dropna().to_numpy(dtype=np.float64)
    retention_values = chain_summary["average_asr_retention"].dropna().to_numpy(dtype=np.float64)
    median_runtime_speedup = float(np.median(runtime_values)) if runtime_values.size else float("nan")
    median_asr_retention = float(np.median(retention_values)) if retention_values.size else float("nan")
    runtime_ci = _bootstrap_median_ci(runtime_values)
    retention_ci = _bootstrap_median_ci(retention_values)
    no_chain_below_floor = bool(chain_summary["average_asr_retention"].fillna(0.0).ge(0.60).all())
    eligible = bool(
        tested_chain_count == 12
        and successful_chain_count >= 10
        and median_runtime_speedup > 1.0
        and median_asr_retention >= 0.75
        and no_chain_below_floor
    )
    paper_positioning = (
        "tested_backbones_claim"
        if eligible
        else "single_chain_headline_plus_mixed_pairwise_validation"
    )
    return {
        "tested_chain_count": tested_chain_count,
        "successful_chain_count": successful_chain_count,
        "successful_chain_fraction": successful_chain_fraction,
        "successful_chain_fraction_wilson_ci": list(successful_wilson),
        "median_runtime_speedup": median_runtime_speedup,
        "median_runtime_speedup_bootstrap_ci": list(runtime_ci),
        "median_asr_retention": median_asr_retention,
        "median_asr_retention_bootstrap_ci": list(retention_ci),
        "eligible_for_tested_backbones_claim": eligible,
        "paper_positioning": paper_positioning,
    }


def write_pairwise_bundle(
    base_config: str,
    study_overrides: dict[str, str | Path],
    output_dir: str | Path,
) -> PairwiseBundleOutputs:
    tradeoff_frames: dict[str, pd.DataFrame] = {}
    for study_name, override_path in study_overrides.items():
        _, tradeoff = load_pairwise_tradeoff(base_config, override_path)
        tradeoff_frames[study_name] = tradeoff

    chain_suite_tradeoff = combine_chain_suite_tradeoff(tradeoff_frames)
    chain_summary = summarize_chain_level(chain_suite_tradeoff)
    claim_check = build_pairwise_claim_check(chain_summary)

    target = ensure_dir(resolve_project_path(output_dir))
    chain_suite_path = target / "chain_suite_tradeoff.csv"
    chain_summary_path = target / "chain_summary.csv"
    claim_path = target / "claim_check.json"
    manifest_path = target / "pairwise_bundle_manifest.json"

    chain_suite_tradeoff.to_csv(chain_suite_path, index=False)
    chain_summary.to_csv(chain_summary_path, index=False)
    write_json(claim_path, claim_check)
    write_json(
        manifest_path,
        {
            "chain_suite_tradeoff": str(chain_suite_path),
            "chain_summary": str(chain_summary_path),
            "claim_check": str(claim_path),
        },
    )
    return PairwiseBundleOutputs(
        chain_suite_tradeoff=chain_suite_path,
        chain_summary=chain_summary_path,
        claim_check=claim_path,
    )


def load_pairwise_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    root = resolve_project_path(bundle_dir)
    return {
        "chain_suite_tradeoff": _load_optional_csv(root / "chain_suite_tradeoff.csv"),
        "chain_summary": _load_optional_csv(root / "chain_summary.csv"),
        "claim_check": read_json(root / "claim_check.json") if (root / "claim_check.json").exists() else {},
    }
