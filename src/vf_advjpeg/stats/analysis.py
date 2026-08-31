from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from vf_advjpeg.results import collect_result_frame
from vf_advjpeg.stats.baseline_viability import write_baseline_viability_outputs
from vf_advjpeg.study import study_metadata
from vf_advjpeg.utils.fs import ensure_dir, read_json, resolve_project_path, write_json


PRIMARY_STATIC_SUITES = ["static_q70", "static_q80", "static_q90"]
SUMMARY_METRICS = [
    "asr",
    "clean_accuracy",
    "jpeg_clean_accuracy",
    "clean_drop",
    "lpips",
    "ssim",
    "linf",
    "duration_sec",
    "mean_estimator_time",
]
GROUP_COLUMNS = [
    "study_id",
    "source_profile",
    "dataset",
    "model_family",
    "source_model",
    "defense_profile",
    "method",
    "target_model",
    "suite",
]
PAIR_METADATA_COLUMNS = [
    "study_id",
    "source_profile",
    "dataset",
    "model_family",
    "source_model",
    "defense_profile",
    "target_model",
    "suite",
    "seed",
]


def _analysis_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("analysis", {})


def _targets(config: dict[str, Any]) -> dict[str, float]:
    targets = dict(
        _analysis_config(config).get(
            "efficiency_targets",
            {
                "runtime_speedup_min": 3.0,
                "estimator_speedup_min": 3.0,
                "clean_drop_delta_max": 0.1,
                "asr_retention_min": 0.75,
            },
        )
    )
    if "clean_drop_delta_max" not in targets:
        targets["clean_drop_delta_max"] = float(targets.get("clean_drop_max", 0.1))
    return targets


def _ensure_metadata_columns(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    metadata = study_metadata(config)
    normalized = frame.copy()
    for column, default_value in metadata.items():
        if column not in normalized.columns:
            normalized[column] = default_value
        else:
            normalized[column] = normalized[column].fillna(default_value)
    return normalized


def _safe_ratio(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    result = np.full_like(numerator_array, np.nan, dtype=np.float64)
    mask = denominator_array > 0
    np.divide(numerator_array, denominator_array, out=result, where=mask)
    return result


def _mean_std(values: pd.Series) -> tuple[float, float]:
    if values.empty:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def _descriptive_ci(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean
    sem = stats.sem(values)
    margin = float(stats.t.ppf(0.975, values.size - 1) * sem)
    return mean - margin, mean + margin


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    indexed = [(index, value) for index, value in enumerate(p_values) if not np.isnan(value)]
    indexed = sorted(indexed, key=lambda item: item[1])
    adjusted = [float("nan")] * len(p_values)
    max_seen = 0.0
    total = len(indexed)
    for rank, (original_index, p_value) in enumerate(indexed):
        adjusted_value = min((total - rank) * p_value, 1.0)
        max_seen = max(max_seen, adjusted_value)
        adjusted[original_index] = max_seen
    return adjusted


def summarize_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        columns = GROUP_COLUMNS.copy()
        for metric in SUMMARY_METRICS:
            columns.extend([f"{metric}_mean", f"{metric}_std"])
        return pd.DataFrame(columns=columns)
    frame = _ensure_metadata_columns(frame, {})
    aggregations = {metric: ["mean", "std"] for metric in SUMMARY_METRICS}
    summary = frame.groupby(GROUP_COLUMNS).agg(aggregations)
    summary.columns = ["_".join(column).strip("_") for column in summary.columns]
    return summary.reset_index()


def build_pairwise_frame(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    frame = _ensure_metadata_columns(frame, config)
    baseline_method = _analysis_config(config).get("baseline_method", "baseline_strong")
    candidate_method = _analysis_config(config).get("candidate_method", "vf_advjpeg")
    targets = _targets(config)

    baseline = frame[frame["method"] == baseline_method].copy()
    candidate = frame[frame["method"] == candidate_method].copy()
    if baseline.empty or candidate.empty:
        return pd.DataFrame(
            columns=[
                "study_id",
                "source_profile",
                "dataset",
                "model_family",
                "source_model",
                "defense_profile",
                "target_model",
                "suite",
                "seed",
                "baseline_method",
                "candidate_method",
                "baseline_asr",
                "candidate_asr",
                "delta_asr",
                "asr_retention",
                "runtime_speedup",
                "estimator_speedup",
                "clean_drop_delta",
                "efficiency_win",
            ]
        )

    join_keys = PAIR_METADATA_COLUMNS.copy()
    metric_columns = ["asr", "duration_sec", "mean_estimator_time", "clean_drop", "lpips", "ssim", "linf"]
    baseline = baseline[join_keys + metric_columns].rename(columns={column: f"baseline_{column}" for column in metric_columns})
    candidate = candidate[join_keys + metric_columns].rename(columns={column: f"candidate_{column}" for column in metric_columns})
    pairs = baseline.merge(candidate, on=join_keys, how="inner")
    if pairs.empty:
        return pairs

    pairs["baseline_method"] = baseline_method
    pairs["candidate_method"] = candidate_method
    pairs["delta_asr"] = pairs["candidate_asr"] - pairs["baseline_asr"]
    pairs["asr_retention"] = _safe_ratio(pairs["candidate_asr"], pairs["baseline_asr"])
    pairs["runtime_speedup"] = _safe_ratio(pairs["baseline_duration_sec"], pairs["candidate_duration_sec"])
    pairs["estimator_speedup"] = _safe_ratio(
        pairs["baseline_mean_estimator_time"],
        pairs["candidate_mean_estimator_time"],
    )
    pairs["clean_drop_delta"] = (pairs["candidate_clean_drop"] - pairs["baseline_clean_drop"]).abs()
    pairs["runtime_delta"] = pairs["baseline_duration_sec"] - pairs["candidate_duration_sec"]
    pairs["estimator_delta"] = pairs["baseline_mean_estimator_time"] - pairs["candidate_mean_estimator_time"]
    pairs["clean_drop_guardrail"] = pairs["clean_drop_delta"] <= targets["clean_drop_delta_max"]
    pairs["efficiency_win"] = (
        (pairs["runtime_speedup"] >= targets["runtime_speedup_min"])
        & (pairs["estimator_speedup"] >= targets["estimator_speedup_min"])
        & (pairs["asr_retention"] >= targets["asr_retention_min"])
        & pairs["clean_drop_guardrail"]
    )
    return pairs


def summarize_efficiency_tradeoff(pairs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    pairs = _ensure_metadata_columns(pairs, config)
    if pairs.empty:
        return pd.DataFrame(
            columns=[
                "study_id",
                "source_profile",
                "dataset",
                "model_family",
                "source_model",
                "defense_profile",
                "target_model",
                "suite",
                "n",
                "baseline_asr_mean",
                "candidate_asr_mean",
                "delta_asr_mean",
                "delta_asr_ci_low",
                "delta_asr_ci_high",
                "asr_retention_mean",
                "runtime_speedup_mean",
                "estimator_speedup_mean",
                "clean_drop_delta_mean",
                "runtime_p_value",
                "runtime_p_value_holm",
                "runtime_faster_significant",
                "efficiency_win",
            ]
        )

    targets = _targets(config)
    records = []
    for (
        study_id,
        source_profile,
        dataset,
        model_family,
        source_model,
        defense_profile,
        target_model,
        suite,
    ), group in pairs.groupby(
        [
            "study_id",
            "source_profile",
            "dataset",
            "model_family",
            "source_model",
            "defense_profile",
            "target_model",
            "suite",
        ],
        sort=True,
    ):
        delta_asr = group["delta_asr"].dropna().to_numpy(dtype=np.float64)
        runtime_delta = group["runtime_delta"].dropna().to_numpy(dtype=np.float64)
        baseline_asr_mean, baseline_asr_std = _mean_std(group["baseline_asr"])
        candidate_asr_mean, candidate_asr_std = _mean_std(group["candidate_asr"])
        baseline_duration_mean, baseline_duration_std = _mean_std(group["baseline_duration_sec"])
        candidate_duration_mean, candidate_duration_std = _mean_std(group["candidate_duration_sec"])
        baseline_estimator_mean, baseline_estimator_std = _mean_std(group["baseline_mean_estimator_time"])
        candidate_estimator_mean, candidate_estimator_std = _mean_std(group["candidate_mean_estimator_time"])
        retention_mean, retention_std = _mean_std(group["asr_retention"].dropna())
        runtime_speedup_mean, runtime_speedup_std = _mean_std(group["runtime_speedup"].dropna())
        estimator_speedup_mean, estimator_speedup_std = _mean_std(group["estimator_speedup"].dropna())
        delta_asr_ci_low, delta_asr_ci_high = _descriptive_ci(delta_asr)

        if runtime_delta.size > 1:
            runtime_t_stat, runtime_p_value = stats.ttest_1samp(runtime_delta, popmean=0.0, alternative="greater")
        else:
            runtime_t_stat, runtime_p_value = float("nan"), float("nan")

        record = {
            "study_id": study_id,
            "source_profile": source_profile,
            "dataset": dataset,
            "model_family": model_family,
            "source_model": source_model,
            "defense_profile": defense_profile,
            "target_model": target_model,
            "suite": suite,
            "n": int(len(group)),
            "baseline_asr_mean": baseline_asr_mean,
            "baseline_asr_std": baseline_asr_std,
            "candidate_asr_mean": candidate_asr_mean,
            "candidate_asr_std": candidate_asr_std,
            "delta_asr_mean": float(delta_asr.mean()) if delta_asr.size else float("nan"),
            "delta_asr_std": float(delta_asr.std(ddof=1)) if delta_asr.size > 1 else float("nan"),
            "delta_asr_ci_low": delta_asr_ci_low,
            "delta_asr_ci_high": delta_asr_ci_high,
            "asr_retention_mean": retention_mean,
            "asr_retention_std": retention_std,
            "baseline_duration_sec_mean": baseline_duration_mean,
            "baseline_duration_sec_std": baseline_duration_std,
            "candidate_duration_sec_mean": candidate_duration_mean,
            "candidate_duration_sec_std": candidate_duration_std,
            "runtime_speedup_mean": runtime_speedup_mean,
            "runtime_speedup_std": runtime_speedup_std,
            "baseline_estimator_time_mean": baseline_estimator_mean,
            "baseline_estimator_time_std": baseline_estimator_std,
            "candidate_estimator_time_mean": candidate_estimator_mean,
            "candidate_estimator_time_std": candidate_estimator_std,
            "estimator_speedup_mean": estimator_speedup_mean,
            "estimator_speedup_std": estimator_speedup_std,
            "baseline_clean_drop_mean": float(group["baseline_clean_drop"].mean()),
            "candidate_clean_drop_mean": float(group["candidate_clean_drop"].mean()),
            "clean_drop_delta_mean": float(group["clean_drop_delta"].mean()),
            "baseline_lpips_mean": float(group["baseline_lpips"].mean()),
            "candidate_lpips_mean": float(group["candidate_lpips"].mean()),
            "baseline_ssim_mean": float(group["baseline_ssim"].mean()),
            "candidate_ssim_mean": float(group["candidate_ssim"].mean()),
            "runtime_delta_mean": float(runtime_delta.mean()) if runtime_delta.size else float("nan"),
            "runtime_t_stat": float(runtime_t_stat),
            "runtime_p_value": float(runtime_p_value),
        }
        record["efficiency_win"] = bool(
            (runtime_speedup_mean >= targets["runtime_speedup_min"])
            and (estimator_speedup_mean >= targets["estimator_speedup_min"])
            and (retention_mean >= targets["asr_retention_min"])
            and (record["clean_drop_delta_mean"] <= targets["clean_drop_delta_max"])
        )
        records.append(record)

    result = pd.DataFrame.from_records(records)
    adjusted = _holm_bonferroni(result["runtime_p_value"].tolist())
    result["runtime_p_value_holm"] = adjusted
    result["runtime_faster_significant"] = result["runtime_p_value_holm"] < 0.05
    return result


def build_efficiency_target_report(tradeoff: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    analysis_cfg = _analysis_config(config)
    targets = _targets(config)
    main_suites = analysis_cfg.get("main_suites", PRIMARY_STATIC_SUITES)
    main = tradeoff[tradeoff["suite"].isin(main_suites)].copy()
    if main.empty:
        return {
            "main_suites": main_suites,
            "suite_count": 0,
            "failed_suites": [],
            "average_asr_retention": float("nan"),
            "average_runtime_speedup": float("nan"),
            "average_estimator_speedup": float("nan"),
            "recovery_required": False,
            "targets": targets,
        }

    failed_mask = main["asr_retention_mean"].isna() | (main["asr_retention_mean"] < targets["asr_retention_min"])
    failed_suites = main.loc[failed_mask, "suite"].tolist()
    report = {
        "main_suites": main_suites,
        "suite_count": int(len(main)),
        "failed_suites": failed_suites,
        "average_asr_retention": float(main["asr_retention_mean"].mean(skipna=True)),
        "average_runtime_speedup": float(main["runtime_speedup_mean"].mean(skipna=True)),
        "average_estimator_speedup": float(main["estimator_speedup_mean"].mean(skipna=True)),
        "efficiency_wins": int(main["efficiency_win"].sum()),
        "targets": targets,
    }
    report["recovery_required"] = len(failed_suites) > (len(main_suites) / 2.0)
    return report


def write_analysis_outputs(config: dict[str, Any]) -> dict[str, str]:
    frame = collect_result_frame(config)
    analysis_root = ensure_dir(resolve_project_path(config["paths"]["analysis_root"]))
    raw_path = analysis_root / "all_results.csv"
    summary_path = analysis_root / "summary.csv"
    pairs_path = analysis_root / "efficiency_pairs.csv"
    tradeoff_path = analysis_root / "efficiency_tradeoff.csv"
    target_report_path = analysis_root / "efficiency_targets.json"
    viability_path = analysis_root / "baseline_viability_report.csv"
    viability_json_path = analysis_root / "baseline_viability_report.json"
    manifest_path = analysis_root / "analysis_manifest.json"

    frame.to_csv(raw_path, index=False)
    summary = summarize_results(frame)
    summary.to_csv(summary_path, index=False)
    pairs = build_pairwise_frame(frame, config)
    pairs.to_csv(pairs_path, index=False)
    tradeoff = summarize_efficiency_tradeoff(pairs, config)
    tradeoff.to_csv(tradeoff_path, index=False)
    targets = build_efficiency_target_report(tradeoff, config)
    write_json(target_report_path, targets)
    write_baseline_viability_outputs(frame, viability_path, viability_json_path)

    manifest = {
        "raw_results": str(raw_path),
        "summary": str(summary_path),
        "efficiency_pairs": str(pairs_path),
        "efficiency_tradeoff": str(tradeoff_path),
        "efficiency_targets": str(target_report_path),
        "baseline_viability_report": str(viability_path),
        "baseline_viability_summary": str(viability_json_path),
        "record_count": int(len(frame)),
    }
    write_json(manifest_path, manifest)
    return {key: str(value) for key, value in manifest.items()}


def _load_optional_csv(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_analysis(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    analysis_root = resolve_project_path(config["paths"]["analysis_root"])
    return {
        "raw": _load_optional_csv(analysis_root / "all_results.csv"),
        "summary": _load_optional_csv(analysis_root / "summary.csv"),
        "efficiency_pairs": _load_optional_csv(analysis_root / "efficiency_pairs.csv"),
        "efficiency_tradeoff": _load_optional_csv(analysis_root / "efficiency_tradeoff.csv"),
    }


def load_efficiency_targets(config: dict[str, Any]) -> dict[str, Any]:
    analysis_root = resolve_project_path(config["paths"]["analysis_root"])
    return read_json(analysis_root / "efficiency_targets.json")
