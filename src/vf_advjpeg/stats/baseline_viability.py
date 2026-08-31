from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vf_advjpeg.utils.fs import write_json


GROUP_COLUMNS = [
    "study_id",
    "source_profile",
    "dataset",
    "model_family",
    "source_model",
    "defense_profile",
    "target_model",
    "suite",
]
REQUIRED_RESULT_COLUMNS = [
    *GROUP_COLUMNS,
    "method",
    "seed",
    "adversarial_successes",
    "eligible_examples",
    "asr",
]


@dataclass(frozen=True)
class ViabilityThresholds:
    min_seeds: int = 5
    standard_min_success_total: int = 10
    standard_min_asr_mean: float = 0.10
    standard_max_retention_gap: float = 0.15
    defended_min_success_total: int = 20
    defended_min_asr_mean: float = 0.05
    defended_max_retention_gap: float = 0.20


def _is_defended(row: pd.Series) -> bool:
    model_family = str(row.get("model_family", "")).lower()
    defense_profile = str(row.get("defense_profile", "")).lower()
    target_model = str(row.get("target_model", "")).lower()
    return "robustbench" in model_family or "robust" in defense_profile or "robustbench" in target_model


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _seed_count(baseline: pd.DataFrame, candidate: pd.DataFrame) -> int:
    baseline_seeds = set(pd.to_numeric(baseline["seed"], errors="coerce").dropna().astype(int))
    candidate_seeds = set(pd.to_numeric(candidate["seed"], errors="coerce").dropna().astype(int))
    return int(len(baseline_seeds.intersection(candidate_seeds)))


def _low_seed_count(baseline_asr: pd.Series, floor: float) -> int:
    return int((baseline_asr < floor).sum())


def summarize_baseline_viability(
    frame: pd.DataFrame,
    baseline_method: str = "baseline_strong",
    candidate_method: str = "vf_advjpeg",
) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_RESULT_COLUMNS if column not in frame.columns]
    if frame.empty or missing_columns:
        return pd.DataFrame(columns=[*GROUP_COLUMNS, "baseline_viability_pass", "baseline_viability_reason"])

    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(GROUP_COLUMNS, dropna=False, sort=True):
        key_payload = dict(zip(GROUP_COLUMNS, keys))
        baseline = group[group["method"] == baseline_method].copy()
        candidate = group[group["method"] == candidate_method].copy()
        if baseline.empty or candidate.empty:
            continue

        baseline = baseline.sort_values("seed")
        candidate = candidate.sort_values("seed")
        baseline_asr = pd.to_numeric(baseline["asr"], errors="coerce").fillna(0.0)
        candidate_asr = pd.to_numeric(candidate["asr"], errors="coerce").fillna(0.0)
        baseline_successes = pd.to_numeric(baseline["adversarial_successes"], errors="coerce").fillna(0)
        candidate_successes = pd.to_numeric(candidate["adversarial_successes"], errors="coerce").fillna(0)
        eligible = pd.to_numeric(baseline["eligible_examples"], errors="coerce").fillna(0)

        pair = baseline[["seed", "asr"]].rename(columns={"asr": "baseline_asr"}).merge(
            candidate[["seed", "asr"]].rename(columns={"asr": "candidate_asr"}),
            on="seed",
            how="inner",
        )
        denominator = pd.to_numeric(pair["baseline_asr"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        numerator = pd.to_numeric(pair["candidate_asr"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        per_seed = np.full(len(pair), np.nan, dtype=float)
        mask = denominator > 0
        np.divide(numerator, denominator, out=per_seed, where=mask)
        per_seed_retention = float(np.nanmean(per_seed)) if np.isfinite(per_seed).any() else float("nan")
        aggregate_retention = _safe_ratio(float(candidate_asr.mean()), float(baseline_asr.mean()))
        if np.isfinite(per_seed_retention) and np.isfinite(aggregate_retention):
            retention_gap = float(abs(per_seed_retention - aggregate_retention))
        else:
            retention_gap = float("nan")

        records.append(
            {
                **key_payload,
                "paper_visible_retention_row": True,
                "n": _seed_count(baseline, candidate),
                "eligible_total": int(eligible.sum()),
                "baseline_success_total": int(baseline_successes.sum()),
                "candidate_success_total": int(candidate_successes.sum()),
                "baseline_asr_mean": float(baseline_asr.mean()),
                "candidate_asr_mean": float(candidate_asr.mean()),
                "zero_baseline_seed_count": int((baseline_asr == 0).sum()),
                "low_baseline_seed_count": 0,
                "baseline_seed_asr_values": ";".join(f"{value:.12g}" for value in baseline_asr.to_list()),
                "per_seed_asr_retention_mean": per_seed_retention,
                "aggregate_asr_retention": aggregate_retention,
                "retention_metric_gap": retention_gap,
                "baseline_viability_pass": False,
                "baseline_viability_reason": "not_evaluated",
            }
        )
    return pd.DataFrame.from_records(records)


def evaluate_baseline_viability(
    report: pd.DataFrame,
    thresholds: ViabilityThresholds = ViabilityThresholds(),
) -> pd.DataFrame:
    evaluated = report.copy()
    if evaluated.empty:
        for column in ["baseline_viability_pass", "baseline_viability_reason"]:
            if column not in evaluated.columns:
                evaluated[column] = pd.Series(dtype=object)
        return evaluated

    passes: list[bool] = []
    reasons: list[str] = []
    low_counts: list[int] = []
    for _, row in evaluated.iterrows():
        defended = _is_defended(row)
        min_success = thresholds.defended_min_success_total if defended else thresholds.standard_min_success_total
        min_asr = thresholds.defended_min_asr_mean if defended else thresholds.standard_min_asr_mean
        max_gap = thresholds.defended_max_retention_gap if defended else thresholds.standard_max_retention_gap
        baseline_asr_mean = float(row["baseline_asr_mean"])
        retention_gap = float(row["retention_metric_gap"])
        failures: list[str] = []

        if int(row["n"]) < thresholds.min_seeds:
            failures.append("n")
        if int(row["zero_baseline_seed_count"]) != 0:
            failures.append("zero_baseline_seed_count")
        if int(row["baseline_success_total"]) < min_success:
            failures.append("baseline_success_total")
        if baseline_asr_mean < min_asr:
            failures.append("baseline_asr_mean")
        if not np.isfinite(retention_gap) or retention_gap > max_gap:
            failures.append("retention_metric_gap")

        low_count = int(row.get("low_baseline_seed_count", 0))
        if "baseline_seed_asr_values" in row and isinstance(row["baseline_seed_asr_values"], str):
            values = [float(item) for item in row["baseline_seed_asr_values"].split(";") if item]
            low_count = _low_seed_count(pd.Series(values), min_asr)
        elif baseline_asr_mean < min_asr:
            low_count = int(row["n"])
        low_counts.append(low_count)
        passes.append(not failures)
        reasons.append("pass" if not failures else ",".join(failures))

    evaluated["low_baseline_seed_count"] = low_counts
    evaluated["baseline_viability_pass"] = passes
    evaluated["baseline_viability_reason"] = reasons
    return evaluated


def baseline_viability_payload(report: pd.DataFrame, thresholds: ViabilityThresholds = ViabilityThresholds()) -> dict[str, Any]:
    failed_rows = []
    if not report.empty and "baseline_viability_pass" in report.columns:
        failed = report[~report["baseline_viability_pass"].astype(bool)].copy()
        for row in failed.itertuples():
            failed_rows.append(
                {
                    "study_id": str(row.study_id),
                    "target_model": str(row.target_model),
                    "suite": str(row.suite),
                    "baseline_viability_reason": str(row.baseline_viability_reason),
                }
            )
    return {
        "ok": bool((not report.empty) and report["baseline_viability_pass"].astype(bool).all()),
        "row_count": int(len(report)),
        "thresholds": asdict(thresholds),
        "failed_rows": failed_rows,
    }


def write_baseline_viability_outputs(
    frame: pd.DataFrame,
    csv_path: str | Path,
    json_path: str | Path,
    baseline_method: str = "baseline_strong",
    candidate_method: str = "vf_advjpeg",
    thresholds: ViabilityThresholds = ViabilityThresholds(),
) -> pd.DataFrame:
    report = evaluate_baseline_viability(
        summarize_baseline_viability(frame, baseline_method=baseline_method, candidate_method=candidate_method),
        thresholds=thresholds,
    )
    target_csv = Path(csv_path)
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(target_csv, index=False)
    write_json(json_path, baseline_viability_payload(report, thresholds=thresholds))
    return report
