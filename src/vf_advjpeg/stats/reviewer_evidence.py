from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from vf_advjpeg.utils.fs import ensure_dir, resolve_project_path, sha256_file, write_json


DEFAULT_REVIEWER_STUDY_ROOTS: dict[str, Path] = {
    "reviewer_cifar10": Path("results/reviewer_cifar10"),
    "reviewer_imagenet": Path("results/reviewer_imagenet"),
    "reviewer_defended": Path("results/reviewer_defended"),
    "reviewer_ablation": Path("results/reviewer_ablation"),
}

TRADEOFF_COLUMNS = [
    "study_id",
    "dataset",
    "model_family",
    "source_model",
    "defense_profile",
    "target_model",
    "suite",
    "n",
    "baseline_asr_mean",
    "candidate_asr_mean",
    "asr_retention_mean",
    "runtime_speedup_mean",
    "estimator_speedup_mean",
    "clean_drop_delta_mean",
    "runtime_p_value_holm",
]
SUMMARY_COLUMNS = [
    "study_id",
    "dataset",
    "model_family",
    "defense_profile",
    "target_model_count",
    "suite_count",
    "row_count",
    "mean_runtime_speedup",
    "mean_asr_retention",
]
METHOD_COMPARISON_COLUMNS = [
    "method",
    "row_count",
    "runtime_speedup_vs_strong_mean",
    "estimator_speedup_vs_strong_mean",
    "retention_vs_strong_mean",
]
ABLATION_COLUMNS = [
    "study_id",
    "axis",
    "setting",
    "suite",
    "n",
    "runtime_speedup_mean",
    "asr_retention_mean",
]
SIGNATURE_FILES = [
    "analysis/reviewer_evidence_summary.csv",
    "analysis/reviewer_evidence_tradeoff.csv",
    "analysis/reviewer_method_comparison.csv",
    "analysis/reviewer_evidence_coverage.json",
    "analysis/reviewer_ablation_summary.csv",
]


@dataclass(slots=True)
class ReviewerEvidenceOutputs:
    root: Path
    summary: Path
    tradeoff: Path
    method_comparison: Path
    coverage: Path
    ablation: Path
    signature: Path


def _resolve_root(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_project_path(candidate)


def _read_tradeoff(study_id: str, root: Path) -> pd.DataFrame:
    path = root / "analysis" / "efficiency_tradeoff.csv"
    if not path.exists():
        return pd.DataFrame(columns=TRADEOFF_COLUMNS)
    frame = pd.read_csv(path)
    for column in TRADEOFF_COLUMNS:
        if column not in frame.columns:
            if column == "study_id":
                frame[column] = study_id
            elif column == "dataset":
                frame[column] = study_id.replace("reviewer_", "")
            elif column in {"model_family", "source_model", "defense_profile", "target_model", "suite"}:
                frame[column] = "unknown"
            else:
                frame[column] = float("nan")
    frame["study_id"] = frame["study_id"].fillna(study_id)
    return frame[TRADEOFF_COLUMNS].copy()


def _combine_tradeoff(study_roots: Mapping[str, str | Path]) -> pd.DataFrame:
    frames = []
    for study_id, path in study_roots.items():
        frame = _read_tradeoff(study_id, _resolve_root(path))
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=TRADEOFF_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    return combined.dropna(subset=["study_id"], how="all")


def _summarize_tradeoff(tradeoff: pd.DataFrame) -> pd.DataFrame:
    if tradeoff.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    records: list[dict[str, Any]] = []
    group_columns = ["study_id", "dataset", "model_family", "defense_profile"]
    for keys, group in tradeoff.groupby(group_columns, sort=True):
        study_id, dataset, model_family, defense_profile = keys
        records.append(
            {
                "study_id": study_id,
                "dataset": dataset,
                "model_family": model_family,
                "defense_profile": defense_profile,
                "target_model_count": int(group["target_model"].nunique(dropna=True)),
                "suite_count": int(group["suite"].nunique(dropna=True)),
                "row_count": int(len(group)),
                "mean_runtime_speedup": float(group["runtime_speedup_mean"].mean(skipna=True)),
                "mean_asr_retention": float(group["asr_retention_mean"].mean(skipna=True)),
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def _read_summary(study_id: str, root: Path) -> pd.DataFrame:
    path = root / "analysis" / "summary.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    for column in [
        "study_id",
        "method",
        "target_model",
        "suite",
        "asr_mean",
        "duration_sec_mean",
        "mean_estimator_time_mean",
    ]:
        if column not in frame.columns:
            if column == "study_id":
                frame[column] = study_id
            elif column in {"method", "target_model", "suite"}:
                frame[column] = "unknown"
            else:
                frame[column] = float("nan")
    frame["study_id"] = frame["study_id"].fillna(study_id)
    return frame


def _combine_summaries(study_roots: Mapping[str, str | Path]) -> pd.DataFrame:
    frames = []
    for study_id, path in study_roots.items():
        frame = _read_summary(study_id, _resolve_root(path))
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _method_comparison(summaries: pd.DataFrame) -> pd.DataFrame:
    if summaries.empty:
        return pd.DataFrame(columns=METHOD_COMPARISON_COLUMNS)
    index_columns = ["study_id", "target_model", "suite"]
    baseline = summaries[summaries["method"] == "baseline_strong"][
        index_columns + ["asr_mean", "duration_sec_mean", "mean_estimator_time_mean"]
    ].rename(
        columns={
            "asr_mean": "baseline_asr_mean",
            "duration_sec_mean": "baseline_duration_sec_mean",
            "mean_estimator_time_mean": "baseline_estimator_time_mean",
        }
    )
    records: list[dict[str, Any]] = []
    for method in ["baseline_diffjpeg", "vf_advjpeg"]:
        candidate = summaries[summaries["method"] == method][
            index_columns + ["asr_mean", "duration_sec_mean", "mean_estimator_time_mean"]
        ].rename(
            columns={
                "asr_mean": "candidate_asr_mean",
                "duration_sec_mean": "candidate_duration_sec_mean",
                "mean_estimator_time_mean": "candidate_estimator_time_mean",
            }
        )
        merged = baseline.merge(candidate, on=index_columns, how="inner")
        if merged.empty:
            continue
        records.append(
            {
                "method": method,
                "row_count": int(len(merged)),
                "runtime_speedup_vs_strong_mean": float(
                    (merged["baseline_duration_sec_mean"] / merged["candidate_duration_sec_mean"]).mean(skipna=True)
                ),
                "estimator_speedup_vs_strong_mean": float(
                    (merged["baseline_estimator_time_mean"] / merged["candidate_estimator_time_mean"]).mean(skipna=True)
                ),
                "retention_vs_strong_mean": float(
                    (merged["candidate_asr_mean"] / merged["baseline_asr_mean"]).mean(skipna=True)
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=METHOD_COMPARISON_COLUMNS)


def _empty_ablation() -> pd.DataFrame:
    return pd.DataFrame(columns=ABLATION_COLUMNS)


def _read_ablation(study_roots: Mapping[str, str | Path]) -> pd.DataFrame:
    root_value = study_roots.get("reviewer_ablation")
    if root_value is None:
        return _empty_ablation()
    path = _resolve_root(root_value) / "analysis" / "reviewer_ablation_summary.csv"
    if not path.exists():
        return _empty_ablation()
    frame = pd.read_csv(path)
    for column in ABLATION_COLUMNS:
        if column not in frame.columns:
            if column == "study_id":
                frame[column] = "reviewer_ablation"
            elif column in {"axis", "setting", "suite"}:
                frame[column] = "unknown"
            else:
                frame[column] = float("nan")
    frame["study_id"] = frame["study_id"].fillna("reviewer_ablation")
    return frame[ABLATION_COLUMNS].copy()


def _status(condition: bool, partial: bool) -> str:
    if condition:
        return "full"
    return "partial" if partial else "not_addressed"


def _coverage_payload(
    tradeoff: pd.DataFrame,
    study_roots: Mapping[str, str | Path],
    ablation: pd.DataFrame | None = None,
) -> dict[str, Any]:
    studies_present = set(tradeoff["study_id"].dropna().astype(str).unique()) if not tradeoff.empty else set()
    datasets_present = set(tradeoff["dataset"].dropna().astype(str).unique()) if not tradeoff.empty else set()
    model_families = set(tradeoff["model_family"].dropna().astype(str).str.lower().unique()) if not tradeoff.empty else set()
    defense_profiles = set(tradeoff["defense_profile"].dropna().astype(str).str.lower().unique()) if not tradeoff.empty else set()
    row_count = int(len(tradeoff))
    has_any = row_count > 0
    has_cifar = "cifar10" in datasets_present or "reviewer_cifar10" in studies_present
    has_imagenet = "imagenet1k_subset" in datasets_present or "imagenet" in datasets_present or "reviewer_imagenet" in studies_present
    model_names = []
    for column in ["source_model", "target_model"]:
        if column in tradeoff.columns:
            model_names.extend(tradeoff[column].dropna().astype(str).tolist())
    has_transformer = "transformer" in model_families or any(
        token in str(value).lower()
        for value in model_names
        for token in ["vit", "deit", "swin"]
    )
    has_defended = "reviewer_defended" in studies_present or any("robust" in item for item in defense_profiles)
    methods = set()
    for study_id, root in study_roots.items():
        path = _resolve_root(root) / "analysis" / "all_results.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if "method" in frame.columns:
                methods.update(frame["method"].dropna().astype(str).tolist())
        if study_id == "reviewer_cifar10" and "reviewer_cifar10" in studies_present:
            methods.update(["baseline_strong", "vf_advjpeg"])
    has_diffjpeg = "baseline_diffjpeg" in methods
    if tradeoff.empty:
        max_n = 0
        max_suite_count = 0
    else:
        n_values = pd.to_numeric(tradeoff["n"], errors="coerce").fillna(0)
        max_n = int(n_values.max()) if not n_values.empty else 0
        suite_counts = tradeoff.groupby("study_id")["suite"].nunique(dropna=True)
        max_suite_count = int(suite_counts.max()) if not suite_counts.empty else 0
    full_suites = max_suite_count >= 4 and max_n >= 5
    required_ablation_axes = {"calibration_size", "amplitude_bucket_count", "neighbor_span"}
    ablation_axes = set(ablation["axis"].dropna().astype(str).tolist()) if ablation is not None and not ablation.empty else set()
    has_ablation = required_ablation_axes.issubset(ablation_axes)

    return {
        "row_count": row_count,
        "studies_present": sorted(studies_present),
        "datasets_present": sorted(datasets_present),
        "items": {
            "broader_datasets": {
                "status": _status(has_cifar and has_imagenet, has_any),
                "evidence": sorted(datasets_present),
            },
            "modern_architectures": {
                "status": _status(has_transformer, has_any),
                "evidence": sorted(model_families),
            },
            "diffjpeg_baseline": {
                "status": _status(has_diffjpeg, has_any),
                "evidence": sorted(methods),
            },
            "defended_models": {
                "status": _status(has_defended, has_any),
                "evidence": sorted(defense_profiles),
            },
            "ablation_stability": {
                "status": _status(has_ablation, has_any),
                "evidence": (
                    sorted(
                        {
                            f"{row.axis}:{row.setting}"
                            for row in ablation.itertuples()
                            if str(row.axis) and str(row.setting)
                        }
                    )
                    if ablation is not None and not ablation.empty
                    else []
                ),
            },
            "multi_seed_ci": {
                "status": _status(bool(full_suites), has_any),
                "evidence": {"max_n": max_n},
            },
        },
    }


def reviewer_source_signature(output_root: str | Path) -> dict[str, Any]:
    root = _resolve_root(output_root)
    files: dict[str, Any] = {}
    for relative in SIGNATURE_FILES:
        path = root / relative
        if not path.exists():
            continue
        payload: dict[str, Any] = {
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        if path.suffix.lower() == ".csv":
            payload["rows"] = int(len(pd.read_csv(path)))
        files[relative] = payload
    return {"root": str(root), "files": files}


def aggregate_reviewer_evidence(
    study_roots: Mapping[str, str | Path] = DEFAULT_REVIEWER_STUDY_ROOTS,
    output_root: str | Path = "results/reviewer_evidence",
) -> ReviewerEvidenceOutputs:
    root = ensure_dir(_resolve_root(output_root))
    analysis_root = ensure_dir(root / "analysis")

    tradeoff = _combine_tradeoff(study_roots)
    summary = _summarize_tradeoff(tradeoff)
    method_comparison = _method_comparison(_combine_summaries(study_roots))
    ablation = _read_ablation(study_roots)
    coverage = _coverage_payload(tradeoff, study_roots, ablation)

    tradeoff_path = analysis_root / "reviewer_evidence_tradeoff.csv"
    summary_path = analysis_root / "reviewer_evidence_summary.csv"
    method_comparison_path = analysis_root / "reviewer_method_comparison.csv"
    ablation_path = analysis_root / "reviewer_ablation_summary.csv"
    coverage_path = analysis_root / "reviewer_evidence_coverage.json"
    signature_path = analysis_root / "reviewer_source_signature.json"

    tradeoff.to_csv(tradeoff_path, index=False)
    summary.to_csv(summary_path, index=False)
    method_comparison.to_csv(method_comparison_path, index=False)
    ablation.to_csv(ablation_path, index=False)
    write_json(coverage_path, coverage)
    signature = reviewer_source_signature(root)
    write_json(signature_path, signature)

    return ReviewerEvidenceOutputs(
        root=root,
        summary=summary_path,
        tradeoff=tradeoff_path,
        method_comparison=method_comparison_path,
        coverage=coverage_path,
        ablation=ablation_path,
        signature=signature_path,
    )
