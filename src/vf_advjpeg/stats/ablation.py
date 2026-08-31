from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from vf_advjpeg.config import deep_merge
from vf_advjpeg.utils.fs import ensure_dir, resolve_project_path, write_json


ABLATION_SUMMARY_COLUMNS = [
    "study_id",
    "axis",
    "setting",
    "suite",
    "n",
    "runtime_speedup_mean",
    "asr_retention_mean",
]


@dataclass(frozen=True, slots=True)
class AblationVariant:
    axis: str
    setting: str
    config: dict[str, Any]


def _variant_slug(axis: str, setting: str) -> str:
    return f"{axis}_{setting}".replace(".", "p").replace(",", "_").replace(" ", "_").replace("[", "").replace("]", "")


def _artifact_suffix(axis: str, setting: str) -> str:
    return _variant_slug(axis, setting)


def _paths_for_variant(base_config: dict[str, Any], axis: str, setting: str) -> dict[str, str]:
    suffix = _artifact_suffix(axis, setting)
    return {
        "artifact_root": f"artifacts/reviewer_ablation/{suffix}",
        "vf_artifact": f"artifacts/reviewer_ablation/{suffix}/vf/vf_first_pass.json",
        "plot_root": f"results/reviewer_ablation/{suffix}/plots",
        "analysis_root": f"results/reviewer_ablation/{suffix}/analysis",
        "run_root": f"results/reviewer_ablation/{suffix}/runs",
    }


def build_ablation_variants(config: dict[str, Any]) -> list[AblationVariant]:
    ablation_cfg = config.get("reviewer_ablation", {})
    variants: list[AblationVariant] = []

    for calibration_size in ablation_cfg.get("calibration_sizes", []):
        setting = f"n{int(calibration_size)}"
        override = {
            "paths": _paths_for_variant(config, "calibration_size", setting),
            "data": {"calibration_size": int(calibration_size)},
        }
        variants.append(AblationVariant("calibration_size", setting, deep_merge(config, override)))

    for bucket_edges in ablation_cfg.get("amplitude_bucket_sets", []):
        edges = [float(item) for item in bucket_edges]
        setting = f"{len(edges) + 1}_buckets"
        override = {
            "paths": _paths_for_variant(config, "amplitude_bucket_count", setting),
            "vf": {"amplitude_buckets": edges},
        }
        variants.append(AblationVariant("amplitude_bucket_count", setting, deep_merge(config, override)))

    for neighbor_span in ablation_cfg.get("neighbor_spans", []):
        setting = f"k{int(neighbor_span)}"
        override = {
            "paths": _paths_for_variant(config, "neighbor_span", setting),
            "attack": {"eot_neighbor_span": int(neighbor_span)},
        }
        variants.append(AblationVariant("neighbor_span", setting, deep_merge(config, override)))

    return variants


def summarize_ablation_variants(
    variants: list[AblationVariant],
    output_root: str | Path = "results/reviewer_ablation",
) -> Path:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        tradeoff_path = resolve_project_path(variant.config["paths"]["analysis_root"]) / "efficiency_tradeoff.csv"
        if not tradeoff_path.exists():
            continue
        tradeoff = pd.read_csv(tradeoff_path)
        for row in tradeoff.itertuples(index=False):
            rows.append(
                {
                    "study_id": "reviewer_ablation",
                    "axis": variant.axis,
                    "setting": variant.setting,
                    "suite": str(getattr(row, "suite")),
                    "n": int(getattr(row, "n")),
                    "runtime_speedup_mean": float(getattr(row, "runtime_speedup_mean")),
                    "asr_retention_mean": float(getattr(row, "asr_retention_mean")),
                }
            )

    analysis_root = ensure_dir(resolve_project_path(output_root) / "analysis")
    summary = pd.DataFrame.from_records(rows, columns=ABLATION_SUMMARY_COLUMNS)
    summary_path = analysis_root / "reviewer_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "variant_count": len(variants),
        "completed_variant_count": int(summary[["axis", "setting"]].drop_duplicates().shape[0]) if not summary.empty else 0,
        "row_count": int(len(summary)),
        "summary": str(summary_path),
    }
    write_json(analysis_root / "reviewer_ablation_manifest.json", manifest)
    return summary_path
