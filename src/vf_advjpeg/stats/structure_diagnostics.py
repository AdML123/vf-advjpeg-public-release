from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResponseBankFootprint:
    response_bank_bytes: int
    response_bank_mib: float
    coefficient_count: int
    quality_count: int
    frequency_count: int
    bucket_count: int


def _weighted_mean(values: np.ndarray, weights: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    weighted = np.where(weights > 0, values * weights, 0.0)
    numerator = weighted.sum(axis=axis)
    denominator = weights.sum(axis=axis)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)


def _roughness(values: np.ndarray) -> float:
    ordered = np.asarray(values, dtype=float)
    if ordered.size <= 1:
        return 0.0
    return float(np.mean(np.abs(np.diff(ordered))))


def _interaction_eta_squared(values: np.ndarray, weights: np.ndarray) -> float:
    collapsed = _weighted_mean(values, weights, axis=2)
    quality_mean = collapsed.mean(axis=1, keepdims=True)
    frequency_mean = collapsed.mean(axis=0, keepdims=True)
    grand_mean = float(collapsed.mean())
    additive = quality_mean + frequency_mean - grand_mean
    residual = collapsed - additive
    total_ss = float(np.sum((collapsed - grand_mean) ** 2))
    residual_ss = float(np.sum(residual**2))
    if total_ss <= 0:
        return 0.0
    return float(min(max(residual_ss / total_ss, 0.0), 1.0))


def analyze_vf_structure(calibration_npz: Path) -> tuple[pd.DataFrame, dict[str, object], ResponseBankFootprint]:
    raw = np.load(calibration_npz)
    responses = raw["responses"].astype(float)
    counts = raw["counts"].astype(float)
    if responses.shape != counts.shape or responses.ndim != 3:
        raise ValueError("Expected responses/counts arrays with shape quality x frequency x bucket.")

    quality_mean = _weighted_mean(responses, counts, axis=(1, 2))
    frequency_mean = _weighted_mean(responses, counts, axis=(0, 2))
    bucket_mean = _weighted_mean(responses, counts, axis=(0, 1))
    residual = responses - responses.mean(axis=0, keepdims=True)

    summary = {
        "quality_roughness": _roughness(quality_mean),
        "frequency_roughness": _roughness(frequency_mean),
        "bucket_roughness": _roughness(bucket_mean),
        "fit_residual_mean_abs": float(np.mean(np.abs(residual))),
        "fit_residual_p95_abs": float(np.percentile(np.abs(residual), 95)),
        "interaction_eta_squared": _interaction_eta_squared(responses, counts),
        "nonzero_cell_count": int((counts > 0).sum()),
        "total_observations": int(counts.sum()),
    }

    rows: list[dict[str, object]] = []
    for index, value in enumerate(frequency_mean.tolist(), start=1):
        rows.append({"axis": "frequency", "index": index, "mean_response": value})
    for index, value in enumerate(bucket_mean.tolist(), start=1):
        rows.append({"axis": "bucket", "index": index, "mean_response": value})
    for index, value in enumerate(quality_mean.tolist(), start=1):
        rows.append({"axis": "quality", "index": index, "mean_response": value})

    footprint = ResponseBankFootprint(
        response_bank_bytes=int(responses.nbytes + counts.nbytes),
        response_bank_mib=float((responses.nbytes + counts.nbytes) / (1024 * 1024)),
        coefficient_count=int(responses.size),
        quality_count=int(responses.shape[0]),
        frequency_count=int(responses.shape[1]),
        bucket_count=int(responses.shape[2]),
    )
    return pd.DataFrame.from_records(rows), summary, footprint


def write_vf_structure_diagnostics(calibration_npz: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, summary, footprint = analyze_vf_structure(calibration_npz)
    csv_path = output_dir / "vf_structure_diagnostics.csv"
    json_path = output_dir / "vf_structure_diagnostics.json"
    footprint_path = output_dir / "response_bank_footprint.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    footprint_path.write_text(json.dumps(asdict(footprint), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "footprint": footprint_path}
