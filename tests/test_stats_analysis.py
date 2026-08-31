from __future__ import annotations

import math

import pandas as pd

from vf_advjpeg.config import load_config
from vf_advjpeg.stats.analysis import build_pairwise_frame, summarize_efficiency_tradeoff, summarize_results


def test_summarize_results_empty_frame() -> None:
    frame = pd.DataFrame()
    summary = summarize_results(frame)
    assert summary.empty


def test_efficiency_tradeoff_metrics_are_computed() -> None:
    config = load_config("configs/default.yaml", "configs/smoke.yaml")
    frame = pd.DataFrame(
        [
            {
                "method": "baseline_strong",
                "target_model": "mobilenet_v2",
                "suite": "static_q70",
                "seed": 42,
                "asr": 0.40,
                "duration_sec": 12.0,
                "mean_estimator_time": 4.0,
                "clean_drop": 0.01,
                "lpips": 0.10,
                "ssim": 0.90,
                "linf": 0.03,
            },
            {
                "method": "vf_advjpeg",
                "target_model": "mobilenet_v2",
                "suite": "static_q70",
                "seed": 42,
                "asr": 0.32,
                "duration_sec": 3.0,
                "mean_estimator_time": 1.0,
                "clean_drop": 0.01,
                "lpips": 0.11,
                "ssim": 0.89,
                "linf": 0.03,
            },
            {
                "method": "baseline_strong",
                "target_model": "mobilenet_v2",
                "suite": "static_q70",
                "seed": 43,
                "asr": 0.20,
                "duration_sec": 9.0,
                "mean_estimator_time": 3.0,
                "clean_drop": 0.02,
                "lpips": 0.12,
                "ssim": 0.88,
                "linf": 0.03,
            },
            {
                "method": "vf_advjpeg",
                "target_model": "mobilenet_v2",
                "suite": "static_q70",
                "seed": 43,
                "asr": 0.16,
                "duration_sec": 3.0,
                "mean_estimator_time": 1.0,
                "clean_drop": 0.02,
                "lpips": 0.13,
                "ssim": 0.87,
                "linf": 0.03,
            },
        ]
    )

    pairs = build_pairwise_frame(frame, config)
    assert set(pairs.columns) >= {
        "study_id",
        "source_profile",
        "asr_retention",
        "runtime_speedup",
        "estimator_speedup",
        "clean_drop_delta",
        "efficiency_win",
    }
    assert len(pairs) == 2
    assert math.isclose(float(pairs.iloc[0]["asr_retention"]), 0.8)
    assert math.isclose(float(pairs.iloc[0]["runtime_speedup"]), 4.0)
    assert math.isclose(float(pairs.iloc[0]["estimator_speedup"]), 4.0)
    assert math.isclose(float(pairs.iloc[0]["clean_drop_delta"]), 0.0)
    assert bool(pairs.iloc[0]["efficiency_win"]) is True

    tradeoff = summarize_efficiency_tradeoff(pairs, config)
    assert len(tradeoff) == 1
    row = tradeoff.iloc[0]
    assert "study_id" in tradeoff.columns
    assert "source_profile" in tradeoff.columns
    assert math.isclose(float(row["asr_retention_mean"]), 0.8)
    assert math.isclose(float(row["runtime_speedup_mean"]), 3.5)
    assert math.isclose(float(row["estimator_speedup_mean"]), 3.5)
    assert math.isclose(float(row["clean_drop_delta_mean"]), 0.0)
    assert bool(row["efficiency_win"]) is True
    assert "runtime_p_value_holm" in tradeoff.columns


def test_clean_drop_guardrail_uses_delta_not_absolute_value() -> None:
    config = load_config("configs/default.yaml", "configs/smoke.yaml")
    frame = pd.DataFrame(
        [
            {
                "method": "baseline_strong",
                "target_model": "mobilenet_v2",
                "suite": "static_q70",
                "seed": 42,
                "asr": 0.40,
                "duration_sec": 12.0,
                "mean_estimator_time": 4.0,
                "clean_drop": 0.18,
                "lpips": 0.10,
                "ssim": 0.90,
                "linf": 0.03,
            },
            {
                "method": "vf_advjpeg",
                "target_model": "mobilenet_v2",
                "suite": "static_q70",
                "seed": 42,
                "asr": 0.32,
                "duration_sec": 3.0,
                "mean_estimator_time": 1.0,
                "clean_drop": 0.19,
                "lpips": 0.11,
                "ssim": 0.89,
                "linf": 0.03,
            },
        ]
    )

    pairs = build_pairwise_frame(frame, config)
    assert math.isclose(float(pairs.iloc[0]["clean_drop_delta"]), 0.01)
    assert bool(pairs.iloc[0]["efficiency_win"]) is True

    tradeoff = summarize_efficiency_tradeoff(pairs, config)
    assert math.isclose(float(tradeoff.iloc[0]["clean_drop_delta_mean"]), 0.01)
    assert bool(tradeoff.iloc[0]["efficiency_win"]) is True


def test_missing_baseline_or_candidate_pairs_are_skipped() -> None:
    config = load_config("configs/default.yaml", "configs/smoke.yaml")
    frame = pd.DataFrame(
        [
            {
                "method": "baseline_strong",
                "target_model": "mobilenet_v2",
                "suite": "static_q70",
                "seed": 42,
                "asr": 0.4,
                "duration_sec": 10.0,
                "mean_estimator_time": 4.0,
                "clean_drop": 0.01,
                "lpips": 0.10,
                "ssim": 0.90,
                "linf": 0.03,
            },
            {
                "method": "vf_advjpeg",
                "target_model": "mobilenet_v2",
                "suite": "static_q80",
                "seed": 42,
                "asr": 0.3,
                "duration_sec": 3.0,
                "mean_estimator_time": 1.0,
                "clean_drop": 0.01,
                "lpips": 0.11,
                "ssim": 0.89,
                "linf": 0.03,
            },
        ]
    )

    pairs = build_pairwise_frame(frame, config)
    assert pairs.empty

    tradeoff = summarize_efficiency_tradeoff(pairs, config)
    assert tradeoff.empty
