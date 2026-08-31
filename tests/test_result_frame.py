from __future__ import annotations

from pathlib import Path

from vf_advjpeg.config import load_config
from vf_advjpeg.experiments import result_path
from vf_advjpeg.results import collect_result_frame
from vf_advjpeg.utils.fs import write_json


def test_collect_result_frame_reads_nested_metrics(tmp_path: Path) -> None:
    config = load_config("configs/default.yaml", "configs/smoke.yaml")
    config["paths"]["run_root"] = str(tmp_path / "runs")
    run_dir = Path(config["paths"]["run_root"]) / "baseline_strong" / "mobilenet_v2" / "static_q70"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "seed_7.json",
        {
            "method": "baseline_strong",
            "target_model": "mobilenet_v2",
            "suite": "static_q70",
            "seed": 7,
            "study_id": "reconfirm",
            "source_profile": "r18_only",
            "dataset": "pet37",
            "model_family": "cnn",
            "source_model": "resnet18",
            "defense_profile": "standard",
            "device": "cpu",
            "hardware": {"device": "cpu", "thread_env": {"OMP_NUM_THREADS": "1"}},
            "duration_sec": 1.0,
            "metrics": {"asr": 0.2, "clean_accuracy": 0.5, "jpeg_clean_accuracy": 0.4, "clean_drop": 0.1, "lpips": 0.1, "ssim": 0.9, "linf": 0.03},
        },
    )
    frame = collect_result_frame(config)
    assert len(frame) == 1
    assert frame.iloc[0]["asr"] == 0.2
    assert frame.iloc[0]["study_id"] == "reconfirm"
    assert frame.iloc[0]["source_profile"] == "r18_only"
    assert frame.iloc[0]["dataset"] == "pet37"
    assert frame.iloc[0]["model_family"] == "cnn"
    assert frame.iloc[0]["source_model"] == "resnet18"
    assert frame.iloc[0]["defense_profile"] == "standard"
    assert frame.iloc[0]["device"] == "cpu"
    assert frame.iloc[0]["thread_env_OMP_NUM_THREADS"] == "1"


def test_result_path_sanitizes_model_name_for_windows(tmp_path: Path) -> None:
    config = load_config("configs/default.yaml", "configs/reviewer_defended.yaml")
    config["paths"]["run_root"] = str(tmp_path / "runs")

    path = result_path(
        config,
        method="baseline_strong",
        target_model="robustbench:cifar10:Linf:Rice2020Overfitting",
        suite_name="static_q70",
        seed=42,
    )

    assert path.parent.exists()
    assert ":" not in path.relative_to(tmp_path / "runs").as_posix()
    assert path.name == "seed_42.json"
