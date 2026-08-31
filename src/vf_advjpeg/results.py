from __future__ import annotations

from typing import Any

import pandas as pd

from vf_advjpeg.study import study_metadata
from vf_advjpeg.utils.fs import read_json, resolve_project_path


def collect_result_frame(config: dict[str, Any]) -> pd.DataFrame:
    root = resolve_project_path(config["paths"]["run_root"])
    records = []
    for path in root.rglob("seed_*.json"):
        payload = read_json(path)
        metadata = study_metadata(config, payload)
        record = {
            "path": str(path),
            "method": payload["method"],
            "target_model": payload["target_model"],
            "suite": payload["suite"],
            "seed": int(payload["seed"]),
            "duration_sec": float(payload["duration_sec"]),
            "device": str(payload.get("device") or payload.get("hardware", {}).get("device") or ""),
            "study_id": metadata["study_id"],
            "source_profile": metadata["source_profile"],
            "dataset": metadata["dataset"],
            "model_family": metadata["model_family"],
            "source_model": metadata["source_model"],
            "defense_profile": metadata["defense_profile"],
        }
        thread_env = payload.get("hardware", {}).get("thread_env", {})
        if isinstance(thread_env, dict):
            for key, value in thread_env.items():
                record[f"thread_env_{key}"] = str(value)
        record.update(dict(payload["metrics"]))
        records.append(record)
    return pd.DataFrame.from_records(records)
