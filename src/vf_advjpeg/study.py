from __future__ import annotations

from typing import Any


def derive_source_profile(config: dict[str, Any]) -> str:
    analysis_cfg = config.get("analysis", {})
    explicit = analysis_cfg.get("source_profile")
    if explicit:
        return str(explicit)

    source_items = config.get("models", {}).get("source", [])
    names = [str(item.get("name")) for item in source_items if isinstance(item, dict) and item.get("name")]
    if not names:
        return "unknown"
    if names == ["resnet18"]:
        return "r18_only"
    if names == ["resnet18", "vgg16"]:
        return "r18_vgg16"
    return "_".join(name.replace("-", "_") for name in names)


def derive_source_model(config: dict[str, Any]) -> str:
    source_items = config.get("models", {}).get("source", [])
    names = [str(item.get("name")) for item in source_items if isinstance(item, dict) and item.get("name")]
    return "+".join(names) if names else "unknown"


def derive_model_family(config: dict[str, Any], payload: dict[str, Any] | None = None) -> str:
    payload = payload or {}
    model_names = [str(payload.get("target_model", "")), derive_source_model(config)]
    joined = " ".join(model_names).lower()
    if "robustbench:" in joined:
        return "robustbench"
    if any(name in joined for name in ["vit", "deit", "swin"]):
        return "transformer"
    return "cnn"


def derive_defense_profile(config: dict[str, Any], payload: dict[str, Any] | None = None) -> str:
    payload = payload or {}
    model_names = [str(payload.get("target_model", "")), derive_source_model(config)]
    joined = " ".join(model_names).lower()
    if "robustbench:" in joined or "robustbench" in joined:
        if ":linf:" in joined or "linf" in joined:
            return "linf_robust"
        return "robust"
    return "standard"


def study_metadata(config: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, str]:
    analysis_cfg = config.get("analysis", {})
    data_cfg = config.get("data", {})
    payload = payload or {}

    study_id = payload.get("study_id") or analysis_cfg.get("study_id") or "default"
    source_profile = payload.get("source_profile") or derive_source_profile(config)
    dataset = payload.get("dataset") or data_cfg.get("dataset") or "pet37"
    source_model = payload.get("source_model") or derive_source_model(config)
    model_family = payload.get("model_family") or analysis_cfg.get("model_family") or derive_model_family(config, payload)
    defense_profile = (
        payload.get("defense_profile")
        or analysis_cfg.get("defense_profile")
        or derive_defense_profile(config, payload)
    )
    return {
        "study_id": str(study_id),
        "source_profile": str(source_profile),
        "dataset": str(dataset),
        "model_family": str(model_family),
        "source_model": str(source_model),
        "defense_profile": str(defense_profile),
    }
