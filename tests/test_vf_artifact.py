from __future__ import annotations

import numpy as np
import torch

from vf_advjpeg.attacks.vf import VFArtifact, VFGradientCorrector, VFResponseModel, build_vf_artifact


def test_vf_artifact_round_trip_evaluation() -> None:
    fits = {
        "bin_00_bucket_0": VFResponseModel(kind="pchip", qualities=[70, 80, 90], responses=[1.0, 1.2, 1.4]),
    }
    artifact = VFArtifact(amplitude_buckets=[0.5], fits=fits, fit_diagnostics={})
    payload = artifact.to_json()
    restored = VFArtifact.from_json(payload)
    assert abs(restored.evaluate(0, 0, 85) - 1.3) < 0.2


def test_single_quality_vf_response_uses_constant_fallback() -> None:
    model = VFResponseModel(kind="pchip", qualities=[80], responses=[1.25])

    assert model.evaluate(70) == 1.25
    assert model.evaluate(90) == 1.25


def test_build_vf_artifact_returns_all_bins() -> None:
    responses = np.ones((2, 2, 3), dtype=np.float64)
    artifact = build_vf_artifact([70, 80, 90], responses, [0.5], max_model_order=4, fit_error_threshold=0.2)
    assert "bin_00_bucket_0" in artifact.fits
    assert "bin_01_bucket_1" in artifact.fits


def _artifact_with_bin_responses() -> VFArtifact:
    fits = {}
    for bin_index in range(63):
        for bucket_index, response in enumerate([float(bin_index + 1), 100.0 + bin_index]):
            fits[f"bin_{bin_index:02d}_bucket_{bucket_index}"] = VFResponseModel(
                kind="pchip",
                qualities=[80],
                responses=[response],
            )
    return VFArtifact(amplitude_buckets=[0.5], fits=fits, fit_diagnostics={})


def test_vf_corrector_can_disable_frequency_or_bucket_structure() -> None:
    dct_grad = torch.ones(1, 1, 1, 1, 8, 8)
    delta_dct = torch.zeros_like(dct_grad)
    artifact = _artifact_with_bin_responses()

    full = VFGradientCorrector(artifact, structure_mode="full").correct(dct_grad, delta_dct, quality=80)
    no_vf = VFGradientCorrector(artifact, structure_mode="no_vf").correct(dct_grad, delta_dct, quality=80)
    bucket_only = VFGradientCorrector(artifact, structure_mode="bucket_only").correct(dct_grad, delta_dct, quality=80)
    frequency_only = VFGradientCorrector(artifact, structure_mode="frequency_only").correct(dct_grad, delta_dct, quality=80)

    assert torch.equal(no_vf, dct_grad)
    assert full[..., 0, 1].item() == 1.0
    assert full[..., 0, 2].item() == 2.0
    assert bucket_only[..., 0, 1].item() == bucket_only[..., 0, 2].item()
    assert round(bucket_only[..., 0, 1].item(), 1) == 32.0
    assert frequency_only[..., 0, 1].item() == 1.0
