from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from skrf import Network
from skrf.vectorFitting import VectorFitting

from vf_advjpeg.attacks.dct_ops import ac_frequency_indices
from vf_advjpeg.jpeg.tables import quality_to_fit_frequency, scaled_quant_table
from vf_advjpeg.utils.fs import read_json, resolve_project_path, write_json


def _complex_to_pairs(values: np.ndarray) -> list[list[float]]:
    return [[float(np.real(item)), float(np.imag(item))] for item in values.tolist()]


def _pairs_to_complex(values: list[list[float]]) -> np.ndarray:
    return np.asarray([complex(item[0], item[1]) for item in values], dtype=np.complex128)


@dataclass(slots=True)
class VFResponseModel:
    kind: str
    qualities: list[int]
    responses: list[float]
    poles: list[list[float]] | None = None
    residues: list[list[float]] | None = None
    constant_term: list[float] | None = None
    proportional_term: list[float] | None = None
    rms_error: float | None = None

    def evaluate(self, quality: int) -> float:
        if self.kind == "vector_fit" and self.poles and self.residues and self.constant_term and self.proportional_term:
            freqs = np.asarray([quality_to_fit_frequency(quality)], dtype=np.float64)
            s = 2j * np.pi * freqs
            poles = _pairs_to_complex(self.poles)
            residues = _pairs_to_complex(self.residues)
            d = complex(self.constant_term[0], self.constant_term[1])
            e = complex(self.proportional_term[0], self.proportional_term[1])
            response = d + e * s[0]
            for pole, residue in zip(poles, residues):
                response += residue / (s[0] - pole)
            return float(np.real(response))
        if len(self.qualities) == 1:
            return float(self.responses[0])
        interpolator = PchipInterpolator(self.qualities, self.responses, extrapolate=True)
        return float(interpolator(quality))


@dataclass(slots=True)
class VFArtifact:
    amplitude_buckets: list[float]
    fits: dict[str, VFResponseModel]
    fit_diagnostics: dict[str, Any]

    def key(self, bin_index: int, bucket_index: int) -> str:
        return f"bin_{bin_index:02d}_bucket_{bucket_index}"

    def evaluate(self, bin_index: int, bucket_index: int, quality: int) -> float:
        return self.fits[self.key(bin_index, bucket_index)].evaluate(quality)

    def to_json(self) -> dict[str, Any]:
        return {
            "amplitude_buckets": self.amplitude_buckets,
            "fit_diagnostics": self.fit_diagnostics,
            "fits": {
                key: {
                    "kind": model.kind,
                    "qualities": model.qualities,
                    "responses": model.responses,
                    "poles": model.poles,
                    "residues": model.residues,
                    "constant_term": model.constant_term,
                    "proportional_term": model.proportional_term,
                    "rms_error": model.rms_error,
                }
                for key, model in self.fits.items()
            },
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "VFArtifact":
        return cls(
            amplitude_buckets=list(payload["amplitude_buckets"]),
            fit_diagnostics=dict(payload.get("fit_diagnostics", {})),
            fits={
                key: VFResponseModel(
                    kind=value["kind"],
                    qualities=list(value["qualities"]),
                    responses=list(value["responses"]),
                    poles=value.get("poles"),
                    residues=value.get("residues"),
                    constant_term=value.get("constant_term"),
                    proportional_term=value.get("proportional_term"),
                    rms_error=value.get("rms_error"),
                )
                for key, value in payload["fits"].items()
            },
        )


class VFGradientCorrector:
    def __init__(self, artifact: VFArtifact, structure_mode: str = "full") -> None:
        if structure_mode not in {"full", "frequency_only", "bucket_only", "no_vf"}:
            raise ValueError(f"Unsupported VF structure_mode: {structure_mode}")
        self.artifact = artifact
        self.frequency_indices = ac_frequency_indices()
        self.structure_mode = structure_mode

    def _bucket_index(self, normalized_magnitude: torch.Tensor) -> torch.Tensor:
        edges = [float(edge) for edge in self.artifact.amplitude_buckets]
        bucket = torch.zeros_like(normalized_magnitude, dtype=torch.long)
        if edges:
            bucket = bucket + (normalized_magnitude >= edges[0]).long()
        if len(edges) > 1:
            bucket = bucket + (normalized_magnitude >= edges[1]).long()
        return bucket.clamp(min=0, max=len(edges))

    def correct(self, dct_grad: torch.Tensor, delta_dct: torch.Tensor, quality: int) -> torch.Tensor:
        if self.structure_mode == "no_vf":
            return dct_grad.clone()
        corrected = dct_grad.clone()
        quant = scaled_quant_table(quality, "luma", device=dct_grad.device, dtype=dct_grad.dtype).view(1, 1, 1, 1, 8, 8)
        magnitudes = delta_dct.abs() / quant
        buckets = self._bucket_index(magnitudes)
        bucket_range = range(len(self.artifact.amplitude_buckets) + 1)
        if self.structure_mode == "bucket_only":
            bucket_corrections = {
                bucket_index: float(
                    np.mean(
                        [
                            self.artifact.evaluate(bin_index, bucket_index, quality)
                            for bin_index in range(len(self.frequency_indices))
                        ]
                    )
                )
                for bucket_index in bucket_range
            }
        for bin_index, (u, v) in enumerate(self.frequency_indices):
            for bucket_index in bucket_range:
                mask = buckets[..., u, v] == bucket_index
                if not mask.any():
                    continue
                if self.structure_mode == "frequency_only":
                    correction = self.artifact.evaluate(bin_index, 0, quality)
                elif self.structure_mode == "bucket_only":
                    correction = bucket_corrections[bucket_index]
                else:
                    correction = self.artifact.evaluate(bin_index, bucket_index, quality)
                corrected[..., u, v] = torch.where(mask, corrected[..., u, v] * correction, corrected[..., u, v])
        return corrected


def _fit_response_model(
    qualities: list[int],
    responses: np.ndarray,
    max_model_order: int,
    fit_error_threshold: float,
) -> tuple[VFResponseModel, dict[str, Any]]:
    freqs = np.asarray([quality_to_fit_frequency(item) for item in qualities], dtype=np.float64)
    network = Network(f=freqs, s=responses.reshape(-1, 1, 1), z0=50.0)
    vector_fit = VectorFitting(network)
    diagnostics: dict[str, Any] = {"fallback": False}
    try:
        vector_fit.auto_fit(
            n_poles_init_real=2,
            n_poles_init_cmplx=2,
            model_order_max=max_model_order,
            target_error=fit_error_threshold,
        )
        rms_error = float(vector_fit.get_rms_error(i=-1, j=-1))
        model = VFResponseModel(
            kind="vector_fit",
            qualities=qualities,
            responses=[float(np.real(item)) for item in responses.tolist()],
            poles=_complex_to_pairs(vector_fit.poles),
            residues=_complex_to_pairs(vector_fit.residues.reshape(-1)),
            constant_term=_complex_to_pairs(vector_fit.constant_coeff.reshape(-1))[0],
            proportional_term=_complex_to_pairs(vector_fit.proportional_coeff.reshape(-1))[0],
            rms_error=rms_error,
        )
        diagnostics["rms_error"] = rms_error
        diagnostics["stable"] = bool(np.all(np.real(vector_fit.poles) <= 0))
        return model, diagnostics
    except Exception as exc:
        diagnostics["fallback"] = True
        diagnostics["error"] = repr(exc)
        fallback = VFResponseModel(
            kind="pchip",
            qualities=qualities,
            responses=[float(np.real(item)) for item in responses.tolist()],
            rms_error=None,
        )
        return fallback, diagnostics


def build_vf_artifact(
    quality_grid: list[int],
    aggregated_responses: np.ndarray,
    amplitude_buckets: list[float],
    max_model_order: int,
    fit_error_threshold: float,
) -> VFArtifact:
    fits: dict[str, VFResponseModel] = {}
    diagnostics: dict[str, Any] = {}
    for bin_index in range(aggregated_responses.shape[0]):
        for bucket_index in range(aggregated_responses.shape[1]):
            key = f"bin_{bin_index:02d}_bucket_{bucket_index}"
            model, model_diag = _fit_response_model(
                qualities=quality_grid,
                responses=aggregated_responses[bin_index, bucket_index].astype(np.complex128),
                max_model_order=max_model_order,
                fit_error_threshold=fit_error_threshold,
            )
            fits[key] = model
            diagnostics[key] = model_diag
    return VFArtifact(amplitude_buckets=amplitude_buckets, fits=fits, fit_diagnostics=diagnostics)


def save_vf_artifact(path: str, artifact: VFArtifact) -> None:
    write_json(resolve_project_path(path), artifact.to_json())


def load_vf_artifact(path: str) -> VFArtifact:
    return VFArtifact.from_json(read_json(resolve_project_path(path)))
