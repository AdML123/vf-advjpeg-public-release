from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class QualityPolicy:
    kind: str
    quality: int | None = None
    quality_min: int | None = None
    quality_max: int | None = None

    def sample(self, rng: np.random.Generator, step: int | None = None) -> int:
        if self.kind == "fixed":
            if self.quality is None:
                raise ValueError("Fixed policy requires quality.")
            return int(self.quality)
        if self.kind == "random_uniform_int":
            if self.quality_min is None or self.quality_max is None:
                raise ValueError("Random quality policy requires min/max.")
            return int(rng.integers(self.quality_min, self.quality_max + 1))
        raise KeyError(f"Unsupported quality policy: {self.kind}")


def quality_policy_from_config(payload: dict[str, Any]) -> QualityPolicy:
    return QualityPolicy(
        kind=str(payload["kind"]),
        quality=payload.get("quality"),
        quality_min=payload.get("quality_min"),
        quality_max=payload.get("quality_max"),
    )


def neighbor_qualities(center: int, span: int, samples: int) -> list[int]:
    if samples <= 1:
        return [int(center)]
    grid = np.linspace(center - span, center + span, samples)
    return [int(np.clip(round(item), 1, 100)) for item in grid.tolist()]

