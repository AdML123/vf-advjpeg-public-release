from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch


LUMA_QUANT_TABLE = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)

CHROMA_QUANT_TABLE = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=np.float32,
)


def quality_to_scale_factor(quality: int) -> float:
    quality = int(max(1, min(quality, 100)))
    if quality < 50:
        return 5000.0 / quality
    return 200.0 - 2.0 * quality


def quality_to_lambda(quality: int) -> float:
    return quality_to_scale_factor(quality) / 100.0


def quality_to_fit_frequency(quality: int) -> float:
    lam = max(quality_to_lambda(quality), 1e-6)
    return 1.0 + math.log1p(1.0 / lam)


@lru_cache(maxsize=128)
def _scaled_table_cached(quality: int, channel: str) -> np.ndarray:
    base = LUMA_QUANT_TABLE if channel == "luma" else CHROMA_QUANT_TABLE
    scale = quality_to_scale_factor(quality)
    table = np.floor((base * scale + 50.0) / 100.0)
    return np.clip(table, 1.0, 255.0).astype(np.float32)


def scaled_quant_table(
    quality: int,
    channel: str,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    array = _scaled_table_cached(int(quality), channel)
    return torch.tensor(array, device=device, dtype=dtype)
