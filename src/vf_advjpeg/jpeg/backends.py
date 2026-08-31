from __future__ import annotations

import io
import math
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from vf_advjpeg.attacks.dct_ops import (
    block_dct,
    block_idct,
    blocks_to_image,
    image_to_blocks,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)
from vf_advjpeg.jpeg.tables import scaled_quant_table


def _to_quality_list(quality: int | Sequence[int], batch_size: int) -> list[int]:
    if isinstance(quality, int):
        return [quality] * batch_size
    qualities = list(quality)
    if len(qualities) != batch_size:
        raise ValueError("Quality list length must match batch size.")
    return [int(item) for item in qualities]


def apply_real_jpeg(images: torch.Tensor, quality: int | Sequence[int]) -> torch.Tensor:
    device = images.device
    qualities = _to_quality_list(quality, images.size(0))
    restored = []
    for image, item_quality in zip(images.detach().cpu(), qualities):
        array = (image.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="JPEG", quality=int(item_quality))
        buffer.seek(0)
        decoded = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32) / 255.0
        restored.append(torch.from_numpy(decoded).permute(2, 0, 1))
    return torch.stack(restored, dim=0).to(device=device, dtype=images.dtype)


class _JPEGIdentityBPDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, images: torch.Tensor, quality: int) -> torch.Tensor:
        return apply_real_jpeg(images, int(quality))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def jpeg_identity_bpda(images: torch.Tensor, quality: int) -> torch.Tensor:
    return _JPEGIdentityBPDA.apply(images, int(quality))


def soft_round(tensor: torch.Tensor) -> torch.Tensor:
    return tensor - torch.sin(2.0 * math.pi * tensor) / (2.0 * math.pi)


def diffjpeg_surrogate(images: torch.Tensor, quality: int, smooth_round: bool = True) -> torch.Tensor:
    ycbcr = rgb_to_ycbcr(images)
    height, width = images.shape[-2:]
    restored_channels = []
    for channel_index in range(3):
        channel = ycbcr[:, channel_index : channel_index + 1]
        blocks = image_to_blocks(channel)
        coeffs = block_dct(blocks)
        table = scaled_quant_table(
            quality,
            "luma" if channel_index == 0 else "chroma",
            device=images.device,
            dtype=images.dtype,
        ).view(1, 1, 1, 1, 8, 8)
        normalized = coeffs / table
        quantized = soft_round(normalized) if smooth_round else torch.round(normalized)
        restored = block_idct(quantized * table)
        restored_channels.append(blocks_to_image(restored, height, width))
    return ycbcr_to_rgb(torch.cat(restored_channels, dim=1)).clamp(0.0, 1.0)

