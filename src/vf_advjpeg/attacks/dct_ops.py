from __future__ import annotations

import math
from functools import lru_cache

import torch


def rgb_to_ycbcr(images: torch.Tensor) -> torch.Tensor:
    r, g, b = images[:, 0:1], images[:, 1:2], images[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.cat([y, cb, cr], dim=1)


def ycbcr_to_rgb(images: torch.Tensor) -> torch.Tensor:
    y = images[:, 0:1]
    cb = images[:, 1:2] - 0.5
    cr = images[:, 2:3] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)


@lru_cache(maxsize=16)
def _dct_matrix(size: int, dtype_name: str) -> torch.Tensor:
    dtype = getattr(torch, dtype_name)
    matrix = torch.zeros(size, size, dtype=dtype)
    for u in range(size):
        alpha = math.sqrt(1.0 / size) if u == 0 else math.sqrt(2.0 / size)
        for x in range(size):
            matrix[u, x] = alpha * math.cos(math.pi * (2 * x + 1) * u / (2 * size))
    return matrix


def dct_matrix(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _dct_matrix(size, str(dtype).split(".")[-1]).to(device=device, dtype=dtype)


def image_to_blocks(channel: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    batch, channels, height, width = channel.shape
    if channels != 1:
        raise ValueError("Expected a single-channel tensor.")
    if height % block_size or width % block_size:
        raise ValueError("Height and width must be divisible by block size.")
    return (
        channel.view(batch, 1, height // block_size, block_size, width // block_size, block_size)
        .permute(0, 2, 4, 1, 3, 5)
        .contiguous()
    )


def blocks_to_image(blocks: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch, h_blocks, w_blocks, channels, block_h, block_w = blocks.shape
    image = blocks.permute(0, 3, 1, 4, 2, 5).contiguous()
    return image.view(batch, channels, height, width)


def block_dct(blocks: torch.Tensor) -> torch.Tensor:
    matrix = dct_matrix(blocks.shape[-1], blocks.device, blocks.dtype)
    return matrix @ blocks @ matrix.transpose(-1, -2)


def block_idct(coeffs: torch.Tensor) -> torch.Tensor:
    matrix = dct_matrix(coeffs.shape[-1], coeffs.device, coeffs.dtype)
    return matrix.transpose(-1, -2) @ coeffs @ matrix


def y_channel_dct(images: torch.Tensor) -> torch.Tensor:
    y = rgb_to_ycbcr(images)[:, 0:1]
    return block_dct(image_to_blocks(y))


def reconstruct_rgb_from_y_dct(clean_ycbcr: torch.Tensor, y_coeffs: torch.Tensor) -> torch.Tensor:
    height, width = clean_ycbcr.shape[-2:]
    restored_y = blocks_to_image(block_idct(y_coeffs), height, width)
    merged = clean_ycbcr.clone()
    merged[:, 0:1] = restored_y
    return ycbcr_to_rgb(merged)


def ac_mask_like(y_coeffs: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(y_coeffs)
    mask[..., 0, 0] = 0.0
    return mask


def y_pixel_grad_to_dct(pixel_grad: torch.Tensor) -> torch.Tensor:
    y_grad = pixel_grad.sum(dim=1, keepdim=True)
    return block_dct(image_to_blocks(y_grad))


def ac_frequency_indices() -> list[tuple[int, int]]:
    indices = []
    for u in range(8):
        for v in range(8):
            if (u, v) != (0, 0):
                indices.append((u, v))
    return indices

