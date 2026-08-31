from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def input_diversity(
    images: torch.Tensor,
    rng: np.random.Generator,
    probability: float,
    scale_min: float,
    scale_max: float,
) -> torch.Tensor:
    if probability <= 0 or float(rng.random()) > probability:
        return images

    _, _, height, width = images.shape
    scale = float(rng.uniform(scale_min, scale_max))
    target_h = max(1, int(round(height * scale)))
    target_w = max(1, int(round(width * scale)))
    resized = F.interpolate(images, size=(target_h, target_w), mode="bilinear", align_corners=False)

    if target_h <= height and target_w <= width:
        pad_h = height - target_h
        pad_w = width - target_w
        top = int(rng.integers(0, pad_h + 1))
        left = int(rng.integers(0, pad_w + 1))
        bottom = pad_h - top
        right = pad_w - left
        return F.pad(resized, (left, right, top, bottom), mode="constant", value=0.0)

    max_top = max(target_h - height, 0)
    max_left = max(target_w - width, 0)
    top = int(rng.integers(0, max_top + 1))
    left = int(rng.integers(0, max_left + 1))
    return resized[:, :, top : top + height, left : left + width]


def gaussian_kernel(kernel_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(grid_x.pow(2) + grid_y.pow(2)) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)


def translation_invariant_smoothing(gradients: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    kernel = gaussian_kernel(kernel_size, sigma, gradients.size(1), gradients.device, gradients.dtype)
    padding = kernel_size // 2
    return F.conv2d(gradients, kernel, padding=padding, groups=gradients.size(1))

