from __future__ import annotations

import torch


def normalize_batch(tensor: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=tensor.device, dtype=tensor.dtype).view(1, -1, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def denormalize_batch(tensor: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=tensor.device, dtype=tensor.dtype).view(1, -1, 1, 1)
    return tensor * std_tensor + mean_tensor

