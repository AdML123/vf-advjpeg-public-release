from __future__ import annotations

import torch

from vf_advjpeg.attacks.dct_ops import block_dct, block_idct, image_to_blocks, blocks_to_image, y_channel_dct


def test_dct_idct_round_trip() -> None:
    images = torch.rand(2, 1, 16, 16)
    blocks = image_to_blocks(images)
    restored = blocks_to_image(block_idct(block_dct(blocks)), 16, 16)
    assert torch.allclose(images, restored, atol=1e-5)


def test_y_channel_dct_shape() -> None:
    images = torch.rand(2, 3, 16, 16)
    coeffs = y_channel_dct(images)
    assert coeffs.shape == (2, 2, 2, 1, 8, 8)

