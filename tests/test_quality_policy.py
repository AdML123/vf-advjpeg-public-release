from __future__ import annotations

import numpy as np

from vf_advjpeg.attacks.quality import QualityPolicy, neighbor_qualities


def test_fixed_quality_policy() -> None:
    policy = QualityPolicy(kind="fixed", quality=80)
    rng = np.random.default_rng(0)
    assert policy.sample(rng) == 80


def test_random_quality_policy_bounds() -> None:
    policy = QualityPolicy(kind="random_uniform_int", quality_min=70, quality_max=72)
    rng = np.random.default_rng(0)
    samples = [policy.sample(rng) for _ in range(10)]
    assert all(70 <= item <= 72 for item in samples)


def test_neighbor_qualities() -> None:
    assert neighbor_qualities(80, 5, 3) == [75, 80, 85]

