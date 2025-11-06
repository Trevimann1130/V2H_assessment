# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import Sequence

def sample_random(n: int,
                  lower: Sequence[float],
                  upper: Sequence[float],
                  seed: int | None = None) -> np.ndarray:
    """
    Gleichverteiltes Zufallssampling im Hyperrechteck [lower, upper].
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    assert lower.shape == upper.shape
    d = lower.size
    assert n > 0 and d > 0

    rng = np.random.default_rng(seed)
    U = rng.random((n, d))
    X = lower + U * (upper - lower)
    return X
