# Optimization/framework/src/optfw/samplers/base.py
from __future__ import annotations
import numpy as np
from typing import List

class Sampler:
    def sample(self, n: int, lower: List[float], upper: List[float], seed: int) -> np.ndarray:
        raise NotImplementedError
