# Optimization/framework/src/optfw/samplers/sobol.py
from __future__ import annotations
import numpy as np
from scipy.stats import qmc
from .base import Sampler

class SobolSampler(Sampler):
    def sample(self, n: int, lower, upper, seed: int) -> np.ndarray:
        d = len(lower)
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        U = sampler.random(n)
        L = np.asarray(lower, float)
        UPPER = np.asarray(upper, float)
        return qmc.scale(U, L, UPPER)
