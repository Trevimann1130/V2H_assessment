# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import Sequence
from .lhs import sample_lhs
from .sobol import sample_sobol
from .random import sample_random


def _snap_to_steps(X: np.ndarray, steps: Sequence[float] | None) -> np.ndarray:
    if steps is None:
        return X
    X = np.asarray(X, dtype=float).copy()
    for j, step in enumerate(steps):
        if step is None or float(step) == 0.0:
            continue
        s = float(step)
        X[:, j] = np.round(X[:, j] / s) * s
    return X


def _drop_duplicates(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X
    # round to mitigate floating noise before unique (use 8 decimals)
    Xr = np.round(X, 8)
    _, idx = np.unique(Xr, axis=0, return_index=True)
    idx = np.sort(idx)
    return X[idx]


def sample_from_settings(settings) -> np.ndarray:
    """
    Zentrale Sampling-Funktion.
    Nutzt ausschließlich die Settings:
      - settings.bounds.lower / .upper / .steps
      - settings.sampler.name ∈ {"lhs","sobol","random"}
      - settings.sampler.n_samples
      - settings.sampler.seed
      - settings.sampler.kwargs (z. B. criterion, trials, scramble, skip, ...)
    """
    name = str(settings.sampler.name).lower().strip()
    n = int(settings.sampler.n_samples)
    seed = int(settings.sampler.seed)
    kwargs = dict(getattr(settings.sampler, "kwargs", {}) or {})

    lower = list(settings.bounds.lower)
    upper = list(settings.bounds.upper)
    steps = getattr(settings.bounds, "steps", None)

    if len(lower) != len(upper):
        raise ValueError("[Samplers.factory] bounds.lower und bounds.upper haben unterschiedliche Längen.")

    if name == "lhs":
        # optionales 'criterion' und 'trials' aus kwargs
        crit = kwargs.get("criterion", None)
        trials = int(kwargs.get("trials", 1))
        X = sample_lhs(n=n, lower=lower, upper=upper, seed=seed, criterion=crit, trials=trials)
    elif name == "sobol":
        X = sample_sobol(n=n, lower=lower, upper=upper, seed=seed, **kwargs)
    elif name == "random":
        X = sample_random(n=n, lower=lower, upper=upper, seed=seed)
    else:
        raise ValueError(f"[Samplers.factory] Unbekannter Sampler: '{name}'. Erlaubt: lhs, sobol, random.")

    n0 = X.shape[0]
    X = _snap_to_steps(X, steps)
    X = _drop_duplicates(X)
    n1 = X.shape[0]

    return X
