# Optimization/framework/engines/Surrogat_model/split/split.py
from __future__ import annotations
from typing import Tuple
import numpy as np


def train_holdout_split(
    X: np.ndarray,
    YF: np.ndarray,
    YG: np.ndarray,
    holdout_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < float(holdout_frac) < 1.0):
        raise ValueError("[split] holdout_frac muss in (0,1) liegen.")

    X = np.asarray(X, float)
    YF = np.asarray(YF, float) if YF is not None else None
    YG = np.asarray(YG, float) if YG is not None else None

    n = X.shape[0]
    if YF is not None and YF.shape[0] != n:
        raise ValueError("[split] YF hat nicht dieselbe Zeilenanzahl wie X.")
    if YG is not None and YG.shape[0] != n:
        raise ValueError("[split] YG hat nicht dieselbe Zeilenanzahl wie X.")

    rng = np.random.default_rng(int(seed))
    idx = np.arange(n)
    rng.shuffle(idx)
    n_hold = max(1, int(round(float(holdout_frac) * n)))

    hold_idx = idx[:n_hold]
    tr_idx   = idx[n_hold:]

    def sel(A):
        if A is None: return None
        return A[tr_idx], A[hold_idx]

    X_tr, X_hold = X[tr_idx], X[hold_idx]
    YF_tr, YF_hold = sel(YF)
    YG_tr, YG_hold = sel(YG)

    return X_tr, X_hold, YF_tr, YF_hold, YG_tr, YG_hold
