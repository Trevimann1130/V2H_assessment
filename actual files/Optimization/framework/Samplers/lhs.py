# Optimization/framework/src/optfw/samplers/lhs.py
from __future__ import annotations
import numpy as np
try:
    from pyDOE2 import lhs as _lhs
except Exception:
    _lhs = None

from .base import Sampler

class LHSSampler(Sampler):
    def __init__(self, criterion: str | None = None):
        self.criterion = criterion

    def sample(self, n: int, lower, upper, seed: int) -> np.ndarray:
        d = len(lower)
        if _lhs is None:
            # Fallback: gleichverteiltes Random
            rng = np.random.default_rng(seed)
            U = rng.random((n, d))
        else:
            U = _lhs(d, samples=n, criterion=self.criterion or None)
        L = np.asarray(lower, float)
        UPPER = np.asarray(upper, float)
        return L + U * (UPPER - L)

# --- Kompatibilitäts-Helper für bestehende Skripte (validation.py) ---

def make_lhs_matrix(n: int, bounds: list[tuple[float, float]], seed: int):
    """
    Liefert ein (n,d)-Array. Skaliert LHS (oder Fallback-Random) in die gegebenen bounds.
    """
    sampler = LHSSampler()
    lower = [lo for lo, _ in bounds]
    upper = [hi for _, hi in bounds]
    return sampler.sample(n, lower, upper, seed)

def snap_to_steps(X, steps):
    """
    Rundet jede Spalte i von X auf das nächstliegende Vielfache von steps[i] (um 0 verankert).
    """
    import numpy as np
    X = np.asarray(X, float).copy()
    for i, s in enumerate(steps):
        if s and s > 0:
            X[:, i] = np.round(X[:, i] / s) * s
    return X

def evaluate_batch(X, base_params, profiles, use_v2h: bool, backend="threads",
                   n_workers=1, show_progress=False):
    """
    Platzhalter: Hier ggf. deinen FAST-Batch-Evaluator integrieren.
    Aktuell: Rückgabe eines minimalen DataFrames mit PV/BESS (damit validation.py startet).
    """
    import pandas as pd
    df = pd.DataFrame(X, columns=["PV_kWp", "BESS_kWh"])
    # TODO: Hier deine echte FAST-Pipeline aufrufen und die Spalten (Targets) befüllen.
    return df
