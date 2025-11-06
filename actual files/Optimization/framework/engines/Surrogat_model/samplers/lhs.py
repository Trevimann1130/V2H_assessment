# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import Sequence, Tuple


def _lhs_1d(n: int, rng: np.random.Generator) -> np.ndarray:
    # gleich große Intervalle, jeweils ein uniformer Punkt, dann permutieren
    cut = np.linspace(0.0, 1.0, n + 1, dtype=float)
    u = rng.uniform(low=cut[:-1], high=cut[1:])
    rng.shuffle(u)
    return u


def _scale_to_bounds(U: np.ndarray, lower: Sequence[float], upper: Sequence[float]) -> np.ndarray:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return lower + U * (upper - lower)


def sample_lhs(n: int,
               lower: Sequence[float],
               upper: Sequence[float],
               seed: int | None = None,
               criterion: str | None = None,
               trials: int = 1) -> np.ndarray:
    """
    Einfache LHS-Implementierung (unabhängig je Dimension).
    Optional: rudimentäres 'maximin'-Kriterium via Mehrfachstichprobe (trials) und Auswahl
              der Stichprobe mit maximalem minimalen Paarabstand.

    Parameter
    ---------
    n : Anzahl Stichpunkte
    lower, upper : gleiche Länge = Dimensionalität
    seed : RNG-Seed
    criterion : None | "maximin"
    trials : int, nur relevant wenn criterion == "maximin"

    Rückgabe
    --------
    X : (n, d) ndarray
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    assert lower.shape == upper.shape
    assert n > 0

    d = lower.size
    rng = np.random.default_rng(seed)

    def one_draw() -> np.ndarray:
        U = np.column_stack([_lhs_1d(n, rng) for _ in range(d)])
        return _scale_to_bounds(U, lower, upper)

    if criterion is None or criterion.lower() != "maximin" or trials <= 1:
        return one_draw()

    # grobes maximin über mehrere Kandidaten
    best = None
    best_minpair = -np.inf
    for _ in range(int(trials)):
        X = one_draw()
        # paarweise Abstände (nur grob; O(n^2) ist für moderate n ok)
        # wir nehmen hier euklidische Distanz
        # um Kosten niedrig zu halten, ziehen wir eine Zufalls-Teilmenge wenn n sehr groß ist
        if n > 400:
            idx = rng.choice(n, size=400, replace=False)
            Y = X[idx]
        else:
            Y = X
        # Distanzmatrix ohne Diagonale
        d2 = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)
        d2 = d2 + np.eye(Y.shape[0]) * 1e18
        minpair = float(np.min(d2))
        if minpair > best_minpair:
            best_minpair = minpair
            best = X
    return best
