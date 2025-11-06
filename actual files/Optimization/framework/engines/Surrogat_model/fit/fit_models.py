# Optimization/framework/engines/Surrogat_model/fit/fit_models.py
from __future__ import annotations

import numpy as np
from typing import List
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor


def fit_random_forest_per_column(
    X: np.ndarray,
    Y: np.ndarray | None,
    n_estimators: int,
    n_jobs: int,
    seed: int,
) -> List[RandomForestRegressor]:
    """
    Trainiert pro Zielspalte ein RandomForestRegressor-Modell.
    - Eine schlanke tqdm-Anzeige zeigt den Fortschritt (1 Balken für alle Zielspalten).
    - Keine sonstigen Prints.

    Parameters
    ----------
    X : (n_samples, n_features)
    Y : (n_samples, n_targets) oder None
    n_estimators : Anzahl Trees pro RF
    n_jobs : Parallelität für RF
    seed : Random-State für Reproduzierbarkeit

    Returns
    -------
    List[RandomForestRegressor]
        Liste trainierter Modelle (eine pro Y-Spalte).
    """
    if Y is None or (hasattr(Y, "shape") and Y.shape[1] == 0):
        return []

    models: List[RandomForestRegressor] = []
    n_cols = int(Y.shape[1])

    with tqdm(total=n_cols, desc="RF fit (targets)", unit="model") as pbar:
        for j in range(n_cols):
            y = Y[:, j]
            rf = RandomForestRegressor(
                n_estimators=int(n_estimators),
                random_state=int(seed),
                n_jobs=int(n_jobs),
            )
            rf.fit(X, y)
            models.append(rf)
            pbar.update(1)

    return models
