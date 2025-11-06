# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import Sequence

def sample_sobol(n: int,
                 lower: Sequence[float],
                 upper: Sequence[float],
                 seed: int | None = None,
                 **kwargs) -> np.ndarray:
    """
    Sobol-Sequenz über scipy.stats.qmc.Sobol (scrambled). Erfordert scipy>=1.7.
    Keine Fallbacks – wenn scipy fehlt, wird ein klarer Fehler geworfen.

    Unterstützte kwargs (werden direkt an qmc.Sobol übergeben, soweit sinnvoll):
      - scramble: bool (default True)
      - skip: int (default 0) – kann via seed beeinflusst werden
    """
    try:
        from scipy.stats import qmc  # type: ignore
    except Exception as e:
        raise ImportError(
            "[Samplers.sobol] scipy ist erforderlich (scipy.stats.qmc.Sobol). "
            "Bitte 'pip install scipy' ausführen."
        ) from e

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    assert lower.shape == upper.shape
    d = lower.size
    assert n > 0 and d > 0

    scramble = kwargs.get("scramble", True)
    skip = kwargs.get("skip", 0)

    # seed beeinflusst bei Sobol meist das Scrambling; nutzen wir als 'seed' Parameter
    eng = qmc.Sobol(d=d, scramble=bool(scramble), seed=seed)
    if int(skip) > 0:
        _ = eng.reset()
        _ = eng.random_base2(m=0)  # reset needed for some versions
        # 'fast-forward' via draw and discard
        _ = eng.random(n=int(skip))

    U = eng.random(n)
    X = lower + U * (upper - lower)
    return X
