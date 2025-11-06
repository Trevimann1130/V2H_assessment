from __future__ import annotations
import numpy as np
from pymoo.core.problem import Problem

# EngineAdapter: erwartet .evaluate(X) -> (F, G)
class EngineAdapter:
    def evaluate(self, X: np.ndarray):
        raise NotImplementedError

def build_pymoo_problem(bounds, objectives, constraints, engine: EngineAdapter) -> Problem:
    n_var = len(bounds.names)
    n_obj = len(objectives.names)
    n_con = len(constraints.names)

    xl = np.asarray(bounds.lower, float)
    xu = np.asarray(bounds.upper, float)

    # Pymoo-Problem, das einfach an die Engine delegiert
    class _Problem(Problem):
        def __init__(self):
            super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_con, xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs):
            F, G = engine.evaluate(np.asarray(X, float))
            # Sicherheitsformate (ohne Defaults): Struktur muss bereits stimmen
            F = np.asarray(F, float).reshape((len(X), n_obj))
            if n_con > 0:
                G = np.asarray(G, float).reshape((len(X), n_con))
                out["G"] = G
            out["F"] = F

    return _Problem()
