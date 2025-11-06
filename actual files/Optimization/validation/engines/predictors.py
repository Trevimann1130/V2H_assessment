from __future__ import annotations
import pandas as pd
from ..utils import ensure_columns
from Optimization.framework.Orchestrator.registry import resolve_engine

def run_surrogate(X: pd.DataFrame, params, xschema_names):
    Eng = resolve_engine("surrogate"); eng = Eng()
    ensure_columns(X, xschema_names)
    out = eng.evaluate(X, params)
    if isinstance(out, tuple) and len(out)==2:
        f,k = out
    elif isinstance(out, dict) and "flows" in out and "kpis" in out:
        f,k = pd.DataFrame(out["flows"]), pd.DataFrame(out["kpis"])
    else:
        raise TypeError("Surrogate evaluate must return (flows_df,kpis_df) or {'flows','kpis'}")
    return f,k
