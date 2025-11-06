from __future__ import annotations
import pandas as pd
from ..utils import ensure_columns

# Use your registry exactly as provided:
from Optimization.framework.Orchestrator.registry import resolve_engine

def _eval(engine_obj, X: pd.DataFrame, params):
    out = engine_obj.evaluate(X, params)
    if isinstance(out, tuple) and len(out)==2:
        f,k = out
    elif isinstance(out, dict) and "flows" in out and "kpis" in out:
        f,k = pd.DataFrame(out["flows"]), pd.DataFrame(out["kpis"])
    else:
        raise TypeError("Engine evaluate must return (flows_df,kpis_df) or {'flows','kpis'}")
    if not isinstance(f,pd.DataFrame) or not isinstance(k,pd.DataFrame):
        raise TypeError("Engine outputs must be pandas.DataFrame")
    return f,k

def run_fast(X: pd.DataFrame, params, xschema_names):
    Eng = resolve_engine("fast"); eng = Eng()
    ensure_columns(X, xschema_names)
    return _eval(eng, X, params)

def run_gold(X: pd.DataFrame, params, xschema_names):
    Eng = resolve_engine("gold"); eng = Eng()
    ensure_columns(X, xschema_names)
    return _eval(eng, X, params)
