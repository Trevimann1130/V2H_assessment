from __future__ import annotations
import numpy as np
import pandas as pd

from ..utils import XSchema

def pick_fast(df_all: pd.DataFrame, strategy: str, k_total: int | None, xschema: XSchema, rng: np.random.Generator) -> pd.DataFrame:
    if strategy == "all": return df_all.copy()
    if not (k_total and k_total>0): raise ValueError("fast_subset.k_total must be >0 for strategy!='all'")
    if strategy == "fixed_k": return df_all.head(k_total).copy()
    if strategy == "edges":
        idx = set()
        for name in xschema.names:
            idx.add(int(df_all[name].idxmin())); idx.add(int(df_all[name].idxmax()))
        chosen = list(idx)[:k_total]
        rest_need = max(0, k_total - len(chosen))
        if rest_need:
            rest = df_all.drop(index=chosen)
            choice = rng.choice(len(rest), size=min(rest_need,len(rest)), replace=False)
            return pd.concat([df_all.loc[chosen], rest.iloc[choice]], axis=0).reset_index(drop=True)
        return df_all.loc[chosen].reset_index(drop=True)
    if strategy in ("pareto_focus","diverse_kcenter"):
        choice = rng.choice(len(df_all), size=min(k_total,len(df_all)), replace=False)
        return df_all.iloc[choice].copy()
    if strategy == "match_gold":
        raise RuntimeError("fast_subset.strategy='match_gold' must be applied after gold subset is known.")
    raise ValueError(f"Unknown fast_subset.strategy='{strategy}'")

def pick_gold(df_all: pd.DataFrame, strategy: str, k_total: int, xschema: XSchema, rng: np.random.Generator) -> pd.DataFrame:
    if not (k_total and k_total>0): raise ValueError("gold_subset.k_total must be >0")
    if strategy == "fixed_k": return df_all.head(k_total).copy()
    if strategy == "mixed":
        idx = set()
        for name in xschema.names:
            idx.add(int(df_all[name].idxmin())); idx.add(int(df_all[name].idxmax()))
        sel = df_all.loc[list(idx)] if idx else df_all.head(0)
        rest_need = max(0, k_total - len(sel))
        if rest_need:
            rest = df_all.drop(index=sel.index)
            choice = rng.choice(len(rest), size=min(rest_need, len(rest)), replace=False)
            sel = pd.concat([sel, rest.iloc[choice]], axis=0)
        return sel.reset_index(drop=True)
    if strategy in ("pareto_focus","top_error"):
        choice = rng.choice(len(df_all), size=min(k_total,len(df_all)), replace=False)
        return df_all.iloc[choice].copy()
    raise ValueError(f"Unknown gold_subset.strategy='{strategy}'")
