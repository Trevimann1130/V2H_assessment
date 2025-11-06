# Technical_model/energy_system/runners/system_model_precomputed.py
# -----------------------------------------------------------------------------
# STRIKT & FLAGLOS:
# - keine stillen Defaults/Fallbacks
# - erwartete Flüsse müssen geliefert werden (res[k] = pd.Series mit DatetimeIndex)
# - Bilanzsummen explizit übergeben (generation_keys, consumption_keys)
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Any, Dict, Iterable, Set
import numpy as np
import pandas as pd


def _require_precomputed(params: Dict[str, Any], profiles: Dict[str, Any]) -> None:
    if not isinstance(params, dict):
        raise ValueError("params must be a dict.")
    if not isinstance(profiles, dict):
        raise ValueError("profiles must be a dict.")
    if "load" not in profiles:
        raise ValueError("profiles['load'] is required.")
    load = profiles["load"]
    if not isinstance(load, pd.Series):
        raise ValueError("profiles['load'] must be a pandas Series.")
    if not isinstance(load.index, pd.DatetimeIndex):
        raise ValueError("profiles['load'] index must be a DatetimeIndex.")


def _validate_res(
    res: Dict[str, Any],
    idx: pd.DatetimeIndex,
    expected_keys: Iterable[str],
) -> Set[str]:
    if not isinstance(res, dict):
        raise ValueError("res must be a dict.")
    exp: Set[str] = set(expected_keys or [])
    if not exp:
        raise ValueError("expected_keys must be a non-empty iterable.")

    missing = [k for k in exp if k not in res]
    if missing:
        raise KeyError(f"Missing expected result keys: {missing}")

    for k in exp:
        s = res[k]
        if not isinstance(s, pd.Series):
            raise TypeError(f"res['{k}'] must be a pandas Series.")
        if not s.index.equals(idx):
            raise ValueError(f"Index mismatch for '{k}': must equal profiles['load'].index.")
        if s.isna().any():
            raise ValueError(f"res['{k}'] contains NaNs.")

    return exp


def set_ec_shares(params: Dict[str, Any], ec_share: float, ec_export_share: float) -> Dict[str, Any]:
    """Strikt: EC-Block muss existieren; Werte ∈ [0,1]."""
    if "EC" not in params or not isinstance(params["EC"], dict):
        raise KeyError("params['EC'] must exist and be a dict before setting EC shares.")
    if not (0.0 <= float(ec_share) <= 1.0):
        raise ValueError("ec_share must be within [0,1].")
    if not (0.0 <= float(ec_export_share) <= 1.0):
        raise ValueError("ec_export_share must be within [0,1].")

    new_params = dict(params)
    ec_block = dict(new_params["EC"])
    ec_block["share"] = float(ec_share)
    ec_block["export_share"] = float(ec_export_share)
    new_params["EC"] = ec_block
    return new_params


def _hourly_table(
    res: Dict[str, Any],
    profiles: Dict[str, Any],
    params: Dict[str, Any],
    *,
    expected_keys: Iterable[str],
    generation_keys: Iterable[str],
    consumption_keys: Iterable[str],
    grid_import_key: str = "grid_import",
    grid_export_key: str = "grid_export",
) -> pd.DataFrame:
    """Baut eine Stundentabelle ausschließlich aus explizit erwarteten Keys (strikt)."""
    _require_precomputed(params, profiles)
    idx = profiles["load"].index

    exp = _validate_res(res, idx, expected_keys)
    gen_set = set(generation_keys or [])
    con_set = set(consumption_keys or [])

    if not gen_set.issubset(exp):
        missing = list(gen_set - exp)
        raise KeyError(f"generation_keys not in expected_keys: {missing}")
    if not (set([grid_import_key, grid_export_key]).issubset(exp)):
        raise KeyError(f"'{grid_import_key}' and '{grid_export_key}' must be in expected_keys.")
    # 'load' kommt aus profiles und muss NICHT in exp stehen
    if not con_set.issubset(exp | {"load"}):
        missing = list(con_set - (exp | {"load"}))
        raise KeyError(f"consumption_keys not in expected_keys (or 'load'): {missing}")

    tab = pd.DataFrame(index=idx)
    tab["load"] = profiles["load"].astype("float64")

    for k in exp:
        tab[k] = res[k].astype("float64")

    generation = sum((tab[k] for k in gen_set), start=pd.Series(0.0, index=idx))
    consumption = tab["load"] + sum((tab[k] for k in con_set if k != "load"),
                                    start=pd.Series(0.0, index=idx))

    tab["net_balance_before_grid"] = generation - consumption
    tab["imbalance_after_grid"] = tab["net_balance_before_grid"] - (tab[grid_export_key] - tab[grid_import_key])
    return tab


def check_mass_balances(
    res: Dict[str, Any],
    profiles: Dict[str, Any],
    params: Dict[str, Any],
    *,
    expected_keys: Iterable[str],
    generation_keys: Iterable[str],
    consumption_keys: Iterable[str],
    grid_import_key: str = "grid_import",
    grid_export_key: str = "grid_export",
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """Strikter Bilanz-Check (keine Ergänzungen)."""
    table = _hourly_table(
        res, profiles, params,
        expected_keys=expected_keys,
        generation_keys=generation_keys,
        consumption_keys=consumption_keys,
        grid_import_key=grid_import_key,
        grid_export_key=grid_export_key,
    )
    imb = table["imbalance_after_grid"]
    max_abs = float(np.abs(imb).max()) if len(imb) else 0.0
    mean_abs = float(np.abs(imb).mean()) if len(imb) else 0.0
    ok = bool(max_abs <= atol)

    return {
        "summary": {
            "ok": ok,
            "max_abs_imbalance": max_abs,
            "mean_abs_imbalance": mean_abs,
            "n_hours": int(len(imb)),
        },
        "imbalance": imb,
        "table": table,
    }
