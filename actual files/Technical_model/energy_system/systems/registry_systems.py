# Technical_model/energy_system/systems/registry_systems.py
# -----------------------------------------------------------------------------
# Einzige Quelle der Wahrheit pro system_id:
# - runner callable
# - expected keys (strikt)
# - generation/consumption keys für die Bilanz
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Dict, Any

import numpy as np
import pandas as pd

# 👉 direkter Import, kein Re-Export über runners.__init__ (vermeidet IDE-Warnung)
from Technical_model.energy_system.runners.system_model_precomputed import _hourly_table

from Technical_model.energy_system.systems.PV_BESS_HP_EV import (
    simulate_energy_system as run_ev,
)
from Technical_model.energy_system.systems.PV_BESS_HP_V2H import (
    simulate_energy_system_with_v2h as run_v2h,
)


SYSTEM_SPECS: Dict[str, Dict[str, Any]] = {
    "pv_bess_hp_ev": {
        "runner": run_ev,
        "expected": [
            "pv_generation",
            "grid_import", "grid_export",
            "bess_charged", "bess_discharged",
            "ev_charge_ac",
            "ev_charge_from_pv_ac", "ev_charge_from_bess_ac",
            "ev_charge_from_ec_ac", "ev_charge_from_grid_ac",
            "ec_import_from_pv", "ec_import_from_ev", "ec_export_from_pv",
        ],
        "generation": ["pv_generation", "bess_discharged"],
        "consumption": ["load", "ev_charge_ac", "bess_charged", "ec_export_from_pv"],
    },
    "pv_bess_hp_v2h": {
        "runner": run_v2h,
        "expected": [
            "pv_generation",
            "grid_import", "grid_export",
            "bess_charged", "bess_discharged",
            "ev_charge_ac",
            "ev_charge_from_pv_ac", "ev_charge_from_bess_ac",
            "ev_charge_from_ec_ac", "ev_charge_from_grid_ac",
            "ec_import_from_pv", "ec_import_from_ev", "ec_export_from_pv",
            # "ev_discharged" optional (falls geliefert)
        ],
        "generation": ["pv_generation", "bess_discharged", "ev_discharged"],
        "consumption": ["load", "ev_charge_ac", "bess_charged", "ec_export_from_pv"],
    },
}


def _to_series_dict(res: dict, idx: pd.DatetimeIndex) -> dict:
    """
    Wandelt 1D-Arrays in Series mit Index um; höhere Dimensionen werden durchgereicht.
    """
    out = {}
    for k, v in res.items():
        if isinstance(v, pd.Series):
            out[k] = v.reindex(idx)
        elif isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v)
            if arr.ndim == 1 and arr.size == idx.size:
                out[k] = pd.Series(arr, index=idx, dtype="float64")
            else:
                out[k] = arr  # 2D+ (z. B. ev_discharged (T,N)), Zahlen etc. durchreichen
        else:
            out[k] = v
    return out


def _make_index(res: dict, profiles: dict) -> pd.DatetimeIndex:
    """
    Robust: timestamps aus results > profiles; sonst default per Länge.
    """
    ts = res.get("timestamps", profiles.get("timestamps", None))
    if ts is None:
        # Fallback-Länge: bevorzugt pv_generation, sonst load
        n = None
        if "pv_generation" in res:
            n = np.asarray(res["pv_generation"]).size
        elif "load" in profiles:
            n = np.asarray(profiles["load"]).size
        if n is None:
            raise ValueError("Cannot infer time index length (need pv_generation or load).")
        return pd.date_range("2023-01-01", periods=int(n), freq="h")
    return pd.to_datetime(ts)


def get(system_id: str):
    sid = (system_id or "").lower()
    if sid not in SYSTEM_SPECS:
        raise ValueError(f"Unknown system_id: {system_id}")

    spec = SYSTEM_SPECS[sid]
    runner = spec["runner"]

    def _fn(params, profiles, pv_size, run_checks: bool = False):
        # 1) System rechnen
        res = runner(params, profiles, pv_size)

        # 2) Zeitachse
        idx = _make_index(res, profiles)

        # 3) Arrays -> Series casten
        res = _to_series_dict(res, idx)

        # 4) erwartete Keys dynamisch um optionale erweitern
        expected = list(spec["expected"])
        for opt_key in ("ev_discharged", "hp_electricity", "pv_to_load_direct"):
            if opt_key in res and opt_key not in expected:
                expected.append(opt_key)

        # 5) Stundentabelle bauen (strikt)
        df = _hourly_table(
            res,
            profiles,
            params,
            expected_keys=expected,
            generation_keys=[k for k in spec["generation"] if k in res],
            consumption_keys=[k for k in spec["consumption"] if (k == "load" or k in res)],
            grid_import_key="grid_import",
            grid_export_key="grid_export",
        )
        return res, df

    return _fn
