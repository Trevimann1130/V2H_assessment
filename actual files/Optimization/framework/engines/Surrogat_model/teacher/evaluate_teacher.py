# Optimization/framework/engines/Surrogat_model/teacher/evaluate_teacher.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Tuple

from Data.data import get_parameters, load_profiles
from Technical_model.energy_system.precompute.precompute import prepare_profiles
from Technical_model.energy_system.runners.system_model_core import (
    simulate_energy_system,
    simulate_energy_system_with_v2h,
)

# ---- helpers ----

def _sum(result: Dict[str, Any], key: str) -> float:
    arr = result.get(key, None)
    if arr is None:
        return 0.0
    a = np.asarray(arr)
    return float(np.sum(a))

def _year_flows(params: Dict[str, Any],
                profiles: Dict[str, Any],
                use_v2h: bool,
                pv_kwp: float,
                bess_kwh: float) -> Dict[str, float]:
    """Simuliere 1 Jahr, gib Jahres-Summen der relevanten Flüsse zurück."""
    p = dict(params)
    p["pv_size"] = float(pv_kwp)
    p["battery_capacity_kWh"] = float(bess_kwh)

    sim = simulate_energy_system_with_v2h if use_v2h else simulate_energy_system
    res = sim(p, profiles, pv_size=float(pv_kwp))

    flows_year = {
        "E_import_grid_kWh":      _sum(res, "grid_import"),
        "E_export_grid_kWh":      _sum(res, "grid_export"),
        "E_import_ec_pv_kWh":     _sum(res, "ec_import_from_pv"),
        "E_import_ec_ev_kWh":     _sum(res, "ec_import_from_ev"),
        "E_bess_throughput_kWh":  _sum(res, "bess_charged"),
        # optional:
        "E_ev_charged_kWh":       _sum(res, "ev_charged"),
        "E_ev_discharged_kWh":    _sum(res, "ev_discharged"),
        "E_hp_heat_kWh":          _sum(res, "heatpump_results_heating"),
        "E_hp_cool_kWh":          _sum(res, "heatpump_results_cooling"),
        "E_pv_gen_kWh":           _sum(res, "pv_generation"),
    }
    return flows_year

# ---- main API ----

def evaluate_teacher_dataset(settings,
                             X: np.ndarray,
                             targets: List[str] | None = None,
                             batch_size: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Liefert:
      YF: Matrix der LEBENSDAUER-SUMMEN der angeforderten Flow-Targets (Spaltenreihenfolge = targets)
      YG: (derzeit leer) Constraints-Säulen, falls du später welche als Surrogates lernst
    """
    eng = settings.engine
    loc = eng.location
    use_v2h = bool(eng.use_v2h)

    base_params = get_parameters(loc)
    base_params["location"] = loc
    # EC-Shares aus Settings überschreiben
    if "EC" not in base_params:
        base_params["EC"] = {}
    base_params["EC"]["share"] = float(eng.ec_share_import)
    base_params["EC"]["export_share"] = float(eng.ec_share_export)

    # optionale Skalierungen
    base_params["N_HH"] = int(eng.N_HH)
    base_params["N_EV"] = int(eng.N_EV_total)
    base_params["N_EV_bidirectional"] = int(eng.N_EV_bidirectional)

    profiles_raw = load_profiles(loc)
    profiles = prepare_profiles(base_params, profiles_raw, do_hp_electricity=True, do_coeffs=False)

    L = int(base_params["lifetime"])

    # Zielspalten
    tnames = list(targets or settings.surrogate_train.targets)

    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    YF = np.zeros((X.shape[0], len(tnames)), dtype=float)

    # Einfache sequenzielle Abarbeitung (klein, verständlich)
    for i, (pv, bess) in enumerate(X):
        year = _year_flows(base_params, profiles, use_v2h, pv_kwp=float(pv), bess_kwh=float(bess))
        # → Lebensdauer-Summen
        life = {k: float(year.get(k, 0.0)) * L for k in year.keys()}
        YF[i, :] = [life.get(t, 0.0) for t in tnames]

    # Aktuell keine G-Targets
    YG = np.zeros((X.shape[0], 0), dtype=float)
    return YF, YG
