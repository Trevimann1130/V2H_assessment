# Optimization/framework/engines/Gold/gold_engine.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np

from Optimization.framework.Problem.builder import EngineAdapter

# Technisches (nicht-vektorisiertes) Jahresmodell → "Gold"
# Achtung: Import aus dem *precomputed* Runner, nicht aus dem Core (FAST)
from Technical_model.energy_system.runners.system_model_core import (
    simulate_energy_system,
    simulate_energy_system_with_v2h,
)

# Daten / Profile
from Data.data import get_parameters, load_profiles
from Technical_model.energy_system.precompute.precompute import prepare_profiles

# Finanzmodell
from Cost_model.financial_model import calculate_npc_yearly


# -------- kleine Helfer (ohne Fallbacks) --------
def _require_attr(obj: Any, name: str):
    if not hasattr(obj, name):
        raise AttributeError(f"[GoldEngine] Settings.engine.{name} ist nicht gesetzt")
    return getattr(obj, name)

def _require(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"[GoldEngine] Fehlendes Pflichtfeld '{key}' im Designpunkt {list(d.keys())}")
    return d[key]

def _sum(result: Dict[str, Any], key: str) -> float:
    arr = result.get(key, None)
    if arr is None:
        return 0.0
    a = np.asarray(arr)
    return float(np.sum(a))


class GoldEngine(EngineAdapter):
    """
    Referenz-Engine (nicht-vektorisiertes Modell).
    Erwartet X gemäß settings.bounds.names (z. B. ["pv_kwp","bess_kwh"]).
    Liefert (F, G) gemäß settings.objectives / settings.constraints.
    """

    def __init__(self, settings):
        self.s = settings

        # ---- Settings prüfen/ziehen (ohne Fallbacks) ----
        eng = self.s.engine
        self.location: str = _require_attr(eng, "location")
        self.use_v2h: bool = bool(_require_attr(eng, "use_v2h"))
        self.ec_share_import: float = float(_require_attr(eng, "ec_share_import"))
        self.ec_share_export: float = float(_require_attr(eng, "ec_share_export"))

        # Optional gespiegelt, falls vorhanden
        self.n_households: int | None = getattr(eng, "N_HH", None)
        self.n_evs: int | None = getattr(eng, "N_EV_total", None)
        self.n_evs_bidirectional: int | None = getattr(eng, "N_EV_bidirectional", None)

        # ---- Parameter + Profile laden ----
        self.params_base: Dict[str, Any] = get_parameters(self.location)
        self.params_base["location"] = self.location
        # EC-Anteile strikt aus Settings spiegeln
        if "EC" not in self.params_base:
            self.params_base["EC"] = {}
        self.params_base["EC"]["share"] = self.ec_share_import
        self.params_base["EC"]["export_share"] = self.ec_share_export
        # Skalierung spiegeln (nur wenn gesetzt)
        if self.n_households is not None:
            self.params_base["N_HH"] = int(self.n_households)
        if self.n_evs is not None:
            self.params_base["N_EV"] = int(self.n_evs)
        if self.n_evs_bidirectional is not None:
            self.params_base["N_EV_bidirectional"] = int(self.n_evs_bidirectional)

        profiles_raw = load_profiles(self.location)
        self.profiles = prepare_profiles(self.params_base, profiles_raw, do_hp_electricity=True, do_coeffs=False)

        # Namen der Ziele/Constraints
        self.obj_names: List[str] = list(self.s.objectives.names)
        self.con_names: List[str] = list(self.s.constraints.names or [])

    # ---- 1 Jahr simulieren (nicht vektorisiert) ----
    def _simulate_year(self, pv_kwp: float, bess_kwh: float) -> Dict[str, Any]:
        params = dict(self.params_base)
        params["pv_size"] = float(pv_kwp)
        params["battery_capacity_kWh"] = float(bess_kwh)

        sim = simulate_energy_system_with_v2h if self.use_v2h else simulate_energy_system
        return sim(params, self.profiles, pv_size=float(pv_kwp))

    # ---- KPIs aus Jahresflüssen ableiten (NPC / PEF etc.) ----
    def _kpis_from_year_flows(self, flows: dict, pv_kwp: float, bess_kwh: float) -> Tuple[np.ndarray, np.ndarray]:
        params = self.params_base
        L = int(params["lifetime"])
        PV = params["PV"]
        BESS = params["BESS"]
        Grid = params["Grid"]
        EV = params.get("EV", {})

        # Jahres-Summen
        E_imp_grid_Y = _sum(flows, "grid_import")
        E_exp_grid_Y = _sum(flows, "grid_export")
        E_imp_ec_pv_Y = _sum(flows, "ec_import_from_pv")
        E_imp_ec_ev_Y = _sum(flows, "ec_import_from_ev")
        E_bess_thr_Y = _sum(flows, "bess_charged")
        E_hp_heat_Y = _sum(flows, "heatpump_results_heating")
        E_hp_cool_Y = _sum(flows, "heatpump_results_cooling")

        # Lebensdauer-Summen (für PEF)
        E_imp_grid_L = E_imp_grid_Y * L
        E_imp_ec_pv_L = E_imp_ec_pv_Y * L
        E_imp_ec_ev_L = E_imp_ec_ev_Y * L

        # Ziele berechnen in settings-Reihenfolge
        F_by_name: Dict[str, float] = {}

        if "npc_eur" in self.obj_names:
            params_fin = dict(params)
            params_fin["pv_size"] = float(pv_kwp)
            params_fin["battery_capacity_kWh"] = float(bess_kwh)
            npc = calculate_npc_yearly(
                params_fin,
                e_import_grid_year=E_imp_grid_Y,
                e_import_ec_pv_year=E_imp_ec_pv_Y,
                e_import_ec_ev_year=E_imp_ec_ev_Y,
                e_export_grid_year=E_exp_grid_Y,
                e_export_pv_ec_year=0.0,
                e_export_ev_ec_year=0.0,
            )
            F_by_name["npc_eur"] = float(npc)

        if "pef_pt" in self.obj_names:
            pef_embodied = float(PV["PEF"]) * float(pv_kwp) + float(BESS["PEF"]) * float(bess_kwh)
            pef_oper = (
                float(Grid["PEF"]) * E_imp_grid_L
                + float(PV["PEF"]) * E_imp_ec_pv_L
                + float(EV.get("PEF", 0.0)) * E_imp_ec_ev_L
            )
            F_by_name["pef_pt"] = float(pef_embodied + pef_oper)

        if "grid_import_kwh" in self.obj_names:
            F_by_name["grid_import_kwh"] = E_imp_grid_L

        F = [float(F_by_name[name]) for name in self.obj_names]
        G = []  # aktuell keine Constraints
        return np.asarray(F, float), np.asarray(G, float)

    # ---- Pymoo-Entry ----
    def evaluate(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        names = list(self.s.bounds.names)  # z. B. ["pv_kwp","bess_kwh"]
        if X.shape[1] != len(names):
            raise ValueError(f"[GoldEngine] X hat {X.shape[1]} Spalten, erwartet {len(names)} (names={names}).")

        F_rows, G_rows = [], []
        for row in X:
            point = dict(zip(names, row))
            pv = float(_require(point, "pv_kwp"))
            bess = float(_require(point, "bess_kwh"))

            flows = self._simulate_year(pv, bess)
            F, G = self._kpis_from_year_flows(flows, pv, bess)
            F_rows.append(F)
            G_rows.append(G)

        F_out = np.asarray(F_rows, float) if F_rows else np.zeros((0, len(self.obj_names)), float)
        G_out = np.zeros((F_out.shape[0], 0), float) if len(self.con_names) == 0 else np.asarray(G_rows, float)
        return F_out, G_out
