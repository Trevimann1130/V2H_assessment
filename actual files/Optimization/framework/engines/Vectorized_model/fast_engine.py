# Optimization/framework/engines/Vectorized_model/fast_engine.py
from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Any, List

# EngineAdapter robust importieren (je nach Projektstruktur)
try:
    from Optimization.framework.Problem.builder import EngineAdapter
except Exception:
    from Optimization.framework.Problem import EngineAdapter  # nur Import-Fallback, keine Werte-Fallbacks!

# Kostenmodell (NPC, arbeitet mit Jahreswerten)
from Cost_model.financial_model import calculate_npc_yearly

# Daten+Profile (Standortparameter, Last-/PV-Profile)
from Data.data import get_parameters, load_profiles

# Profile vorbereiten (z. B. HP-Strom, etc.) – vektorisierte Vorbereitung
from Technical_model.energy_system.precompute.precompute import prepare_profiles

# Vektorisierte Jahres-Systemmodelle (FAST)
from Technical_model.energy_system.runners.system_model_core import (
    simulate_energy_system,
    simulate_energy_system_with_v2h,
)


def _require(d: Dict[str, Any], key: str) -> Any:
    """Pflichtfeld aus Dict holen – ohne Fallbacks."""
    if key not in d:
        raise KeyError(f"[FastEngine] Fehlendes Pflichtfeld '{key}' im Designpunkt {list(d.keys())}")
    return d[key]


def _require_attr(obj: Any, name: str):
    """Pflichtattribut aus Settings holen – ohne Fallbacks."""
    if not hasattr(obj, name):
        raise AttributeError(f"[FastEngine] Settings.engine.{name} ist nicht gesetzt")
    return getattr(obj, name)


def _sum(result: Dict[str, Any], key: str) -> float:
    """Robustes Summieren eines Ergebnisarrays (fehlender Key -> 0.0)."""
    arr = result.get(key, None)
    if arr is None:
        return 0.0
    a = np.asarray(arr)
    return float(np.sum(a))


class FastEngine(EngineAdapter):
    """
    FAST/vektorisierte Ein-Jahres-Engine.
    Erwartet Designvariablen in der Reihenfolge von settings.bounds.names (z. B. ["pv_kwp","bess_kwh"]).
    Liefert (F, G) passend zu settings.objectives / settings.constraints.
    """

    def __init__(self, settings):
        self.s = settings

        # ---- Settings prüfen (ohne Fallbacks) ----
        eng = self.s.engine
        self.location: str = _require_attr(eng, "location")
        self.use_v2h: bool = bool(_require_attr(eng, "use_v2h"))
        self.ec_share_import: float = float(_require_attr(eng, "ec_share_import"))
        self.ec_share_export: float = float(_require_attr(eng, "ec_share_export"))

        self.obj_names = list(getattr(settings.objectives, "names", []))

        # Optional, aber häufig vorhanden (wir erzwingen hier NICHT, weil technische Modelle evtl. nicht alle brauchen):
        self.n_households: int | None = getattr(eng, "n_households", None)
        self.n_evs: int | None = getattr(eng, "n_evs", None)
        self.n_evs_bidirectional: int | None = getattr(eng, "n_evs_bidirectional", None)
        self.system_id: str | None = getattr(eng, "system_id", None)  # z. B. "PV_BESS_HP_EV" vs. "PV_BESS_HP_V2H"

        # ---- Standort-Parameter und Profile laden ----
        self.base_params: Dict[str, Any] = get_parameters(self.location)
        self.base_params["location"] = self.location  # explizit

        # EC-Shares strikt aus Settings in Params spiegeln (ohne Defaults)
        if "EC" not in self.base_params:
            self.base_params["EC"] = {}
        self.base_params["EC"]["share"] = self.ec_share_import
        self.base_params["EC"]["export_share"] = self.ec_share_export

        # optionale Zähler in Params spiegeln (nur wenn gesetzt)
        if self.n_households is not None:
            self.base_params["N_HH"] = int(self.n_households)
        if self.n_evs is not None:
            self.base_params["N_EV"] = int(self.n_evs)
        if self.n_evs_bidirectional is not None:
            self.base_params["N_EV_bidirectional"] = int(self.n_evs_bidirectional)

        # Rohprofile + aufbereitete Profile (ein Jahr)
        profiles_raw = load_profiles(self.location)
        # do_hp_electricity=True (wie in deinem bestehenden FAST-Workflow)
        self.profiles = prepare_profiles(self.base_params, profiles_raw, do_hp_electricity=True, do_coeffs=False)

        # Jahres-Last (für Autarkie-Constraint)
        load_arr = np.asarray(self.profiles.get("load", []), dtype=float)
        self.year_load_kwh: float = float(np.sum(load_arr)) if load_arr.size else 0.0

        # Vorbereitungen: Namen der Ziele/Constraints
        self.obj_names: List[str] = list(self.s.objectives.names)
        self.con_names: List[str] = list(self.s.constraints.names or [])
        self.con_senses: List[str] = list(self.s.constraints.senses or [])
        self.con_rhs: List[float] = list(self.s.constraints.rhs or [])

    # ---------------------------------------------------------------------
    # Kern: ein Punkt -> Simulation (ein Jahr) -> KPIs (Ziele/Constraints)
    # ---------------------------------------------------------------------

    def _simulate_year_flows(self, pv_kwp: float, bess_kwh: float) -> Dict[str, Any]:
        """
        Ruft das vektorisierte Jahres-Systemmodell auf (ein Jahr).
        Erwartet, dass simulate_* Dicts mit Zeitreihen (np.ndarray) zurückgeben.
        """
        params = dict(self.base_params)
        params["pv_size"] = float(pv_kwp)
        params["battery_capacity_kWh"] = float(bess_kwh)

        if self.use_v2h:
            sim = simulate_energy_system_with_v2h
        else:
            sim = simulate_energy_system

        # Ergebnis enthält Zeitreihen (kWh/h) für ein Jahr
        res = sim(params, self.profiles, pv_size=float(pv_kwp))
        return res

    def _kpis_from_year_flows(self, flows: dict, pv_kwp: float, bess_kwh: float):
        """
        Ziele (F) und Constraints (G) aus Jahres-Zeitreihen des Technical Models ableiten.

        Annahmen:
          - flows enthält 1-Jahres-Zeitreihen (np.ndarray) mit Keys:
              "grid_import", "grid_export", "ec_import_from_pv", "ec_import_from_ev",
              "bess_charged", "heatpump_results_heating", "heatpump_results_cooling"
          - NPC arbeitet mit *Jahreswerten* (€/NPV). PEF nutzt Lebensdauer-Summen (wie in deinem alten evaluate).
          - Keine Fallbacks: benötigte Parameter sind in self.base_params gesetzt.
        """
        # ---- Parameter ohne Fallbacks aus Settings/Data ----
        params = self.base_params
        L = int(params["lifetime"])
        PV = params["PV"]
        BESS = params["BESS"]
        Grid = params["Grid"]
        EV = params.get("EV", {})  # EV darf fehlen → dann operativer EV-PEF = 0

        # ---- Jahres-Summen aus Zeitreihen (Technical Model gibt arrays zurück) ----
        E_imp_grid_Y = _sum(flows, "grid_import")
        E_exp_grid_Y = _sum(flows, "grid_export")
        E_imp_ec_pv_Y = _sum(flows, "ec_import_from_pv")
        E_imp_ec_ev_Y = _sum(flows, "ec_import_from_ev")
        E_bess_thr_Y = _sum(flows, "bess_charged")  # geladene Energiemenge = Durchsatz
        E_hp_heat_Y = _sum(flows, "heatpump_results_heating")
        E_hp_cool_Y = _sum(flows, "heatpump_results_cooling")

        # ---- Lebensdauer-Summen (für PEF etc.) aus Jahreswerten ableiten ----
        E_imp_grid_L = E_imp_grid_Y * L
        E_exp_grid_L = E_exp_grid_Y * L
        E_imp_ec_pv_L = E_imp_ec_pv_Y * L
        E_imp_ec_ev_L = E_imp_ec_ev_Y * L
        E_bess_thr_L = E_bess_thr_Y * L
        E_hp_heat_L = E_hp_heat_Y * L
        E_hp_cool_L = E_hp_cool_Y * L

        # ---- Ziele berechnen gemäß settings.objectives.names ----
        F_values_by_name = {}

        # NPC [€] via Financial Model (benötigt *Jahres*-Energieflüsse)
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
            F_values_by_name["npc_eur"] = float(npc)

        # PEF [Pt] (embodied + operational) – Lebensdauer-Summen
        if "pef_pt" in self.obj_names:
            pef_embodied = (
                    float(PV["PEF"]) * float(pv_kwp) +
                    float(BESS["PEF"]) * float(bess_kwh)
            )
            pef_oper = (
                    float(Grid["PEF"]) * E_imp_grid_L
                    + float(PV["PEF"]) * E_imp_ec_pv_L
                    + float(EV.get("PEF", 0.0)) * E_imp_ec_ev_L
            )
            F_values_by_name["pef_pt"] = float(pef_embodied + pef_oper)

        # Abwärtskompatible Ziele (falls noch genutzt)
        if "grid_import_kwh" in self.obj_names:
            F_values_by_name["grid_import_kwh"] = float(E_imp_grid_L)

        if "total_cost_eur" in self.obj_names and "npc_eur" in F_values_by_name:
            F_values_by_name["total_cost_eur"] = F_values_by_name["npc_eur"]

        # ---- Rückgabe in exakt der Reihenfolge der Settings ----
        F = [float(F_values_by_name[name]) for name in self.obj_names]
        # derzeit keine Constraints hier berechnet → leeres Array
        G = []
        return np.asarray(F, dtype=float), np.asarray(G, dtype=float)

    # ---------------------------------------------------------------------
    # Pymoo-Entry: mehrere Punkte auf einmal auswerten
    # ---------------------------------------------------------------------

    def evaluate(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Erwartet X als (n, d). Spaltenreihenfolge = settings.bounds.names.
        Rückgabe:
            F: (n, n_obj)
            G: (n, n_con)  – ggf. shape (n, 0) wenn keine Constraints.
        """
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        var_names = list(self.s.bounds.names)  # z. B. ["pv_kwp", "bess_kwh"]
        if X.shape[1] != len(var_names):
            raise ValueError(
                f"[FastEngine] X hat {X.shape[1]} Spalten, erwartet {len(var_names)} (names={var_names})."
            )

        F_rows: List[List[float]] = []
        G_rows: List[List[float]] = []

        for row in X:
            point = dict(zip(var_names, row))
            pv = float(_require(point, "pv_kwp"))
            bess = float(_require(point, "bess_kwh"))

            flows = self._simulate_year_flows(pv, bess)
            F, G = self._kpis_from_year_flows(flows, pv, bess)

            F_rows.append(F)
            G_rows.append(G)

        F_out = np.asarray(F_rows, dtype=float) if F_rows else np.zeros((0, len(self.obj_names)), dtype=float)
        if len(self.con_names) == 0:
            G_out = np.zeros((F_out.shape[0], 0), dtype=float)
        else:
            G_out = np.asarray(G_rows, dtype=float) if G_rows else np.zeros((F_out.shape[0], len(self.con_names)))

        return F_out, G_out
