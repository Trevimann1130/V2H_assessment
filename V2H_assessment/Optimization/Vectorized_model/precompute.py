# Optimization/Vectorized_model/precompute.py

from typing import Dict, Any
import numpy as np
import pandas as pd

# Nur über Heatpump-API importieren – kein direkter heating_and_cooling-Import
from Technical_model.heatpump_model import (
    simulate_heatpump_heating_system,
    simulate_heatpump_cooling_system,
)


def _as_np(a):
    return np.asarray(a, dtype=float)


def prepare_profiles(params: Dict[str, Any],
                     profiles: Dict[str, Any],
                     do_hp_electricity: bool = True,
                     do_coeffs: bool = False) -> Dict[str, Any]:
    """
    Erzeugt das 1-Jahres-Precompute-Paket für das FAST-Modell.
    Erwartete Eingaben in `profiles` (aus Data.load_profiles):
    - 'load' (kWh/h), 'pv_generation' (kWh/h bei Referenz-kWp), 'T_outdoor',
      'usage_profile' (DataFrame), optional: 'timestamps',
      optional (für V2H): 'availability_profile', 'driving_profile', 'min_SOC', 'ev_profiles'.

    Ausgaben (immer als NumPy-Vektoren, außer usage_profile als DF):
    - 'hp_elec_heat', 'hp_elec_cool', 'hotwater_HH_kWh',
      'pv_generation', 'load', 'T_outdoor', 'usage_profile' (+ EV-Profile falls vorhanden).
    """
    out: Dict[str, Any] = {}

    # 1) Pflicht-Pass-Through
    required = ["load", "pv_generation", "T_outdoor", "usage_profile"]
    missing = [k for k in required if k not in profiles]
    assert not missing, f"Profiles fehlen Keys: {missing}"

    out["load"] = _as_np(profiles["load"])
    out["pv_generation"] = _as_np(profiles["pv_generation"])
    out["T_outdoor"] = _as_np(profiles["T_outdoor"])
    out["usage_profile"] = profiles["usage_profile"]  # als DF belassen

    # Timestamps – falls vorhanden durchreichen, sonst 1 Jahr generieren
    if "timestamps" in profiles:
        out["timestamps"] = profiles["timestamps"]
    else:
        nH = len(out["load"])
        out["timestamps"] = pd.date_range(start="2023-01-01", periods=nH, freq="h")

    # 2) Optional-Pass-Through für V2H
    for k in ["availability_profile", "driving_profile", "min_SOC"]:
        if k in profiles:
            out[k] = _as_np(profiles[k])

    # 3) Warmwasser vorrechnen (aus m²-Bedarf und Fläche)
    usage_df = profiles["usage_profile"]
    assert "Warmwasserbedarf_W_m2" in usage_df.columns, "usage_profile fehlt Spalte 'Warmwasserbedarf_W_m2'"
    hotwater_W_m2 = usage_df["Warmwasserbedarf_W_m2"].to_numpy(dtype=float)
    A_floor = float(params["building"]["A_floor"])
    out["hotwater_HH_kWh"] = (hotwater_W_m2 * A_floor) / 1000.0  # W → kWh

    # 4) Wärmepumpen-Strom vorrechnen (ein Jahr)
    if do_hp_electricity:
        hpH = simulate_heatpump_heating_system(params=params, profiles=profiles)
        hpC = simulate_heatpump_cooling_system(params=params, profiles=profiles)
        out["hp_elec_heat"] = _as_np(hpH["electric_consumption_series"])
        out["hp_elec_cool"] = _as_np(hpC["electric_consumption_series"])
    else:
        nH = len(out["load"])
        out["hp_elec_heat"] = np.zeros(nH, dtype=float)
        out["hp_elec_cool"] = np.zeros(nH, dtype=float)

    # 5) EV-Profile absichern (für NoV2H auch Dummy erzeugen)
    nH = len(out["load"])
    N_EV = int(params.get("N_EV", 1))
    if "ev_profiles" in profiles:
        out["ev_profiles"] = _as_np(profiles["ev_profiles"])
    else:
        # Dummy-Profil: keine zusätzliche EV-Last
        out["ev_profiles"] = np.zeros((nH, N_EV))

    # 6) (Optional) Platzhalter für Koeffizienten
    if do_coeffs:
        out["coeffs"] = {"note": "hier könnten Regressions-Koeffizienten stehen"}

    return out
