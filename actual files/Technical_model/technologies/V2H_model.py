# Technical_model/technologies/V2H_model.py

import numpy as np


def get_c_rate(temp, c_rate_table):
    """
    Liefert die C-Rate (DC-seitig) für eine gegebene Temperatur.
    """
    temp = float(temp)
    for entry in c_rate_table:
        if entry["min_temp"] <= temp < entry["max_temp"]:
            return float(entry["c_rate"])
    # Fallback, falls Temperatur in keinem Intervall liegt
    return 0.1


def get_availability_for_hour(hour, availability_profile):
    """
    Bernoulli-Sampling gegen ein Verfügbarkeitsprofil (0..1).
    """
    availability_value = float(availability_profile[hour])
    return np.random.random() <= availability_value


def simulate_v2h_battery(temperature, params):
    """
    Baut stündliche Lade-/Entlade-Leistungsgrenzen (kW_AC) für EVs
    basierend auf C-Rate-Tabellen (DC), Effizienzen und Port-Limits.

    Rückgabe-Keys:
      - charge_power_limit:           kW_AC (nur C-Rate/η, ohne Port-Limit)
      - charge_power_limit_capped:    kW_AC (mit Port-Limit gekappt)
      - discharge_power_limit:        kW_AC (mit Port-Limit gekappt)
      - temperature, temperature_series
      - c_rate_charge, c_rate_discharge
    """
    capacity_kWh = float(params["capacity_kWh"])
    charge_c_rate_table = params["charge_c_rate_table"]
    discharge_c_rate_table = params["discharge_c_rate_table"]

    charging_efficiency = float(params["charging_efficiency"])        # η_ch
    discharging_efficiency = float(params["discharging_efficiency"])  # η_dis

    n_steps = len(temperature)

    charge_power_limit = []             # kW_AC (nur C-Rate/η, ohne Port)
    discharge_power_limit = []          # kW_AC (mit Port gekappt)
    temperature_series = []
    c_rate_charge_series = []
    c_rate_discharge_series = []
    charge_power_limit_series = []      # kW_AC (mit Port gekappt)

    # ----------------- LOOP -----------------
    for t in range(n_steps):
        # Temperatur
        T = float(temperature[t])

        # C-Rates (DC)
        c_rate_charge = get_c_rate(T, charge_c_rate_table)
        c_rate_discharge = get_c_rate(T, discharge_c_rate_table)

        # DC-Leistungsgrenzen (kW_DC)
        max_charge_power_dc = c_rate_charge * capacity_kWh
        max_discharge_power_dc = c_rate_discharge * capacity_kWh

        # Auf AC mappen (Effizienzen NICHT doppelt einrechnen):
        max_charge_power = max_charge_power_dc / max(charging_efficiency, 1e-9)

        max_discharge_power = min(
            max_discharge_power_dc * max(discharging_efficiency, 1e-9),
            float(params.get("max_discharge_power", params.get("max_charge_power", float("inf"))))
        )

        # Port-Limit für Laden (z. B. 11 kW)
        port_ch = float(params.get("max_charge_power", float("inf")))
        charge_power_capped = min(max_charge_power, port_ch)

        # Serien füllen
        temperature_series.append(T)
        c_rate_charge_series.append(c_rate_charge)
        c_rate_discharge_series.append(c_rate_discharge)

        # Laden: roh (nur C-Rate/η) + gekappt (Port)
        charge_power_limit.append(max_charge_power)              # ohne Port
        charge_power_limit_series.append(charge_power_capped)    # mit Port

        # Entladen: direkt als gekappte AC-Leistung hinterlegen
        discharge_power_limit.append(max_discharge_power)

    # --------------- /LOOP ------------------

    return {
        "charge_power_limit": np.array(charge_power_limit, dtype=float),
        "charge_power_limit_capped": np.array(charge_power_limit_series, dtype=float),
        "discharge_power_limit": np.array(discharge_power_limit, dtype=float),
        "temperature": np.array(temperature, dtype=float),
        "temperature_series": np.array(temperature_series, dtype=float),
        "c_rate_charge": np.array(c_rate_charge_series, dtype=float),
        "c_rate_discharge": np.array(c_rate_discharge_series, dtype=float),
    }
