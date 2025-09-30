# Technical_model/system_model_precomputed.py

import numpy as np
import pandas as pd

from Technical_model.PV_model import simulate_pv_system
from Technical_model.battery_model import simulate_battery_flow
from Technical_model.V2H_model import simulate_v2h_battery


def _require_precomputed(profiles: dict, require_ev: bool = False):
    required = ["hp_elec_heat", "hp_elec_cool", "hotwater_HH_kWh",
                "pv_generation", "load", "T_outdoor"]
    missing = [k for k in required if k not in profiles]
    assert not missing, f"Precompute missing in profiles: {missing}"

    if require_ev:
        ev_keys = ["availability_profile", "driving_profile", "min_SOC"]
        ev_missing = [k for k in ev_keys if k not in profiles]
        assert not ev_missing, f"EV/V2H profiles missing: {ev_missing}"


def simulate_energy_system(params: dict, profiles: dict, pv_size: float) -> dict:
    _require_precomputed(profiles, require_ev=True)

    load = np.asarray(profiles["load"], dtype=float)
    hp_heat = np.asarray(profiles["hp_elec_heat"], dtype=float)
    hp_cool = np.asarray(profiles["hp_elec_cool"], dtype=float)
    hotwater = np.asarray(profiles["hotwater_HH_kWh"], dtype=float)
    base_load = load + hp_heat + hp_cool + hotwater
    n_steps = len(base_load)

    # --- EV Setup ---
    ev_params = params["EV"]
    N_EV_total = int(ev_params["N_EV_total"])

    availability_profile = np.asarray(profiles["availability_profile"], dtype=float)
    driving_profile = np.asarray(profiles["driving_profile"], dtype=float)
    min_soc_data = np.asarray(profiles["min_SOC"], dtype=float)

    # Arrays
    ev_soc = np.ones((n_steps, N_EV_total)) * ev_params["capacity_kWh"] * ev_params["initial_soc"]
    ev_charged = np.zeros((n_steps, N_EV_total))
    trip_loss = np.zeros((n_steps, N_EV_total))   # Fahrenergieverlust wie bei V2H, aber ohne Einspeisung

    pv_results = simulate_pv_system(
        pv_size=float(pv_size),
        load_demand=base_load,
        pv_generation=np.asarray(profiles["pv_generation"], dtype=float),
        params=params
    )
    pv_generation = np.asarray(pv_results["pv_production"], dtype=float)

    battery_in_request = np.zeros(n_steps)
    battery_out_request = np.zeros(n_steps)
    grid_import = np.zeros(n_steps)
    grid_export = np.zeros(n_steps)
    ec_import_from_pv = np.zeros(n_steps)
    ec_import_from_ev = np.zeros(n_steps)  # bleibt leer bei NoV2H

    EC_SHARE = float(params.get("EC_share", 1.0))  # Anteil des Defizits, der aus der EC gedeckt wird (0..1)
    timestep_h = 1.0

    # --- Hauptschleife ---
    for t in range(n_steps):
        if t > 0:
            ev_soc[t] = ev_soc[t-1]

        load_t = base_load[t]
        pv_t = pv_generation[t]

        direct_pv = min(pv_t, load_t)
        pv_surplus = max(0, pv_t - load_t)
        load_deficit = max(0, load_t - pv_t)

        # --- Alle EVs durchgehen ---
        for i in range(N_EV_total):
            # Verfügbarkeit
            is_available = (np.random.random() <= availability_profile[t])

            # Fahrenergiebedarf
            driving_energy = driving_profile[t] * ev_params["capacity_kWh"]

            if not is_available:
                # Fahrzeug unterwegs -> SOC abziehen
                min_soc = ev_params["capacity_kWh"] * min_soc_data[t]
                ev_soc[t, i] = max(min_soc, ev_soc[t, i] - driving_energy)
                trip_loss[t, i] = driving_energy
                continue

            # --- SOC-Grenzen ---
            min_soc = ev_params["capacity_kWh"] * min_soc_data[t]
            max_soc = ev_params["capacity_kWh"] * ev_params["max_soc"]

            # --- EV laden mit PV-Überschuss ---
            if pv_surplus > 0 and ev_soc[t, i] < max_soc:
                max_ev_charge = min(pv_surplus, max_soc - ev_soc[t, i])
                ev_soc[t, i] += max_ev_charge * ev_params["charging_efficiency"]
                ev_charged[t, i] = max_ev_charge
                pv_surplus -= max_ev_charge

            # --- EV laden aus BESS ---
            if pv_surplus <= 0 and load_deficit <= 0 and ev_soc[t, i] < max_soc:
                bess_to_ev = min(max_soc - ev_soc[t, i], ev_params["max_charge_power"] * timestep_h)
                if bess_to_ev > 0:
                    drawn_from_bess = bess_to_ev / ev_params["charging_efficiency"]
                    ev_soc[t, i] += bess_to_ev
                    ev_charged[t, i] += bess_to_ev
                    battery_out_request[t] += drawn_from_bess

            # --- EV laden aus Netz/EC (nur wenn unter minSOC) ---
            if ev_soc[t, i] < min_soc:
                needed = min_soc - ev_soc[t, i]
                drawn_from_supply = needed / ev_params["charging_efficiency"]
                ev_soc[t, i] += needed
                ev_charged[t, i] += needed
                # EC zuerst, Rest aus dem Grid
                ec_take = EC_SHARE * drawn_from_supply
                grid_take = drawn_from_supply - ec_take
                ec_import_from_pv[t] += ec_take
                grid_import[t] += grid_take

        battery_in_request[t] = pv_surplus
        battery_out_request[t] += load_deficit

    # --- BESS Simulation ---
    battery_results = simulate_battery_flow(
        battery_in_request=battery_in_request,
        battery_out_request=battery_out_request,
        capacity_kWh=float(params["battery_capacity_kWh"]),
        power_kW=float(params["BESS"]["power_kW"]),
        efficiency=float(params["BESS"]["efficiency"]),
        self_discharge=float(params["BESS"]["self_discharge"]),
        max_cycles=float(params["BESS"]["max_cycles"]),
        battery_eol_capacity=float(params["BESS"]["eol_capacity"]),
        DoD=float(params["BESS"]["DoD"]),
    )

    # Restdefizit nach BESS: EC zuerst, Rest Grid
    residual_deficit = (battery_out_request - battery_results["battery_out_series"])
    residual_pos = np.maximum(residual_deficit, 0.0)
    ec_import_from_pv += EC_SHARE * residual_pos
    grid_import += residual_pos - EC_SHARE * residual_pos

    # PV-Überschuss → Grid
    grid_export += battery_in_request - battery_results["battery_in_series"]

    # --- Nur EV-Ladung berücksichtigt, keine Entladung ---
    total_load_out = base_load + np.sum(ev_charged, axis=1)

    return {
        "heatpump_results_heating": hp_heat,
        "heatpump_results_cooling": hp_cool,
        "pv_results": pv_results,
        "ev_charged": ev_charged,
        "ev_discharged": np.zeros((n_steps, N_EV_total)),
        "ev_soc": ev_soc,
        "trip_loss": trip_loss,
        "bess_charged": battery_results["battery_in_series"],
        "bess_discharged": battery_results["battery_out_series"],
        "bess_soc": battery_results["soc_history"],
        "grid_import": grid_import,
        "grid_export": grid_export,
        "ec_import_from_pv": ec_import_from_pv,
        "ec_import_from_ev": ec_import_from_ev,
        "total_load": total_load_out,
        "pv_generation": pv_generation,
        "BESS_replacements": battery_results.get("BESS_replacements", 0),
        "timestamps": profiles.get("timestamps", pd.date_range(start="2023-01-01", periods=n_steps, freq="h")),
        "hotwater_load": hotwater,
    }




def simulate_energy_system_with_v2h(params: dict, profiles: dict, pv_size: float) -> dict:
    _require_precomputed(profiles, require_ev=True)

    load = np.asarray(profiles["load"], dtype=float)
    hp_heat = np.asarray(profiles["hp_elec_heat"], dtype=float)
    hp_cool = np.asarray(profiles["hp_elec_cool"], dtype=float)
    hotwater = np.asarray(profiles["hotwater_HH_kWh"], dtype=float)
    base_load = load + hp_heat + hp_cool + hotwater

    T_outdoor = np.asarray(profiles["T_outdoor"], dtype=float)
    availability_profile = np.asarray(profiles["availability_profile"], dtype=float)
    driving_profile = np.asarray(profiles["driving_profile"], dtype=float)
    min_soc_data = np.asarray(profiles["min_SOC"], dtype=float)

    n_steps = len(base_load)
    ev_params = params["EV"]

    # Nur bidirektionale EVs für V2H berücksichtigen
    N_EV_bidir = int(ev_params.get("N_EV_bidirectional"))
    N_EV_total = int(params.get("N_EV"))

    pv_results = simulate_pv_system(
        pv_size=float(pv_size),
        load_demand=base_load,
        pv_generation=np.asarray(profiles["pv_generation"], dtype=float),
        params=params
    )
    pv_generation = np.asarray(pv_results["pv_production"], dtype=float)

    ev_sim = simulate_v2h_battery(temperature=T_outdoor, params=ev_params)

    ev_soc = np.ones((n_steps, N_EV_total)) * ev_params["capacity_kWh"] * ev_params["initial_soc"]
    ev_charged = np.zeros((n_steps, N_EV_total))
    ev_discharged = np.zeros((n_steps, N_EV_bidir))
    ev_soc_series = []
    trip_loss = np.zeros((n_steps, N_EV_total))
    driving_energy = np.zeros((n_steps, N_EV_total))
    ev_availability = np.zeros((n_steps, N_EV_total), dtype=bool)
    ev_is_active_series = np.zeros((n_steps, N_EV_total), dtype=bool)

    battery_in_request = np.zeros(n_steps)
    battery_out_request = np.zeros(n_steps)
    grid_import = np.zeros(n_steps)
    grid_export = np.zeros(n_steps)
    ec_import_from_pv = np.zeros(n_steps)
    ec_import_from_ev = np.zeros(n_steps)

    EC_SHARE = float(params.get("EC_share", 1.0))  # 1.0 = alles aus EC; <1 erlaubt Grid-Anteil

    timestep_h = 1.0

    for t in range(n_steps):
        if t > 0:
            ev_soc[t] = ev_soc[t - 1]

        load_t = base_load[t]
        pv_t = pv_generation[t]
        pv_surplus = max(0.0, pv_t - load_t)
        load_deficit = max(0.0, load_t - pv_t)

        for i in range(N_EV_total):
            is_available = (np.random.random() <= availability_profile[t])
            ev_availability[t, i] = is_available
            ev_is_active_series[t, i] = is_available

            driving_energy[t, i] = driving_profile[t] * ev_params["capacity_kWh"]

            if not is_available:
                trip_loss[t, i] = driving_energy[t, i]
                min_soc = ev_params["capacity_kWh"] * min_soc_data[t]
                ev_soc[t, i] = max(min_soc, ev_soc[t, i] - driving_energy[t, i])
                continue

            # SOC-Grenzen
            min_soc = ev_params["capacity_kWh"] * min_soc_data[t]
            max_soc = ev_params["capacity_kWh"] * ev_params["max_soc"]

            charge_power_limit = ev_sim["charge_power_limit_capped"][t]
            max_charge_energy = min(charge_power_limit * timestep_h, max_soc - ev_soc[t, i])
            max_ev_discharge = ev_sim["discharge_power_limit"][t] * timestep_h

            # --- Schritt 1: Versorgung bis min_SOC (EC zuerst, Rest Grid) ---
            if ev_soc[t, i] < min_soc:
                needed = min_soc - ev_soc[t, i]
                drawn = needed / ev_params["charging_efficiency"]
                ev_soc[t, i] += needed
                ev_charged[t, i] += needed
                ec_take = EC_SHARE * drawn
                grid_take = drawn - ec_take
                ec_import_from_pv[t] += ec_take
                grid_import[t] += grid_take
                max_charge_energy = 0

            # --- Schritt 2: PV zuerst ---
            if ev_soc[t, i] < max_soc and max_charge_energy > 0:
                pv_used = min(pv_surplus, max_charge_energy)
                if pv_used > 0:
                    ev_soc[t, i] += pv_used * ev_params["charging_efficiency"]
                    ev_charged[t, i] += pv_used
                    pv_surplus -= pv_used
                    max_charge_energy -= pv_used

            # --- Schritt 3: BESS danach ---
            if ev_soc[t, i] < max_soc and max_charge_energy > 0 and pv_surplus <= 0:
                bess_to_ev = min(max_charge_energy, ev_params["capacity_kWh"] - ev_soc[t, i])
                if bess_to_ev > 0:
                    ev_soc[t, i] += bess_to_ev * ev_params["charging_efficiency"]
                    ev_charged[t, i] += bess_to_ev
                    battery_out_request[t] += bess_to_ev / ev_params["charging_efficiency"]
                    max_charge_energy -= bess_to_ev

            # --- Schritt 4: Entladen bei Defizit (nur für bidirektionale) ---
            if (i < N_EV_bidir) and load_deficit > 0 and ev_soc[t, i] > min_soc:
                ev_discharge_potential = min(load_deficit,
                                             max_ev_discharge,
                                             ev_soc[t, i] - min_soc)
                if ev_discharge_potential > 0:
                    ev_soc[t, i] -= ev_discharge_potential / ev_params["discharging_efficiency"]
                    ev_discharged[t, i] = ev_discharge_potential
                    load_deficit -= ev_discharge_potential
                    ec_import_from_ev[t] += ev_discharge_potential

            # --- Selbstentladung ---
            ev_soc[t, i] *= (1 - ev_params["self_discharge_EV"])

        ev_soc_series.append(ev_soc[t].copy())
        battery_in_request[t] = pv_surplus
        battery_out_request[t] += load_deficit  # wichtig: += statt =

    # --- BESS-Simulation ---
    battery_results = simulate_battery_flow(
        battery_in_request=battery_in_request,
        battery_out_request=battery_out_request,
        capacity_kWh=float(params["battery_capacity_kWh"]),
        power_kW=float(params["BESS"]["power_kW"]),
        efficiency=float(params["BESS"]["efficiency"]),
        self_discharge=float(params["BESS"]["self_discharge"]),
        max_cycles=float(params["BESS"]["max_cycles"]),
        battery_eol_capacity=float(params["BESS"]["eol_capacity"]),
        DoD=float(params["BESS"]["DoD"]),
    )

    # Restdefizit nach BESS: EC zuerst, Rest Grid
    residual_deficit = (battery_out_request - battery_results["battery_out_series"])
    residual_pos = np.maximum(residual_deficit, 0.0)
    grid_import += residual_pos - EC_SHARE * residual_pos
    ec_import_from_pv += EC_SHARE * residual_pos

    # PV-Überschuss → Grid (EVs speisen nicht ins Grid)
    grid_export += battery_in_request - battery_results["battery_in_series"]

    total_load_out = base_load + np.sum(ev_charged, axis=1) - np.sum(ev_discharged, axis=1)

    return {
        "heatpump_results_heating": hp_heat,
        "heatpump_results_cooling": hp_cool,
        "thermal_output_heating": np.zeros(n_steps, dtype=float),
        "thermal_output_cooling": np.zeros(n_steps, dtype=float),
        "pv_results": pv_results,
        "ev_charged": ev_charged,
        "ev_discharged": ev_discharged,
        "ev_soc": np.array(ev_soc_series),
        "bess_charged": battery_results["battery_in_series"],
        "bess_discharged": battery_results["battery_out_series"],
        "bess_soc": battery_results["soc_history"],
        "grid_import": grid_import,
        "grid_export": grid_export,
        "ec_import_from_pv": ec_import_from_pv,
        "ec_import_from_ev": ec_import_from_ev,
        "total_load": total_load_out,
        "pv_generation": pv_generation,
        "trip_loss": trip_loss,
        "driving_energy": driving_energy,
        "ev_availability": ev_availability,
        "availability_profile": availability_profile,
        "BESS_replacements": battery_results.get("BESS_replacements", 0),
        "timestamps": profiles.get("timestamps", pd.date_range(start="2023-01-01", periods=n_steps, freq="h")),
        "ev_active": ev_is_active_series,
        "hotwater_load": hotwater,
    }



'''
if __name__ == "__main__":
    from Data.data import get_parameters, load_profiles
    from Optimization.Vectorized_model.precompute import prepare_profiles
    import pandas as pd

    # Standort & Parameter laden
    location = "Vienna"
    base_params = get_parameters(location)
    base_params["location"] = location
    profiles_raw = load_profiles(location)
    profiles = prepare_profiles(base_params, profiles_raw, do_hp_electricity=True)

    # Beispielgrößen
    base_params["pv_size"] = 3000     # kWp
    base_params["battery_capacity_kWh"] = 500
    pv_size = base_params["pv_size"]

    # Simulation mit V2H
    res = simulate_energy_system(base_params, profiles, pv_size=pv_size)

    # DataFrame erzeugen (z. B. für 100 Stunden)
    n_show = 8760
    df = pd.DataFrame({
        "Timestamp": res["timestamps"][:n_show],
        "Grid_Import": res["grid_import"][:n_show],
        "Grid_Export": res["grid_export"][:n_show],
        "BESS_Charge": res["bess_charged"][:n_show],
        "BESS_Discharge": res["bess_discharged"][:n_show],
        "BESS_SOC": res["bess_soc"][:n_show],
        "Feedin_Tariff": [base_params.get("Cfeed", 0.0)] * n_show,
        "BESS_Degradation": [0.0] * n_show,
        "BESS_Replaced": [res["BESS_replacements"]] * n_show,
        "EV_Charge": res["ev_charged"][:n_show].sum(axis=1),
        "EV_Discharge": res["ev_discharged"][:n_show].sum(axis=1),
        "EV_SOC": res["ev_soc"][:n_show].mean(axis=1),
        "Trip_Loss": res["trip_loss"][:n_show].sum(axis=1),
        "Driving_Energy": res["driving_energy"][:n_show].sum(axis=1),
        "EV_Availability_Prob": profiles["availability_profile"][:n_show],
        "EV_Active": res["ev_active"][:n_show].any(axis=1),
        "PV_Production": res["pv_generation"][:n_show],
        "EV_Temperature": res["ev_temperature"][:n_show],
        "EV_CRate_Charge": res["ev_c_rate_charge"][:n_show],
        "EV_Charge_Limit_kW": res["ev_charge_limit_kW"][:n_show],
        "EV_Charge_Limit_kWh": res["ev_charge_limit_kWh"][:n_show],
        "EV_min_SOC": profiles["min_SOC"][:n_show],
    })

    # Ausgabe – Index ausblenden, Spaltenüberschriften behalten
    print("\n===== Stundenweise Lastflüsse (erste Zeilen) =====")
    print(df.head(8760).to_string(index=False, header=True, float_format=lambda x: f"{x:8.2f}"))
'''

if __name__ == "__main__":
    from Data.data import get_parameters, load_profiles
    from Optimization.Vectorized_model.precompute import prepare_profiles

    # Standort & Parameter laden
    location = "Vienna"
    base_params = get_parameters(location)
    base_params["location"] = location
    profiles_raw = load_profiles(location)
    profiles = prepare_profiles(base_params, profiles_raw, do_hp_electricity=True)

    # Beispielgrößen
    base_params["pv_size"] = 1   # kWp
    base_params["battery_capacity_kWh"] = 1
    pv_size = base_params["pv_size"]

    # --- Simulation wählen ---
    USE_V2H = False   # ⚡ hier umschalten zwischen NoV2H (False) und V2H (True)

    if USE_V2H:
        res = simulate_energy_system_with_v2h(base_params, profiles, pv_size=pv_size)
        tag = "V2H"
    else:
        res = simulate_energy_system(base_params, profiles, pv_size=pv_size)
        tag = "NoV2H"

    # --- Ergebnisse aggregieren ---
    lifetime = int(base_params.get("lifetime", 25))

    grid_import = np.sum(res["grid_import"])
    grid_export = np.sum(res["grid_export"])
    ec_import_pv = np.sum(res["ec_import_from_pv"])
    ec_import_ev = np.sum(res["ec_import_from_ev"])
    ev_charged = np.sum(res["ev_charged"])
    ev_discharged = np.sum(res["ev_discharged"])
    bess_throughput = (np.sum(res["bess_charged"]) + np.sum(res["bess_discharged"]))

    # --- PV-Generation berechnen ---
    total_pv_generation = np.sum(res["pv_generation"])

    # Degradation pro Jahr anwenden (falls noch nicht richtig berücksichtigt)
    # Beispiel: PV-Degradation = 0.5% pro Jahr
    degradation_rate = base_params['PV']['PVdegradation']
    years = np.arange(lifetime)
    degradation_factors = (1 - degradation_rate) ** years

    # Summiere die PV-Generation unter Berücksichtigung der Degradation
    pv_generation_lifetime = 0
    for i, factor in enumerate(degradation_factors):
        pv_generation_lifetime += total_pv_generation * factor

    # --- Anzahl aktiver EVs ---
    N_EV_total = int(base_params["EV"]["N_EV_total"])
    N_EV_bidir = int(base_params["EV"]["N_EV_bidirectional"])

    if USE_V2H:
        n_ev_charge = np.count_nonzero(np.sum(res["ev_charged"], axis=0) > 0)
        n_ev_discharge = np.count_nonzero(np.sum(res["ev_discharged"], axis=0) > 0)
    else:
        n_ev_charge = np.count_nonzero(np.sum(res["ev_charged"], axis=0) > 0)
        n_ev_discharge = 0  # NoV2H -> kein Entladen

    # --- Ausgabe ---
    print(f"\n=== Test-Run für feste Parameter ===")
    print(f"Mode: {tag} | PV: {base_params['pv_size']} kWp | BESS: {base_params['battery_capacity_kWh']} kWh")
    print(f"Grid Import:     {grid_import:,.0f} kWh/a")
    print(f"Grid Export:     {grid_export:,.0f} kWh/a")
    print(f"EC Import (PV):  {ec_import_pv:,.0f} kWh/a")
    print(f"EC Import (EV):  {ec_import_ev:,.0f} kWh/a")
    print(f"EV Charged:      {ev_charged:,.0f} kWh/a")
    print(f"EV Discharged:   {ev_discharged:,.0f} kWh/a")
    print(f"BESS Throughput: {bess_throughput:,.0f} kWh/a")
    print(f"Total PV Generation (Lebensdauer): {pv_generation_lifetime:,.0f} kWh (über {lifetime} Jahre)")  # PV-Menge unter Berücksichtigung der Degradation

    print("--- EV Stats ---")
    print(f"Total EVs:       {N_EV_total}")
    print(f"Bidirectional:   {N_EV_bidir}")
    print(f"EVs charged:     {n_ev_charge}")
    print(f"EVs discharged:  {n_ev_discharge}")
