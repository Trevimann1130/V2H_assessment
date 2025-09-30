import numpy as np
import pandas as pd

from Technical_model.V2H_model import simulate_v2h_battery
from Technical_model.PV_model import simulate_pv_system
from Technical_model.heatpump_model import (
    simulate_heatpump_heating_system,
    simulate_heatpump_cooling_system,
)
from Technical_model.battery_model import simulate_battery_flow

def simulate_energy_system(params, profiles, pv_size):

    T_outdoor = profiles['T_outdoor']
    usage_df = profiles['usage_profile']

    hotwater_W_m2 = usage_df['Warmwasserbedarf_W_m2'].to_numpy()
    building_area = params['building']['A_floor']
    hotwater_HH_kWh = (hotwater_W_m2 * building_area) / 1000

    # Wärmepumpen
    hpH = simulate_heatpump_heating_system(params=params, profiles=profiles)
    hpC = simulate_heatpump_cooling_system(params=params, profiles=profiles)
    elec_hp_heat = np.asarray(hpH['electric_consumption_series'], dtype=float)
    elec_hp_cool = np.asarray(hpC['electric_consumption_series'], dtype=float)

    # Basislast (ohne EV)
    base_load = profiles['load'] + elec_hp_heat + elec_hp_cool + hotwater_HH_kWh
    n_steps = len(base_load)

    # EV-Profile
    ev_load_profile = profiles["ev_profiles"].sum(axis=1)   # Ladebedarf pro Stunde
    ev_params = params["EV"]
    ev_soc = np.ones(n_steps) * ev_params["capacity_kWh"] * ev_params.get("initial_soc", 0.5)
    ev_charged = np.zeros(n_steps)

    # PV-Simulation
    pv_results = simulate_pv_system(
        pv_size=pv_size,
        load_demand=base_load,
        pv_generation=profiles['pv_generation'],
        params=params
    )
    pv_generation = np.asarray(pv_results['pv_production'], dtype=float)

    battery_in_request = np.zeros(n_steps)
    battery_out_request = np.zeros(n_steps)
    grid_import = np.zeros(n_steps)
    grid_export = np.zeros(n_steps)

    for t in range(n_steps):
        load_t = base_load[t]
        pv_t = pv_generation[t]

        # --- Schritt 1: Basislast durch PV decken ---
        direct_pv = min(pv_t, load_t)
        pv_surplus = max(0, pv_t - load_t)
        load_deficit = max(0, load_t - pv_t)

        # --- Schritt 2: EV laden mit PV-Überschuss ---
        if ev_load_profile[t] > 0 and pv_surplus > 0:
            max_ev_charge = min(ev_load_profile[t], pv_surplus, ev_params["capacity_kWh"] - ev_soc[t])
            ev_soc[t] += max_ev_charge * ev_params["charging_efficiency"]
            ev_charged[t] = max_ev_charge                  # Quelle = PV-Überschuss
            pv_surplus -= max_ev_charge

        # --- Schritt 3: EV laden aus BESS (falls kein PV, Restkapazität da) ---
        if ev_load_profile[t] > 0 and pv_surplus <= 0 and load_deficit <= 0:
            bess_source = min(ev_load_profile[t], ev_params["capacity_kWh"] - ev_soc[t])
            if bess_source > 0:
                ev_soc[t]     += bess_source * ev_params["charging_efficiency"]
                ev_charged[t] += bess_source               # Quelle = BESS
                battery_out_request[t] += bess_source      # BESS-Entnahme

        # --- Schritt 4: EV laden aus Netz (nur wenn unter minSOC) ---
        min_soc_frac = profiles["min_SOC"]
        min_soc_frac = (min_soc_frac[t] if np.ndim(min_soc_frac) > 0 else float(min_soc_frac))
        min_soc = ev_params["capacity_kWh"] * min_soc_frac

        if ev_soc[t] < min_soc:
            needed_ev = min_soc - ev_soc[t]                # Ziel: im EV ankommen
            grid_source = needed_ev / ev_params["charging_efficiency"]
            ev_soc[t]     += needed_ev
            ev_charged[t] += grid_source                   # Quelle = Netz
            grid_import[t] += grid_source

        # --- Rest für BESS/Netz ---
        battery_in_request[t] = pv_surplus
        battery_out_request[t] += load_deficit

    # BESS-Simulation
    battery_results = simulate_battery_flow(
        battery_in_request=battery_in_request,
        battery_out_request=battery_out_request,
        capacity_kWh=params['battery_capacity_kWh'],
        power_kW=params['BESS']['power_kW'],
        efficiency=params['BESS']['efficiency'],
        self_discharge=params['BESS']['self_discharge'],
        max_cycles=params['BESS']['max_cycles'],
        battery_eol_capacity=params['BESS']['eol_capacity'],
        DoD=params['BESS']['DoD']
    )

    grid_import += battery_out_request - battery_results['battery_out_series']
    grid_export += battery_in_request - battery_results['battery_in_series']

    total_load = base_load + ev_charged

    return {
        'heatpump_results_heating': elec_hp_heat,
        'heatpump_results_cooling': elec_hp_cool,
        'pv_results': pv_results,
        'ev_charged': ev_charged,
        'ev_discharged': np.zeros(n_steps),
        'ev_soc': ev_soc,
        'bess_charged': battery_results['battery_in_series'],
        'bess_discharged': battery_results['battery_out_series'],
        'bess_soc': battery_results['soc_history'],
        'grid_import': grid_import,
        'grid_export': grid_export,
        'total_load': total_load,
        'pv_generation': pv_generation,
        'BESS_replacements': battery_results['BESS_replacements'],
        'timestamps': profiles.get('timestamps', pd.date_range(start="2023-01-01", periods=n_steps, freq="h")),
        'hotwater_load': hotwater_HH_kWh,
    }



def simulate_energy_system_with_v2h(params, profiles, pv_size):

    availability_profile = profiles["availability_profile"]
    driving_profile = profiles["driving_profile"]
    min_soc_data = profiles["min_SOC"]

    T_outdoor = profiles["T_outdoor"]
    usage_df = profiles['usage_profile']
    load = profiles["load"]
    n_steps = len(load)

    hotwater_W_m2 = usage_df['Warmwasserbedarf_W_m2'].to_numpy()
    building_area = params['building']['A_floor']
    hotwater_HH_kWh = (hotwater_W_m2 * building_area) / 1000

    # WP-Simulation
    heatpump_results = simulate_heatpump_heating_system(params=params, profiles=profiles)
    heatpump_cooling_results = simulate_heatpump_cooling_system(params=params, profiles=profiles)
    elec_hp_heat = heatpump_results["electric_consumption_series"]
    elec_hp_cool = heatpump_cooling_results["electric_consumption_series"]

    base_load = profiles['load'] + elec_hp_heat + elec_hp_cool + hotwater_HH_kWh

    # PV-Simulation
    pv_results = simulate_pv_system(
        pv_size=pv_size,
        load_demand=base_load,
        pv_generation=profiles['pv_generation'],
        params=params
    )
    pv_generation = pv_results['pv_production']

    # V2H Lade-/Entladeraten
    ev_params = params["EV"]
    ev_sim = simulate_v2h_battery(
        temperature=np.asarray(T_outdoor, dtype=float),
        params=ev_params,
    )

    N_EV = params.get("N_EV")

    # Initialisierung
    ev_soc = np.ones((n_steps, N_EV)) * ev_params["capacity_kWh"] * ev_params.get("initial_soc", 0.5)
    ev_charged = np.zeros((n_steps, N_EV))      # Quelle (PV, BESS, Netz)
    ev_discharged = np.zeros((n_steps, N_EV))   # Energie ins Hausnetz
    ev_soc_series = []
    trip_loss = np.zeros((n_steps, N_EV))
    driving_energy = np.zeros((n_steps, N_EV))
    ev_availability = np.zeros((n_steps, N_EV), dtype=bool)
    ev_is_active_series = np.zeros((n_steps, N_EV), dtype=bool)

    battery_in_request = np.zeros(n_steps)
    battery_out_request = np.zeros(n_steps)
    grid_import = np.zeros(n_steps)
    grid_export = np.zeros(n_steps)

    for t in range(n_steps):
        if t > 0:
            ev_soc[t] = ev_soc[t - 1]  # Vorherigen SOC übernehmen

        load_t = base_load[t]
        pv_t = pv_generation[t]
        pv_surplus = max(0, pv_t - load_t)
        load_deficit = max(0, load_t - pv_t)

        for i in range(N_EV):
            is_available = np.random.random() <= availability_profile[t]
            ev_availability[t, i] = is_available
            ev_is_active_series[t, i] = is_available

            driving_energy[t, i] = driving_profile[t] * ev_params["capacity_kWh"]

            if not is_available:
                trip_loss[t, i] = driving_energy[t, i]
                soc_before = ev_soc[t, i]
                min_soc = ev_params["capacity_kWh"] * min_soc_data[t]
                ev_soc[t, i] = max(min_soc, soc_before - driving_energy[t, i])
                continue

            # Wenn verfügbar:
            min_soc = ev_params["capacity_kWh"] * min_soc_data[t]
            max_soc = ev_params["capacity_kWh"] * ev_params["max_soc"]
            timestep_h = 1
            charge_power_limit = ev_sim["charge_power_limit_capped"][t]
            max_charge_energy = min(charge_power_limit * timestep_h, max_soc - ev_soc[t, i])
            max_ev_discharge = ev_sim["discharge_power_limit"][t]

            # --- Entladung zur Lastdeckung ---
            if ev_soc[t, i] > min_soc:
                ev_discharge_potential = min(load_deficit, max_ev_discharge, ev_soc[t, i] - min_soc)
                ev_soc[t, i] -= ev_discharge_potential / ev_params["discharging_efficiency"]
                ev_discharged[t, i] = ev_discharge_potential
                load_deficit -= ev_discharge_potential

            # --- Laden (PV → BESS → Netz) ---
            if ev_soc[t, i] < max_soc and max_charge_energy > 0:
                # PV zuerst
                pv_used = min(pv_surplus, max_charge_energy)
                ev_soc[t, i] += pv_used * ev_params["charging_efficiency"]
                ev_charged[t, i] += pv_used  # Quelle = PV
                pv_surplus -= pv_used
                max_charge_energy -= pv_used

                # dann BESS
                if pv_surplus <= 0 and ev_soc[t, i] < max_soc and max_charge_energy > 0:
                    bess_source = min(max_charge_energy, ev_params["capacity_kWh"] - ev_soc[t, i])
                    if bess_source > 0:
                        ev_soc[t, i] += bess_source * ev_params["charging_efficiency"]
                        ev_charged[t, i] += bess_source  # Quelle = BESS
                        battery_out_request[t] += bess_source
                        max_charge_energy -= bess_source

                # zuletzt Netz (nur wenn unter minSOC)
                if ev_soc[t, i] < min_soc and max_charge_energy > 0:
                    needed_ev = min_soc - ev_soc[t, i]  # im Akku ankommen
                    grid_source = min(max_charge_energy, needed_ev / ev_params["charging_efficiency"])
                    if grid_source > 0:
                        ev_soc[t, i] += grid_source * ev_params["charging_efficiency"]
                        ev_charged[t, i] += needed_ev  # Quelle = Netz
                        grid_import[t] += grid_source
                        max_charge_energy -= grid_source

            # --- Selbstentladung ---
            ev_soc[t, i] *= (1 - ev_params.get("self_discharge_EV", 0.0))

        ev_soc_series.append(ev_soc[t].copy())
        battery_in_request[t] = pv_surplus
        battery_out_request[t] = load_deficit

    battery_results = simulate_battery_flow(
        battery_in_request=battery_in_request,
        battery_out_request=battery_out_request,
        capacity_kWh=params["battery_capacity_kWh"],
        power_kW=params["BESS"]["power_kW"],
        efficiency=params["BESS"]["efficiency"],
        self_discharge=params["BESS"]["self_discharge"],
        max_cycles=params["BESS"]["max_cycles"],
        battery_eol_capacity=params["BESS"]["eol_capacity"],
        DoD=params["BESS"]["DoD"],
    )

    grid_import += battery_out_request - battery_results["battery_out_series"]
    grid_export += battery_in_request - battery_results["battery_in_series"]

    total_load = base_load + np.sum(ev_charged, axis=1) - np.sum(ev_discharged, axis=1)
    return {
        "heatpump_results_heating": elec_hp_heat,
        "heatpump_results_cooling": elec_hp_cool,
        "pv_results": pv_results,
        "ev_charged": ev_charged,
        "ev_discharged": ev_discharged,
        "ev_soc": np.array(ev_soc_series),
        "bess_charged": battery_results["battery_in_series"],
        "bess_discharged": battery_results["battery_out_series"],
        "bess_soc": battery_results["soc_history"],
        "grid_import": grid_import,
        "grid_export": grid_export,
        "total_load": total_load,
        "pv_generation": pv_generation,
        "trip_loss": trip_loss,
        "driving_energy": driving_energy,
        "ev_availability": ev_availability,
        "availability_profile": availability_profile,
        "BESS_replacements": battery_results["BESS_replacements"],
        "timestamps": profiles.get("timestamps", pd.date_range(start="2023-01-01", periods=n_steps, freq="h")),
        "ev_active": ev_is_active_series,
        'hotwater_load': hotwater_HH_kWh,
    }
