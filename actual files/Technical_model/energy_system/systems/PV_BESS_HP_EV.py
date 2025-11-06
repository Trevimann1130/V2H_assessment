# Technical_model/energy_system/runners/systems/pv_bess_hp_ev.py

import numpy as np
import pandas as pd

from Technical_model.technologies.PV_model import simulate_pv_system
from Technical_model.technologies.battery_model import simulate_battery_flow

def simulate_energy_system(params: dict, profiles: dict, pv_size: float) -> dict:

    ec_cfg = params.get("EC", {})
    EC_SHARE_IMPORT = float(ec_cfg["share"])
    EC_SHARE_EXPORT = float(ec_cfg["export_share"])

    # --- Grundlast AC ---
    load = np.asarray(profiles["load"], dtype=float)
    hp_heat = np.asarray(profiles["hp_elec_heat"], dtype=float)
    hp_cool = np.asarray(profiles["hp_elec_cool"], dtype=float)
    hotwater = np.asarray(profiles["hotwater_HH_kWh"], dtype=float)
    base_load_ac = load + hp_heat + hp_cool + hotwater
    n_steps = len(base_load_ac)

    # --- EV Setup ---
    ev_params = params["EV"]
    N_EV_total     = int(ev_params["N_EV_total"])
    cap = float(ev_params["capacity_kWh"])
    eta_ch = float(ev_params["charging_efficiency"])
    self_dis_ev = float(ev_params["self_discharge_EV"])
    max_soc_frac = float(ev_params["max_soc"])
    init_soc_frac = float(ev_params["initial_soc"])
    max_chg_power = float(ev_params["max_charge_power"])

    availability_profile = np.asarray(profiles["availability_profile"], dtype=float)
    driving_profile = np.asarray(profiles["driving_profile"], dtype=float)   # relativ → *capacity_kWh
    min_soc_data = np.asarray(profiles["min_SOC"], dtype=float)              # relativ

    # deterministischer RNG
    rng = np.random.default_rng(int(params.get("rng_seed", 0)))

    # --- PV (AC-Seite) ---
    pv_results = simulate_pv_system(
        pv_size=float(pv_size),
        load_demand=base_load_ac,
        pv_generation=np.asarray(profiles["pv_generation"], dtype=float),
        params=params
    )
    pv_generation_ac = np.asarray(pv_results["pv_production"], dtype=float)

    # --- Requests (AC) + Statistiken ---
    battery_in_request_ac  = np.zeros(n_steps)   # PV→BESS (AC)
    battery_out_request_ac = np.zeros(n_steps)   # BESS→(EV + Last) (AC)

    # Aufteilung der BESS-Entlade-Anfrage (wichtig!)
    bess_out_req_ev_ac   = np.zeros(n_steps)  # EV möchte aus BESS laden
    bess_out_req_load_ac = np.zeros(n_steps)  # Basislast-Defizit an BESS

    grid_import_ac = np.zeros(n_steps)
    grid_export_ac = np.zeros(n_steps)

    # EC (aggregiertes internes Matching)
    ec_import_from_pv_ac = np.zeros(n_steps)  # EC deckt Defizit aus PV-EC-Pool
    ec_import_from_ev_ac = np.zeros(n_steps)  # hier 0 (kein V2H)
    ec_export_from_pv_ac = np.zeros(n_steps)  # unser PV-Überschuss an EC

    # EV: SOC (DC), DC-Ladung (Info), AC-Ladequellen
    ev_soc_dc = np.ones((n_steps, N_EV_total)) * cap * init_soc_frac
    ev_charged_dc = np.zeros((n_steps, N_EV_total))
    ev_charge_from_pv_ac   = np.zeros(n_steps)
    ev_charge_from_bess_ac = np.zeros(n_steps)  # wird NACH BESS-Sim gesetzt
    ev_charge_from_ec_ac   = np.zeros(n_steps)
    ev_charge_from_grid_ac = np.zeros(n_steps)

    trip_loss_dc = np.zeros((n_steps, N_EV_total))
    timestep_h = 1.0


    for t in range(n_steps):
        if t > 0:
            ev_soc_dc[t] = ev_soc_dc[t-1]

        load_t = base_load_ac[t]
        pv_t   = pv_generation_ac[t]

        # PV direkt in Grundlast (AC)
        pv_surplus_ac  = max(0.0, pv_t - load_t)
        load_deficit_ac= max(0.0, load_t - pv_t)

        # --- EVs (nur laden), AC bilanzieren, SOC in DC updaten ---
        for i in range(N_EV_total):
            is_available = (rng.random() <= availability_profile[t])

            # Fahrenergie (DC) abziehen, wenn nicht verfügbar
            drive_dc = driving_profile[t] * cap
            if not is_available:
                ev_soc_dc[t, i] = max(0.0, ev_soc_dc[t, i] - drive_dc)
                trip_loss_dc[t, i] = drive_dc
                continue

            min_soc_dc = cap * min_soc_data[t]
            max_soc_dc = cap * max_soc_frac

            # 1) Unter minSOC: Nachladen via EC/Grid (AC)
            if ev_soc_dc[t, i] < min_soc_dc:
                need_dc = min_soc_dc - ev_soc_dc[t, i]
                need_ac = min(need_dc / max(eta_ch, 1e-9),
                              max_chg_power * timestep_h)
                ec_take   = EC_SHARE_IMPORT * need_ac
                grid_take = need_ac - ec_take

                ev_soc_dc[t, i]  += need_ac * eta_ch
                ev_charged_dc[t, i] += need_ac * eta_ch
                ev_charge_from_ec_ac[t]   += ec_take
                ev_charge_from_grid_ac[t] += grid_take
                grid_import_ac[t]         += grid_take
                ec_import_from_ev_ac[t] += ec_take

            # 2) PV-Überschuss in EV (AC→DC)
            if (pv_surplus_ac > 1e-12) and (ev_soc_dc[t, i] < max_soc_dc):
                room_dc        = max(0.0, max_soc_dc - ev_soc_dc[t, i])
                max_ac_by_soc  = room_dc / max(eta_ch, 1e-9)
                max_ac_by_power= max_chg_power * timestep_h
                take_ac = min(pv_surplus_ac, max_ac_by_soc, max_ac_by_power)
                if take_ac > 1e-12:
                    ev_soc_dc[t, i]         += take_ac * eta_ch
                    ev_charged_dc[t, i]     += take_ac * eta_ch
                    ev_charge_from_pv_ac[t] += take_ac
                    pv_surplus_ac           -= take_ac

            # 3) BESS→EV (nur ANFRAGE; reale Zuteilung nach BESS-Sim!)
            if (pv_surplus_ac <= 1e-12) and (ev_soc_dc[t, i] < max_soc_dc):
                room_dc        = max(0.0, max_soc_dc - ev_soc_dc[t, i])
                max_ac_by_soc  = room_dc / max(eta_ch, 1e-9)
                max_ac_by_power= max_chg_power * timestep_h
                take_ac = min(max_ac_by_soc, max_ac_by_power)
                if take_ac > 1e-12:
                    ev_soc_dc[t, i]       += take_ac * eta_ch
                    ev_charged_dc[t, i]   += take_ac * eta_ch
                    battery_out_request_ac[t] += take_ac
                    bess_out_req_ev_ac[t]     += take_ac

            # Selbstentladung (DC)
            if self_dis_ev > 0:
                ev_soc_dc[t, i] *= (1.0 - self_dis_ev)

        # Nach EV-Laden: PV-Überschuss → BESS-Ladeanfrage
        battery_in_request_ac[t] = max(0.0, pv_surplus_ac)

        # Lastdefizit an BESS als **Anfrage** (separat tracken)
        battery_out_request_ac[t] += load_deficit_ac
        bess_out_req_load_ac[t]   += load_deficit_ac

    # --- BESS Simulation (AC) ---
    battery_results = simulate_battery_flow(
        battery_in_request=battery_in_request_ac,
        battery_out_request=battery_out_request_ac,
        capacity_kWh=float(params["battery_capacity_kWh"]),
        power_kW=float(params["BESS"]["power_kW"]),
        efficiency=float(params["BESS"]["efficiency"]),
        self_discharge=float(params["BESS"]["self_discharge"]),
        max_cycles=float(params["BESS"]["max_cycles"]),
        battery_eol_capacity=float(params["BESS"]["eol_capacity"]),
        DoD=float(params["BESS"]["DoD"]),
    )
    bess_in_ac  = np.asarray(battery_results["battery_in_series"], dtype=float)
    bess_out_ac = np.asarray(battery_results["battery_out_series"], dtype=float)

    # --- Reale BESS-Entladung proportional auf EV vs. Last verteilen ---
    total_req = bess_out_req_ev_ac + bess_out_req_load_ac
    with np.errstate(divide='ignore', invalid='ignore'):
        share_ev = np.where(total_req > 1e-12, bess_out_req_ev_ac / total_req, 0.0)
    bess_out_to_ev_ac   = share_ev * bess_out_ac
    bess_out_to_load_ac = bess_out_ac - bess_out_to_ev_ac

    # Jetzt erst die EV-Quelle „aus BESS“ mit der **realen** Zuteilung buchen
    ev_charge_from_bess_ac = bess_out_to_ev_ac.copy()

    residual_deficit_ac = bess_out_req_load_ac - bess_out_to_load_ac
    residual_pos_ac     = np.maximum(residual_deficit_ac, 0.0)

    post_bess_surplus_ac  = battery_in_request_ac - bess_in_ac
    post_bess_surplus_pos = np.maximum(post_bess_surplus_ac, 0.0)

    ec_internal_match = np.minimum(EC_SHARE_IMPORT * residual_pos_ac,
                                   EC_SHARE_EXPORT * post_bess_surplus_pos)

    ec_import_from_pv_ac += ec_internal_match
    ec_export_from_pv_ac += ec_internal_match

    grid_import_ac += residual_pos_ac - ec_internal_match
    grid_export_ac += post_bess_surplus_pos - ec_internal_match

    ev_charge_ac = ev_charge_from_pv_ac + ev_charge_from_bess_ac + ev_charge_from_ec_ac + ev_charge_from_grid_ac

    results = {
        "heatpump_results_heating": hp_heat,
        "heatpump_results_cooling": hp_cool,
        "pv_results": pv_results,
        "ev_charged": ev_charged_dc,
        "ev_discharge_ac": np.zeros(n_steps),
        "ev_charged_dc": ev_charged_dc,
        "ev_soc": ev_soc_dc,
        "trip_loss": trip_loss_dc,
        "bess_charged": bess_in_ac,
        "bess_discharged": bess_out_ac,
        "bess_soc": np.asarray(battery_results["soc_history"], dtype=float),
        "BESS_replacements": battery_results.get("BESS_replacements", 0),
        "grid_import": grid_import_ac,
        "grid_export": grid_export_ac,
        "ec_import_from_pv": ec_import_from_pv_ac,
        "ec_import_from_ev": ec_import_from_ev_ac,
        "ec_export_from_pv": ec_export_from_pv_ac,
        "ev_charge_ac": ev_charge_ac,
        "ev_charge_from_pv_ac": ev_charge_from_pv_ac,
        "ev_charge_from_bess_ac": ev_charge_from_bess_ac,
        "ev_charge_from_ec_ac": ev_charge_from_ec_ac,
        "ev_charge_from_grid_ac": ev_charge_from_grid_ac,
        "total_load": base_load_ac + ev_charge_ac,
        "pv_generation": pv_generation_ac,
        "timestamps": profiles.get("timestamps", pd.date_range(start="2023-01-01", periods=n_steps, freq="h")),
        "hotwater_load": hotwater,
        "bess_to_ev_ac": bess_out_to_ev_ac,
        "bess_to_load_ac": bess_out_to_load_ac,
    }
    return results
