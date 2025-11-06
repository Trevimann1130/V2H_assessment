# Technical_model/energy_system/systems/PV_BESS_HP_V2H.py
import numpy as np
import pandas as pd

from Technical_model.technologies.PV_model import simulate_pv_system
from Technical_model.technologies.battery_model import simulate_battery_flow
from Technical_model.technologies.V2H_model import simulate_v2h_battery

def simulate_energy_system_with_v2h(params: dict, profiles: dict, pv_size: float) -> dict:
    # EC strikt aus params lesen
    ec_cfg = params["EC"]
    EC_SHARE_IMPORT  = float(ec_cfg["share"])
    EC_SHARE_EXPORT  = float(ec_cfg["export_share"])

    # Grundlast AC
    load     = np.asarray(profiles["load"], dtype=float)
    hp_heat  = np.asarray(profiles["hp_elec_heat"], dtype=float)
    hp_cool  = np.asarray(profiles["hp_elec_cool"], dtype=float)
    hotwater = np.asarray(profiles["hotwater_HH_kWh"], dtype=float)
    base_load_ac = load + hp_heat + hp_cool + hotwater

    T_outdoor            = np.asarray(profiles["T_outdoor"], dtype=float)
    availability_profile = np.asarray(profiles["availability_profile"], dtype=float)
    driving_profile      = np.asarray(profiles["driving_profile"], dtype=float)
    min_soc_data         = np.asarray(profiles["min_SOC"], dtype=float)

    n_steps   = len(base_load_ac)
    ev_params = params["EV"]                       # ← nur EINMAL

    N_EV_bidir = int(ev_params["N_EV_bidirectional"])
    N_EV_total = int(ev_params["N_EV_total"])

    cap           = float(ev_params["capacity_kWh"])
    eta_ch        = float(ev_params["charging_efficiency"])
    eta_dis       = float(ev_params["discharging_efficiency"])
    self_dis_ev   = float(ev_params["self_discharge_EV"])
    max_soc_frac  = float(ev_params["max_soc"])
    init_soc_frac = float(ev_params["initial_soc"])
    max_chg_power = float(ev_params["max_charge_power"])

    rng = np.random.default_rng(int(params.get("rng_seed", 0)))

    # PV (AC)
    pv_results = simulate_pv_system(
        pv_size=float(pv_size),
        load_demand=base_load_ac,
        pv_generation=np.asarray(profiles["pv_generation"], dtype=float),
        params=params
    )
    pv_generation_ac = np.asarray(pv_results["pv_production"], dtype=float)

    # EV-Limits (Temp-abhängig)
    ev_sim = simulate_v2h_battery(temperature=T_outdoor, params=ev_params)

    # Grenzserien
    if N_EV_bidir == 0:
        charge_limit_series    = np.full(n_steps, max_chg_power)  # kW
        discharge_limit_series = np.zeros(n_steps)                # keine Entladung
    else:
        charge_limit_series    = np.asarray(ev_sim["charge_power_limit_capped"], dtype=float)
        discharge_limit_series = np.asarray(ev_sim["discharge_power_limit"], dtype=float)

    # EV-Zustände/Container
    ev_soc_dc          = np.ones((n_steps, N_EV_total)) * cap * init_soc_frac
    ev_charged_dc      = np.zeros((n_steps, N_EV_total))
    ev_discharged_ac   = np.zeros((n_steps, N_EV_bidir))
    trip_loss_dc       = np.zeros((n_steps, N_EV_total))
    driving_energy_dc  = np.zeros((n_steps, N_EV_total))
    ev_availability    = np.zeros((n_steps, N_EV_total), dtype=bool)
    ev_is_active_series= np.zeros((n_steps, N_EV_total), dtype=bool)
    ev_bess_req_ac_ev = np.zeros((n_steps, N_EV_total))

    # BESS/Netz/EC/EV-Quellen
    battery_in_request_ac   = np.zeros(n_steps)
    battery_out_request_ac  = np.zeros(n_steps)
    bess_out_req_ev_ac      = np.zeros(n_steps)
    bess_out_req_load_ac    = np.zeros(n_steps)

    grid_import_ac          = np.zeros(n_steps)
    grid_export_ac          = np.zeros(n_steps)
    ec_import_from_pv_ac    = np.zeros(n_steps)
    ec_import_from_ev_ac    = np.zeros(n_steps)
    ec_export_from_pv_ac    = np.zeros(n_steps)

    ev_charge_from_pv_ac    = np.zeros(n_steps)
    ev_charge_from_bess_ac  = np.zeros(n_steps)
    ev_charge_from_ec_ac    = np.zeros(n_steps)
    ev_charge_from_grid_ac  = np.zeros(n_steps)

    timestep_h = 1.0

    for t in range(n_steps):
        if t > 0:
            ev_soc_dc[t] = ev_soc_dc[t - 1]

        load_t         = base_load_ac[t]
        pv_t           = pv_generation_ac[t]
        pv_surplus_ac  = max(0.0, pv_t - load_t)
        load_deficit_ac= max(0.0, load_t - pv_t)

        for i in range(N_EV_total):
            is_available              = (rng.random() <= availability_profile[t])
            ev_availability[t, i]     = is_available
            ev_is_active_series[t, i] = is_available

            drive_dc = driving_profile[t] * cap
            driving_energy_dc[t, i] = drive_dc

            if not is_available:
                ev_soc_dc[t, i] = max(0.0, ev_soc_dc[t, i] - drive_dc)
                trip_loss_dc[t, i] = drive_dc
                continue

            min_soc_dc = cap * min_soc_data[t]
            max_soc_dc = cap * max_soc_frac

            # Lade-/Entlade-Limits (AC Energiemenge pro h)
            charge_limit_ac    = min(charge_limit_series[t] * timestep_h,
                                     max(0.0, (max_soc_dc - ev_soc_dc[t, i]) / max(eta_ch, 1e-9)))
            discharge_limit_ac = discharge_limit_series[t] * timestep_h

            # (1) minSOC via EC/Grid
            if ev_soc_dc[t, i] < min_soc_dc and charge_limit_ac > 1e-12:
                need_dc = min_soc_dc - ev_soc_dc[t, i]
                need_ac = min(need_dc / max(eta_ch, 1e-9), charge_limit_ac)
                ec_take, grid_take = EC_SHARE_IMPORT * need_ac, need_ac - EC_SHARE_IMPORT * need_ac

                ev_soc_dc[t, i]        += need_ac * eta_ch
                ev_charged_dc[t, i]    += need_ac * eta_ch
                ev_charge_from_ec_ac[t]+= ec_take
                ev_charge_from_grid_ac[t]+= grid_take
                grid_import_ac[t]      += grid_take
                charge_limit_ac        -= need_ac

            # (2) PV→EV
            if charge_limit_ac > 1e-12 and pv_surplus_ac > 1e-12:
                take_ac = min(charge_limit_ac, pv_surplus_ac)
                ev_soc_dc[t, i]          += take_ac * eta_ch
                ev_charged_dc[t, i]      += take_ac * eta_ch
                ev_charge_from_pv_ac[t]  += take_ac
                pv_surplus_ac            -= take_ac
                charge_limit_ac          -= take_ac

            # (3) BESS→EV (Anfrage)
            if charge_limit_ac > 1e-12 and pv_surplus_ac <= 1e-12:
                take_ac = charge_limit_ac
                ev_soc_dc[t, i]       += take_ac * eta_ch
                ev_charged_dc[t, i]   += take_ac * eta_ch
                battery_out_request_ac[t] += take_ac
                bess_out_req_ev_ac[t]     += take_ac

            # (4) EV→Haus (nur bidirektionale)
            if (i < N_EV_bidir) and load_deficit_ac > 1e-12 and (ev_soc_dc[t, i] > min_soc_dc):
                room_dc        = max(0.0, ev_soc_dc[t, i] - min_soc_dc)
                max_ac_by_soc  = room_dc * max(eta_dis, 0.0)  # AC verfügbar
                give_ac = min(load_deficit_ac, discharge_limit_ac, max_ac_by_soc)
                if give_ac > 1e-12:
                    ev_soc_dc[t, i]    -= give_ac / max(eta_dis, 1e-9)
                    ev_discharged_ac[t, i] = give_ac
                    load_deficit_ac     -= give_ac

            if self_dis_ev > 0:
                ev_soc_dc[t, i] *= (1.0 - self_dis_ev)

        # PV/BESS-Anfragen nach EV
        battery_in_request_ac[t]  = max(0.0, pv_surplus_ac)
        battery_out_request_ac[t] += load_deficit_ac
        bess_out_req_load_ac[t]   += load_deficit_ac

    # BESS Simulation
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

    # Reale BESS-Entladung proportional zuteilen
    total_req = bess_out_req_ev_ac + bess_out_req_load_ac
    with np.errstate(divide='ignore', invalid='ignore'):
        share_ev = np.where(total_req > 1e-12, bess_out_req_ev_ac / total_req, 0.0)
    bess_out_to_ev_ac   = share_ev * bess_out_ac
    bess_out_to_load_ac = bess_out_ac - bess_out_to_ev_ac
    ev_charge_from_bess_ac = bess_out_to_ev_ac  # ← direkt nutzen, dann keine Lint-Warnung

    # EC-Logik (gewichtet), dann auf Netz/Export
    residual_pos_ac      = np.maximum(bess_out_req_load_ac - bess_out_to_load_ac, 0.0)
    post_bess_surplus_pos= np.maximum(battery_in_request_ac - bess_in_ac, 0.0)

    ev_ec_need_unw = ev_charge_from_ec_ac.copy()
    load_need_w    = EC_SHARE_IMPORT * residual_pos_ac
    ev_need_w      = EC_SHARE_IMPORT * ev_ec_need_unw
    total_need_w   = load_need_w + ev_need_w

    ec_possible_w  = EC_SHARE_EXPORT * post_bess_surplus_pos
    ec_match_w     = np.minimum(ec_possible_w, total_need_w)

    with np.errstate(divide="ignore", invalid="ignore"):
        frac_ev    = np.where(total_need_w > 1e-12, ev_need_w / total_need_w, 0.0)
    ec_to_ev_w     = ec_match_w * frac_ev
    ec_to_load_w   = ec_match_w - ec_to_ev_w

    ec_export_from_pv_ac += ec_match_w
    ec_import_from_pv_ac += ec_to_load_w

    small = 1e-12 if EC_SHARE_IMPORT <= 0 else EC_SHARE_IMPORT
    ev_ec_matched_unw = ec_to_ev_w / small
    delta_ev_ec = np.maximum(0.0, ev_charge_from_ec_ac - ev_ec_matched_unw)
    if np.any(delta_ev_ec > 1e-12):
        ev_charge_from_ec_ac  -= delta_ev_ec
        ev_charge_from_grid_ac+= delta_ev_ec
        grid_import_ac        += delta_ev_ec
        ec_import_from_ev_ac  -= EC_SHARE_IMPORT * delta_ev_ec

    grid_import_ac += residual_pos_ac - (ec_to_load_w / max(EC_SHARE_IMPORT, 1e-12))
    grid_export_ac += post_bess_surplus_pos - (ec_match_w / max(EC_SHARE_EXPORT, 1e-12))

    ev_charge_ac = ev_charge_from_pv_ac + ev_charge_from_bess_ac + ev_charge_from_ec_ac + ev_charge_from_grid_ac
    total_load_out_ac = base_load_ac + ev_charge_ac - np.sum(ev_discharged_ac, axis=1)

    return {
        "heatpump_results_heating": hp_heat,
        "heatpump_results_cooling": hp_cool,
        "thermal_output_heating": np.zeros(n_steps, dtype=float),
        "thermal_output_cooling": np.zeros(n_steps, dtype=float),
        "pv_results": pv_results,
        # EV
        "ev_charged": ev_charged_dc,
        "ev_discharged": ev_discharged_ac,   # ← KEY-NAME korrigiert
        "ev_soc": ev_soc_dc,
        "trip_loss": trip_loss_dc,
        "driving_energy": driving_energy_dc,
        "ev_availability": ev_availability,
        "ev_active": ev_is_active_series,
        # EV-Quellen
        "ev_charge_ac": ev_charge_ac,
        "ev_charge_from_pv_ac": ev_charge_from_pv_ac,
        "ev_charge_from_bess_ac": ev_charge_from_bess_ac,
        "ev_charge_from_ec_ac": ev_charge_from_ec_ac,
        "ev_charge_from_grid_ac": ev_charge_from_grid_ac,
        # BESS
        "bess_charged": bess_in_ac,
        "bess_discharged": bess_out_ac,
        "bess_soc": np.asarray(battery_results["soc_history"], dtype=float),
        "BESS_replacements": battery_results.get("BESS_replacements", 0),
        "bess_to_ev_ac": bess_out_to_ev_ac,
        "bess_to_load_ac": bess_out_to_load_ac,
        # Netz/EC/Basis
        "grid_import": grid_import_ac,
        "grid_export": grid_export_ac,
        "ec_import_from_pv": ec_import_from_pv_ac,
        "ec_import_from_ev": ec_import_from_ev_ac,
        "ec_export_from_pv": ec_export_from_pv_ac,
        "total_load": total_load_out_ac,
        "pv_generation": pv_generation_ac,
        "timestamps": profiles.get("timestamps", pd.date_range(start="2023-01-01", periods=n_steps, freq="h")),
        "hotwater_load": hotwater,
    }
