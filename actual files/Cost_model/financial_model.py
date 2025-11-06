import math
import pandas as pd
from Data import data as data

# Alte Version: Monats-basierter NPC (mit Referenzmarktpreisen pro Monat)
# Derzeit deaktiviert, nur für spätere Referenz.
'''def calculate_npc(params, energy_flows):
    # (UNVERÄNDERT) – deine bestehende Monats-basierte NPC-Funktion
    capex_pv = params['CPV'] * params['pv_size']
    capex_batt = params['CBESS'] * params['battery_capacity_kWh']
    capex_EV = params.get('CEV', 0)
    total_capex = capex_pv + capex_batt + capex_EV

    opex_pv = capex_pv * params['PV']['maintenance_rate_PV']
    opex_batt = capex_batt * params['BESS']['maintenance_rate_BESS']
    opex_EV = capex_EV * params['EV']['maintenance_rate_EV']
    opex_annual = opex_pv + opex_batt + opex_EV

    r_grid = params['electricity_price_growth']
    r_feed = params['feedin_growth_rate']

    timestamps = pd.to_datetime(energy_flows['timestamps'])
    df = pd.DataFrame({
        'import': energy_flows['grid_import'],
        'export': energy_flows['grid_export'],
        'time': timestamps
    })
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    start_year = df['year'].min()
    df = df.drop(columns=['time'])
    grouped = df.groupby(['year', 'month']).sum().reset_index()

    base_prices = params['reference_market_price_pv'][2023]  # Liste 12 Monate
    lifetime = params['lifetime']
    wacc = params['WACC']

    capex_batt_replacement = 0.0
    if 'battery_cycles' in energy_flows and 'BESS_replacements' in energy_flows:
        replacements = energy_flows['BESS_replacements']
        capex_batt_replacement = replacements * (0.6 * params['CBESS'] * params['battery_capacity_kWh'])

    npc = total_capex + capex_batt_replacement

    for year_offset in range(lifetime):
        year = start_year + year_offset
        year_factor_grid = (1 + r_grid) ** year_offset
        year_factor_feed = (1 + r_feed) ** year_offset

        annual_cost = 0.0
        for month in range(1, 13):
            base_price_feed = base_prices[month - 1]
            export = grouped.loc[(grouped['year'] == year) & (grouped['month'] == month), 'export'].sum()
            import_ = grouped.loc[(grouped['year'] == year) & (grouped['month'] == month), 'import'].sum()
            revenue = export * base_price_feed * year_factor_feed
            cost = import_ * params['Cbuy'] * year_factor_grid
            annual_cost += (cost - revenue)

        discounted_opex = opex_annual / (1 + wacc) ** (year_offset + 1)
        discounted_energy = annual_cost / (1 + wacc) ** (year_offset + 1)
        npc += discounted_opex + discounted_energy

    return npc

'''
# financial_model.py

import numpy as np


import numpy as np

def calculate_npc_yearly(params,
                         e_import_grid_year,
                         e_import_ec_pv_year=0.0,
                         e_import_ec_ev_year=0.0,
                         e_export_grid_year=0.0,
                         e_export_pv_ec_year=0.0,
                         e_export_ev_ec_year=0.0):
    """
    NPC-Berechnung für Energiegemeinschaft mit PV, BESS und EVs (flussgenau).

    Inputs:
      e_import_grid_year    : kWh/a Bezug aus öffentlichem Netz
      e_import_ec_pv_year   : kWh/a Bezug aus Energiegemeinschaft (PV-Strom)
      e_import_ec_ev_year   : kWh/a Bezug aus Energiegemeinschaft (EV-Strom, V2H)
      e_export_grid_year    : kWh/a Export ins öffentliche Netz (PV-Überschuss)
      e_export_pv_ec_year   : kWh/a PV-Export in Energiegemeinschaft
      e_export_ev_ec_year   : kWh/a EV-Export in Energiegemeinschaft (nur V2H)

    Annahmen:
      - Charge-only EVs zahlen CAPEX = CEV.
      - Bidirektionale EVs zahlen CAPEX = CEV_V2H.
      - Alle EVs haben denselben OPEX-Satz (maintenance_rate_EV).
      - Nur V2H-EVs speisen ein (Vergütung Cfeed_community_EV).
      - EVs speisen nie ins öffentliche Netz.
      - BESS Replacement nach battery_lifetime.
    """

    # --- CAPEX ---
    capex_pv   = float(params['CPV'])   * float(params.get('pv_size'))
    capex_batt = float(params['CBESS']) * float(params.get('battery_capacity_kWh'))

    n_ev_charge = int(params['EV'].get('N_EV_chargeonly', 0))
    n_ev_bidir  = int(params['EV'].get('N_EV_bidirectional', 0))

    capex_ev_charge = float(params.get('CEV', 0))     * n_ev_charge
    capex_ev_bidir  = float(params.get('CEV_V2H', 0)) * n_ev_bidir
    capex_ev        = capex_ev_charge + capex_ev_bidir

    total_capex = capex_pv + capex_batt + capex_ev

    # --- OPEX ---
    opex_pv   = capex_pv   * float(params['PV'].get('maintenance_rate_PV', 0))
    opex_bess = capex_batt * float(params['BESS'].get('maintenance_rate_BESS', 0))
    opex_ev   = capex_ev   * float(params['EV'].get('maintenance_rate_EV', 0))
    opex_annual = opex_pv + opex_bess + opex_ev

    # --- Parameter & Raten ---
    lifetime = int(params.get('lifetime'))
    wacc     = float(params.get('WACC'))

    r_grid = float(params.get('electricity_price_growth', 0))
    r_feed = float(params.get('feedin_growth_rate', 0))
    r_ec   = float(params.get('ec_price_growth', r_feed))

    # --- Preise ---
    c_buy_grid   = float(params.get('Cbuy_grid'))
    c_feed_grid  = float(params.get('Cfeed_grid'))

    # Differenzierte Community-Preise (Import und Export)
    c_buy_ec_pv  = float(params.get('Cbuy_community_PV', params.get('Cbuy_community', c_buy_grid)))
    c_buy_ec_ev  = float(params.get('Cbuy_community_EV', params.get('Cbuy_community', c_buy_grid)))
    c_feed_pv_ec = float(params.get('Cfeed_community_PV', params.get('Cfeed_community', c_feed_grid)))
    c_feed_ev_ec = float(params.get('Cfeed_community_EV', params.get('Cfeed_community', c_feed_grid)))

    # --- BESS Replacement ---
    batt_life = params['BESS'].get('battery_lifetime')
    batt_life = int(batt_life) if batt_life else None
    repl_cost_nominal = 0.6 * float(params['CBESS']) * float(params.get('battery_capacity_kWh')) if batt_life else 0.0

    # --- NPC Berechnung ---
    npc = total_capex

    for y in range(1, lifetime + 1):
        # Preise mit Eskalation
        price_buy_grid   = c_buy_grid   * ((1 + r_grid) ** (y - 1))
        price_feed_grid  = c_feed_grid  * ((1 + r_feed) ** (y - 1))
        price_buy_ec_pv  = c_buy_ec_pv  * ((1 + r_ec)   ** (y - 1))
        price_buy_ec_ev  = c_buy_ec_ev  * ((1 + r_ec)   ** (y - 1))
        price_feed_pv    = c_feed_pv_ec * ((1 + r_ec)   ** (y - 1))
        price_feed_ev    = c_feed_ev_ec * ((1 + r_ec)   ** (y - 1))

        # Kosten & Erlöse
        cost_import_grid = e_import_grid_year  * price_buy_grid
        cost_import_pv   = e_import_ec_pv_year * price_buy_ec_pv
        cost_import_ev   = e_import_ec_ev_year * price_buy_ec_ev

        rev_export_grid  = e_export_grid_year  * price_feed_grid
        rev_export_pv_ec = e_export_pv_ec_year * price_feed_pv
        rev_export_ev_ec = e_export_ev_ec_year * price_feed_ev  # nur V2H liefert hier > 0

        annual_net = (cost_import_grid + cost_import_pv + cost_import_ev) \
                   - (rev_export_grid + rev_export_pv_ec + rev_export_ev_ec) \
                   + opex_annual

        # BESS Replacement
        if batt_life and (y % batt_life == 0) and y < lifetime:
            annual_net += repl_cost_nominal

        # Diskontierung
        npc += annual_net / ((1 + wacc) ** y)

    return float(npc)

