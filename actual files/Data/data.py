import pandas as pd
import numpy as np


# number of households
N_HH = 200

# number of EVs in the community
N_EV = 120

# share of EVs with/without V2H
r_EV_V2H = 0.2
r_EV_noV2H = 0

# Globale Simulationsparameter
lifetime_params = {'lifetime':25}         # Lebensdauer in Jahren, z.B. 25 Jahre für alle Komponenten


# Standortspezifische Tabellen
# Pfade zu den Lastprofilen (Load) pro Standort
location_loadprofiles = {
    'Vienna': r"C:\Users\Philipp Thunshirn\Desktop\PhD\Papers\Journals\SOO paper\Datenimport python\loadprofiles.xlsx",
    'VilaReal': r"C:\Users\Philipp Thunshirn\Desktop\PhD\Papers\Journals\SOO paper\Datenimport python\loadprofiles.xlsx",
    'Kemi': r"C:\Users\Philipp Thunshirn\Desktop\PhD\Papers\Journals\SOO paper\Datenimport python\loadprofiles.xlsx"
}

# Pfade zu den PV-Erzeugungsprofilen pro Standort
location_PVprofiles = {
    'Vienna': r"C:\Users\Philipp Thunshirn\Desktop\PhD\Papers\Journals\SOO paper\Datenimport python\Vienna\PV_Erzeugung.csv",
    'VilaReal': r'Pfad\zu\VilaReal_pv.csv',
    'Kemi': r'Pfad\zu\Kemi_pv.csv',
}

# Pfade zu Temperaturprofilen pro Standort
location_temp_profiles = {
    'Vienna': r"C:\Users\Philipp Thunshirn\Desktop\PhD\Papers\Journals\SOO paper\Datenimport python\Vienna\Tempearturdaten_Felixgasse_22.csv",
    'VilaReal': r'Pfad\zu\VilaReal_temp.csv',
    'Kemi': r'Pfad\zu\Kemi_temp.csv',
}

# Pfade zu Sonneneinstrahlungsprofilen pro Standort
location_solarirradiation_profiles = {
    'Vienna': r"C:\Users\Philipp Thunshirn\Desktop\PhD\Papers\Journals\SOO paper\Datenimport python\Vienna\Strahlungsdaten_Felixgasse22.csv",
    'VilaReal': r'Pfad\zu\VilaReal_irradiance.csv',
    'Kemi': r'Pfad\zu\Kemi_irradiance.csv',
}


# Pfade zu solar gains pro Standort
location_solargains_profiles = {
    'Vienna': r"C:\Users\Philipp Thunshirn\Desktop\PhD\Papers\Journals\SOO paper\Datenimport python\Vienna\Solar_gains.csv",

}

# Globale Tabellen
# Pfade zu usage profiles für dynamisches heat load model und Warmwasser
location_usage_profiles =  r"C:\Users\Philipp Thunshirn\Desktop\PhD\Papers\Journals\SOO paper\Datenimport python\usage_profiles.xlsx"

# Pfade zu V2H Profile auf Basis der ENTSO-E
location_V2H_profiles = r"C:\Users\Philipp Thunshirn\Desktop\PhD\Papers\Journals\SOO paper\Datenimport python\V2H profiles.xlsx"


def load_v2h_profiles(file_path):
    # Lese die Excel-Datei ein
    v2h_df = pd.read_excel(file_path, sheet_name="Yearly_profiles")  # Sheetname anpassen

    # Extrahiere Verfügbarkeit, Fahrprofile und Min SOC
    min_SOC = v2h_df['min_SOC'].values                              # Anpassen je nach Struktur
    availability_profile = v2h_df['availability_profile'].values    # Anpassen je nach Struktur
    driving_profile = v2h_df['driving_profile'].values              # Anpassen je nach Struktur

    return min_SOC, availability_profile, driving_profile


def load_profiles(location):
    # Checks
    if location not in location_loadprofiles:
        raise KeyError(f"Lastprofil für Standort '{location}' nicht definiert.")
    if location not in location_PVprofiles:
        raise KeyError(f"PV-Profil für Standort '{location}' nicht definiert.")
    if location not in location_temp_profiles:
        raise KeyError(f"Temperaturprofil für Standort '{location}' nicht definiert.")
    if location not in location_solarirradiation_profiles:
        raise KeyError(f"Sonneneinstrahlungsprofil für Standort '{location}' nicht definiert.")

    # Lastprofil: exakt 1 Spalte aus dem Excel lesen
    col_map = {
        "Vienna":   "Summe_h_multipliziert_AT",
        "Kemi":     "Summe_h_multipliziert_FI",
        "VilaReal": "Summe_h_multipliziert_PT",
    }
    col = col_map[location]
    df_load = pd.read_excel(
        location_loadprofiles[location],
        sheet_name="Haushalt_Stundenwerte",
        usecols=[col]
    )
    load_array = pd.to_numeric(df_load[col], errors="coerce").to_numpy() * N_HH

    # weitere Profile
    df_pv   = pd.read_csv(location_PVprofiles[location], sep=';', decimal=',')
    df_temp = pd.read_csv(location_temp_profiles[location], sep=';', decimal=',')
    df_irr  = pd.read_csv(location_solarirradiation_profiles[location], sep=';', decimal=',')
    usage_df = pd.read_excel(location_usage_profiles, sheet_name=0)

    pv_array   = pd.to_numeric(df_pv["PPV"], errors='coerce').to_numpy()
    temp_series = pd.Series(pd.to_numeric(df_temp["T2m"], errors='coerce').values, index=df_temp['time'])
    irr_array  = pd.to_numeric(df_irr["solar_irradiance_total"], errors='coerce').to_numpy()

    df_solargains = pd.read_csv(location_solargains_profiles[location], sep=';', decimal=',')
    solargains_array = pd.to_numeric(df_solargains["solar_gains_(W/m2)"], errors='coerce').to_numpy()

    # V2H
    min_SOC, availability_profile, driving_profile = load_v2h_profiles(location_V2H_profiles)

    return {
        'load': np.array(load_array),
        'pv_generation': np.array(pv_array),
        'T_outdoor': np.array(temp_series),
        'irradiance': np.array(irr_array),
        'min_SOC': min_SOC,
        'availability_profile': availability_profile,
        'driving_profile': driving_profile,
        'usage_profile': usage_df,
        'solargains': np.array(solargains_array)
    }

# global technical parameter (same for all locations)
technologies_global = {
    'PV': {
        'PEF': 1,                     # product environmental footprint of the PV [Pt/kWp]
        'PVdegradation': 0.0057,          # annual PV degradation  [%/a]
        'pv_reference_kwp': 1.0,         # reference value
        'maintenance_rate_PV': 0.01
    },
    'BESS': {
        'PEF': 50,                     # product environmental footprint of the BESS [Pt/kWh]
        'DoD': 0.85,                     # Depth of discharge of the BESS (%)
        'efficiency': 0.85,              # round trip efficiency of the BESS (%)
        'max_cycles': 6500,             # maximum number of cycles until BESS has to be replaced (n)
        'eol_capacity': 0.8,            # capacity of the BESS when it has to be replaced (%)
        'battery_lifetime': 15,
        'self_discharge': 0.001,        # self discharge rate. 0.001 bedeutet 0.1% pro Tag Verlust
        'power_kW': 5 * N_HH,                  # max charging/discharging capacities of battery in kW
        'maintenance_rate_BESS': 0.01        # annual maintenance costs as a percentage of investment costs.
    },
    'Grid': {
        'PEF': 270                      #  product environmental footprint of the grid in [Pt/kWp]
    },
    'heatpump': {
        'PEF': 1,                     # product environmental footprint of the HP in [Pt/kWp]
        'T_flow': 303.15,               # flow temperature of the heat transfer medium (heating water) in °C
        'T_flow_cool': 290.15,          # flow temperature of the heat transfer medium (cooling water) in °C
        'thermal_capacity_kW': 10,      # nominal heating output of the heat pump in kW
        'quality factor': 0.65,         # Carnot quality factor (φ), dimensionless
        'eer_max': 5,                   # Maximum energy efficiency ratio (dimensionless)
        'cop_max': 5,                   # Maximum COP (dimensionless)
        'eta_cop': 0.2,

    },
    'building': {
        # Exterior wall
        'U_wall': 0.12,                # Gesamt-U-Wert der Außenwand (inkl. Dämmung) in W/(m²K) -> ersetzt U_wall0 + d_insulation + lambda_insulation
        'A_wall': 752,              # Außenwandfläche [m²]

        # Windows
        'U_window': 0.7,              # U-Wert der Fenster [W/m²K]
        'A_window': {                 # Fensterflächen nach Himmelsrichtung [m²]
            'south': 30,
            'east': 30,
            'west': 30,
            'north': 30
        },

        'solar_multipliers': {        # Multiplikatoren für solare Gewinne
            'south': 1,
            'east': 0.2,
            'west': 0.2,
            'north': 0
        },
        'g_glazing': 0.6,             # g-Wert für solare Gewinne im Winter
        'g_glazing_shaded': 0.0,      # g-Wert für verschattete Fenster im Sommer

        'A_roof': 360,                # Dachfläche [m²]
        'U_roof': 0.12,                # U-Wert Dach [W/m²K]

        'A_floor': 360,               # Bodenplatte [m²]
        'U_floor': 0.2,               # U-Wert Boden [W/m²K]

        'cp_air': 0.34,               # spezifische Wärmekapazität Luft [Wh/m³K]
        'room_height': 2.5,           # Raumhöhe [m]
        'heat_capacity': 200,         # Gebäude-Wärmespeicherkapazität [Wh/K]

        'T_min': 294.15,              # Mindest-Innentemperatur [K]
        'T_max': 300.15,              # Maximal-Innentemperatur [K]

        'N_HH': N_HH                  # Anzahl Haushalte
}
    ,

    'EV': {
        'N_EV_total': N_EV,
        'N_EV_chargeonly': N_EV * r_EV_noV2H,
        '2222222222222222222222222222222222222222222': N_EV * r_EV_V2H,

        'PEF': 5,

        'capacity_kWh': 60,

        'max_soc': 1.0,
        'initial_soc': 0.5,

        'charging_efficiency': 0.95,
        'discharging_efficiency': 0.95,

        'max_charge_power': 11,          # Begrenzung Ladeleistung
        'max_discharge_power': 11,       # Begrenzung Entladeleistung

        'degradation_rate': 0.005,
        'self_discharge_EV': 0.001,      # stündlicher prozentsatz
        'maintenance_rate_EV': 0.001,

        'charge_c_rate_table': [
            {'min_temp': -273, 'max_temp': 5, 'c_rate': 1},
            {'min_temp': 278.15, 'max_temp': 283.15, 'c_rate': 1},
            {'min_temp': 283.15, 'max_temp': 318.15, 'c_rate': 1},
            {'min_temp': 318.15, 'max_temp': 333.15, 'c_rate': 1},
            {'min_temp': 333.15, 'max_temp': 1000, 'c_rate': 1},
        ],
        'discharge_c_rate_table': [
            {'min_temp': 0, 'max_temp': 263.15, 'c_rate': 1},
            {'min_temp': 263.15, 'max_temp': 273.15, 'c_rate': 1},
            {'min_temp': 273.15, 'max_temp': 318.15, 'c_rate': 1.0},
            {'min_temp': 318.15, 'max_temp': 333.15, 'c_rate': 1},
            {'min_temp': 333.15, 'max_temp': 1000, 'c_rate': 1},
        ]
    }
}

# location specific parameter
technologies_local = {
    'Vienna': {
        # PV costs
        'CPV': 1500,    # investment costs PV [€/kWp]
        'Cfeed_community_PV': 0.1,

        # BESS costs
        'CBESS': 500,   # investment costs BESS [€/kWh]

        # EV costs
        'CEV': 2000,
        'CEV_V2H': 2500,    # V2H Kosten pro Auto
        'Cfeed_community_EV': 0.10,

        # electricity prices from grid
        'Cbuy_grid': 0.33,   # price electricity from the grid [€/kWh]
        'Cfeed_grid': 0.07,  # revenue from selling electricity to the grid [€/kWh]

        # electricity prices energy community
        'Cbuy_community': 0.32,

        # heatpump costs
        'CHP': 1000,    # investment costs HP [€/kW_th]
        'WACC': 0.09,   # discount rate
        'feedin_growth_rate': 0.0 ,      # jährliches Wachstum der Einspeisevergütung
        'electricity_price_growth': 0.02, # jährliches Wachstum der Netzstrompreise
        #'reference_market_price_pv': {2023: [0.1556, 0.1399, 0.1048, 0.0856, 0.0543, 0.0754, 0.0653, 0.075, 0.077, 0.0814, 0.092, 0.0748]
        },


    'VilaReal': {
        'CPV': 1100,
        'CBESS': 600,
        'Cbuy': 0.30,
        'Csell': 0.08,
        'CHP': 950,
        'WACC': 0.08,
        'feedin_growth_rate': 0.01,
        'electricity_price_growth': 0.02,
        'reference_market_price_pv': {
            2023: [0.110, 0.115, 0.118, 0.122, 0.124, 0.120, 0.118, 0.114, 0.110, 0.108, 0.106, 0.105]
        }
    },

    'Kemi': {
        'CPV': 1300,
        'CBESS': 580,
        'Cbuy': 0.32,
        'Csell': 0.09,
        'CHP': 980,
        'WACC': 0.08,
        'feedin_growth_rate': 0.01,
        'electricity_price_growth': 0.02,
        'reference_market_price_pv': {
            2023: [0.110, 0.115, 0.118, 0.122, 0.124, 0.120, 0.118, 0.114, 0.110, 0.108, 0.106, 0.105]
        }
    }
}


def get_parameters(location):
    if location not in technologies_local:
        raise KeyError(f"Ökonomische Parameter für Standort '{location}' nicht definiert.")

    params = {}

    # technologies_global als verschachtelte Struktur übernehmen
    params.update(technologies_global)

    # Füge lifetime aus globalen Simulationsparametern hinzu
    params['lifetime'] = lifetime_params['lifetime']

    params['N_EV'] = N_EV

    # Standortabhängige Parameter hinzufügen
    params.update(technologies_local[location])

    return params




