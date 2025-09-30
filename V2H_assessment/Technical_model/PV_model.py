import numpy as np

def simulate_pv_system(pv_size, load_demand, pv_generation, params):

# scaling PV generation
    gen = (pv_generation/1000) * (pv_size / params['PV']['pv_reference_kwp'])
    hours = len(gen)

# Considering degradation
    years = np.arange(hours) / (365 * 24)
    degradation_factor = (1 - params['PV']['PVdegradation']) ** years
    gen *= degradation_factor



    return {
        'pv_production': gen
    }

