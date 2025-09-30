# -*- coding: utf-8 -*-
# 01_make_dataset_fast.py – FAST-Datensatz (LHS) erzeugen

from __future__ import annotations
import os, argparse, numpy as np, pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from Optimization.Vectorized_model.precompute import prepare_profiles
from Optimization.Vectorized_model.system_model_precomputed import (
    simulate_energy_system, simulate_energy_system_with_v2h,
)
from Data.data import get_parameters, load_profiles

# ---------- Worker-Globals ----------
_G = {"params": None, "profiles": None, "use_v2h": False}

def _worker_init(base_params, profiles, use_v2h, v2h_seed: int | None):
    _G["params"] = base_params
    _G["profiles"] = profiles
    _G["use_v2h"] = bool(use_v2h)
    if v2h_seed is not None:
        np.random.seed(int(v2h_seed) + (os.getpid() % 100_000))

def _eval_point_only(point):
    return evaluate_one(point, _G["params"], _G["profiles"], _G["use_v2h"])

# ---------- Simulate lifetime ----------
def simulate_over_lifetime_fast(sim_func, params, profiles, pv_size):
    lifetime = int(params.get("lifetime"))
    degr     = float(params["PV"].get("PVdegradation"))
    nH       = len(profiles["load"])
    base = sim_func(params, profiles, pv_size)

    def arr1d(key):
        a = base.get(key, None)
        if a is None: return np.zeros(nH, dtype=float)
        a = np.asarray(a)
        if a.ndim == 2: a = a.sum(axis=1)
        return a.astype(float, copy=False)

    pv_base  = arr1d("pv_generation")
    gi_base  = arr1d("grid_import")
    ge_base  = arr1d("grid_export")
    bc_base  = arr1d("bess_charged")
    bd_base  = arr1d("bess_discharged")
    hpH_base = arr1d("heatpump_results_heating")
    hpC_base = arr1d("heatpump_results_cooling")
    ec_pv_base = arr1d("ec_import_from_pv")  # ⬅️ neu
    ec_ev_base = arr1d("ec_import_from_ev")  # ⬅️ neu
    ev_ch_base = arr1d("ev_charged")
    ev_ds_base = arr1d("ev_discharged")
    trip_loss  = arr1d("trip_loss")

    pv_self_base = pv_base - ge_base
    with np.errstate(divide="ignore", invalid="ignore"):
        s_self = np.where(pv_base > 1e-9,
                          np.clip(pv_self_base / pv_base, 0.0, 1.0), 0.0)

    f_year = np.array([(1.0 - degr) ** y for y in range(lifetime)], dtype=float)

    def stack_scaled(a_base, scale="pv"):
        if a_base.size == 0: return np.zeros(0, dtype=float)
        if scale == "pv":
            return np.concatenate([np.clip(a_base * f, 0.0, None) for f in f_year])
        else:
            return np.concatenate([np.clip(a_base, 0.0, None) for _ in f_year])

    pv_all, gi_all, ge_all = [], [], []
    for f in f_year:
        pv_y = pv_base * f
        ge_y = np.clip(ge_base + (pv_y - pv_base) * (1.0 - s_self), 0.0, None)
        pv_self_y = pv_y * s_self
        gi_y = np.clip(gi_base + (pv_self_base - pv_self_y), 0.0, None)
        pv_all.append(pv_y); ge_all.append(ge_y); gi_all.append(gi_y)

    cat = (lambda lst: np.concatenate(lst) if lst else np.zeros(0, dtype=float))
    results = {
        "pv_generation": cat(pv_all),
        "grid_import":   cat(gi_all),
        "grid_export":   cat(ge_all),
        "bess_charged":  stack_scaled(bc_base, "pv"),
        "bess_discharged": stack_scaled(bd_base, "pv"),
        "heatpump_results_heating": stack_scaled(hpH_base, "flat"),
        "heatpump_results_cooling": stack_scaled(hpC_base, "flat"),
        "ec_import_from_pv": stack_scaled(ec_pv_base, "flat"),
        "ec_import_from_ev": stack_scaled(ec_ev_base, "flat"),
    }
    if ev_ch_base.any() or ev_ds_base.any() or trip_loss.any():
        results["ev_charged"]    = stack_scaled(ev_ch_base, "pv")
        results["ev_discharged"] = stack_scaled(ev_ds_base, "pv")
        results["trip_loss"]     = stack_scaled(trip_loss, "flat")
    return results

# ---------- LHS ----------
def lhs_in_bounds(n, low, high, rng):
    cut = np.linspace(0, 1, n + 1, dtype=float)
    u = rng.uniform(cut[:-1], cut[1:])
    rng.shuffle(u)
    return low + u * (high - low)

def make_lhs_matrix(n, bounds, seed):
    rng = np.random.default_rng(seed)
    cols = [lhs_in_bounds(n, lo, hi, rng) for (lo, hi) in bounds]
    return np.column_stack(cols)

def snap_to_steps(X, steps):
    Xs = X.copy()
    for j, step in enumerate(steps):
        if step and step > 0:
            Xs[:, j] = np.round(Xs[:, j] / step) * step
    return Xs

# ---------- Evaluate ----------
def evaluate_one(point, base_params, profiles, use_v2h):
    pv_size, batt_kwh = float(point[0]), float(point[1])
    params = dict(base_params)
    params["pv_size"] = pv_size
    params["battery_capacity_kWh"] = batt_kwh
    sim = simulate_energy_system_with_v2h if use_v2h else simulate_energy_system
    res = simulate_over_lifetime_fast(sim, params, profiles, pv_size)

    return (
        pv_size, batt_kwh,
        float(np.sum(res["grid_import"])),
        float(np.sum(res["ec_import_from_pv"])),
        float(np.sum(res["ec_import_from_ev"])),
        float(np.sum(res["grid_export"])),
        float(np.sum(res["bess_charged"])),
        float(np.sum(res["heatpump_results_heating"])),
        float(np.sum(res["heatpump_results_cooling"])),
        float(np.sum(res["pv_generation"])),
        float(np.sum(res["ev_charged"])) if "ev_charged" in res else 0.0,
        float(np.sum(res["ev_discharged"])) if "ev_discharged" in res else 0.0,
        float(np.sum(res["trip_loss"])) if "trip_loss" in res else 0.0,
    )


# ---------- Batch ----------
def evaluate_batch(X, base_params, profiles, use_v2h, backend, n_workers, chunksize=16, show_progress=True, v2h_seed=None):
    if backend == "processes":
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_worker_init,
                                 initargs=(base_params, profiles, use_v2h, v2h_seed)) as ex:
            it = ex.map(_eval_point_only, X, chunksize=chunksize)
            it = tqdm(it, total=len(X), desc="FAST dataset (proc)") if show_progress else it
            rows = list(it)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            it = ex.map(lambda p: evaluate_one(p, base_params, profiles, use_v2h), X)
            it = tqdm(it, total=len(X), desc="FAST dataset (thread)") if show_progress else it
            rows = list(it)

    cols = ["PV_kWp",
            "BESS_kWh",
            "E_import_grid_kWh",
            "E_import_ec_pv_kWh",
            "E_import_ec_ev_kWh",
            "E_export_grid_kWh",
            "E_bess_throughput_kWh",
            "E_hp_heat_kWh",
            "E_hp_cool_kWh",
            "E_pv_gen_kWh",
            "E_ev_charged_kWh",
            "E_ev_discharged_kWh",
            "E_ev_trip_loss_kWh"]

    df = pd.DataFrame.from_records(rows, columns=cols)
    ev_cols = ["E_ev_charged_kWh","E_ev_discharged_kWh","E_ev_trip_loss_kWh"]
    if all(c in df.columns for c in ev_cols) and df[ev_cols].sum().sum() == 0.0:
        df = df.drop(columns=ev_cols)
    return df

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", required=True)
    ap.add_argument("--v2h", type=str, default="false")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--pv-min", type=float, required=True)
    ap.add_argument("--pv-max", type=float, required=True)
    ap.add_argument("--bess-min", type=float, required=True)
    ap.add_argument("--bess-max", type=float, required=True)
    ap.add_argument("--pv-step", type=float, required=True)
    ap.add_argument("--bess-step", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1,(os.cpu_count() or 4)-1))
    ap.add_argument("--backend", choices=["processes","threads"], default="processes")
    ap.add_argument("--chunksize", type=int, default=16)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--v2h-seed", type=int, default=None)
    ap.add_argument("--out-dir", default="Optimization/Surrogat_model/results")
    return ap.parse_args()

def main():
    args = parse_args()
    LOCATION = args.location
    USE_V2H  = str(args.v2h).lower() in ("1","true","yes","y")
    TAG      = "V2H" if USE_V2H else "NoV2H"

    # ✅ konsistente Ordnerstruktur
    out_dir = os.path.join(args.out_dir, LOCATION, TAG, "dataset")
    os.makedirs(out_dir, exist_ok=True)

    base_params = get_parameters(LOCATION)
    base_params["location"] = LOCATION
    profiles_raw = load_profiles(LOCATION)
    profiles = prepare_profiles(base_params, profiles_raw, do_hp_electricity=True, do_coeffs=False)

    bounds = [(args.pv_min,args.pv_max),(args.bess_min,args.bess_max)]
    X0 = make_lhs_matrix(args.samples, bounds, args.seed)
    X  = snap_to_steps(X0, [args.pv_step,args.bess_step])
    X  = np.unique(X, axis=0)

    df = evaluate_batch(X, base_params, profiles, USE_V2H, args.backend,
                        args.workers, args.chunksize, not args.no_progress, args.v2h_seed)
    df["location"]=LOCATION; df["use_v2h"]=USE_V2H; df["lifetime_y"]=int(base_params.get("lifetime",25))

    base=f"{LOCATION}.{TAG}_fast_dataset"
    df.to_parquet(os.path.join(out_dir, base+".parquet"), index=False)
    df.to_csv(os.path.join(out_dir, base+".csv"), index=False)
    print(f"✔ Dataset gespeichert in {out_dir}")

if __name__=="__main__":
    import multiprocessing as mp; mp.freeze_support(); main()
