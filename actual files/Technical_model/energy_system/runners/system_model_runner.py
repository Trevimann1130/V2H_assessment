# Technical_model/energy_system/runners/system_model_runner.py
# -----------------------------------------------------------------------------
# System-Runner (system_id-first, settings-only)
# - alles kommt aus settings (keine Defaults hier!)
# - Precompute -> params, profiles
# - EC-Shares strikt im params-Dict setzen (kein Import nötig)
# - System über Registry wählen & ausführen
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Dict, Any

# Technical_model/energy_system/runners/system_model_runner.py
from Technical_model.energy_system.precompute.adapter import prepare_profiles_adapter
from Technical_model.energy_system.systems.registry_systems import get as get_system



def _get(d: Dict[str, Any], path: str) -> Any:
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(f"Missing settings key: '{path}'")
        cur = cur[p]
    return cur


def run_from_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    # 1) Pflichtangaben
    location   = _get(settings, "location")
    system_id  = _get(settings, "system_id")
    data_src   = settings.get("data_source", None)
    pv_size    = float(_get(settings, "tech.pv.pv_size_kwp"))

    # 2) Precompute
    params, profiles = prepare_profiles_adapter(location=location, data_source=data_src)

    # 3) EC-Shares strikt ins params-Dict schreiben (kein set_ec_shares-Import nötig)
    ec_share        = float(_get(settings, "ec.share"))
    ec_export_share = float(_get(settings, "ec.export_share"))
    if "EC" not in params or not isinstance(params["EC"], dict):
        raise KeyError("params['EC'] must exist and be a dict before setting EC shares.")
    params = dict(params)
    params["EC"] = dict(params["EC"])
    params["EC"]["share"] = ec_share
    params["EC"]["export_share"] = ec_export_share

    # 4) System via Registry und ausführen
    run_system = get_system(system_id)  # -> callable(params, profiles, pv_size, run_checks=False)
    results, hourly = run_system(params, profiles, pv_size, run_checks=False)

    return {
        "system_id": system_id,
        "location": location,
        "settings_used": settings,
        "results": results,
        "hourly": hourly,
    }

# --- am Ende von Technical_model/energy_system/runners/system_model_runner.py ---
if __name__ == "__main__":
    import argparse, json, importlib

    ap = argparse.ArgumentParser(description="Run technical energy system from settings")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--settings-json", help="Pfad zu JSON-Datei mit Settings-Dict")
    g.add_argument("--settings-py",   help="Import-Spezifikation 'paket.modul:VARNAME' mit Settings-Dict")
    args = ap.parse_args()

    if args.settings_json:
        with open(args.settings_json, "r", encoding="utf-8") as f:
            settings = json.load(f)
    else:
        mod_name, var_name = args.settings_py.split(":")
        mod = importlib.import_module(mod_name)
        settings = getattr(mod, var_name)
        if not isinstance(settings, dict):
            raise TypeError(f"{args.settings_py} must be a dict")

    out = run_from_settings(settings)
    print("✔ run ok")
    print("system_id:", out["system_id"])
    print("location :", out["location"])
    try:
        print("hourly shape:", out["hourly"].shape)
    except Exception:
        print("hourly type :", type(out["hourly"]))
