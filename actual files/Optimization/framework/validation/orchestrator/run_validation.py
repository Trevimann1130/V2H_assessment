# Optimization/Surrogat_model/validation_multi/orchestrator/run_validation.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, subprocess, itertools, shutil
from pathlib import Path

# =============== ZENTRALE EINSTELLUNGEN =================
CONFIG = {
    # Orte, Varianten, Vergleiche
    "locations": ["Vienna"],               # z.B. ["Vienna","Kemi","VilaReal"]
    "use_v2h":   [False, True],            # beide Varianten prüfen
    "comparisons": [                       # Reihenfolge egal; nimm raus, was du nicht willst
        "surrogat_vs_gold",
        "surrogat_vs_fast",
        "fast_vs_gold",
    ],
    # Sampling / Parameterraum
    "samples": 30,                         # nur für LHS-Workloads relevant
    "pv":   {"min": 100.0, "max": 2000.0, "step": 10.0},
    "bess": {"min":   0.0, "max": 2000.0, "step": 10.0},
    "seed": 111,

    # Pfade (relativ zum Projektroot)
    "models_dir": "Optimization/Surrogat_model/results",
    "out_dir":    "Optimization/Surrogat_model/results",

    # Parallelisierung (Subprozesse); 1 = seriell
    "max_workers": 1,
}
# =========================================================

# Hilfsfunktion: Projektroot heuristisch (…/actual files/…)
def _project_root() -> Path:
    # dieses file: .../Optimization/Surrogat_model/validation_multi/orchestrator/run_validation.py
    here = Path(__file__).resolve()
    # 4 Ebenen nach oben bis zum Ordner "actual files" oder Projektordner
    for p in [here.parents[i] for i in range(1, 7)]:
        if (p / "Optimization").exists():
            return p
    return here.parent.parent.parent.parent  # fallback

def _bool_str(b: bool) -> str:
    return "true" if b else "false"

def _run_one(modname: str, args: list[str], cwd: Path) -> int:
    """Startet ein Modul via `python -m` mit Args; gibt Rückgabecode zurück."""
    cmd = [sys.executable, "-m", modname] + args
    print("→", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(cwd))

def main():
    root = _project_root()
    os.chdir(root)  # wichtig: relative Pfade stimmen dann überall

    # Sanity: relations-Module vorhanden?
    rel_pkg = "Optimization.Surrogat_model.validation_multi.relations"
    required = {
        "surrogat_vs_gold": f"{rel_pkg}.surrogat_vs_gold",
        "surrogat_vs_fast": f"{rel_pkg}.surrogat_vs_fast",
        "fast_vs_gold":     f"{rel_pkg}.fast_vs_gold",
    }
    missing = [k for k,v in required.items()
               if not (root / Path(*v.split("."))).with_suffix(".py").exists()]
    if missing:
        raise SystemExit(f"Relations-Module fehlen: {missing}")

    # Ausgabeordner vorbereiten
    out_root = root / CONFIG["out_dir"]
    out_root.mkdir(parents=True, exist_ok=True)

    # Schleifen über Location x V2H x Vergleich
    jobs = list(itertools.product(CONFIG["locations"], CONFIG["use_v2h"], CONFIG["comparisons"]))
    print(f"Starte {len(jobs)} Validierungs-Jobs …\n")

    for location, use_v2h, comp in jobs:
        mod = required[comp]

        common = [
            "--location", location,
            "--v2h", _bool_str(use_v2h),
            "--samples", str(CONFIG["samples"]),
            "--pv-min", str(CONFIG["pv"]["min"]),
            "--pv-max", str(CONFIG["pv"]["max"]),
            "--bess-min", str(CONFIG["bess"]["min"]),
            "--bess-max", str(CONFIG["bess"]["max"]),
            "--pv-step", str(CONFIG["pv"]["step"]),
            "--bess-step", str(CONFIG["bess"]["step"]),
            "--seed", str(CONFIG["seed"]),
            "--models-dir", CONFIG["models_dir"],
            "--out-dir",    CONFIG["out_dir"],
        ]

        # Einige Relations-Skripte (z.B. fast_vs_gold) nutzen nicht alle Flags – macht nichts.
        rc = _run_one(mod, common, cwd=root)
        if rc != 0:
            print(f"✖ Job fehlgeschlagen: {comp} | {location} | V2H={use_v2h} (rc={rc})")
            # hier NICHT abbrechen; weiterlaufen lassen
        else:
            print(f"✔ Fertig: {comp} | {location} | V2H={use_v2h}\n")

    print("\nAlle Jobs durch. Ergebnisse unter:", out_root)

if __name__ == "__main__":
    main()
