# -*- coding: utf-8 -*-
# run_all.py – Pipeline für Datensatz, Training, Optimierung, Validation

from __future__ import annotations
import os, sys, subprocess, json

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)

# ---------------- ZENTRALE PARAMETER ----------------
LOCATION   = "Vienna"
USE_V2H    = True

# Designraum & Diskretisierung
PV_MIN, PV_MAX, PV_STEP       = 1, 4000, 1
BESS_MIN, BESS_MAX, BESS_STEP = 1,   4000, 1

# Algorithmus, Stochastik/Umfang
OPT_ALGO = 'smsemoa'
SEED     = 10
SAMPLES  = 10

# Optimierungsparameter
POP = 30
GEN = 50

# Parallelisierung
WORKERS  = max(1, (os.cpu_count() or 8) - 2)
BACKEND  = "threads"   # "processes" oder "threads"

TAG      = "V2H" if USE_V2H else "NoV2H"

# Basisverzeichnisse
BASE_DIR   = os.path.join(HERE, "results", LOCATION, TAG)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR  = os.path.join(BASE_DIR, "training")
OPT_DIR     = os.path.join(BASE_DIR, "optimization")
VAL_DIR     = os.path.join(BASE_DIR, "validation")


# ---------------- Helpers ----------------
def save_config():
    cfg = {
        "location": LOCATION, "use_v2h": USE_V2H,
        "pv_min": PV_MIN, "pv_max": PV_MAX,
        "bess_min": BESS_MIN, "bess_max": BESS_MAX,
        "pv_step": PV_STEP, "bess_step": BESS_STEP,
        "samples": SAMPLES, "seed": SEED,
        "pop": POP, "gen": GEN,
        'algo': OPT_ALGO
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    out_cfg = os.path.join(MODELS_DIR, "config.json")
    with open(out_cfg, "w") as f: json.dump(cfg, f, indent=2)
    print("✔ Config gespeichert:", out_cfg)


def run_lhs():
    cmd = [sys.executable, "-m", "Optimization.Surrogat_model.LHS",
           "--location", LOCATION, "--v2h", "true" if USE_V2H else "false",
           "--samples", str(SAMPLES), "--seed", str(SEED),
           "--pv-min", str(PV_MIN), "--pv-max", str(PV_MAX),
           "--bess-min", str(BESS_MIN), "--bess-max", str(BESS_MAX),
           "--pv-step", str(PV_STEP), "--bess-step", str(BESS_STEP),
           "--workers", str(WORKERS), "--backend", BACKEND,
           "--out-dir", os.path.join(HERE, "results")]
    subprocess.run(cmd, check=True)


def run_training():
    # HINWEIS: Training-Skript geht davon aus, dass LHS-CSV in DATASET_DIR liegt
    cmd = [sys.executable, "-m", "Optimization.Surrogat_model.training",
           "--location", LOCATION, "--v2h", "true" if USE_V2H else "false",
           "--data-dir", os.path.join(HERE, "results"),
           "--models-dir", os.path.join(HERE, "results"),
           "--cv", "5", "--n-iter", "80", "--n-jobs", str(min(WORKERS, 12))]
    subprocess.run(cmd, check=True)


def run_optimize():
    cmd = [sys.executable, "-m", "Optimization.Surrogat_model.optimize_surrogat",
           "--location", LOCATION, "--v2h", "true" if USE_V2H else "false",
           "--models-dir", os.path.join(HERE, "results"),
           "--out-dir", os.path.join(HERE, "results"),
           "--pv-min", str(PV_MIN), "--pv-max", str(PV_MAX),
           "--bess-min", str(BESS_MIN), "--bess-max", str(BESS_MAX),
           "--pop", str(POP), "--gen", str(GEN), "--seed", str(SEED),
           "--algo", OPT_ALGO]

    subprocess.run(cmd, check=True)


def run_validation():
    cmd = [sys.executable, "-m", "Optimization.Surrogat_model.validation",
           "--location", LOCATION, "--v2h", "true" if USE_V2H else "false",
           "--samples", str(SAMPLES), "--seed", str(SEED),
           "--pv-min", str(PV_MIN), "--pv-max", str(PV_MAX), "--pv-step", str(PV_STEP),
           "--bess-min", str(BESS_MIN), "--bess-max", str(BESS_MAX), "--bess-step", str(BESS_STEP),
           "--models-dir", os.path.join(HERE, "results"),
           "--out-dir", os.path.join(HERE, "results")]
    subprocess.run(cmd, check=True)


# ---------------- Main ----------------
if __name__ == "__main__":
    print("== Schritt 0: Config speichern =="); save_config()
    print("== Schritt 1: Datensatz erzeugen =="); run_lhs()
    print("== Schritt 2: Training =="); run_training()
    print("== Schritt 3: Optimierung =="); run_optimize()
    print("== Schritt 4: Validation =="); run_validation()
    print("✔ Pipeline fertig.")
