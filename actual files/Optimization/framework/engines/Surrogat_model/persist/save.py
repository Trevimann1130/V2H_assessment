# Optimization/framework/engines/Surrogat_model/persist/save.py
from __future__ import annotations

import os
import json
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _join(*parts: str) -> str:
    return os.path.join(*parts)


def make_outdir(settings) -> str:
    """
    Artefakt-Ordner:
      {settings.reporting.output_root}/{location}/{tag}/surrogate_{timestamp}/
    """
    base = str(settings.reporting.output_root)
    loc = str(settings.engine.location)
    tag = str(settings.run.tag)
    d = _join(base, loc, tag, f"surrogate_{_ts()}")
    os.makedirs(d, exist_ok=True)
    return d


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build_meta_dict(settings, holdout_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Schlankes, vollständiges Meta – ohne Defaults/Fallbacks.
    """
    return {
        "engine": {
            "name": settings.engine.name,
            "system_id": settings.engine.system_id,
            "location": settings.engine.location,
            "use_v2h": bool(settings.engine.use_v2h),
            "ec_share_import": float(settings.engine.ec_share_import),
            "ec_share_export": float(settings.engine.ec_share_export),
            "N_HH": int(settings.engine.N_HH),
            "N_EV_total": int(settings.engine.N_EV_total),
            "N_EV_bidirectional": int(settings.engine.N_EV_bidirectional),
            "rng_seed": int(settings.engine.rng_seed),
        },
        "bounds": {
            "names": list(settings.bounds.names),
            "lower": list(settings.bounds.lower),
            "upper": list(settings.bounds.upper),
            "steps": list(settings.bounds.steps or [None] * len(settings.bounds.names)),
        },
        "objectives": {
            "names": list(settings.objectives.names),
            "minimize": list(settings.objectives.minimize),
        },
        "constraints": {
            "names": list(settings.constraints.names or []),
            "senses": list(settings.constraints.senses or []),
            "rhs": list(settings.constraints.rhs or []),
        },
        "sampler": {
            "name": settings.sampler.name,
            "n_samples": int(settings.sampler.n_samples),
            "seed": int(settings.sampler.seed),
            "kwargs": dict(settings.sampler.kwargs or {}),
        },
        "surrogate_train": {
            "model_type": settings.surrogate_train.model_type,
            "rf_n_estimators": int(settings.surrogate_train.rf_n_estimators),
            "rf_n_jobs": int(settings.surrogate_train.rf_n_jobs),
            "holdout_frac": float(settings.surrogate_train.holdout_frac),
        },
        "optimizer": {
            "name": settings.optimizer.name,
            "kwargs": dict(settings.optimizer.kwargs or {}),
            "seed": int(settings.optimizer.seed),
            "n_jobs": int(settings.optimizer.n_jobs),
        },
        "reporting": {
            "output_root": settings.reporting.output_root,
        },
        "run": {
            "tag": settings.run.tag,
        },
        "holdout": holdout_metrics or {},
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def _write_holdout_files(artifact_dir: str, holdout: Dict[str, Any]) -> None:
    """
    Schreibt Holdout-Metriken (falls vorhanden) ins Artefakt:
      surrogate_.../holdout/metrics_F_objectives.csv
      surrogate_.../holdout/metrics_G_constraints.csv (optional)
      surrogate_.../holdout/metrics.json
      surrogate_.../holdout/summary.md
    """
    if not holdout:
        return

    hdir = _join(artifact_dir, "holdout")
    os.makedirs(hdir, exist_ok=True)

    # JSON (vollständige Struktur)
    _write_json(_join(hdir, "metrics.json"), holdout)

    def _df_from_metrics(m: Any) -> pd.DataFrame:
        """
        Akzeptiert:
          - neue Dict-Form: {"targets":[...], "r2":[...], "rmse":[...], "mae":[...], "rel_mae_percent":[...]}
          - alte Listenform: [{"target": "...", "r2":..., "rmse":..., "mae":..., "rel_mae_percent":...}, ...]
        """
        if isinstance(m, dict) and "targets" in m:
            return pd.DataFrame({
                "target": m["targets"],
                "r2": m["r2"],
                "rmse": m["rmse"],
                "mae": m["mae"],
                "rel_mae_percent": m["rel_mae_percent"],
            })
        if isinstance(m, list):
            return pd.DataFrame(m)
        raise ValueError("[persist] Unknown metrics format for holdout.")

    # CSVs
    if "F" in holdout and holdout["F"]:
        dfF = _df_from_metrics(holdout["F"])
        dfF.to_csv(_join(hdir, "metrics_F_objectives.csv"), index=False)
    if "G" in holdout and holdout["G"]:
        dfG = _df_from_metrics(holdout["G"])
        dfG.to_csv(_join(hdir, "metrics_G_constraints.csv"), index=False)

    # Summary
    with open(_join(hdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Holdout Validation\n\n")
        if "F" in holdout and holdout["F"]:
            f.write("## Objectives (F)\n")
            f.write("- Targets: " + ", ".join(holdout["F"].get("targets", [])) + "\n")
            f.write("- R² (median): " + f"{np.nanmedian(holdout['F'].get('r2', [])):.3f}\n")
            f.write("- RMSE (median): " + f"{np.nanmedian(holdout['F'].get('rmse', [])):.3f}\n\n")
        if "G" in holdout and holdout["G"]:
            f.write("## Constraints (G)\n")
            f.write("- Targets: " + ", ".join(holdout["G"].get("targets", [])) + "\n")
            f.write("- R² (median): " + f"{np.nanmedian(holdout['G'].get('r2', [])):.3f}\n")
            f.write("- RMSE (median): " + f"{np.nanmedian(holdout['G'].get('rmse', [])):.3f}\n\n")


def persist_artifact(artifact_dir: str,
                     models_F: List[Any],
                     models_G: List[Any],
                     meta: Dict[str, Any]) -> str:
    """
    Speichert:
      - surrogate_rf.joblib (Modelle + Meta)
      - meta.json
      - holdout/* (Metriken) – wenn vorhanden
    """
    obj = {"F": models_F, "G": models_G, "meta": meta}
    joblib_path = _join(artifact_dir, "surrogate_rf.joblib")
    joblib.dump(obj, joblib_path)

    _write_json(_join(artifact_dir, "meta.json"), meta)

    # Holdout-Metriken als Dateien (wenn vorhanden)
    holdout = meta.get("holdout", {})
    _write_holdout_files(artifact_dir, holdout)

    print(f"[surrogate] artifact saved: {joblib_path}")
    return joblib_path


# --------- Spiegelung in zentralen validation/-Ordner ---------

def _validation_root(settings) -> str:
    # Parallel zu results: Optimization/run/validation/<Location>/<Tag>/
    return _join("Optimization", "run", "validation", settings.engine.location, settings.run.tag)


def _copy_if_exists(src: str, dst: str) -> None:
    if os.path.exists(src):
        shutil.copy2(src, dst)


def mirror_holdout_to_validation(settings, artifact_dir: str, meta: Dict[str, Any]) -> None:
    """
    Spiegelt die Holdout-CSV(s) in:
      Optimization/run/validation/<Location>/<Tag>/
    und führt einen 'runs_index.csv', der Artefakte übersichtlich auflistet.
    """
    vroot = _validation_root(settings)
    os.makedirs(vroot, exist_ok=True)

    # Hole Zeitstempel aus Artefakt-Verzeichnisnamen surrogate_<ts>
    base = os.path.basename(artifact_dir)
    ts = base.replace("surrogate_", "") if base.startswith("surrogate_") else _ts()

    # Quelldateien
    srcF = _join(artifact_dir, "holdout", "metrics_F_objectives.csv")
    srcG = _join(artifact_dir, "holdout", "metrics_G_constraints.csv")

    # Ziel-Dateinamen mit Kontext
    dstF = _join(vroot, f"{settings.engine.location}.{settings.run.tag}_{ts}_metrics_F.csv")
    dstG = _join(vroot, f"{settings.engine.location}.{settings.run.tag}_{ts}_metrics_G.csv")

    _copy_if_exists(srcF, dstF)
    _copy_if_exists(srcG, dstG)

    # Index-CSV anhängen/erstellen
    index_path = _join(vroot, "runs_index.csv")
    row = {
        "timestamp": ts,
        "location": settings.engine.location,
        "tag": settings.run.tag,
        "sampler": meta["sampler"]["name"],
        "n_samples": meta["sampler"]["n_samples"],
        "seed": meta["sampler"]["seed"],
        "rf_n_estimators": meta["surrogate_train"]["rf_n_estimators"],
        "rf_n_jobs": meta["surrogate_train"]["rf_n_jobs"],
        "holdout_frac": meta["surrogate_train"]["holdout_frac"],
        "artifact_dir": os.path.abspath(artifact_dir),
        "metrics_F_csv": os.path.abspath(dstF) if os.path.exists(dstF) else "",
        "metrics_G_csv": os.path.abspath(dstG) if os.path.exists(dstG) else "",
    }

    if os.path.exists(index_path):
        df = pd.read_csv(index_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(index_path, index=False)
    print(f"[surrogate] validation mirror: {vroot}")

def write_holdout_csv(validation_dir: str,
                      location: str,
                      tag: str,
                      rows: list[dict],
                      kind: str) -> str:
    """
    Schreibt eine Holdout-CSV in den Validation-Ordner.
    kind: "F" oder "G"
    """
    import csv, os
    os.makedirs(validation_dir, exist_ok=True)
    out_path = os.path.join(validation_dir, f"{location}.{tag}_holdout_{kind}.csv")
    if not rows:
        # leere Datei mit Header
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["target","r2","rmse","mae","rel_mae_percent"])
            w.writeheader()
        return out_path

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["target","r2","rmse","mae","rel_mae_percent"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out_path
