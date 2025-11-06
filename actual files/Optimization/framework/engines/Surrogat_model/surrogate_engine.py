# Optimization/framework/engines/Surrogat_model/surrogate_engine.py
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
import joblib

from Optimization.framework.Problem.builder import EngineAdapter
from Optimization.framework.engines.Surrogat_model.training import auto_train_surrogate
from Cost_model.financial_model import calculate_npc_yearly
from Data.data import get_parameters


class SurrogateEngine(EngineAdapter):
    def __init__(self, settings):
        self.s = settings
        self._models_F: List[Any] = []
        self._targets: List[str] = []
        self._meta: Dict[str, Any] = {}

        # Train/Load surrogate artifact (deine bestehende Pipeline)
        path = getattr(self.s.engine, "surrogate_artifact_path", None)
        if not (isinstance(path, str) and path.strip() and os.path.exists(path)):
            # Triggert: Sampling → Teacher → Training → Holdout → Speichern ... (deine Logs)
            path = auto_train_surrogate(self.s)
            self.s.engine.surrogate_artifact_path = path

        obj = joblib.load(path)
        self._models_F = obj["F"]
        self._meta = obj.get("meta", {})
        self._targets = list(self._meta.get("surrogate_targets", []))

        # Quelle des Artefakts für spätere Spiegelung ins run_dir
        self._artifact_path = Path(path).resolve()
        self._artifact_dir = self._artifact_path.parent  # z. B. .../surrogate_YYYYMMDD_HHMMSS

    # ---------------- helpers ----------------
    def _flows_dict(self, y_row: np.ndarray) -> Dict[str, float]:
        return {k: float(v) for k, v in zip(self._targets, y_row)}

    def _params(self) -> Dict[str, Any]:
        eng = self.s.engine
        p = get_parameters(eng.location)
        p["location"] = eng.location
        if "EC" not in p:
            p["EC"] = {}
        p["EC"]["share"] = float(eng.ec_share_import)
        p["EC"]["export_share"] = float(eng.ec_share_export)
        p["N_HH"] = int(eng.N_HH)
        p["N_EV"] = int(eng.N_EV_total)
        p["N_EV_bidirectional"] = int(eng.N_EV_bidirectional)
        return p

    def _compute_objectives(self, flows_L: Dict[str, float], pv_kwp: float, bess_kwh: float) -> Dict[str, float]:
        params = self._params()
        L = int(params["lifetime"])
        PVp, BSp, Grid = params["PV"], params["BESS"], params["Grid"]
        EV = params.get("EV", {})

        def Y(k: str) -> float:
            return float(flows_L.get(k, 0.0)) / L

        e_import_grid_Y = Y("E_import_grid_kWh")
        e_export_grid_Y = Y("E_export_grid_kWh")
        e_import_ec_pv_Y = Y("E_import_ec_pv_kWh")
        e_import_ec_ev_Y = Y("E_import_ec_ev_kWh")

        params_fin = dict(params)
        params_fin["pv_size"] = float(pv_kwp)
        params_fin["battery_capacity_kWh"] = float(bess_kwh)
        npc = float(
            calculate_npc_yearly(
                params_fin,
                e_import_grid_year=e_import_grid_Y,
                e_import_ec_pv_year=e_import_ec_pv_Y,
                e_import_ec_ev_year=e_import_ec_ev_Y,
                e_export_grid_year=e_export_grid_Y,
                e_export_pv_ec_year=0.0,
                e_export_ev_ec_year=0.0,
            )
        )

        pef_embodied = float(PVp["PEF"]) * float(pv_kwp) + float(BSp["PEF"]) * float(bess_kwh)
        pef_oper = (
            float(Grid["PEF"]) * float(flows_L.get("E_import_grid_kWh", 0.0))
            + float(PVp["PEF"]) * float(flows_L.get("E_import_ec_pv_kWh", 0.0))
            + float(EV.get("PEF", 0.0)) * float(flows_L.get("E_import_ec_ev_kWh", 0.0))
        )
        pef = pef_embodied + pef_oper

        out = {}
        for name in self.s.objectives.names:
            if name == "npc_eur":
                out[name] = npc
            elif name == "pef_pt":
                out[name] = pef
            elif name == "grid_import_kwh":
                out[name] = float(flows_L.get("E_import_grid_kWh", 0.0))
            else:
                raise ValueError(f"[surrogate] unknown objective '{name}'")
        return out

    # ---------------- API ----------------
    def evaluate(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Y_pred = np.column_stack([m.predict(X) for m in self._models_F])

        F_rows = []
        for i in range(X.shape[0]):
            pv, bess = float(X[i, 0]), float(X[i, 1])
            flows_L = self._flows_dict(Y_pred[i, :])
            obj = self._compute_objectives(flows_L, pv, bess)
            F_rows.append([float(obj[n]) for n in self.s.objectives.names])

        F = np.asarray(F_rows, float)
        G = np.zeros((F.shape[0], 0), float)
        return F, G

    # ---------------- NEW: thin wrapper for orchestrator ----------------
    def run(self, run_dir: str | Path) -> Dict[str, Any]:
        """
        Minimal-invasiv:
        - nutzt die bestehende Trainings-/Holdout-Pipeline (lief bereits im __init__ via auto_train_surrogate)
        - spiegelt **alle** Artefakte/Reports aus dem artefakt-Quellordner in das vom Orchestrator
          vorgegebene run_dir, damit CSV/Plots/Summary/Joblib unter der RUN-ID liegen.
        """
        out = Path(run_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)

        src = self._artifact_dir
        if src.exists() and src.is_dir():
            # Inhalt 1:1 nach run_dir mergen (Dateien + Unterordner)
            for item in src.iterdir():
                dst = out / item.name
                if item.is_dir():
                    if dst.exists():
                        # merge: kopiere Inhalte rekursiv
                        for sub in item.rglob("*"):
                            rel = sub.relative_to(item)
                            target = dst / rel
                            if sub.is_dir():
                                target.mkdir(parents=True, exist_ok=True)
                            else:
                                target.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(str(sub), str(target))
                    else:
                        shutil.copytree(str(item), str(dst))
                else:
                    shutil.copy2(str(item), str(dst))
        else:
            # Fallback: nur das Modell selbst sichern
            (out / "artifact").mkdir(exist_ok=True, parents=True)
            if self._artifact_path.exists():
                shutil.copy2(str(self._artifact_path), str(out / "artifact" / self._artifact_path.name))

        # Kleine Zusatz-Meta (hilft Debug)
        meta_hint = {
            "surrogate_targets": self._targets,
            "artifact_path": str(self._artifact_path),
            "artifact_source_dir": str(self._artifact_dir),
        }
        with (out / "surrogate_meta_hint.json").open("w", encoding="utf-8") as f:
            import json
            json.dump(meta_hint, f, indent=2)

        return {
            "ok": True,
            "run_dir": str(out),
            "n_models": len(self._models_F),
            "targets": self._targets,
        }
