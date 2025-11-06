# Optimization/framework/Orchestrator/optimize.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any
import os

# Deine Engine – exakt so wie in deiner Struktur
from Optimization.framework.engines.Surrogat_model.surrogate_engine import SurrogateEngine


def _resolve_runs_root(output_root: str) -> Path:
    """
    Normalisiert den Basispfad der Optimierungsergebnisse.
    - Absolute Pfade: unverändert
    - Relative Pfade: relativ zur Projektwurzel
    - Entfernt Doppel-Präfixe wie '.../Optimization/run/Optimization/run/results'
    """
    project_root = Path(__file__).resolve().parents[3]
    p = Path(output_root)

    runs_root = p if p.is_absolute() else (project_root / p)
    runs_root = runs_root.resolve()

    # Doppeltes ".../Optimization/run/results" bereinigen
    parts = list(runs_root.parts)
    canonical = ("Optimization", "run", "results")
    cleaned, seen, i = [], 0, 0
    while i < len(parts):
        if i + 3 <= len(parts) and tuple(parts[i:i+3]) == canonical:
            seen += 1
            if seen > 1:
                i += 3
                continue
        cleaned.append(parts[i]); i += 1
    if cleaned != parts:
        runs_root = Path(*cleaned)

    runs_root.mkdir(parents=True, exist_ok=True)
    return runs_root


def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def run(s) -> Dict[str, Any]:
    """
    Startet die echte Surrogat-Pipeline (die bereits im __init__ der Engine läuft)
    und setzt nur Pfade/_LATEST_RUN sauber.
    """
    # 1) Kanonischer Root
    runs_root = _resolve_runs_root(str(s.reporting.output_root))

    # 2) Pfadkomponenten
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{s.run.tag}"
    loc = s.engine.location
    mode = "V2H" if getattr(s.engine, "use_v2h", False) else "NoV2H"
    engine_label = (getattr(s.engine, "name", "surrogate") or "surrogate").lower().strip()

    run_dir = (runs_root / loc / mode / engine_label / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[optimize] results root: {runs_root}")
    print(f"[optimize] run dir     : {run_dir}")
    print(f"[optimize] run id      : {run_id}")

    # 3) ECHTE Pipeline: die läuft in __init__.
    #    Falls deine Engine ein run_dir-Argument im ctor unterstützt, übergeben wir es.
    #    Wenn nicht, stiller Fallback ohne run_dir.
    try:
        # viele Implementationen akzeptieren 'run_dir' im ctor:
        _ = SurrogateEngine(settings=s, run_dir=str(run_dir))
    except TypeError:
        # ältere Signaturen ohne run_dir:
        _ = SurrogateEngine(settings=s)

    # 4) _LATEST_RUN Marker (damit Validation 'latest' findet)
    engine_root = (runs_root / loc / mode / engine_label).resolve()
    _write_json(engine_root / "_LATEST_RUN.json", {"run_id": run_id})
    (engine_root / "_LATEST_RUN.txt").write_text(run_id, encoding="utf-8")

    # 5) Rückgabe
    return {"run_dir": str(run_dir), "run_id": run_id}
