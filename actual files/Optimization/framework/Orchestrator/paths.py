# Optimization/framework/Orchestrator/paths.py
from pathlib import Path
from datetime import datetime
import os

def _scenario_tag(system_id: str) -> str:
    return "V2H" if "V2H" in system_id.upper() else "NoV2H"

def make_run_dir(output_root: str, tag: str, location: str, system_id: str, engine_name: str) -> str:
    """
    Baut den Run-Ordner **am Projekt-Root** (robust ggü. CWD).
    Struktur: <project_root>/<output_root>/<Location>/<Scenario>/<Engine>/<timestamp>_<tag>
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario = _scenario_tag(system_id)

    # Projekt-Root = eine Ebene über ".../Optimization"
    project_root = Path(__file__).resolve().parents[4]   # .../<proj> / Optimization / framework / Orchestrator / paths.py
    base = (project_root / output_root).resolve()        # output_root ist idR "Optimization/run/results"
    d = base / location / scenario / engine_name / f"{ts}_{tag}"
    return str(d)
