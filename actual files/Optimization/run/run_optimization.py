# Optimization/framework/run/run_optimization.py
from Optimization.framework.Settings.settings import get_settings
from Optimization.framework.Orchestrator.optimize import run

if __name__ == "__main__":
    s = get_settings()       # EINZIGE Quelle der Wahrheit
    result = run(s)
    print("Run fertig. Ergebnisse:", result.get("run_dir"))
