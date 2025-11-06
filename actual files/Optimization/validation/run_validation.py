from Optimization.framework.Settings.settings import validation, paths
from Optimization.validation.core.orchestrator import validate

if __name__ == "__main__":
    print(f"[validation] runs_root = {paths.runs_root}")
    print(f"[validation] run_id    = {validation.run_id}")
    print(f"[validation] teacher   = {validation.teacher_mode}")
    validate()
