# Optimization/framework/engines/Surrogat_model/training.py
from __future__ import annotations
import numpy as np
from Optimization.framework.engines.Surrogat_model.samplers.factory import sample_from_settings
from Optimization.framework.engines.Surrogat_model.teacher.evaluate_teacher import evaluate_teacher_dataset
from Optimization.framework.engines.Surrogat_model.split.split import train_holdout_split
from Optimization.framework.engines.Surrogat_model.fit.fit_models import fit_random_forest_per_column
from Optimization.framework.engines.Surrogat_model.validate.holdout import metrics_by_column
from Optimization.framework.engines.Surrogat_model.persist.save import (
    make_outdir, build_meta_dict, persist_artifact, mirror_holdout_to_validation
)

def auto_train_surrogate(settings) -> str:
    # 1) Sampling
    X = sample_from_settings(settings)
    print(f"[surrogate] sample: {settings.sampler.name} n={len(X)}")

    # 2) Teacher (Flows → LEBENSDAUER-Summen)
    YF, YG = evaluate_teacher_dataset(settings, X, targets=settings.surrogate_train.targets, batch_size=None)
    print(f"[surrogate] teacher: X={X.shape} YF={YF.shape}")

    # 3) Split
    X_tr, X_hold, YF_tr, YF_hold, YG_tr, YG_hold = train_holdout_split(
        X, YF, YG,
        holdout_frac=float(settings.surrogate_train.holdout_frac),
        seed=int(settings.sampler.seed),
    )

    # 4) Fit
    models_F = fit_random_forest_per_column(
        X_tr, YF_tr,
        n_estimators=int(settings.surrogate_train.rf_n_estimators),
        n_jobs=int(settings.surrogate_train.rf_n_jobs),
        seed=int(settings.sampler.seed),
    )

    # 5) Holdout
    holdout = {}
    if X_hold.shape[0] and YF_hold.shape[0]:
        YF_pred = np.column_stack([m.predict(X_hold) for m in models_F])
        holdout["F"] = metrics_by_column(YF_hold, YF_pred, target_names=list(settings.surrogate_train.targets))
    print("[surrogate] holdout: ok")

    # 6) Persist + Mirror
    outdir = make_outdir(settings)
    meta = build_meta_dict(settings, holdout_metrics=holdout)
    meta["surrogate_targets"] = list(settings.surrogate_train.targets)
    artifact = persist_artifact(outdir, models_F, [], meta)
    mirror_holdout_to_validation(settings, artifact_dir=outdir, meta=meta)
    return artifact
