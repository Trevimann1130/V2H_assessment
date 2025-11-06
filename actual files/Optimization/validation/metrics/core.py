from __future__ import annotations
import numpy as np, pandas as pd

def _align(pred: pd.DataFrame, ref: pd.DataFrame):
    pred, ref = pred.reset_index(drop=True), ref.reset_index(drop=True)
    cols = [c for c in ref.columns if c in pred.columns]
    if not cols: raise ValueError("No common columns for metrics alignment")
    return pred[cols], ref[cols]

def _mae_rmse_mape(pred: pd.DataFrame, ref: pd.DataFrame):
    eps=1e-12; out={}
    for c in pred.columns:
        e = pred[c].to_numpy(float) - ref[c].to_numpy(float)
        mae=float(np.mean(np.abs(e))); rmse=float(np.sqrt(np.mean(e*e)))
        denom=np.maximum(np.abs(ref[c].to_numpy(float)), eps)
        out[c]={"mae":mae,"rmse":rmse,"mape_%":float(np.mean(np.abs(e)/denom)*100.0)}
    return out

def _front(Fp: np.ndarray, Fr: np.ndarray):
    if len(Fp)==0 or len(Fr)==0: return {"gd":float("nan"),"igd":float("nan"),"delta_hv":float("nan")}
    dist=lambda a,b: np.linalg.norm(a-b,2)
    gd=float(np.mean([min(dist(p,r) for r in Fr) for p in Fp]))
    igd=float(np.mean([min(dist(r,p) for p in Fp) for r in Fr]))
    res={"gd":gd,"igd":igd,"delta_hv":float("nan")}
    if Fp.shape[1]==2:
        ref=np.maximum(Fp.max(0), Fr.max(0))
        def hv(F):
            xs=np.unique(np.concatenate([F[:,0],[ref[0]]])); xs.sort(); h=0.0
            for i in range(len(xs)-1):
                x0,x1=xs[i],xs[i+1]
                y=np.min(F[F[:,0]<=x0][:,1]) if np.any(F[:,0]<=x0) else ref[1]
                h+=(x1-x0)*(ref[1]-y)
            return h
        res["delta_hv"]=float(hv(Fp)-hv(Fr))
    return res

def compute(s_flows: pd.DataFrame, s_kpis: pd.DataFrame, t_flows: pd.DataFrame|None, t_kpis: pd.DataFrame|None):
    out={}
    if t_flows is not None:
        p,r=_align(s_flows,t_flows); out["flows"]=_mae_rmse_mape(p,r)
    if t_kpis is not None:
        p,r=_align(s_kpis,t_kpis); out["kpis"]=_mae_rmse_mape(p,r)
        if r.shape[1]>=2:
            out["front"]=_front(p.iloc[:,:2].to_numpy(float), r.iloc[:,:2].to_numpy(float))
    return out
