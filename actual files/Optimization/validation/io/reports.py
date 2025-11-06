from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

from ..utils import write_json, write_text

def save_audit(outdir: Path, audit: dict) -> None:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / "audit.json", audit)

def save_probes(outdir: Path, df: pd.DataFrame) -> None:
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "tables" / "probes.csv", index=False)

def save_predictions(outdir: Path, df: pd.DataFrame) -> None:
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "tables" / "predictions.csv", index=False)

def save_metrics(outdir: Path, metrics: dict) -> None:
    write_json(outdir / "metrics.json", metrics)

def save_report_md(outdir: Path, header: dict, metrics: dict) -> None:
    lines = ["# Validation Report", ""]
    for k, v in header.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Metrics")
    lines.append("```json")
    lines.append(json.dumps(metrics, indent=2))
    lines.append("```")
    write_text(outdir / "report.md", "\n".join(lines))

def plot_scatter(outdir: Path, k_ref: pd.DataFrame, k_teacher: pd.DataFrame, label: str) -> None:
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(k_ref.iloc[:,0], k_ref.iloc[:,1], alpha=0.5, label="surrogate")
    ax.scatter(k_teacher.iloc[:,0], k_teacher.iloc[:,1], alpha=0.5, label=label)
    ax.set_xlabel(k_ref.columns[0]); ax.set_ylabel(k_ref.columns[1])
    ax.legend()
    fig.savefig(outdir / "plots" / f"scatter_{label.lower()}.png", dpi=150)
    plt.close(fig)

def plot_front(outdir: Path, k_ref: pd.DataFrame, k_teacher: pd.DataFrame, label: str) -> None:
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(k_ref.iloc[:,0], k_ref.iloc[:,1], ".", alpha=0.6, label="surrogate")
    ax.plot(k_teacher.iloc[:,0], k_teacher.iloc[:,1], ".", alpha=0.6, label=label)
    ax.set_xlabel(k_ref.columns[0]); ax.set_ylabel(k_ref.columns[1])
    ax.legend()
    fig.savefig(outdir / "plots" / f"front_{label.lower()}.png", dpi=150)
    plt.close(fig)
