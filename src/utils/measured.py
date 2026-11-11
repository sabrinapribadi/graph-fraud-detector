"""Measured model quality (ADR-004): every number shown in the app comes from data/eval_results.json, never typed."""
import json
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "data" / "eval_results.json"
ILLICIT, LICIT, NODES = 4545, 42019, 203769   # Elliptic labels (Weber et al. 2019)


def measured() -> dict:
    e = json.loads(_PATH.read_text())
    rf, gs = e["random_forest_temporal"], e["graphsage_as_built"]
    tn, fp, fn, tp = (v for row in gs["confusion"] for v in row)
    return {"rf_f1": rf["f1"], "rf_precision": rf["precision"], "rf_recall": rf["recall"], "rf_f1_after": rf["f1_after_shutdown"],
            "rf_f1_before": rf["f1_before_shutdown"], "paper_f1": rf["paper"]["f1"], "gs_auc": gs["auc"], "gs_f1": gs["f1"],
            "gs_precision": gs["precision"], "gs_recall": gs["recall"], "gs_accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
            "illicit_share": ILLICIT / (ILLICIT + LICIT)}
