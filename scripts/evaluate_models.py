"""ADR-004 evaluation with the fixed labels.

    python scripts/evaluate_models.py

1. Pipeline check against Weber et al. (2019): random forest (50 trees, max_features=50, seed 42) on all 166 features,
   trained on labelled time steps 1-34, tested on 35-49; illicit precision, recall, F1, and F1 per test step.
   Passes if illicit F1 is within 0.788 +/- 0.05 (the paper's random forest, all features).
2. The dashboard's GraphSAGE as built (src/ui/dashboard.py): reported with the caveat that its balanced sample and
   random split are not comparable with the temporal benchmark.
Writes data/eval_results.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PAPER_F1, TOL = 0.788, 0.05


def rf_temporal() -> dict:
    f = pd.read_parquet(ROOT / "data/processed/features.parquet")
    c = pd.read_parquet(ROOT / "data/processed/classes.parquet")
    f.columns = ["txId", "time_step"] + [f"f{i}" for i in range(f.shape[1] - 2)]
    c["txId"] = c.txId.astype(str); f["txId"] = f.txId.astype(str)
    d = f.merge(c, on="txId")
    d = d[d["class"].isin(["1", "2"])].copy()
    d["y"] = (d["class"] == "1").astype(int)               # class "1" = illicit (Weber et al. 2019)
    X = d.drop(columns=["txId", "class", "y"])              # time step + 165 features = 166 ("AF")
    tr, te = d.time_step <= 34, d.time_step >= 35
    rf = RandomForestClassifier(n_estimators=50, max_features=50, random_state=42, n_jobs=4).fit(X[tr], d.y[tr])
    p = rf.predict(X[te]); s = rf.predict_proba(X[te])[:, 1]; y = d.y[te].to_numpy()
    steps = {int(t): float(f1_score(y[(d.time_step[te] == t).to_numpy()], p[(d.time_step[te] == t).to_numpy()], zero_division=0))
             for t in sorted(d.time_step[te].unique())}
    res = {"train_rows": int(tr.sum()), "test_rows": int(te.sum()), "test_illicit": int(y.sum()),
           "precision": float(precision_score(y, p)), "recall": float(recall_score(y, p)), "f1": float(f1_score(y, p)),
           "auc": float(roc_auc_score(y, s)), "f1_by_step": steps,
           "f1_before_shutdown": float(f1_score(y[(d.time_step[te] <= 42).to_numpy()], p[(d.time_step[te] <= 42).to_numpy()])),
           "f1_after_shutdown": float(f1_score(y[(d.time_step[te] >= 43).to_numpy()], p[(d.time_step[te] >= 43).to_numpy()], zero_division=0)),
           "paper": {"precision": 0.956, "recall": 0.670, "f1": PAPER_F1}}
    res["check_passed"] = bool(abs(res["f1"] - PAPER_F1) <= TOL)
    # threshold what-if for the results page: flag a transaction when its illicit score is above the threshold (as predict does at 0.5)
    res["threshold_curve"] = []
    for t in [round(0.05 * k, 2) for k in range(1, 20)]:
        f = s > t; tp = int((f & (y == 1)).sum()); fp = int((f & (y == 0)).sum()); fn = int((~f & (y == 1)).sum())
        res["threshold_curve"].append({"threshold": t, "flagged": int(f.sum()), "caught": tp, "missed": fn, "false_alarms": fp,
                                       "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)})
    return res


def graphsage_as_built() -> dict:
    import torch
    from sklearn.metrics import confusion_matrix

    from src.data.loader import EllipticDataLoader
    from src.models.gnn_model import FraudDetector
    torch.manual_seed(42); np.random.seed(42)
    loader = EllipticDataLoader(); loader.load_data(); loader.preprocess_features(); loader.prepare_labels()
    G = loader.build_graph()
    det = FraudDetector(hidden_dim=64, num_layers=2, dropout=0.3, learning_rate=0.001)   # src/ui/dashboard.py
    data = det.build_graph_data(G, sample_size=3000, balance_classes=True)
    det.train(data, epochs=100, early_stopping=20)
    det.model.eval()
    with torch.no_grad():
        out = torch.sigmoid(det.model(data["x"], data["adj"])[data["test_mask"]]).squeeze().cpu().numpy()
    y = data["y"][data["test_mask"]].cpu().numpy(); p = (out > 0.5).astype(int)
    return {"sample_nodes": int(data["x"].shape[0]), "test_nodes": int(len(y)), "test_illicit_share": float(y.mean()),
            "auc": float(roc_auc_score(y, out)), "precision": float(precision_score(y, p, zero_division=0)),
            "recall": float(recall_score(y, p)), "f1": float(f1_score(y, p)), "confusion": confusion_matrix(y, p).tolist(),
            "caveat": "Balanced 3,000-node sample with a random split: illicit is ~50% of test nodes, not ~10%; flatters the model."}


def main() -> None:
    t0 = time.time()
    res = {"design": "ADR-004", "random_forest_temporal": rf_temporal()}
    print("RF temporal:", {k: round(v, 3) for k, v in res["random_forest_temporal"].items() if isinstance(v, float)}, "check", res["random_forest_temporal"]["check_passed"], flush=True)
    res["graphsage_as_built"] = graphsage_as_built()
    print("GraphSAGE as built:", {k: round(v, 3) for k, v in res["graphsage_as_built"].items() if isinstance(v, float)}, flush=True)
    res["seconds"] = round(time.time() - t0)
    (ROOT / "data" / "eval_results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
