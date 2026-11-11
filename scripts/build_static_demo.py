"""Static data for the GitHub Pages demo (ADR-004): no live API host, no server cost.

    python scripts/build_static_demo.py      # after scripts/evaluate_models.py

Writes docs/data/demo.json with: graph stats and topology (fixed labels), auto-discovery insights, 12 real example
transactions from the held-out time steps 35-49 scored by the validated random forest (6 illicit, 6 licit, seed 42),
and the ADR-004 evaluation results.
"""
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def examples() -> list:
    f = pd.read_parquet(ROOT / "data/processed/features.parquet")
    c = pd.read_parquet(ROOT / "data/processed/classes.parquet")
    f.columns = ["txId", "time_step"] + [f"f{i}" for i in range(f.shape[1] - 2)]
    f["txId"] = f.txId.astype(str); c["txId"] = c.txId.astype(str)
    d = f.merge(c, on="txId"); d = d[d["class"].isin(["1", "2"])].copy(); d["y"] = (d["class"] == "1").astype(int)
    X = d.drop(columns=["txId", "class", "y"])
    tr, te = d.time_step <= 34, (d.time_step >= 35) & (d.time_step <= 42)    # examples before the dark-market shutdown
    rf = RandomForestClassifier(n_estimators=50, max_features=50, random_state=42, n_jobs=4).fit(X[tr], d.y[tr])
    pool = d[te]
    pick = pd.concat([pool[pool.y == 1].sample(6, random_state=42), pool[pool.y == 0].sample(6, random_state=42)])
    prob = rf.predict_proba(pick.drop(columns=["txId", "class", "y"]))[:, 1]
    risk = lambda p: "HIGH" if p >= 0.7 else "MEDIUM" if p >= 0.4 else "LOW"
    return [{"tx_id": t, "time_step": int(s), "true_label": "illicit" if yy else "licit", "fraud_probability": round(float(p), 3),
             "risk_level": risk(p), "prediction": "illicit" if p >= 0.5 else "licit"} for t, s, yy, p in zip(pick.txId, pick.time_step, pick.y, prob)]


def graph_and_insights() -> tuple:
    from src.data.loader import EllipticDataLoader
    from src.models.gnn_model import FraudDetector
    loader = EllipticDataLoader(); loader.load_data(); loader.preprocess_features(); loader.prepare_labels()
    G = loader.build_graph()
    labels = [d.get("label", -1) for _, d in G.nodes(data=True)]; deg = [x for _, x in G.degree()]
    illicit, licit = labels.count(1), labels.count(0)
    stats = {"total_nodes": G.number_of_nodes(), "total_edges": G.number_of_edges(), "illicit_count": illicit, "licit_count": licit,
             "unknown_count": labels.count(-1), "fraud_rate": f"{100 * illicit / (illicit + licit):.1f}%",
             "avg_degree": round(float(np.mean(deg)), 3), "components": int(nx.number_weakly_connected_components(G))}
    det = FraudDetector(hidden_dim=32, num_layers=2, dropout=0.2)
    data = det.build_graph_data(G, sample_size=2000, balance_classes=True); det.train(data, epochs=50)
    insights = []
    try:
        from src.analytics.auto_discovery import AutoDiscovery
        insights = [{"title": i.title, "description": i.description, "category": i.category, "severity": i.severity}
                    for i in AutoDiscovery(G=G, detector=det, data=data).run_full_discovery()
                    # Elliptic features are anonymised and normalised, not amounts: a "values below thresholds" (structuring)
                    # insight cannot be read as money, so it is not published.
                    if "structuring" not in i.title.lower()]
    except Exception as e:  # noqa: BLE001
        print("auto-discovery skipped:", e)
    return stats, insights


def main() -> None:
    stats, insights = graph_and_insights()
    out = {"stats": stats, "insights": insights, "examples": examples(),
           "evaluation": json.loads((ROOT / "data/eval_results.json").read_text())}
    (ROOT / "docs/data").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs/data/demo.json").write_text(json.dumps(out, indent=1))
    print(stats, len(insights), "insights;", [(e["true_label"], e["fraud_probability"]) for e in out["examples"]])


if __name__ == "__main__":
    main()
