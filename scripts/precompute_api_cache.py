"""
Pre-compute API cache during Docker build.

Runs during `docker build` (build env has 8GB RAM).
Saves model weights + pre-computed stats so the runtime
container (512MB RAM) never needs to load the full dataset.

Outputs:
  data/api_cache.json       - stats, topology, insights (~50 KB)
  data/api_model_weights.pt - trained GraphSAGE weights   (~5 MB)
"""
import sys
import json
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("precompute")

# ── 1. Load data ──────────────────────────────────────────────────────────────
logger.info("Loading data...")
from src.data.loader import EllipticDataLoader
loader = EllipticDataLoader()
features_df, classes_df, edgelist_df = loader.load_data()
loader.preprocess_features()
loader.prepare_labels()
G = loader.build_graph()
logger.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── 2. Train model ────────────────────────────────────────────────────────────
logger.info("Training model (sample_size=2000, epochs=50)...")
from src.models.gnn_model import FraudDetector
detector = FraudDetector(hidden_dim=32, num_layers=2, dropout=0.2)
data = detector.build_graph_data(G, sample_size=2000, balance_classes=True)
detector.train(data, epochs=50)
n_features = int(data["x"].shape[1])
logger.info(f"Model trained. n_features={n_features}, hidden_dim=32")

# ── 3. Save model weights ─────────────────────────────────────────────────────
weights_path = project_root / "data" / "api_model_weights.pt"
torch.save(detector.model.state_dict(), weights_path)
logger.info(f"Weights saved → {weights_path} ({weights_path.stat().st_size / 1e6:.1f} MB)")

# ── 4. Compute stats ──────────────────────────────────────────────────────────
logger.info("Computing stats...")
node_labels = [d.get("label", -1) for _, d in G.nodes(data=True)]
degrees = [d for _, d in G.degree()]
licit   = sum(1 for l in node_labels if l == 0)
illicit = sum(1 for l in node_labels if l == 1)
unknown = sum(1 for l in node_labels if l == -1)
labeled = licit + illicit

graph_stats = {
    "total_nodes": G.number_of_nodes(),
    "total_edges": G.number_of_edges(),
    "licit_count": licit,
    "illicit_count": illicit,
    "unknown_count": unknown,
    "avg_degree": round(float(np.mean(degrees)), 4),
    "max_degree": int(max(degrees)),
    "fraud_rate": f"{illicit / max(1, labeled) * 100:.1f}%",
}

topology_stats = {
    "nodes": G.number_of_nodes(),
    "edges": G.number_of_edges(),
    "density": round(float(nx.density(G)), 8),
    "components": int(nx.number_weakly_connected_components(G)),
    "isolates": int(nx.number_of_isolates(G)),
    "avg_degree": round(float(np.mean(degrees)), 4),
    "max_degree": int(max(degrees)),
    "min_degree": int(min(degrees)),
    "degree_percentiles": {
        "25%": float(np.percentile(degrees, 25)),
        "50%": float(np.percentile(degrees, 50)),
        "75%": float(np.percentile(degrees, 75)),
        "90%": float(np.percentile(degrees, 90)),
        "95%": float(np.percentile(degrees, 95)),
        "99%": float(np.percentile(degrees, 99)),
    },
}

# ── 5. Run auto-discovery ─────────────────────────────────────────────────────
insights_json = []
try:
    from src.analytics.auto_discovery import AutoDiscovery
    discoverer = AutoDiscovery(G=G, detector=detector, data=data)
    insights = discoverer.run_full_discovery()
    insights_json = [
        {
            "title": i.title,
            "description": i.description,
            "category": i.category,
            "severity": i.severity,
        }
        for i in insights
    ]
    logger.info(f"Auto-discovery: {len(insights_json)} insights")
except Exception as exc:
    logger.warning(f"Auto-discovery skipped: {exc}")

# ── 6. Save cache JSON ────────────────────────────────────────────────────────
cache = {
    "model_config": {
        "n_features": n_features,
        "hidden_dim": 32,
        "num_layers": 2,
        "dropout": 0.2,
    },
    "stats": graph_stats,
    "topology": topology_stats,
    "insights": insights_json,
}

cache_path = project_root / "data" / "api_cache.json"
with open(cache_path, "w") as f:
    json.dump(cache, f, indent=2)
logger.info(f"Cache saved → {cache_path} ({cache_path.stat().st_size / 1e3:.1f} KB)")
logger.info("Pre-compute complete.")
