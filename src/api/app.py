"""
FastAPI Application for Fraud Detection

Runtime memory budget: 512 MB (Render Starter plan).

Strategy: all heavy computation (data loading, GNN training) happens during
`docker build` via scripts/precompute_api_cache.py.  At runtime the app loads:
  - data/api_cache.json       (~50 KB)  — pre-computed stats + insights
  - data/api_model_weights.pt (~5 MB)   — trained GraphSAGE weights

No parquet files, no NetworkX graph, no training at runtime.
"""
import sys
import json
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from src.models.gnn_model import GraphSAGE
from src.analytics.risk_analysis import QuantitativeRiskAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Graph Fraud Detection API",
    description="API for fraud detection in Bitcoin transaction networks",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state (loaded once at startup, tiny footprint) ─────────────────────
_CACHE: Dict[str, Any] = {}
_MODEL: Optional[GraphSAGE] = None
_LOADED: bool = False

_DATA_DIR = project_root / "data"
_CACHE_PATH = _DATA_DIR / "api_cache.json"
_WEIGHTS_PATH = _DATA_DIR / "api_model_weights.pt"

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    node_id: str = Field(..., description="Transaction node ID")
    features: List[float] = Field(..., description="166 node features")

class PredictionResponse(BaseModel):
    node_id: str
    fraud_probability: float
    risk_level: str
    prediction: str

class RiskRequest(BaseModel):
    n_transactions: int  = Field(10000, description="Number of transactions")
    fraud_rate: float    = Field(0.02,  description="Base fraud rate")
    avg_loss: float      = Field(5000,  description="Average loss per fraud ($)")
    detection_days: int  = Field(30,    description="Detection delay in days")

# ── Startup loader ────────────────────────────────────────────────────────────

def _load():
    global _CACHE, _MODEL, _LOADED
    if _LOADED:
        return

    # Load pre-computed stats / insights
    if not _CACHE_PATH.exists():
        raise RuntimeError(
            f"{_CACHE_PATH} not found. "
            "Run scripts/precompute_api_cache.py during Docker build."
        )
    with open(_CACHE_PATH) as f:
        _CACHE = json.load(f)
    logger.info("Loaded api_cache.json")

    # Re-create model architecture and load trained weights
    cfg = _CACHE["model_config"]
    model = GraphSAGE(
        in_features=cfg["n_features"],
        hidden_dim=cfg["hidden_dim"],
        out_features=1,
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    if not _WEIGHTS_PATH.exists():
        raise RuntimeError(
            f"{_WEIGHTS_PATH} not found. "
            "Run scripts/precompute_api_cache.py during Docker build."
        )
    model.load_state_dict(
        torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    _MODEL = model
    _LOADED = True
    logger.info(
        f"Model loaded from weights. "
        f"n_features={cfg['n_features']}, hidden_dim={cfg['hidden_dim']}"
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "Graph Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health")
async def health_check():
    try:
        _load()
        stats = _CACHE.get("stats", {})
        return {
            "status": "healthy",
            "model_loaded": _LOADED,
            "total_nodes": stats.get("total_nodes"),
            "total_edges": stats.get("total_edges"),
            "version": "1.0.0",
        }
    except Exception as exc:
        return {"status": "unhealthy", "error": str(exc), "model_loaded": False}

@app.get("/stats")
async def get_stats():
    _load()
    return _CACHE.get("stats", {})

@app.post("/predict", response_model=PredictionResponse)
async def predict_transaction(request: TransactionRequest):
    _load()
    try:
        features = torch.FloatTensor(request.features).unsqueeze(0)
        adj = torch.eye(1)
        with torch.no_grad():
            prob = float(torch.sigmoid(_MODEL(features, adj)).squeeze())

        if prob > 0.8:
            risk_level, prediction = "HIGH",   "FRAUD"
        elif prob > 0.5:
            risk_level, prediction = "MEDIUM", "SUSPICIOUS"
        else:
            risk_level, prediction = "LOW",    "LEGITIMATE"

        return PredictionResponse(
            node_id=request.node_id,
            fraud_probability=prob,
            risk_level=risk_level,
            prediction=prediction,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/analyze/risk")
async def analyze_risk(request: RiskRequest):
    try:
        analyzer = QuantitativeRiskAnalyzer(discount_rate=0.10)
        results = analyzer.full_risk_assessment(
            n_transactions=request.n_transactions,
            fraud_probability=request.fraud_rate,
            avg_loss_per_fraud=request.avg_loss,
            exposure_per_transaction=request.avg_loss * 0.2,
            detection_time=request.detection_days,
            n_simulations=10_000,
        )
        return {
            "status": "success",
            "params": {
                "n_transactions": request.n_transactions,
                "fraud_rate": request.fraud_rate,
                "avg_loss": request.avg_loss,
                "detection_days": request.detection_days,
            },
            "results": {
                "expected_loss":     results["expected_loss"]["expected_loss"],
                "value_at_risk_95":  results["monte_carlo"]["value_at_risk"],
                "cost_of_delay":     results["tvm_adjusted"]["time_value_cost"],
                "total_risk_score":  results["total_risk_score"],
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/network/stats")
async def get_network_stats():
    _load()
    return _CACHE.get("topology", {})

@app.get("/discover/insights")
async def discover_insights():
    _load()
    insights = _CACHE.get("insights", [])
    return {
        "status": "success",
        "count": len(insights),
        "insights": insights,
    }

# ── Startup event ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    try:
        _load()
        logger.info("API ready.")
    except Exception as exc:
        logger.error(f"Startup error: {exc}")
