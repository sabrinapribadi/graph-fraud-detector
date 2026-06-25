"""
FastAPI Application for Fraud Detection
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
import torch
import networkx as nx
import pandas as pd
from datetime import datetime
import logging

# Local imports
from src.data.loader import EllipticDataLoader
from src.models.gnn_model import FraudDetector
from src.analytics.risk_analysis import QuantitativeRiskAnalyzer
from src.analytics.auto_discovery import AutoDiscovery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Graph Fraud Detection API",
    description="API for fraud detection in Bitcoin transaction networks",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
GLOBAL_G = None
GLOBAL_DETECTOR = None
GLOBAL_DATA = None
GLOBAL_LOADED = False

# ============================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================

class TransactionRequest(BaseModel):
    """Request for single transaction prediction"""
    node_id: str = Field(..., description="Transaction node ID")
    features: List[float] = Field(..., description="Node features")

class PredictionResponse(BaseModel):
    """Response for prediction"""
    node_id: str
    fraud_probability: float
    risk_level: str
    prediction: str

class RiskRequest(BaseModel):
    """Request for risk analysis"""
    n_transactions: int = Field(10000, description="Number of transactions")
    fraud_rate: float = Field(0.02, description="Base fraud rate")
    avg_loss: float = Field(5000, description="Average loss per fraud")
    detection_days: int = Field(30, description="Detection delay in days")

# ============================================================
# Helper Functions
# ============================================================

def load_models():
    """Load models once at startup"""
    global GLOBAL_G, GLOBAL_DETECTOR, GLOBAL_DATA, GLOBAL_LOADED
    
    if GLOBAL_LOADED:
        return
    
    logger.info("Loading models...")
    
    try:
        # Load data
        loader = EllipticDataLoader()
        features, classes, edgelist = loader.load_data()
        loader.preprocess_features()
        loader.prepare_labels()
        G = loader.build_graph()
        
        # Train detector
        detector = FraudDetector(hidden_dim=32, num_layers=2, dropout=0.2)
        data = detector.build_graph_data(G, sample_size=2000, balance_classes=True)
        detector.train(data, epochs=50)
        
        GLOBAL_G = G
        GLOBAL_DETECTOR = detector
        GLOBAL_DATA = data
        GLOBAL_LOADED = True
        
        logger.info("Models loaded successfully!")
        logger.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

# ============================================================
# API Endpoints
# ============================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Graph Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        load_models()
        return {
            "status": "healthy",
            "model_loaded": GLOBAL_LOADED,
            "total_nodes": GLOBAL_G.number_of_nodes() if GLOBAL_G else None,
            "total_edges": GLOBAL_G.number_of_edges() if GLOBAL_G else None,
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "model_loaded": False,
            "version": "1.0.0"
        }

@app.get("/stats")
async def get_stats():
    """Get overall statistics"""
    load_models()
    
    G = GLOBAL_G
    labels = [data.get('label', -1) for _, data in G.nodes(data=True)]
    
    stats = {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'licit_count': sum(1 for l in labels if l == 0),
        'illicit_count': sum(1 for l in labels if l == 1),
        'unknown_count': sum(1 for l in labels if l == -1),
        'avg_degree': float(np.mean([d for n, d in G.degree()])),
        'max_degree': max([d for n, d in G.degree()]) if G.nodes() else 0,
        'fraud_rate': f"{(sum(1 for l in labels if l == 1) / max(1, sum(1 for l in labels if l in [0, 1])) * 100):.1f}%"
    }
    
    return stats

@app.post("/predict", response_model=PredictionResponse)
async def predict_transaction(request: TransactionRequest):
    """Predict fraud probability for a single transaction"""
    load_models()
    
    try:
        detector = GLOBAL_DETECTOR
        features = np.array(request.features)
        
        # Get prediction
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(detector.device)
        detector.model.eval()
        
        with torch.no_grad():
            output = detector.model(features_tensor, torch.eye(1).to(detector.device))
            prob = torch.sigmoid(output).squeeze().item()
        
        # Determine risk level
        if prob > 0.8:
            risk_level = "HIGH"
            prediction = "FRAUD"
        elif prob > 0.5:
            risk_level = "MEDIUM"
            prediction = "SUSPICIOUS"
        else:
            risk_level = "LOW"
            prediction = "LEGITIMATE"
        
        return PredictionResponse(
            node_id=request.node_id,
            fraud_probability=prob,
            risk_level=risk_level,
            prediction=prediction
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/risk")
async def analyze_risk(request: RiskRequest):
    """Run quantitative risk analysis"""
    try:
        analyzer = QuantitativeRiskAnalyzer(discount_rate=0.10)
        
        results = analyzer.full_risk_assessment(
            n_transactions=request.n_transactions,
            fraud_probability=request.fraud_rate,
            avg_loss_per_fraud=request.avg_loss,
            exposure_per_transaction=request.avg_loss * 0.2,
            detection_time=request.detection_days,
            n_simulations=10000
        )
        
        return {
            "status": "success",
            "params": {
                "n_transactions": request.n_transactions,
                "fraud_rate": request.fraud_rate,
                "avg_loss": request.avg_loss,
                "detection_days": request.detection_days
            },
            "results": {
                "expected_loss": results['expected_loss']['expected_loss'],
                "value_at_risk_95": results['monte_carlo']['value_at_risk'],
                "cost_of_delay": results['tvm_adjusted']['time_value_cost'],
                "total_risk_score": results['total_risk_score']
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/network/stats")
async def get_network_stats():
    """Get detailed network statistics"""
    load_models()
    
    G = GLOBAL_G
    
    degrees = [d for n, d in G.degree()]
    
    stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': float(nx.density(G)),
        'components': int(nx.number_weakly_connected_components(G)),
        'isolates': int(nx.number_of_isolates(G)),
        'avg_degree': float(np.mean(degrees)) if degrees else 0,
        'max_degree': int(max(degrees)) if degrees else 0,
        'min_degree': int(min(degrees)) if degrees else 0,
        'degree_percentiles': {
            '25%': float(np.percentile(degrees, 25)) if degrees else 0,
            '50%': float(np.percentile(degrees, 50)) if degrees else 0,
            '75%': float(np.percentile(degrees, 75)) if degrees else 0,
            '90%': float(np.percentile(degrees, 90)) if degrees else 0,
            '95%': float(np.percentile(degrees, 95)) if degrees else 0,
            '99%': float(np.percentile(degrees, 99)) if degrees else 0
        }
    }
    
    return stats

@app.get("/discover/insights")
async def discover_insights():
    """Run auto-discovery to find fraud patterns"""
    load_models()
    
    try:
        discoverer = AutoDiscovery(
            G=GLOBAL_G,
            detector=GLOBAL_DETECTOR,
            data=GLOBAL_DATA
        )
        
        insights = discoverer.run_full_discovery()
        
        # Convert insights to JSON-serializable format
        insights_json = []
        for insight in insights:
            insights_json.append({
                'title': insight.title,
                'description': insight.description,
                'category': insight.category,
                'severity': insight.severity,
                'data': insight.data
            })
        
        return {
            "status": "success",
            "count": len(insights_json),
            "insights": insights_json
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/network/stats")
async def get_network_stats():
    """Get detailed network statistics"""
    load_models()
    
    G = GLOBAL_G
    
    degrees = [d for n, d in G.degree()]
    
    stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': float(nx.density(G)),
        'components': int(nx.number_weakly_connected_components(G)),
        'isolates': int(nx.number_of_isolates(G)),
        'avg_degree': float(np.mean(degrees)) if degrees else 0,
        'max_degree': int(max(degrees)) if degrees else 0,
        'min_degree': int(min(degrees)) if degrees else 0,
        'degree_percentiles': {
            '25%': float(np.percentile(degrees, 25)) if degrees else 0,
            '50%': float(np.percentile(degrees, 50)) if degrees else 0,
            '75%': float(np.percentile(degrees, 75)) if degrees else 0,
            '90%': float(np.percentile(degrees, 90)) if degrees else 0,
            '95%': float(np.percentile(degrees, 95)) if degrees else 0,
            '99%': float(np.percentile(degrees, 99)) if degrees else 0
        }
    }
    
    return stats

# ============================================================
# Startup Event
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        load_models()
        logger.info("API is ready!")
    except Exception as e:
        logger.error(f"Startup error: {e}")
