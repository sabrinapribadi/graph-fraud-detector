PRODUCT REQUIREMENT DOCUMENT (PRD)
Project: Graph Fraud Detector – GNN-Powered Financial Fraud Detection with Quantitative Risk Analysis
Version: 2.0
Author: Sabrina Pribadi
Date: June 25, 2026
Status: Production Ready (v1.0 Deployed)

## 1. EXECUTIVE SUMMARY

**Problem:** Financial fraud costs the global economy over $5 trillion annually. Traditional fraud detection systems rely on rule-based heuristics or isolated transaction analysis, missing the complex network patterns that characterize sophisticated fraud rings. Analysts struggle to identify hidden connections, quantify fraud risk, and prioritize investigations without specialized graph analysis tools.

**Solution:** A GNN-powered agentic system that detects fraudulent transactions by analyzing the Bitcoin transaction graph, automatically identifies anomalous patterns, quantifies fraud risk using Monte Carlo simulations and TVM-adjusted metrics, and validates its predictions through rigorous testing. The system is deployed as a production-ready web application with both a user-friendly dashboard and a REST API.

**Value Proposition:** Reduce false positives by 30-50% compared to traditional methods. Enable non-technical stakeholders to query fraud patterns in natural language. Demonstrate end-to-end ML engineering with GNNs, quantitative finance, and agentic AI.

**Success Metrics:**
- ✅ GNN model achieves **0.955 AUC** on illicit transaction detection
- ✅ Monte Carlo simulations quantify fraud risk with confidence intervals
- ✅ Agent correctly answers 6+ test questions about fraud patterns
- ✅ Dashboard loads in < 3 seconds
- ✅ Hallucination test suite passes 6/6 questions against ground truth

## 2. PROJECT CONTEXT & BACKGROUND

This project demonstrates capabilities in:
1. **Graph Neural Networks**: PyTorch-based GNN (GraphSAGE) with 95.5% AUC
2. **Quantitative Finance**: Monte Carlo simulations, TVM-adjusted risk scoring, Expected Loss calculations, VaR
3. **LLM & Agentic Design**: LangChain agent with 6 tools, fallback strategies
4. **Data Engineering**: Processing 200K+ transaction nodes with 166 features each
5. **MLOps**: Production pipeline, Docker containerization, Render deployment
6. **AI Safety**: Automated hallucination detection against Parquet ground truth
7. **Full-Stack Development**: Streamlit dashboard + FastAPI REST API

**Data Source:** Elliptic Bitcoin Dataset (https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

## 3. SCOPE

**In-Scope:**
- **Data**: Elliptic Bitcoin transaction dataset (203,769 nodes, 234,355 edges)
- **Modelling**: Graph Neural Networks (GraphSAGE) for node classification
- **Quantitative Finance**: Monte Carlo simulations, TVM-adjusted risk scoring, Expected Loss calculations, VaR
- **LLM Agent**: Natural language Q&A about fraud patterns, 6 tools
- **Auto-Discovery**: Proactively surface 5 types of fraud patterns
- **Temporal Analysis**: Transaction velocity, anomaly detection, fraud trends
- **Advanced ML**: Ensemble models, hyperparameter optimization, explainability
- **UI**: Streamlit dashboard with 8 tabs, dark theme, Plotly charts
- **API**: FastAPI REST API with interactive documentation
- **Deployment**: Docker containerization, Render deployment
- **Testing**: Automated hallucination detection suite against ground truth

## 4. USER PERSONAS & STORIES

| Persona | Goal | Implemented Feature |
|---------|------|---------------------|
| Alex (Analyst) | Deep dives on fraud patterns | Network Explorer, Auto-Discovery, Temporal Analysis |
| Maria (Risk Manager) | Quantify fraud risk exposure | Risk Analysis with Monte Carlo simulations |
| James (Compliance Officer) | Demonstrate regulatory compliance | API for integration, Report generation |
| Dana (Data Scientist) | Train and improve fraud models | Advanced ML: ensemble models, hyperparameter optimization |
| API User | Programmatic access | REST API with Swagger documentation |

## 5. FUNCTIONAL REQUIREMENTS

### Module A: Data Pipeline
- **A.1 Data Ingestion**: Load Elliptic dataset (features, classes, edgelist) ✅
- **A.2 Data Preprocessing**: Handle missing values, normalize features ✅
- **A.3 Graph Construction**: Build directed graph with 203,769 nodes, 234,355 edges ✅
- **A.4 Train/Test Split**: 80/20 split on labelled nodes ✅
- **A.5 Data Export**: Processed data in memory for fast access ✅

### Module B: GNN Model
- **B.1 Architecture**: GraphSAGE with 2 layers, 32 hidden dimensions ✅
- **B.2 Training**: Binary classification (licit vs illicit) ✅
- **B.3 Evaluation**: AUC: 0.955, F1: 0.898, Accuracy: 89.0% ✅
- **B.4 Explainability**: Feature importance, node-level explanations ✅

### Module C: Quantitative Risk Analysis
- **C.1 Monte Carlo Simulation**: 10,000 iterations with confidence intervals ✅
- **C.2 TVM-Adjusted Risk Scoring**: Discount fraud costs over time ✅
- **C.3 Expected Loss Calculation**: `EL = PD × EAD × LGD` ✅
- **C.4 Value at Risk (VaR)**: 95% confidence level ✅
- **C.5 Risk Heatmap**: Visualize risk by transaction characteristics ✅

### Module D: LLM Agent (FraudAgent)
- **D.1 Agent Framework**: LangChain with fallback mode ✅
- **D.2 Tools**: 
  - `get_fraud_stats`: Overall fraud statistics ✅
  - `find_suspicious_nodes`: Top suspicious transactions ✅
  - `analyze_network`: Network structure analysis ✅
  - `predict_transaction`: Single transaction prediction ✅
  - `run_risk_analysis`: Monte Carlo risk simulation ✅
  - `get_anomalous_patterns`: Pattern discovery ✅

### Module E: Auto-Discovery
- **E.1 Money Laundering Rings**: Detect high-degree illicit hubs ✅
- **E.2 Structuring Patterns**: Transactions just below thresholds ✅
- **E.3 Rapid Transaction Chains**: High velocity patterns ✅
- **E.4 Mixed Signals**: Nodes with both licit and illicit connections ✅
- **E.5 Anomaly Outliers**: Statistical outliers in degree distribution ✅

### Module F: Temporal Analysis
- **F.1 Transaction Velocity**: Analyze high-velocity nodes ✅
- **F.2 Time Patterns**: Activity distribution patterns ✅
- **F.3 Fraud Trends**: Class distribution and degree comparison ✅
- **F.4 Temporal Anomalies**: Z-score based anomaly detection ✅

### Module G: Advanced ML
- **G.1 Ensemble Model**: GAT + GraphSAGE (mean) + GraphSAGE (sum) ✅
- **G.2 Hyperparameter Optimization**: Optuna for parameter tuning ✅
- **G.3 Model Explainability**: Feature importance visualization ✅

### Module H: Web Interface
- **H.1 Framework**: Streamlit with dark theme ✅
- **H.2 Pages**: 8 tabs (Overview, Network Explorer, AI Assistant, Risk Analysis, Data Explorer, Auto-Discovery, Advanced ML, Temporal Analysis) ✅
- **H.3 Charts**: Plotly with transparent dark backgrounds ✅
- **H.4 Network Visualization**: Interactive graph with Plotly ✅
- **H.5 Data Explorer**: Filterable transaction table with export ✅

### Module I: API & Deployment
- **I.1 FastAPI Application**: REST API with 8+ endpoints ✅
- **I.2 API Documentation**: Interactive Swagger UI ✅
- **I.3 Docker Containerization**: Dockerfile for reproducible builds ✅
- **I.4 Deployment**: Render deployment with free tier ✅
- **I.5 Health Checks**: `/health` endpoint for monitoring ✅

### Module J: Hallucination Detection
- **J.1 Ground Truth**: Load from Parquet file ✅
- **J.2 Error Detection**: Flag deviations >10% from ground truth ✅
- **J.3 Test Runner**: 6 standard questions, PASS/FAIL per question ✅

## 6. NON-FUNCTIONAL REQUIREMENTS

- **Performance**: Dashboard loads < 3 seconds ✅
- **Cost**: GPT-4o-mini throughout, caching in session state ✅
- **Resilience**: LLM fallbacks (hardcoded narratives, fuzzy matching) ✅
- **Code Quality**: Modular structure, centralised helpers ✅
- **Theme**: Consistent dark theme via config.toml ✅

## 7. TECHNICAL ARCHITECTURE
