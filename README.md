# Graph Fraud Detector

An end-to-end Graph Neural Network system for detecting fraudulent Bitcoin transactions,
featuring quantitative risk analysis, a 14-tab interactive dashboard, a LangChain AI agent,
RAG-powered fraud knowledge search, Phase 6 advanced quant-finance analytics, and a FastAPI
REST service — designed so non-finance users can interpret every result in plain English.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://graph-fraud-detector-dashboards.onrender.com)
[![API](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://graph-fraud-detector-api-service.onrender.com/docs)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)
![alt text](image.png)
## Live Services

| Service | URL | Status |
|---------|-----|--------|
| **Streamlit Dashboard** | [graph-fraud-detector-dashboards.onrender.com](https://graph-fraud-detector-dashboards.onrender.com) | Live |
| **API Documentation** | [graph-fraud-detector-api-service.onrender.com/docs](https://graph-fraud-detector-api-service.onrender.com/docs) | Live |
| **Health Check** | [.../health](https://graph-fraud-detector-api-service.onrender.com/health) | Live |

## Features

### Core Dashboard (Tabs 1–9)

| Dashboard Tab | Business Question | What it does |
|--------------|-------------------|-------------|
| **Overview** | Are we catching the fraudsters? | AUC, fraud rate, dual donut (ground-truth labels + model-predicted classification for all 203k nodes), degree histogram, fraud probability histogram, top suspicious nodes table |
| **Network Explorer** | Who are the most connected suspects? | Interactive Plotly graph; colour by label, degree, fraud probability, or **AI-Predicted Label** (model inference on subgraph with caution callout); network topology stats |
| **AI Assistant** | Can I just ask questions in plain English? | LangChain agent with 6 analytical tools; native `st.chat_message()` bubbles; Plotly bar charts for statistics/suspect results; agent cached across messages; multiple questions via semicolons |
| **Risk Analysis** | How much money could we lose? | Monte Carlo (10,000 scenarios); Expected Loss (PD × EAD × LGD); VaR 95%; Time Value of Money |
| **Data Explorer** | Can I filter and export specific transactions? | Filterable node table with label and degree filters; degree distribution chart; CSV export |
| **Auto-Discovery** | What fraud patterns should I worry about right now? | Proactively detects 5 fraud pattern types with severity badges and Plotly charts |
| **Advanced ML** | Can the model explain why it flagged a transaction? | Optuna hyperparameter tuning + **Apply button to retrain with best params**; GAT + GraphSAGE ensemble; gradient-based attribution as horizontal bar chart (red = raises risk, blue = lowers) |
| **Temporal Analysis** | Is fraud getting worse over time? | Class distribution with Ground Truth / AI-Predicted Labels toggle; velocity percentiles; z-score anomaly table; downloadable report |
| **Knowledge Base** | Can I look up a fraud term or concept? | RAG search over 25 curated documents; ChromaDB + OpenAI with TF-IDF fallback; DuckDuckGo web search fallback when knowledge base has no match |

### Phase 6 — Advanced Quant Finance (Tabs 10–14)

| Dashboard Tab | Business Question | What it does |
|--------------|-------------------|-------------|
| **Stress Testing** | What if there's a financial crisis? | 5 named crisis scenarios (2008, COVID-19, Crypto Winter, etc.); Monte Carlo × 5,000 per scenario; severity ratio vs baseline; loss distribution overlay; KDE heatmap |
| **Risk-Adjusted Metrics** | Is our fraud detector worth the investment? | Sharpe, Sortino, Information Ratio, Calmar — bootstrapped over 49 pseudo-periods; radar scorecard; TPR series vs naive benchmark; drawdown chart |
| **Loss Forecasting** | How much should we budget for fraud? | Prophet (with Holt-Winters fallback) fraud-loss forecast over 49 bi-weekly steps (Jan 2011–Jan 2013); 80% confidence intervals; trend direction; per-period budget estimate |
| **Regulatory Capital** | How much money do we need in reserve? | Basel III Standardised Approach vs IRB Vasicek formula at 99.9% confidence; rho and PD sensitivity curves; SA vs IRB capital saving |
| **Contagion Score** | If we miss one fraudster, how many others are at risk? | Stochastic SIR diffusion; Composite Risk = fraud_prob × (1 + log(1 + mean_at_risk)); ranked scatter; top investigation targets |

### Business Translation Layer

Every tab surfaces a green **Business Question** panel that frames the technical content as
a plain-English question and answer — no finance background required.

Phase 6 tabs additionally show an amber **Plain English** callout after results are computed
(e.g. *"In a 2008-style crisis, your expected fraud losses would be 3.0× your baseline —
from $X to $Y."*). All technical sliders (PD, LGD, rho, diffusion steps) include `help=`
tooltip explanations.

## Model Performance

| Metric | Value |
|--------|-------|
| **AUC** | 0.955 |
| **F1 Score** | 0.898 |
| **Accuracy** | 89.0% |
| **Precision** | 90.6% |
| **Recall** | 89.0% |
| **Total Nodes** | 203,769 |
| **Total Edges** | 234,355 |
| **Illicit Nodes** | 42,019 |
| **Licit Nodes** | 4,545 |

## Project Structure

```
graph-fraud-detector/
├── .streamlit/
│   └── config.toml                  # Dark theme + coral accent
├── src/
│   ├── agent/
│   │   ├── fraud_agent.py           # LangChain Q&A agent (6 tools)
│   │   └── rag_agent.py             # FraudRAGAgent: ChromaDB + OpenAI RAG
│   ├── analytics/
│   │   ├── risk_analysis.py         # Monte Carlo, TVM, Expected Loss
│   │   ├── auto_discovery.py        # 5-method fraud pattern detector
│   │   ├── temporal_analysis.py     # Velocity, z-score anomaly detection
│   │   ├── stress_testing.py        # 5 crisis scenarios × Monte Carlo
│   │   ├── risk_adjusted_metrics.py # Sharpe, Sortino, IR, Calmar
│   │   ├── loss_forecasting.py      # Prophet / Holt-Winters forecast
│   │   ├── regulatory_capital.py    # Basel III SA + IRB Vasicek
│   │   └── contagion.py             # SIR diffusion, Composite Risk Score
│   ├── api/
│   │   └── app.py                   # FastAPI REST service (8 endpoints)
│   ├── data/
│   │   └── loader.py                # EllipticDataLoader: parquet-first load
│   ├── models/
│   │   ├── gnn_model.py             # GraphSAGE implementation
│   │   ├── advanced_gnn.py          # GAT + Ensemble model
│   │   └── hyperparameter_optimization.py  # Optuna tuner
│   └── ui/
│       └── dashboard.py             # Streamlit 14-tab dashboard
├── scripts/
│   ├── preprocess_data.py           # Run once: CSVs → zstd parquet (690 MB → 87 MB)
│   ├── build_fraud_index.py         # Run once: build ChromaDB RAG index
│   ├── test_agent.py                # Agent integration test
│   ├── test_env.py                  # Environment variable check
│   └── validate_data.py             # Dataset validation report
├── data/
│   ├── processed/                   # Committed parquet files (~87 MB, no LFS needed)
│   │   ├── features.parquet         # 203,769 nodes × 166 features  (84.5 MB, zstd)
│   │   ├── classes.parquet          # Ground-truth labels            (1.0 MB)
│   │   └── edgelist.parquet         # 234,355 directed edges         (2.1 MB)
│   └── raw/                         # Not committed — place CSVs here only if re-generating parquet
├── tests/
├── Dockerfile                       # Dashboard container (port 8501)
├── Dockerfile.api                   # API container (port 8000)
├── render.yaml                      # Render deployment config
├── pyproject.toml                   # Poetry dependencies
└── PRD.md                           # Full product requirements document
```

## Setup

**Prerequisites:** Python 3.12, [Poetry](https://python-poetry.org/), optional OpenAI API key.

```bash
# Clone the repository
git clone https://github.com/sabrinapribadi/graph-fraud-detector.git
cd graph-fraud-detector

# Install dependencies (includes Prophet and DuckDuckGo search)
poetry install

# Copy and configure environment variables
cp .env.example .env
```

> **Optional:** Prophet improves forecast accuracy. It is included in `pyproject.toml` but may
> require additional system dependencies on some platforms:
> ```bash
> pip install prophet          # standalone pip install if poetry install fails
> ```

Edit `.env`:

```
OPENAI_API_KEY=sk-...        # Required for LLM agent and RAG synthesis
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0
OPENAI_MAX_TOKENS=2000
```

## Data Setup

The processed parquet files (`data/processed/`) are committed to the repository — no extra
download step is needed for most users. The app loads from parquet automatically on startup.

**Only needed if regenerating from scratch** (e.g. after downloading a dataset update):

```bash
# 1. Download from Kaggle and place CSVs here
mkdir -p data/raw/elliptic_bitcoin_dataset
# copy elliptic_txs_features.csv, elliptic_txs_classes.csv, elliptic_txs_edgelist.csv

# 2. Regenerate parquet (690 MB CSV → 87 MB parquet, zstd)
poetry run python scripts/preprocess_data.py
```

The loader auto-detects parquet on first call and falls back to raw CSVs if parquet is absent,
saving a fresh cache automatically.

## Running the App

```bash
# Dashboard
PYTHONPATH=. poetry run streamlit run src/ui/dashboard.py

# API server (separate terminal)
PYTHONPATH=. poetry run uvicorn src.api.app:app --reload
```

Dashboard opens at `http://localhost:8501` · API at `http://localhost:8000`.

## Building the RAG Index

The Knowledge Base tab uses ChromaDB vector search. Build the index once before using it:

```bash
python scripts/build_fraud_index.py          # build (skips if already exists)
python scripts/build_fraud_index.py --rebuild # force full rebuild
```

Cost: approximately $0.01 one-time (25 documents × ~150 tokens × text-embedding-3-small pricing).
The index is saved to `data/chroma_fraud/`. If ChromaDB is not installed, the app automatically
falls back to TF-IDF keyword search.

## Docker Setup

```bash
# Dashboard
docker build -t fraud-detector-dashboard .
docker run -p 8501:8501 fraud-detector-dashboard

# API
docker build -f Dockerfile.api -t fraud-detector-api .
docker run -p 8000:8000 fraud-detector-api
```

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check with model status |
| `/stats` | GET | Dataset and model statistics |
| `/predict` | POST | Single transaction fraud probability |
| `/network/stats` | GET | Network topology metrics |
| `/analyze/risk` | POST | Monte Carlo risk assessment |
| `/discover/insights` | GET | Auto-discovery fraud pattern results |
| `/docs` | GET | Interactive Swagger documentation |

Example:

```python
import requests

response = requests.post(
    "https://graph-fraud-detector-api-service.onrender.com/predict",
    json={"node_id": "5530458", "features": [0.0] * 166}
)
print(response.json())
```

## Tech Stack

| Layer | Library / Tool |
|-------|---------------|
| Frontend | Streamlit 1.58, Material Icons |
| Charts | Plotly Express / Graph Objects |
| Graph ML | PyTorch, GraphSAGE, GAT, torch-geometric |
| Optimisation | Optuna (TPE sampler, MedianPruner) |
| AI Agent | LangChain, LangGraph, OpenAI GPT-4o-mini |
| RAG | ChromaDB, OpenAI text-embedding-3-small (256-dim), TF-IDF fallback |
| API | FastAPI, Uvicorn |
| Quant Finance | Monte Carlo, TVM, VaR, Sharpe/Sortino/IR/Calmar, Basel III SA + IRB Vasicek, SIR diffusion |
| Forecasting | Prophet (optional), Holt-Winters double exponential smoothing fallback |
| Graph Analysis | NetworkX, SciPy (stats, KDE) |
| Data Storage | Parquet (pyarrow, zstd) — 690 MB CSV → 87 MB committed |
| Deployment | Docker, Render |
| Language | Python 3.12 |

## Key Dataset Facts

- **Total transactions:** 203,769
- **Fraud rate (labeled):** 90.2% of labeled nodes are illicit
- **Network structure:** Sparse directed graph, avg degree 2.3, 49 connected components
- **Top hub node:** Node 2984918 with degree 473
- **High-velocity nodes:** 13,752 nodes above the 90th percentile degree threshold
- **Time range:** 49 bi-weekly snapshots (approx. Jan 2011 – Jan 2013)

## Data Source

[Elliptic Bitcoin Dataset — Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
