# Graph Fraud Detector — ML & Analytics Summary
> GNN-powered Bitcoin transaction fraud detection on the Elliptic dataset, with quantitative risk analytics and business translation layer

**Author:** Sabrina Pribadi  
**Status:** Completed (Sprint 23)  
**Live dashboard:** Deployed on Render via Docker

---

## 1. Project Overview

**Goal:** Classify Bitcoin transactions as illicit or licit using Graph Neural Networks that learn from both per-transaction features AND network topology — something tabular-only models miss. The system wraps the model in a 14-tab Streamlit dashboard, FastAPI REST API, LangChain AI agent, RAG knowledge base, and Phase 6 quantitative finance analytics (stress testing, loss forecasting, regulatory capital, contagion scoring).

**Performance targets achieved:**
| Metric | Target | Achieved |
|--------|--------|----------|
| AUC | ≥ 0.95 | 0.955 |
| F1 | ≥ 0.88 | 0.898 |
| Accuracy | ≥ 88% | 89.0% |

---

## 2. Data Inputs

### Dataset — Elliptic Bitcoin Dataset (Kaggle)

| File | Content | Size |
|------|---------|------|
| `elliptic_txs_features.csv` | 203,769 nodes × 166 features | 658 MB raw → 84.5 MB parquet |
| `elliptic_txs_classes.csv` | Ground-truth labels per node | 1.0 MB parquet |
| `elliptic_txs_edgelist.csv` | 234,355 directed edges | 2.1 MB parquet |

**Parquet pipeline:** Raw CSVs converted once via `scripts/preprocess_data.py` → zstd-compressed parquet in `data/processed/`. On every subsequent run, the loader checks for parquet first (fast path, ~0.5s) before falling back to raw CSVs (slow path, ~25s).

### Dataset Characteristics
- **Total nodes:** 203,769 Bitcoin transactions
- **Total edges:** 234,355 directed payment flows
- **Time coverage:** 49 bi-weekly snapshots, Jan 2011 – Jan 2013
- **Class distribution:** 21% illicit, 2% licit, 77% unknown (unlabeled)
- **Graph density:** Sparse — average degree 2.3
- **Top hub:** Node 2984918, degree 473 (205 standard deviations above mean)
- **High-velocity nodes:** 13,752 nodes exceed the 90th percentile degree threshold

### Labels
| Raw class value | Mapped label | Meaning |
|----------------|-------------|---------|
| `'1'` | 0 | Licit transaction |
| `'2'` | 1 | Illicit transaction |
| `'unknown'` | -1 | Unlabeled (excluded from training) |

---

## 3. Data Pre-Processing Pipeline

### Step 1 — Load & Cache
```
EllipticDataLoader.load_data()
  → check data/processed/*.parquet (fast path)
  → fallback: read raw CSVs → auto-save parquet (slow path, first run only)
```

### Step 2 — Feature Extraction & Normalisation
```python
# Column 0 = node ID, columns 1-166 = features
node_ids = features.iloc[:, 0].values.astype(str)
feature_matrix = features.iloc[:, 1:].values.astype(np.float32)

# Fill missing values with column means
col_means = np.nanmean(features, axis=0)

# Normalize
scaler = StandardScaler()
features_scaled = scaler.fit_transform(feature_matrix)
```

### Step 3 — Label Mapping
```python
# classes.csv: '1' → illicit, '2' → licit, 'unknown' → -1
label_dict = dict(zip(classes['txId'], classes['label']))
```

### Step 4 — Graph Construction
```python
G = nx.DiGraph()
# Nodes: feature vector + label + train/test masks as attributes
# Edges: source, target, optional timestamp
```

### Step 5 — Balanced Sampling for GNN Training
```python
# Problem: full graph with 200k nodes too large for dense adjacency matrix in memory
# Solution: balanced random sample of labeled nodes
sample_per_class = min(sample_size // 2, len(licit_nodes), len(illicit_nodes))
# Default: 3,000 nodes total → 1,500 licit + 1,500 illicit
np.random.seed(42)
```

### Step 6 — Adjacency Matrix Construction
```python
# Dense adj matrix for sampled subgraph
# Directed graph made undirected for message passing (symmetrized)
adj_matrix[i, j] = 1.0
adj_matrix[j, i] = 1.0  # symmetric
```

### Step 7 — Train / Test Split
```
80% train, 20% test (random shuffle within labeled nodes only)
np.random.seed(42) for reproducibility
```

---

## 4. Feature Engineering

### 166 Transaction Features (Elliptic feature groups)

| Feature Group | Index Range | Description |
|--------------|------------|-------------|
| Local metadata | 0–29 | Transaction amounts, fees, input/output counts, coin ages |
| Network centrality | 30–49 | PageRank, betweenness, clustering coefficient |
| Temporal | 50–69 | Hour, day, frequency, transaction velocity |
| Structural graph | 70–89 | Graph diameter, k-core number, triangle count |
| Statistical anomaly | 100–119 | Z-scores, fee ratio, fraud indicators |
| Derived / embeddings | 130–165 | PCA components, autoencoder embeddings, GNN embeddings |

**Most predictive features (from domain knowledge):** Transaction Amount, Fee Ratio, Input Count, Betweenness Centrality.

### Graph Structural Features (Implicit via GNN)
The GNN learns structural patterns that are not explicit features — these are encoded through the message-passing aggregation:
- **Neighbor fraud contamination:** How many neighbors are illicit
- **Hub connectivity:** High-degree nodes with illicit connections (money laundering rings)
- **Graph depth patterns:** How fraud propagates through k-hop neighborhoods

### Auto-Discovery Pattern Features (Rule-based)
| Pattern | Detection Logic |
|---------|----------------|
| Money Laundering Rings | Degree > 10 AND >50% illicit neighbors |
| Structuring | Feature values just below round numbers (smurfing detection) |
| Rapid Transaction Chains | Top-50 by degree, illicit label, >3 illicit connections |
| Mixed Signal Nodes | 30–70% illicit connections among labeled neighbors |
| Anomaly Outliers | Degree z-score > 2.5 standard deviations |

---

## 5. Models

### A. GraphSAGE (Base Model)
**File:** [src/models/gnn_model.py](src/models/gnn_model.py)

```python
class GraphSAGE(nn.Module):
    # Architecture
    in_features → Linear(in, hidden_dim) → ReLU
    hidden_dim=64, num_layers=2, dropout=0.3
    
    # Message passing (forward pass)
    degrees = adj.sum(dim=1)
    aggr = adj @ x / degrees          # mean aggregation
    x = 0.5 * x + 0.5 * aggr         # combine self + neighbors
    
    # Hidden layers with skip connections + BatchNorm
    # Output: Linear(hidden_dim, 1) → BCEWithLogitsLoss
```

**Training:**
```python
optimizer = Adam(lr=0.001)
loss_fn = BCEWithLogitsLoss()
epochs = 100, early_stopping patience = 20
grad_clip: clip_grad_norm_(max_norm=1.0)
device: MPS (Apple Silicon) → CPU fallback
```

### B. GAT — Graph Attention Network
**File:** [src/models/advanced_gnn.py](src/models/advanced_gnn.py)

```python
class GraphAttentionLayer(nn.Module):
    # Learnable attention coefficients per edge
    W = Linear(in_features, out_features)  # Xavier init
    a = Parameter(shape=(2*out_features, 1))  # Xavier init
    
    # Attention mechanism
    e = LeakyReLU(alpha=0.2)(a^T [Wh_i || Wh_j])
    attention = softmax(e masked by adjacency)
    h_prime = attention @ Wh

class GAT(nn.Module):
    # 2 attention heads in first layer
    # Concat heads → BatchNorm → single-head output layer
    num_heads=2, hidden_dim=64, dropout=0.3
```

### C. EnsembleFraudDetector
**File:** [src/models/advanced_gnn.py](src/models/advanced_gnn.py)

```python
class EnsembleFraudDetector:
    models = {
        'gat':       GAT(in_features, hidden_dim=64, num_heads=2),
        'sage_mean': GraphSAGE(aggregator='mean'),
        'sage_sum':  GraphSAGE(aggregator='sum')
    }
    # Equal initial weights: 1/3 each
    # Weight optimization: grid search over w1, w2, w3 combinations
    # Objective: maximize validation AUC
    
    # Final prediction: weighted average of sigmoid outputs
    ensemble_pred = sum(prob * weight for each model)
```

**Aggregator comparison:**
| Aggregator | Behavior | Best for |
|-----------|---------|---------|
| mean | Normalises by degree — stable for hub nodes | General-purpose, prevents hub dominance |
| sum | Accumulates neighbor signals — hub nodes amplified | Detecting high-degree fraud hubs |
| GAT attention | Learned per-edge weights | Selective neighbor influence |

---

## 6. Hyperparameter Tuning

**File:** [src/models/hyperparameter_optimization.py](src/models/hyperparameter_optimization.py)

### Optuna (TPE Sampler + MedianPruner)
```python
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=50)  # default; configurable 8-50
```

### Search Space
| Hyperparameter | Type | Range |
|---------------|------|-------|
| `hidden_dim` | int | 16–128 (step 16) |
| `num_layers` | int | 2–4 |
| `dropout` | float | 0.1–0.5 (step 0.1) |
| `learning_rate` | float | 1e-4 to 1e-2 (log scale) |
| `aggregator` | categorical | `['mean', 'sum']` |

### Objective
```python
# Train 30 epochs per trial (reduced for speed)
# Early stopping within trial: patience=5
# Optimize: maximize validation AUC
# Fallback: if only 1 class in val set → use accuracy instead
```

### UI Integration
- "Apply Optimised Parameters" button in dashboard retrains the model with best trial params
- Updates session state — new model propagates across all 14 tabs
- Best trial hyperparameters shown in a table with Description column per param

---

## 7. Validation & Evaluation Metrics

### Primary Metrics
| Metric | Description | Usage |
|--------|-------------|-------|
| **AUC-ROC** | Area under ROC curve | Primary model selection metric in Optuna |
| **F1 Score** | Harmonic mean of precision/recall | Fraud detection balance metric |
| **Accuracy** | Overall correctness | Baseline check |
| **Precision** | TP / (TP + FP) | False alarm rate |
| **Recall** | TP / (TP + FN) | Fraud catch rate |

### Train / Test Split
- **80% train / 20% test** — random shuffle within labeled nodes
- Only 23% of nodes are labeled (licit + illicit); unknown nodes excluded
- Balanced sampling: equal licit and illicit in training batch

### Risk-Adjusted Performance Metrics
**File:** [src/analytics/risk_adjusted_metrics.py](src/analytics/risk_adjusted_metrics.py)

Bootstraps a per-period True Positive Rate (TPR) series across the 49 Elliptic time steps, then computes finance-style ratios:

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Sharpe Ratio** | `(mean_TPR - risk_free) / std_TPR × √n_periods` | Risk-adjusted catch rate |
| **Sortino Ratio** | Uses semi-deviation (only periods below risk-free) | Downside-adjusted performance |
| **Information Ratio** | `excess_TPR / tracking_error` vs degree-threshold benchmark | Consistency vs baseline |
| **Calmar Ratio** | `annualised mean TPR / max TPR drawdown` | Worst-case scenario robustness |

`risk_free = 5% annualised` (scaled to per-period)

---

## 8. Explainability AI

**File:** [src/analytics/model_explainability.py](src/analytics/model_explainability.py)

### Method 1 — SHAP (KernelExplainer)
```python
explainer = shap.KernelExplainer(predict, X_background[:100])
shap_values = explainer.shap_values(features.reshape(1, -1))

# Top 10 most influential features by |SHAP value|
# Positive SHAP = raises fraud risk (red bars in UI)
# Negative SHAP = lowers fraud risk (blue bars in UI)
```

- Background dataset: first 100 training nodes
- Adjacency: identity matrix (self-only) for single-node explanation
- Output: `top_features` list with name, raw value, and SHAP importance score

### Method 2 — Gradient-Based Attribution (Saliency Map)
```python
x_tensor.requires_grad = True
output = model(x_tensor, identity_adj)
output.backward()
gradients = x_tensor.grad.squeeze()

# Top 10 features by |gradient|
```

- Faster alternative to SHAP (no background sample needed)
- Approximates feature importance via local gradient magnitude

### Subgraph Influence Explanation
```python
# For a given node: get k neighbors from NetworkX graph
# Count illicit neighbors → compute illicit ratio
# influence_score = illicit_neighbors / total_neighbors
```

### UI Rendering
- Horizontal Plotly bar chart: red bars = raises fraud risk, blue = lowers risk
- Attribution score on x-axis
- Single-sentence caption per chart

---

## 9. Loss Forecasting

**File:** [src/analytics/loss_forecasting.py](src/analytics/loss_forecasting.py)

This module derives a fraud-loss **time series** from the graph (the Elliptic dataset has 49 time steps but no explicit timestamp per node — degree percentiles are used as a proxy for time ordering).

### Time-Series Construction
```python
# Bucket nodes into 49 bins by degree percentile
buckets = pd.qcut(degrees, q=49, duplicates='drop')

# For each time step bucket:
est_loss = illicit_count * mean_fraud_prob * loss_per_fraud
# loss_per_fraud: configurable, default $10,000 per illicit node

# Rolling 3-period smoothing to reduce bucket-imbalance noise
df["y"] = df["y"].rolling(window=3, min_periods=1, center=True).mean()

# Time axis: bi-weekly from 2011-01-01
# Step 0 = 2011-01-01, Step 48 = 2013-01-xx
```

### Forecasting Methods

#### Prophet (Primary)
```python
from prophet import Prophet

m = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.3,   # flexible trend
    interval_width=0.80            # 80% confidence band
)
m.fit(df[["ds", "y"]])
future = m.make_future_dataframe(periods=n_periods, freq="W")
fc = m.predict(future)
# Returns: yhat, yhat_lower, yhat_upper
```

#### Holt-Winters Double Exponential Smoothing (Fallback)
```python
# Pure numpy, no Prophet dependency
# alpha=0.3 (level smoothing), beta=0.1 (trend smoothing)
level[t] = alpha * y[t] + (1-alpha) * (level[t-1] + trend[t-1])
trend[t] = beta * (level[t] - level[t-1]) + (1-beta) * trend[t-1]
forecast = [level[-1] + i*trend[-1] for i in range(1, n_periods+1)]
CI: ±1.96 * residual_std
```

**Fallback trigger:** `PROPHET_AVAILABLE = False` when Prophet not installed (e.g., Render free tier).

### Summary Stats
```python
{
  'historical_mean', 'historical_peak',
  'forecast_mean', 'forecast_peak',
  'forecast_vs_hist',  # ratio: forecast mean / historical mean
  'trend_direction'    # 'increasing' or 'decreasing'
}
```

---

## 10. Quantitative Risk Analytics

### 10.1 Core Risk Analysis
**File:** [src/analytics/risk_analysis.py](src/analytics/risk_analysis.py)

#### Expected Loss (Basel Credit Risk)
```
EL = PD × EAD × LGD
  PD  = Probability of Default (= fraud probability from model)
  EAD = Exposure at Default (= exposure per transaction)
  LGD = Loss Given Default (= fraction of loss unrecovered)
  capital_requirement = EL × 8%
```

#### Monte Carlo Simulation
```python
n_simulations = 10,000
fraud_counts = np.random.binomial(n_transactions, fraud_probability, n_simulations)
losses = fraud_counts * avg_loss_per_fraud
loss_variation = np.random.normal(1.0, 0.2, n_simulations)  # 20% std on loss size
losses_varied = losses * loss_variation

outputs: mean_loss, median_loss, std_loss,
         VaR (95th pct), Expected Shortfall (ES above 95th),
         lower/upper confidence bounds
```

#### Time Value of Money Adjustment
```python
daily_rate = discount_rate / 365
present_value = expected_loss / ((1 + daily_rate) ** detection_time_days)
time_value_cost = expected_loss - present_value  # cost of delayed detection
```

### 10.2 Stress Testing
**File:** [src/analytics/stress_testing.py](src/analytics/stress_testing.py)

5 named crisis scenarios applied to base risk parameters:

| Scenario | PD Multiplier | LGD Shift | Volume Shock | Delay (days) |
|----------|--------------|-----------|-------------|-------------|
| Baseline | ×1.0 | +0.00 | ×1.0 | +0 |
| 2008 Financial Crisis | ×3.0 | +0.20 | ×0.6 | +30 |
| COVID-19 Pandemic | ×1.8 | +0.10 | ×0.7 | +15 |
| Crypto Winter | ×2.2 | +0.15 | ×0.4 | +20 |
| Regulatory Crackdown | ×0.5 | −0.10 | ×1.0 | −10 |

Each scenario runs 5,000 Monte Carlo simulations and computes:
- Stressed mean_loss, VaR (95%), Expected Shortfall
- TVM-adjusted total loss
- `severity_ratio` = scenario total loss / baseline total loss

### 10.3 Regulatory Capital (Basel III)
**File:** [src/analytics/regulatory_capital.py](src/analytics/regulatory_capital.py)

#### Standardised Approach (SA)
```
RWA = EAD × risk_weight
  risk weights: retail=75%, corporate=100%, high_risk=150%, crypto_exchange=100%
Capital (SA) = RWA × 8%
```

#### Internal Ratings-Based (IRB) — Vasicek Model
```python
# Stressed PD at 99.9% confidence
PD_stressed = N((N⁻¹(PD) + √rho × N⁻¹(0.999)) / √(1-rho))

# Capital requirement
capital = (LGD × PD_stressed - LGD × PD) × maturity_adjustment × EAD
total_capital = min_capital × (10.5% / 8%)  # capital conservation buffer

# rho: asset correlation parameter (0-1)
# Outputs: SA capital, IRB capital, savings % from using precise model
```

### 10.4 Fraud Contagion Score (SIR Model)
**File:** [src/analytics/contagion.py](src/analytics/contagion.py)

```python
# Stochastic SIR diffusion from each candidate node
def _diffuse(G, seed, steps=3, infection_prob=0.30):
    infected = {seed}
    frontier = {seed}
    for step in range(steps):
        # Each neighbor infected with probability 0.30
        next_frontier = {n for n in neighbors if random < infection_prob}
    return infected - {seed}  # at-risk nodes excluding seed

# Run 10 independent simulations per node
mean_at_risk = mean(|infected| across 10 runs)

# Composite risk: fraud probability weighted by network danger
composite_risk = fraud_prob × (1 + log(1 + mean_at_risk))

# Ranked by composite_risk (highest = most dangerous to miss)
```

**Candidates:** Top 200 nodes by degree (highest-degree nodes are most dangerous if missed).

---

## 11. Auto-Discovery of Fraud Patterns

**File:** [src/analytics/auto_discovery.py](src/analytics/auto_discovery.py)

5 unsupervised pattern types, each returns an `Insight` object:

| Pattern | Method | Severity |
|---------|--------|---------|
| Money Laundering Rings | Degree > 10 AND illicit_neighbor_ratio > 50% | HIGH |
| Structuring (Smurfing) | Feature values within 0.05 of round numbers | MEDIUM |
| Rapid Transaction Chains | Degree in top-50, illicit label, >3 illicit connections | HIGH |
| Mixed Signal Nodes | 30–70% illicit ratio among labeled neighbors | MEDIUM |
| Anomaly Outliers | Degree z-score > 2.5 | HIGH |

Each insight includes: title, description, category, severity, data dict, chart metadata.

---

## 12. Temporal Analysis

**File:** [src/analytics/temporal_analysis.py](src/analytics/temporal_analysis.py)

Pre-computes node degrees and labels in one pass (cached), then provides:

| Analysis | Description |
|----------|-------------|
| Activity summary | Total nodes, avg/max/min degree, class distribution |
| Transaction velocity | Degree percentiles (25/50/75/90/95/99), high-velocity threshold = 90th pct |
| Time patterns | High (>20 edges), medium (5–20), low (<5) activity buckets |
| Fraud trends | Degree comparison: illicit vs licit avg/max degree |
| Temporal anomalies | Z-score > 2.5: top-N highest absolute z-score nodes |

**Anomaly threshold:** `|z-score| > 2.5` (both high-degree and low-degree outliers flagged).

---

## 13. LLM Agent & RAG

### FraudAgent (LangChain)
**File:** [src/agent/fraud_agent.py](src/agent/fraud_agent.py)

- **Framework:** LangChain + LangGraph v1.3
- **Model:** GPT-4o-mini (OpenAI)
- **6 Tools:**
  1. `get_fraud_stats` — dataset-level statistics
  2. `find_suspicious_nodes` — top-N by fraud probability
  3. `analyze_network` — graph topology metrics
  4. `predict_transaction` — single-node inference
  5. `run_risk_analysis` — Monte Carlo + Expected Loss
  6. `get_anomalous_patterns` — trigger auto-discovery
- **Offline fallback:** All 6 tools run via rule-based keyword routing when no API key

### FraudRAGAgent (ChromaDB + TF-IDF)
**File:** [src/agent/rag_agent.py](src/agent/rag_agent.py)

- **Knowledge base:** 25 curated domain documents (fraud typologies, GNN architecture, risk concepts, dataset description)
- **Vector store:** ChromaDB PersistentClient, OpenAI `text-embedding-3-small` (256-dim matryoshka)
- **Retrieval:** Cosine similarity → top-5 documents
- **Synthesis:** GPT-4o-mini with strict grounding prompt
- **TF-IDF fallback:** sklearn `TfidfVectorizer` with bigrams (no OpenAI key needed)
- **Web fallback:** DuckDuckGo search → Google link fallback

---

## 14. API & Deployment

### FastAPI REST Endpoints
| Endpoint | Method | Description |
|---------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/stats` | GET | Graph and model statistics |
| `/predict` | POST | Node fraud probability |
| `/network/stats` | GET | Network topology stats |
| `/analyze/risk` | POST | Monte Carlo risk analysis |
| `/discover/insights` | GET | Auto-discovery patterns |
| `/docs` | GET | Swagger UI |

### Deployment
- **Dashboard:** Streamlit Docker container, port 8501, Render
- **API:** Separate FastAPI Docker container, port 8000, Render
- **MPS fix:** `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to prevent memory exhaustion on Apple Silicon

---

## 15. Libraries Used

```
# GNN & Deep Learning
torch                 → GraphSAGE, GAT, EnsembleFraudDetector
torch.nn              → BCEWithLogitsLoss, BatchNorm1d, Dropout

# Graph
networkx              → DiGraph construction, degree, centrality

# Classical ML & Preprocessing
sklearn               → StandardScaler, TfidfVectorizer, metrics
shap                  → KernelExplainer for feature attribution
scipy.stats           → Statistical functions for VaR, regression

# Hyperparameter Optimization
optuna                → TPE sampler, MedianPruner, 8–50 trials

# Time-Series Forecasting
prophet               → Loss forecasting (changepoint_prior_scale=0.3)
numpy                 → Holt-Winters double exponential smoothing fallback

# LLM & RAG
langchain, langgraph  → Agent orchestration (v1.3)
chromadb              → Persistent vector store (cosine similarity)
openai                → GPT-4o-mini (agent + RAG), text-embedding-3-small

# Data
pandas                → Data loading, time-series bucketing
pyarrow               → zstd parquet read/write

# UI & API
streamlit             → 14-tab dashboard, dark theme, session state
fastapi               → REST API with Pydantic validation
pydantic              → Request schemas

# Deployment
docker                → Container images for dashboard + API
```

---

## 16. Key Design Decisions

- **GNN over tabular models:** Transaction graph topology (who sent to whom) contains fraud signals invisible to standalone feature models — laundering rings, layering patterns, and hub contamination only appear when edges are modeled.

- **Balanced class sampling in training:** 90% of labeled nodes are illicit — training on full labeled set would bias the model. Balanced 50/50 sampling during GNN graph data construction prevents this at the cost of not using all available data.

- **Degree percentile as time proxy:** The Elliptic dataset has 49 time steps but no explicit timestamp per node in the feature file. Degree percentiles serve as an ordered proxy for temporal activity level — enabling the loss forecasting time series without requiring a separate join.

- **Prophet → Holt-Winters fallback:** Prophet has C/Stan compile dependencies that can fail on Render's free tier. Pure-numpy Holt-Winters is always available as a fallback with similar trend extrapolation for the short forecasting horizons needed.

- **SIR over deterministic contagion:** Stochastic diffusion with 10 runs per node produces a distribution of at-risk counts rather than a single deterministic count — more realistic for modeling varied network response to a missed fraud node.

- **IRB vs SA capital comparison:** Shows compliance officers exactly how much regulatory capital a high-precision fraud model saves (IRB requires less capital when PD is well-estimated vs SA's flat risk weights).

- **Composite Risk Score = fraud_prob × log(1 + at_risk):** Logarithm prevents high-degree hub nodes from completely dominating the ranking — a moderately fraudulent highly-connected node should not always outrank a very fraudulent moderately-connected one.

- **No emoji in source files:** All print statements, return strings, and log messages use plain text. Dashboard uses Streamlit Material icons and inline SVG instead. Reason: emoji in log streams break certain terminal encodings and look unprofessional in production logs.
