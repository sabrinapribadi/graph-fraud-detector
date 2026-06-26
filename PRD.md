PRODUCT REQUIREMENT DOCUMENT (PRD)
Project: Graph Fraud Detector — GNN-Powered Bitcoin Transaction Fraud Detection
Version: 3.2 (Network AI Labels, Hyperparameter Apply, Git History Restore)
Author: Sabrina Pribadi
Date: June 26, 2026
Status: Completed


1. EXECUTIVE SUMMARY

Problem: Compliance analysts and data scientists tasked with Bitcoin fraud detection lack tools
that combine graph-structural intelligence, quantitative risk quantification, and natural-language
exploration in a single interface. Existing approaches use static rule-based systems or train
models on feature tables that ignore network topology — missing fraud patterns that are only
visible in the transaction graph. Beyond the technical gap, non-finance users (operations,
executives, auditors) cannot act on outputs they cannot understand.

Solution: A Graph Neural Network (GNN) system that learns from both transaction features and
network topology to classify illicit nodes in the Elliptic Bitcoin dataset. The system wraps
the model in a 14-tab Streamlit dashboard, a FastAPI REST API, a LangChain AI agent, a
Retrieval-Augmented Generation (RAG) knowledge base, Phase 6 advanced quant-finance analytics,
and a Business Translation Layer that frames every technical output as a plain-English business
question — enabling both technical and non-technical users to explore, query, and act on fraud
insights without a finance background.

Value Proposition: Reduce time-to-insight for Bitcoin fraud investigators from days to minutes.
Provide quantifiable risk exposure metrics (VaR, Expected Loss, regulatory capital requirements,
crisis stress scenarios, contagion risk) directly from model output. Frame every analytic as
a business question so results are immediately actionable by non-specialists. Demonstrate
end-to-end data science, ML engineering, LLM orchestration, RAG design, quant-finance modelling,
golden-ratio UI/UX, and MLOps deployment practices in a single cohesive portfolio project.

Success Metrics:
- GNN model achieves AUC > 0.95 on the Elliptic held-out test set
- AI agent answers 6 standard test questions with correct statistics
- RAG knowledge base retrieves relevant documents for 5+ question types
- Dashboard loads in under 3 seconds on first visit
- All 5 auto-discovery insight types surface without user prompting
- Docker containers deploy successfully to Render free tier
- Every tab displays a plain-English business question alongside the technical context
- Phase 6 tabs show dynamic plain-English result interpretation after computation


2. PROJECT CONTEXT AND BACKGROUND

This project was built as a portfolio piece to demonstrate capabilities across:
1. Graph Machine Learning: Training GraphSAGE and GAT on a real financial fraud dataset.
2. Quantitative Risk Analysis: Implementing Monte Carlo simulation, VaR, TVM adjustments,
   stress testing, risk-adjusted metrics, loss forecasting, regulatory capital, and contagion.
3. LLM and Agentic Design: Building a LangChain agent with 6 domain-specific tools and a
   fallback mode that works without an API key.
4. RAG Engineering: Designing a ChromaDB + OpenAI embedding pipeline with TF-IDF fallback,
   a static knowledge base of 25 fraud domain documents, and dynamic insight ingestion.
5. Software Engineering: Modular, testable Python codebase with separation of concerns across
   data, models, analytics, agent, API, and UI layers.
6. MLOps: Docker containerisation, Render deployment, environment variable management.
7. UI/UX Design: Golden ratio column layouts (1:1.618), Fibonacci spacing, Material icons,
   dark theme, dual context panels (technical + business) on each tab, SVG icons.
8. Business Translation: Every tab includes a green Business Question panel; Phase 6 tabs
   include a dynamic amber Plain English callout that converts numbers into decisions.
9. Code Quality: No emoji in any source file — scripts use plain-text logging; dashboard
   uses Streamlit Material icons and inline SVG/CSS for all visual indicators.

Data Source: Elliptic Bitcoin Dataset (Kaggle)
- elliptic_txs_features.csv: 203,769 nodes, 166 features per node
- elliptic_txs_classes.csv: Ground-truth labels (1=illicit, 2=licit, unknown)
- elliptic_txs_edgelist.csv: 234,355 directed edges


3. SCOPE

In-Scope:
- Data: Elliptic Bitcoin transaction dataset (parquet-first pipeline)
- Models: GraphSAGE, GAT, Ensemble (majority voting), Optuna hyperparameter optimisation
- Core Risk Analysis: Monte Carlo simulation, Expected Loss (PD × EAD × LGD), VaR (95%),
  Time Value of Money adjustments
- Phase 6 — Advanced Quant Finance:
    - Stress Testing: 5 named crisis scenarios × Monte Carlo 5,000 simulations
    - Risk-Adjusted Metrics: Sharpe, Sortino, Information Ratio, Calmar (bootstrapped TPR series)
    - Loss Forecasting: Prophet + Holt-Winters fallback; 80% confidence intervals
    - Regulatory Capital: Basel III Standardised Approach + IRB Vasicek formula (99.9% CI)
    - Fraud Contagion Score: Stochastic SIR diffusion; Composite Risk Score
- Business Translation Layer: Green Business Question panel + amber Plain English callout on all
  14 tabs; help= tooltip text on all technical parameter inputs
- LLM Agent: LangChain agent with 6 tools; offline fallback mode
- RAG Knowledge Base: 25 domain documents, ChromaDB vector store, TF-IDF fallback,
  OpenAI GPT-4o-mini synthesis
- Auto-Discovery: 5 proactive fraud pattern detection methods
- Model Explainability: Gradient-based feature attribution
- UI: Streamlit 14-tab dashboard with dark theme, golden ratio layout, Material icons
- API: FastAPI REST service with 8 endpoints and Swagger documentation
- Deployment: Docker containers on Render
- Documentation: README.md (GitHub front page) and PRD.md (engineering reference)

Out-of-Scope:
- Real-time streaming fraud detection
- User authentication or multi-tenancy
- Mobile-native application
- Advanced LLM fine-tuning on domain data
- Cloud orchestration (Kubeflow, Vertex AI)
- Geo-spatial analysis (Bitcoin transactions lack location data)


4. USER PERSONAS AND STORIES

| Persona              | Goal                                                          | Pain Point                               |
|----------------------|---------------------------------------------------------------|------------------------------------------|
| Alex (Compliance)    | Identify illicit transaction hubs for investigation           | Manual graph traversal takes days        |
| Maria (Risk Officer) | Quantify portfolio exposure and regulatory capital            | No quantitative risk model exists        |
| James (Data Sci.)    | Understand how the GNN model makes predictions                | Black-box model, no explainability       |
| Dana (Analyst)       | Ask plain-English questions without writing Python or SQL     | High technical barrier to ML tools      |
| Sam (Executive)      | See the business impact without learning quant finance        | Technical dashboards don't answer "so what?" |

User Stories Implemented:
- As a compliance officer, I can view the top suspicious transactions ranked by fraud probability.
- As a risk officer, I can run a Monte Carlo simulation and see VaR at 95% confidence.
- As a risk officer, I can simulate 5 crisis scenarios and see how much worse losses get.
- As a risk officer, I can calculate regulatory capital under SA and IRB approaches and see
  how much capital a precise model saves.
- As a data scientist, I can explain why the model flagged a specific transaction by seeing its
  top 5 influential features.
- As an analyst, I can ask "What are the main fraud patterns?" and receive a grounded answer
  with source documents from the knowledge base.
- As an executive with no finance background, I can read the plain-English Business Question
  panel on every tab and immediately understand what the numbers mean for the business.
- As a user, I can explore the transaction graph interactively, coloured by label or degree.
- As a user, I can filter, sort, and export the transaction table to CSV.
- As a user, I can see automatically surfaced fraud insights without writing any query.
- As a user, I can see a budget estimate for fraud losses next quarter without knowing what
  Prophet or Holt-Winters means.
- As a user, I can see which specific transaction is most dangerous to miss — the one that
  would put the most other accounts at risk if overlooked.


5. FUNCTIONAL REQUIREMENTS

Module A: Data Pipeline
- A.1 Load dataset — parquet-first: checks data/processed/{features,classes,edgelist}.parquet;
     falls back to raw CSVs and auto-saves a parquet cache on first run.
- A.2 Preprocessing script (scripts/preprocess_data.py): converts three raw CSVs (690 MB total)
     to zstd-compressed parquet (87 MB total) committed directly to the repository.
     features.parquet: 84.5 MB | classes.parquet: 1.0 MB | edgelist.parquet: 2.1 MB.
     No Git LFS required — all three files are under GitHub's 100 MB per-file limit.
- A.3 Normalise features using StandardScaler.
- A.4 Map class labels: '1' -> 1 (illicit), '2' -> 0 (licit), 'unknown' -> -1.
- A.5 Build a NetworkX directed graph with node features and labels as attributes.
- A.6 Expose graph statistics: nodes, edges, degree distribution, class counts.

Module B: GNN Models
- B.1 GraphSAGE: 2-layer model, 32 hidden dimensions, mean aggregation, batch normalisation,
     dropout 0.2, BCEWithLogitsLoss, early stopping.
- B.2 GAT: Multi-head attention layers, Xavier initialisation, LeakyReLU activations.
- B.3 EnsembleFraudDetector: Trains GAT + GraphSAGE(mean) + GraphSAGE(sum) in parallel;
     majority voting for final prediction.
- B.4 HyperparameterOptimizer: Optuna TPE sampler, 8-50 trials, MedianPruner;
     tunes hidden_dim (16-128), num_layers (2-4), dropout (0.1-0.5), learning rate, aggregator.
- B.5 Model Explainability: Gradient-based feature attribution; renders top-10 features as a
     horizontal Plotly bar chart — red bars raise fraud risk, blue bars lower it — with an
     attribution score axis and a plain-language caption. Previously showed as a plain text list.
- B.6 Performance targets: AUC >= 0.95, F1 >= 0.88, Accuracy >= 88%.

Module C: Core Quantitative Risk Analysis
- C.1 Expected Loss: EL = PD × EAD × LGD with configurable parameters.
- C.2 Monte Carlo: 10,000 binomial fraud-count draws × normal loss variation (20% std);
     returns mean, median, VaR (95%), Expected Shortfall.
- C.3 TVM Adjustment: Computes additional loss cost for detection delays using discount rate.
- C.4 Full Assessment: Combines all three methods into a unified risk score.

Module D: LLM Agent (FraudAgent)
- D.1 Framework: LangChain with LangGraph v1.3.
- D.2 Tools: 6 tools — get_fraud_stats, find_suspicious_nodes, analyze_network,
     predict_transaction, run_risk_analysis, get_anomalous_patterns.
- D.3 Model: GPT-4o-mini.
- D.4 Fallback: When no API key is configured, all tool implementations run directly via
     rule-based routing (keyword matching on the question).
- D.5 Output format: Professional plain-text with ASCII-style headers; no emoji in outputs.

Module E: RAG Knowledge Base (FraudRAGAgent)
- E.1 Knowledge base: 25 curated documents covering fraud typologies, GNN architecture,
     risk analysis concepts, dataset description, and deployment details.
- E.2 Vector store: ChromaDB PersistentClient at data/chroma_fraud/; collection name
     'fraud_knowledge'; OpenAI text-embedding-3-small with 256-dim matryoshka reduction.
- E.3 Retrieval: Cosine-similarity search returns top-5 most relevant documents.
- E.4 Synthesis: Retrieved context passed to GPT-4o-mini with a strict grounding prompt.
- E.5 TF-IDF fallback: When ChromaDB or OpenAI is unavailable, sklearn TfidfVectorizer
     with bigrams performs cosine-similarity keyword search over the static knowledge base.
- E.6 Dynamic update: update_with_insights() adds auto-discovered patterns to the vector
     store so users can query the current run's findings.
- E.7 Index build: scripts/build_fraud_index.py. Estimated cost: ~$0.01 one-time.
- E.8 UI: 9th dashboard tab with 6 preset questions, free-text search, synthesised answer,
     and source document cards showing similarity scores.

Module F: Auto-Discovery (AutoDiscovery)
- F.1 Method 1 — Money Laundering Rings: Nodes with degree > 10 and > 50% illicit neighbours.
- F.2 Method 2 — Structuring: Transaction amounts clustered near detection thresholds.
- F.3 Method 3 — Rapid Transaction Chains: Degree above 90th percentile with illicit connections.
- F.4 Method 4 — Mixed Signal Nodes: Licit-labeled nodes with at least one illicit neighbour.
- F.5 Method 5 — Anomaly Outliers: Degree z-score > 2.5 standard deviations.
- F.6 Each insight returns: title, description, category, severity (HIGH/MEDIUM/LOW), data, chart.
- F.7 UI: Severity and category badges, 2-column card grid, inline Plotly charts, CSV export.

Module G: FastAPI REST Service
- G.1 Endpoints: GET /, GET /health, GET /stats, POST /predict, GET /network/stats,
     POST /analyze/risk, GET /discover/insights, GET /docs.
- G.2 Pydantic schemas for request validation.
- G.3 Model loaded at startup via lifespan event.
- G.4 Deployed as a separate Docker container on Render (port 8000).

Module H: Streamlit Dashboard (14-tab)
- H.1 14 tabs with Material icon labels — no emoji in any tab name or heading.
- H.2 Golden ratio column splits (1:1.618) applied throughout.
- H.3 Fibonacci-scale spacing variables (8/13/21/34/55 px) in CSS.
- H.4 Technical context panel (blue-border) at the top of each tab explaining the methodology.
- H.5 Business Translation panel (green-border) immediately below the technical panel on every
     tab; frames content as a plain-English business question + answer.
- H.6 SVG shield icon + styled CSS dots (no emoji circles) for status indicators.
- H.7 Plotly charts: dark backgrounds, proper axis titles, reference lines, captions.
     _to_rgba() helper converts any hex or rgb() color to rgba(r,g,b,alpha) for Plotly
     fillcolor — avoids 8-character hex (#RRGGBBAA) which Plotly 6.x rejects.
     Feature importance chart uses horizontal bar chart (red = raises risk, blue = lowers risk).
     Stress test scenario legend positioned above chart (y=1.12) to avoid overlapping x-axis labels.
- H.8 Sidebar: SVG brand icon; 2×2 CSS grid cards showing full numeric values (Nodes, Edges,
     Licit, Illicit) — replaces st.metric() columns which truncated large numbers to "203,…";
     model performance progress bars; nav guide expander.
- H.9 Session state management for data, model, agent, RAG agent, chat history, discovery
     insights, ensemble model, temporal analyser, stress results, RAM report, LF result,
     RC comparison, and contagion scores. rag_do_search flag enables auto-search when a
     preset question button is clicked (st.rerun() loop).
- H.10 Dark theme via .streamlit/config.toml (primaryColor #FF5A5F coral).
- H.11 AI Chat rendering uses st.chat_message("user") and st.chat_message("assistant") with
     _render_agent_response() helper: detects Key : Value metric lines via regex and renders
     them as st.metric() grid cards; remaining prose rendered as markdown. Replaces previous
     custom HTML <pre> block that forced monospace and showed all text on one line.
- H.12 Overview tab Transaction Classification section: dual donut charts side by side —
     left shows ground-truth labels (77% unknown), right shows GNN model predictions for ALL
     203k nodes including unlabeled ones (addresses user question about unknown classification).
     The model's probability threshold (>=0.5) is used to classify all nodes without requiring
     a separate unsupervised step.
- H.13 Network Explorer: 4th coloring option "AI-Predicted Label" — runs model inference on
     the displayed subgraph, colours predicted-illicit nodes red (#FF5A5F) and predicted-licit
     green (#00CC96). Displays a yellow caution callout explaining these are model predictions
     not verified ground truth. Shows confusion stats (predicted vs actual illicit in sample).
- H.14 Advanced ML → Hyperparameter Optimisation: after Optuna trial completes, "Apply
     Optimised Parameters" button retrains the model with the best hidden_dim, num_layers,
     dropout and lr found, updates session_state.detector, and shows new AUC. This closes the
     loop — users previously got best params displayed but had no way to act on them in-app.

Module I: Phase 6 — Advanced Quant Finance

I.1 Stress Testing (src/analytics/stress_testing.py)
- StressScenario dataclass: name, description, pd_multiplier, lgd_delta, ead_multiplier,
  volume_shock, delay_delta.
- 5 named scenarios: Baseline, 2008 Financial Crisis, COVID-19 Pandemic, Crypto Winter,
  Regulatory Crackdown.
- StressTester: runs 5,000 Monte Carlo simulations per scenario; returns mean_loss, var_95,
  expected_shortfall, total_loss, severity_ratio vs Baseline.
- UI: base parameter controls (PD, LGD, avg loss, delay); KPI row (worst scenario, severity
  ratio, VaR); KDE loss distribution overlay for all 5 scenarios; severity heatmap; parameter
  table; expandable scenario descriptions.

I.2 Risk-Adjusted Metrics (src/analytics/risk_adjusted_metrics.py)
- RiskAdjustedAnalyzer: bootstraps per-period TPR series from 200 labelled nodes per period
  across 49 pseudo-periods; synthetic fallback when model data is unavailable.
- Sharpe Ratio: (mean_TPR - risk_free) / std_TPR × sqrt(49); risk_free = 5% annualised.
- Sortino Ratio: uses correct semi-deviation formula — only returns BELOW the risk-free threshold
  contribute to downside variance (downside_sq = where(excess < 0, excess², 0)); produces a
  meaningful Sortino > Sharpe when the model consistently beats the risk-free target.
  Bug fixed: previous implementation was mathematically identical to Sharpe (std(1-r) = std(r)).
- Information Ratio: excess TPR over degree-threshold benchmark / tracking error.
- Calmar Ratio: annualised mean TPR / maximum TPR drawdown.
- full_report(): all 4 metrics + text interpretation array.
- UI: KPI row (4 ratios); radar scorecard; TPR series vs benchmark chart; drawdown chart;
  active return histogram; metric interpretation table; business summary markdown table.

I.3 Loss Forecasting (src/analytics/loss_forecasting.py)
- LossForecaster: buckets nodes into N_STEPS=49 time bins by degree percentile; computes
  estimated loss per bin as illicit_count × mean_fraud_prob × loss_per_fraud; rolling 3-period
  smoothing to reduce bucket-imbalance noise.
- Time axis: bi-weekly snapshots starting 2011-01-01, step size = 2 weeks → covers Jan 2011 –
  Jan 2013 (correct Elliptic dataset date range). Bug fixed: previous code used weekly steps
  which compressed all 49 steps into 11 months of 2011 and confused users.
- forecast_prophet(): fits Prophet (changepoint_prior_scale=0.3, no seasonality, 80% interval).
- forecast_holt_winters(): double exponential smoothing (alpha=0.3, beta=0.1) fallback when
  Prophet is not installed; future_dates at freq="2W" to match historical bi-weekly axis.
- forecast(): unified entry point — uses Prophet when available, Holt-Winters otherwise.
- UI: n_forecast slider; loss_per_fraud input; KPI row (historical mean, peak, forecast mean,
  trend); main forecast chart with CI band; x-axis title explains bi-weekly cadence and
  2011–2013 date range; updated caption; period-by-period bar chart; summary table; plain-English
  budget estimate.

I.4 Regulatory Capital (src/analytics/regulatory_capital.py)
- RegulatoryCapitalCalculator: configurable total_exposure, fraud_probability, LGD, rho,
  exposure_class.
- standardised_approach(): Basel III SA — RWA = EAD × risk_weight; capital = RWA × 8%.
  Risk weights: retail 75%, corporate 100%, high_risk 150%, crypto_exchange 100%.
- irb_approach(): Vasicek single-factor model at 99.9% confidence.
  Stressed PD = N((N⁻¹(PD) + √rho × N⁻¹(0.999)) / √(1 - rho)); capital = (LGD × PD_stressed -
  LGD × PD) × maturity_adjustment × EAD; total capital = min_capital × (10.5% / 8%).
- rho_sensitivity() / pd_sensitivity(): sweep capital requirement across parameter ranges.
- compare(): returns capital_diff and saving_pct between SA and IRB.
- UI: exposure parameters with help= tooltips; KPI row (SA capital, IRB capital, stressed PD,
  RWA); plain-English saving/cost explanation; SA vs IRB bar chart; rho sensitivity curve;
  PD sensitivity curve; side-by-side detail table.

I.5 Fraud Contagion Score (src/analytics/contagion.py)
- _diffuse(): stochastic SIR diffusion from a single seed node; steps=3, infection_prob=0.30;
  returns set of infected nodes excluding seed.
- ContagionAnalyzer: runs N_RUNS=10 independent diffusion simulations per candidate node;
  computes mean_at_risk, max_at_risk, contagion_multiplier, composite_risk.
- Composite Risk Score = fraud_prob × (1 + log(1 + mean_at_risk)).
- network_summary(): network-level KPIs including illicit_mean_at_risk and top_node.
- UI: diffusion parameter controls with help= tooltips; KPI row (candidates, mean/max at risk,
  top composite risk); plain-English contagion interpretation; fraud-prob vs mean-at-risk scatter;
  composite risk histogram; ranked top-20 table.

Module J: Business Translation Layer
- J.1 CSS class .business-box: green/teal left border (var(--color-low)), background
  rgba(0,204,150,0.07); placed below the blue .context-box on every tab.
- J.2 CSS class .plain-english: amber left border (#FFA726), background rgba(255,167,38,0.08);
  rendered inline after Phase 6 results are computed.
- J.3 Helper function _biz_box(question, answer): renders the green Business Question panel.
- J.4 Helper function _plain_english(text): renders the amber Plain English callout.
- J.5 All 14 tabs have a Business Question framing appropriate to the tab's content.
- J.6 Phase 6 tabs (10–14) generate dynamic plain-English text from computed results:
    - Stress Testing: "In a {worst_scenario}, losses would be {ratio}× your baseline —
      from ${baseline} to ${worst_total}. In 95% of scenarios, losses won't exceed ${var_95}."
    - Risk-Adjusted Metrics: model catch rate vs baseline %, Sharpe verdict, Sortino observation.
    - Loss Forecasting: direction + % change, per-period budget estimate.
    - Regulatory Capital: reserve amount, expected loss, IRB saving/cost explanation.
    - Contagion Score: mean accounts at risk, max exposure, highest-priority investigation target.
- J.7 Technical sliders with finance jargon (PD, LGD, rho, diffusion steps, infection
  probability) all carry help= tooltip text explaining the concept in plain English.


6. NON-FUNCTIONAL REQUIREMENTS

- Performance: Dashboard loads < 3 seconds; GNN training < 60 seconds (2,000-node sample);
  Monte Carlo 10,000 runs < 10 seconds; RAG search < 5 seconds;
  Contagion Score (200 nodes × 10 runs) < 60 seconds.
- Cost: LLM calls use GPT-4o-mini throughout; RAG index build ~$0.01 one-time;
  LLM agent and RAG synthesis cached in session state to avoid redundant calls.
- Resilience: All LLM and ChromaDB calls have working fallbacks. Prophet falls back to
  Holt-Winters. Model inference in Phase 6 falls back to label-based probability proxy.
  GNN training falls back to CPU when MPS fails.
- Code Quality: Modular structure (src/data/, src/models/, src/analytics/, src/agent/,
  src/api/, src/ui/). No emoji in Python source files. Plain-text logging in scripts.
- Theme: Consistent dark theme via config.toml; coral accent (#FF5A5F) for alerts;
  teal/green (var(--color-low)) for business context; amber (#FFA726) for plain-English
  callouts; blue/indigo for technical context.
- Documentation: README.md covers setup, data, run, Docker, RAG index, and API.
  PRD.md covers all modules, architecture, risks, and success criteria.
- Version Control: Clean Git history; sensitive files (.env, data/) excluded via .gitignore.


7. DATA STRATEGY

Data Dictionary (Key Fields):

| Field      | Type   | Description                                      | Example    |
|------------|--------|--------------------------------------------------|------------|
| node_id    | string | Unique transaction identifier                    | "5530458"  |
| label      | int    | Ground truth: 0=licit, 1=illicit, -1=unknown     | 1          |
| features   | list   | 166 normalised transaction features              | [0.12, ...] |
| degree     | int    | Number of connected transactions                 | 3          |
| fraud_prob | float  | GNN output probability (0-1)                     | 0.932      |
| risk_level | string | HIGH / MEDIUM / LOW                              | "HIGH"     |

Feature Groups (166 total):
- Features 0-29: Local transaction metadata (amounts, fees, input/output counts, ages)
- Features 30-49: Network centrality measures (PageRank, betweenness, clustering)
- Features 50-69: Temporal features (hour, day, frequency, velocity)
- Features 70-89: Structural graph features (diameter, k-core, triangle count)
- Features 100-119: Statistical anomaly scores (z-score, fee ratio, fraud indicators)
- Features 130-165: Derived features (PCA components, autoencoder embeddings, GNN embeddings)

Data Pipeline:
1. EllipticDataLoader.load_data() checks data/processed/*.parquet (fast path, ~0.5s)
2. If parquet absent: load raw CSVs, auto-save parquet cache for next run (slow path, ~25s)
3. Preprocessing script (run once): scripts/preprocess_data.py -> 690 MB CSV -> 87 MB parquet
4. Normalise 166 features with StandardScaler
5. Map class labels ('1' -> illicit, '2' -> licit, 'unknown' -> unknown)
6. Build NetworkX directed graph with node attributes
7. Sample 2,000 balanced nodes for GNN training (50% illicit, 50% licit)
8. Build PyTorch tensors (feature matrix, adjacency matrix, labels)

Render Services:
- Dashboard: graph-fraud-detector-dashboards.onrender.com  (Docker, port 8501)
- API:       graph-fraud-detector-api-service.onrender.com  (Docker, port 8000)


8. TECHNICAL ARCHITECTURE

+──────────────────────────────────────────────────────────────────────────────────+
|                       STREAMLIT DASHBOARD (14 tabs)                               |
|  Overview | Network | AI Agent | Risk | Explorer | Discovery | ML | Temporal      |
|  Knowledge Base (RAG)                                                             |
|  Stress Testing | Risk-Adjusted Metrics | Loss Forecasting                        |
|  Regulatory Capital | Fraud Contagion Score                                       |
+──────────────────────────────┬───────────────────────────────────────────────────+
                               |
         +─────────────────────+────────────────────────────────────+
         |                     |                                    |
+────────+──────+  +───────────+──────+  +──────────────────────────+──────────+
| FraudAgent    |  | FraudRAGAgent    |  | Analytics Layer                      |
| LangChain     |  | ChromaDB /       |  | RiskAnalyzer / AutoDiscovery         |
| 6 tools       |  | TF-IDF           |  | TemporalAnalyzer / ModelExplainer    |
+────────+──────+  +──────────────────+  | StressTester / RiskAdjustedAnalyzer  |
         |                               | LossForecaster / RegCapCalculator    |
         |                               | ContagionAnalyzer                    |
         |                               | Business Translation (_biz_box,      |
         |                               |   _plain_english, .business-box CSS) |
         |                               +──────────────────────────────────────+
         |
+────────+────────────────────────────────────────────────────────+
| GNN Models: GraphSAGE | GAT | EnsembleFraudDetector | Optuna    |
+────────+────────────────────────────────────────────────────────+
         |
+────────+─────────────────────────+
| EllipticDataLoader                |
| Normalise -> Graph -> Tensors     |
+────────+─────────────────────────+
         |
+────────+──────────────────────────────────────+
| data/processed/ (committed, 87 MB total)      |
| features.parquet (84.5 MB, zstd)              |
| classes.parquet  (1.0 MB)                     |
| edgelist.parquet (2.1 MB)                     |
| Generated from raw CSVs (690 MB, gitignored)  |
+───────────────────────────────────────────────+

              +────────────────────────────────+
              | FastAPI REST API (separate svc) |
              | 8 endpoints | Swagger /docs     |
              | Docker container, port 8000     |
              +────────────────────────────────+


9. IMPLEMENTATION PLAN

| Sprint | Duration | Deliverables                                                      | Status    |
|--------|----------|-------------------------------------------------------------------|-----------|
| 1      | 3 days   | Data loading, preprocessing, graph construction                   | Completed |
| 2      | 3 days   | GraphSAGE training and evaluation (AUC 0.955)                     | Completed |
| 3      | 3 days   | Quantitative risk analysis (Monte Carlo, VaR, TVM)                | Completed |
| 4      | 3 days   | LangChain agent with 6 tools and offline fallback                 | Completed |
| 5      | 2 days   | Auto-Discovery (5 pattern types) with Insight objects             | Completed |
| 6      | 2 days   | Temporal analysis with z-score anomaly detection                  | Completed |
| 7      | 2 days   | Advanced ML: GAT, Ensemble, Optuna, Explainability                | Completed |
| 8      | 3 days   | Streamlit dashboard with dark theme (9 core tabs)                 | Completed |
| 9      | 1 day    | FastAPI REST API, Docker containers, Render deployment            | Completed |
| 10     | 2 days   | RAG: FraudRAGAgent, ChromaDB, 25 knowledge documents,            | Completed |
|        |          | build_fraud_index.py, Knowledge Base tab (9th tab)                |           |
| 11     | 1 day    | UX overhaul: golden ratio, Material icons, remove emoji,          | Completed |
|        |          | context panels, improved charts, sidebar redesign                 |           |
| 12     | 2 days   | Parquet pipeline, LFS removal, Render deployment fix              | Completed |
| 13     | 3 days   | Phase 6: Stress Testing, Risk-Adjusted Metrics, Loss Forecasting, | Completed |
|        |          | Regulatory Capital, Fraud Contagion Score (5 new tabs)            |           |
| 14     | 1 day    | Business Translation Layer: .business-box CSS, _biz_box(),        | Completed |
|        |          | _plain_english(), 14 business questions, 5 dynamic callouts,      |           |
|        |          | help= tooltip text on all technical inputs                        |           |
| 15     | 1 day    | UI/UX hardening + quant bug fixes: Sortino semi-deviation fix;    | Completed |
|        |          | forecast bi-weekly date axis (2011–2013); feature importance bar  |           |
|        |          | chart; dual classification donut; sidebar 2×2 grid cards;        |           |
|        |          | st.chat_message() rendering; stress test legend; Regulatory       |           |
|        |          | Capital max_value fix; KDE fillcolor rgba fix; KB auto-search;   |           |
|        |          | Network Explorer AI-Predicted Label option with caution callout;  |           |
|        |          | Hyperparameter Apply button to retrain model with optimised params |           |


10. TESTING STRATEGY

- Unit Tests: Data loading functions, label mapping, graph construction.
- Integration Tests: Agent tools tested via scripts/test_agent.py with 6 standard questions.
- Manual QA: All 14 dashboard tabs tested on Python 3.12, macOS + MPS.
- Phase 6 QA: All 5 analytics modules verified with synthetic data; fallback paths tested
  (Holt-Winters triggered when Prophet not installed; label-proxy used when model unavailable).
- RAG Testing: 6 preset questions verified to return relevant source documents.
- API Testing: All 8 endpoints tested via Swagger UI at /docs.
- Docker Testing: Both containers build and run locally before pushing to Render.
- Memory Testing: MPS environment variables verified to prevent out-of-memory errors.
- Business Translation QA: All 14 Business Question panels and 5 Plain English callouts
  verified to display correct dynamic values after computation.


11. RISKS AND MITIGATIONS

| Risk                              | Mitigation                                                  | Status   |
|-----------------------------------|-------------------------------------------------------------|----------|
| MPS memory exhaustion (M1/M2)     | PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0; CPU fallback          | Resolved |
| LLM hallucination in agent        | Fallback mode returns direct tool output; no synthesis      | Resolved |
| OpenAI API unavailable or no key  | All LLM features degrade gracefully to rule-based mode      | Resolved |
| ChromaDB not installed            | FraudRAGAgent falls back to sklearn TF-IDF automatically    | Resolved |
| Class imbalance (illicit >> licit)| Balanced class sampling in training; AUC-ROC evaluation     | Resolved |
| Render free-tier memory (512 MB)  | 2,000-node training sample; lazy model loading              | Resolved |
| Slow dashboard first load         | @st.cache_resource on data loading and model training       | Resolved |
| Emoji in source files             | Full audit pass; Material icons + CSS dots in dashboard;    | Resolved |
|                                   | plain-text logging in all scripts                           |          |
| Large knowledge base cost         | 25 docs at ~150 tokens = ~$0.01 one-time embedding cost     | Resolved |
| Raw CSV too large for GitHub      | zstd parquet compression (690 MB -> 87 MB); no LFS needed   | Resolved |
| Dockerfile COPY failure on Render | Removed explicit raw CSV COPY; parquet bundled via COPY .   | Resolved |
| Prophet not installed on Render   | Holt-Winters fallback implemented; PROPHET_AVAILABLE flag   | Resolved |
| Non-technical users can't act     | Business Translation Layer: green Q&A + amber plain-English | Resolved |
|   on technical outputs            | callouts on all 14 tabs; help= tooltips on all jargon inputs|          |
| Sharpe == Sortino always          | Sortino denominator was std(1-r)=std(r); fixed with proper  | Resolved |
|                                   | semi-deviation: RMS of returns below risk-free threshold    |          |
| Forecast dates showed only 2011   | Used weekly step; corrected to bi-weekly (2×Timedelta) so  | Resolved |
|                                   | 49 steps span Jan 2011 – Jan 2013 correctly                 |          |
| Regulatory Capital crash          | value=n_nodes×10,000 exceeded max_value=1B; raised to 10B  | Resolved |
|   (StreamlitValueAboveMaxError)   | and capped default with min()                               |          |
| KDE fillcolor ValueError          | Plotly 6.x rejects 8-char hex (#RRGGBBAA); added _to_rgba()| Resolved |
|   (#RRGGBBAA invalid in Plotly)   | helper converting any hex/rgb to rgba(r,g,b,a)              |          |
| Knowledge Base presets show nothing| Buttons set state but didn't trigger search; added         | Resolved |
|                                   | rag_do_search flag + st.rerun() to auto-execute search      |          |
| Explainability margin kwarg clash | _dark_layout() and caller both passed margin= to            | Resolved |
|                                   | update_layout(); moved override inside _dark_layout() call  |          |
| Sidebar numbers truncated         | st.metric() in narrow columns clips "203,769" to "203,…";  | Resolved |
|                                   | replaced with 2×2 HTML grid cards showing full numbers      |          |
| AI Chat ugly pre-formatted block  | Custom <pre> tag forced monospace one-line wall; switched   | Resolved |
|                                   | to st.chat_message() + _render_agent_response() metric grid |          |
| git filter-repo wiped working tree| filter-repo removed origin remote and reset working tree to  | Resolved |
|   (un-committed dashboard lost)   | committed state, destroying 2 sessions of un-committed edits.|          |
|                                   | Fix: all dashboard files are now committed after each session |          |


12. SUCCESS CRITERIA

Model performance (all achieved):
- AUC >= 0.95 on held-out test set (achieved: 0.955)
- F1 Score >= 0.88 (achieved: 0.898)
- Accuracy >= 88% (achieved: 89.0%)

Functional requirements (all achieved):
- LangChain agent answers 6 test questions with correct statistics
- Monte Carlo simulation runs in under 10 seconds for 10,000 iterations
- 5 auto-discovery insight types surface without user input
- RAG returns relevant source documents for 5+ question types
- 14-tab Streamlit dashboard loads in under 3 seconds
- FastAPI serves all 8 endpoints with Swagger documentation
- Docker containers deploy to Render and pass health checks
- All 5 Phase 6 analytics modules run end-to-end with graceful fallbacks

Business Translation requirements (all achieved):
- Every tab has a green Business Question panel explaining the content in plain English
- Phase 6 tabs display amber Plain English callouts with dynamic values after computation
- All technical parameter inputs (PD, LGD, rho, diffusion steps, infection probability)
  carry help= tooltip explanations in plain English
- A user with no finance background can read the result and know what action to take

Code quality requirements (all achieved):
- No emoji in any Python source file
- All LLM, ChromaDB, and Prophet features have working offline fallbacks
- Golden ratio column layouts (1:1.618) applied throughout dashboard
- Fibonacci-scale spacing (8/13/21/34/55 px) in CSS
- Dual context panels (technical + business) on each dashboard tab


13. KEY INSIGHTS FROM DATA

- Total Transactions: 203,769
- Fraud Rate: 90.2% of labeled transactions are illicit
- Network Structure: Sparse directed graph, average degree 2.3, 49 weakly connected components
- Top Hub Node: Node 2984918, degree 473 (205 standard deviations above mean)
- High-Velocity Nodes: 13,752 nodes exceed the 90th percentile degree threshold
- Time Coverage: 49 bi-weekly snapshots (Jan 2011 – Jan 2013, approximately)
- Most Predictive Features: Transaction Amount, Fee Ratio, Input Count, Betweenness Centrality
- Class Imbalance: Illicit 21%, Licit 2%, Unknown 77% of all nodes


14. APPENDIX

- Data Source: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- GitHub Repository: https://github.com/sabrinapribadi/graph-fraud-detector
- Live Dashboard: https://graph-fraud-detector-dashboards.onrender.com
- API Documentation: https://graph-fraud-detector-api-service.onrender.com/docs
- Tech Stack: PyTorch, torch-geometric, NetworkX, Streamlit, FastAPI, LangChain, LangGraph,
  ChromaDB, OpenAI GPT-4o-mini, text-embedding-3-small, Optuna, Prophet, SciPy, Docker, Render
- GNN Models: GraphSAGE, GAT, EnsembleFraudDetector
- Core Quant Finance: Monte Carlo simulation, Expected Loss (PD × EAD × LGD), VaR (95%), TVM
- Phase 6 Quant Finance: Stress Testing (5 scenarios), Sharpe/Sortino/IR/Calmar, Prophet/Holt-
  Winters loss forecast, Basel III SA + IRB Vasicek regulatory capital, SIR contagion score
- RAG Stack: ChromaDB cosine similarity, text-embedding-3-small 256-dim, 25 knowledge docs
- Agent Tools: get_fraud_stats, find_suspicious_nodes, analyze_network, predict_transaction,
  run_risk_analysis, get_anomalous_patterns
- Discovery Methods: Money Laundering Rings, Structuring, Rapid Chains, Mixed Signals, Outliers
- Business Translation: _biz_box() green panels + _plain_english() amber callouts on all 14 tabs
