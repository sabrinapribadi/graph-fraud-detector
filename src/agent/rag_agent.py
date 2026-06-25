"""
RAG Agent for Fraud Detection Knowledge Base

Provides semantic search over a curated knowledge base of Bitcoin fraud patterns,
risk analysis concepts, and GNN model explanations using ChromaDB + OpenAI embeddings.
Falls back to TF-IDF search when ChromaDB or an API key is not available.
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    import chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_AVAILABLE = bool(OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-your-"))

CHROMA_PATH = project_root / "data" / "chroma_fraud"
COLLECTION_NAME = "fraud_knowledge"

# ── Knowledge Base ────────────────────────────────────────────────────────────
FRAUD_KNOWLEDGE_BASE: List[Dict[str, str]] = [
    {
        "id": "kb_001",
        "text": (
            "Money laundering through Bitcoin typically occurs in three stages: "
            "placement (introducing illicit funds into the network), layering (obscuring the "
            "origin through multiple transactions), and integration (reintroducing funds as "
            "seemingly legitimate assets). In the Elliptic dataset, money laundering rings "
            "appear as high-degree nodes with more than 50% illicit connections. These hubs "
            "aggregate transactions from many sources before distributing to fewer outputs, "
            "creating a characteristic fan-in fan-out pattern visible in the transaction graph."
        ),
        "category": "fraud_pattern",
        "source": "Money Laundering Typology",
    },
    {
        "id": "kb_002",
        "text": (
            "Structuring (also called smurfing) breaks large illicit transactions into many "
            "smaller amounts to avoid detection thresholds. In Bitcoin networks, structuring "
            "appears as transaction clusters with amounts just below round-number thresholds. "
            "The Elliptic dataset captures structuring as nodes whose normalised transaction-"
            "amount features are concentrated in narrow quantile bands just below threshold "
            "boundaries. Structuring is classified as HIGH severity because it directly "
            "indicates intentional evasion of compliance controls."
        ),
        "category": "fraud_pattern",
        "source": "Structuring Pattern",
    },
    {
        "id": "kb_003",
        "text": (
            "Rapid transaction chains involve a high volume of transactions executed in quick "
            "succession, often to obscure the origin of funds. In the transaction graph these "
            "chains appear as long directed paths where each node has a high out-degree to "
            "in-degree ratio. Nodes with degree above the 90th percentile (roughly 13,752 "
            "nodes in the Elliptic dataset) and at least one illicit connection are flagged as "
            "high-velocity anomalies. Detection relies on transaction velocity features "
            "and temporal proximity metrics."
        ),
        "category": "fraud_pattern",
        "source": "Velocity Analysis",
    },
    {
        "id": "kb_004",
        "text": (
            "Mixed signal nodes are transactions connected to both licit and illicit counterparties. "
            "They represent potential intermediaries or exchanges that unknowingly process "
            "fraudulent funds. In the Elliptic dataset, approximately 4-8% of labeled nodes "
            "have at least one illicit neighbour despite being labeled licit. These nodes "
            "require manual review because the GNN model may assign them moderate fraud "
            "probabilities (0.3-0.7) reflecting genuine uncertainty."
        ),
        "category": "fraud_pattern",
        "source": "Mixed Signal Analysis",
    },
    {
        "id": "kb_005",
        "text": (
            "Anomaly outliers are nodes whose degree deviates more than 2.5 standard deviations "
            "from the mean degree in the transaction graph. The Elliptic dataset has a mean "
            "degree of approximately 2.3 with high variance due to hub nodes. Outlier hubs "
            "can represent exchanges, mixers, or darknet markets. The top hub node in the "
            "dataset (node 2984918) has a degree of 473, making it 205 standard deviations "
            "above the mean — a clear candidate for investigation."
        ),
        "category": "fraud_pattern",
        "source": "Anomaly Detection",
    },
    {
        "id": "kb_006",
        "text": (
            "GraphSAGE (Graph Sample and Aggregate) is the primary GNN architecture used in "
            "this fraud detector. It learns node representations by sampling and aggregating "
            "features from a fixed-size neighbourhood. Unlike traditional GNNs that require "
            "the full graph during training, GraphSAGE uses mini-batch training with "
            "neighbourhood sampling, making it scalable to the 203,769-node Elliptic graph. "
            "The model uses two layers, 32 hidden dimensions, mean aggregation, batch "
            "normalisation, and dropout (0.2) to achieve AUC 0.955 and F1 0.898."
        ),
        "category": "model",
        "source": "GNN Architecture",
    },
    {
        "id": "kb_007",
        "text": (
            "Graph Attention Networks (GAT) extend GraphSAGE by learning attention weights "
            "over neighbours rather than using a fixed aggregation function. Multi-head "
            "attention allows the model to attend to different aspects of a node's "
            "neighbourhood simultaneously. In the ensemble model, GAT is combined with two "
            "GraphSAGE variants (mean and sum aggregation) using majority voting, which "
            "reduces variance and improves robustness on imbalanced classes like illicit "
            "Bitcoin transactions."
        ),
        "category": "model",
        "source": "GAT Architecture",
    },
    {
        "id": "kb_008",
        "text": (
            "The Elliptic dataset contains 203,769 Bitcoin transaction nodes and 234,355 "
            "directed edges representing BTC flows. Nodes are labeled as illicit (class 1, "
            "42,019 nodes, ~21%), licit (class 0, 4,545 nodes, ~2%), or unknown (~77%). "
            "Each node has 166 features: the first 94 are local transaction features "
            "(amounts, fees, input/output counts) and the remaining 72 are aggregated "
            "neighbourhood features. The dataset covers 49 time steps from 2011 to 2014. "
            "It is publicly available on Kaggle from the Elliptic company."
        ),
        "category": "dataset",
        "source": "Elliptic Dataset Overview",
    },
    {
        "id": "kb_009",
        "text": (
            "Expected Loss (EL) in fraud risk quantification is calculated as: "
            "EL = PD x EAD x LGD, where PD is the Probability of Default (fraud occurrence "
            "rate), EAD is the Exposure at Default (average transaction value at risk), and "
            "LGD is the Loss Given Default (fraction of exposure lost once fraud is confirmed, "
            "typically 80-100% for Bitcoin fraud). For a portfolio of 10,000 transactions "
            "with 2% fraud rate, $5,000 average loss, and 80% LGD, the expected loss is "
            "approximately $800,000."
        ),
        "category": "risk",
        "source": "Expected Loss Formula",
    },
    {
        "id": "kb_010",
        "text": (
            "Value at Risk (VaR) is the maximum expected loss at a given confidence level "
            "over a specified time period. A 95% VaR means that in 95% of simulated scenarios "
            "the actual loss will not exceed the VaR amount. Monte Carlo simulation generates "
            "10,000 random fraud scenarios using a binomial distribution for fraud count and "
            "a normal distribution (20% std) for loss variation. The 95th percentile of the "
            "resulting loss distribution is the VaR. Expected Shortfall (CVaR) — the average "
            "loss in the worst 5% of scenarios — is a more conservative risk measure."
        ),
        "category": "risk",
        "source": "Value at Risk",
    },
    {
        "id": "kb_011",
        "text": (
            "Time Value of Money (TVM) adjustments account for the fact that fraud losses "
            "detected later are more costly due to opportunity cost. If fraud is detected "
            "after a delay of d days at a discount rate r, the time-adjusted loss is "
            "L x (1 + r)^(d/365). For a 30-day detection delay at a 10% annual discount "
            "rate, losses are approximately 0.8% higher than immediate detection. This "
            "incentivises investment in faster detection pipelines."
        ),
        "category": "risk",
        "source": "Time Value of Money",
    },
    {
        "id": "kb_012",
        "text": (
            "Betweenness centrality measures how often a node lies on the shortest path "
            "between other node pairs. High betweenness in a Bitcoin transaction graph "
            "indicates a node that bridges many transaction flows — typically an exchange, "
            "mixer, or relay service. Illicit nodes with high betweenness are particularly "
            "dangerous as they can propagate funds across many subgraphs simultaneously."
        ),
        "category": "network",
        "source": "Betweenness Centrality",
    },
    {
        "id": "kb_013",
        "text": (
            "The clustering coefficient of a node measures the fraction of its neighbours "
            "that are also connected to each other. Low clustering in the Bitcoin fraud "
            "network means illicit nodes tend to form tree-like chains rather than cliques. "
            "Unusually high clustering around a hub can indicate a tightly-knit fraud ring "
            "where members transact with each other to layer funds."
        ),
        "category": "network",
        "source": "Clustering Coefficient",
    },
    {
        "id": "kb_014",
        "text": (
            "PageRank assigns importance to nodes based on the quantity and quality of their "
            "incoming connections. In a Bitcoin transaction graph, a high-PageRank node "
            "receives funds from many well-connected sources. Illicit nodes with high "
            "PageRank are significant targets for investigation because they act as "
            "consolidation points for funds from multiple fraudulent sources."
        ),
        "category": "network",
        "source": "PageRank in Fraud Networks",
    },
    {
        "id": "kb_015",
        "text": (
            "Temporal anomaly detection identifies nodes whose transaction activity deviates "
            "significantly from expected patterns over time. In the Elliptic dataset, each "
            "node has a time step (1-49). Nodes that appear in many consecutive time steps "
            "with increasing degree may represent long-running fraud operations. Z-score "
            "anomaly detection flags nodes whose degree is more than 2.5 standard deviations "
            "above the mean for their time step cohort."
        ),
        "category": "temporal",
        "source": "Temporal Anomaly Detection",
    },
    {
        "id": "kb_016",
        "text": (
            "Bitcoin mixers (tumblers) are services that pool and redistribute Bitcoin to "
            "obscure transaction trails. In the transaction graph, mixers appear as nodes "
            "with high in-degree from many small inputs and matching high out-degree to many "
            "different outputs. The Elliptic dataset includes labeled mixer transactions "
            "as part of the illicit class. Mixers are distinguishable from exchanges by "
            "their lack of KYC and the near-equal values of inputs and outputs."
        ),
        "category": "fraud_pattern",
        "source": "Mixer Detection",
    },
    {
        "id": "kb_017",
        "text": (
            "Hyperparameter optimisation with Optuna uses Tree-structured Parzen Estimators "
            "(TPE) to efficiently search the parameter space. For the fraud detector, Optuna "
            "tunes hidden dimensions (16-128), number of GNN layers (2-4), dropout rate "
            "(0.1-0.5), learning rate, and aggregation function. A MedianPruner stops "
            "unpromising trials early based on AUC at intermediate epochs. Typical best "
            "parameters are hidden_dim=64, num_layers=3, dropout=0.2, lr=0.001."
        ),
        "category": "model",
        "source": "Hyperparameter Optimisation",
    },
    {
        "id": "kb_018",
        "text": (
            "Model explainability for the GNN uses gradient-based feature importance: the "
            "gradient of the output probability with respect to each input feature indicates "
            "how much changing that feature changes the fraud prediction. Features with large "
            "absolute gradients are most influential. For Bitcoin fraud, the most predictive "
            "features are typically transaction amount (feature 0), fee ratio (feature 102), "
            "input count (feature 2), and network centrality measures (features 30-50)."
        ),
        "category": "model",
        "source": "Model Explainability",
    },
    {
        "id": "kb_019",
        "text": (
            "Darknet marketplace transactions constitute a major category of illicit Bitcoin "
            "activity in the Elliptic dataset. These involve purchases of illegal goods where "
            "Bitcoin is used as payment. Transaction signatures include: small amounts sent "
            "to fresh addresses, short time-to-spend intervals, and lack of change outputs. "
            "The GNN detects these because they form recognisable sub-graph patterns that "
            "differ from legitimate e-commerce or peer-to-peer transfers."
        ),
        "category": "fraud_pattern",
        "source": "Darknet Marketplace Patterns",
    },
    {
        "id": "kb_020",
        "text": (
            "The class imbalance in the Elliptic dataset (illicit 21%, licit 2%, unknown 77%) "
            "presents a challenge for model training. The GNN training pipeline addresses "
            "this by using balanced class sampling — equal numbers of illicit and licit nodes "
            "are selected for each training batch. Binary cross-entropy loss with logits is "
            "used rather than softmax, and AUC-ROC is the primary evaluation metric because "
            "it is robust to class imbalance."
        ),
        "category": "model",
        "source": "Class Imbalance Handling",
    },
    {
        "id": "kb_021",
        "text": (
            "The FastAPI REST API exposes 8 endpoints for programmatic access to the fraud "
            "detector: GET /health (service status), GET /stats (dataset statistics), "
            "POST /predict (single transaction fraud probability), GET /network/stats "
            "(network topology metrics), POST /analyze/risk (Monte Carlo risk assessment), "
            "and GET /discover/insights (auto-discovery results). The API uses Pydantic "
            "schemas for request validation and returns JSON responses with fraud probability, "
            "risk level, and supporting statistics."
        ),
        "category": "api",
        "source": "API Overview",
    },
    {
        "id": "kb_022",
        "text": (
            "Fraud detection precision vs recall trade-off: increasing the decision threshold "
            "above 0.5 increases precision (fewer false positives) but reduces recall (more "
            "missed fraud). For high-stakes financial compliance, recall is often prioritised "
            "over precision — it is less costly to investigate a false positive than to miss "
            "an illicit transaction. At threshold 0.5 the model achieves precision 90.6% "
            "and recall 89.0%. Lowering the threshold to 0.3 increases recall to ~95% at "
            "the cost of doubling false positives."
        ),
        "category": "model",
        "source": "Precision-Recall Trade-off",
    },
    {
        "id": "kb_023",
        "text": (
            "Deployment architecture: the dashboard (Streamlit on port 8501) and API "
            "(FastAPI on port 8000) are containerised separately using Docker with Python "
            "3.12-slim base images. Both services are deployed on Render's free tier with "
            "512 MB RAM. Auto-deploy on Git push is configured via render.yaml. Memory "
            "management for Apple Silicon MPS is handled by setting PYTORCH_MPS_HIGH_"
            "WATERMARK_RATIO=0.0. On CPU-only hosts, PyTorch automatically falls back to CPU."
        ),
        "category": "deployment",
        "source": "Deployment Architecture",
    },
    {
        "id": "kb_024",
        "text": (
            "Auto-Discovery runs five independent analysis methods on the transaction graph "
            "without requiring any user query. Method 1 (Money Laundering Rings) identifies "
            "nodes with degree > 10 and more than 50% illicit neighbours. Method 2 "
            "(Structuring) detects near-threshold transaction amounts. Method 3 (Rapid Chains) "
            "flags high-velocity nodes above the 90th percentile degree. Method 4 (Mixed "
            "Signals) finds licit nodes with illicit connections. Method 5 (Outliers) uses "
            "z-score > 2.5 to detect anomalous degree nodes. Each result is packaged as an "
            "Insight object with title, description, severity, and a Plotly chart."
        ),
        "category": "feature",
        "source": "Auto-Discovery Methods",
    },
    {
        "id": "kb_025",
        "text": (
            "Ensemble learning combines predictions from multiple models to improve robustness. "
            "The EnsembleFraudDetector trains three models in parallel: GAT, GraphSAGE with "
            "mean aggregation, and GraphSAGE with sum aggregation. Final predictions use "
            "majority voting — a transaction is labeled illicit only if at least two of the "
            "three models agree. This reduces the impact of any single model's errors and "
            "typically improves AUC by 1-3 percentage points over the base model."
        ),
        "category": "model",
        "source": "Ensemble Model",
    },
]


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    category: str
    source: str
    score: float


class FraudRAGAgent:
    """
    Retrieval-Augmented Generation agent for fraud detection knowledge.

    Supports two retrieval modes:
    - Full mode: ChromaDB vector search + OpenAI GPT-4o-mini synthesis
    - Fallback mode: TF-IDF keyword search + direct retrieval (no API key needed)
    """

    def __init__(self, insights: Optional[List[Any]] = None):
        self.insights = insights or []
        self._collection = None
        self._tfidf = None
        self._tfidf_matrix = None

        if CHROMADB_AVAILABLE and OPENAI_AVAILABLE:
            self._init_chroma()
        else:
            self._init_tfidf()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_chroma(self) -> None:
        try:
            embed_fn = OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-3-small",
                dimensions=256,
            )
            client = chromadb.PersistentClient(path=str(CHROMA_PATH))
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
            if self._collection.count() == 0:
                self._populate_collection()
        except Exception as e:
            print(f"ChromaDB init failed ({e}). Falling back to TF-IDF.")
            self._collection = None
            self._init_tfidf()

    def _populate_collection(self) -> None:
        docs = list(FRAUD_KNOWLEDGE_BASE)
        docs += self._insights_to_docs()
        if not docs:
            return
        self._collection.upsert(
            ids=[d["id"] for d in docs],
            documents=[d["text"] for d in docs],
            metadatas=[{"category": d["category"], "source": d["source"]} for d in docs],
        )

    def _insights_to_docs(self) -> List[Dict[str, str]]:
        result = []
        for i, insight in enumerate(self.insights):
            result.append(
                {
                    "id": f"insight_{i:03d}",
                    "text": f"{insight.title}. {insight.description}",
                    "category": getattr(insight, "category", "discovery"),
                    "source": "Auto-Discovery",
                }
            )
        return result

    def _init_tfidf(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        self._all_docs = list(FRAUD_KNOWLEDGE_BASE) + self._insights_to_docs()
        texts = [d["text"] for d in self._all_docs]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)

    # ── Public API ────────────────────────────────────────────────────────────

    def is_indexed(self) -> bool:
        if self._collection is not None:
            return self._collection.count() > 0
        return self._tfidf_matrix is not None

    def search(self, query: str, n_results: int = 5) -> List[RetrievedDoc]:
        if self._collection is not None:
            return self._search_chroma(query, n_results)
        return self._search_tfidf(query, n_results)

    def answer(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        docs = self.search(question, n_results=n_results)
        if not docs:
            return {
                "answer": "No relevant documents found in the knowledge base.",
                "sources": [],
            }

        if OPENAI_AVAILABLE:
            return self._synthesise_with_llm(question, docs)
        return self._direct_retrieval(docs)

    def update_with_insights(self, insights: List[Any]) -> None:
        self.insights = insights
        if self._collection is not None:
            insight_docs = self._insights_to_docs()
            if insight_docs:
                self._collection.upsert(
                    ids=[d["id"] for d in insight_docs],
                    documents=[d["text"] for d in insight_docs],
                    metadatas=[
                        {"category": d["category"], "source": d["source"]}
                        for d in insight_docs
                    ],
                )
        elif self._tfidf_matrix is not None:
            self._init_tfidf()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _search_chroma(self, query: str, n: int) -> List[RetrievedDoc]:
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n, self._collection.count()),
            )
            docs = []
            for doc_id, text, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                docs.append(
                    RetrievedDoc(
                        doc_id=doc_id,
                        text=text,
                        category=meta.get("category", ""),
                        source=meta.get("source", ""),
                        score=round(1 - dist, 4),
                    )
                )
            return docs
        except Exception as e:
            print(f"ChromaDB search failed: {e}")
            return []

    def _search_tfidf(self, query: str, n: int) -> List[RetrievedDoc]:
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._tfidf_matrix)[0]
        top_idx = np.argsort(scores)[::-1][:n]
        docs = []
        for idx in top_idx:
            if scores[idx] > 0:
                d = self._all_docs[idx]
                docs.append(
                    RetrievedDoc(
                        doc_id=d["id"],
                        text=d["text"],
                        category=d["category"],
                        source=d["source"],
                        score=round(float(scores[idx]), 4),
                    )
                )
        return docs

    def _synthesise_with_llm(
        self, question: str, docs: List[RetrievedDoc]
    ) -> Dict[str, Any]:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)
            context = "\n\n".join(
                f"[Source: {d.source}]\n{d.text}" for d in docs
            )
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0,
                max_tokens=600,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a fraud detection analyst. Answer the user's question "
                            "using only the provided context documents. Be precise and "
                            "professional. If the context does not contain the answer, "
                            "say so clearly."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{context}\n\nQuestion: {question}"
                        ),
                    },
                ],
            )
            return {
                "answer": response.choices[0].message.content,
                "sources": docs,
            }
        except Exception as e:
            return self._direct_retrieval(docs, error=str(e))

    def _direct_retrieval(
        self, docs: List[RetrievedDoc], error: str = ""
    ) -> Dict[str, Any]:
        top = docs[0] if docs else None
        answer = (
            f"Top match from knowledge base ({top.source}):\n\n{top.text}"
            if top
            else "No relevant documents found."
        )
        if error:
            answer = f"[LLM unavailable: {error}]\n\n{answer}"
        return {"answer": answer, "sources": docs}
