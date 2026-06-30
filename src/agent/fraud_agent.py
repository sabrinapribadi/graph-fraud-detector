"""
LLM Agent for Natural Language Fraud Analysis
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import re
from typing import Dict, Any, List, Optional
import numpy as np
import networkx as nx
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# Local imports
from src.data.loader import EllipticDataLoader
from src.models.gnn_model import FraudDetector
from src.analytics.risk_analysis import QuantitativeRiskAnalyzer

load_dotenv(override=True)

# Follow-up suggestions keyed by which tool branch was used
_FOLLOWUPS: Dict[str, List[str]] = {
    "stats": [
        "What are the top 10 suspicious transactions?",
        "Analyze the network structure",
        "Run a risk analysis for 5000 transactions",
    ],
    "suspicious": [
        "Show fraud statistics",
        "Predict fraud probability for node 5530458",
        "What anomalous patterns exist in the network?",
    ],
    "network": [
        "Analyze fraud trends over time",
        "Find anomalous patterns",
        "Show fraud statistics",
    ],
    "predict": [
        "Explain the features that flagged this node",
        "Find the top 10 suspicious transactions",
        "Run a risk analysis",
    ],
    "risk": [
        "Forecast fraud losses for the next 10 periods",
        "Show fraud statistics",
        "What are the top suspicious transactions?",
    ],
    "anomaly": [
        "Analyze fraud trends over time",
        "Show fraud statistics",
        "Run a risk analysis",
    ],
    "temporal": [
        "Run a risk analysis",
        "Forecast fraud losses for the next 10 periods",
        "Find anomalous patterns",
    ],
    "forecast": [
        "Run a risk analysis for 10000 transactions with 2% fraud rate",
        "Show fraud statistics",
        "Analyze fraud trends over time",
    ],
    "explain": [
        "Find the top 10 suspicious transactions",
        "Show fraud statistics",
        "Predict fraud probability for another node",
    ],
}


def _append_followups(response: str, key: str) -> str:
    suggestions = _FOLLOWUPS.get(key, [])
    if not suggestions:
        return response
    lines = "\n".join(f"- {q}" for q in suggestions)
    return f"{response}\n\n---\n**Related questions you might ask:**\n{lines}"


class FraudAgent:
    """
    LLM-powered agent for fraud detection
    """
    def __init__(self, G: nx.DiGraph, detector: FraudDetector, data: Dict[str, Any]):
        self.G = G
        self.detector = detector
        self.data = data
        self.analyzer = QuantitativeRiskAnalyzer()
        self.tools = []
        self.conversation_history: List = []  # accumulated HumanMessage/AIMessage for LLM path

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "sk-your-actual-key-here" or api_key == "sk-your-actual-key-here...":
            print("WARNING: Valid OPENAI_API_KEY not found in environment variables.")
            print("Running in fallback mode - will return direct tool responses.")
            self.mock_mode = True
            self.agent = None
            self.llm = None
        else:
            self.mock_mode = False
            print("OpenAI API key loaded.")

            self.llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
                api_key=api_key
            )

            self.tools = self._create_tools()

            try:
                self.agent = create_agent(
                    model=self.llm,
                    tools=self.tools,
                    system_prompt="""You are a specialized fraud detection analyst for Bitcoin transaction networks.
                    You have access to a graph neural network model that can detect fraudulent transactions.

                    When answering questions:
                    1. Be precise with numbers - use the data from the tools
                    2. Provide actionable insights
                    3. If you don't know something, say so
                    4. Use a professional, analytical tone
                    5. Explain technical terms in plain language

                    Your goal is to help users understand fraud patterns and make data-driven decisions.
                    """
                )
                print("Agent initialized successfully.")
            except Exception as e:
                print(f"Agent initialization failed: {e}")
                print("Running in fallback mode")
                self.mock_mode = True
                self.agent = None

        if not self.tools:
            self.tools = self._create_tools()

    # ── Tool registry ─────────────────────────────────────────────────────────

    def _create_tools(self):
        """Create @tool wrappers that delegate to the _*_impl methods."""

        @tool
        def get_fraud_stats(query: str = "") -> str:
            """Get overall fraud statistics including total illicit/licit transactions, class distribution"""
            return self._get_fraud_stats_impl()

        @tool
        def find_suspicious_nodes(n: int = 10) -> str:
            """Find top N most suspicious transactions (nodes) with highest fraud probability"""
            return self._find_suspicious_nodes_impl(n=n)

        @tool
        def analyze_network(query: str = "") -> str:
            """Analyze network structure including degree distribution, components, and connectivity"""
            return self._analyze_network_impl()

        @tool
        def predict_transaction(node_id: str) -> str:
            """Predict fraud probability for a specific transaction node ID"""
            return self._predict_transaction_impl(node_id=node_id)

        @tool
        def run_risk_analysis(n_transactions: int = 10000, fraud_rate: float = 0.02, avg_loss: float = 5000) -> str:
            """Run Monte Carlo risk analysis to assess fraud exposure"""
            return self._run_risk_analysis_impl(
                n_transactions=n_transactions, fraud_rate=fraud_rate, avg_loss=avg_loss
            )

        @tool
        def get_anomalous_patterns(query: str = "") -> str:
            """Find anomalous patterns in the transaction graph"""
            return self._get_anomalous_patterns_impl()

        @tool
        def analyze_temporal_trends(query: str = "") -> str:
            """Analyze fraud trends, transaction velocity, and temporal anomalies over time"""
            return self._temporal_trend_impl()

        @tool
        def forecast_fraud_losses(n_periods: int = 10) -> str:
            """Forecast projected fraud losses for the next N periods using time-series analysis"""
            return self._budget_forecast_impl(n_periods=n_periods)

        @tool
        def explain_node_features(node_id: str) -> str:
            """Explain why a transaction node was flagged as fraudulent using gradient-based feature importance"""
            return self._feature_explanation_impl(node_id=node_id)

        return [
            get_fraud_stats, find_suspicious_nodes, analyze_network,
            predict_transaction, run_risk_analysis, get_anomalous_patterns,
            analyze_temporal_trends, forecast_fraud_losses, explain_node_features,
        ]

    def _call_tool(self, tool_name: str, **kwargs) -> str:
        tool_map = {
            "get_fraud_stats":        self._get_fraud_stats_impl,
            "find_suspicious_nodes":  self._find_suspicious_nodes_impl,
            "analyze_network":        self._analyze_network_impl,
            "predict_transaction":    self._predict_transaction_impl,
            "run_risk_analysis":      self._run_risk_analysis_impl,
            "get_anomalous_patterns": self._get_anomalous_patterns_impl,
            "analyze_temporal_trends": self._temporal_trend_impl,
            "forecast_fraud_losses":  self._budget_forecast_impl,
            "explain_node_features":  self._feature_explanation_impl,
        }
        if tool_name in tool_map:
            try:
                return tool_map[tool_name](**kwargs)
            except Exception as e:
                return f"Error calling tool {tool_name}: {str(e)}"
        return f"Tool {tool_name} not found."

    # ── Tool implementations ──────────────────────────────────────────────────

    def _get_fraud_stats_impl(self, query: str = "") -> str:
        labels = [data.get('label', -1) for _, data in self.G.nodes(data=True)]
        total = len(labels)
        licit = sum(1 for l in labels if l == 0)
        illicit = sum(1 for l in labels if l == 1)
        unknown = sum(1 for l in labels if l == -1)

        return f"""
FRAUD STATISTICS
================================
Total Transactions: {total:,}
├── Licit: {licit:,} ({licit/total*100:.1f}%)
├── Illicit: {illicit:,} ({illicit/total*100:.1f}%)
└── Unknown: {unknown:,} ({unknown/total*100:.1f}%)

Fraud Rate (of labeled): {(illicit/(licit+illicit)*100):.1f}%
"""

    def _find_suspicious_nodes_impl(self, n: int = 10) -> str:
        n = min(n, 50)

        if not self.data:
            return "No model data available for prediction"

        import torch
        features = self.data['features']
        labels = self.data['labels']
        node_ids = self.data['node_ids']

        x = torch.FloatTensor(features).to(self.detector.device)
        adj = torch.eye(len(features)).to(self.detector.device)

        self.detector.model.eval()
        with torch.no_grad():
            output = self.detector.model(x, adj)
            probs = torch.sigmoid(output).squeeze().cpu().numpy()

        if isinstance(probs, (float, np.float32, np.float64)):
            probs = np.array([probs])

        results = sorted(
            [
                {'node_id': nid, 'fraud_probability': float(p)}
                for nid, lbl, p in zip(node_ids, labels, probs)
                if lbl == 1
            ],
            key=lambda x: x['fraud_probability'],
            reverse=True,
        )[:n]

        output = f"TOP {n} MOST SUSPICIOUS TRANSACTIONS\n" + "=" * 50 + "\n\n"
        if not results:
            return output + "No illicit transactions found in the sampled data."

        for i, result in enumerate(results, 1):
            prob = result['fraud_probability'] * 100
            risk_level = "HIGH" if prob > 80 else "MEDIUM" if prob > 50 else "LOW"
            output += (
                f"{i}. Transaction ID: {result['node_id']}\n"
                f"   Fraud Probability: {prob:.1f}%\n"
                f"   Risk Level: {risk_level}\n\n"
            )
        return output

    def _analyze_network_impl(self, query: str = "") -> str:
        G = self.G
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        degrees = [d for _, d in G.degree()]
        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        components = nx.number_weakly_connected_components(G)
        isolates = nx.number_of_isolates(G)

        return f"""
NETWORK ANALYSIS
================================

Basic Statistics:
├── Total Nodes: {n_nodes:,}
├── Total Edges: {n_edges:,}
├── Network Density: {nx.density(G):.6f}
└── Connected Components: {components:,}

Degree Distribution:
├── Average Degree: {avg_degree:.2f}
├── Maximum Degree: {max_degree}
└── Isolated Nodes: {isolates:,} ({isolates/n_nodes*100:.1f}%)
"""

    def _predict_transaction_impl(self, node_id: str) -> str:
        try:
            node_id = node_id.strip()
            if node_id not in self.G:
                return f"Transaction ID {node_id} not found in the dataset."

            node_data = self.G.nodes[node_id]
            if 'features' not in node_data:
                return f"Node {node_id} has no features."

            import torch
            features = node_data['features']
            label = node_data.get('label', -1)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.detector.device)

            self.detector.model.eval()
            with torch.no_grad():
                out = self.detector.model(features_tensor, torch.eye(1).to(self.detector.device))
                prob = torch.sigmoid(out).squeeze().item()

            actual = "LICIT" if label == 0 else "ILLICIT" if label == 1 else "UNKNOWN"
            risk_level = "HIGH" if prob > 0.8 else "MEDIUM" if prob > 0.5 else "LOW"

            return f"""
TRANSACTION FRAUD ANALYSIS
================================

Transaction ID: {node_id}
Actual Label: {actual}
Fraud Probability: {prob:.1%}
Risk Level: {risk_level}

Recommendation:
{'BLOCK' if prob > 0.8 else 'REVIEW' if prob > 0.5 else 'APPROVE'}
"""
        except Exception as e:
            return f"Error analyzing transaction: {str(e)}"

    def _run_risk_analysis_impl(
        self, n_transactions: int = 10000, fraud_rate: float = 0.02, avg_loss: float = 5000
    ) -> str:
        try:
            results = self.analyzer.full_risk_assessment(
                n_transactions=n_transactions,
                fraud_probability=fraud_rate,
                avg_loss_per_fraud=avg_loss,
                exposure_per_transaction=avg_loss * 0.2,
                detection_time=30,
                n_simulations=10000,
            )
            return f"""
QUANTITATIVE RISK ANALYSIS
================================

Parameters:
├── Transactions Analyzed: {n_transactions:,}
├── Base Fraud Rate: {fraud_rate*100:.1f}%
└── Average Loss per Fraud: ${avg_loss:,.2f}

Results:
├── Expected Loss: ${results['expected_loss']['expected_loss']:,.2f}
├── Value at Risk (95%): ${results['monte_carlo']['value_at_risk']:,.2f}
├── Cost of Delay (30 days): ${results['tvm_adjusted']['time_value_cost']:,.2f}
└── Total Risk Score: ${results['total_risk_score']:,.2f}
"""
        except Exception as e:
            return f"Error running risk analysis: {str(e)}"

    def _get_anomalous_patterns_impl(self, query: str = "") -> str:
        G = self.G
        degrees = sorted([(n, G.degree(n)) for n in G.nodes()], key=lambda x: x[1], reverse=True)
        top_hubs = degrees[:5]

        patterns = []
        if top_hubs:
            patterns.append("High-Degree Hubs:")
            for node, degree in top_hubs:
                label = G.nodes[node].get('label', -1)
                label_str = "Illicit" if label == 1 else "Licit" if label == 0 else "Unknown"
                patterns.append(f"   - Node {node}: Degree {degree} ({label_str})")

        output = "ANOMALOUS PATTERNS DETECTED\n" + "=" * 50 + "\n\n"
        return output + ("\n".join(patterns) if patterns else "No significant anomalous patterns detected.")

    def _temporal_trend_impl(self, query: str = "") -> str:
        try:
            from src.analytics.temporal_analysis import TemporalAnalyzer
            analyzer = TemporalAnalyzer(self.G, self.data)
            return analyzer.generate_temporal_report()
        except Exception as e:
            return f"Error running temporal analysis: {str(e)}"

    def _budget_forecast_impl(self, n_periods: int = 10) -> str:
        try:
            from src.analytics.loss_forecasting import LossForecaster
            forecaster = LossForecaster(self.G, self.detector, self.data)
            result = forecaster.forecast(n_periods=n_periods)
            stats = forecaster.summary_stats(result)

            output = f"""
FRAUD LOSS FORECAST ({result['method']})
================================
Historical Mean Loss:  ${stats['historical_mean']:,.2f}
Historical Peak Loss:  ${stats['historical_peak']:,.2f}
Forecast Mean Loss:    ${stats['forecast_mean']:,.2f}
Forecast Peak Loss:    ${stats['forecast_peak']:,.2f}
Trend vs Historical:   {stats['forecast_vs_hist']:.1%}
Direction:             {stats['trend_direction'].title()}

Next {n_periods} periods:
"""
            for _, row in result['forecast'].iterrows():
                output += (
                    f"  {row['ds'].strftime('%Y-%m-%d')}: ${row['yhat']:,.0f} "
                    f"(range ${row['yhat_lower']:,.0f} – ${row['yhat_upper']:,.0f})\n"
                )
            return output
        except Exception as e:
            return f"Error running loss forecast: {str(e)}"

    def _feature_explanation_impl(self, node_id: str) -> str:
        try:
            node_id = node_id.strip()
            if node_id not in self.G:
                return f"Transaction ID {node_id} not found in the dataset."

            node_data = self.G.nodes[node_id]
            if 'features' not in node_data:
                return f"Node {node_id} has no features."

            import torch
            from src.analytics.model_explainability import ModelExplainer

            features = np.array(node_data['features'])
            x_all = torch.FloatTensor(self.data['features']).to(self.detector.device)
            explainer = ModelExplainer(self.detector.model, x_all)
            exp = explainer._gradient_explanation(features)

            if 'error' in exp:
                return f"Explanation failed for node {node_id}: {exp['error']}"

            top5 = sorted(exp['top_features'], key=lambda f: abs(f['importance']), reverse=True)[:5]
            output = f"""
FEATURE EXPLANATION FOR NODE {node_id}
=========================================
Fraud Probability: {exp['prediction']:.1%}

Top 5 Most Influential Features:
"""
            for i, feat in enumerate(top5, 1):
                direction = "increases" if feat['importance'] > 0 else "decreases"
                output += (
                    f"  {i}. {feat['name']}: value={feat['value']:.4f} "
                    f"→ {direction} fraud risk (importance: {feat['importance']:+.4f})\n"
                )
            return output
        except Exception as e:
            return f"Error explaining node: {str(e)}"

    # ── Public interface ──────────────────────────────────────────────────────

    def ask(self, question: str) -> str:
        """Ask a question to the agent"""
        if self.mock_mode or self.agent is None:
            return self._ask_mock(question)

        try:
            self.conversation_history.append(HumanMessage(content=question))
            response = self.agent.invoke({"messages": self.conversation_history})
            answer = response['messages'][-1].content
            self.conversation_history.append(AIMessage(content=answer))
            return answer
        except Exception as e:
            return f"Error processing question: {str(e)}"

    def _ask_mock(self, question: str) -> str:
        """Keyword-based dispatcher used when no OpenAI key is available."""
        q = question.lower()

        if "temporal" in q or "trend" in q or "velocity" in q or "time" in q:
            return _append_followups(self._temporal_trend_impl(), "temporal")

        if "forecast" in q or "budget" in q or "future" in q or "project" in q:
            numbers = re.findall(r'\d+', question)
            n_periods = int(numbers[0]) if numbers else 10
            return _append_followups(self._budget_forecast_impl(n_periods=n_periods), "forecast")

        if "explain" in q or "why" in q or "feature" in q or "importan" in q:
            node_match = re.search(r'\d+', question)
            node_id = node_match.group() if node_match else ""
            if not node_id:
                return "Please specify a node ID. Example: \"Explain why node 5530458 was flagged\""
            return _append_followups(self._feature_explanation_impl(node_id=node_id), "explain")

        if "risk" in q or "monte" in q:
            numbers = re.findall(r'\d+', question)
            if len(numbers) >= 3:
                n_tx, fraud_rate, avg_loss = int(numbers[0]), float(numbers[1]) / 100, float(numbers[2])
            elif len(numbers) == 2:
                n_tx, fraud_rate, avg_loss = int(numbers[0]), float(numbers[1]) / 100, 5000
            elif len(numbers) == 1:
                n_tx, fraud_rate, avg_loss = int(numbers[0]), 0.02, 5000
            else:
                n_tx, fraud_rate, avg_loss = 10000, 0.02, 5000
            return _append_followups(
                self._run_risk_analysis_impl(n_transactions=n_tx, fraud_rate=fraud_rate, avg_loss=avg_loss),
                "risk",
            )

        if "statistic" in q or ("fraud" in q and "stat" in q):
            return _append_followups(self._get_fraud_stats_impl(), "stats")

        if "suspicious" in q or "top" in q:
            numbers = re.findall(r'\d+', question)
            n = int(numbers[0]) if numbers else 10
            return _append_followups(self._find_suspicious_nodes_impl(n=n), "suspicious")

        if "network" in q or "structure" in q:
            return _append_followups(self._analyze_network_impl(), "network")

        if "predict" in q or "transaction" in q:
            node_match = re.search(r'\d+', question)
            node_id = node_match.group() if node_match else "5530458"
            return _append_followups(self._predict_transaction_impl(node_id=node_id), "predict")

        if "anomal" in q or "pattern" in q:
            return _append_followups(self._get_anomalous_patterns_impl(), "anomaly")

        return (
            "I can help with fraud analysis. Try asking about:\n"
            "- Fraud statistics\n"
            "- Suspicious transactions\n"
            "- Network structure\n"
            "- Transaction predictions\n"
            "- Risk analysis\n"
            "- Anomalous patterns\n"
            "- Fraud trends over time\n"
            "- Loss forecasts\n"
            "- Feature explanations for a specific node"
        )
