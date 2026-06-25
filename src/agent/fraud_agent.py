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
from langchain_core.messages import HumanMessage

# Local imports
from src.data.loader import EllipticDataLoader
from src.models.gnn_model import FraudDetector
from src.analytics.risk_analysis import QuantitativeRiskAnalyzer

load_dotenv(override=True)

class FraudAgent:
    """
    LLM-powered agent for fraud detection
    """
    def __init__(self, G: nx.DiGraph, detector: FraudDetector, data: Dict[str, Any]):
        self.G = G
        self.detector = detector
        self.data = data
        self.analyzer = QuantitativeRiskAnalyzer()
        self.tools = []  # Initialize empty tools list
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "sk-your-actual-key-here" or api_key == "sk-your-actual-key-here...":
            print("⚠️  WARNING: Valid OPENAI_API_KEY not found in environment variables.")
            print("Running in fallback mode - will return direct tool responses.")
            self.mock_mode = True
            self.agent = None
            self.llm = None
        else:
            self.mock_mode = False
            print(f"✅ OpenAI API key loaded")
            
            # Initialize LLM
            self.llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
                api_key=api_key
            )
            
            # Create tools
            self.tools = self._create_tools()
            
            # Create agent using create_agent
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
                print("✅ Agent initialized successfully!")
            except Exception as e:
                print(f"⚠️  Agent initialization failed: {e}")
                print("Running in fallback mode")
                self.mock_mode = True
                self.agent = None
        
        # Always create tools for fallback mode
        if not self.tools:
            self.tools = self._create_tools()
    
    def _create_tools(self):
        """Create tools using the @tool decorator"""
        
        @tool
        def get_fraud_stats(query: str = "") -> str:
            """Get overall fraud statistics including total illicit/licit transactions, class distribution"""
            labels = [data.get('label', -1) for _, data in self.G.nodes(data=True)]
            total = len(labels)
            licit = sum(1 for l in labels if l == 0)
            illicit = sum(1 for l in labels if l == 1)
            unknown = sum(1 for l in labels if l == -1)
            
            output = f"""
📊 FRAUD STATISTICS
================================
Total Transactions: {total:,}
├── Licit: {licit:,} ({licit/total*100:.1f}%)
├── Illicit: {illicit:,} ({illicit/total*100:.1f}%)
└── Unknown: {unknown:,} ({unknown/total*100:.1f}%)

Fraud Rate (of labeled): {(illicit/(licit+illicit)*100):.1f}%
"""
            return output
        
        @tool
        def find_suspicious_nodes(n: int = 10) -> str:
            """Find top N most suspicious transactions (nodes) with highest fraud probability"""
            n = min(n, 50)
            
            if self.data:
                data = self.data
                features = data['features']
                labels = data['labels']
                node_ids = data['node_ids']
                
                import torch
                x = torch.FloatTensor(features).to(self.detector.device)
                adj = torch.eye(len(features)).to(self.detector.device)
                
                self.detector.model.eval()
                with torch.no_grad():
                    output = self.detector.model(x, adj)
                    probs = torch.sigmoid(output).squeeze().cpu().numpy()
                
                if isinstance(probs, (float, np.float32, np.float64)):
                    probs = np.array([probs])
                
                results = []
                for i, (node_id, label, prob) in enumerate(zip(node_ids, labels, probs)):
                    if label == 1:
                        results.append({
                            'node_id': node_id,
                            'fraud_probability': float(prob),
                        })
                
                results.sort(key=lambda x: x['fraud_probability'], reverse=True)
                top_results = results[:n]
                
                output = f"🔍 TOP {n} MOST SUSPICIOUS TRANSACTIONS\n"
                output += "=" * 50 + "\n\n"
                
                if not top_results:
                    output += "No illicit transactions found in the sampled data."
                else:
                    for i, result in enumerate(top_results, 1):
                        prob = result['fraud_probability'] * 100
                        risk_level = "🔴 HIGH" if prob > 80 else "🟡 MEDIUM" if prob > 50 else "🟢 LOW"
                        output += f"{i}. Transaction ID: {result['node_id']}\n"
                        output += f"   Fraud Probability: {prob:.1f}%\n"
                        output += f"   Risk Level: {risk_level}\n\n"
                
                return output
            else:
                return "No model data available for prediction"
        
        @tool
        def analyze_network(query: str = "") -> str:
            """Analyze network structure including degree distribution, components, and connectivity"""
            G = self.G
            
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            degrees = [d for n, d in G.degree()]
            avg_degree = np.mean(degrees) if degrees else 0
            max_degree = max(degrees) if degrees else 0
            components = nx.number_weakly_connected_components(G)
            isolates = nx.number_of_isolates(G)
            
            output = f"""
🌐 NETWORK ANALYSIS
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
            return output
        
        @tool
        def predict_transaction(node_id: str) -> str:
            """Predict fraud probability for a specific transaction node ID"""
            try:
                node_id = node_id.strip()
                
                if node_id not in self.G:
                    return f"❌ Transaction ID {node_id} not found in the dataset"
                
                node_data = self.G.nodes[node_id]
                if 'features' not in node_data:
                    return f"❌ Node {node_id} has no features"
                
                features = node_data['features']
                label = node_data.get('label', -1)
                
                import torch
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.detector.device)
                
                self.detector.model.eval()
                with torch.no_grad():
                    output = self.detector.model(features_tensor, torch.eye(1).to(self.detector.device))
                    prob = torch.sigmoid(output).squeeze().item()
                
                actual = "LICIT" if label == 0 else "ILLICIT" if label == 1 else "UNKNOWN"
                risk_level = "🔴 HIGH" if prob > 0.8 else "🟡 MEDIUM" if prob > 0.5 else "🟢 LOW"
                
                output = f"""
🎯 TRANSACTION FRAUD ANALYSIS
================================

Transaction ID: {node_id}
Actual Label: {actual}
Fraud Probability: {prob:.1%}
Risk Level: {risk_level}

Recommendation:
{'🚨 BLOCK' if prob > 0.8 else '⚠️ REVIEW' if prob > 0.5 else '✅ APPROVE'}
"""
                return output
            
            except Exception as e:
                return f"❌ Error analyzing transaction: {str(e)}"
        
        @tool
        def run_risk_analysis(n_transactions: int = 10000, fraud_rate: float = 0.02, avg_loss: float = 5000) -> str:
            """Run Monte Carlo risk analysis to assess fraud exposure"""
            try:
                results = self.analyzer.full_risk_assessment(
                    n_transactions=n_transactions,
                    fraud_probability=fraud_rate,
                    avg_loss_per_fraud=avg_loss,
                    exposure_per_transaction=avg_loss * 0.2,
                    detection_time=30,
                    n_simulations=10000
                )
                
                output = f"""
💰 QUANTITATIVE RISK ANALYSIS
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
                return output
            
            except Exception as e:
                return f"❌ Error running risk analysis: {str(e)}"
        
        @tool
        def get_anomalous_patterns(query: str = "") -> str:
            """Find anomalous patterns in the transaction graph"""
            G = self.G
            
            patterns = []
            
            degrees = [(n, G.degree(n)) for n in G.nodes()]
            degrees.sort(key=lambda x: x[1], reverse=True)
            top_hubs = degrees[:5]
            
            if top_hubs:
                patterns.append("🏢 High-Degree Hubs:")
                for node, degree in top_hubs:
                    label = G.nodes[node].get('label', -1)
                    label_str = "🔴 Illicit" if label == 1 else "🟢 Licit" if label == 0 else "❓ Unknown"
                    patterns.append(f"   - Node {node}: Degree {degree} ({label_str})")
            
            output = "🔍 ANOMALOUS PATTERNS DETECTED\n"
            output += "=" * 50 + "\n\n"
            output += "\n".join(patterns) if patterns else "No significant anomalous patterns detected."
            
            return output
        
        return [get_fraud_stats, find_suspicious_nodes, analyze_network, predict_transaction, run_risk_analysis, get_anomalous_patterns]
    
    def _call_tool(self, tool_name: str, **kwargs) -> str:
        """Helper to call a tool directly by name"""
        # Map tool names to their implementations
        tool_map = {
            "get_fraud_stats": self._get_fraud_stats_impl,
            "find_suspicious_nodes": self._find_suspicious_nodes_impl,
            "analyze_network": self._analyze_network_impl,
            "predict_transaction": self._predict_transaction_impl,
            "run_risk_analysis": self._run_risk_analysis_impl,
            "get_anomalous_patterns": self._get_anomalous_patterns_impl
        }
        
        if tool_name in tool_map:
            try:
                return tool_map[tool_name](**kwargs)
            except Exception as e:
                return f"❌ Error calling tool {tool_name}: {str(e)}"
        return f"❌ Tool {tool_name} not found"
    
    # Tool implementations for fallback mode
    def _get_fraud_stats_impl(self, query: str = "") -> str:
        """Get fraud statistics implementation"""
        labels = [data.get('label', -1) for _, data in self.G.nodes(data=True)]
        total = len(labels)
        licit = sum(1 for l in labels if l == 0)
        illicit = sum(1 for l in labels if l == 1)
        unknown = sum(1 for l in labels if l == -1)
        
        output = f"""
📊 FRAUD STATISTICS
================================
Total Transactions: {total:,}
├── Licit: {licit:,} ({licit/total*100:.1f}%)
├── Illicit: {illicit:,} ({illicit/total*100:.1f}%)
└── Unknown: {unknown:,} ({unknown/total*100:.1f}%)

Fraud Rate (of labeled): {(illicit/(licit+illicit)*100):.1f}%
"""
        return output
    
    def _find_suspicious_nodes_impl(self, n: int = 10) -> str:
        """Find suspicious nodes implementation"""
        n = min(n, 50)
        
        if self.data:
            data = self.data
            features = data['features']
            labels = data['labels']
            node_ids = data['node_ids']
            
            import torch
            x = torch.FloatTensor(features).to(self.detector.device)
            adj = torch.eye(len(features)).to(self.detector.device)
            
            self.detector.model.eval()
            with torch.no_grad():
                output = self.detector.model(x, adj)
                probs = torch.sigmoid(output).squeeze().cpu().numpy()
            
            if isinstance(probs, (float, np.float32, np.float64)):
                probs = np.array([probs])
            
            results = []
            for i, (node_id, label, prob) in enumerate(zip(node_ids, labels, probs)):
                if label == 1:
                    results.append({
                        'node_id': node_id,
                        'fraud_probability': float(prob),
                    })
            
            results.sort(key=lambda x: x['fraud_probability'], reverse=True)
            top_results = results[:n]
            
            output = f"🔍 TOP {n} MOST SUSPICIOUS TRANSACTIONS\n"
            output += "=" * 50 + "\n\n"
            
            if not top_results:
                output += "No illicit transactions found in the sampled data."
            else:
                for i, result in enumerate(top_results, 1):
                    prob = result['fraud_probability'] * 100
                    risk_level = "🔴 HIGH" if prob > 80 else "🟡 MEDIUM" if prob > 50 else "🟢 LOW"
                    output += f"{i}. Transaction ID: {result['node_id']}\n"
                    output += f"   Fraud Probability: {prob:.1f}%\n"
                    output += f"   Risk Level: {risk_level}\n\n"
            
            return output
        else:
            return "No model data available for prediction"
    
    def _analyze_network_impl(self, query: str = "") -> str:
        """Analyze network implementation"""
        G = self.G
        
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        degrees = [d for n, d in G.degree()]
        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        components = nx.number_weakly_connected_components(G)
        isolates = nx.number_of_isolates(G)
        
        output = f"""
🌐 NETWORK ANALYSIS
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
        return output
    
    def _predict_transaction_impl(self, node_id: str) -> str:
        """Predict transaction implementation"""
        try:
            node_id = node_id.strip()
            
            if node_id not in self.G:
                return f"❌ Transaction ID {node_id} not found in the dataset"
            
            node_data = self.G.nodes[node_id]
            if 'features' not in node_data:
                return f"❌ Node {node_id} has no features"
            
            features = node_data['features']
            label = node_data.get('label', -1)
            
            import torch
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.detector.device)
            
            self.detector.model.eval()
            with torch.no_grad():
                output = self.detector.model(features_tensor, torch.eye(1).to(self.detector.device))
                prob = torch.sigmoid(output).squeeze().item()
            
            actual = "LICIT" if label == 0 else "ILLICIT" if label == 1 else "UNKNOWN"
            risk_level = "🔴 HIGH" if prob > 0.8 else "🟡 MEDIUM" if prob > 0.5 else "🟢 LOW"
            
            output = f"""
🎯 TRANSACTION FRAUD ANALYSIS
================================

Transaction ID: {node_id}
Actual Label: {actual}
Fraud Probability: {prob:.1%}
Risk Level: {risk_level}

Recommendation:
{'🚨 BLOCK' if prob > 0.8 else '⚠️ REVIEW' if prob > 0.5 else '✅ APPROVE'}
"""
            return output
        
        except Exception as e:
            return f"❌ Error analyzing transaction: {str(e)}"
    
    def _run_risk_analysis_impl(self, n_transactions: int = 10000, fraud_rate: float = 0.02, avg_loss: float = 5000) -> str:
        """Run risk analysis implementation"""
        try:
            results = self.analyzer.full_risk_assessment(
                n_transactions=n_transactions,
                fraud_probability=fraud_rate,
                avg_loss_per_fraud=avg_loss,
                exposure_per_transaction=avg_loss * 0.2,
                detection_time=30,
                n_simulations=10000
            )
            
            output = f"""
💰 QUANTITATIVE RISK ANALYSIS
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
            return output
        
        except Exception as e:
            return f"❌ Error running risk analysis: {str(e)}"
    
    def _get_anomalous_patterns_impl(self, query: str = "") -> str:
        """Get anomalous patterns implementation"""
        G = self.G
        
        patterns = []
        
        degrees = [(n, G.degree(n)) for n in G.nodes()]
        degrees.sort(key=lambda x: x[1], reverse=True)
        top_hubs = degrees[:5]
        
        if top_hubs:
            patterns.append("🏢 High-Degree Hubs:")
            for node, degree in top_hubs:
                label = G.nodes[node].get('label', -1)
                label_str = "🔴 Illicit" if label == 1 else "🟢 Licit" if label == 0 else "❓ Unknown"
                patterns.append(f"   - Node {node}: Degree {degree} ({label_str})")
        
        output = "🔍 ANOMALOUS PATTERNS DETECTED\n"
        output += "=" * 50 + "\n\n"
        output += "\n".join(patterns) if patterns else "No significant anomalous patterns detected."
        
        return output
    
    def ask(self, question: str) -> str:
        """Ask a question to the agent"""
        if self.mock_mode or self.agent is None:
            # Direct tool response (fallback mode)
            question_lower = question.lower()
            
            # Check for risk analysis FIRST (before predict)
            if "risk" in question_lower or "monte" in question_lower:
                # Improved parsing for risk analysis
                numbers = re.findall(r'\d+', question)
                
                if len(numbers) >= 3:
                    n_transactions = int(numbers[0])
                    fraud_rate = float(numbers[1]) / 100
                    avg_loss = float(numbers[2])
                elif len(numbers) == 2:
                    n_transactions = int(numbers[0])
                    fraud_rate = float(numbers[1]) / 100
                    avg_loss = 5000
                elif len(numbers) == 1:
                    n_transactions = int(numbers[0])
                    fraud_rate = 0.02
                    avg_loss = 5000
                else:
                    n_transactions = 10000
                    fraud_rate = 0.02
                    avg_loss = 5000
                
                return self._run_risk_analysis_impl(
                    n_transactions=n_transactions, 
                    fraud_rate=fraud_rate, 
                    avg_loss=avg_loss
                )
            elif "statistic" in question_lower or ("fraud" in question_lower and "stat" in question_lower):
                return self._get_fraud_stats_impl()
            elif "suspicious" in question_lower or "top" in question_lower:
                numbers = re.findall(r'\d+', question)
                n = int(numbers[0]) if numbers else 10
                return self._find_suspicious_nodes_impl(n=n)
            elif "network" in question_lower or "structure" in question_lower:
                return self._analyze_network_impl()
            elif "predict" in question_lower or "transaction" in question_lower:
                node_match = re.search(r'\d+', question)
                node_id = node_match.group() if node_match else "5530458"
                return self._predict_transaction_impl(node_id=node_id)
            elif "anomalous" in question_lower or "pattern" in question_lower:
                return self._get_anomalous_patterns_impl()
            else:
                return """I can help with fraud analysis. Try asking about:
    - Fraud statistics
    - Suspicious transactions
    - Network structure
    - Transaction predictions
    - Risk analysis
    - Anomalous patterns"""
        
        try:
            response = self.agent.invoke({"messages": [HumanMessage(content=question)]})
            return response['messages'][-1].content
        except Exception as e:
            return f"❌ Error processing question: {str(e)}"