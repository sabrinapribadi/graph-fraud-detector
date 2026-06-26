"""
Auto-Discovery Module for Fraud Patterns
Automatically detects and surfaces 5 types of fraud patterns
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import torch
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Insight:
    """Data class for an auto-discovered insight"""
    title: str
    description: str
    category: str  # 'anomaly', 'pattern', 'risk', 'trend', 'gap'
    severity: str  # 'HIGH', 'MEDIUM', 'LOW'
    data: Dict[str, Any] = field(default_factory=dict)
    chart_data: Optional[Dict[str, Any]] = None

class AutoDiscovery:
    """
    Automatic discovery of fraud patterns in transaction graphs
    """
    def __init__(self, G: nx.DiGraph, detector=None, data=None):
        self.G = G
        self.detector = detector
        self.data = data
        self.insights = []
        
    def run_full_discovery(self) -> List[Insight]:
        """Run all discovery methods and return insights"""
        logger.info("Running full auto-discovery...")
        self.insights = []
        
        # Run each discovery method
        self._discover_money_laundering_rings()
        self._discover_structuring_patterns()
        self._discover_rapid_transaction_chains()
        self._discover_mixed_signals()
        self._discover_anomaly_outliers()
        
        logger.info(f"Discovered {len(self.insights)} insights")
        return self.insights

    def _discover_money_laundering_rings(self):
        """Detect potential money laundering rings (high-degree hubs with illicit connections)"""
        logger.info("Discovering money laundering rings...")
        
        # Find high-degree nodes
        degrees = [(n, self.G.degree(n)) for n in self.G.nodes()]
        degrees.sort(key=lambda x: x[1], reverse=True)
        
        # Find hubs connected to many illicit nodes
        rings = []
        for node, degree in degrees[:100]:  # Top 100 nodes
            if degree > 10:
                # Count illicit neighbors
                illicit_neighbors = 0
                for neighbor in self.G.neighbors(node):
                    if self.G.nodes[neighbor].get('label') == 1:
                        illicit_neighbors += 1
                
                if illicit_neighbors > degree * 0.5:  # >50% illicit connections
                    rings.append({
                        'node': node,
                        'degree': degree,
                        'illicit_neighbors': illicit_neighbors,
                        'illicit_ratio': illicit_neighbors / degree if degree > 0 else 0
                    })
        
        if rings:
            # Sort by illicit ratio
            rings.sort(key=lambda x: x['illicit_ratio'], reverse=True)
            top_rings = rings[:5]
            
            insight = Insight(
                title="Money Laundering Rings Detected",
                description=f"Found {len(rings)} potential money laundering hubs. Top hub Node {top_rings[0]['node']} has {top_rings[0]['illicit_neighbors']} illicit connections ({top_rings[0]['illicit_ratio']*100:.1f}% of all connections).",
                category="pattern",
                severity="HIGH",
                data={
                    'rings': rings,
                    'total_rings': len(rings),
                    'top_hub': top_rings[0]['node']
                },
                chart_data={
                    'type': 'bar',
                    'labels': [f"Node {r['node']}" for r in top_rings[:5]],
                    'values': [r['illicit_ratio'] * 100 for r in top_rings[:5]],
                    'title': 'Money Laundering Ring Risk Score',
                    'xlabel': 'Node',
                    'ylabel': 'Illicit Connection Ratio (%)'
                }
            )
            self.insights.append(insight)
        else:
            insight = Insight(
                title="No Money Laundering Rings Detected",
                description="No significant money laundering patterns were found in the current dataset. The network appears to have a low concentration of illicit connections.",
                category="pattern",
                severity="LOW",
                data={}
            )
            self.insights.append(insight)

    def _discover_structuring_patterns(self):
        """Detect structuring patterns (transactions just below thresholds)"""
        logger.info("Discovering structuring patterns...")
        
        structuring_nodes = []
        
        if self.data and self.detector:
            features = self.data['features']
            node_ids = self.data['node_ids']
            labels = self.data['labels']
            
            # Get predictions
            import torch
            x = torch.FloatTensor(features).to(self.detector.device)
            adj = torch.eye(len(features)).to(self.detector.device)
            
            self.detector.model.eval()
            with torch.no_grad():
                output = self.detector.model(x, adj)
                probs = torch.sigmoid(output).squeeze().cpu().numpy()
            
            if isinstance(probs, (float, np.float32, np.float64)):
                probs = np.array([probs])
            
            # Check for features near thresholds (simplified)
            for i, (node_id, features, label, prob) in enumerate(zip(node_ids, features, labels, probs)):
                # Check if any feature is just below a round number
                for j, feat in enumerate(features[:20]):  # Check first 20 features
                    if abs(feat - round(feat)) < 0.05 and feat > 0:  # Just below round number
                        structuring_nodes.append({
                            'node_id': node_id,
                            'feature_index': j,
                            'feature_value': float(feat),
                            'fraud_probability': float(prob),
                            'label': int(label) if not isinstance(label, (np.integer,)) else int(label)
                        })
                        break
        
        if structuring_nodes:
            # Take up to 50 for display
            display_nodes = structuring_nodes[:min(50, len(structuring_nodes))]
            
            insight = Insight(
                title="Structuring Patterns Detected",
                description=f"Found {len(structuring_nodes)} transactions with values just below common thresholds. This could indicate structuring (smurfing) behavior.",
                category="pattern",
                severity="MEDIUM",
                data={
                    'structuring_nodes': display_nodes,
                    'total_nodes': len(structuring_nodes)
                },
                chart_data={
                    'type': 'scatter',
                    'x': [s['fraud_probability'] * 100 for s in display_nodes],
                    'y': [s['feature_value'] for s in display_nodes],
                    'title': 'Structuring Pattern Analysis',
                    'xlabel': 'Fraud Probability (%)',
                    'ylabel': 'Feature Value'
                }
            )
            self.insights.append(insight)
        else:
            insight = Insight(
                title="No Structuring Patterns Detected",
                description="No significant structuring patterns were found in the current dataset. Transaction values appear normally distributed.",
                category="pattern",
                severity="LOW",
                data={}
            )
            self.insights.append(insight)

    def _discover_rapid_transaction_chains(self):
        """Detect rapid transaction chains (high velocity patterns)"""
        logger.info("Discovering rapid transaction chains...")
        
        # Find nodes with high degree (many transactions)
        degrees = [(n, self.G.degree(n)) for n in self.G.nodes()]
        degrees.sort(key=lambda x: x[1], reverse=True)
        
        rapid_chains = []
        
        for node, degree in degrees[:50]:  # Top 50 nodes
            # Check if node has high degree and is illicit
            label = self.G.nodes[node].get('label', -1)
            if label == 1 and degree > 10:
                # Get neighbors
                neighbors = list(self.G.neighbors(node))
                illicit_count = sum(1 for n in neighbors if self.G.nodes[n].get('label') == 1)
                
                if illicit_count > 3:  # Has multiple illicit connections
                    rapid_chains.append({
                        'node': node,
                        'degree': degree,
                        'illicit_connections': illicit_count,
                        'velocity_score': degree / 10  # Simplified velocity score
                    })
        
        if rapid_chains:
            rapid_chains.sort(key=lambda x: x['velocity_score'], reverse=True)
            top_chains = rapid_chains[:5]
            
            insight = Insight(
                title="Rapid Transaction Chains Detected",
                description=f"Found {len(rapid_chains)} nodes with high transaction velocity. Top node {top_chains[0]['node']} has {top_chains[0]['degree']} connections with {top_chains[0]['illicit_connections']} illicit connections.",
                category="pattern",
                severity="HIGH",
                data={
                    'rapid_chains': rapid_chains[:10],
                    'total_chains': len(rapid_chains)
                },
                chart_data={
                    'type': 'bar',
                    'labels': [f"Node {c['node']}" for c in top_chains],
                    'values': [c['velocity_score'] for c in top_chains],
                    'title': 'Transaction Velocity Score',
                    'xlabel': 'Node',
                    'ylabel': 'Velocity Score'
                }
            )
            self.insights.append(insight)
        else:
            insight = Insight(
                title="No Rapid Chains Detected",
                description="No rapid transaction chains were found in the current dataset. Transaction velocity appears normal.",
                category="pattern",
                severity="LOW",
                data={}
            )
            self.insights.append(insight)

    def _discover_mixed_signals(self):
        """Detect nodes with both licit and illicit connections (ambiguous signals)"""
        logger.info("Discovering mixed signals...")
        
        mixed_nodes = []
        
        for node in self.G.nodes():
            # Get neighbors
            neighbors = list(self.G.neighbors(node))
            if len(neighbors) < 3:  # Need at least 3 neighbors
                continue
            
            # Count licit and illicit neighbors
            licit_count = 0
            illicit_count = 0
            
            for neighbor in neighbors:
                label = self.G.nodes[neighbor].get('label', -1)
                if label == 0:
                    licit_count += 1
                elif label == 1:
                    illicit_count += 1
            
            # Mixed signal: both licit and illicit connections
            if licit_count > 0 and illicit_count > 0 and (licit_count + illicit_count) > 3:
                ratio = illicit_count / (licit_count + illicit_count) if (licit_count + illicit_count) > 0 else 0
                if 0.3 < ratio < 0.7:  # Mixed, not clearly one or the other
                    mixed_nodes.append({
                        'node': node,
                        'licit_connections': licit_count,
                        'illicit_connections': illicit_count,
                        'total_connections': licit_count + illicit_count,
                        'illicit_ratio': ratio,
                        'label': self.G.nodes[node].get('label', -1)
                    })
        
        if mixed_nodes:
            mixed_nodes.sort(key=lambda x: x['illicit_ratio'])
            display_nodes = mixed_nodes[:min(10, len(mixed_nodes))]
            
            insight = Insight(
                title="Mixed Signal Nodes Detected",
                description=f"Found {len(mixed_nodes)} nodes with mixed licit/illicit connections. These nodes may represent money laundering gateways or compromised accounts.",
                category="anomaly",
                severity="MEDIUM",
                data={
                    'mixed_nodes': display_nodes,
                    'total_nodes': len(mixed_nodes)
                },
                chart_data={
                    'type': 'scatter',
                    'x': [m['licit_connections'] for m in display_nodes],
                    'y': [m['illicit_connections'] for m in display_nodes],
                    'title': 'Mixed Signal Analysis',
                    'xlabel': 'Licit Connections',
                    'ylabel': 'Illicit Connections'
                }
            )
            self.insights.append(insight)
        else:
            insight = Insight(
                title="No Mixed Signals Detected",
                description="No significant mixed signal patterns were found in the current dataset. Nodes appear to have clear licit or illicit connection patterns.",
                category="anomaly",
                severity="LOW",
                data={}
            )
            self.insights.append(insight)

    def _discover_anomaly_outliers(self):
        """Detect statistical outliers in the network"""
        logger.info("Discovering anomaly outliers...")
        
        outliers = []
        
        # Calculate network statistics
        degrees = [d for n, d in self.G.degree()]
        if not degrees:
            insight = Insight(
                title="No Outliers Detected",
                description="No nodes found in the graph.",
                category="anomaly",
                severity="LOW",
                data={}
            )
            self.insights.append(insight)
            return
        
        mean_degree = np.mean(degrees)
        std_degree = np.std(degrees)
        
        # Find degree outliers
        for node in self.G.nodes():
            degree = self.G.degree(node)
            label = self.G.nodes[node].get('label', -1)
            
            # Z-score for degree
            if std_degree > 0:
                z_score = (degree - mean_degree) / std_degree
                if abs(z_score) > 2.5:  # Outlier threshold
                    outliers.append({
                        'node': node,
                        'degree': degree,
                        'z_score': float(z_score),
                        'label': int(label) if not isinstance(label, (np.integer,)) else int(label),
                        'type': 'high' if z_score > 0 else 'low'
                    })
        
        if outliers:
            outliers.sort(key=lambda x: abs(x['z_score']), reverse=True)
            top_outliers = outliers[:10]
            
            # Count high vs low outliers
            high_outliers = sum(1 for o in outliers if o['type'] == 'high')
            low_outliers = sum(1 for o in outliers if o['type'] == 'low')
            
            insight = Insight(
                title="Anomaly Outliers Detected",
                description=f"Found {len(outliers)} outlier nodes ({high_outliers} high-degree, {low_outliers} low-degree). Top outlier Node {top_outliers[0]['node']} has degree {top_outliers[0]['degree']} (z-score: {top_outliers[0]['z_score']:.2f}).",
                category="anomaly",
                severity="HIGH",
                data={
                    'outliers': top_outliers,
                    'total_outliers': len(outliers),
                    'high_outliers': high_outliers,
                    'low_outliers': low_outliers
                },
                chart_data={
                    'type': 'histogram',
                    'data': degrees[:500],  # Sample for performance
                    'title': 'Degree Distribution with Outliers',
                    'xlabel': 'Degree',
                    'ylabel': 'Frequency',
                    'outlier_threshold': float(mean_degree + 2.5 * std_degree)
                }
            )
            self.insights.append(insight)
        else:
            insight = Insight(
                title="No Outliers Detected",
                description="No significant outliers were found in the current dataset. The network appears well-distributed.",
                category="anomaly",
                severity="LOW",
                data={}
            )
            self.insights.append(insight)

    def get_insights_by_category(self, category: str) -> List[Insight]:
        """Get insights filtered by category"""
        return [i for i in self.insights if i.category == category]
    
    def get_insights_by_severity(self, severity: str) -> List[Insight]:
        """Get insights filtered by severity"""
        return [i for i in self.insights if i.severity == severity]
    
    def generate_report(self) -> str:
        """Generate a text report of all insights"""
        report = "AUTO-DISCOVERY REPORT\n"
        report += "=" * 50 + "\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Total Insights: {len(self.insights)}\n"
        report += "=" * 50 + "\n\n"
        
        for i, insight in enumerate(self.insights, 1):
            report += f"{i}. {insight.title}\n"
            report += f"   Category: {insight.category} | Severity: {insight.severity}\n"
            report += f"   {insight.description}\n\n"
        
        return report                                                                    