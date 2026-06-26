"""
Fast Temporal Analysis for Fraud Detection
Optimized with caching and vectorized operations
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemporalAnalyzer:
    """
    Fast temporal analysis with caching
    """
    def __init__(self, G: nx.DiGraph, data: Dict[str, Any] = None):
        self.G = G
        self.data = data
        self._cache = {}
        
        # Pre-compute frequently used data
        self._precompute()
    
    def _precompute(self):
        """Pre-compute common data once"""
        logger.info("Pre-computing temporal data...")
        start_time = time.time()
        
        # Get degrees and labels in one pass
        self.degrees = []
        self.labels = []
        self.nodes = []
        
        for node in self.G.nodes():
            self.nodes.append(node)
            self.degrees.append(self.G.degree(node))
            self.labels.append(self.G.nodes[node].get('label', -1))
        
        # Convert to numpy arrays for fast operations
        self.degrees_np = np.array(self.degrees)
        self.labels_np = np.array(self.labels)
        
        # Basic stats
        self.mean_degree = np.mean(self.degrees_np)
        self.std_degree = np.std(self.degrees_np)
        
        logger.info(f"Pre-computation complete in {time.time() - start_time:.2f}s")
    
    def get_activity_summary(self) -> Dict[str, Any]:
        """Get overall activity summary (fast)"""
        if 'summary' in self._cache:
            return self._cache['summary']
        
        degrees = self.degrees_np
        labels = self.labels_np
        
        summary = {
            'total_nodes': len(self.nodes),
            'avg_degree': float(np.mean(degrees)),
            'max_degree': int(np.max(degrees)),
            'min_degree': int(np.min(degrees)),
            'std_degree': float(np.std(degrees)),
            'total_illicit': int(np.sum(labels == 1)),
            'total_licit': int(np.sum(labels == 0)),
            'total_unknown': int(np.sum(labels == -1))
        }
        
        self._cache['summary'] = summary
        return summary
    
    def analyze_transaction_velocity(self) -> Dict[str, Any]:
        """Fast velocity analysis"""
        if 'velocity' in self._cache:
            return self._cache['velocity']
        
        logger.info("Analyzing transaction velocity...")
        
        # Use pre-computed data
        degrees = self.degrees_np
        
        # Calculate percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(degrees, percentiles)
        
        # High velocity threshold (90th percentile)
        high_velocity_threshold = percentile_values[3]  # 90th percentile
        
        # Get high velocity nodes (only top 20 for display)
        high_velocity_indices = np.where(degrees > high_velocity_threshold)[0]
        high_velocity_nodes = []
        
        # Only process top 20 to save time
        sorted_indices = np.argsort(degrees)[::-1]  # Descending
        for idx in sorted_indices[:20]:
            node = self.nodes[idx]
            label = self.labels[idx]
            high_velocity_nodes.append({
                'node': node,
                'degree': int(degrees[idx]),
                'label': 'Illicit' if label == 1 else 'Licit' if label == 0 else 'Unknown'
            })
        
        results = {
            'avg_degree': float(np.mean(degrees)),
            'max_degree': int(np.max(degrees)),
            'min_degree': int(np.min(degrees)),
            'high_velocity_threshold': float(high_velocity_threshold),
            'num_high_velocity_nodes': int(len(high_velocity_indices)),
            'high_velocity_nodes': high_velocity_nodes,
            'velocity_distribution': {
                'percentiles': {
                    f'{p}%': float(v) for p, v in zip(percentiles, percentile_values)
                }
            }
        }
        
        self._cache['velocity'] = results
        return results
    
    def analyze_time_patterns(self) -> Dict[str, Any]:
        """Fast time pattern analysis"""
        if 'time_patterns' in self._cache:
            return self._cache['time_patterns']
        
        logger.info("Analyzing time patterns...")
        
        # Use pre-computed data
        labels = self.labels_np
        
        # Count labels
        illicit_count = int(np.sum(labels == 1))
        licit_count = int(np.sum(labels == 0))
        unknown_count = int(np.sum(labels == -1))
        total = len(labels)
        
        # Activity levels based on degree
        degrees = self.degrees_np
        
        high_activity = int(np.sum(degrees > 20))
        medium_activity = int(np.sum((degrees >= 5) & (degrees <= 20)))
        low_activity = int(np.sum(degrees < 5))
        
        results = {
            'total_labeled_nodes': total,
            'illicit_nodes': illicit_count,
            'licit_nodes': licit_count,
            'unknown_nodes': unknown_count,
            'pattern_analysis': {
                'high_activity': {
                    'count': high_activity,
                    'percentage': high_activity / total * 100 if total > 0 else 0
                },
                'medium_activity': {
                    'count': medium_activity,
                    'percentage': medium_activity / total * 100 if total > 0 else 0
                },
                'low_activity': {
                    'count': low_activity,
                    'percentage': low_activity / total * 100 if total > 0 else 0
                }
            }
        }
        
        self._cache['time_patterns'] = results
        return results
    
    def analyze_fraud_trends(self) -> Dict[str, Any]:
        """Fast fraud trend analysis"""
        if 'fraud_trends' in self._cache:
            return self._cache['fraud_trends']
        
        logger.info("Analyzing fraud trends...")
        
        # Use pre-computed data
        labels = self.labels_np
        degrees = self.degrees_np
        
        # Class distribution
        illicit = int(np.sum(labels == 1))
        licit = int(np.sum(labels == 0))
        unknown = int(np.sum(labels == -1))
        total = len(labels)
        
        # Degree stats by class
        illicit_degrees = degrees[labels == 1]
        licit_degrees = degrees[labels == 0]
        
        results = {
            'class_distribution': {
                'illicit': illicit,
                'licit': licit,
                'unknown': unknown,
                'total': total
            },
            'degree_comparison': {
                'avg_illicit_degree': float(np.mean(illicit_degrees)) if len(illicit_degrees) > 0 else 0,
                'avg_licit_degree': float(np.mean(licit_degrees)) if len(licit_degrees) > 0 else 0,
                'illicit_max_degree': int(np.max(illicit_degrees)) if len(illicit_degrees) > 0 else 0,
                'licit_max_degree': int(np.max(licit_degrees)) if len(licit_degrees) > 0 else 0
            },
            'fraud_concentration': {
                'top_10_percent_illicit': int(np.sum(illicit_degrees > np.percentile(illicit_degrees, 90))) if len(illicit_degrees) > 0 else 0,
                'total_illicit': illicit
            }
        }
        
        self._cache['fraud_trends'] = results
        return results
    
    def detect_temporal_anomalies(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Fast temporal anomaly detection"""
        cache_key = f'anomalies_{top_n}'
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        logger.info("Detecting temporal anomalies...")
        
        # Use pre-computed data
        degrees = self.degrees_np
        mean = self.mean_degree
        std = self.std_degree
        
        if std == 0:
            return []
        
        # Calculate z-scores for all nodes (vectorized)
        z_scores = (degrees - mean) / std
        
        # Find anomalies (|z-score| > 2.5)
        anomaly_mask = np.abs(z_scores) > 2.5
        anomaly_indices = np.where(anomaly_mask)[0]
        
        # Create results
        anomalies = []
        for idx in anomaly_indices:
            z_score = z_scores[idx]
            anomalies.append({
                'node': self.nodes[idx],
                'degree': int(degrees[idx]),
                'z_score': float(z_score),
                'label': 'Illicit' if self.labels[idx] == 1 else 'Licit' if self.labels[idx] == 0 else 'Unknown',
                'type': 'high_activity' if z_score > 0 else 'low_activity'
            })
        
        # Sort by absolute z-score
        anomalies.sort(key=lambda x: abs(x['z_score']), reverse=True)
        
        # Cache only top N
        self._cache[cache_key] = anomalies[:top_n]
        return anomalies[:top_n]
    
    def generate_temporal_report(self) -> str:
        """Generate a comprehensive report"""
        velocity = self.analyze_transaction_velocity()
        time_patterns = self.analyze_time_patterns()
        trends = self.analyze_fraud_trends()
        anomalies = self.detect_temporal_anomalies()
        summary = self.get_activity_summary()
        
        report = f"""
TEMPORAL ANALYSIS REPORT
===========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. OVERVIEW
-------------------------
Total Nodes: {summary['total_nodes']:,}
Average Degree: {summary['avg_degree']:.2f}
Max Degree: {summary['max_degree']:,}
Min Degree: {summary['min_degree']}

2. CLASS DISTRIBUTION
-------------------------
Illicit: {trends['class_distribution']['illicit']:,} ({trends['class_distribution']['illicit']/summary['total_nodes']*100:.1f}%)
Licit: {trends['class_distribution']['licit']:,} ({trends['class_distribution']['licit']/summary['total_nodes']*100:.1f}%)
Unknown: {trends['class_distribution']['unknown']:,} ({trends['class_distribution']['unknown']/summary['total_nodes']*100:.1f}%)

3. TRANSACTION VELOCITY
-------------------------
Average Degree: {velocity['avg_degree']:.2f}
Maximum Degree: {velocity['max_degree']:,}
High Velocity Threshold: {velocity['high_velocity_threshold']:.1f}
Nodes with High Velocity: {velocity['num_high_velocity_nodes']:,}

4. ACTIVITY PATTERNS
-------------------------
High Activity (>20 connections): {time_patterns['pattern_analysis']['high_activity']['count']:,} ({time_patterns['pattern_analysis']['high_activity']['percentage']:.1f}%)
Medium Activity (5-20 connections): {time_patterns['pattern_analysis']['medium_activity']['count']:,} ({time_patterns['pattern_analysis']['medium_activity']['percentage']:.1f}%)
Low Activity (<5 connections): {time_patterns['pattern_analysis']['low_activity']['count']:,} ({time_patterns['pattern_analysis']['low_activity']['percentage']:.1f}%)

5. TEMPORAL ANOMALIES
-------------------------
Found {len(anomalies)} anomalies
Top Anomalies:
"""
        
        for i, anomaly in enumerate(anomalies[:5], 1):
            report += f"  {i}. Node {anomaly['node']}: {anomaly['type']} (z-score: {anomaly['z_score']:.2f}, degree: {anomaly['degree']})\n"
        
        return report