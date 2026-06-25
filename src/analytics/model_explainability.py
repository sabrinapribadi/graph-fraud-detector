"""
Model Explainability for Fraud Detection
Using SHAP and Captum for interpretability
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import shap
from typing import Dict, Any, List, Optional, Tuple
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelExplainer:
    """
    Model explainability for fraud detection
    """
    def __init__(self, model, x: torch.Tensor, feature_names: Optional[List[str]] = None):
        self.model = model
        self.x = x
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(x.shape[1])]
        self.device = x.device
        
    def explain_prediction(
        self,
        node_id: str,
        features: np.ndarray,
        method: str = 'shap'
    ) -> Dict[str, Any]:
        """
        Explain a single prediction
        """
        if method == 'shap':
            return self._shap_explanation(features)
        elif method == 'gradient':
            return self._gradient_explanation(features)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _shap_explanation(self, features: np.ndarray) -> Dict[str, Any]:
        """
        SHAP-based explanation
        """
        try:
            # Create a wrapper for the model
            def predict(x):
                x_tensor = torch.FloatTensor(x).to(self.device)
                self.model.eval()
                with torch.no_grad():
                    # Create simple adjacency (identity matrix)
                    adj = torch.eye(x_tensor.shape[0]).to(self.device)
                    output = self.model(x_tensor, adj)
                    return torch.sigmoid(output).squeeze().cpu().numpy()
            
            # Create explainer
            explainer = shap.KernelExplainer(predict, self.x[:100].cpu().numpy())
            shap_values = explainer.shap_values(features.reshape(1, -1))
            
            # Get top features
            shap_values_flat = shap_values[0] if isinstance(shap_values, list) else shap_values
            top_indices = np.argsort(np.abs(shap_values_flat))[-10:]
            
            explanation = {
                'method': 'shap',
                'shap_values': shap_values_flat.tolist(),
                'top_features': [
                    {
                        'name': self.feature_names[i],
                        'value': float(features[i]),
                        'importance': float(shap_values_flat[i])
                    }
                    for i in top_indices
                ],
                'prediction': float(predict(features.reshape(1, -1))[0])
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {
                'method': 'shap',
                'error': str(e),
                'prediction': None,
                'top_features': []
            }
    
    def _gradient_explanation(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Gradient-based explanation (simpler alternative)
        """
        try:
            x_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            x_tensor.requires_grad = True
            
            self.model.eval()
            adj = torch.eye(x_tensor.shape[0]).to(self.device)
            output = self.model(x_tensor, adj)
            
            # Compute gradients
            output.backward()
            gradients = x_tensor.grad.squeeze().cpu().numpy()
            
            # Get top features
            top_indices = np.argsort(np.abs(gradients))[-10:]
            
            explanation = {
                'method': 'gradient',
                'gradients': gradients.tolist(),
                'top_features': [
                    {
                        'name': self.feature_names[i],
                        'value': float(features[i]),
                        'importance': float(gradients[i])
                    }
                    for i in top_indices
                ],
                'prediction': float(torch.sigmoid(output).squeeze().item())
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Gradient explanation failed: {e}")
            return {
                'method': 'gradient',
                'error': str(e),
                'prediction': None,
                'top_features': []
            }
    
    def explain_node_subgraph(
        self,
        node_id: str,
        G: nx.DiGraph,
        num_neighbors: int = 3
    ) -> Dict[str, Any]:
        """
        Explain node prediction using its neighborhood
        """
        try:
            explanation = {
                'node_id': node_id,
                'neighbors': [],
                'influence_score': 0.0
            }
            
            # Get neighbors
            neighbors = list(G.neighbors(node_id))[:num_neighbors]
            
            for neighbor in neighbors:
                neighbor_data = G.nodes[neighbor]
                label = neighbor_data.get('label', -1)
                label_str = 'Illicit' if label == 1 else 'Licit' if label == 0 else 'Unknown'
                
                explanation['neighbors'].append({
                    'id': neighbor,
                    'label': label_str,
                    'degree': G.degree(neighbor)
                })
            
            # Calculate influence score (simplified)
            illicit_neighbors = sum(1 for n in neighbors if G.nodes[n].get('label') == 1)
            if neighbors:
                explanation['influence_score'] = illicit_neighbors / len(neighbors)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Subgraph explanation failed: {e}")
            return {
                'node_id': node_id,
                'error': str(e),
                'neighbors': []
            }
    
    def generate_explanation_report(self, node_id: str, features: np.ndarray) -> str:
        """
        Generate a human-readable explanation report
        """
        shap_exp = self._shap_explanation(features)
        
        report = f"""
🔍 EXPLANATION REPORT FOR NODE {node_id}
=========================================

Prediction: {shap_exp.get('prediction', 'N/A'):.2%}

Top Influential Features:
"""
        for i, feat in enumerate(shap_exp.get('top_features', [])[:5], 1):
            report += f"  {i}. {feat['name']}: {feat['value']:.4f} (importance: {feat['importance']:.4f})\n"
        
        if 'error' in shap_exp:
            report += f"\n⚠️  Note: {shap_exp['error']}"
        
        return report