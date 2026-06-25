"""
Advanced GNN Architectures for Fraud Detection
Includes GAT and GraphSAGE variants (mean and sum only)
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT)
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention
        """
        Wh = torch.mm(h, self.W)
        N = Wh.size(0)
        
        # Compute attention coefficients
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)
    
    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        """
        Prepare input for attentional mechanism
        """
        N = Wh.size(0)
        Wh_repeated_in = Wh.repeat(N, 1).view(N, N, -1)
        Wh_repeated_out = Wh.repeat(N, 1).view(N, N, -1)
        return torch.cat([Wh_repeated_in, Wh_repeated_out], dim=2)

class GAT(nn.Module):
    """
    Graph Attention Network (GAT) for node classification
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        out_features: int = 1,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # First layer with multiple heads
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, hidden_dim, dropout=dropout)
            for _ in range(num_heads)
        ])
        
        # Output layer
        self.out_att = GraphAttentionLayer(hidden_dim * num_heads, out_features, dropout=dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * num_heads)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.batch_norm(x)
        x = self.out_att(x, adj)
        return x

class GraphSAGE(nn.Module):
    """
    GraphSAGE with different aggregators (mean and sum only)
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        out_features: int = 1,
        num_layers: int = 2,
        dropout: float = 0.3,
        aggregator: str = 'mean'
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregator = aggregator
        
        # Input layer
        self.input_layer = nn.Linear(in_features, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, out_features)
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
        ])
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with message passing (mean and sum only)
        """
        # Compute degrees for aggregation
        degrees = adj.sum(dim=1, keepdim=True)
        degrees = torch.clamp(degrees, min=1.0)
        
        # Message passing - only mean and sum
        if self.aggregator == 'mean':
            aggr = adj @ x / degrees
        elif self.aggregator == 'sum':
            aggr = adj @ x
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}. Use 'mean' or 'sum'.")
        
        # Combine self and neighbors
        x = 0.5 * x + 0.5 * aggr
        
        # Input layer
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        # Hidden layers
        for i, (layer, bn) in enumerate(zip(self.hidden_layers, self.batch_norms)):
            residual = x
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout_layer(x)
            
            # Skip connection
            if i < len(self.hidden_layers) - 1:
                x = x + residual
        
        # Output
        x = self.output_layer(x)
        return x

class EnsembleFraudDetector:
    """
    Ensemble of multiple GNN models for robust fraud detection
    Uses only 'mean' and 'sum' aggregators (no 'max')
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        out_features: int = 1,
        device: str = 'mps'
    ):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.device = device if torch.backends.mps.is_available() else 'cpu'
        
        # Initialize models - only mean and sum (no max)
        self.models = {
            'gat': GAT(in_features, hidden_dim, out_features, num_heads=2),
            'sage_mean': GraphSAGE(in_features, hidden_dim, out_features, aggregator='mean'),
            'sage_sum': GraphSAGE(in_features, hidden_dim, out_features, aggregator='sum')
        }
        
        # Move models to device
        for model in self.models.values():
            model.to(self.device)
        
        # Equal weights for 3 models
        self.weights = {name: 1/3 for name in self.models.keys()}
    
    def train_models(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        adj: torch.Tensor,
        train_mask: torch.Tensor,
        epochs: int = 100,
        lr: float = 0.001
    ) -> Dict[str, List[float]]:
        """
        Train all models in the ensemble
        """
        losses = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()
            
            model.train()
            model_losses = []
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                output = model(x, adj)
                loss = criterion(output[train_mask].squeeze(), y[train_mask])
                loss.backward()
                optimizer.step()
                
                model_losses.append(loss.item())
                
                if epoch % 20 == 0:
                    logger.info(f"  {name} Epoch {epoch}: Loss = {loss.item():.4f}")
            
            losses[name] = model_losses
        
        return losses
    
    def predict(self, x: torch.Tensor, adj: torch.Tensor) -> np.ndarray:
        """
        Ensemble prediction using weighted voting
        """
        self._set_eval_mode()
        
        with torch.no_grad():
            predictions = []
            
            for name, model in self.models.items():
                output = model(x, adj)
                # Move to CPU before converting to numpy
                prob = torch.sigmoid(output).squeeze().cpu().numpy()
                predictions.append(prob * self.weights[name])
            
            # Weighted average
            ensemble_pred = np.sum(predictions, axis=0)
        
        return ensemble_pred
    
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        adj: torch.Tensor,
        test_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance
        """
        self._set_eval_mode()
        
        with torch.no_grad():
            preds = self.predict(x, adj)
            
            # Move to CPU and convert to numpy
            y_true = y[test_mask].cpu().numpy()
            y_pred = (preds[test_mask] > 0.5).astype(int)
            y_score = preds[test_mask]
            
            # Check if we have both classes
            unique_classes = np.unique(y_true)
            logger.info(f"Unique classes in test set: {unique_classes}")
            
            # Calculate all metrics
            if len(unique_classes) > 1:
                auc = roc_auc_score(y_true, y_score)
            else:
                logger.warning(f"Only one class found in test set: {unique_classes}")
                auc = 0.5  # Random baseline
                
            f1 = f1_score(y_true, y_pred, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            
            # Calculate precision and recall
            from sklearn.metrics import precision_score, recall_score
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            metrics = {
                'auc': auc,
                'f1': f1,
                'accuracy': accuracy,
                'precision': precision,  # MAKE SURE THIS IS INCLUDED
                'recall': recall,
                'predictions': y_pred.tolist(),
                'probabilities': y_score.tolist(),
                'unique_classes': unique_classes.tolist() if len(unique_classes) > 0 else []
            }
            
            logger.info(f"Ensemble Evaluation:")
            logger.info(f"  AUC: {auc:.4f}")
            logger.info(f"  F1: {f1:.4f}")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            
            return metrics
    
    def _set_eval_mode(self):
        """Set all models to evaluation mode"""
        for model in self.models.values():
            model.eval()
    
    def optimize_weights(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        adj: torch.Tensor,
        val_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Optimize ensemble weights using validation data
        """
        self._set_eval_mode()
        
        with torch.no_grad():
            predictions = []
            for name, model in self.models.items():
                output = model(x, adj)
                prob = torch.sigmoid(output).squeeze().cpu().numpy()
                predictions.append(prob[val_mask])
            
            best_weights = None
            best_auc = 0.0
            
            y_true = y[val_mask].cpu().numpy()
            
            # Try different weight combinations for 3 models
            for w1 in [0.1, 0.2, 0.3, 0.4, 0.5]:
                for w2 in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    w3 = 1.0 - w1 - w2
                    if w3 < 0.1 or w3 > 0.5:
                        continue
                    
                    weights = {'gat': w1, 'sage_mean': w2, 'sage_sum': w3}
                    
                    ensemble_pred = (
                        predictions[0] * weights['gat'] +
                        predictions[1] * weights['sage_mean'] +
                        predictions[2] * weights['sage_sum']
                    )
                    
                    if len(np.unique(y_true)) > 1:
                        auc = roc_auc_score(y_true, ensemble_pred)
                    else:
                        auc = 0.0
                    
                    if auc > best_auc:
                        best_auc = auc
                        best_weights = weights
            
            if best_weights:
                self.weights = best_weights
                logger.info(f"Optimized weights: {best_weights}")
                logger.info(f"Validation AUC: {best_auc:.4f}")
            
            return self.weights