"""
Graph Neural Network for Fraud Detection
Using pure PyTorch (no PyTorch Geometric dependency)
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import networkx as nx
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphSAGE(nn.Module):
    """
    GraphSAGE-inspired GNN for node classification
    """
    def __init__(
        self, 
        in_features: int, 
        hidden_dim: int = 64,  # Reduced from 128
        out_features: int = 1,
        num_layers: int = 2,   # Reduced from 3
        dropout: float = 0.3,
        aggregate: str = 'mean'
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregate = aggregate
        
        # Input projection
        self.input_proj = nn.Linear(in_features, hidden_dim)
        
        # Hidden layers with skip connections
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, out_features)
        
        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with message passing
        """
        # Message passing: aggregate neighbors
        degrees = adj_matrix.sum(dim=1, keepdim=True)
        degrees = torch.clamp(degrees, min=1.0)
        
        # Aggregate neighbor features
        if self.aggregate == 'mean':
            aggr = adj_matrix @ x / degrees
        elif self.aggregate == 'sum':
            aggr = adj_matrix @ x
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregate}")
        
        # Combine self and neighbor features
        x = 0.5 * x + 0.5 * aggr
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
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

class FraudDetector:
    """
    Complete fraud detection pipeline with GNN
    """
    def __init__(
        self,
        hidden_dim: int = 64,  # Reduced
        num_layers: int = 2,   # Reduced
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        device: str = 'mps'
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = device if torch.backends.mps.is_available() else 'cpu'
        
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        
        logger.info(f"Using device: {self.device}")
        
    def build_graph_data(
        self, 
        G: nx.DiGraph, 
        sample_size: int = 3000,  # Reduced default
        balance_classes: bool = True
    ) -> Dict[str, Any]:
        """
        Convert NetworkX graph to PyTorch tensors with efficient sampling
        """
        logger.info("Building graph data for training...")
        
        # Extract node features and labels efficiently
        node_ids = []
        features = []
        labels = []
        train_mask = []
        test_mask = []
        
        # Collect all labeled nodes
        labeled_nodes = []
        for node, data in G.nodes(data=True):
            if 'features' in data and 'label' in data and data['label'] != -1:
                labeled_nodes.append((node, data))
        
        logger.info(f"Found {len(labeled_nodes)} labeled nodes")
        
        if len(labeled_nodes) == 0:
            raise ValueError("No labeled nodes found in graph")
        
        # Sample nodes (balanced if needed)
        if balance_classes:
            licit_nodes = [(n, d) for n, d in labeled_nodes if d['label'] == 0]
            illicit_nodes = [(n, d) for n, d in labeled_nodes if d['label'] == 1]
            
            logger.info(f"Licit: {len(licit_nodes)}, Illicit: {len(illicit_nodes)}")
            
            # Sample equally from both classes
            sample_per_class = min(sample_size // 2, len(licit_nodes), len(illicit_nodes))
            
            np.random.seed(42)
            licit_sample = np.random.choice(len(licit_nodes), sample_per_class, replace=False)
            illicit_sample = np.random.choice(len(illicit_nodes), sample_per_class, replace=False)
            
            sampled_nodes = [licit_nodes[i] for i in licit_sample] + [illicit_nodes[i] for i in illicit_sample]
        else:
            # Random sample
            np.random.seed(42)
            indices = np.random.choice(len(labeled_nodes), min(sample_size, len(labeled_nodes)), replace=False)
            sampled_nodes = [labeled_nodes[i] for i in indices]
        
        logger.info(f"Sampled {len(sampled_nodes)} nodes")
        
        # Build data arrays
        node_ids = [n for n, _ in sampled_nodes]
        features = [d['features'] for _, d in sampled_nodes]
        labels = [d['label'] for _, d in sampled_nodes]
        
        # Convert to numpy
        features = np.array(features)
        labels = np.array(labels)
        labels_binary = (labels == 1).astype(np.float32)  # 1 = illicit
        
        # Create adjacency matrix for sampled nodes
        n_nodes = len(node_ids)
        node_to_idx = {node: i for i, node in enumerate(node_ids)}
        
        adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        
        # Add edges between sampled nodes
        edge_count = 0
        for source, target in G.edges():
            if source in node_to_idx and target in node_to_idx:
                i = node_to_idx[source]
                j = node_to_idx[target]
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0  # Make undirected for message passing
                edge_count += 1
        
        logger.info(f"Added {edge_count} edges between sampled nodes")
        
        # Create train/test split (80/20)
        np.random.seed(42)
        indices = np.arange(n_nodes)
        np.random.shuffle(indices)
        split = int(0.8 * n_nodes)
        
        train_indices = indices[:split]
        test_indices = indices[split:]
        
        train_mask = np.zeros(n_nodes, dtype=bool)
        test_mask = np.zeros(n_nodes, dtype=bool)
        train_mask[train_indices] = True
        test_mask[test_indices] = True
        
        # Convert to tensors and move to device
        x = torch.FloatTensor(features).to(self.device)
        y = torch.FloatTensor(labels_binary).to(self.device)
        adj = torch.FloatTensor(adj_matrix).to(self.device)
        train_mask_tensor = torch.BoolTensor(train_mask).to(self.device)
        test_mask_tensor = torch.BoolTensor(test_mask).to(self.device)
        
        logger.info(f"Data shape: {x.shape}, Positive labels: {y.sum().item():.0f}")
        logger.info(f"Train: {train_mask_tensor.sum().item()}, Test: {test_mask_tensor.sum().item()}")
        
        # Clear memory
        gc.collect()
        
        return {
            'x': x,
            'y': y,
            'adj': adj,
            'train_mask': train_mask_tensor,
            'test_mask': test_mask_tensor,
            'features': features,
            'labels': labels_binary,
            'node_ids': node_ids
        }
    
    def train(
        self,
        data: Dict[str, Any],
        epochs: int = 100,
        batch_size: Optional[int] = None,
        early_stopping: int = 20
    ) -> Dict[str, Any]:
        """
        Train the GNN model
        """
        x = data['x']
        y = data['y']
        adj = data['adj']
        train_mask = data['train_mask']
        
        # Initialize model with smaller architecture
        in_features = x.shape[1]
        self.model = GraphSAGE(
            in_features=in_features,
            hidden_dim=self.hidden_dim,
            out_features=1,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Training loop
        self.model.train()
        losses = []
        best_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(x, adj)
            loss = self.loss_fn(output[train_mask].squeeze(), y[train_mask])
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            losses.append(loss.item())
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                with torch.no_grad():
                    preds = torch.sigmoid(output[train_mask]).squeeze()
                    preds_binary = (preds > 0.5).float()
                    accuracy = (preds_binary == y[train_mask]).float().mean().item()
                
                logger.info(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}, Train Acc = {accuracy:.4f}")
        
        logger.info("Training complete!")
        return {'losses': losses}
    
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the trained model
        """
        self.model.eval()
        
        with torch.no_grad():
            x = data['x']
            y = data['y']
            adj = data['adj']
            test_mask = data['test_mask']
            
            output = self.model(x, adj)
            preds = torch.sigmoid(output[test_mask]).squeeze()
            preds_binary = (preds > 0.5).float()
            
            # Calculate metrics
            y_true = y[test_mask].cpu().numpy()
            y_pred = preds_binary.cpu().numpy()
            y_score = preds.cpu().numpy()
            
            # AUC
            try:
                auc = roc_auc_score(y_true, y_score)
            except:
                auc = 0.0
            
            # F1
            try:
                f1 = f1_score(y_true, y_pred, zero_division=0)
            except:
                f1 = 0.0
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Accuracy
            accuracy = (y_pred == y_true).mean()
            
            # Precision and Recall
            if cm.shape == (2, 2):
                tp = cm[1, 1]
                fp = cm[0, 1]
                fn = cm[1, 0]
                tn = cm[0, 0]
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            else:
                precision = 0
                recall = 0
                tp = fp = fn = tn = 0
            
            metrics = {
                'auc': auc,
                'f1': f1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'confusion_matrix': cm,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            }
            
            logger.info(f"Evaluation Results:")
            logger.info(f"  AUC: {metrics['auc']:.4f}")
            logger.info(f"  F1: {metrics['f1']:.4f}")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            if cm.shape == (2, 2):
                logger.info(f"  Confusion Matrix: \n{cm}")
            
            return metrics
    
    def predict(self, features: np.ndarray, adj: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict fraud probability for new nodes
        """
        self.model.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            if adj is None:
                adj = torch.eye(len(features)).to(self.device)
            else:
                adj = torch.FloatTensor(adj).to(self.device)
            
            output = self.model(x, adj)
            probs = torch.sigmoid(output).squeeze().cpu().numpy()
            
        return probs

if __name__ == "__main__":
    # Test the model
    from src.data.loader import EllipticDataLoader
    
    # Load data
    loader = EllipticDataLoader()
    features, classes, edgelist = loader.load_data()
    loader.preprocess_features()
    loader.prepare_labels()
    G = loader.build_graph()
    
    # Initialize detector with smaller architecture
    detector = FraudDetector(
        hidden_dim=32,  # Smaller
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001
    )
    
    # Build graph data with balanced sampling (1000 nodes per class)
    data = detector.build_graph_data(
        G, 
        sample_size=2000,  # 1000 licit + 1000 illicit
        balance_classes=True
    )
    
    # Train
    train_results = detector.train(data, epochs=100)
    
    # Evaluate
    metrics = detector.evaluate(data)
    
    print("\nModel trained and evaluated.")
    print(f"  - AUC: {metrics['auc']:.4f}")
    print(f"  - F1: {metrics['f1']:.4f}")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall: {metrics['recall']:.4f}")