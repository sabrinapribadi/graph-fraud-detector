"""
Hyperparameter Optimization for GNN Models using Optuna
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import optuna
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import logging
import warnings
warnings.filterwarnings('ignore')

from src.models.gnn_model import GraphSAGE as BaseGraphSAGE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna
    """
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        adj: torch.Tensor,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        device: str = 'mps',
        n_trials: int = 50
    ):
        self.x = x
        self.y = y
        self.adj = adj
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.device = device if torch.backends.mps.is_available() else 'cpu'
        self.n_trials = n_trials
        
        # Move tensors to device
        self.x = self.x.to(self.device)
        self.y = self.y.to(self.device)
        self.adj = self.adj.to(self.device)
        self.train_mask = self.train_mask.to(self.device)
        self.val_mask = self.val_mask.to(self.device)
        
        # Check if validation set has both classes
        y_val = self.y[self.val_mask].cpu().numpy()
        unique_classes = np.unique(y_val)
        logger.info(f"Validation set classes: {unique_classes}")
        
        if len(unique_classes) < 2:
            logger.warning("Validation set has only one class! AUC will be 0.5.")
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization
        """
        # Suggest hyperparameters
        hidden_dim = trial.suggest_int('hidden_dim', 16, 128, step=16)
        num_layers = trial.suggest_int('num_layers', 2, 4)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        aggregator = trial.suggest_categorical('aggregator', ['mean', 'sum'])
        
        # Create model
        model = BaseGraphSAGE(
            in_features=self.x.shape[1],
            hidden_dim=hidden_dim,
            out_features=1,
            num_layers=num_layers,
            dropout=dropout,
            aggregate=aggregator
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        # Early stopping
        best_loss = float('inf')
        patience = 0
        
        # Train for a few epochs
        model.train()
        for epoch in range(30):  # Reduced for speed
            optimizer.zero_grad()
            output = model(self.x, self.adj)
            loss = criterion(output[self.train_mask].squeeze(), self.y[self.train_mask])
            loss.backward()
            optimizer.step()
            
            # Early stopping check
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience = 0
            else:
                patience += 1
                if patience > 5:
                    break
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            output = model(self.x, self.adj)
            # Get predictions for validation set
            preds = torch.sigmoid(output[self.val_mask]).squeeze().cpu().numpy()
            y_true = self.y[self.val_mask].cpu().numpy()
            
            # Check if we have both classes
            unique_classes = np.unique(y_true)
            
            if len(unique_classes) > 1:
                # Calculate AUC
                try:
                    auc = roc_auc_score(y_true, preds)
                except:
                    auc = 0.0
            else:
                # Only one class - use accuracy instead
                preds_binary = (preds > 0.5).astype(int)
                auc = accuracy_score(y_true, preds_binary)
                logger.warning(f"Only one class in validation. Using accuracy: {auc:.4f}")
        
        return auc
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization
        """
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials...")
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best validation score: {best_value:.4f}")
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'study': study
        }
    
    def get_trial_history(self) -> Dict[str, Any]:
        """
        Get optimization history
        """
        return {
            'best_value': self.study.best_value if hasattr(self, 'study') else None,
            'best_params': self.study.best_params if hasattr(self, 'study') else None,
            'n_trials': len(self.study.trials) if hasattr(self, 'study') else 0
        }