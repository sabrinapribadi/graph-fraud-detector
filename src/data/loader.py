"""
Data loader for Elliptic Bitcoin Fraud Dataset

Fast path: loads from data/processed/*.parquet (zstd, ~86 MB total).
Slow path: falls back to raw CSVs and auto-saves a parquet cache for next run.
Generate parquet files once with:  python scripts/preprocess_data.py
"""
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PROCESSED = Path("data/processed")
_FEAT_PQ   = _PROCESSED / "features.parquet"
_CLS_PQ    = _PROCESSED / "classes.parquet"
_EDGE_PQ   = _PROCESSED / "edgelist.parquet"


class EllipticDataLoader:
    """Load and preprocess Elliptic Bitcoin fraud dataset"""

    def __init__(self, data_path: str = "data/raw/elliptic_bitcoin_dataset"):
        self.data_path = Path(data_path)
        self.features = None
        self.classes = None
        self.edgelist = None
        self.graph = None
        self.node_features = None
        self.node_labels = None
        self.train_mask = None
        self.test_mask = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load dataset — parquet cache first, raw CSVs as fallback."""
        if _FEAT_PQ.exists() and _CLS_PQ.exists() and _EDGE_PQ.exists():
            logger.info("Loading from parquet cache...")
            self.features  = pd.read_parquet(_FEAT_PQ)
            self.classes   = pd.read_parquet(_CLS_PQ)
            self.edgelist  = pd.read_parquet(_EDGE_PQ)
            logger.info(f"Loaded {len(self.features):,} nodes, {len(self.edgelist):,} edges")
            return self.features, self.classes, self.edgelist

        logger.info("Parquet cache not found — loading from raw CSVs (first run is slow)...")
        self._load_from_csv()
        self._save_parquet_cache()
        return self.features, self.classes, self.edgelist

    def _load_from_csv(self) -> None:
        """Read the three raw CSV files and normalise column names."""
        self.features = pd.read_csv(
            self.data_path / "elliptic_txs_features.csv",
            header=None,
        )

        self.classes = pd.read_csv(self.data_path / "elliptic_txs_classes.csv")
        logger.info(f"Classes value counts:\n{self.classes['class'].value_counts()}")

        self.edgelist = pd.read_csv(
            self.data_path / "elliptic_txs_edgelist.csv",
            header=None,
        )

        if self.edgelist.shape[1] == 3:
            self.edgelist.columns = ["source", "target", "timestamp"]
        elif self.edgelist.shape[1] == 2:
            self.edgelist.columns = ["source", "target"]
            self.edgelist["timestamp"] = 0
        else:
            raise ValueError(f"Unexpected edgelist shape: {self.edgelist.shape}")

        self.edgelist["source"] = self.edgelist["source"].astype(str)
        self.edgelist["target"] = self.edgelist["target"].astype(str)

        initial_len = len(self.edgelist)
        self.edgelist = self.edgelist.dropna(subset=["source", "target"])
        if len(self.edgelist) < initial_len:
            logger.warning(f"Dropped {initial_len - len(self.edgelist)} rows with NaN source/target")

        logger.info(f"Loaded {len(self.features):,} nodes, {len(self.edgelist):,} edges")

    def _save_parquet_cache(self) -> None:
        """Write parquet cache so future loads are fast."""
        try:
            _PROCESSED.mkdir(parents=True, exist_ok=True)
            self.features.to_parquet(_FEAT_PQ,  compression="zstd", index=False)
            self.classes.to_parquet( _CLS_PQ,   compression="zstd", index=False)
            self.edgelist.to_parquet(_EDGE_PQ,  compression="zstd", index=False)
            logger.info(f"Parquet cache saved to {_PROCESSED}")
        except Exception as exc:
            logger.warning(f"Could not write parquet cache: {exc}")
    
    def preprocess_features(self) -> np.ndarray:
        """Extract and normalize node features"""
        logger.info("Preprocessing features...")
        
        # First column is node ID, rest are features
        self.node_ids = self.features.iloc[:, 0].values.astype(str)
        features = self.features.iloc[:, 1:].values.astype(np.float32)
        
        # Handle missing values (fill with column means)
        col_means = np.nanmean(features, axis=0)
        for i in range(features.shape[1]):
            if np.isnan(features[:, i]).any():
                features[:, i] = np.nan_to_num(features[:, i], nan=col_means[i])
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        self.node_features = features_scaled
        logger.info(f"Features shape: {features_scaled.shape}")
        return features_scaled
    
    def prepare_labels(self) -> np.ndarray:
        """Prepare node labels (0=licit, 1=illicit, -1=unknown)"""
        logger.info("Preparing labels...")
        
        # Map classes to numeric labels
        def map_label(class_val):
            if class_val == '1':
                return 0  # licit
            elif class_val == '2':
                return 1  # illicit
            else:
                return -1  # unknown
        
        self.classes['label'] = self.classes['class'].apply(map_label)
        self.classes['txId'] = self.classes['txId'].astype(str)
        
        # Create label array aligned with node IDs
        labels = np.full(len(self.node_ids), -1, dtype=int)  # Default to unknown
        
        # Create a dictionary for fast lookup
        label_dict = dict(zip(self.classes['txId'], self.classes['label']))
        
        # Fill labels
        for idx, node_id in enumerate(self.node_ids):
            if node_id in label_dict:
                labels[idx] = label_dict[node_id]
        
        self.node_labels = labels
        
        # Create train/test masks (only for labeled nodes)
        labeled_mask = labels != -1
        labeled_indices = np.where(labeled_mask)[0]
        
        logger.info(f"Labeled nodes: {len(labeled_indices):,} ({len(labeled_indices)/len(labels)*100:.1f}%)")
        
        if len(labeled_indices) > 0:
            # Split 80/20
            np.random.seed(42)
            np.random.shuffle(labeled_indices)
            split = int(0.8 * len(labeled_indices))
            
            self.train_mask = np.zeros(len(labels), dtype=bool)
            self.test_mask = np.zeros(len(labels), dtype=bool)
            
            self.train_mask[labeled_indices[:split]] = True
            self.test_mask[labeled_indices[split:]] = True
        
        # Log distribution
        licit_count = sum(labels == 0)
        illicit_count = sum(labels == 1)
        unknown_count = sum(labels == -1)
        
        logger.info(f"Labels: Licit={licit_count:,}, Illicit={illicit_count:,}, Unknown={unknown_count:,}")
        logger.info(f"Train: {sum(self.train_mask):,}, Test: {sum(self.test_mask):,}")
        
        return labels
    
    def build_graph(self) -> nx.DiGraph:
        """Build NetworkX directed graph"""
        logger.info("Building directed graph...")
        
        if self.node_features is None:
            self.preprocess_features()
        if self.node_labels is None:
            self.prepare_labels()
        
        G = nx.DiGraph()
        
        # Add nodes with features and labels
        for i, node_id in enumerate(self.node_ids):
            G.add_node(
                node_id,
                features=self.node_features[i],
                label=self.node_labels[i],
                train_mask=self.train_mask[i] if hasattr(self, 'train_mask') else False,
                test_mask=self.test_mask[i] if hasattr(self, 'test_mask') else False
            )
        
        # Add edges
        edge_count = 0
        for _, row in self.edgelist.iterrows():
            source = str(row['source'])
            target = str(row['target'])
            timestamp = row.get('timestamp', 0)
            
            # Only add edges if both nodes exist in graph
            if source in G and target in G:
                G.add_edge(source, target, timestamp=timestamp)
                edge_count += 1
        
        if edge_count < len(self.edgelist):
            logger.warning(f"Added {edge_count:,} of {len(self.edgelist):,} edges (some nodes may be missing)")
        
        self.graph = G
        logger.info(f"Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        return G
    
    def get_graph_stats(self) -> dict:
        """Get basic graph statistics"""
        if self.graph is None:
            self.build_graph()
        
        G = self.graph
        
        # Calculate connected components
        try:
            components = nx.number_weakly_connected_components(G)
        except:
            components = 0
        
        # Calculate isolates
        try:
            isolates = nx.number_of_isolates(G)
        except:
            isolates = 0
        
        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'num_components': components,
            'isolates': isolates,
        }
        
        if G.number_of_edges() > 0:
            try:
                stats['density'] = nx.density(G)
            except:
                stats['density'] = 0.0
            
            degrees = [d for n, d in G.degree()]
            stats.update({
                'avg_degree': np.mean(degrees),
                'max_degree': np.max(degrees),
                'min_degree': np.min(degrees),
            })
        else:
            stats['density'] = 0.0
            stats['avg_degree'] = 0.0
            stats['max_degree'] = 0
            stats['min_degree'] = 0
        
        return stats

if __name__ == "__main__":
    # Test the loader
    loader = EllipticDataLoader()
    features, classes, edgelist = loader.load_data()
    loader.preprocess_features()
    loader.prepare_labels()
    G = loader.build_graph()
    
    print(f"\nGraph built successfully.")
    print(f"  - Nodes: {G.number_of_nodes():,}")
    print(f"  - Edges: {G.number_of_edges():,}")
    print(f"  - Isolated nodes: {nx.number_of_isolates(G):,}")

    # Print some stats
    stats = loader.get_graph_stats()
    print(f"\nGraph Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.4f}")
        else:
            print(f"  - {key}: {value:,}")