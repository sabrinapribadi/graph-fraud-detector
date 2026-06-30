"""
Unit tests for GraphSAGE and FraudDetector.
Run with: pytest tests/test_gnn_model.py -v
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import networkx as nx
import pytest
import torch

from src.models.gnn_model import GraphSAGE, FraudDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(n_licit: int = 20, n_illicit: int = 20, n_features: int = 8) -> nx.DiGraph:
    """Return a small synthetic DiGraph with labeled nodes and features."""
    G = nx.DiGraph()
    rng = np.random.default_rng(0)

    for i in range(n_licit):
        G.add_node(f"l{i}", features=rng.random(n_features).tolist(), label=0)
    for i in range(n_illicit):
        G.add_node(f"i{i}", features=rng.random(n_features).tolist(), label=1)

    # Add some edges so the adjacency matrix isn't all-zero
    nodes = list(G.nodes())
    for _ in range(30):
        u, v = rng.choice(nodes, size=2, replace=False)
        G.add_edge(u, v)

    return G


# ---------------------------------------------------------------------------
# GraphSAGE tests
# ---------------------------------------------------------------------------

class TestGraphSAGE:
    def test_output_shape_mean(self):
        n, f = 10, 8
        model = GraphSAGE(in_features=f, hidden_dim=16, out_features=1, aggregator='mean')
        x = torch.randn(n, f)
        adj = torch.eye(n)
        out = model(x, adj)
        assert out.shape == (n, 1)

    def test_output_shape_sum(self):
        n, f = 10, 8
        model = GraphSAGE(in_features=f, hidden_dim=16, out_features=1, aggregator='sum')
        x = torch.randn(n, f)
        adj = torch.eye(n)
        out = model(x, adj)
        assert out.shape == (n, 1)

    def test_unknown_aggregator_raises(self):
        with pytest.raises(ValueError, match="Unknown aggregator"):
            model = GraphSAGE(in_features=4, aggregator='max')
            x = torch.randn(5, 4)
            model(x, torch.eye(5))

    def test_no_nan_in_output(self):
        n, f = 12, 8
        model = GraphSAGE(in_features=f, hidden_dim=16, aggregator='mean')
        x = torch.randn(n, f)
        adj = torch.rand(n, n).clamp(0, 1)
        out = model(x, adj)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# FraudDetector.build_graph_data tests
# ---------------------------------------------------------------------------

class TestFraudDetectorBuildGraphData:
    def setup_method(self):
        self.G = _make_graph(n_licit=20, n_illicit=20)
        self.detector = FraudDetector(hidden_dim=16, num_layers=2, device='cpu')

    def test_returns_required_keys(self):
        data = self.detector.build_graph_data(self.G, sample_size=20, balance_classes=True)
        for key in ('x', 'y', 'adj', 'train_mask', 'test_mask', 'node_ids'):
            assert key in data

    def test_balanced_classes(self):
        data = self.detector.build_graph_data(self.G, sample_size=20, balance_classes=True)
        y = data['y'].cpu().numpy()
        assert abs(y.sum() - (1 - y).sum()) <= 1  # within one node

    def test_raises_on_empty_graph(self):
        G_empty = nx.DiGraph()
        G_empty.add_node("x")  # no label
        with pytest.raises(ValueError, match="No labeled nodes"):
            self.detector.build_graph_data(G_empty, sample_size=10)

    def test_train_test_split_sizes(self):
        data = self.detector.build_graph_data(self.G, sample_size=20, balance_classes=True)
        n = len(data['node_ids'])
        n_train = data['train_mask'].sum().item()
        n_test = data['test_mask'].sum().item()
        assert n_train + n_test == n
        assert n_train > n_test  # 80/20 split


# ---------------------------------------------------------------------------
# FraudDetector.evaluate — single-class guard (the bare-except fix)
# ---------------------------------------------------------------------------

class TestFraudDetectorEvaluate:
    def test_auc_single_class_does_not_crash(self):
        """evaluate() must not raise when the test set has only one class."""
        G = _make_graph(n_licit=20, n_illicit=0)
        # Add unlabeled illicit nodes so build_graph_data has something to do
        for i in range(5):
            G.add_node(f"i{i}", features=np.zeros(8).tolist(), label=1)

        detector = FraudDetector(hidden_dim=16, num_layers=2, device='cpu')
        data = detector.build_graph_data(G, sample_size=20, balance_classes=False)
        detector.train(data, epochs=5, early_stopping=5)
        metrics = detector.evaluate(data)
        assert 'auc' in metrics
        assert metrics['auc'] == 0.0 or 0.0 <= metrics['auc'] <= 1.0


# ---------------------------------------------------------------------------
# preprocess_data — file existence guard
# ---------------------------------------------------------------------------

class TestPreprocessDataGuard:
    def test_raises_on_missing_files(self, tmp_path):
        from scripts.preprocess_data import main as preprocess_main
        import scripts.preprocess_data as ppd

        original_raw = ppd.RAW_DIR
        original_processed = ppd.PROCESSED_DIR
        try:
            ppd.RAW_DIR = tmp_path / "raw" / "elliptic_bitcoin_dataset"
            ppd.PROCESSED_DIR = tmp_path / "processed"
            with pytest.raises(FileNotFoundError, match="Missing raw data files"):
                preprocess_main()
        finally:
            ppd.RAW_DIR = original_raw
            ppd.PROCESSED_DIR = original_processed
