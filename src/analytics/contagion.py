"""
Fraud Contagion Score Module
Models how fraud propagates through the transaction graph using an adapted
SIR (Susceptible-Infected-Recovered) diffusion process.

For each seed node, 3-step diffusion counts how many additional nodes become
"at risk".  The contagion multiplier = at-risk neighbours / 1 missed node.
"""
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


def _diffuse(G: nx.DiGraph, seed: str, steps: int = 3,
             infection_prob: float = 0.30) -> Set[str]:
    """
    Stochastic SIR diffusion from a single seed node.
    Returns the set of nodes that entered the 'Infected' state.
    """
    infected: Set[str] = {seed}
    frontier: Set[str] = {seed}
    rng = np.random.default_rng(abs(hash(seed)) % (2**31))

    for _ in range(steps):
        next_frontier: Set[str] = set()
        for node in frontier:
            for nbr in G.neighbors(node):
                if nbr not in infected:
                    if rng.random() < infection_prob:
                        infected.add(nbr)
                        next_frontier.add(nbr)
        frontier = next_frontier
        if not frontier:
            break

    return infected - {seed}


class ContagionAnalyzer:
    """
    Computes a per-node Contagion Multiplier: how many additional nodes are
    put at risk if this node is missed (not detected as fraudulent).

    Contagion Score = mean at-risk count across N_RUNS stochastic diffusion runs.
    Composite Risk Score = fraud_prob × (1 + log(1 + contagion_score)).
    """

    def __init__(
        self,
        G: nx.DiGraph,
        detector=None,
        model_data: Optional[dict] = None,
        infection_prob: float = 0.30,
        diffusion_steps: int = 3,
        n_runs: int = 10,
    ):
        self.G               = G
        self.detector        = detector
        self.data            = model_data or {}
        self.infection_prob  = infection_prob
        self.diffusion_steps = diffusion_steps
        self.n_runs          = n_runs
        self._scores: Optional[pd.DataFrame] = None

    # ── Fraud probability lookup ───────────────────────────────────────────────

    def _get_fraud_probs(self) -> Dict[str, float]:
        """Returns {node_id: fraud_prob} from the model or fallback."""
        import torch

        features = self.data.get("features")
        node_ids = self.data.get("node_ids")

        if features is not None and self.detector is not None:
            try:
                self.detector.model.eval()
                with torch.no_grad():
                    x   = torch.FloatTensor(features).to(self.detector.device)
                    adj = torch.eye(len(features)).to(self.detector.device)
                    out = self.detector.model(x, adj)
                    probs = torch.sigmoid(out).squeeze().cpu().numpy()
                if probs.ndim == 0:
                    probs = np.array([float(probs)])
                return {str(nid): float(p) for nid, p in zip(node_ids, probs)}
            except Exception as e:
                logger.warning(f"Model inference failed: {e}")

        # Fallback: use label as proxy
        fallback = {}
        for n, d in self.G.nodes(data=True):
            lbl = d.get("label", -1)
            fallback[str(n)] = 0.9 if lbl == 1 else (0.05 if lbl == 0 else 0.21)
        return fallback

    # ── Core computation ───────────────────────────────────────────────────────

    def compute_scores(
        self,
        candidate_nodes: Optional[List[str]] = None,
        top_n: int = 200,
    ) -> pd.DataFrame:
        """
        Runs SIR diffusion for each candidate node and builds a ranked DataFrame.

        candidate_nodes: subset to evaluate (defaults to top_n highest-degree nodes).
        """
        if self._scores is not None:
            return self._scores

        fraud_probs = self._get_fraud_probs()

        # Select candidates — default to high-degree nodes (most dangerous if missed)
        if candidate_nodes is None:
            degrees = sorted(self.G.degree(), key=lambda x: x[1], reverse=True)
            candidate_nodes = [n for n, _ in degrees[:top_n]]

        logger.info(
            "Computing contagion scores for %d nodes (%d runs each)...",
            len(candidate_nodes), self.n_runs,
        )

        rows = []
        for node in candidate_nodes:
            at_risk_counts = []
            for _ in range(self.n_runs):
                at_risk = _diffuse(self.G, node, self.diffusion_steps, self.infection_prob)
                at_risk_counts.append(len(at_risk))

            mean_at_risk  = float(np.mean(at_risk_counts))
            max_at_risk   = int(np.max(at_risk_counts))
            fraud_prob    = fraud_probs.get(str(node), 0.21)
            label         = self.G.nodes[node].get("label", -1)
            degree        = self.G.degree(node)

            # Composite risk: fraud probability weighted by contagion potential
            composite = float(fraud_prob * (1.0 + np.log1p(mean_at_risk)))

            rows.append({
                "node_id":             str(node),
                "fraud_prob":          round(fraud_prob, 4),
                "degree":              degree,
                "label":               int(label),
                "mean_at_risk":        round(mean_at_risk, 1),
                "max_at_risk":         max_at_risk,
                "contagion_multiplier": round(mean_at_risk / max(degree, 1), 4),
                "composite_risk":      round(composite, 4),
            })

        df = pd.DataFrame(rows).sort_values("composite_risk", ascending=False)
        self._scores = df
        return df

    # ── Network-level summary ─────────────────────────────────────────────────

    def network_summary(self, scores: pd.DataFrame) -> Dict[str, Any]:
        illicit = scores[scores["label"] == 1]
        return {
            "total_candidates":    len(scores),
            "mean_at_risk":        round(float(scores["mean_at_risk"].mean()), 1),
            "max_at_risk":         int(scores["max_at_risk"].max()),
            "mean_composite_risk": round(float(scores["composite_risk"].mean()), 4),
            "top_node":            scores.iloc[0]["node_id"] if len(scores) else "N/A",
            "top_composite_risk":  round(float(scores.iloc[0]["composite_risk"]), 4) if len(scores) else 0.0,
            "illicit_mean_at_risk": round(float(illicit["mean_at_risk"].mean()), 1) if len(illicit) else 0.0,
            "infection_prob":      self.infection_prob,
            "diffusion_steps":     self.diffusion_steps,
        }
