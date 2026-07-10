"""LinUCB contextual bandit — ridge-regularised per-arm linear model."""
import numpy as np
from typing import List, Tuple


class LinUCBBandit:
    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.A = [np.eye(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> Tuple[int, np.ndarray]:
        ctx = context.astype(np.float64)
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            theta = np.linalg.solve(self.A[a], self.b[a])
            variance = float(ctx @ np.linalg.solve(self.A[a], ctx))
            scores[a] = theta @ ctx + self.alpha * np.sqrt(max(variance, 0.0))
        return int(np.argmax(scores)), scores

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        ctx = context.astype(np.float64)
        self.A[arm] += np.outer(ctx, ctx)
        self.b[arm] += reward * ctx

    def get_weights(self) -> List[np.ndarray]:
        return [np.linalg.solve(self.A[a], self.b[a]) for a in range(self.n_arms)]
