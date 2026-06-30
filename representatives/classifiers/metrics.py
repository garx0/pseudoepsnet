"""Metrics and distances"""

import numpy as np

class MetricSpace:
    """Metric space with various metrics"""

    @staticmethod
    def euclidean(x: np.ndarray, y: np.ndarray) -> float:
        return np.linalg.norm(x - y, axis=-1, ord=2)

    @staticmethod
    def manhattan(x: np.ndarray, y: np.ndarray) -> float:
        return np.sum(np.abs(x - y), axis=-1)

    @staticmethod
    def chebyshev(x: np.ndarray, y: np.ndarray) -> float:
        return np.max(np.abs(x - y), axis=-1)

    @staticmethod
    def cosine(x: np.ndarray, y: np.ndarray) -> float:
        denoms = np.linalg.norm(x, axis=-1, ord=2) * np.linalg.norm(y, axis=-1, ord=2)
        denoms[denoms == 0] = 1
        return 1. - np.sum(x * y, axis=-1) / denoms

    # Dictionary for fast metric access
    _METRICS = {
        'euclidean': euclidean,
        'manhattan': manhattan,
        'chebyshev': chebyshev,
        'cosine': cosine
    }

    _METRICS_sklearn = {
        'euclidean': 'euclidean',
        'manhattan': 'cityblock',
        'chebyshev': 'chebyshev',
        'cosine': 'cosine'
    }

    def __init__(self, metric: str = 'euclidean'):
        if metric not in self._METRICS:
            raise ValueError(f"Unknown metric: {metric}. Choose from {list(self._METRICS.keys())}")

        self.metric_name = metric
        self.metric_name_sklearn = self._METRICS_sklearn[metric]
        self.distance = self._METRICS[metric]

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """Convenient call: space(x, y)"""
        return self.distance(x, y)
