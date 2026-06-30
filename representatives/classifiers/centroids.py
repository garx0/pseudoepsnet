"""Centroids classifier - one representative per class"""

import numpy as np
from typing import Tuple, Union
from .base import BaseClassifier
from .metrics import MetricSpace
from classifiers.search import SearcherFactory


class CentroidsClassifier(BaseClassifier):
    def __init__(self, metric: Union[str, MetricSpace], agent: int, iteration: int, config):
        self.metric = metric if isinstance(metric, MetricSpace) else MetricSpace(metric)
        self.theta = config.theta
        self.train_len = 0
        self.searcher = SearcherFactory.create(config.searcher, metric=self.metric,
                                               agent=agent, iteration=iteration, config=config)
        self._centroids = None
        self._radiuses = None
        self._labels = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CentroidsClassifier':
        self.train_len = len(X)
        X, y = np.array(X), np.array(y)

        centroids, radiuses, labels = [], [], []
        for label in np.unique(y):
            pts = X[y == label]
            if len(pts) == 0:
                continue
            centroid = np.mean(pts, axis=0)
            radius = np.max(self.metric(pts, centroid)) if len(pts) > 0 else 0.0
            centroids.append(centroid)
            radiuses.append(radius)
            labels.append(label)

        self._centroids = np.array(centroids)
        self._radiuses = np.array(radiuses)
        self._labels = np.array(labels)
        self.searcher.fit(self._centroids, self._radiuses)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._centroids is None:
            raise ValueError("Not fitted")

        X = np.array(X)
        n = len(X)
        pred = np.zeros(n, dtype=int)
        conf = np.zeros(n, dtype=float)
        conf_raw = np.zeros(n, dtype=float)
        is_conf = np.zeros(n, dtype=bool)
        covered = np.zeros(n, dtype=bool)
        for_update = np.zeros(n, dtype=bool)

        idxs, dists = self.searcher(X)

        for i, (idx, d) in enumerate(zip(idxs, dists)):
            if idx == -1:
                pred[i] = -1
                continue

            r = self._radiuses[idx]
            # covered[i] = d <= r
            covered[i] = True
            # confidence = 1 - d / r if r > 0 else float(d == 0)
            confidence = 1 if r > 0 else float(d == 0)
            conf_raw[i] = confidence
            is_conf[i] = confidence >= self.theta
            if covered[i] and is_conf[i]:
                pred[i] = self._labels[idx]
                conf[i] = confidence
            else:
                pred[i] = -1
                conf[i] = 0.
                for_update[i] = True

        return pred, conf, is_conf, covered, conf_raw, for_update

    def are_points_covered(self, points: np.ndarray) -> np.ndarray:
        if self._centroids is None:
            return np.zeros(len(points), dtype=bool)
        idxs, dists = self.searcher(np.array(points))
        return np.array([i != -1 and d <= self._radiuses[i] for i, d in zip(idxs, dists)])

    @property
    def n_representatives(self) -> int:
        return len(self._centroids) if self._centroids is not None else 0

    @property
    def train_data_len(self) -> int:
        return self.train_len

    def __repr__(self) -> str:
        return f"CentroidsClassifier(metric={self.metric.metric_name}, theta={self.theta}, n={self.n_representatives})"
