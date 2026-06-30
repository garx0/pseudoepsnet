"""Data structure for linear search of nearest representatives"""

import numpy as np
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Union

from .base import BaseSearcher
from dte_stand.algorithm.mate.lib.classifiers.utils import *
from dte_stand.algorithm.mate.lib.classifiers.metrics import MetricSpace

class LinearSearcher(BaseSearcher):
    """
    Linear search of nearest representatives
    """

    def __init__(self,
                 metric: Union[str, MetricSpace],
                 agent: int,
                 iteration: int,
                 config):
        """
        Initialize search structure

        Args:
            metric: metric ('euclidean', 'manhattan', etc.) or MetricSpace
            agent: which agent the classifier is for
            iteration: which iteration the classifier is for
            config: input parameters
        """
        # Metric
        self.metric = metric if isinstance(metric, MetricSpace) else MetricSpace(metric)

        # Parameters
        self.verbose = config.verbose

        # State
        self.representatives: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, r: np.ndarray) -> 'LinearSearcher':
        """
        Build index

        Args:
            X: representative points
            r: radii of representatives
        """
        self.representatives = np.array(X)
        return self

    def _predict_1batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find nearest elements for multiple points (without additional memory optimization)

        Returns:
            (indices_of_nearest_elements, distances_to_nearest_elements)
        """
        if len(self.representatives) == 0:
            return -np.ones(len(X), dtype='int'), np.array([float('inf')] * len(X))

        dists = self.metric(X[:, None, :], self.representatives[None, :, :])
        nearest_reprs_idx = np.argmin(dists, axis=1)
        nearest_reprs_dists = dists[np.arange(len(dists)), nearest_reprs_idx]
        return nearest_reprs_idx, nearest_reprs_dists

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find nearest elements for multiple points

        Returns:
            (indices_of_nearest_elements, distances_to_nearest_elements)
        """
        max_gb = 4 # don't allocate an array more than max_gb GB
        if len(self.representatives) > 0:
            batch_size = min(len(X), int(max_gb * 1024**3 / 8 / X.shape[-1] / len(self.representatives)))
        else:
            batch_size = len(X)
        if batch_size == len(X):
            return self._predict_1batch(X)
        else:
            shift = 0
            nearest_reprs_idx = np.zeros(len(X), dtype='int')
            nearest_reprs_dists = np.zeros(len(X), dtype='float')
            while shift < len(X):
                nearest_reprs_idx_batch, nearest_reprs_dists_batch = self._predict_1batch(X[shift:shift + batch_size])
                nearest_reprs_idx[shift:shift + batch_size] = nearest_reprs_idx_batch
                nearest_reprs_dists[shift:shift + batch_size] = nearest_reprs_dists_batch
                shift += batch_size
            return nearest_reprs_idx, nearest_reprs_dists

    @property
    def n_representatives(self) -> int:
        return len(self.representatives)
