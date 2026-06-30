"""Data structure for BallTree search of nearest representatives"""

import numpy as np
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Union
from sklearn.neighbors import BallTree

from .base import BaseSearcher
from classifiers.utils import *
from classifiers.metrics import MetricSpace

class BallTreeSearcher(BaseSearcher):
    """
    Nearest representative search using BallTree
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
        self._n_representatives = 0
        self.fitted = False

    def fit(self, X: np.ndarray, r: np.ndarray) -> 'BallTreeSearcher':
        """
        Build index

        Args:
            X: representative points
            r: radii of representatives
        """
        self._n_representatives = len(X)
        self.tree = BallTree(X.copy(), metric=self.metric.metric_name_sklearn)
        self.max_radius = np.max(r)
        self.fitted = True
        return self

    def _predict_1batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find nearest elements for multiple points (without additional memory optimization)

        Returns:
            (indices_of_nearest_elements, distances_to_nearest_elements)
        """
        if not self.fitted:
            return -np.ones(len(X), dtype='int'), np.array([float('inf')] * len(X))

        use_radius = False # affects calculation time only
        only_count_within_radius = False # affects calculation time only
        use_argmin = True # affects calculation time only
        # (use_radius = False):
        # finds nearest element and ignores radius
        # (use_radius = True, only_count_within_radius = False):
        # finds all elements within given radius and either sorts them (use_argmin = False) or takes minimum (use_argmin = True)
        # (use_radius = True, only_count_within_radius = True):
        # returns number of elements within given radius, and where it's non-zero, finds nearest element again

        if use_radius:
            if only_count_within_radius:
                nearest_reprs_count = self.tree.query_radius(X, self.max_radius, count_only=True)
                mask = nearest_reprs_count > 0
                nearest_reprs_idx = -np.ones(len(X), dtype='int')
                nearest_reprs_dists = np.zeros(len(X), dtype='float')
                nearest_reprs_dists[~mask] = float('inf')
                if np.any(mask):
                    nearest_reprs_dists_multi, nearest_reprs_idx_multi = self.tree.query(X[mask], k=1)
                    nearest_reprs_idx[mask] = nearest_reprs_idx_multi[:, 0]
                    nearest_reprs_dists[mask] = nearest_reprs_dists_multi[:, 0]
            else:
                if use_argmin:
                    nearest_reprs_idx_multi, nearest_reprs_dists_multi = self.tree.query_radius(
                        X, self.max_radius, count_only=False, return_distance=True, sort_results=False)
                    nearest_reprs_argmins = np.array([np.argmin(dists) if len(dists) > 0 else -1
                        for dists in nearest_reprs_dists_multi])
                    nearest_reprs_idx = np.array([indices[argmin] if argmin >= 0 else -1
                        for indices, argmin in zip(nearest_reprs_idx_multi, nearest_reprs_argmins)])
                    nearest_reprs_dists = np.array([dists[argmin] if argmin >= 0 else float('inf')
                        for dists, argmin in zip(nearest_reprs_dists_multi, nearest_reprs_argmins)])
                else:
                    nearest_reprs_idx_multi, nearest_reprs_dists_multi = self.tree.query_radius(
                        X, self.max_radius, count_only=False, return_distance=True, sort_results=True)
                    nearest_reprs_idx = np.array([arr[0] if len(arr) > 0 else -1 for arr in nearest_reprs_idx_multi])
                    nearest_reprs_dists = np.array([arr[0] if len(arr) > 0 else float('inf') for arr in nearest_reprs_dists_multi])
        else:
            nearest_reprs_dists_multi, nearest_reprs_idx_multi = self.tree.query(X, k=1)
            nearest_reprs_idx = nearest_reprs_idx_multi[:, 0]
            nearest_reprs_dists = nearest_reprs_dists_multi[:, 0]
        return nearest_reprs_idx, nearest_reprs_dists

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find nearest elements for multiple points

        Returns:
            (indices_of_nearest_elements, distances_to_nearest_elements)
        """
        max_gb = 4 # don't allocate an array more than max_gb GB
        if not self.fitted:
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
        return self._n_representatives
