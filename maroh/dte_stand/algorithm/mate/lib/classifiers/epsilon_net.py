"""Unified classifier based on pseudo-ε-net"""

import os
import numpy as np
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Union
from sklearn.neighbors import BallTree
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from datetime import datetime

from .base import BaseClassifier
from .utils import *
from .metrics import MetricSpace
from dte_stand.algorithm.mate.lib.classifiers.search import SearcherFactory

class EpsilonNetClassifier(BaseClassifier):
    """
    Unified classifier based on pseudo-ε-net.
    Combines network construction, streaming classification, and buffer.
    """

    def __init__(self,
                 metric: Union[str, MetricSpace],
                 agent: int,
                 iteration: int,
                 config):
        """
        Initialize classifier

        Args:
            metric: metric ('euclidean', 'manhattan', etc.) or MetricSpace
            agent: which agent the classifier is for
            iteration: which iteration the classifier is for
            config: input parameters
        """
        # Metric
        self.metric = metric if isinstance(metric, MetricSpace) else MetricSpace(metric)

        # Parameters
        self.delta = config.delta
        self.theta = config.theta
        self.buffer_size = config.buffer_size
        self.dist_percentile = config.dist_percentile
        self.use_acceleration = config.use_acceleration
        self.verbose = config.verbose
        self.agent = agent
        self.iteration = iteration
        self.use_antireprs = not config.disable_antireprs if hasattr(config, 'disable_antireprs') else True
        self.use_adaptive_radius = not config.disable_adaptive_radius if hasattr(config, 'disable_adaptive_radius') else True
        self.ndim = None
        # State
        self.dists = None
        self.searcher_construct = lambda: SearcherFactory.create(config.searcher,
                                               metric = self.metric,
                                               agent = agent,
                                               iteration = iteration,
                                               config = config)
        self.searcher = self.searcher_construct()
        if hasattr(config, 'saved_dists'):
            try:
                loaded = np.load(os.path.join(config.data_path, config.saved_dists), allow_pickle=True)
                p = format_percentile_key(self.dist_percentile)
                if f"a{agent}_i{iteration}" in loaded and p in loaded[f"a{agent}_i{iteration}"].item():
                    dists_p = loaded[f"a{agent}_i{iteration}"].item()
                    self.dists = dists_p[p]
            except Exception as e:
                if self.verbose:
                    print(f" Warning: {e}")
                    print(f" Saved distances could not be loaded, calculating distances from scratch")

    # ================== Network construction ==================

    def _distance_between_classes(self, class_points: Dict[int, np.ndarray], calc_percentile=True) -> Dict[int, float]:
        """Calculate percentiles and minimum distances from each class to other classes"""
        if self.dists is not None:
            # won't work with precalculated percentiles now, need to precalculate minimums
            # in the corresponding script as well, like here
            return self.dists, None

        distances = {}
        for a, points_a in class_points.items():
            for b, points_b in class_points.items():
                if a < b:
                    # result of self.metric() before flatten - dimension (len(points_a), len(points_b)),
                    # after flatten - first distances from 0-th point of class a
                    # to all points of class b, then from 1-st point of class a to them, etc.
                    if len(points_a) == 0 or len(points_b) == 0:
                        distances[(a, b)] = np.array([])
                    else:
                        distances[(a, b)] = pairwise_distances(points_a, points_b, metric=self.metric.metric_name_sklearn)


        if calc_percentile:
            percentile_dist_to_other = {}
            for a in class_points.keys():
                dists_a = []
                for b in class_points.keys():
                    if a != b:
                        a1, b1 = (a, b) if a < b else (b, a)
                        dists_a.append(distances.get((a1, b1), np.array([])).flatten())
                dists_a = np.concatenate(dists_a) if len(dists_a) > 0 else np.array([])
                percentile_dist_to_other[a] = np.percentile(dists_a, self.dist_percentile) if len(dists_a) > 0 else float('inf')

        min_dist_to_other = {}
        for a, points in class_points.items():
            dists_a_min = None
            for b in class_points.keys():
                if a < b:
                    dists_ab_full = distances.get((a, b), np.array([]))
                elif a > b:
                    dists_ab_full = distances.get((b, a), np.array([])).T
                else:
                    continue
                if len(dists_ab_full) == 0:
                    continue
                dists_ab = np.min(dists_ab_full, axis=1)
                if dists_a_min is None:
                    dists_a_min = dists_ab
                else:
                    print("ENTERED BRANCH THAT HAD BUG")
                    dists_a_min = np.minimum(dists_a_min, dists_ab)
            if dists_a_min is None:
                if calc_percentile:
                    epsilon = (1 - self.delta) * percentile_dist_to_other.get(a, 1.0) / 2
                else:
                    epsilon = 0
                dists_a_min = np.full(len(points), epsilon)
            min_dist_to_other[a] = dists_a_min
        if calc_percentile:
            return percentile_dist_to_other, min_dist_to_other
        else:
            return min_dist_to_other


    def _create_uncertainty_representatives(self, class_points: Dict[int, np.ndarray], min_radius: float,
        min_dist_to_other: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create anti-representatives in areas where classes are close together

        Args:
            class_points: dictionary {label: points of class}

        Returns:
            list of anti-representatives
        """
        antirepr_centers = np.empty((0, self.ndim), dtype=float)

        # For each pair of classes
        classes = list(class_points.keys())
        classes = sorted(classes, key=lambda c: len(class_points[c]))
        for i, label_a in enumerate(classes[:-1]):
            antirepr_centers = np.append(antirepr_centers,
                class_points[label_a][min_dist_to_other[label_a] < min_radius], axis=0)

        antirepr_labels = np.full(len(antirepr_centers), -1, dtype=int)
        antirepr_radiuses = np.full(len(antirepr_centers), min_radius, dtype=float)

        return antirepr_centers, antirepr_labels, antirepr_radiuses

    def _merge_antireprs(self, antirepr_centers: np.ndarray, antirepr_labels: np.ndarray, antirepr_radiuses: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove anti-representatives whose centers lie inside others"""
        if len(antirepr_centers) <= 1:
            return antirepr_centers, antirepr_labels, antirepr_radiuses

        keep = np.zeros(len(antirepr_centers), dtype=bool)
        for r1_idx, r1_center in enumerate(antirepr_centers):
            # Check if r1's center is inside any already kept
            if not np.any(keep) or np.all(self.metric(r1_center, antirepr_centers[keep]) > antirepr_radiuses[keep]):
                keep[r1_idx] = True
        return antirepr_centers[keep], antirepr_labels[keep], antirepr_radiuses[keep]

    def _filter_points_with_dists(self, points: np.ndarray, dists: np.ndarray, class_label: int,
        antirepr_centers: np.ndarray, antirepr_radiuses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove points falling into anti-representatives, and synchronously filter distances"""
        if len(antirepr_centers) == 0:
            return points, dists

        keep = np.ones(len(points), dtype=bool)
        dists_to_a = pairwise_distances(points, antirepr_centers)
        keep = np.all(dists_to_a > antirepr_radiuses, axis=1)

        return points[keep], dists[keep]

    def _build_greedy(self, points: np.ndarray, dists_to_other: np.ndarray, epsilon: float, class_label: int) \
        -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Greedy algorithm for building pseudo-ε-net"""
        n = len(points)
        if n == 0:
            return []
        points = points.copy()
        if class_label is not None:
            # Sort points: first those with LARGER distance to other classes
            sorted_indices = np.argsort(-dists_to_other)  # minus sign for descending sort
            points = points[sorted_indices]
            # Save sorted distances for fast access
            sorted_dists_to_other = dists_to_other[sorted_indices]
        else:
            sorted_dists_to_other = None
        representatives = []
        radiuses = []
        if self.use_acceleration and n > 4000:
            tree = BallTree(points.copy(), metric=self.metric.metric_name)
            remaining_idx = np.arange(n)
            remaining = points.copy()
            remaining_dists = sorted_dists_to_other.copy() if sorted_dists_to_other is not None else None
            while len(remaining_idx) > 0:
                rep = remaining[0]
                radius = epsilon
                if sorted_dists_to_other is not None:
                    dist_to_other = remaining_dists[0]

                    if self.use_adaptive_radius:
                        # Radius = distance to other class, but not less than epsilon
                        radius = max(epsilon, dist_to_other * 0.95 / 2)  # 0.95 - margin
                    else:
                        radius = epsilon
                representatives.append(rep)
                radiuses.append(radius)
                remaining_idx = remaining_idx[1:]
                remaining = remaining[1:]
                if remaining_dists is not None:
                    remaining_dists = remaining_dists[1:]
                if len(remaining_idx) == 0:
                    break
                if len(remaining_idx) > 3000:
                    indices = tree.query_radius([rep], r=radius)[0]
                    mask = ~np.isin(remaining_idx, indices, assume_unique=True)
                    remaining_idx = remaining_idx[mask]
                    remaining = remaining[mask]
                    if remaining_dists is not None:
                        remaining_dists = remaining_dists[mask]
                else:
                    remaining_dists_to_rep = self.metric(remaining, rep)
                    mask = remaining_dists_to_rep > radius
                    remaining_idx = remaining_idx[mask]
                    remaining = remaining[mask]
                    if remaining_dists is not None:
                        remaining_dists = remaining_dists[mask]
        else:
            remaining = points.copy()
            remaining_dists = sorted_dists_to_other.copy() if sorted_dists_to_other is not None else None
            while len(remaining) > 0:
                rep = remaining[0]
                radius = epsilon
                if remaining_dists is not None:
                    dist_to_other = remaining_dists[0]

                    # Radius = distance to other class, but not less than epsilon
                    if self.use_adaptive_radius:
                        radius = max(epsilon, dist_to_other * 0.95 / 2)  # 0.95 - margin
                    else:
                        radius = epsilon
                representatives.append(rep)
                radiuses.append(radius)
                remaining = remaining[1:]
                if remaining_dists is not None:
                    remaining_dists = remaining_dists[1:]
                if len(remaining) == 0:
                    break
                remaining_dists_to_rep = self.metric(remaining, rep)
                mask = remaining_dists_to_rep > radius
                remaining = remaining[mask]
                if remaining_dists is not None:
                    remaining_dists = remaining_dists[mask]

        return representatives, radiuses

    def _append_representatives(self, repr_centers, repr_labels, repr_radiuses):
        self.repr_centers = np.append(self.repr_centers, repr_centers, axis=0)
        self.repr_labels = np.append(self.repr_labels, repr_labels, axis=0)
        self.repr_radiuses = np.append(self.repr_radiuses, repr_radiuses, axis=0)

    def _append_antireprs(self, antirepr_centers, antirepr_radiuses):
        self._antirepr_centers = np.append(self._antirepr_centers, antirepr_centers, axis=0)
        self._antirepr_radiuses = np.append(self._antirepr_radiuses, antirepr_radiuses, axis=0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EpsilonNetClassifier':
        """
        Train classifier (build ε-net)

        Args:
            X, y: training data - points and labels

        Returns:
            self for method chaining
        """
        self.train_len = len(X)
        self.ndim = X.shape[1]

        # Group by class
        class_points = {}
        for label in np.unique(y):
            class_points[label] = X[y == label]

        classes = list(class_points.keys())

        # Distances between classes
        percentile_dist_to_other, min_dist_to_other = self._distance_between_classes(class_points)
        self.percentile_dist_to_other = percentile_dist_to_other

        # Build network for each class
        self.repr_centers = np.empty((0, self.ndim), dtype=float)
        self.repr_labels = np.empty(0, dtype=int)
        self.repr_radiuses = np.empty(0, dtype=float)
        rad = max([self.percentile_dist_to_other.get(x, 1.0) for x in class_points.keys()])
        if self.use_antireprs:
            antirepr_centers, antirepr_labels, antirepr_radiuses = \
                self._create_uncertainty_representatives(class_points, rad, min_dist_to_other)
            antirepr_centers, antirepr_labels, antirepr_radiuses = \
                self._merge_antireprs(antirepr_centers, antirepr_labels, antirepr_radiuses)
            self._antirepr_centers = antirepr_centers
            self._antirepr_radiuses = antirepr_radiuses
            self._append_representatives(antirepr_centers, antirepr_labels, antirepr_radiuses)
            if self.verbose:
                print(f"  Added {len(antirepr_centers)} anti-representatives")
        else:
            self._antirepr_centers = np.empty((0, self.ndim), dtype=float)
            self._antirepr_radiuses = np.empty(0, dtype=float)


        for label, points in class_points.items():
            dists = min_dist_to_other[label]
            if self.use_antireprs:
                points, dists = self._filter_points_with_dists(points, dists, label, self._antirepr_centers, self._antirepr_radiuses)

            if len(points) == 0:
                continue

            # Update _min_dist_to_other for this class
            min_dist_to_other[label] = dists
            eps = (1 - self.delta) * percentile_dist_to_other.get(label, 1.0) / 2
            repr_centers, repr_radiuses = self._build_greedy(points, dists, eps, label)
            repr_labels = np.full(len(repr_centers), label, dtype=int)
            self._append_representatives(repr_centers, repr_labels, repr_radiuses)

        self.searcher.fit(self.repr_centers, self.repr_radiuses)

        return self

    def update(self, X, y):
        t_start = datetime.now()
        # Group by class
        class_points = {}
        for label in np.unique(y):
            class_points[label] = X[y == label]

        classes = list(class_points.keys())

        # Distances between classes
        min_dist_to_other = self._distance_between_classes(class_points, calc_percentile=False)

        if self.ndim is None:
            self.ndim = X.shape[1]

        # Build network for each class
        rad = max([self.percentile_dist_to_other.get(x, 1.0) for x in class_points.keys()])
        if self.use_antireprs:
            antirepr_centers, antirepr_labels, antirepr_radiuses = \
                self._create_uncertainty_representatives(class_points, rad, min_dist_to_other)
            antirepr_centers, antirepr_labels, antirepr_radiuses = \
                self._merge_antireprs(antirepr_centers, antirepr_labels, antirepr_radiuses)
            self._append_antireprs(antirepr_centers, antirepr_radiuses)
            self._append_representatives(antirepr_centers, antirepr_labels, antirepr_radiuses)
            # if self.verbose:
                # print(f"  Added {len(antirepr_centers)} anti-representatives")

        repr_points_added = 0
        for label, points in class_points.items():
            dists = min_dist_to_other[label]
            if self.use_antireprs:
                points, dists = self._filter_points_with_dists(points, dists, label, antirepr_centers, antirepr_radiuses)

            if len(points) == 0:
                continue

            # Update _min_dist_to_other for this class
            min_dist_to_other[label] = dists
            eps = (1 - self.delta) * self.percentile_dist_to_other.get(label, 1.0) / 2
            repr_centers, repr_radiuses = self._build_greedy(points, dists, eps, label)
            repr_labels = np.full(len(repr_centers), label, dtype=int)
            repr_points_added += len(repr_centers)
            self._append_representatives(repr_centers, repr_labels, repr_radiuses)
        t_end = datetime.now()
        # if self.verbose:
            # print(f"  Added {repr_points_added} representatives")


        self.searcher = self.searcher_construct()
        self.searcher.fit(self.repr_centers, self.repr_radiuses)

        self.train_len += len(X)
        return (t_end - t_start).total_seconds()

    def _find_nearest(self, X) -> Tuple[np.ndarray, np.ndarray]:
        reps_idx, dists = self.searcher(X)
        return reps_idx, dists

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Classify array of points

        Returns:
            (predicted_classes, confidences, whether_classification_was_confident,
            whether_points_are_covered_by_balls, raw_confidences)
        """
        # reps, dists = self._find_nearest(X)
        reps_idx, dists = self._find_nearest(X)
        repr_radiuses = self.repr_radiuses[reps_idx]
        repr_labels = self.repr_labels[reps_idx]
        are_covered = (dists < repr_radiuses) & (repr_labels >= 0)

        pred = np.zeros(len(X), dtype=int)
        confidences = np.zeros(len(X), dtype=float)
        confidences_raw = np.zeros(len(X), dtype=float)
        are_confident = np.zeros(len(X), dtype=bool)
        for_update = np.zeros(len(X), dtype=bool)
        # for i, (point, rep, dist) in enumerate(zip(X, reps, dists)):
        for i, (point, rad, label, dist) in enumerate(zip(X, repr_radiuses, repr_labels, dists)):
            if rad > 0:
                confidence = 1 - dist / rad
            else:
                confidence = 1.0 if dist == 0 else 0.0
            confidences_raw[i] = confidence if label != -1 else 0.
            # fell into radius and label >= 0
            if confidence >= self.theta and label != -1:
                pred[i] = label
                confidences[i] = confidence
                are_confident[i] = True
            else:
                if confidence < self.theta:
                    # didn't fall into anti-representative area or other representative area
                    for_update[i] = True
                pred[i] = -1
                confidences[i] = 0.
                are_confident[i] = False
        return pred, confidences, are_confident, are_covered, confidences_raw, for_update

    def are_points_covered(self, points: np.ndarray) -> np.ndarray:
        """Check whether points are covered by balls"""
        reps_idx, dists = self._find_nearest(points)
        repr_radiuses = self.repr_radiuses[reps_idx]
        repr_labels = self.repr_labels[reps_idx]
        return (dists < repr_radiuses) & (repr_labels >= 0)

    @property
    def n_representatives(self) -> int:
        return len(self.repr_centers)

    @property
    def train_data_len(self) -> int:
        return self.train_len

    # ================== Save/Load ==================

    def get_params(self) -> Dict:
        """Get classifier parameters"""
        return {
            'metric': self.metric.metric_name,
            'delta': self.delta,
            'theta': self.theta,
            'buffer_size': self.buffer_size,
            'use_acceleration': self.use_acceleration,
            'n_representatives': self.n_representatives
        }

    def load_representatives(self, filename):
        """Load ready representatives from npz file"""
        print("loading representatives...")

        data = np.load(filename)
        self.repr_centers = np.array(data["centers"])
        self.repr_radiuses = np.array(data["radiuses"])
        self.repr_labels = np.array(data["labels"])
        self.searcher.fit(self.repr_centers, self.repr_radiuses)
        self._antirepr_centers = self.repr_centers[self.repr_labels == -1]
        self._antirepr_radiuses = self.repr_radiuses[self.repr_labels == -1]
        classes = sorted([x for x in np.unique(self.repr_labels) if x >= 0])
        self.percentile_dist_to_other = dict(zip(classes, data["percentiles"]))
        self.train_len = data["train_len"][0]
        if len(self.repr_centers) > 0:
            self.ndim = self.repr_centers.shape[1]
        else:
            self.ndim = None

        print(f"loaded {len(self.repr_centers)} representatives")

    def save_representatives(self, filename):
        """Save ready representatives to npz file"""

        classes = sorted([x for x in np.unique(self.repr_labels) if x >= 0])
        percentiles = np.array([self.percentile_dist_to_other[k] for k in classes])
        np.savez(filename, centers=self.repr_centers, radiuses=self.repr_radiuses,
            labels=self.repr_labels, percentiles=percentiles, train_len=np.array([self.train_len]))
