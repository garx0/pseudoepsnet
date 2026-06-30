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
from classifiers.search import SearcherFactory

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

        # State
        self.representatives: List[Representative] = []
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


    def _create_uncertainty_representatives(self, class_points: Dict[int, np.ndarray], min_radius: float) -> List[Representative]:
        """
        Create anti-representatives in areas where classes are close together

        Args:
            class_points: dictionary {label: points of class}

        Returns:
            list of anti-representatives
        """
        uncertainty_reprs = []

        # For each pair of classes
        classes = list(class_points.keys())
        classes = sorted(classes, key=lambda c: len(class_points[c]))

        for i, label_a in enumerate(classes):
            for label_b in classes[i+1:]:
                points_a = class_points[label_a]
                points_b = class_points[label_b]

                # Find areas where classes are close
                # For each point from A find distance to nearest point from B
                if len(points_b) > 1000:
                    tree_b = BallTree(points_b, metric=self.metric.metric_name_sklearn)
                    dists, _ = tree_b.query(points_a, k=1)
                    dists = dists.flatten()
                else:
                    dists = np.min(pairwise_distances(points_a, points_b, metric=self.metric.metric_name_sklearn), axis=1)
                close_mask = dists < min_radius

                # Points that are too close to another class
                close_points = points_a[close_mask]
                close_dists = dists[close_mask]

                # Add anti-representatives
                for point in close_points:
                    uncertainty_reprs.append(Representative(
                        center=point,
                        epsilon=min_radius,
                        label=-1  # special label for uncertainty
                    ))

        return uncertainty_reprs

    def _merge_antireprs(self, antireprs: List[Representative]) -> List[Representative]:
        """Remove anti-representatives whose centers lie inside others"""
        if len(antireprs) <= 1:
            return antireprs

        keep = np.zeros(len(antireprs), dtype=bool)
        antireprs_centers = np.array([a.center for a in antireprs])
        antireprs_radiuses = np.array([a.epsilon for a in antireprs])
        for r1_idx, r1 in enumerate(antireprs):
            # Check if r1's center is inside any already kept
            if not np.any(keep) or np.all(self.metric(r1.center, antireprs_centers[keep]) > antireprs_radiuses[keep]):
                keep[r1_idx] = True
        return [antireprs[i] for i in np.where(keep)[0]]

    def _filter_points_with_dists(self, points: np.ndarray, dists: np.ndarray, class_label: int, antireprs) -> Tuple[np.ndarray, np.ndarray]:
        """Remove points falling into anti-representatives, and synchronously filter distances"""
        if len(antireprs) == 0:
            return points, dists

        keep = np.ones(len(points), dtype=bool)
        antireprs_centers = np.array([a.center for a in antireprs])
        antireprs_radiuses = np.array([a.epsilon for a in antireprs])
        dists_to_a = pairwise_distances(points, antireprs_centers)
        keep = np.all(dists_to_a > antireprs_radiuses, axis=1)

        return points[keep], dists[keep]

    def _build_greedy(self, points: np.ndarray, dists_to_other: np.ndarray, epsilon: float, class_label: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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

                    # Radius = distance to other class, but not less than epsilon
                    radius = max(epsilon, dist_to_other * 0.95)  # 0.95 - margin
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
                    radius = max(epsilon, dist_to_other * 0.95)  # 0.95 - margin
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EpsilonNetClassifier':
        """
        Train classifier (build ε-net)

        Args:
            X, y: training data - points and labels

        Returns:
            self for method chaining
        """
        self.train_len = len(X)
        # Group by class
        class_points = {}
        for label in np.unique(y):
            class_points[label] = X[y == label]

        classes = list(class_points.keys())

        # Distances between classes
        percentile_dist_to_other, min_dist_to_other = self._distance_between_classes(class_points)
        self.percentile_dist_to_other = percentile_dist_to_other

        # Build network for each class
        self.representatives = []
        rad = max([self.percentile_dist_to_other.get(x, 1.0) for x in class_points.keys()])
        antireprs = self._create_uncertainty_representatives(class_points, rad)
        antireprs = self._merge_antireprs(antireprs)
        self._antireprs = antireprs
        self.representatives.extend(antireprs)
        if self.verbose:
            print(f"  Added {len(antireprs)} anti-representatives")

        for label, points in class_points.items():
            dists = min_dist_to_other[label]
            points, dists = self._filter_points_with_dists(points, dists, label, self._antireprs)

            if len(points) == 0:
                continue

            # Update _min_dist_to_other for this class
            min_dist_to_other[label] = dists
            eps = (1 - self.delta) * percentile_dist_to_other.get(label, 1.0) / 2
            repr_points, radiuses = self._build_greedy(points, dists, eps, label)

            for p, r in zip(repr_points, radiuses):
                self.representatives.append(Representative(
                    center=p, epsilon=r, label=label
                ))

        repr_points = np.array([rep.center for rep in self.representatives])
        repr_radiuses = np.array([rep.epsilon for rep in self.representatives])
        self.searcher.fit(repr_points, repr_radiuses)

        return self

    # ================== Classification ==================

    def update(self, X, y):
        t_start = datetime.now()
        # Group by class
        class_points = {}
        for label in np.unique(y):
            class_points[label] = X[y == label]

        classes = list(class_points.keys())

        # Distances between classes
        min_dist_to_other = self._distance_between_classes(class_points, calc_percentile=False)

        # Build network for each class
        rad = max([self.percentile_dist_to_other.get(x, 1.0) for x in class_points.keys()])
        antireprs = self._create_uncertainty_representatives(class_points, rad)
        antireprs = self._merge_antireprs(antireprs)
        self._antireprs.extend(antireprs)
        self.representatives.extend(antireprs)
        if self.verbose:
            print(f"  Added {len(antireprs)} anti-representatives")

        repr_points_added = 0
        for label, points in class_points.items():
            dists = min_dist_to_other[label]
            points, dists = self._filter_points_with_dists(points, dists, label, antireprs)

            if len(points) == 0:
                continue

            # Update _min_dist_to_other for this class
            min_dist_to_other[label] = dists
            eps = (1 - self.delta) * self.percentile_dist_to_other.get(label, 1.0) / 2
            repr_points, radiuses = self._build_greedy(points, dists, eps, label)
            repr_points_added += len(repr_points)
            for p, r in zip(repr_points, radiuses):
                self.representatives.append(Representative(
                    center=p, epsilon=r, label=label
                ))
        t_end = datetime.now()
        if self.verbose:
            print(f"  Added {repr_points_added} representatives")


        repr_points = np.array([rep.center for rep in self.representatives])
        repr_radiuses = np.array([rep.epsilon for rep in self.representatives])
        self.searcher = self.searcher_construct()
        self.searcher.fit(repr_points, repr_radiuses)

        self.train_len += len(X)
        return (t_end - t_start).total_seconds()

    def _find_nearest(self, X) -> Tuple[List[Representative], np.ndarray]:
        reps_idx, dists = self.searcher(X)
        reps = [self.representatives[i] for i in reps_idx]
        return reps, dists

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Classify array of points

        Returns:
            (predicted_classes, confidences, whether_classification_was_confident,
            whether_points_are_covered_by_balls, raw_confidences)
        """
        reps, dists = self._find_nearest(X)
        are_covered = np.array([rep is not None and d < rep.epsilon and rep.label != -1 for rep, d in zip(reps, dists)])

        pred = np.zeros(len(X), dtype=int)
        confidences = np.zeros(len(X), dtype=float)
        confidences_raw = np.zeros(len(X), dtype=float)
        are_confident = np.zeros(len(X), dtype=bool)
        for_update = np.zeros(len(X), dtype=bool)
        for i, (point, rep, dist) in enumerate(zip(X, reps, dists)):
            if rep is None: # currently impossible
                pred[i] = -1
                confidences[i] = 0.
                confidences_raw[i] = 0.
                are_confident[i] = False
            else:
                if rep.epsilon > 0:
                    confidence = 1 - dist / rep.epsilon
                else:
                    confidence = 1.0 if dist == 0 else 0.0
                confidences_raw[i] = confidence if rep.label != -1 else 0.
                # fell into radius and label >= 0
                if confidence >= self.theta and rep.label != -1:
                    pred[i] = rep.label
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
        reps, dists = self._find_nearest(points)
        return np.array([rep is not None and d < rep.epsilon and rep.label != -1 for rep, d in zip(reps, dists)])

    @property
    def n_representatives(self) -> int:
        return len(self.representatives)

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

    # def set_representatives(self, representatives: List[Representative]):
    #     """Set ready representatives (without training)"""
    #     self.representatives = representatives
    #     repr_points = np.array([rep.center for rep in self.representatives])
    #     repr_radiuses = np.array([rep.epsilon for rep in self.representatives])
    #     self.searcher.fit(repr_points, repr_radiuses)
    #     self.train_len = 0
    #     return self

    def load_representatives(self, filename):
        """Load ready representatives from npz file"""
        print("loading representatives...")
        data = np.load(filename)
        self.representatives = [Representative(center, radius, label) for
            center, radius, label in zip(data["centers"], data["radiuses"], data["labels"])]
        self.searcher.fit(data["centers"], data["radiuses"])
        self._antireprs = [r for r in self.representatives if r.label == -1]
        classes = sorted([x for x in np.unique(data["labels"]) if x >= 0])
        self.percentile_dist_to_other = dict(zip(classes, data["percentiles"]))
        self.train_len = data["train_len"][0]
        print(f"loaded {len(self.representatives)} representatives")

    def save_representatives(self, filename):
        """Save ready representatives to npz file"""
        centers = np.array([r.center for r in self.representatives])
        radiuses = np.array([r.epsilon for r in self.representatives])
        labels = np.array([r.label for r in self.representatives])
        classes = sorted([x for x in np.unique(labels) if x >= 0])
        percentiles = np.array([self.percentile_dist_to_other[k] for k in classes])
        np.savez(filename, centers=centers, radiuses=radiuses, labels=labels, percentiles=percentiles, train_len=np.array([self.train_len]))
