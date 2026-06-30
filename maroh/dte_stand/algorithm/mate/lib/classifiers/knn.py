"""KNN classifier"""

import numpy as np
from typing import List, Tuple, Optional
from .base import BaseClassifier
from sklearn.neighbors import BallTree

class kNNClassifier(BaseClassifier):
    """
    K-Nearest Neighbors classifier

    Parameters:
        k: number of neighbors for voting
    """

    def __init__(self, k: int = 5, **kwargs):
        self.k = k
        self._X_train = None
        self._y_train = None
        self._is_fitted = False

    def fit(self, X, y):
        """
        Train classifier

        Args:
            X, y: training data - points and labels
        """
        self._X_train = np.array(X)
        self._y_train = np.array(y)
        self.tree = BallTree(self._X_train, metric='euclidean')
        self._is_fitted = True

    def predict(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Classify points

        Returns:
            (predicted_classes, confidences, whether_classification_was_confident,
            whether_points_are_covered_by_balls, raw_confidences)
            here classification is always confident
        """
        if not self._is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")

        pred_labels = np.zeros(len(points), dtype=int)
        confidences = np.zeros(len(points), dtype=float)
        are_confident = np.zeros(len(points), dtype=bool)
        for_update = np.zeros(len(points), dtype=bool)

        vectorize_for_all_points = False # True option is somehow slower
        for i, point in enumerate(points):
            # Find k nearest neighbors
            k_dists, k_indices = self.tree.query([point], k=self.k)
            k_indices = k_indices[0]
            k_dists = k_dists[0]
            k_labels = self._y_train[k_indices]

            # Voting
            unique, counts = np.unique(k_labels, return_counts=True)
            counts_argmax = np.argmax(counts)
            predicted = unique[counts_argmax]
            confidence = counts[counts_argmax] / self.k
            pred_labels[i] = predicted
            confidences[i] = confidence
            are_confident[i] = True
        are_covered = np.array([self._is_fitted] * len(points))
        return pred_labels, confidences, are_confident, are_covered, confidences.copy(), for_update

    def are_points_covered(self, points: np.ndarray) -> bool:
        """For KNN all points are covered"""
        return np.array([self._is_fitted] * len(points))

    @property
    def n_representatives(self) -> int:
        """Number of representatives (all training points)"""
        return len(self._X_train) if self._is_fitted else 0

    @property
    def train_data_len(self) -> int:
        """Number of elements in training set"""
        return self.n_representatives

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k}, n_reprs={self.n_representatives})"
