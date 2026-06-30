"""Base abstract class for all classifiers"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple, List, Dict
from .utils import *

class BaseClassifier(ABC):
    """Base class for all classifiers"""
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train classifier

        Args:
            X: training data points
            y: training data labels
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Classify multiple points

        Returns:
            (predicted_classes (-1 instead of None), confidences, whether_classification_was_confident,
            whether_points_are_covered_by_balls, raw_confidences)
        """
        pass

    def predict_one(self, point: np.ndarray) -> Tuple[Optional[int], float, bool, bool, float, bool]:
        """
        Classify a single point

        Returns:
            (predicted_class, confidence, whether_classification_was_confident,
            whether_point_is_covered_by_ball, raw_confidence)
        """
        classes, confidence, is_confident, is_covered, confidence_raw, for_update = self.predict(np.array([point]))
        return classes[0], confidence[0], is_confident[0], is_covered[0], confidence_raw[0], for_update[0]

    def evaluate(self, X: np.ndarray, y: np.ndarray, update_buffer_length: int = None, max_updates: int = None) -> Tuple[EvaluationMetrics, Dict[str, np.ndarray]]:
        """
        Evaluate classifier on test data

        Args:
            X: test data - numpy array of points
            y: test data - numpy array of true labels
        Returns:
            EvaluationMetrics with metrics, dictionary with np-arrays pred, confidences,
            confident, covered, confidences_raw
        """
        results = []

        pred_labels, confidences, are_confident, are_covered, confidences_raw, for_update = self.predict(X)
        for point, true_label, pred_label, confidence, is_confident, is_covered in zip(
            X, y, pred_labels, confidences, are_confident, are_covered):

            results.append(ClassificationResult(
                point=point,
                true_label=true_label,
                predicted_label=pred_label if pred_label >= 0 else None,
                confidence=confidence,
                is_confident=is_confident,
                is_covered=is_covered,
                processing_time=0.0  # TODO: add timing
            ))
        if update_buffer_length is not None:
            X_update, y_update = X[for_update], y[for_update]
            period = np.searchsorted(np.cumsum(for_update), update_buffer_length+1)
            print(f"From {period} points accumulated {np.sum(for_update[:period])} points for update")
            n_updates = 0
            update_durations = []
            while update_buffer_length <= len(X_update):
                X_batch, y_batch = X_update[:update_buffer_length], y_update[:update_buffer_length]
                t = self.update(X_batch, y_batch)
                update_durations.append(t)
                print(f"update: {t*1000000:.3f} µs")
                n_updates += 1
                if max_updates is not None and n_updates >= max_updates:
                    break
                X_update, y_update = X_update[update_buffer_length:], y_update[update_buffer_length:]
                pred_labels2, confidences2, are_confident2, are_covered2, confidences_raw2, for_update2 = self.predict(X_update)
                X_update, y_update = X_update[for_update2], y_update[for_update2]
                period = np.searchsorted(np.cumsum(for_update2), update_buffer_length+1)
                print(f"From {period} points accumulated {np.sum(for_update2[:period])} points for update")
            print(f"Average update time: {np.mean(update_durations)*1000000:.3f} µs")

        preds_np = dict(pred=pred_labels, confidences=confidences, confident=are_confident,
                        covered=are_covered, confidences_raw=confidences_raw)
        if update_buffer_length is None:
            return EvaluationMetrics.from_results(results, self.n_representatives, self.train_data_len), preds_np
        else:
            return EvaluationMetrics.from_results(results, self.n_representatives, self.train_data_len), preds_np, update_durations

    @abstractmethod
    def are_points_covered(self, points: np.ndarray) -> list[bool]:
        """
        Check whether points are covered by balls
        """
        pass

    @property
    @abstractmethod
    def n_representatives(self) -> int:
        """Number of representatives"""
        pass

    @property
    @abstractmethod
    def train_data_len(self) -> int:
        """Number of elements in training set"""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_reprs={self.n_representatives})"


class ClassifierFactory:
    """Factory for creating classifiers"""

    _classifiers = {}

    @classmethod
    def register(cls, name: str, classifier_class):
        """Register a new classifier"""
        cls._classifiers[name] = classifier_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseClassifier:
        """Create classifier by name"""
        if name not in cls._classifiers:
            raise ValueError(f"Unknown classifier: {name}. Available: {list(cls._classifiers.keys())}")

        return cls._classifiers[name](**kwargs)

    @classmethod
    def list_available(cls) -> list:
        """List available classifiers"""
        return list(cls._classifiers.keys())
