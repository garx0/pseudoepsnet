"""Base abstract class for all nearest neighbor search structures"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple, List
from classifiers.utils import *

class BaseSearcher(ABC):
    """Base class for search structure"""
    @abstractmethod
    def fit(self, X: np.ndarray, r: np.ndarray):
        """
        Build index

        Args:
            X: representative points
            r: radii of representatives
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find nearest elements for multiple points

        Returns:
            (indices_of_nearest_elements, distances_to_nearest_elements)
        """
        pass

    def predict_one(self, point: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Find nearest element to a point

        Returns:
            (nearest_element, distance_to_nearest_element)
        """
        a, b = self.predict(np.array([point]))
        return a[0], b[0]

    def __call__(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convenient call: searcher(X)"""
        return self.predict(X)

    @property
    @abstractmethod
    def n_representatives(self) -> int:
        """Number of representatives"""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_reprs={self.n_representatives})"


class SearcherFactory:
    """Factory for creating search structures"""

    _searchers = {}

    @classmethod
    def register(cls, name: str, searcher_class):
        """Register a new search structure"""
        cls._searchers[name] = searcher_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseSearcher:
        """Create search structure by name"""
        if name not in cls._searchers:
            raise ValueError(f"Unknown searcher: {name}. Available: {list(cls._searchers.keys())}")

        return cls._searchers[name](**kwargs)

    @classmethod
    def list_available(cls) -> list:
        """List available search structures"""
        return list(cls._searchers.keys())
