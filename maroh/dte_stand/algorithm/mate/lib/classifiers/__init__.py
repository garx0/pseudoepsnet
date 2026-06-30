"""Classifiers"""

from .base import BaseClassifier, ClassifierFactory
from .metrics import MetricSpace
from .utils import Representative, ClassificationResult, EvaluationMetrics
from .epsilon_net import EpsilonNetClassifier
from .knn import kNNClassifier
from .centroids import CentroidsClassifier

# Classifiers implementations
ClassifierFactory.register('epsilon_net', EpsilonNetClassifier)
ClassifierFactory.register('kNN', kNNClassifier)
ClassifierFactory.register('centroids', CentroidsClassifier)

__all__ = [
    # Base classes
    'BaseClassifier',
    'MetricSpace',
    'ClassifierFactory',

    # Data structures
    'Representative',
    'ClassificationResult',
    'EvaluationMetrics',

    # Main classifier
    'EpsilonNetClassifier',
    'create_classifier'
]
