"""Nearest neighbor search algorithms and data structures"""

from .base import BaseSearcher, SearcherFactory
from dte_stand.algorithm.mate.lib.classifiers.metrics import MetricSpace
from dte_stand.algorithm.mate.lib.classifiers.utils import Representative
from .linear import LinearSearcher
from .balltree import BallTreeSearcher

SearcherFactory.register('linear', LinearSearcher)
SearcherFactory.register('balltree', BallTreeSearcher)

__all__ = [
    'BaseSearcher',
    'MetricSpace',
    'SearcherFactory',

    'Representative',

    'LinearSearcher',
    'create_Searcher'
]
