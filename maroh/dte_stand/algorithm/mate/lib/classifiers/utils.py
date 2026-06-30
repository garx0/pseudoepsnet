"""Common utilities and data structures"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import time

@dataclass
class Representative:
    """Structure for storing a representative"""
    center: np.ndarray
    epsilon: float
    label: int  # combined agent and action

    def __post_init__(self):
        self.center = np.asarray(self.center)

    def __repr__(self):
        return f"Repr(label={self.label}, eps={self.epsilon:.3f})"

@dataclass
class ClassificationResult:
    """Classification result for a single point"""
    point: np.ndarray
    true_label: int
    predicted_label: Optional[int]
    confidence: float
    is_confident: bool
    is_covered: bool # Whether point fell into any representative's area
    processing_time: float # Processing time in seconds

@dataclass
class EvaluationMetrics:
    """Classifier evaluation metrics"""
    accuracy: float
    coverage: float
    compression_ratio: float
    num_reprs: int
    total_points: int
    correct_classifications: int
    confident_classifications: int
    covered_points: int
    avg_processing_time: float
    total_processing_time: float

    @classmethod
    def from_results(cls, results: List[ClassificationResult], n_reprs: int, train_size: int):
        """Create metrics from list of results"""
        total = len(results)
        if total == 0:
            return cls(0, 0, 0, n_reprs, 0, 0, 0, 0, 0, 0)

        covered = sum(1 for r in results if r.is_covered)
        confident = sum(1 for r in results if r.is_confident)
        correct = sum(1 for r in results if r.is_confident and r.predicted_label == r.true_label)

        times = [r.processing_time for r in results]

        return cls(
            accuracy=correct / confident if confident > 0 else 0,
            coverage=confident / total,
            compression_ratio=n_reprs / train_size if train_size > 0 else 0,
            num_reprs=n_reprs,
            total_points=total,
            correct_classifications=correct,
            confident_classifications=confident,
            covered_points=covered,
            avg_processing_time=float(np.mean(times)),
            total_processing_time=float(np.sum(times))
        )

    def __str__(self):
        return (f"Acc: {self.accuracy:.3f}, Cov: {self.coverage:.3f}, "
                f"Comp: {self.compression_ratio:.3f}, Reprs: {self.num_reprs}, "
                f"Avg_time: {self.avg_processing_time:.3f}, Total_time: {self.total_processing_time:.3f}")

def format_percentile_key(percent):
    """
    Format percentile in percent to a readable key

    Examples:
    0.001 -> 'p0_001'    (0.001%)
    0.005 -> 'p0_005'    (0.005%)
    0.01  -> 'p0_01'     (0.01%)
    0.05  -> 'p0_05'     (0.05%)
    0.1   -> 'p0_1'      (0.1%)
    0.5   -> 'p0_5'      (0.5%)
    1     -> 'p1'        (1%)
    5     -> 'p5'        (5%)
    50    -> 'p50'       (50%)
    99.5  -> 'p99_5'     (99.5%)
    99.99 -> 'p99_99'    (99.99%)
    """
    # If value is integer (e.g., 5.0, 50.0)
    if percent == int(percent):
        return f'p{int(percent)}'

    # If fractional, replace dot with underscore
    # 0.001 -> p0_001, 99.5 -> p99_5
    str_percent = f"{percent:.10f}".rstrip('0').rstrip('.')
    return f'p{str_percent.replace(".", "_")}'
