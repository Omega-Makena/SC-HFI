"""
Evaluation Module - P1 Essential
Comprehensive metrics including fairness
"""

from .metrics import (
    compute_fairness_metrics,
    healthcare_fairness_index,
    per_client_evaluation
)

__all__ = [
    'compute_fairness_metrics',
    'healthcare_fairness_index',
    'per_client_evaluation',
]

