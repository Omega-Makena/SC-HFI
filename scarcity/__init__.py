"""
Scarcity Framework (SF-HFE)

A Hierarchical Federated Ensemble framework for machine learning
with scarcity constraints.
"""

__version__ = "0.1.0"
__author__ = "Scarcity Framework Team"

from scarcity.core import Expert, StructureExpert, DriftExpert, Router, Client, Server, MetaLearner

__all__ = ['Expert', 'StructureExpert', 'DriftExpert', 'Router', 'Client', 'Server', 'MetaLearner']

