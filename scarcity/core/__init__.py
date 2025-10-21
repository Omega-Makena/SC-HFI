"""
Scarcity Framework - Core Module

This module contains the core components of the Scarcity Framework (SF-HFE).
"""

from .expert import Expert, StructureExpert, DriftExpert, MemoryConsolidationExpert, MetaAdaptationExpert
from .router import Router
from .client import Client
from .server import Server
from .meta_learner import MetaLearner

__all__ = [
    'Expert', 'StructureExpert', 'DriftExpert', 
    'MemoryConsolidationExpert', 'MetaAdaptationExpert',
    'Router', 'Client', 'Server', 'MetaLearner'
]

