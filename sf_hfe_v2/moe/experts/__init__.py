"""
Expert Portfolio - All 10 Specialized Experts
Organized by category for better maintainability
"""

from .structure import GeometryExpert, TemporalExpert, ReconstructionExpert
from .intelligence import CausalInferenceExpert, DriftDetectionExpert
from .guardrail import GovernanceExpert, StatisticalConsistencyExpert
from .specialized import (
    PeerSelectionExpert,
    MetaAdaptationExpert,
    MemoryConsolidationExpert
)

__all__ = [
    # Core Structure (3)
    'GeometryExpert',
    'TemporalExpert',
    'ReconstructionExpert',
    
    # Intelligence (2)
    'CausalInferenceExpert',
    'DriftDetectionExpert',
    
    # Guardrail (2)
    'GovernanceExpert',
    'StatisticalConsistencyExpert',
    
    # Specialized (3)
    'PeerSelectionExpert',
    'MetaAdaptationExpert',
    'MemoryConsolidationExpert',
]

