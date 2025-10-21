"""
Mixture of Experts (MoE) Module
Client-side components (Users with local data)
"""

from .base_expert import BaseExpert
from .router import ContextualBanditRouter
from .client import SFHFEClient

# Import all expert categories
from .experts.structure import GeometryExpert, TemporalExpert, ReconstructionExpert
from .experts.intelligence import CausalInferenceExpert, DriftDetectionExpert
from .experts.guardrail import GovernanceExpert, StatisticalConsistencyExpert
from .experts.specialized import (
    PeerSelectionExpert,
    MetaAdaptationExpert,
    MemoryConsolidationExpert
)

# Import memory system
from .memory import HierarchicalMemory

__all__ = [
    # Core MoE components
    'BaseExpert',
    'ContextualBanditRouter',
    'SFHFEClient',
    
    # Structure experts
    'GeometryExpert',
    'TemporalExpert',
    'ReconstructionExpert',
    
    # Intelligence experts
    'CausalInferenceExpert',
    'DriftDetectionExpert',
    
    # Guardrail experts
    'GovernanceExpert',
    'StatisticalConsistencyExpert',
    
    # Specialized experts
    'PeerSelectionExpert',
    'MetaAdaptationExpert',
    'MemoryConsolidationExpert',
    
    # Memory system
    'HierarchicalMemory',
]

