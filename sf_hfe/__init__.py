"""
SF-HFE: Scarcity Framework - Hybrid Federated Expertise
Online Continual Learning System for General AI

Core Components:
- 10 Specialized Experts
- Contextual Bandit Router
- 3-Tier Hierarchical Memory
- Online MAML Meta-Learning
- P2P Gossip Protocol
- Insight-Based Federated Learning
"""

__version__ = "1.0.0"
__author__ = "SF-HFE Team"

from .client import SFHFEClient
from .server import SFHFEServer
from .router import ContextualBanditRouter
from .memory import HierarchicalMemory
from .data_stream import ConceptDriftStream, MultiClientStreamGenerator
from .p2p_gossip import P2PGossipManager

# Expert imports
from .experts import (
    GeometryExpert,
    TemporalExpert,
    ReconstructionExpert,
    CausalInferenceExpert,
    DriftDetectionExpert,
    GovernanceExpert,
    StatisticalConsistencyExpert,
    PeerSelectionExpert,
    MetaAdaptationExpert,
    MemoryConsolidationExpert
)

__all__ = [
    # Core components
    'SFHFEClient',
    'SFHFEServer',
    'ContextualBanditRouter',
    'HierarchicalMemory',
    
    # Data streaming
    'ConceptDriftStream',
    'MultiClientStreamGenerator',
    
    # P2P
    'P2PGossipManager',
    
    # All 10 experts
    'GeometryExpert',
    'TemporalExpert',
    'ReconstructionExpert',
    'CausalInferenceExpert',
    'DriftDetectionExpert',
    'GovernanceExpert',
    'StatisticalConsistencyExpert',
    'PeerSelectionExpert',
    'MetaAdaptationExpert',
    'MemoryConsolidationExpert',
]

