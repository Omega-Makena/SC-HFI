"""
SF-HFE v2.0: Scarcity Framework - Hybrid Federated Expertise
Production-Grade Online Continual Learning System

Organized Structure:
- federated/   → Server-side (Developer with ZERO data)
- moe/         → Client-side (Users with data)
- p2p/         → Decentralized communication
- data/        → Data streaming
"""

__version__ = "2.0.0"
__author__ = "SF-HFE Team"

# Federated Learning components
from .federated import SFHFEServer, GlobalMemory, OnlineMAMLEngine

# MoE components
from .moe import (
    SFHFEClient,
    ContextualBanditRouter,
    BaseExpert,
    HierarchicalMemory,
)

# All 10 experts
from .moe.experts import (
    # Structure
    GeometryExpert,
    TemporalExpert,
    ReconstructionExpert,
    
    # Intelligence
    CausalInferenceExpert,
    DriftDetectionExpert,
    
    # Guardrail
    GovernanceExpert,
    StatisticalConsistencyExpert,
    
    # Specialized
    PeerSelectionExpert,
    MetaAdaptationExpert,
    MemoryConsolidationExpert,
)

# P2P
from .p2p import P2PGossipManager

# Data
from .data import ConceptDriftStream, MultiClientStreamGenerator

__all__ = [
    # Federated Learning
    'SFHFEServer',
    'GlobalMemory',
    'OnlineMAMLEngine',
    
    # MoE Core
    'SFHFEClient',
    'ContextualBanditRouter',
    'BaseExpert',
    'HierarchicalMemory',
    
    # Structure Experts
    'GeometryExpert',
    'TemporalExpert',
    'ReconstructionExpert',
    
    # Intelligence Experts
    'CausalInferenceExpert',
    'DriftDetectionExpert',
    
    # Guardrail Experts
    'GovernanceExpert',
    'StatisticalConsistencyExpert',
    
    # Specialized Experts
    'PeerSelectionExpert',
    'MetaAdaptationExpert',
    'MemoryConsolidationExpert',
    
    # P2P
    'P2PGossipManager',
    
    # Data
    'ConceptDriftStream',
    'MultiClientStreamGenerator',
]

