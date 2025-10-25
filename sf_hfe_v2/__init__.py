"""
SF-HFE v2.0: Scarcity Framework - Hybrid Federated Expertise
Production-Grade Online Continual Learning System

Organized Structure:
- federated/ Server-side (Developer with ZERO data) + P2P gossip protocol
- moe/ Client-side (Users with data)
- data/ Data streaming
"""

__version__ = "2.0.0"
__author__ = "SF-HFE Team"

# Federated Learning components (including P2P gossip)
from .federated import SFHFEServer, GlobalMemory, OnlineMAMLEngine, P2PGossipManager

# MoE components
from .moe import (
SFHFEClient,
ScarcityMoE,
AdvancedOnlineLearningRouter,
UnifiedStorage,
SimulationEngine,
)

# All 30 experts for online learning
from .moe.structural_experts import (
SchemaMapperExpert, TypeFormatExpert, MissingnessNoiseExpert, ScalingEncodingExpert
)
from .moe.statistical_experts import (
DescriptiveExpert, CorrelationExpert, DensityExpert, AnomalyExpert
)
from .moe.temporal_experts import (
TrendExpert, DriftExpert, CyclicExpert, TemporalCausalityExpert
)
from .moe.relational_experts import (
GraphBuilderExpert, InfluenceExpert, GroupDynamicsExpert, FeedbackLoopExpert
)
from .moe.comprehensive_experts import (
CausalDiscoveryExpert, CounterfactualExpert, MediationExpert, PolicyEffectExpert,
ContextualExpert, DomainOntologyExpert, CrossDomainTransferExpert, RepresentationConsistencyExpert,
CognitiveExpert, SimulationExpert, ForecastExpert, MetaFeedbackExpert, MemoryCuratorExpert, EthicalConstraintExpert
)
from .moe.online_router import AdvancedOnlineLearningRouter

# P2P (now part of federated learning)

# Data
from .data import ConceptDriftStream, MultiClientStreamGenerator

__all__ = [
# Federated Learning
'SFHFEServer',
'GlobalMemory',
'OnlineMAMLEngine',

# MoE Core
'SFHFEClient',
'ScarcityMoE',
'AdvancedOnlineLearningRouter',
'UnifiedStorage',
'SimulationEngine',

# Structure Experts
'SchemaMapperExpert',
'TypeFormatExpert',
'MissingnessNoiseExpert',
'ScalingEncodingExpert',

# Statistical Experts
'DescriptiveExpert',
'CorrelationExpert',
'DensityExpert',
'AnomalyExpert',

# Temporal Experts
'TrendExpert',
'DriftExpert',
'CyclicExpert',
'TemporalCausalityExpert',

# Relational Experts
'GraphBuilderExpert',
'InfluenceExpert',
'GroupDynamicsExpert',
'FeedbackLoopExpert',

# Causal Experts
'CausalDiscoveryExpert',
'CounterfactualExpert',
'MediationExpert',
'PolicyEffectExpert',

# Semantic Experts
'ContextualExpert',
'DomainOntologyExpert',
'CrossDomainTransferExpert',
'RepresentationConsistencyExpert',

# Cognitive Experts
'CognitiveExpert',
'SimulationExpert',
'ForecastExpert',
'MetaFeedbackExpert',
'MemoryCuratorExpert',
'EthicalConstraintExpert',

# Advanced Router
'AdvancedOnlineLearningRouter',

# P2P (integrated with federated learning)
'P2PGossipManager',

# Data
'ConceptDriftStream',
'MultiClientStreamGenerator',
]

