"""
Scarcity Mixture of Experts (MoE) - 30-Expert Online Learning System
Implements the complete 30-expert architecture for online learning
with cross-expert compositional reasoning capabilities
"""

from .client import SFHFEClient
from .unified_storage import UnifiedStorage
from .simulation import SimulationEngine
from .online_router import AdvancedOnlineLearningRouter

# Import all 30 experts
from .structural_experts import (
SchemaMapperExpert, TypeFormatExpert, MissingnessNoiseExpert, ScalingEncodingExpert
)
from .statistical_experts import (
DescriptiveExpert, CorrelationExpert, DensityExpert, AnomalyExpert
)
from .temporal_experts import (
TrendExpert, DriftExpert, CyclicExpert, TemporalCausalityExpert
)
from .relational_experts import (
GraphBuilderExpert, InfluenceExpert, GroupDynamicsExpert, FeedbackLoopExpert
)
from .comprehensive_experts import (
CausalDiscoveryExpert, CounterfactualExpert, MediationExpert, PolicyEffectExpert,
ContextualExpert, DomainOntologyExpert, CrossDomainTransferExpert, RepresentationConsistencyExpert,
CognitiveExpert, SimulationExpert, ForecastExpert, MetaFeedbackExpert, MemoryCuratorExpert, EthicalConstraintExpert
)

# Import cross-expert reasoning system
from .cross_expert_reasoning import (
CrossExpertReasoningSystem,
ExpertOutput,
CompositionalInsight,
InterExpertGraph,
CompositionalReasoningEngine,
MetaController,
MemoryCurator
)

from .expert_integration import (
ExpertIntegrationManager,
ExpertCompositionVisualizer
)

from .cross_expert_demo import (
CrossExpertReasoningDemo,
run_cross_expert_demo
)

__version__ = "2.0.0"
__author__ = "SCARCITY Framework Team"

# All available experts
ALL_EXPERTS = [
# Structural Experts (1-4)
SchemaMapperExpert, TypeFormatExpert, MissingnessNoiseExpert, ScalingEncodingExpert,

# Statistical Experts (5-8)
DescriptiveExpert, CorrelationExpert, DensityExpert, AnomalyExpert,

# Temporal Experts (9-12)
TrendExpert, DriftExpert, CyclicExpert, TemporalCausalityExpert,

# Relational Experts (13-16)
GraphBuilderExpert, InfluenceExpert, GroupDynamicsExpert, FeedbackLoopExpert,

# Causal Experts (17-20)
CausalDiscoveryExpert, CounterfactualExpert, MediationExpert, PolicyEffectExpert,

# Semantic Experts (21-24)
ContextualExpert, DomainOntologyExpert, CrossDomainTransferExpert, RepresentationConsistencyExpert,

# Cognitive Experts (25-30)
CognitiveExpert, SimulationExpert, ForecastExpert, MetaFeedbackExpert, MemoryCuratorExpert, EthicalConstraintExpert
]

class ScarcityMoE:
"""
Main Scarcity Mixture of Experts system implementing the 30-expert architecture
with cross-expert compositional reasoning capabilities
"""

def __init__(self, config=None):
"""
Initialize the complete MoE system with all 30 experts

Args:
config: Configuration dictionary for the MoE system
"""
self.config = config or {}

# Initialize client with 30 experts
self.client = SFHFEClient(client_id=0, domain="general", config=self.config)

# Initialize post-expert processes
self.storage = UnifiedStorage(config=self.config.get('storage', {}))
self.simulation = SimulationEngine(config=self.config.get('simulation', {}))

# Initialize cross-expert reasoning system
self.cross_expert_system = CrossExpertReasoningSystem(config=self.config.get('cross_expert', {}))
self.expert_integration = ExpertIntegrationManager(config=self.config.get('integration', {}))

# Track active experts and routing
self.active_experts = []
self.routing_history = []

def process_data(self, data, metadata=None):
"""
Main entry point for processing data through the complete MoE system

Args:
data: Input dataset (numpy array or similar)
metadata: Optional metadata about the data source

Returns:
dict: Complete analysis results from all experts
"""
# Process data through the 30-expert system
results = self.client.process_data(data, metadata)

# Store insights in unified storage
if 'expert_results' in results:
self.storage.store_results(results['expert_results'], metadata)

# Run simulations if requested
if metadata and metadata.get('generate_simulation', False):
simulation_results = self.simulation.generate_simulation(data, results)
results['simulation_results'] = simulation_results

return results

def process_data_with_cross_expert_reasoning(self, data, metadata=None):
"""
Process data through the MoE system with cross-expert compositional reasoning

Args:
data: Input dataset (numpy array or similar)
metadata: Optional metadata about the data source

Returns:
dict: Complete analysis results including cross-expert insights
"""
# Process data through the 30-expert system
results = self.client.process_data(data, metadata)

# Store insights in unified storage
if 'expert_results' in results:
self.storage.store_results(results['expert_results'], metadata)

# Run cross-expert compositional reasoning
if 'expert_results' in results:
cross_expert_insights = self.cross_expert_system.generate_insights(results['expert_results'])
results['cross_expert_insights'] = cross_expert_insights

# Run simulations if requested
if metadata and metadata.get('generate_simulation', False):
simulation_results = self.simulation.generate_simulation(data, results)
results['simulation_results'] = simulation_results

return results

def get_system_status(self):
"""
Get current status of the MoE system

Returns:
dict: System status information
"""
router_stats = self.client.router.get_routing_stats()

return {
'total_experts': len(self.client.experts),
'active_experts': router_stats.get('active_experts', 0),
'routing_history_count': router_stats.get('routing_history_size', 0),
'storage_insights_count': self.storage.get_insight_count() if hasattr(self.storage, 'get_insight_count') else 0,
'expert_groups': router_stats.get('expert_groups', {}),
'group_performance': router_stats.get('group_performance', {}),
'cross_expert_insights_count': len(self.cross_expert_system.insights) if hasattr(self.cross_expert_system, 'insights') else 0
}

def update_meta_parameters(self, meta_params):
"""
Update meta-learning parameters from external meta-learning engine

Args:
meta_params: Dictionary of updated meta-parameters
"""
# Update client configuration
if 'client' in meta_params:
self.client.config.update(meta_params['client'])

# Update storage and simulation engine
if 'storage' in meta_params:
self.storage.update_parameters(meta_params['storage'])
if 'simulation' in meta_params:
self.simulation.update_parameters(meta_params['simulation'])

# Update cross-expert reasoning system
if 'cross_expert' in meta_params:
self.cross_expert_system.update_parameters(meta_params['cross_expert'])
if 'integration' in meta_params:
self.expert_integration.update_parameters(meta_params['integration'])