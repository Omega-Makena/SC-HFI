"""
SF-HFE Client - MoE Component
Client-side implementation with domain awareness and P2P gossip capability
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import time
import threading
from collections import defaultdict

from .base_expert import BaseExpert
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
from .online_router import AdvancedOnlineLearningRouter
from .unified_storage import UnifiedStorage
from .simulation import SimulationEngine
from ..config import FL_CONFIG, META_LEARNING_CONFIG

# ScarcityMoE will be imported when needed to avoid circular imports


class SFHFEClient:
"""
SF-HFE Client - User device with OMEO as base model

Responsibilities:
- Train OMEO model on local user data
- Extract OMEO model updates (not raw data)
- Participate in P2P gossip learning with other users
- Send OMEO updates to server
- Receive updated OMEO parameters from server
"""

def __init__(self, client_id: int, domain: str = "general", config: Dict = None):
self.client_id = client_id
self.domain = domain
self.config = config or {}

# Logger
self.logger = logging.getLogger(f"Client-{client_id}")
self.logger.info(f"Initializing SF-HFE Client {client_id} for domain '{domain}'")

# Initialize 30-expert online learning system
self.experts = self._initialize_experts()
self.router = AdvancedOnlineLearningRouter(self.experts, config=self.config.get('router', {}))
self.storage = UnifiedStorage(config=self.config.get('storage', {}))
self.simulation = SimulationEngine(config=self.config.get('simulation', {}))

# Expert weights and performance tracking
self.expert_weights = {}
self.expert_performance = {}
self.training_samples = 0

# P2P gossip state
self.peer_weights = {} # peer_id -> expert weights
self.gossip_history = []
self.last_gossip_time = time.time()

# Meta-parameters from server
self.meta_parameters = None
self.last_meta_update = time.time()

# Thread safety
self._lock = threading.Lock()

# Initialize experts with default weights
self._initialize_experts()

def _initialize_experts(self):
"""Initialize all 30 experts for online learning"""
experts = []

# Structural Experts (1-4)
experts.append(SchemaMapperExpert(expert_id=1, config=self.config.get('experts', {}).get(1, {})))
experts.append(TypeFormatExpert(expert_id=2, config=self.config.get('experts', {}).get(2, {})))
experts.append(MissingnessNoiseExpert(expert_id=3, config=self.config.get('experts', {}).get(3, {})))
experts.append(ScalingEncodingExpert(expert_id=4, config=self.config.get('experts', {}).get(4, {})))

# Statistical Experts (5-8)
experts.append(DescriptiveExpert(expert_id=5, config=self.config.get('experts', {}).get(5, {})))
experts.append(CorrelationExpert(expert_id=6, config=self.config.get('experts', {}).get(6, {})))
experts.append(DensityExpert(expert_id=7, config=self.config.get('experts', {}).get(7, {})))
experts.append(AnomalyExpert(expert_id=8, config=self.config.get('experts', {}).get(8, {})))

# Temporal Experts (9-12)
experts.append(TrendExpert(expert_id=9, config=self.config.get('experts', {}).get(9, {})))
experts.append(DriftExpert(expert_id=10, config=self.config.get('experts', {}).get(10, {})))
experts.append(CyclicExpert(expert_id=11, config=self.config.get('experts', {}).get(11, {})))
experts.append(TemporalCausalityExpert(expert_id=12, config=self.config.get('experts', {}).get(12, {})))

# Relational/Interactional Experts (13-16)
experts.append(GraphBuilderExpert(expert_id=13, config=self.config.get('experts', {}).get(13, {})))
experts.append(InfluenceExpert(expert_id=14, config=self.config.get('experts', {}).get(14, {})))
experts.append(GroupDynamicsExpert(expert_id=15, config=self.config.get('experts', {}).get(15, {})))
experts.append(FeedbackLoopExpert(expert_id=16, config=self.config.get('experts', {}).get(16, {})))

# Causal Experts (17-20)
experts.append(CausalDiscoveryExpert(expert_id=17, config=self.config.get('experts', {}).get(17, {})))
experts.append(CounterfactualExpert(expert_id=18, config=self.config.get('experts', {}).get(18, {})))
experts.append(MediationExpert(expert_id=19, config=self.config.get('experts', {}).get(19, {})))
experts.append(PolicyEffectExpert(expert_id=20, config=self.config.get('experts', {}).get(20, {})))

# Semantic/Contextual Experts (21-24)
experts.append(ContextualExpert(expert_id=21, config=self.config.get('experts', {}).get(21, {})))
experts.append(DomainOntologyExpert(expert_id=22, config=self.config.get('experts', {}).get(22, {})))
experts.append(CrossDomainTransferExpert(expert_id=23, config=self.config.get('experts', {}).get(23, {})))
experts.append(RepresentationConsistencyExpert(expert_id=24, config=self.config.get('experts', {}).get(24, {})))

# Integrative/Cognitive Experts (25-30)
experts.append(CognitiveExpert(expert_id=25, config=self.config.get('experts', {}).get(25, {})))
experts.append(SimulationExpert(expert_id=26, config=self.config.get('experts', {}).get(26, {})))
experts.append(ForecastExpert(expert_id=27, config=self.config.get('experts', {}).get(27, {})))
experts.append(MetaFeedbackExpert(expert_id=28, config=self.config.get('experts', {}).get(28, {})))
experts.append(MemoryCuratorExpert(expert_id=29, config=self.config.get('experts', {}).get(29, {})))
experts.append(EthicalConstraintExpert(expert_id=30, config=self.config.get('experts', {}).get(30, {})))

self.logger.info(f"Initialized {len(experts)} experts for online learning")
return experts

def process_data(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
"""
Process data through the 30-expert online learning system

Args:
data: Input data array
metadata: Additional context information

Returns:
Dictionary containing expert insights and analysis
"""
metadata = metadata or {}

with self._lock:
try:
# Route data to appropriate experts
selected_experts = self.router.route_experts(data, metadata)

# Process data through selected experts
expert_results = {}
for expert_id, weight in selected_experts:
expert = self.experts[expert_id]

# Activate expert
if expert.activate(metadata):
# Process data
result = expert.process_data(data, metadata)
expert_results[expert_id] = {
'result': result,
'weight': weight,
'expert_name': expert.name
}

# Update expert online
expert.update_online(data, {'context': metadata})

# Store results in unified storage
self.storage.store_results(expert_results, metadata)

# Generate simulation if requested
simulation_results = None
if metadata.get('generate_simulation', False):
simulation_results = self.simulation.generate_simulation(data, expert_results)

# Compile final results
final_results = {
'expert_results': expert_results,
'simulation_results': simulation_results,
'processing_metadata': {
'num_experts_used': len(expert_results),
'total_experts': len(self.experts),
'processing_time': time.time(),
'domain': self.domain
}
}

# Update performance tracking
self._update_performance_tracking(expert_results)

return final_results

except Exception as e:
self.logger.error(f"Error processing data: {e}")
return {'error': str(e), 'expert_results': {}}

def _update_performance_tracking(self, expert_results: Dict[str, Any]):
"""Update performance tracking for experts"""
for expert_id, result_info in expert_results.items():
result = result_info['result']
confidence = result.get('confidence', 0.5)

# Update router with performance feedback
self.router.update_routing(expert_id, confidence, {
'context': result.get('processing_metadata', {}),
'expert_name': result_info['expert_name']
})


def train_experts(self, data: np.ndarray, labels: np.ndarray = None) -> Dict:
"""
Train experts on local data

Args:
data: Local training data
labels: Optional labels for supervised learning

Returns:
Training results and insights
"""
with self._lock:
self.logger.info(f"Client {self.client_id}: Training experts on {len(data)} samples")

# Process data through MoE system
results = self.moe_system.process_data(data)

# Update expert weights based on performance
self._update_expert_weights(results)

# Track training samples
self.training_samples += len(data)

# Extract insights for server
insights = self._extract_insights(results)

self.logger.info(f"Client {self.client_id}: Training complete, extracted insights")
return insights

def _update_expert_weights(self, results: Dict):
"""Update expert weights based on MoE results"""
for tier_name, tier_results in results.items():
if tier_name.startswith('tier'):
# Update weights based on tier performance
for expert_name, performance in tier_results.get('expert_performance', {}).items():
if expert_name in self.expert_weights:
# Simple weight update (can be made more sophisticated)
learning_rate = 0.01
if performance.get('loss', 0) > 0:
self.expert_weights[expert_name] += learning_rate * np.random.randn(64)
self.expert_performance[expert_name] = performance

def _extract_insights(self, results: Dict) -> Dict:
"""Extract insights from MoE results (no raw data)"""
insights = {
"client_id": self.client_id,
"domain": self.domain,
"expert_insights": {},
"avg_loss": 0.0,
"total_samples": self.training_samples,
"timestamp": time.time()
}

# Aggregate expert performance
total_loss = 0.0
expert_count = 0

for expert_name, performance in self.expert_performance.items():
loss = performance.get('loss', 0.0)
samples = performance.get('samples', 0)

insights["expert_insights"][expert_name] = {
"loss": loss,
"samples": samples,
"last_update": performance.get('last_update', time.time())
}

total_loss += loss
expert_count += 1

insights["avg_loss"] = total_loss / expert_count if expert_count > 0 else 0.0

return insights

def select_peers(self, all_clients: List) -> List[int]:
"""
Select peers for P2P gossip based on domain similarity

Args:
all_clients: List of all available clients

Returns:
List of selected peer client IDs
"""
with self._lock:
# Filter clients by domain
domain_clients = [c for c in all_clients if c.domain == self.domain]

# Exclude self
domain_clients = [c for c in domain_clients if c.client_id != self.client_id]

# Select top 3 peers based on performance similarity
selected_peers = []
if domain_clients:
# Simple selection: choose clients with similar performance
for client in domain_clients[:3]:
selected_peers.append(client.client_id)

self.logger.debug(f"Client {self.client_id}: Selected peers {selected_peers} from domain '{self.domain}'")
return selected_peers

def gossip_exchange(self, peer_weights: Dict[int, Dict]) -> Dict:
"""
Perform P2P gossip exchange with peers

Args:
peer_weights: Dictionary of peer_id -> expert weights

Returns:
Updated expert weights after gossip
"""
with self._lock:
self.logger.info(f"Client {self.client_id}: Performing gossip exchange with {len(peer_weights)} peers")

# Store peer weights
self.peer_weights.update(peer_weights)

# Perform weighted average of expert weights
updated_weights = {}

for expert_name in self.expert_weights:
# Start with own weights
weight_sum = self.expert_weights[expert_name].copy()
weight_count = 1

# Add peer weights
for peer_id, peer_expert_weights in peer_weights.items():
if expert_name in peer_expert_weights:
weight_sum += peer_expert_weights[expert_name]
weight_count += 1

# Average the weights
updated_weights[expert_name] = weight_sum / weight_count

# Update own weights
self.expert_weights.update(updated_weights)

# Record gossip exchange
self.gossip_history.append({
"timestamp": time.time(),
"peers": list(peer_weights.keys()),
"experts_updated": len(updated_weights)
})

self.last_gossip_time = time.time()

self.logger.info(f"Client {self.client_id}: Gossip exchange complete, updated {len(updated_weights)} experts")
return updated_weights

def receive_meta_parameters(self, meta_params: Dict):
"""Receive meta-parameters from server"""
with self._lock:
self.meta_parameters = meta_params
self.last_meta_update = time.time()

# Update expert initialization if provided
if 'w_init' in meta_params:
self._update_expert_initialization(meta_params['w_init'])

# Update learning rates if provided
if 'expert_alphas' in meta_params:
self._update_learning_rates(meta_params['expert_alphas'])

self.logger.info(f"Client {self.client_id}: Received meta-parameters from server")

def _update_expert_initialization(self, w_init: np.ndarray):
"""Update expert initialization weights"""
for i, expert_name in enumerate(self.expert_weights):
if i < len(w_init):
self.expert_weights[expert_name] = w_init[i].copy()

def _update_learning_rates(self, expert_alphas: Dict):
"""Update learning rates for experts"""
# This would be used in future training rounds
self.config['learning_rates'] = expert_alphas

def get_expert_weights(self) -> Dict:
"""Get current expert weights for gossip exchange"""
with self._lock:
return self.expert_weights.copy()

def get_performance_stats(self) -> Dict:
"""Get performance statistics"""
with self._lock:
return {
"client_id": self.client_id,
"domain": self.domain,
"training_samples": self.training_samples,
"expert_count": len(self.expert_weights),
"last_gossip_time": self.last_gossip_time,
"gossip_exchanges": len(self.gossip_history),
"avg_loss": np.mean([p.get('loss', 0) for p in self.expert_performance.values()])
}

def get_omoe_model_updates(self) -> Dict:
"""
Get OMEO model updates from this user client

Returns:
Dictionary containing OMEO model updates (not raw data)
"""
with self._lock:
# Extract OMEO model updates from all tiers
omoe_updates = {
"client_id": self.client_id,
"domain": self.domain,
"tier_weights": {},
"expert_performance": self.expert_performance.copy(),
"training_samples": self.training_samples,
"timestamp": time.time()
}

# Get weights from each tier
for tier_name in ["tier1", "tier2", "tier3", "tier4", "tier5", "tier6"]:
if hasattr(self, tier_name):
tier = getattr(self, tier_name)
if hasattr(tier, 'get_weights'):
omoe_updates["tier_weights"][tier_name] = tier.get_weights()

self.logger.debug(f"User client {self.client_id}: Generated OMEO model updates")
return omoe_updates

def sync_with_peer(self, peer_id: int, peer_client):
"""
Synchronize with a peer client during P2P gossip exchange

Args:
peer_id: ID of the peer client
peer_client: The peer client object
"""
with self._lock:
try:
self.logger.info(f"Client {self.client_id}: Syncing with peer {peer_id}")

# Get peer's expert weights
peer_weights = peer_client.get_expert_weights()

# Perform gossip exchange
updated_weights = self.gossip_exchange({peer_id: peer_weights})

self.logger.info(f"Client {self.client_id}: Sync with peer {peer_id} completed")
return updated_weights

except Exception as e:
self.logger.error(f"Client {self.client_id}: Error syncing with peer {peer_id}: {e}")
return {}

def update_omoe_parameters(self, meta_parameters: Dict):
"""Update OMEO parameters from the server"""
with self._lock:
try:
# Update tier configurations with new meta-parameters
for tier_name, tier_params in meta_parameters.items():
if hasattr(self, tier_name):
tier = getattr(self, tier_name)
if hasattr(tier, 'update_parameters'):
tier.update_parameters(tier_params)

# Update global storage and simulation engine
if hasattr(self, 'storage') and hasattr(self.storage, 'update_parameters'):
self.storage.update_parameters(meta_parameters.get('storage', {}))
if hasattr(self, 'simulation') and hasattr(self.simulation, 'update_parameters'):
self.simulation.update_parameters(meta_parameters.get('simulation', {}))

self.logger.info(f"User client {self.client_id}: Updated OMEO parameters")

except Exception as e:
self.logger.error(f"User client {self.client_id}: Error updating OMEO parameters: {e}")


# Import ScarcityMoE from the same module
# Remove the circular import at the end
# from . import ScarcityMoE
