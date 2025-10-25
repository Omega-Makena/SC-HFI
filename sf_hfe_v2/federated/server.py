"""
SF-HFE Server - Federated Learning Component
Central coordinator operated by Developer (with ZERO training data)
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
import threading
import time
import json
from datetime import datetime
from collections import defaultdict

from ..config import FL_CONFIG, META_LEARNING_CONFIG, SYSTEM_CONFIG
from .global_memory import GlobalMemory
from .meta_learning import OnlineMAMLEngine
from .gossip import P2PGossipManager
from .initialization import setup_reproducibility, get_client_id_type, safe_get_config


class DomainAggregator:
"""
Domain Aggregator - Tier 2
Acts as mini-server for a domain (e.g., Economics-Kenya, Medical-Imaging, etc.)
"""

def __init__(self, domain: str):
self.domain = domain
self.logger = logging.getLogger(f"DomainAggregator-{domain}")

# Domain-specific components
self.domain_adapters = {} # Domain adapters φ_d,k for each expert k
self.domain_router = np.random.randn(30) # Domain router head r_d

# Client management
self.clients = []
self.client_deltas = []

# Domain meta-learning
self.domain_meta_params = {}
self.domain_performance_history = []

# Statistics
self.total_clients = 0
self.total_aggregations = 0

self.logger.info(f"Initialized domain aggregator for '{domain}'")

def add_client(self, client):
"""Add a client to this domain"""
self.clients.append(client)
self.total_clients += 1
self.logger.info(f"Added client {client.client_id} to domain '{self.domain}'")

def receive_client_deltas(self, deltas: Dict[str, Any]):
"""Receive deltas from clients"""
self.client_deltas.append(deltas)
self.logger.info(f"Received deltas from client {deltas['client_id']} in domain '{self.domain}'")

def aggregate_domain_adapters(self) -> Dict[str, Any]:
"""Aggregate client adapters to build domain adapters φ_d,k and domain router head r_d"""
if not self.client_deltas:
return {}

self.logger.info(f"Aggregating {len(self.client_deltas)} client deltas for domain '{self.domain}'")

# Aggregate router biases
router_biases = [delta['router_delta'] for delta in self.client_deltas]
self.domain_router = np.mean(router_biases, axis=0)

# Aggregate expert adapters
for expert_id in range(30): # 30 experts
expert_adapters = []
for delta in self.client_deltas:
if expert_id in delta['expert_deltas']:
expert_adapters.append(delta['expert_deltas'][expert_id])

if expert_adapters:
# Average U matrices
U_matrices = [ea['U_delta'] for ea in expert_adapters]
avg_U = np.mean(U_matrices, axis=0)

# Average V matrices
V_matrices = [ea['V_delta'] for ea in expert_adapters]
avg_V = np.mean(V_matrices, axis=0)

self.domain_adapters[expert_id] = {
'U': avg_U,
'V': avg_V,
'rank': expert_adapters[0]['rank']
}

# Clear processed deltas
self.client_deltas = []
self.total_aggregations += 1

return {
'domain': self.domain,
'domain_router': self.domain_router.tolist(),
'domain_adapters': {
expert_id: {
'U': adapter['U'].tolist(),
'V': adapter['V'].tolist(),
'rank': adapter['rank']
}
for expert_id, adapter in self.domain_adapters.items()
},
'aggregation_count': self.total_aggregations,
'client_count': self.total_clients
}

def run_domain_meta_learning(self) -> Dict[str, Any]:
"""Run quick domain-meta (e.g., Reptile/ANIL) to produce good inits for new clients"""
self.logger.info(f"Running domain meta-learning for '{self.domain}'")

# Simulate domain meta-learning (Reptile/ANIL style)
meta_params = {
'domain': self.domain,
'meta_learning_rate': 0.01,
'inner_steps': 5,
'meta_initialization': {}
}

# Generate meta-initialization for each expert
for expert_id in range(30):
if expert_id in self.domain_adapters:
adapter = self.domain_adapters[expert_id]
# Meta-initialization is slightly perturbed domain adapter
meta_params['meta_initialization'][expert_id] = {
'U': (adapter['U'] + np.random.randn(*adapter['U'].shape) * 0.01).tolist(),
'V': (adapter['V'] + np.random.randn(*adapter['V'].shape) * 0.01).tolist(),
'rank': adapter['rank']
}

self.domain_meta_params = meta_params
return meta_params

def get_domain_summary(self) -> Dict[str, Any]:
"""Get domain summary to push up to Global Coordinator"""
return {
'domain': self.domain,
'total_clients': self.total_clients,
'total_aggregations': self.total_aggregations,
'domain_performance': {
'avg_router_norm': np.linalg.norm(self.domain_router),
'expert_count': len(self.domain_adapters),
'last_meta_update': datetime.now().isoformat()
},
'meta_params': self.domain_meta_params
}


class SFHFEServer:
"""
SF-HFE Global Coordinator - Tier 3

Operated by Developer (who has ZERO training data)

Responsibilities:
- Maintains global experts θ_k and global router trunk
- Runs global meta-learning (initializations, capacity allocation, domain trust)
- Learns transfer maps between domains
- Distributes global checkpoints + meta-params downstream
- Coordinates domain-based P2P gossip learning
"""

def __init__(self, num_experts: int = 30):
self.num_experts = num_experts

# Ensure reproducibility
setup_reproducibility()

# Logger with structured format
self.logger = logging.getLogger("GlobalCoordinator")
self.logger.info("Initializing SF-HFE Global Coordinator (Developer with ZERO data)")

# Global Memory with bounded storage
max_insights = safe_get_config(FL_CONFIG, "max_insights", 10000)
self.global_memory = GlobalMemory(max_insights=max_insights)

# Meta-Learning Engine
self.meta_engine = OnlineMAMLEngine(num_experts=num_experts)

# Domain-based components
self.domain_aggregators = {} # Domain -> DomainAggregator
self.domain_summaries = {} # Domain -> summary
self.clients = [] # All clients across domains

# Global components
self.global_experts = {} # Global experts θ_k
self.global_router_trunk = np.random.randn(num_experts) # Global router trunk

# Global meta-learning
self.expert_capacity_allocation = {}
self.domain_trust_weights = {}
self.domain_transfer_maps = {}

# Communication round tracking
self.round_count = 0
self.clients_seen = set()
self._lock = threading.Lock() # Thread safety

# Meta-learning trigger state with robust defaults
self.samples_since_meta = 0
self.time_since_meta = time.time()
self.last_meta_loss = 0.0

# Rate limiting for backpressure
self.request_count = 0
self.last_rate_reset = time.time()
self.max_requests_per_minute = safe_get_config(FL_CONFIG, "max_requests_per_minute", 100)

# Standardized client ID type
self.client_id_type = get_client_id_type()

# Initialize global experts
self._initialize_global_experts()

def _initialize_global_experts(self):
"""Initialize global experts θ_k"""
for expert_id in range(self.num_experts):
# Initialize with random low-rank adapters
self.global_experts[expert_id] = {
'U': np.random.randn(64, 16) * 0.01,
'V': np.random.randn(32, 16) * 0.01,
'rank': 16
}
self.expert_capacity_allocation[expert_id] = 1.0

def add_client(self, client):
"""Add a user client to the appropriate domain aggregator"""
with self._lock:
self.clients.append(client)
self.clients_seen.add(client.client_id)

# Add to appropriate domain aggregator
domain = getattr(client, 'domain', 'general')
if domain not in self.domain_aggregators:
self.domain_aggregators[domain] = DomainAggregator(domain)
self.logger.info(f"Created domain aggregator for '{domain}'")

self.domain_aggregators[domain].add_client(client)
self.logger.info(f"Added client {client.client_id} to domain '{domain}'")

def remove_client(self, client_id: int):
"""Remove a client from the server"""
with self._lock:
self.clients = [c for c in self.clients if c.client_id != client_id]
self.clients_seen.discard(client_id)

if self.gossip_manager:
self.gossip_manager.clients = self.clients
self.gossip_manager.num_clients = len(self.clients)

self.logger.info(f"Removed client {client_id}")

def coordinate_user_gossip(self):
"""Coordinate P2P gossip learning between users"""
self.logger.info(f"DEBUG: gossip_manager is None: {self.gossip_manager is None}")
self.logger.info(f"DEBUG: len(self.clients): {len(self.clients)}")
self.logger.info(f"DEBUG: self.clients: {[c.client_id for c in self.clients]}")

if self.gossip_manager is None or len(self.clients) < 2:
self.logger.info(f"DEBUG: Skipping gossip coordination - gossip_manager: {self.gossip_manager is not None}, clients: {len(self.clients)}")
return

self.logger.info("Coordinating user-based gossip learning")

# Update topology based on user similarity
self.logger.info("DEBUG: Calling gossip_manager.update_topology()")
self.gossip_manager.update_topology()
self.logger.info("DEBUG: update_topology() completed")

# Check if it's time for gossip exchange
self.logger.info("DEBUG: Checking should_exchange()")
if self.gossip_manager.should_exchange():
self.logger.info("DEBUG: Should exchange - calling _perform_user_gossip()")
# Perform gossip exchange between users
self._perform_user_gossip()
self.logger.info("DEBUG: _perform_user_gossip() completed")
else:
self.logger.info("DEBUG: Should not exchange yet")

def _perform_user_gossip(self):
"""Perform P2P gossip exchange between user clients"""
if self.gossip_manager is None:
return

try:
# Use the gossip manager to perform exchanges
self.gossip_manager.perform_gossip_round()
self.logger.info("User-based gossip exchange completed")
except Exception as e:
self.logger.error(f"Error in user gossip exchange: {e}")

def _perform_domain_gossip(self):
"""Perform gossip exchange within each domain"""
# Group clients by domain
domain_clients = defaultdict(list)
for client in self.clients:
domain_clients[client.domain].append(client)

# Perform gossip within each domain
for domain, clients in domain_clients.items():
if len(clients) >= 2: # Need at least 2 clients for gossip
self.logger.info(f"Performing gossip exchange in domain '{domain}' with {len(clients)} clients")

for client in clients:
# Get peer weights for this client
peer_weights = {}
for peer in clients:
if peer.client_id != client.client_id:
peer_weights[peer.client_id] = peer.get_expert_weights()

# Perform gossip exchange
if peer_weights:
client.gossip_exchange(peer_weights)

# Update gossip manager
self.gossip_manager.exchange_count += 1
self.gossip_manager.last_exchange_time = time.time()

def run_communication_round(self, clients: List = None) -> Dict:
"""
Run one domain-based federated learning round

Args:
clients: Optional list of user clients (uses self.clients if not provided)

Returns:
Dictionary with round results
"""
if clients is None:
clients = self.clients

with self._lock:
self.round_count += 1
self.logger.info(f"Global Coordinator: Starting domain-based FL round {self.round_count}")

# Step 1: Domain-based P2P gossip learning
self.logger.info("Step 1: Running domain-based P2P gossip learning")
gossip_results = self._run_domain_based_gossip()
self.logger.info(f"Step 1 completed: Gossip in {len(gossip_results)} domains")

# Step 2: Clients send deltas to domain aggregators
self.logger.info("Step 2: Clients sending deltas to domain aggregators")
delta_results = self._collect_client_deltas(clients)
self.logger.info(f"Step 2 completed: Collected deltas from {delta_results['clients_sent']} clients")

# Step 3: Domain aggregators aggregate and run domain meta-learning
self.logger.info("Step 3: Domain aggregators aggregating and running meta-learning")
domain_results = self._run_domain_aggregation()
self.logger.info(f"Step 3 completed: Processed {len(domain_results)} domains")

# Step 4: Global coordinator runs global meta-learning
self.logger.info("Step 4: Running global meta-learning")
global_result = self._run_global_meta_learning()
self.logger.info(f"Step 4 completed: Global meta-learning done")

# Step 5: Distribute global checkpoints
self.logger.info("Step 5: Distributing global checkpoints")
checkpoint_result = self._distribute_global_checkpoints(clients)
self.logger.info(f"Step 5 completed: Distributed to {checkpoint_result['clients_updated']} clients")

self.logger.info(f"Global Coordinator: Domain-based FL round {self.round_count} complete")

return {
"round": self.round_count,
"gossip_domains": len(gossip_results),
"clients_sent_deltas": delta_results['clients_sent'],
"domains_processed": len(domain_results),
"global_meta_learning": global_result,
"clients_updated": checkpoint_result['clients_updated'],
"status": "success"
}

def _run_domain_based_gossip(self) -> List[Dict[str, Any]]:
"""Run domain-based P2P gossip learning (same domain only)"""
gossip_results = []

# Group clients by domain
domain_clients = defaultdict(list)
for client in self.clients:
domain = getattr(client, 'domain', 'general')
domain_clients[domain].append(client)

# Perform gossip within each domain
for domain, clients in domain_clients.items():
if len(clients) >= 2: # Need at least 2 clients for gossip
self.logger.info(f"Gossiping in domain '{domain}' with {len(clients)} clients")

# Simple gossip: each client exchanges with 2-5 neighbors
for client in clients:
other_clients = [c for c in clients if c.client_id != client.client_id]
if len(other_clients) > 0:
# Select 2-5 neighbors (but not more than available)
num_neighbors = min(5, max(2, len(other_clients)))
if num_neighbors <= len(other_clients):
neighbors = np.random.choice(other_clients, size=num_neighbors, replace=False)
else:
neighbors = other_clients

# Simulate gossip exchange (in practice, clients would exchange deltas)
gossip_results.append({
'domain': domain,
'client_id': client.client_id,
'neighbors_count': len(neighbors),
'timestamp': datetime.now().isoformat()
})

return gossip_results

def _collect_client_deltas(self, clients: List) -> Dict[str, Any]:
"""Collect deltas from clients and send to domain aggregators"""
clients_sent = 0

for client in clients:
try:
# Simulate client packaging deltas
domain = getattr(client, 'domain', 'general')
if domain in self.domain_aggregators:
# Create mock deltas (in practice, clients would package real deltas)
deltas = {
'client_id': client.client_id,
'domain': domain,
'timestamp': datetime.now().isoformat(),
'expert_deltas': {i: {'U_delta': np.random.randn(64, 16), 'V_delta': np.random.randn(32, 16), 'rank': 16} for i in range(30)},
'router_delta': np.random.randn(30),
'sufficient_stats': {i: {'total_samples': 100, 'avg_loss': 0.5, 'confidence': 0.8} for i in range(30)},
'total_samples': 100
}

self.domain_aggregators[domain].receive_client_deltas(deltas)
clients_sent += 1

except Exception as e:
self.logger.error(f"Error collecting deltas from client {client.client_id}: {e}")

return {'clients_sent': clients_sent}

def _run_domain_aggregation(self) -> List[Dict[str, Any]]:
"""Run domain aggregation and meta-learning"""
domain_results = []

for domain, aggregator in self.domain_aggregators.items():
try:
# Aggregate domain adapters
domain_result = aggregator.aggregate_domain_adapters()

# Run domain meta-learning
meta_result = aggregator.run_domain_meta_learning()

# Get domain summary
summary = aggregator.get_domain_summary()
self.domain_summaries[domain] = summary

domain_results.append({
'domain': domain,
'aggregation_result': domain_result,
'meta_result': meta_result,
'summary': summary
})

except Exception as e:
self.logger.error(f"Error processing domain {domain}: {e}")

return domain_results

def _run_global_meta_learning(self) -> Dict[str, Any]:
"""Run global meta-learning"""
self.logger.info("Running global meta-learning")

# Learn global expert initializations
self._learn_global_expert_initializations()

# Learn expert capacity allocation
self._learn_expert_capacity_allocation()

# Learn domain trust weights
self._learn_domain_trust_weights()

# Learn domain transfer maps
self._learn_domain_transfer_maps()

return {
'global_experts_count': len(self.global_experts),
'expert_capacity_allocation': self.expert_capacity_allocation,
'domain_trust_weights': self.domain_trust_weights,
'domain_transfer_maps': self.domain_transfer_maps
}

def _learn_global_expert_initializations(self):
"""Learn initializations for global experts θ_k"""
for expert_id in range(self.num_experts):
# Aggregate initializations from all domains
domain_inits = []
for domain, summary in self.domain_summaries.items():
if 'meta_params' in summary and 'meta_initialization' in summary['meta_params']:
if expert_id in summary['meta_params']['meta_initialization']:
init = summary['meta_params']['meta_initialization'][expert_id]
# Convert back to numpy arrays
U = np.array(init['U'])
V = np.array(init['V'])
domain_inits.append({'U': U, 'V': V})

if domain_inits:
# Average across domains
avg_U = np.mean([di['U'] for di in domain_inits], axis=0)
avg_V = np.mean([di['V'] for di in domain_inits], axis=0)

self.global_experts[expert_id] = {
'U': avg_U,
'V': avg_V,
'rank': 16
}

def _learn_expert_capacity_allocation(self):
"""Learn when to spawn/split/merge experts (capacity allocation)"""
for expert_id in range(self.num_experts):
# Simple capacity allocation based on domain usage
total_usage = 0
for domain, summary in self.domain_summaries.items():
if 'domain_performance' in summary:
total_usage += summary['domain_performance'].get('expert_count', 0)

# Capacity based on usage (normalized)
self.expert_capacity_allocation[expert_id] = min(2.0, max(0.5, total_usage / max(1, len(self.domain_summaries))))

def _learn_domain_trust_weights(self):
"""Learn policy for how much to trust each domain in aggregation"""
for domain, summary in self.domain_summaries.items():
# Trust weight based on domain performance and client count
client_count = summary.get('total_clients', 0)
aggregation_count = summary.get('total_aggregations', 0)

# Higher trust for domains with more clients and aggregations
trust_weight = min(1.0, (client_count * 0.1 + aggregation_count * 0.05))
self.domain_trust_weights[domain] = trust_weight

def _learn_domain_transfer_maps(self):
"""Learn transfer maps between domains"""
domains = list(self.domain_summaries.keys())

for source_domain in domains:
self.domain_transfer_maps[source_domain] = {}
for target_domain in domains:
if source_domain != target_domain:
# Simple transfer map based on domain similarity
# In practice, this would be more sophisticated
similarity = np.random.uniform(0.1, 0.8) # Random similarity for now
self.domain_transfer_maps[source_domain][target_domain] = similarity

def _distribute_global_checkpoints(self, clients: List) -> Dict[str, Any]:
"""Distribute global checkpoints to clients"""
clients_updated = 0

global_checkpoints = {
'global_experts': {
expert_id: {
'U': expert['U'].tolist(),
'V': expert['V'].tolist()
}
for expert_id, expert in self.global_experts.items()
},
'global_router': self.global_router_trunk.tolist(),
'meta_params': {
'expert_capacity_allocation': self.expert_capacity_allocation,
'domain_trust_weights': self.domain_trust_weights,
'domain_transfer_maps': self.domain_transfer_maps
},
'round': self.round_count
}

for client in clients:
try:
# In practice, would send global_checkpoints to client
# For now, just count as updated
clients_updated += 1
except Exception as e:
self.logger.error(f"Error updating client {client.client_id}: {e}")

return {'clients_updated': clients_updated}

def _check_rate_limit(self) -> bool:
"""Check if request rate is within limits (backpressure)"""
current_time = time.time()
if current_time - self.last_rate_reset > 60: # Reset every minute
self.request_count = 0
self.last_rate_reset = current_time

if self.request_count >= self.max_requests_per_minute:
self.logger.warning("Rate limit exceeded, rejecting request")
return False

self.request_count += 1
return True

def receive_insights(self, insights: List[Dict]) -> Dict:
"""
Receive insights from clients with robust error handling

Args:
insights: List of insight dictionaries from clients

Returns:
Acknowledgment with current meta-parameters
"""
# Rate limiting check
if not self._check_rate_limit():
return {
"status": "rate_limited",
"message": "Server is overloaded, please try again later"
}

with self._lock: # Thread safety
if not insights:
self.logger.warning("Received empty insights list")
return {
"status": "error",
"message": "No insights provided"
}

self.round_count += 1

# Add to global memory with validation
valid_insights = 0
for insight in insights:
try:
self.global_memory.add_insight(insight)
valid_insights += 1

client_id = insight.get("client_id")
if client_id is not None:
self.clients_seen.add(client_id)

# Track samples
samples = insight.get("total_samples", 0)
if isinstance(samples, (int, float)) and samples >= 0:
self.samples_since_meta += int(samples)
except Exception as e:
self.logger.error(f"Error processing insight: {e}")

self.logger.info(
f"Server: Round {self.round_count} - "
f"Received {valid_insights}/{len(insights)} valid insights from {len(self.clients_seen)} unique clients"
)

# Check if meta-learning should trigger
should_trigger = self._check_meta_trigger(insights)

if should_trigger:
meta_params = self._trigger_meta_learning(insights)
else:
meta_params = self.meta_engine.get_meta_parameters()

return {
"status": "received",
"round": self.round_count,
"meta_params": meta_params,
"meta_learning_triggered": should_trigger,
"valid_insights": valid_insights,
}

def _check_meta_trigger(self, insights: List[Dict]) -> bool:
"""Check if meta-learning should be triggered with robust edge case handling"""
try:
triggers = META_LEARNING_CONFIG.get("triggers", {})

# Sample count trigger with validation
sample_count_threshold = triggers.get("sample_count", 1000)
if self.samples_since_meta >= sample_count_threshold:
self.logger.info(f"Server: Meta-learning triggered by sample count ({self.samples_since_meta})")
return True

# Time-based trigger with validation
time_threshold = triggers.get("time_seconds", 300)
if time.time() - self.time_since_meta >= time_threshold:
self.logger.info(f"Server: Meta-learning triggered by time threshold")
return True

# Drift trigger with robust validation
if insights:
drift_reports = 0
valid_insights = 0

for ins in insights:
if isinstance(ins, dict):
valid_insights += 1
drift_count = ins.get("drift_events_count", 0)
if isinstance(drift_count, (int, float)) and drift_count > 0:
drift_reports += 1

if valid_insights > 0 and drift_reports >= valid_insights * 0.3: # 30% of clients report drift
self.logger.info(f"Server: Meta-learning triggered by drift ({drift_reports}/{valid_insights} clients)")
return True

# Performance drop trigger with NaN/infinity protection
if insights:
losses = []
for ins in insights:
if isinstance(ins, dict):
loss = ins.get("avg_loss", 0.0)
if isinstance(loss, (int, float)) and not (np.isnan(loss) or np.isinf(loss)):
losses.append(float(loss))

if losses and self.last_meta_loss > 0:
avg_loss = np.mean(losses)
performance_drop_threshold = triggers.get("performance_drop", 0.15)

if avg_loss > self.last_meta_loss * (1 + performance_drop_threshold):
self.logger.info(f"Server: Meta-learning triggered by performance drop ({avg_loss:.4f} vs {self.last_meta_loss:.4f})")
return True

return False

except Exception as e:
self.logger.error(f"Error in meta-learning trigger check: {e}")
return False # Fail safe

def _trigger_meta_learning(self, insights: List[Dict]) -> Dict:
"""Trigger meta-learning update with robust error handling"""
try:
self.logger.info(f"Server: Running meta-learning update #{self.meta_engine.meta_updates + 1}")

# Get recent insights for meta-learning
recent_insights = self.global_memory.get_recent_insights(n=100)

# Meta-update
meta_params = self.meta_engine.meta_update(recent_insights)

# Reset trigger counters
self.samples_since_meta = 0
self.time_since_meta = time.time()

# Update last meta loss with validation
if insights:
losses = []
for ins in insights:
if isinstance(ins, dict):
loss = ins.get("avg_loss", 0.0)
if isinstance(loss, (int, float)) and not (np.isnan(loss) or np.isinf(loss)):
losses.append(float(loss))

if losses:
self.last_meta_loss = float(np.mean(losses))

self.logger.info(
f"Server: Meta-learning complete - "
f"Updated alphas for {len(meta_params.get('expert_alphas', {}))} experts"
)

return meta_params

except Exception as e:
self.logger.error(f"Error in meta-learning update: {e}")
# Return current parameters as fallback
return self.meta_engine.get_meta_parameters()

def aggregate_omoe_updates(self, omoe_updates: List) -> Dict:
"""
Aggregate OMEO model updates from multiple user clients

Args:
omoe_updates: List of OMEO model updates from clients

Returns:
Aggregated OMEO model parameters
"""
if not omoe_updates:
return {}

# Simple aggregation strategy - can be enhanced with FedAvg, etc.
aggregated = {
"tier_weights": {},
"expert_performance": {},
"meta_parameters": {},
"timestamp": datetime.now().isoformat(),
"client_count": len(omoe_updates)
}

# Aggregate tier weights from all clients
for update in omoe_updates:
if "tier_weights" in update:
for tier, weights in update["tier_weights"].items():
if tier not in aggregated["tier_weights"]:
aggregated["tier_weights"][tier] = []
aggregated["tier_weights"][tier].append(weights)

# Average the tier weights
for tier in aggregated["tier_weights"]:
weights_list = aggregated["tier_weights"][tier]
if weights_list:
# Simple averaging - can be enhanced
aggregated["tier_weights"][tier] = np.mean(weights_list, axis=0).tolist()

self.logger.info(f"Aggregated OMEO updates from {len(omoe_updates)} user clients")
return aggregated

def broadcast_omoe_parameters(self, clients: List, meta_parameters: Dict) -> int:
"""
Broadcast updated OMEO parameters to user clients

Args:
clients: List of user clients
meta_parameters: Updated meta-parameters from meta-learning

Returns:
Number of clients successfully updated
"""
updated_count = 0

for client in clients:
try:
# Send updated OMEO parameters to each user client
client.update_omoe_parameters(meta_parameters)
updated_count += 1
self.logger.debug(f"Updated OMEO parameters for user client {client.client_id}")
except Exception as e:
self.logger.error(f"Error updating OMEO parameters for client {client.client_id}: {e}")

self.logger.info(f"Broadcasted OMEO parameters to {updated_count}/{len(clients)} user clients")
return updated_count

def broadcast_meta_parameters(self, clients: List) -> int:
"""Broadcast meta-parameters to all clients"""
with self._lock: # Thread safety
meta_params = self.meta_engine.get_meta_parameters()

updated = 0
for client in clients:
try:
client.receive_meta_parameters(meta_params)
updated += 1
except Exception as e:
self.logger.error(f"Error broadcasting to client: {e}")

self.logger.info(f"Server: Broadcast meta-parameters to {updated} clients")

return updated

def get_stats(self) -> Dict:
"""Get comprehensive server statistics"""
with self._lock: # Thread safety
return {
"round_count": self.round_count,
"total_insights": self.global_memory.total_insights,
"unique_clients": len(self.clients_seen),
"memory_stats": self.global_memory.stats(),
"meta_engine_stats": self.meta_engine.stats(),
"samples_since_meta": self.samples_since_meta,
"rate_limit_stats": {
"requests_this_minute": self.request_count,
"max_requests_per_minute": self.max_requests_per_minute,
},
}

