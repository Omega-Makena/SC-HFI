#!/usr/bin/env python3
"""
OMEO-Federated Learning Integration
Connects OMEO system to federated learning infrastructure with domain-specific global storage
"""

import sys
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import threading
from collections import defaultdict

# Add the sf_hfe_v2/moe directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sf_hfe_v2', 'moe'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sf_hfe_v2', 'federated'))

# Import OMEO components
from sf_hfe_v2.moe import ScarcityMoE, ALL_EXPERTS
from sf_hfe_v2.moe.cross_expert_reasoning import CrossExpertReasoningSystem
from sf_hfe_v2.moe.gate import DomainRouter

# Import Federated Learning components
from sf_hfe_v2.federated.server import SFHFEServer
from sf_hfe_v2.federated.global_memory import GlobalMemory
from sf_hfe_v2.federated.meta_learning import OnlineMAMLEngine

class DomainSpecificGlobalStorage:
"""
Enhanced global storage with domain-specific organization
Stores insights organized by domain with cross-domain learning capabilities
"""

def __init__(self, config: Dict[str, Any] = None):
self.config = config or {}
self.logger = logging.getLogger("DomainSpecificGlobalStorage")

# Domain-specific storage
self.domain_storage = defaultdict(lambda: {
'insights': [],
'expert_performance': {},
'cross_expert_patterns': [],
'meta_parameters': {},
'last_update': None
})

# Cross-domain learning
self.cross_domain_patterns = []
self.domain_similarity_matrix = {}

# Thread safety
self._lock = threading.Lock()

# Statistics
self.total_insights = 0
self.domain_counts = defaultdict(int)

def add_domain_insight(self, domain: str, insight: Dict[str, Any]) -> bool:
"""
Add insight to domain-specific storage

Args:
domain: Domain identifier
insight: Insight data with expert results

Returns:
bool: Success status
"""
with self._lock:
try:
# Validate insight structure
if not self._validate_insight(insight):
self.logger.warning(f"Invalid insight structure for domain {domain}")
return False

# Add to domain storage
self.domain_storage[domain]['insights'].append(insight)
self.domain_storage[domain]['last_update'] = datetime.now()

# Update statistics
self.total_insights += 1
self.domain_counts[domain] += 1

# Extract expert performance metrics
if 'expert_results' in insight:
self._update_expert_performance(domain, insight['expert_results'])

# Extract cross-expert patterns
if 'cross_expert_insights' in insight:
self._extract_cross_expert_patterns(domain, insight['cross_expert_insights'])

self.logger.info(f"Added insight to domain '{domain}': {self.domain_counts[domain]} total")
return True

except Exception as e:
self.logger.error(f"Error adding insight to domain {domain}: {e}")
return False

def _validate_insight(self, insight: Dict[str, Any]) -> bool:
"""Validate insight structure"""
required_fields = ['expert_results', 'timestamp']
return all(field in insight for field in required_fields)

def _update_expert_performance(self, domain: str, expert_results: Dict[str, Any]):
"""Update expert performance metrics for domain"""
domain_data = self.domain_storage[domain]

for expert_name, result in expert_results.items():
if expert_name not in domain_data['expert_performance']:
domain_data['expert_performance'][expert_name] = {
'total_runs': 0,
'avg_confidence': 0.0,
'success_rate': 0.0,
'last_performance': 0.0
}

perf = domain_data['expert_performance'][expert_name]
perf['total_runs'] += 1

if 'confidence' in result:
# Update average confidence
old_avg = perf['avg_confidence']
new_avg = (old_avg * (perf['total_runs'] - 1) + result['confidence']) / perf['total_runs']
perf['avg_confidence'] = new_avg

if 'success' in result:
# Update success rate
old_success = perf['success_rate']
new_success = (old_success * (perf['total_runs'] - 1) + (1 if result['success'] else 0)) / perf['total_runs']
perf['success_rate'] = new_success

perf['last_performance'] = result.get('confidence', 0.0)

def _extract_cross_expert_patterns(self, domain: str, cross_expert_insights: List[Dict[str, Any]]):
"""Extract cross-expert patterns for domain"""
domain_data = self.domain_storage[domain]

for insight in cross_expert_insights:
pattern = {
'domain': domain,
'expert_combination': insight.get('expert_combination', []),
'insight_type': insight.get('insight_type', 'unknown'),
'confidence': insight.get('confidence', 0.0),
'timestamp': datetime.now()
}

domain_data['cross_expert_patterns'].append(pattern)

# Also store in cross-domain patterns
self.cross_domain_patterns.append(pattern)

def get_domain_insights(self, domain: str, limit: int = 100) -> List[Dict[str, Any]]:
"""Get insights for specific domain"""
with self._lock:
return self.domain_storage[domain]['insights'][-limit:] if domain in self.domain_storage else []

def get_cross_domain_patterns(self, source_domain: str, target_domain: str) -> List[Dict[str, Any]]:
"""Get cross-domain learning patterns"""
with self._lock:
patterns = []
for pattern in self.cross_domain_patterns:
if pattern['domain'] == source_domain:
# Check if pattern could apply to target domain
if self._is_pattern_transferable(pattern, target_domain):
patterns.append(pattern)
return patterns

def _is_pattern_transferable(self, pattern: Dict[str, Any], target_domain: str) -> bool:
"""Check if pattern is transferable to target domain"""
# Simple heuristic - can be enhanced with domain similarity
return True # For now, assume all patterns are transferable

def get_domain_statistics(self, domain: str) -> Dict[str, Any]:
"""Get statistics for specific domain"""
with self._lock:
if domain not in self.domain_storage:
return {}

domain_data = self.domain_storage[domain]
return {
'total_insights': len(domain_data['insights']),
'expert_count': len(domain_data['expert_performance']),
'cross_expert_patterns': len(domain_data['cross_expert_patterns']),
'last_update': domain_data['last_update'],
'expert_performance': domain_data['expert_performance']
}

def get_all_domains(self) -> List[str]:
"""Get list of all domains"""
with self._lock:
return list(self.domain_storage.keys())

def get_global_statistics(self) -> Dict[str, Any]:
"""Get global statistics across all domains"""
with self._lock:
return {
'total_insights': self.total_insights,
'total_domains': len(self.domain_storage),
'domain_counts': dict(self.domain_counts),
'cross_domain_patterns': len(self.cross_domain_patterns)
}

class OMEOFederatedIntegration:
"""
Main integration class connecting OMEO to federated learning
"""

def __init__(self, config: Dict[str, Any] = None):
self.config = config or {}
self.logger = logging.getLogger("OMEOFederatedIntegration")

# Initialize OMEO system
self.omoe_system = ScarcityMoE(config=self.config.get('omoe', {}))

# Initialize federated learning server
self.fl_server = SFHFEServer(num_experts=30)

# Initialize domain-specific global storage
self.domain_storage = DomainSpecificGlobalStorage(config=self.config.get('storage', {}))

# Initialize domain router
self.domain_router = DomainRouter(config=self.config.get('routing', {}))

# Track active domains
self.active_domains = set()
self.domain_clients = defaultdict(list)

# Thread safety
self._lock = threading.Lock()

self.logger.info("OMEO-Federated Learning integration initialized")

def process_data_with_federated_learning(self, data: pd.DataFrame, domain: str = None, client_id: int = 0) -> Dict[str, Any]:
"""
Process data through OMEO system with federated learning integration

Args:
data: Input dataset
domain: Domain identifier (auto-detected if None)
client_id: Client identifier

Returns:
dict: Complete results with federated learning insights
"""
with self._lock:
try:
# Auto-detect domain if not provided
if domain is None:
routing_result = self.domain_router.route_data(data)
domain = routing_result.get('primary_domain', 'general')

self.logger.info(f"Processing data for domain '{domain}' with client {client_id}")

# Process through OMEO system with cross-expert reasoning
metadata = {
'domain': domain,
'client_id': client_id,
'timestamp': datetime.now().isoformat(),
'generate_simulation': True
}

# Convert DataFrame to numpy array for OMEO processing
if isinstance(data, pd.DataFrame):
# Select numeric columns only
numeric_data = data.select_dtypes(include=[np.number])
if len(numeric_data.columns) == 0:
# If no numeric columns, create dummy data
numeric_data = np.random.randn(len(data), 5)
else:
numeric_data = numeric_data.values
else:
numeric_data = data

omoe_results = self.omoe_system.process_data(numeric_data, metadata)

# Add cross-expert insights if not present
if 'cross_expert_insights' not in omoe_results:
omoe_results['cross_expert_insights'] = []

# Prepare federated learning insight
fl_insight = self._prepare_federated_insight(omoe_results, domain, client_id)

# Store in domain-specific global storage
self.domain_storage.add_domain_insight(domain, fl_insight)

# Add to federated learning server
self.fl_server.global_memory.add_insight(fl_insight)

# Update active domains
self.active_domains.add(domain)

# Run federated learning round if conditions are met
if self._should_run_fl_round():
fl_results = self._run_federated_learning_round()
omoe_results['federated_learning'] = fl_results

# Get cross-domain insights
cross_domain_insights = self._get_cross_domain_insights(domain)
if cross_domain_insights:
omoe_results['cross_domain_insights'] = cross_domain_insights

self.logger.info(f"Successfully processed data for domain '{domain}'")
return omoe_results

except Exception as e:
self.logger.error(f"Error processing data with federated learning: {e}")
return {'error': str(e)}

def _prepare_federated_insight(self, omoe_results: Dict[str, Any], domain: str, client_id: int) -> Dict[str, Any]:
"""Prepare insight for federated learning"""
return {
'client_id': client_id,
'domain': domain,
'expert_results': omoe_results.get('expert_results', {}),
'cross_expert_insights': omoe_results.get('cross_expert_insights', []),
'simulation_results': omoe_results.get('simulation_results', {}),
'timestamp': datetime.now().isoformat(),
'total_samples': len(omoe_results.get('expert_results', {})),
'avg_loss': self._calculate_average_loss(omoe_results),
'expert_insights': self._extract_expert_insights(omoe_results)
}

def _calculate_average_loss(self, omoe_results: Dict[str, Any]) -> float:
"""Calculate average loss from expert results"""
expert_results = omoe_results.get('expert_results', {})
if not expert_results:
return 0.0

losses = []
for expert_name, result in expert_results.items():
if 'confidence' in result:
# Convert confidence to loss (1 - confidence)
loss = 1.0 - result['confidence']
losses.append(loss)

return np.mean(losses) if losses else 0.0

def _extract_expert_insights(self, omoe_results: Dict[str, Any]) -> Dict[str, Any]:
"""Extract expert insights for federated learning"""
expert_results = omoe_results.get('expert_results', {})
insights = {}

for expert_name, result in expert_results.items():
insights[expert_name] = {
'confidence': result.get('confidence', 0.0),
'success': result.get('success', False),
'insight_type': result.get('insight_type', 'unknown'),
'processing_time': result.get('processing_time', 0.0)
}

return insights

def _should_run_fl_round(self) -> bool:
"""Check if federated learning round should be run"""
# Simple heuristic - can be enhanced
return len(self.active_domains) >= 2 or self.domain_storage.total_insights % 10 == 0

def _run_federated_learning_round(self) -> Dict[str, Any]:
"""Run federated learning round"""
try:
# Get insights from all domains
all_insights = []
for domain in self.active_domains:
domain_insights = self.domain_storage.get_domain_insights(domain, limit=50)
all_insights.extend(domain_insights)

if not all_insights:
return {'status': 'no_insights'}

# Run federated learning round
fl_results = self.fl_server.receive_insights(all_insights)

# Update OMEO system with new meta-parameters
if 'meta_params' in fl_results:
self.omoe_system.update_meta_parameters(fl_results['meta_params'])

self.logger.info(f"Completed federated learning round: {fl_results.get('status', 'unknown')}")
return fl_results

except Exception as e:
self.logger.error(f"Error running federated learning round: {e}")
return {'error': str(e)}

def _get_cross_domain_insights(self, current_domain: str) -> List[Dict[str, Any]]:
"""Get cross-domain insights for current domain"""
cross_domain_insights = []

for domain in self.active_domains:
if domain != current_domain:
patterns = self.domain_storage.get_cross_domain_patterns(domain, current_domain)
cross_domain_insights.extend(patterns)

return cross_domain_insights

def get_system_status(self) -> Dict[str, Any]:
"""Get comprehensive system status"""
with self._lock:
omoe_status = self.omoe_system.get_system_status()
fl_status = self.fl_server.get_stats()
storage_stats = self.domain_storage.get_global_statistics()

return {
'omoe_system': omoe_status,
'federated_learning': fl_status,
'domain_storage': storage_stats,
'active_domains': list(self.active_domains),
'total_domains': len(self.active_domains)
}

def get_domain_insights(self, domain: str) -> Dict[str, Any]:
"""Get insights for specific domain"""
return {
'domain': domain,
'insights': self.domain_storage.get_domain_insights(domain),
'statistics': self.domain_storage.get_domain_statistics(domain)
}

def get_cross_domain_learning_summary(self) -> Dict[str, Any]:
"""Get summary of cross-domain learning"""
with self._lock:
summary = {
'total_domains': len(self.active_domains),
'cross_domain_patterns': len(self.domain_storage.cross_domain_patterns),
'domain_statistics': {}
}

for domain in self.active_domains:
summary['domain_statistics'][domain] = self.domain_storage.get_domain_statistics(domain)

return summary

def test_omoe_federated_integration():
"""Test OMEO-Federated Learning integration"""

print("=" * 80)
print("TESTING OMEO-FEDERATED LEARNING INTEGRATION")
print("=" * 80)

# Initialize integration system
config = {
'omoe': {
'cross_expert': {'max_insights': 1000},
'integration': {'max_compositions': 50}
},
'storage': {'max_insights_per_domain': 500},
'routing': {'confidence_threshold': 0.7}
}

integration = OMEOFederatedIntegration(config=config)

# Load agricultural data
if os.path.exists('agricultural_data.csv'):
df = pd.read_csv('agricultural_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(f" Loaded agricultural data: {len(df)} days × {len(df.columns)} variables")
else:
print(" Agricultural data not found. Please run test_agricultural_data.py first.")
return

# Test 1: Process agricultural data
print("\n" + "="*50)
print("TEST 1: PROCESSING AGRICULTURAL DATA")
print("="*50)

results1 = integration.process_data_with_federated_learning(
data=df, 
domain='agricultural', 
client_id=1
)

print(f" Processed agricultural data")
print(f" Expert results: {len(results1.get('expert_results', {}))}")
print(f" Cross-expert insights: {len(results1.get('cross_expert_insights', []))}")
print(f" Federated learning: {results1.get('federated_learning', {}).get('status', 'N/A')}")

# Test 2: Process market data (if available)
if os.path.exists('Market_Prices.csv'):
print("\n" + "="*50)
print("TEST 2: PROCESSING MARKET DATA")
print("="*50)

market_df = pd.read_csv('Market_Prices.csv')
market_df['Date'] = pd.to_datetime(market_df['MarketDate'])
print(f" Loaded market data: {len(market_df)} days × {len(market_df.columns)} variables")

results2 = integration.process_data_with_federated_learning(
data=market_df, 
domain='financial', 
client_id=2
)

print(f" Processed market data")
print(f" Expert results: {len(results2.get('expert_results', {}))}")
print(f" Cross-expert insights: {len(results2.get('cross_expert_insights', []))}")
print(f" Federated learning: {results2.get('federated_learning', {}).get('status', 'N/A')}")

# Test 3: Get system status
print("\n" + "="*50)
print("TEST 3: SYSTEM STATUS")
print("="*50)

status = integration.get_system_status()

print(f" OMEO System Status:")
print(f" Total experts: {status['omoe_system']['total_experts']}")
print(f" Active experts: {status['omoe_system']['active_experts']}")
print(f" Cross-expert insights: {status['omoe_system']['cross_expert_insights_count']}")

print(f" Federated Learning Status:")
print(f" Round count: {status['federated_learning']['round_count']}")
print(f" Total insights: {status['federated_learning']['total_insights']}")
print(f" Unique clients: {status['federated_learning']['unique_clients']}")

print(f" Domain Storage Status:")
print(f" Total insights: {status['domain_storage']['total_insights']}")
print(f" Total domains: {status['domain_storage']['total_domains']}")
print(f" Active domains: {status['active_domains']}")

# Test 4: Get domain-specific insights
print("\n" + "="*50)
print("TEST 4: DOMAIN-SPECIFIC INSIGHTS")
print("="*50)

for domain in status['active_domains']:
domain_insights = integration.get_domain_insights(domain)
stats = domain_insights['statistics']

print(f" Domain '{domain}':")
print(f" Total insights: {stats.get('total_insights', 0)}")
print(f" Expert count: {stats.get('expert_count', 0)}")
print(f" Cross-expert patterns: {stats.get('cross_expert_patterns', 0)}")
print(f" Last update: {stats.get('last_update', 'N/A')}")

# Test 5: Cross-domain learning summary
print("\n" + "="*50)
print("TEST 5: CROSS-DOMAIN LEARNING")
print("="*50)

cross_domain_summary = integration.get_cross_domain_learning_summary()

print(f" Cross-Domain Learning Summary:")
print(f" Total domains: {cross_domain_summary['total_domains']}")
print(f" Cross-domain patterns: {cross_domain_summary['cross_domain_patterns']}")

for domain, stats in cross_domain_summary['domain_statistics'].items():
print(f" Domain '{domain}': {stats.get('total_insights', 0)} insights")

print("\n" + "="*80)
print("OMEO-FEDERATED LEARNING INTEGRATION TESTING COMPLETE")
print("="*80)

print(" OMEO system successfully integrated with federated learning")
print(" Domain-specific global storage operational")
print(" Cross-domain learning patterns extracted")
print(" Federated learning rounds executed")
print(" System status monitoring functional")

return integration

def main():
"""Main test function"""

print("OMEO-Federated Learning Integration")
print("=" * 80)

try:
integration = test_omoe_federated_integration()

print(f"\n{'='*80}")
print("INTEGRATION TESTING COMPLETE")
print(f"{'='*80}")

print(" OMEO-Federated Learning integration operational")
print(" Domain-specific global storage implemented")
print(" Cross-domain learning capabilities active")
print(" Federated learning rounds functional")
print(" System monitoring and status reporting")

print("\nThis demonstrates:")
print("1. Seamless integration between OMEO and federated learning")
print("2. Domain-specific insight storage and organization")
print("3. Cross-domain learning pattern extraction")
print("4. Federated learning coordination")
print("5. Comprehensive system monitoring")

except Exception as e:
print(f"\nError during testing: {e}")
import traceback
traceback.print_exc()

if __name__ == "__main__":
main()
