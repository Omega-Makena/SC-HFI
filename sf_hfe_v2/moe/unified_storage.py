"""
Unified Storage System - Single storage solution for both Federated Learning and MoE
Combines the best features of both global_memory.py and storage.py
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
from datetime import datetime, timedelta
import threading
import logging
from collections import defaultdict

class UnifiedStorage:
"""
Unified Storage System - Single storage solution for both Federated Learning and MoE

Features:
- Federated Learning: Client-based insights with domain partitioning
- MoE System: Tier-based insights with meta-learning triggers
- Privacy-preserving: Stores insights, not raw data
- Thread-safe: Supports concurrent access
- Bounded storage: Prevents unlimited growth
- Meta-learning triggers: Automatic learning updates
"""

def __init__(self, config: Optional[Dict] = None):
"""Initialize Unified Storage System"""
self.config = config or {}

# Storage structure - supports both federated and MoE
self.insights = [] # All insights (bounded)
self.domain_partitions = defaultdict(list) # domain -> insights
self.client_insights = defaultdict(list) # client_id -> insights
self.tier_insights = defaultdict(list) # tier -> insights
self.domain_baskets = defaultdict(list) # domain_basket -> insights

# Aggregated metrics and triggers
self.aggregated_metrics = {}
self.meta_learning_triggers = {}

# Configuration
self.max_insights = self.config.get('max_insights', 10000)
self.storage_threshold = self.config.get('storage_threshold', 100)
self.retention_period = self.config.get('retention_period', 30) # days
self.aggregation_window = self.config.get('aggregation_window', 7) # days

# Thread safety
self._lock = threading.Lock()

# Statistics
self.total_insights = 0
self.unique_clients = set()

# Insight validation schema
self.required_fields = {
"insight_type": str, # 'federated' or 'moe'
"timestamp": (str, datetime),
"insight_data": dict
}

# Initialize logging
self.logger = logging.getLogger(__name__)

def _validate_insight(self, insight: Dict) -> bool:
"""Validate insight schema"""
try:
for field, expected_type in self.required_fields.items():
if field not in insight:
self.logger.warning(f"Missing field {field} in insight")
return False
if not isinstance(insight[field], expected_type):
self.logger.warning(f"Invalid type for field {field}")
return False
return True
except Exception as e:
self.logger.error(f"Error validating insight: {e}")
return False

def _trim_memory(self):
"""Trim memory to prevent unlimited growth"""
if len(self.insights) > self.max_insights:
# Remove oldest insights (FIFO)
to_remove = len(self.insights) - self.max_insights
removed_insights = self.insights[:to_remove]
self.insights = self.insights[to_remove:]

# Clean up all partitions
for insight in removed_insights:
self._remove_from_partitions(insight)

self.logger.info(f"Trimmed memory: removed {to_remove} old insights")

def _remove_from_partitions(self, insight: Dict):
"""Remove insight from all partitions"""
client_id = insight.get("client_id")
domain = insight.get("domain", "general")
tier = insight.get("tier")
domain_basket = insight.get("domain_basket", "general")

# Remove from client insights
if client_id in self.client_insights:
try:
self.client_insights[client_id].remove(insight)
except ValueError:
pass # Already removed

# Remove from domain partitions
if domain in self.domain_partitions:
try:
self.domain_partitions[domain].remove(insight)
except ValueError:
pass # Already removed

# Remove from tier insights
if tier in self.tier_insights:
try:
self.tier_insights[tier].remove(insight)
except ValueError:
pass # Already removed

# Remove from domain baskets
if domain_basket in self.domain_baskets:
try:
self.domain_baskets[domain_basket].remove(insight)
except ValueError:
pass # Already removed

def add_federated_insight(self, insight: Dict):
"""Add federated learning insight"""
with self._lock:
# Validate and format insight
federated_insight = {
"insight_type": "federated",
"timestamp": datetime.now(),
"insight_data": insight,
"client_id": insight.get("client_id"),
"domain": insight.get("domain", "general"),
"avg_loss": insight.get("avg_loss", 0.0),
"total_samples": insight.get("total_samples", 0)
}

if not self._validate_insight(federated_insight):
self.logger.warning("Skipping invalid federated insight")
return

self._store_insight(federated_insight)

def add_moe_insight(self, tier_results: Dict[str, Any], routing_result: Dict[str, Any]):
"""Add MoE system insight"""
with self._lock:
# Create insight entry for MoE
moe_insight = {
"insight_type": "moe",
"timestamp": datetime.now(),
"insight_data": {
"tier_results": tier_results,
"routing_result": routing_result,
"insight_summary": self._create_insight_summary(tier_results)
},
"domain_basket": routing_result.get('basket_label', 'unknown'),
"routing_confidence": routing_result.get('confidence_score', 0.0),
"tier": "all_tiers"
}

if not self._validate_insight(moe_insight):
self.logger.warning("Skipping invalid MoE insight")
return

self._store_insight(moe_insight)

# Store individual tier insights
for tier_name, tier_result in tier_results.items():
tier_insight = {
"insight_type": "moe_tier",
"timestamp": datetime.now(),
"insight_data": tier_result,
"domain_basket": routing_result.get('basket_label', 'unknown'),
"routing_confidence": routing_result.get('confidence_score', 0.0),
"tier": tier_name
}

if self._validate_insight(tier_insight):
self._store_insight(tier_insight)

def _store_insight(self, insight: Dict):
"""Store insight in all relevant partitions"""
self.insights.append(insight)
self.total_insights += 1

# Store in client insights (if applicable)
client_id = insight.get("client_id")
if client_id is not None:
self.unique_clients.add(client_id)
self.client_insights[client_id].append(insight)

# Store in domain partitions
domain = insight.get("domain", "general")
self.domain_partitions[domain].append(insight)

# Store in tier insights
tier = insight.get("tier")
if tier is not None:
self.tier_insights[tier].append(insight)

# Store in domain baskets
domain_basket = insight.get("domain_basket", "general")
self.domain_baskets[domain_basket].append(insight)

# Update aggregated metrics
self._update_aggregated_metrics(insight)

# Check for meta-learning triggers
self._check_meta_learning_triggers(insight)

# Trim memory if needed
self._trim_memory()

def store_results(self, expert_results: Dict[str, Any], metadata: Dict[str, Any] = None):
"""
Store expert results in unified storage

Args:
expert_results: Dictionary of expert results
metadata: Additional metadata about the processing
"""
metadata = metadata or {}

# Create insight from expert results
insight = {
"timestamp": datetime.now().isoformat(),
"type": "expert_results",
"expert_results": expert_results,
"metadata": metadata,
"domain": metadata.get("domain", "general"),
"client_id": metadata.get("client_id"),
"num_experts": len(expert_results),
"total_confidence": sum(
result_info.get('result', {}).get('confidence', 0.0) 
for result_info in expert_results.values()
) / len(expert_results) if expert_results else 0.0
}

# Store the insight
self._store_insight(insight)

# Log storage
self.logger.debug(f"Stored expert results with {len(expert_results)} experts")

def get_insight_count(self) -> int:
"""Get total number of insights stored"""
return len(self.insights)

def _create_insight_summary(self, tier_results: Dict[str, Any]) -> Dict[str, Any]:
"""Create a summary of insights from all tiers"""
summary = {
'tier1_summary': tier_results.get('tier1', {}).get('fingerprint', {}),
'tier2_summary': tier_results.get('tier2', {}).get('relationship_summary', {}),
'tier3_summary': tier_results.get('tier3', {}).get('dynamical_summary', {}),
'tier4_summary': tier_results.get('tier4', {}).get('semantic_summary', {}),
'tier5_summary': tier_results.get('tier5', {}).get('projective_summary', {}),
'tier6_summary': tier_results.get('tier6', {}).get('meta_summary', {}),
'overall_quality': self._calculate_overall_quality(tier_results)
}

return summary

def _calculate_overall_quality(self, tier_results: Dict[str, Any]) -> float:
"""Calculate overall quality score from all tiers"""
quality_scores = []

# Extract quality scores from each tier
if tier_results.get('tier1', {}).get('quality', {}).get('overall_score'):
quality_scores.append(tier_results['tier1']['quality']['overall_score'])

if tier_results.get('tier2', {}).get('relationship_summary', {}).get('strong_correlations'):
strong_correlations = tier_results['tier2']['relationship_summary']['strong_correlations']
quality_scores.append(min(strong_correlations / 10, 1.0)) # Normalize

if tier_results.get('tier3', {}).get('dynamical_summary', {}).get('trending_variables'):
trending_vars = tier_results['tier3']['dynamical_summary']['trending_variables']
quality_scores.append(min(trending_vars / 5, 1.0)) # Normalize

if tier_results.get('tier4', {}).get('semantic_summary', {}).get('important_features'):
important_features = tier_results['tier4']['semantic_summary']['important_features']
quality_scores.append(min(important_features / 5, 1.0)) # Normalize

if tier_results.get('tier5', {}).get('projective_summary', {}).get('average_forecast_confidence'):
forecast_confidence = tier_results['tier5']['projective_summary']['average_forecast_confidence']
quality_scores.append(forecast_confidence)

if tier_results.get('tier6', {}).get('meta_summary', {}).get('performance_score'):
performance_score = tier_results['tier6']['meta_summary']['performance_score']
quality_scores.append(performance_score)

return np.mean(quality_scores) if quality_scores else 0.0

def _update_aggregated_metrics(self, insight: Dict):
"""Update aggregated metrics for insights"""
domain = insight.get("domain", "general")
domain_basket = insight.get("domain_basket", "general")

# Update domain metrics
if domain not in self.aggregated_metrics:
self.aggregated_metrics[domain] = {
'total_insights': 0,
'average_quality': 0.0,
'average_confidence': 0.0,
'last_updated': datetime.now(),
'quality_history': [],
'confidence_history': []
}

metrics = self.aggregated_metrics[domain]
metrics['total_insights'] += 1

# Update quality history
quality = insight.get('insight_data', {}).get('insight_summary', {}).get('overall_quality', 0.0)
if quality > 0:
metrics['quality_history'].append(quality)
metrics['average_quality'] = np.mean(metrics['quality_history'])

# Update confidence history
confidence = insight.get('routing_confidence', 0.0)
if confidence > 0:
metrics['confidence_history'].append(confidence)
metrics['average_confidence'] = np.mean(metrics['confidence_history'])

metrics['last_updated'] = datetime.now()

# Keep only recent history
max_history = 100
if len(metrics['quality_history']) > max_history:
metrics['quality_history'] = metrics['quality_history'][-max_history:]
if len(metrics['confidence_history']) > max_history:
metrics['confidence_history'] = metrics['confidence_history'][-max_history:]

def _check_meta_learning_triggers(self, insight: Dict):
"""Check if meta-learning should be triggered"""
domain = insight.get("domain", "general")
domain_basket = insight.get("domain_basket", "general")

if domain not in self.aggregated_metrics:
return

metrics = self.aggregated_metrics[domain]

# Check various trigger conditions
if metrics['total_insights'] >= self.storage_threshold:
self.meta_learning_triggers[domain] = {
'timestamp': datetime.now(),
'reason': f"Storage threshold reached: {metrics['total_insights']} insights",
'metrics': metrics.copy()
}
elif metrics['average_quality'] < 0.3:
self.meta_learning_triggers[domain] = {
'timestamp': datetime.now(),
'reason': f"Low quality threshold: {metrics['average_quality']:.3f}",
'metrics': metrics.copy()
}
elif metrics['average_confidence'] < 0.4:
self.meta_learning_triggers[domain] = {
'timestamp': datetime.now(),
'reason': f"Low confidence threshold: {metrics['average_confidence']:.3f}",
'metrics': metrics.copy()
}

# Query methods for both federated and MoE
def get_recent_insights(self, n: int = 100) -> List[Dict]:
"""Get n most recent insights"""
with self._lock:
return self.insights[-n:] if len(self.insights) >= n else self.insights

def get_insights_by_domain(self, domain: str) -> List[Dict]:
"""Get all insights for a specific domain"""
with self._lock:
return self.domain_partitions.get(domain, [])

def get_insights_by_client(self, client_id: int) -> List[Dict]:
"""Get all insights from a specific client"""
with self._lock:
return self.client_insights.get(client_id, [])

def get_insights_by_tier(self, tier: str) -> List[Dict]:
"""Get all insights for a specific tier"""
with self._lock:
return self.tier_insights.get(tier, [])

def get_insights_by_domain_basket(self, domain_basket: str) -> List[Dict]:
"""Get all insights for a specific domain basket"""
with self._lock:
return self.domain_baskets.get(domain_basket, [])

def get_federated_insights(self) -> List[Dict]:
"""Get all federated learning insights"""
with self._lock:
return [insight for insight in self.insights if insight.get('insight_type') == 'federated']

def get_moe_insights(self) -> List[Dict]:
"""Get all MoE system insights"""
with self._lock:
return [insight for insight in self.insights if insight.get('insight_type') == 'moe']

def get_insights(self, domain: str = None, start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
"""Retrieve insights from storage with filtering"""
with self._lock:
if domain:
insights = self.domain_partitions.get(domain, [])
else:
insights = self.insights.copy()

# Filter by date range
if start_date:
insights = [insight for insight in insights if insight['timestamp'] >= start_date]
if end_date:
insights = [insight for insight in insights if insight['timestamp'] <= end_date]

return {
'insights': insights,
'total_count': len(insights),
'domains': list(self.domain_partitions.keys()),
'domain_baskets': list(self.domain_baskets.keys()),
'tiers': list(self.tier_insights.keys()),
'aggregated_metrics': self.aggregated_metrics
}

def get_aggregated_metrics(self, domain: str = None) -> Dict[str, Any]:
"""Get aggregated metrics for a domain or all domains"""
with self._lock:
if domain:
return self.aggregated_metrics.get(domain, {})
else:
return self.aggregated_metrics.copy()

def get_meta_learning_triggers(self) -> Dict[str, Any]:
"""Get meta-learning triggers"""
with self._lock:
return self.meta_learning_triggers.copy()

def get_insight_count(self) -> int:
"""Get total number of insights stored"""
with self._lock:
return len(self.insights)

def stats(self) -> Dict:
"""Get comprehensive storage statistics"""
with self._lock:
return {
"total_insights": self.total_insights,
"current_insights": len(self.insights),
"max_insights": self.max_insights,
"unique_clients": len(self.unique_clients),
"domains": list(self.domain_partitions.keys()),
"domain_baskets": list(self.domain_baskets.keys()),
"tiers": list(self.tier_insights.keys()),
"insights_per_domain": {
domain: len(insights)
for domain, insights in self.domain_partitions.items()
},
"insights_per_tier": {
tier: len(insights)
for tier, insights in self.tier_insights.items()
},
"memory_utilization": len(self.insights) / self.max_insights,
"meta_learning_triggers": len(self.meta_learning_triggers),
"aggregated_metrics_count": len(self.aggregated_metrics)
}

def export_insights(self, domain: str = None, format: str = 'json') -> str:
"""Export insights to a specific format"""
insights_data = self.get_insights(domain)

if format == 'json':
return json.dumps(insights_data, default=str, indent=2)
elif format == 'pickle':
return pickle.dumps(insights_data)
else:
raise ValueError(f"Unsupported format: {format}")

def import_insights(self, data: str, format: str = 'json'):
"""Import insights from a specific format"""
if format == 'json':
insights_data = json.loads(data)
elif format == 'pickle':
insights_data = pickle.loads(data)
else:
raise ValueError(f"Unsupported format: {format}")

# Process imported insights
with self._lock:
for insight in insights_data.get('insights', []):
self._store_insight(insight)

def clear_storage(self, domain: str = None):
"""Clear storage for a domain or all domains"""
with self._lock:
if domain:
# Remove insights for specific domain
insights_to_remove = [insight for insight in self.insights if insight.get('domain') == domain]
for insight in insights_to_remove:
self._remove_from_partitions(insight)
self.insights.remove(insight)

# Clean up partitions
if domain in self.domain_partitions:
del self.domain_partitions[domain]
if domain in self.aggregated_metrics:
del self.aggregated_metrics[domain]
if domain in self.meta_learning_triggers:
del self.meta_learning_triggers[domain]
else:
# Clear all storage
self.insights.clear()
self.domain_partitions.clear()
self.client_insights.clear()
self.tier_insights.clear()
self.domain_baskets.clear()
self.aggregated_metrics.clear()
self.meta_learning_triggers.clear()
self.total_insights = 0
self.unique_clients.clear()

def update_parameters(self, params: Dict[str, Any]):
"""Update storage parameters"""
with self._lock:
self.config.update(params)

if 'max_insights' in params:
self.max_insights = params['max_insights']
if 'storage_threshold' in params:
self.storage_threshold = params['storage_threshold']
if 'retention_period' in params:
self.retention_period = params['retention_period']
if 'aggregation_window' in params:
self.aggregation_window = params['aggregation_window']

def get_storage_statistics(self) -> Dict[str, Any]:
"""Get storage statistics"""
return self.stats()
