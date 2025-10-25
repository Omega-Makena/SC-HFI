"""
Expert Integration Layer
Connects the cross-expert reasoning system with existing expert architecture
Implements the reviewer's vision of moving from first-order to high-order understanding
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
from collections import defaultdict, deque
import json

from .cross_expert_reasoning import (
CrossExpertReasoningSystem, 
ExpertOutput, 
CompositionalInsight,
InterExpertGraph,
CompositionalReasoningEngine,
MetaController,
MemoryCurator
)
from .base_expert import BaseExpert


class ExpertIntegrationManager:
"""
Manages integration between individual experts and cross-expert reasoning system
"""

def __init__(self, config: Dict[str, Any] = None):
self.config = config or {}
self.logger = logging.getLogger(__name__)

# Initialize cross-expert reasoning system
self.cross_expert_system = CrossExpertReasoningSystem(config)

# Expert registry
self.expert_registry = {}
self.expert_outputs_cache = deque(maxlen=1000)

# Integration state
self.integration_history = deque(maxlen=1000)
self.insight_evolution = defaultdict(list)

def register_expert(self, expert: BaseExpert) -> bool:
"""Register an expert with the integration system"""
try:
self.expert_registry[expert.expert_id] = expert
self.logger.info(f"Registered expert: {expert.name} (ID: {expert.expert_id})")
return True
except Exception as e:
self.logger.error(f"Error registering expert {expert.name}: {e}")
return False

def process_data_through_experts(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
"""Process data through all registered experts and cross-expert reasoning"""

metadata = metadata or {}
expert_outputs = []

# Process data through each expert
for expert_id, expert in self.expert_registry.items():
try:
# Process data through expert
expert_result = expert.process_data(data, metadata)

# Convert to ExpertOutput format
expert_output = ExpertOutput(
expert_id=expert_id,
expert_name=expert.name,
relation_types=expert.relation_types,
confidence=expert_result.get('confidence', 0.0),
insights=expert_result.get('insights', []),
embeddings=self._extract_embeddings(expert_result),
metadata=expert_result,
timestamp=time.time()
)

expert_outputs.append(expert_output)

# Update expert online
expert.update_online(data, expert_result.get('feedback', {}))

except Exception as e:
self.logger.error(f"Error processing data through expert {expert.name}: {e}")
continue

# Process through cross-expert reasoning system
cross_expert_results = self.cross_expert_system.process_expert_outputs(expert_outputs)

# Store in cache
self.expert_outputs_cache.extend(expert_outputs)

# Generate integration insights
integration_insights = self._generate_integration_insights(expert_outputs, cross_expert_results)

# Store integration results
integration_result = {
'expert_outputs': expert_outputs,
'cross_expert_results': cross_expert_results,
'integration_insights': integration_insights,
'timestamp': time.time()
}

self.integration_history.append(integration_result)

return integration_result

def _extract_embeddings(self, expert_result: Dict[str, Any]) -> np.ndarray:
"""Extract embeddings from expert result"""
try:
# Try to extract embeddings from various possible locations
if 'embeddings' in expert_result:
return np.array(expert_result['embeddings'])
elif 'features' in expert_result:
return np.array(expert_result['features'])
elif 'analysis' in expert_result:
# Extract numerical features from analysis
analysis = expert_result['analysis']
if isinstance(analysis, dict):
features = []
for key, value in analysis.items():
if isinstance(value, (int, float)):
features.append(value)
elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
features.extend(value[:5]) # Take first 5 elements
return np.array(features) if features else np.zeros(128)

# Default embedding
return np.zeros(128)
except Exception as e:
self.logger.error(f"Error extracting embeddings: {e}")
return np.zeros(128)

def _generate_integration_insights(self, expert_outputs: List[ExpertOutput], 
cross_expert_results: Dict[str, Any]) -> List[str]:
"""Generate insights from the integration process"""

insights = []

# Expert collaboration insights
expert_count = len(expert_outputs)
relation_types = set()
for output in expert_outputs:
relation_types.update(output.relation_types)

insights.append(f"🔗 Expert Collaboration: {expert_count} experts across {len(relation_types)} relation types")

# Cross-expert reasoning insights
compositional_insights = cross_expert_results.get('compositional_insights', [])
if compositional_insights:
insights.append(f"🧩 Cross-Expert Reasoning: {len(compositional_insights)} high-order insights generated")

# Highlight highest confidence insight
best_insight = max(compositional_insights, key=lambda x: x.confidence)
insights.append(f" Best Insight: {best_insight.composition_type} (confidence: {best_insight.confidence:.3f})")

# Quality assessment insights
quality_assessment = cross_expert_results.get('quality_assessment', {})
if quality_assessment:
overall_quality = quality_assessment.get('overall_quality', 0.0)
insights.append(f" Overall Quality: {overall_quality:.3f}")

quality_scores = quality_assessment.get('quality_scores', {})
if quality_scores:
insights.append(f" Quality Breakdown: Novelty={quality_scores.get('novelty', 0):.3f}, "
f"Coherence={quality_scores.get('coherence', 0):.3f}, "
f"Predictive={quality_scores.get('predictive_gain', 0):.3f}")

# Memory insights
memory_stats = cross_expert_results.get('memory_stats', {})
if memory_stats:
total_insights = memory_stats.get('total_insights', 0)
avg_quality = memory_stats.get('avg_quality', 0.0)
insights.append(f"🧠 Memory: {total_insights} insights stored (avg quality: {avg_quality:.3f})")

return insights

def get_expert_performance_analysis(self) -> Dict[str, Any]:
"""Analyze performance of individual experts"""

performance_analysis = {
'expert_stats': {},
'relation_type_stats': defaultdict(list),
'confidence_distribution': [],
'insight_generation_stats': defaultdict(int)
}

# Analyze recent expert outputs
for output in list(self.expert_outputs_cache)[-100:]: # Last 100 outputs
expert_id = output.expert_id
expert_name = output.expert_name

if expert_id not in performance_analysis['expert_stats']:
performance_analysis['expert_stats'][expert_id] = {
'name': expert_name,
'confidence_scores': [],
'insight_counts': [],
'relation_types': set()
}

stats = performance_analysis['expert_stats'][expert_id]
stats['confidence_scores'].append(output.confidence)
stats['insight_counts'].append(len(output.insights))
stats['relation_types'].update(output.relation_types)

# Relation type analysis
for rel_type in output.relation_types:
performance_analysis['relation_type_stats'][rel_type].append(output.confidence)

# Confidence distribution
performance_analysis['confidence_distribution'].append(output.confidence)

# Insight generation stats
performance_analysis['insight_generation_stats'][expert_name] += len(output.insights)

# Convert sets to lists for JSON serialization
for expert_id, stats in performance_analysis['expert_stats'].items():
stats['relation_types'] = list(stats['relation_types'])
stats['avg_confidence'] = np.mean(stats['confidence_scores']) if stats['confidence_scores'] else 0.0
stats['avg_insights'] = np.mean(stats['insight_counts']) if stats['insight_counts'] else 0.0

# Convert defaultdict to regular dict
performance_analysis['relation_type_stats'] = dict(performance_analysis['relation_type_stats'])
performance_analysis['insight_generation_stats'] = dict(performance_analysis['insight_generation_stats'])

return performance_analysis

def get_cross_expert_evolution(self) -> Dict[str, Any]:
"""Analyze evolution of cross-expert reasoning over time"""

evolution_analysis = {
'insight_evolution': {},
'quality_trends': [],
'composition_patterns': defaultdict(int),
'expert_collaboration_matrix': defaultdict(lambda: defaultdict(int))
}

# Analyze recent integration history
for integration_result in list(self.integration_history)[-50:]: # Last 50 integrations
timestamp = integration_result['timestamp']
cross_expert_results = integration_result['cross_expert_results']

# Quality trends
quality_assessment = cross_expert_results.get('quality_assessment', {})
overall_quality = quality_assessment.get('overall_quality', 0.0)
evolution_analysis['quality_trends'].append({
'timestamp': timestamp,
'quality': overall_quality
})

# Composition patterns
compositional_insights = cross_expert_results.get('compositional_insights', [])
for insight in compositional_insights:
evolution_analysis['composition_patterns'][insight.composition_type] += 1

# Expert collaboration matrix
experts = insight.participating_experts
for i, expert1 in enumerate(experts):
for expert2 in experts[i+1:]:
evolution_analysis['expert_collaboration_matrix'][expert1][expert2] += 1

# Convert defaultdict to regular dict
evolution_analysis['composition_patterns'] = dict(evolution_analysis['composition_patterns'])
evolution_analysis['expert_collaboration_matrix'] = {
str(k): dict(v) for k, v in evolution_analysis['expert_collaboration_matrix'].items()
}

return evolution_analysis

def get_system_diagnostics(self) -> Dict[str, Any]:
"""Get comprehensive system diagnostics"""

diagnostics = {
'expert_registry': {
'total_experts': len(self.expert_registry),
'expert_list': [{'id': eid, 'name': expert.name, 'relation_types': expert.relation_types} 
for eid, expert in self.expert_registry.items()]
},
'cross_expert_system': self.cross_expert_system.get_system_status(),
'integration_stats': {
'total_integrations': len(self.integration_history),
'expert_outputs_cached': len(self.expert_outputs_cache),
'avg_integration_time': self._compute_avg_integration_time()
},
'performance_analysis': self.get_expert_performance_analysis(),
'evolution_analysis': self.get_cross_expert_evolution()
}

return diagnostics

def _compute_avg_integration_time(self) -> float:
"""Compute average integration time"""
if not self.integration_history:
return 0.0

# Simple approximation - in real implementation, you'd track actual timing
return 0.5 # Placeholder

def export_insights(self, format: str = 'json') -> str:
"""Export insights in specified format"""

if format == 'json':
export_data = {
'system_diagnostics': self.get_system_diagnostics(),
'recent_insights': [
{
'composition_type': insight.composition_type,
'insight_text': insight.insight_text,
'confidence': insight.confidence,
'participating_experts': insight.participating_experts,
'timestamp': insight.timestamp
}
for insight in list(self.cross_expert_system.insight_history)[-20:] # Last 20 insights
],
'high_order_insights': self.cross_expert_system.get_high_order_insights()
}

return json.dumps(export_data, indent=2, default=str)

else:
return "Unsupported format. Use 'json'."


class ExpertCompositionVisualizer:
"""
Visualization engine for expert compositions
Represents experts as dynamic 3-D graph (nodes = experts, edges = influence weights)
"""

def __init__(self):
self.logger = logging.getLogger(__name__)

def generate_composition_graph(self, expert_outputs: List[ExpertOutput], 
cross_expert_results: Dict[str, Any]) -> Dict[str, Any]:
"""Generate graph representation of expert compositions"""

graph_data = {
'nodes': [],
'edges': [],
'compositions': [],
'metadata': {
'total_experts': len(expert_outputs),
'total_compositions': len(cross_expert_results.get('compositional_insights', [])),
'graph_connectivity': cross_expert_results.get('graph_results', {}).get('graph_connectivity', {})
}
}

# Create nodes (experts)
for output in expert_outputs:
node = {
'id': output.expert_id,
'name': output.expert_name,
'relation_types': output.relation_types,
'confidence': output.confidence,
'insight_count': len(output.insights),
'color': self._get_relation_type_color(output.relation_types),
'size': max(10, min(50, output.confidence * 50)) # Size based on confidence
}
graph_data['nodes'].append(node)

# Create edges (expert collaborations)
dependency_matrix = cross_expert_results.get('graph_results', {}).get('dependency_matrix')
if dependency_matrix is not None:
for i, output1 in enumerate(expert_outputs):
for j, output2 in enumerate(expert_outputs):
if i != j and dependency_matrix[i, j] > 0.3: # Threshold for edge
edge = {
'source': output1.expert_id,
'target': output2.expert_id,
'weight': float(dependency_matrix[i, j]),
'type': 'collaboration'
}
graph_data['edges'].append(edge)

# Create compositions
compositional_insights = cross_expert_results.get('compositional_insights', [])
for insight in compositional_insights:
composition = {
'type': insight.composition_type,
'participating_experts': insight.participating_experts,
'confidence': insight.confidence,
'insight_preview': insight.insight_text[:100] + "..." if len(insight.insight_text) > 100 else insight.insight_text
}
graph_data['compositions'].append(composition)

return graph_data

def _get_relation_type_color(self, relation_types: List[str]) -> str:
"""Get color for relation type"""
color_map = {
'structural': '#FF6B6B', # Red
'statistical': '#4ECDC4', # Teal
'temporal': '#45B7D1', # Blue
'relational': '#96CEB4', # Green
'causal': '#FFEAA7', # Yellow
'semantic': '#DDA0DD', # Plum
'cognitive': '#98D8C8', # Mint
'integrative': '#F7DC6F', # Light Yellow
'predictive': '#BB8FCE', # Light Purple
'ethical': '#85C1E9' # Light Blue
}

# Use first relation type for color
if relation_types:
return color_map.get(relation_types[0], '#CCCCCC') # Default gray
return '#CCCCCC'

def generate_insight_timeline(self, integration_history: deque) -> Dict[str, Any]:
"""Generate timeline of insights over time"""

timeline_data = {
'timeline': [],
'quality_trends': [],
'composition_evolution': defaultdict(list)
}

for integration_result in integration_history:
timestamp = integration_result['timestamp']
cross_expert_results = integration_result['cross_expert_results']

# Quality trends
quality_assessment = cross_expert_results.get('quality_assessment', {})
overall_quality = quality_assessment.get('overall_quality', 0.0)
timeline_data['quality_trends'].append({
'timestamp': timestamp,
'quality': overall_quality
})

# Composition evolution
compositional_insights = cross_expert_results.get('compositional_insights', [])
for insight in compositional_insights:
timeline_data['composition_evolution'][insight.composition_type].append({
'timestamp': timestamp,
'confidence': insight.confidence
})

# Convert defaultdict to regular dict
timeline_data['composition_evolution'] = dict(timeline_data['composition_evolution'])

return timeline_data


# Example usage and integration
def create_integrated_expert_system(config: Dict[str, Any] = None) -> ExpertIntegrationManager:
"""Create an integrated expert system with cross-expert reasoning"""

# Initialize integration manager
integration_manager = ExpertIntegrationManager(config)

# Register all experts (this would be done with actual expert instances)
# For now, we'll create a placeholder that shows the structure

return integration_manager


def demonstrate_cross_expert_reasoning():
"""Demonstrate the cross-expert reasoning capabilities"""

# Create integration manager
integration_manager = ExpertIntegrationManager()

# Simulate expert outputs (in real usage, these would come from actual experts)
sample_data = np.random.randn(100, 10) # 100 samples, 10 features

# Process data through experts
result = integration_manager.process_data_through_experts(sample_data)

# Get high-order insights
high_order_insights = integration_manager.cross_expert_system.get_high_order_insights()

# Get system diagnostics
diagnostics = integration_manager.get_system_diagnostics()

return {
'integration_result': result,
'high_order_insights': high_order_insights,
'diagnostics': diagnostics
}
