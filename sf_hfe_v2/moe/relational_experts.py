"""
Relational/Interactional Experts - Entity Behavior Layer
Implements 4 core experts for understanding entity interactions, dependencies, and feedback loops
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict, deque
import networkx as nx
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import time

from .base_expert import BaseExpert


class GraphBuilderExpert(BaseExpert):
"""
Expert 13: Graph Builder Expert
Constructs relation graphs from co-occurrence or reference patterns
"""

def __init__(self, expert_id: int = 13, config: Dict[str, Any] = None):
super().__init__(
expert_id=expert_id,
name="GraphBuilderExpert",
relation_types=["relational", "interactional"],
config=config
)

self.preferred_domains = ["network", "graph", "relational", "social"]
self.preferred_data_types = ["mixed", "object", "float64"]
self.preferred_tasks = ["graph_construction", "network_analysis", "relationship_mapping"]

# Graph building parameters
self.edge_threshold = self.config.get('edge_threshold', 0.3)
self.graph_methods = self.config.get('graph_methods', ['correlation', 'co_occurrence', 'distance'])
self.max_nodes = self.config.get('max_nodes', 100)

# Pattern storage
self.graph_patterns = defaultdict(list)
self.network_patterns = defaultdict(list)

def _initialize_expert(self):
"""Initialize graph building components"""
self.correlation_graph_builder = CorrelationGraphBuilder()
self.cooccurrence_graph_builder = CooccurrenceGraphBuilder()
self.distance_graph_builder = DistanceGraphBuilder()
self.graph_analyzer = GraphAnalyzer()

def process_data(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
"""Process data to construct relation graphs"""
metadata = metadata or {}

# Build correlation graph
correlation_graph = self.correlation_graph_builder.build_graph(data)

# Build co-occurrence graph
cooccurrence_graph = self.cooccurrence_graph_builder.build_graph(data)

# Build distance graph
distance_graph = self.distance_graph_builder.build_graph(data)

# Analyze overall graph patterns
graph_analysis = self.graph_analyzer.analyze_graphs(data, correlation_graph, cooccurrence_graph, distance_graph)

# Compute confidence
confidence = self._compute_graph_confidence(correlation_graph, cooccurrence_graph, distance_graph, graph_analysis)

result = {
'expert_id': self.expert_id,
'expert_name': self.name,
'correlation_graph': correlation_graph,
'cooccurrence_graph': cooccurrence_graph,
'distance_graph': distance_graph,
'graph_analysis': graph_analysis,
'confidence': confidence,
'relation_types': self.relation_types,
'timestamp': time.time()
}

return result

def update_online(self, data: np.ndarray, feedback: Dict[str, Any] = None) -> bool:
"""Update graph patterns based on new data"""
try:
# Extract graph features
graph_features = self.graph_analyzer.extract_features(data)

# Update patterns
self.graph_patterns['recent'].append(graph_features)

# Adapt thresholds based on feedback
if feedback and 'graph_quality' in feedback:
quality = feedback['graph_quality']
if quality > 0.8:
self.edge_threshold *= 1.01
elif quality < 0.6:
self.edge_threshold *= 0.99

self.edge_threshold = max(0.1, min(0.8, self.edge_threshold))

# Store in memory
self.store_memory(data, {'graph_features': graph_features}, feedback)

return True

except Exception as e:
self.logger.error(f"Error updating GraphBuilderExpert: {e}")
return False

def _compute_graph_confidence(self, correlation_graph: Dict, cooccurrence_graph: Dict, 
distance_graph: Dict, graph_analysis: Dict) -> float:
"""Compute confidence in graph construction"""
confidence = 0.7 # Base confidence

# Graph connectivity
connectivity = graph_analysis.get('connectivity_score', 0)
confidence += connectivity * 0.15

# Graph structure quality
structure_quality = graph_analysis.get('structure_quality', 0)
confidence += structure_quality * 0.15

return min(1.0, confidence)


class InfluenceExpert(BaseExpert):
"""
Expert 14: Influence Expert
Detects directional influence between features/entities
"""

def __init__(self, expert_id: int = 14, config: Dict[str, Any] = None):
super().__init__(
expert_id=expert_id,
name="InfluenceExpert",
relation_types=["relational", "causal"],
config=config
)

self.preferred_domains = ["influence", "causal", "directional", "hierarchical"]
self.preferred_data_types = ["float64", "int64", "mixed"]
self.preferred_tasks = ["influence_detection", "directional_analysis", "hierarchy_detection"]

# Influence detection parameters
self.influence_threshold = self.config.get('influence_threshold', 0.2)
self.influence_methods = self.config.get('influence_methods', ['regression', 'information_theory', 'causality'])

# Pattern storage
self.influence_patterns = defaultdict(list)
self.hierarchy_patterns = defaultdict(list)

def _initialize_expert(self):
"""Initialize influence detection components"""
self.regression_influence = RegressionInfluence()
self.information_influence = InformationInfluence()
self.causality_influence = CausalityInfluence()
self.influence_analyzer = InfluenceAnalyzer()

def process_data(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
"""Process data to detect directional influence"""
metadata = metadata or {}

# Detect regression-based influence
regression_influence = self.regression_influence.detect_influence(data)

# Detect information-theoretic influence
information_influence = self.information_influence.detect_influence(data)

# Detect causality-based influence
causality_influence = self.causality_influence.detect_influence(data)

# Analyze overall influence patterns
influence_analysis = self.influence_analyzer.analyze_influence(data, regression_influence, information_influence, causality_influence)

# Compute confidence
confidence = self._compute_influence_confidence(regression_influence, information_influence, causality_influence, influence_analysis)

result = {
'expert_id': self.expert_id,
'expert_name': self.name,
'regression_influence': regression_influence,
'information_influence': information_influence,
'causality_influence': causality_influence,
'influence_analysis': influence_analysis,
'confidence': confidence,
'relation_types': self.relation_types,
'timestamp': time.time()
}

return result

def update_online(self, data: np.ndarray, feedback: Dict[str, Any] = None) -> bool:
"""Update influence patterns based on new data"""
try:
# Extract influence features
influence_features = self.influence_analyzer.extract_features(data)

# Update patterns
self.influence_patterns['recent'].append(influence_features)

# Adapt thresholds based on feedback
if feedback and 'influence_accuracy' in feedback:
accuracy = feedback['influence_accuracy']
if accuracy > 0.8:
self.influence_threshold *= 1.01
elif accuracy < 0.6:
self.influence_threshold *= 0.99

self.influence_threshold = max(0.1, min(0.5, self.influence_threshold))

# Store in memory
self.store_memory(data, {'influence_features': influence_features}, feedback)

return True

except Exception as e:
self.logger.error(f"Error updating InfluenceExpert: {e}")
return False

def _compute_influence_confidence(self, regression_influence: Dict, information_influence: Dict, 
causality_influence: Dict, influence_analysis: Dict) -> float:
"""Compute confidence in influence detection"""
confidence = 0.6 # Base confidence

# Influence strength
influence_strength = influence_analysis.get('strength_score', 0)
confidence += influence_strength * 0.2

# Method consistency
method_consistency = influence_analysis.get('consistency_score', 0)
confidence += method_consistency * 0.2

return min(1.0, confidence)


class GroupDynamicsExpert(BaseExpert):
"""
Expert 15: Group Dynamics Expert
Clusters agents or features by behavioral similarity
"""

def __init__(self, expert_id: int = 15, config: Dict[str, Any] = None):
super().__init__(
expert_id=expert_id,
name="GroupDynamicsExpert",
relation_types=["relational", "statistical"],
config=config
)

self.preferred_domains = ["clustering", "group_behavior", "social", "behavioral"]
self.preferred_data_types = ["float64", "int64", "mixed"]
self.preferred_tasks = ["group_detection", "behavioral_clustering", "similarity_analysis"]

# Group dynamics parameters
self.cluster_range = self.config.get('cluster_range', (2, 10))
self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
self.clustering_methods = self.config.get('clustering_methods', ['kmeans', 'spectral', 'dbscan'])

# Pattern storage
self.group_patterns = defaultdict(list)
self.behavioral_patterns = defaultdict(list)

def _initialize_expert(self):
"""Initialize group dynamics components"""
self.kmeans_clusterer = KMeansClusterer()
self.spectral_clusterer = SpectralClusterer()
self.dbscan_clusterer = DBSCANClusterer()
self.group_analyzer = GroupAnalyzer()

def process_data(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
"""Process data to detect group dynamics and behavioral clusters"""
metadata = metadata or {}

# Detect KMeans clusters
kmeans_clusters = self.kmeans_clusterer.detect_clusters(data)

# Detect spectral clusters
spectral_clusters = self.spectral_clusterer.detect_clusters(data)

# Detect DBSCAN clusters
dbscan_clusters = self.dbscan_clusterer.detect_clusters(data)

# Analyze overall group dynamics
group_analysis = self.group_analyzer.analyze_groups(data, kmeans_clusters, spectral_clusters, dbscan_clusters)

# Compute confidence
confidence = self._compute_group_confidence(kmeans_clusters, spectral_clusters, dbscan_clusters, group_analysis)

result = {
'expert_id': self.expert_id,
'expert_name': self.name,
'kmeans_clusters': kmeans_clusters,
'spectral_clusters': spectral_clusters,
'dbscan_clusters': dbscan_clusters,
'group_analysis': group_analysis,
'confidence': confidence,
'relation_types': self.relation_types,
'timestamp': time.time()
}

return result

def update_online(self, data: np.ndarray, feedback: Dict[str, Any] = None) -> bool:
"""Update group patterns based on new data"""
try:
# Extract group features
group_features = self.group_analyzer.extract_features(data)

# Update patterns
self.group_patterns['recent'].append(group_features)

# Adapt thresholds based on feedback
if feedback and 'cluster_quality' in feedback:
quality = feedback['cluster_quality']
if quality > 0.8:
self.similarity_threshold *= 1.01
elif quality < 0.6:
self.similarity_threshold *= 0.99

self.similarity_threshold = max(0.5, min(0.9, self.similarity_threshold))

# Store in memory
self.store_memory(data, {'group_features': group_features}, feedback)

return True

except Exception as e:
self.logger.error(f"Error updating GroupDynamicsExpert: {e}")
return False

def _compute_group_confidence(self, kmeans_clusters: Dict, spectral_clusters: Dict, 
dbscan_clusters: Dict, group_analysis: Dict) -> float:
"""Compute confidence in group detection"""
confidence = 0.7 # Base confidence

# Cluster quality
cluster_quality = group_analysis.get('quality_score', 0)
confidence += cluster_quality * 0.2

# Group stability
group_stability = group_analysis.get('stability_score', 0)
confidence += group_stability * 0.1

return min(1.0, confidence)


class FeedbackLoopExpert(BaseExpert):
"""
Expert 16: Feedback Loop Expert
Detects mutual causation and reinforcement loops
"""

def __init__(self, expert_id: int = 16, config: Dict[str, Any] = None):
super().__init__(
expert_id=expert_id,
name="FeedbackLoopExpert",
relation_types=["relational", "causal"],
config=config
)

self.preferred_domains = ["feedback", "loops", "reinforcement", "causal"]
self.preferred_data_types = ["float64", "int64", "mixed"]
self.preferred_tasks = ["feedback_detection", "loop_analysis", "reinforcement_detection"]

# Feedback loop parameters
self.loop_threshold = self.config.get('loop_threshold', 0.3)
self.max_loop_length = self.config.get('max_loop_length', 5)
self.feedback_methods = self.config.get('feedback_methods', ['correlation', 'causality', 'temporal'])

# Pattern storage
self.feedback_patterns = defaultdict(list)
self.loop_patterns = defaultdict(list)

def _initialize_expert(self):
"""Initialize feedback loop detection components"""
self.correlation_loops = CorrelationLoops()
self.causality_loops = CausalityLoops()
self.temporal_loops = TemporalLoops()
self.feedback_analyzer = FeedbackAnalyzer()

def process_data(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
"""Process data to detect feedback loops and reinforcement patterns"""
metadata = metadata or {}

# Detect correlation-based loops
correlation_loops = self.correlation_loops.detect_loops(data)

# Detect causality-based loops
causality_loops = self.causality_loops.detect_loops(data)

# Detect temporal loops
temporal_loops = self.temporal_loops.detect_loops(data)

# Analyze overall feedback patterns
feedback_analysis = self.feedback_analyzer.analyze_feedback(data, correlation_loops, causality_loops, temporal_loops)

# Compute confidence
confidence = self._compute_feedback_confidence(correlation_loops, causality_loops, temporal_loops, feedback_analysis)

result = {
'expert_id': self.expert_id,
'expert_name': self.name,
'correlation_loops': correlation_loops,
'causality_loops': causality_loops,
'temporal_loops': temporal_loops,
'feedback_analysis': feedback_analysis,
'confidence': confidence,
'relation_types': self.relation_types,
'timestamp': time.time()
}

return result

def update_online(self, data: np.ndarray, feedback: Dict[str, Any] = None) -> bool:
"""Update feedback patterns based on new data"""
try:
# Extract feedback features
feedback_features = self.feedback_analyzer.extract_features(data)

# Update patterns
self.feedback_patterns['recent'].append(feedback_features)

# Adapt thresholds based on feedback
if feedback and 'loop_accuracy' in feedback:
accuracy = feedback['loop_accuracy']
if accuracy > 0.8:
self.loop_threshold *= 1.01
elif accuracy < 0.6:
self.loop_threshold *= 0.99

self.loop_threshold = max(0.1, min(0.6, self.loop_threshold))

# Store in memory
self.store_memory(data, {'feedback_features': feedback_features}, feedback)

return True

except Exception as e:
self.logger.error(f"Error updating FeedbackLoopExpert: {e}")
return False

def _compute_feedback_confidence(self, correlation_loops: Dict, causality_loops: Dict, 
temporal_loops: Dict, feedback_analysis: Dict) -> float:
"""Compute confidence in feedback detection"""
confidence = 0.6 # Base confidence

# Loop strength
loop_strength = feedback_analysis.get('strength_score', 0)
confidence += loop_strength * 0.2

# Loop consistency
loop_consistency = feedback_analysis.get('consistency_score', 0)
confidence += loop_consistency * 0.2

return min(1.0, confidence)


# Helper classes for relational experts

class CorrelationGraphBuilder:
"""Builds graphs based on correlations"""

def build_graph(self, data: np.ndarray) -> Dict[str, Any]:
"""Build correlation-based graph"""
graph_info = {
'nodes': [],
'edges': [],
'graph_properties': {},
'connectivity_score': 0.0
}

if data.ndim > 1 and data.shape[1] > 1:
try:
# Compute correlation matrix
corr_matrix = np.corrcoef(data.T)

# Create nodes
nodes = list(range(data.shape[1]))
graph_info['nodes'] = nodes

# Create edges based on correlation threshold
edges = []
for i in range(data.shape[1]):
for j in range(i+1, data.shape[1]):
corr_val = abs(corr_matrix[i, j])
if corr_val > 0.3: # Threshold
edges.append({
'source': i,
'target': j,
'weight': corr_val,
'type': 'correlation'
})

graph_info['edges'] = edges

# Compute graph properties
if edges:
graph_info['connectivity_score'] = len(edges) / (data.shape[1] * (data.shape[1] - 1) / 2)
graph_info['graph_properties'] = {
'num_nodes': len(nodes),
'num_edges': len(edges),
'density': len(edges) / (len(nodes) * (len(nodes) - 1) / 2)
}

except:
pass

return graph_info


class CooccurrenceGraphBuilder:
"""Builds graphs based on co-occurrence patterns"""

def build_graph(self, data: np.ndarray) -> Dict[str, Any]:
"""Build co-occurrence-based graph"""
graph_info = {
'nodes': [],
'edges': [],
'graph_properties': {},
'connectivity_score': 0.0
}

if data.ndim > 1 and data.shape[1] > 1:
try:
# For categorical data, compute co-occurrence
# For numerical data, use discretized co-occurrence
nodes = list(range(data.shape[1]))
graph_info['nodes'] = nodes

edges = []
for i in range(data.shape[1]):
for j in range(i+1, data.shape[1]):
# Compute co-occurrence strength
cooccurrence_strength = self._compute_cooccurrence(data[:, i], data[:, j])

if cooccurrence_strength > 0.3: # Threshold
edges.append({
'source': i,
'target': j,
'weight': cooccurrence_strength,
'type': 'cooccurrence'
})

graph_info['edges'] = edges

if edges:
graph_info['connectivity_score'] = len(edges) / (data.shape[1] * (data.shape[1] - 1) / 2)
graph_info['graph_properties'] = {
'num_nodes': len(nodes),
'num_edges': len(edges),
'density': len(edges) / (len(nodes) * (len(nodes) - 1) / 2)
}

except:
pass

return graph_info

def _compute_cooccurrence(self, col1: np.ndarray, col2: np.ndarray) -> float:
"""Compute co-occurrence strength between two columns"""
try:
# Discretize if numerical
if np.issubdtype(col1.dtype, np.number):
col1_disc = np.digitize(col1, np.linspace(np.min(col1), np.max(col1), 5))
else:
col1_disc = col1

if np.issubdtype(col2.dtype, np.number):
col2_disc = np.digitize(col2, np.linspace(np.min(col2), np.max(col2), 5))
else:
col2_disc = col2

# Compute mutual information as co-occurrence measure
from sklearn.metrics import mutual_info_score
return mutual_info_score(col1_disc, col2_disc)
except:
return 0.0


class DistanceGraphBuilder:
"""Builds graphs based on distance metrics"""

def build_graph(self, data: np.ndarray) -> Dict[str, Any]:
"""Build distance-based graph"""
graph_info = {
'nodes': [],
'edges': [],
'graph_properties': {},
'connectivity_score': 0.0
}

if data.ndim > 1 and data.shape[1] > 1:
try:
# Compute distance matrix
distances = pairwise_distances(data.T)

# Create nodes
nodes = list(range(data.shape[1]))
graph_info['nodes'] = nodes

# Create edges based on distance threshold
edges = []
max_distance = np.max(distances)
threshold = max_distance * 0.3 # Threshold

for i in range(data.shape[1]):
for j in range(i+1, data.shape[1]):
distance = distances[i, j]
if distance < threshold:
edges.append({
'source': i,
'target': j,
'weight': 1.0 - (distance / max_distance), # Convert to similarity
'type': 'distance'
})

graph_info['edges'] = edges

if edges:
graph_info['connectivity_score'] = len(edges) / (data.shape[1] * (data.shape[1] - 1) / 2)
graph_info['graph_properties'] = {
'num_nodes': len(nodes),
'num_edges': len(edges),
'density': len(edges) / (len(nodes) * (len(nodes) - 1) / 2)
}

except:
pass

return graph_info


class GraphAnalyzer:
"""Analyzes overall graph patterns"""

def analyze_graphs(self, data: np.ndarray, correlation_graph: Dict, cooccurrence_graph: Dict, distance_graph: Dict) -> Dict[str, Any]:
"""Analyze overall graph patterns"""
analysis = {
'connectivity_score': 0.0,
'structure_quality': 0.0,
'graph_types': [],
'overall_density': 0.0
}

# Collect connectivity scores
connectivity_scores = []
if correlation_graph.get('connectivity_score', 0) > 0:
connectivity_scores.append(correlation_graph['connectivity_score'])
analysis['graph_types'].append('correlation')

if cooccurrence_graph.get('connectivity_score', 0) > 0:
connectivity_scores.append(cooccurrence_graph['connectivity_score'])
analysis['graph_types'].append('cooccurrence')

if distance_graph.get('connectivity_score', 0) > 0:
connectivity_scores.append(distance_graph['connectivity_score'])
analysis['graph_types'].append('distance')

if connectivity_scores:
analysis['connectivity_score'] = np.mean(connectivity_scores)
analysis['overall_density'] = analysis['connectivity_score']
analysis['structure_quality'] = len(connectivity_scores) / 3.0

return analysis

def extract_features(self, data: np.ndarray) -> Dict[str, Any]:
"""Extract graph features"""
if data.ndim > 1 and data.shape[1] > 1:
try:
# Compute basic graph features
corr_matrix = np.corrcoef(data.T)
upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

return {
'max_correlation': float(np.max(np.abs(upper_tri))),
'mean_correlation': float(np.mean(np.abs(upper_tri))),
'correlation_variance': float(np.var(upper_tri)),
'strong_connections': int(np.sum(np.abs(upper_tri) > 0.5))
}
except:
return {'max_correlation': 0.0, 'mean_correlation': 0.0, 'correlation_variance': 0.0, 'strong_connections': 0}
else:
return {'max_correlation': 0.0, 'mean_correlation': 0.0, 'correlation_variance': 0.0, 'strong_connections': 0}


class RegressionInfluence:
"""Detects influence using regression analysis"""

def detect_influence(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect influence using regression"""
influence_info = {
'influence_pairs': [],
'influence_matrix': None,
'strength_score': 0.0
}

if data.ndim > 1 and data.shape[1] > 1:
try:
from sklearn.linear_model import LinearRegression

influence_matrix = np.zeros((data.shape[1], data.shape[1]))
influence_pairs = []

for i in range(data.shape[1]):
for j in range(data.shape[1]):
if i != j:
# Test if variable j influences variable i
X = data[:, j].reshape(-1, 1)
y = data[:, i]

model = LinearRegression()
model.fit(X, y)

# Use R-squared as influence measure
y_pred = model.predict(X)
r_squared = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

influence_matrix[i, j] = r_squared

if r_squared > 0.1: # Threshold
influence_pairs.append({
'influencer': j,
'influenced': i,
'strength': r_squared
})

influence_info['influence_matrix'] = influence_matrix.tolist()
influence_info['influence_pairs'] = influence_pairs
influence_info['strength_score'] = np.mean(influence_matrix) if influence_matrix.size > 0 else 0.0

except:
pass

return influence_info


class InformationInfluence:
"""Detects influence using information theory"""

def detect_influence(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect influence using information theory"""
influence_info = {
'influence_pairs': [],
'influence_matrix': None,
'strength_score': 0.0
}

if data.ndim > 1 and data.shape[1] > 1:
try:
from sklearn.metrics import mutual_info_score

influence_matrix = np.zeros((data.shape[1], data.shape[1]))
influence_pairs = []

for i in range(data.shape[1]):
for j in range(data.shape[1]):
if i != j:
# Discretize data
data_i = self._discretize(data[:, i])
data_j = self._discretize(data[:, j])

# Compute mutual information
mi_value = mutual_info_score(data_i, data_j)
influence_matrix[i, j] = mi_value

if mi_value > 0.1: # Threshold
influence_pairs.append({
'influencer': j,
'influenced': i,
'strength': mi_value
})

influence_info['influence_matrix'] = influence_matrix.tolist()
influence_info['influence_pairs'] = influence_pairs
influence_info['strength_score'] = np.mean(influence_matrix) if influence_matrix.size > 0 else 0.0

except:
pass

return influence_info

def _discretize(self, data: np.ndarray, bins: int = 10) -> np.ndarray:
"""Discretize continuous data"""
try:
_, bin_edges = np.histogram(data, bins=bins)
return np.digitize(data, bin_edges[1:-1])
except:
return np.zeros_like(data, dtype=int)


class CausalityInfluence:
"""Detects influence using causality measures"""

def detect_influence(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect influence using causality"""
influence_info = {
'influence_pairs': [],
'influence_matrix': None,
'strength_score': 0.0
}

if data.ndim > 1 and data.shape[1] > 1 and data.shape[0] > 10:
try:
influence_matrix = np.zeros((data.shape[1], data.shape[1]))
influence_pairs = []

for i in range(data.shape[1]):
for j in range(data.shape[1]):
if i != j:
# Simple causality test using lagged correlation
causality_strength = self._test_causality(data[:, i], data[:, j])
influence_matrix[i, j] = causality_strength

if causality_strength > 0.1: # Threshold
influence_pairs.append({
'influencer': j,
'influenced': i,
'strength': causality_strength
})

influence_info['influence_matrix'] = influence_matrix.tolist()
influence_info['influence_pairs'] = influence_pairs
influence_info['strength_score'] = np.mean(influence_matrix) if influence_matrix.size > 0 else 0.0

except:
pass

return influence_info

def _test_causality(self, y: np.ndarray, x: np.ndarray, max_lag: int = 3) -> float:
"""Test causality between two variables"""
try:
if len(y) <= max_lag:
return 0.0

# Test if x causes y (lagged correlation)
max_corr = 0.0
for lag in range(1, min(max_lag + 1, len(y))):
if len(y) > lag:
corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
max_corr = max(max_corr, abs(corr))

return max_corr
except:
return 0.0


class InfluenceAnalyzer:
"""Analyzes overall influence patterns"""

def analyze_influence(self, data: np.ndarray, regression_influence: Dict, information_influence: Dict, causality_influence: Dict) -> Dict[str, Any]:
"""Analyze overall influence patterns"""
analysis = {
'strength_score': 0.0,
'consistency_score': 0.0,
'influence_methods': [],
'hierarchy_score': 0.0
}

# Collect influence detection results
influence_strengths = []

if regression_influence.get('strength_score', 0) > 0:
influence_strengths.append(regression_influence.get('strength_score', 0))
analysis['influence_methods'].append('regression')

if information_influence.get('strength_score', 0) > 0:
influence_strengths.append(information_influence.get('strength_score', 0))
analysis['influence_methods'].append('information')

if causality_influence.get('strength_score', 0) > 0:
influence_strengths.append(causality_influence.get('strength_score', 0))
analysis['influence_methods'].append('causality')

if influence_strengths:
analysis['strength_score'] = np.mean(influence_strengths)
analysis['consistency_score'] = len(influence_strengths) / 3.0

# Compute hierarchy score (asymmetry in influence)
all_pairs = []
for method in [regression_influence, information_influence, causality_influence]:
pairs = method.get('influence_pairs', [])
all_pairs.extend(pairs)

if all_pairs:
# Count asymmetric relationships
asymmetric_count = 0
for pair in all_pairs:
# Check if reverse relationship exists
reverse_exists = any(
p['influencer'] == pair['influenced'] and p['influenced'] == pair['influencer']
for p in all_pairs
)
if not reverse_exists:
asymmetric_count += 1

analysis['hierarchy_score'] = asymmetric_count / len(all_pairs)

return analysis

def extract_features(self, data: np.ndarray) -> Dict[str, Any]:
"""Extract influence features"""
if data.ndim > 1 and data.shape[1] > 1:
try:
# Compute basic influence features
corr_matrix = np.corrcoef(data.T)

# Compute asymmetry
asymmetry = np.abs(corr_matrix - corr_matrix.T)
asymmetry_score = np.mean(asymmetry)

return {
'mean_correlation': float(np.mean(np.abs(corr_matrix))),
'correlation_asymmetry': float(asymmetry_score),
'max_correlation': float(np.max(np.abs(corr_matrix))),
'influence_diversity': float(len(np.unique(corr_matrix.flatten())))
}
except:
return {'mean_correlation': 0.0, 'correlation_asymmetry': 0.0, 'max_correlation': 0.0, 'influence_diversity': 0.0}
else:
return {'mean_correlation': 0.0, 'correlation_asymmetry': 0.0, 'max_correlation': 0.0, 'influence_diversity': 0.0}


class KMeansClusterer:
"""Detects clusters using KMeans"""

def detect_clusters(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect clusters using KMeans"""
cluster_info = {
'cluster_labels': None,
'cluster_centers': None,
'num_clusters': 0,
'silhouette_score': 0.0,
'inertia': 0.0
}

if data.ndim > 1 and data.shape[1] > 1 and data.shape[0] > 5:
try:
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Determine optimal number of clusters
n_clusters = min(5, data.shape[0] // 2)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(data)

cluster_info['cluster_labels'] = labels.tolist()
cluster_info['cluster_centers'] = kmeans.cluster_centers_.tolist()
cluster_info['num_clusters'] = n_clusters
cluster_info['inertia'] = float(kmeans.inertia_)

# Compute silhouette score
if len(set(labels)) > 1:
silhouette_avg = silhouette_score(data, labels)
cluster_info['silhouette_score'] = silhouette_avg

except:
pass

return cluster_info


class SpectralClusterer:
"""Detects clusters using Spectral Clustering"""

def detect_clusters(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect clusters using Spectral Clustering"""
cluster_info = {
'cluster_labels': None,
'num_clusters': 0,
'silhouette_score': 0.0
}

if data.ndim > 1 and data.shape[1] > 1 and data.shape[0] > 5:
try:
from sklearn.metrics import silhouette_score

# Determine optimal number of clusters
n_clusters = min(5, data.shape[0] // 2)

spectral = SpectralClustering(n_clusters=n_clusters, random_state=42)
labels = spectral.fit_predict(data)

cluster_info['cluster_labels'] = labels.tolist()
cluster_info['num_clusters'] = n_clusters

# Compute silhouette score
if len(set(labels)) > 1:
silhouette_avg = silhouette_score(data, labels)
cluster_info['silhouette_score'] = silhouette_avg

except:
pass

return cluster_info


class DBSCANClusterer:
"""Detects clusters using DBSCAN"""

def detect_clusters(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect clusters using DBSCAN"""
cluster_info = {
'cluster_labels': None,
'num_clusters': 0,
'noise_points': 0,
'silhouette_score': 0.0
}

if data.ndim > 1 and data.shape[1] > 1 and data.shape[0] > 5:
try:
from sklearn.metrics import silhouette_score

# Use DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=3)
labels = dbscan.fit_predict(data)

cluster_info['cluster_labels'] = labels.tolist()
cluster_info['num_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
cluster_info['noise_points'] = int(np.sum(labels == -1))

# Compute silhouette score (excluding noise points)
if cluster_info['num_clusters'] > 1:
non_noise_mask = labels != -1
if np.sum(non_noise_mask) > 1:
silhouette_avg = silhouette_score(data[non_noise_mask], labels[non_noise_mask])
cluster_info['silhouette_score'] = silhouette_avg

except:
pass

return cluster_info


class GroupAnalyzer:
"""Analyzes overall group dynamics"""

def analyze_groups(self, data: np.ndarray, kmeans_clusters: Dict, spectral_clusters: Dict, dbscan_clusters: Dict) -> Dict[str, Any]:
"""Analyze overall group dynamics"""
analysis = {
'quality_score': 0.0,
'stability_score': 0.0,
'cluster_methods': [],
'consensus_clusters': None
}

# Collect clustering results
cluster_results = []

if kmeans_clusters.get('silhouette_score', 0) > 0:
cluster_results.append(kmeans_clusters.get('silhouette_score', 0))
analysis['cluster_methods'].append('kmeans')

if spectral_clusters.get('silhouette_score', 0) > 0:
cluster_results.append(spectral_clusters.get('silhouette_score', 0))
analysis['cluster_methods'].append('spectral')

if dbscan_clusters.get('silhouette_score', 0) > 0:
cluster_results.append(dbscan_clusters.get('silhouette_score', 0))
analysis['cluster_methods'].append('dbscan')

if cluster_results:
analysis['quality_score'] = np.mean(cluster_results)
analysis['stability_score'] = len(cluster_results) / 3.0

return analysis

def extract_features(self, data: np.ndarray) -> Dict[str, Any]:
"""Extract group features"""
if data.ndim > 1 and data.shape[1] > 1:
try:
# Compute basic group features
distances = pairwise_distances(data)

return {
'mean_distance': float(np.mean(distances)),
'distance_variance': float(np.var(distances)),
'max_distance': float(np.max(distances)),
'min_distance': float(np.min(distances[distances > 0]))
}
except:
return {'mean_distance': 0.0, 'distance_variance': 0.0, 'max_distance': 0.0, 'min_distance': 0.0}
else:
return {'mean_distance': 0.0, 'distance_variance': 0.0, 'max_distance': 0.0, 'min_distance': 0.0}


class CorrelationLoops:
"""Detects feedback loops using correlation"""

def detect_loops(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect correlation-based loops"""
loop_info = {
'loops': [],
'loop_count': 0,
'strength_score': 0.0
}

if data.ndim > 1 and data.shape[1] > 2:
try:
# Compute correlation matrix
corr_matrix = np.corrcoef(data.T)

# Find potential loops (cycles of length 3 or more)
loops = []
for i in range(data.shape[1]):
for j in range(data.shape[1]):
if i != j:
for k in range(data.shape[1]):
if k != i and k != j:
# Check for loop: i -> j -> k -> i
corr_ij = abs(corr_matrix[i, j])
corr_jk = abs(corr_matrix[j, k])
corr_ki = abs(corr_matrix[k, i])

if corr_ij > 0.3 and corr_jk > 0.3 and corr_ki > 0.3:
loop_strength = (corr_ij + corr_jk + corr_ki) / 3
loops.append({
'nodes': [i, j, k],
'strength': loop_strength,
'type': 'correlation'
})

loop_info['loops'] = loops
loop_info['loop_count'] = len(loops)
loop_info['strength_score'] = np.mean([loop['strength'] for loop in loops]) if loops else 0.0

except:
pass

return loop_info


class CausalityLoops:
"""Detects feedback loops using causality"""

def detect_loops(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect causality-based loops"""
loop_info = {
'loops': [],
'loop_count': 0,
'strength_score': 0.0
}

if data.ndim > 1 and data.shape[1] > 2 and data.shape[0] > 10:
try:
loops = []

# Find potential causality loops
for i in range(data.shape[1]):
for j in range(data.shape[1]):
if i != j:
for k in range(data.shape[1]):
if k != i and k != j:
# Test causality chain: i -> j -> k -> i
causality_ij = self._test_causality(data[:, i], data[:, j])
causality_jk = self._test_causality(data[:, j], data[:, k])
causality_ki = self._test_causality(data[:, k], data[:, i])

if causality_ij > 0.1 and causality_jk > 0.1 and causality_ki > 0.1:
loop_strength = (causality_ij + causality_jk + causality_ki) / 3
loops.append({
'nodes': [i, j, k],
'strength': loop_strength,
'type': 'causality'
})

loop_info['loops'] = loops
loop_info['loop_count'] = len(loops)
loop_info['strength_score'] = np.mean([loop['strength'] for loop in loops]) if loops else 0.0

except:
pass

return loop_info

def _test_causality(self, y: np.ndarray, x: np.ndarray, max_lag: int = 3) -> float:
"""Test causality between two variables"""
try:
if len(y) <= max_lag:
return 0.0

# Test if x causes y (lagged correlation)
max_corr = 0.0
for lag in range(1, min(max_lag + 1, len(y))):
if len(y) > lag:
corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
max_corr = max(max_corr, abs(corr))

return max_corr
except:
return 0.0


class TemporalLoops:
"""Detects feedback loops using temporal patterns"""

def detect_loops(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect temporal-based loops"""
loop_info = {
'loops': [],
'loop_count': 0,
'strength_score': 0.0
}

if data.ndim > 1 and data.shape[1] > 2 and data.shape[0] > 20:
try:
loops = []

# Find temporal feedback patterns
for i in range(data.shape[1]):
for j in range(data.shape[1]):
if i != j:
# Test for temporal feedback between i and j
temporal_strength = self._test_temporal_feedback(data[:, i], data[:, j])

if temporal_strength > 0.2:
loops.append({
'nodes': [i, j],
'strength': temporal_strength,
'type': 'temporal'
})

loop_info['loops'] = loops
loop_info['loop_count'] = len(loops)
loop_info['strength_score'] = np.mean([loop['strength'] for loop in loops]) if loops else 0.0

except:
pass

return loop_info

def _test_temporal_feedback(self, y: np.ndarray, x: np.ndarray) -> float:
"""Test temporal feedback between two variables"""
try:
if len(y) < 10:
return 0.0

# Test bidirectional temporal influence
# Forward: x -> y
forward_corr = 0.0
for lag in range(1, min(4, len(y))):
if len(y) > lag:
corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
forward_corr = max(forward_corr, abs(corr))

# Backward: y -> x
backward_corr = 0.0
for lag in range(1, min(4, len(x))):
if len(x) > lag:
corr = np.corrcoef(y[:-lag], x[lag:])[0, 1]
backward_corr = max(backward_corr, abs(corr))

# Feedback strength is the minimum of forward and backward
feedback_strength = min(forward_corr, backward_corr)
return feedback_strength
except:
return 0.0


class FeedbackAnalyzer:
"""Analyzes overall feedback patterns"""

def analyze_feedback(self, data: np.ndarray, correlation_loops: Dict, causality_loops: Dict, temporal_loops: Dict) -> Dict[str, Any]:
"""Analyze overall feedback patterns"""
analysis = {
'strength_score': 0.0,
'consistency_score': 0.0,
'feedback_methods': [],
'loop_diversity': 0.0
}

# Collect feedback detection results
feedback_strengths = []

if correlation_loops.get('strength_score', 0) > 0:
feedback_strengths.append(correlation_loops.get('strength_score', 0))
analysis['feedback_methods'].append('correlation')

if causality_loops.get('strength_score', 0) > 0:
feedback_strengths.append(causality_loops.get('strength_score', 0))
analysis['feedback_methods'].append('causality')

if temporal_loops.get('strength_score', 0) > 0:
feedback_strengths.append(temporal_loops.get('strength_score', 0))
analysis['feedback_methods'].append('temporal')

if feedback_strengths:
analysis['strength_score'] = np.mean(feedback_strengths)
analysis['consistency_score'] = len(feedback_strengths) / 3.0

# Compute loop diversity
all_loops = []
for method in [correlation_loops, causality_loops, temporal_loops]:
loops = method.get('loops', [])
all_loops.extend(loops)

if all_loops:
# Count unique loop patterns
unique_patterns = set()
for loop in all_loops:
pattern = tuple(sorted(loop['nodes']))
unique_patterns.add(pattern)

analysis['loop_diversity'] = len(unique_patterns) / len(all_loops)

return analysis

def extract_features(self, data: np.ndarray) -> Dict[str, Any]:
"""Extract feedback features"""
if data.ndim > 1 and data.shape[1] > 1:
try:
# Compute basic feedback features
corr_matrix = np.corrcoef(data.T)

# Compute symmetry (potential feedback indicator)
symmetry = np.abs(corr_matrix - corr_matrix.T)
symmetry_score = np.mean(symmetry)

return {
'correlation_symmetry': float(symmetry_score),
'max_correlation': float(np.max(np.abs(corr_matrix))),
'mean_correlation': float(np.mean(np.abs(corr_matrix))),
'correlation_entropy': float(-np.sum(corr_matrix * np.log(np.abs(corr_matrix) + 1e-10)))
}
except:
return {'correlation_symmetry': 0.0, 'max_correlation': 0.0, 'mean_correlation': 0.0, 'correlation_entropy': 0.0}
else:
return {'correlation_symmetry': 0.0, 'max_correlation': 0.0, 'mean_correlation': 0.0, 'correlation_entropy': 0.0}
