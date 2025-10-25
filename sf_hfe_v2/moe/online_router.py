"""
Online Learning Router for 30-Expert System
Dynamically selects and weights experts based on online learning and performance feedback
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
from collections import defaultdict, deque
import json

from .base_expert import BaseExpert, OnlineLearningRouter


class AdvancedOnlineLearningRouter(OnlineLearningRouter):
"""
Advanced router for the 30-expert online learning system
Implements sophisticated expert selection and weighting strategies
"""

def __init__(self, experts: List[BaseExpert], config: Dict[str, Any] = None):
"""
Initialize advanced online learning router

Args:
experts: List of all 30 experts
config: Configuration dictionary
"""
super().__init__(experts, config)

# Advanced routing parameters
self.adaptation_rate = self.config.get('adaptation_rate', 0.01)
self.exploration_decay = self.config.get('exploration_decay', 0.99)
self.performance_window = self.config.get('performance_window', 50)
self.context_memory_size = self.config.get('context_memory_size', 1000)

# Expert grouping by relation types
self.expert_groups = self._group_experts_by_relation_type()

# Performance tracking
self.expert_performance_history = {expert_id: deque(maxlen=self.performance_window) 
for expert_id in self.experts.keys()}
self.group_performance_history = {group: deque(maxlen=self.performance_window) 
for group in self.expert_groups.keys()}

# Context-based routing
self.context_patterns = defaultdict(list)
self.context_expert_mapping = defaultdict(list)

# Meta-learning components
self.meta_learner = MetaLearner(config=self.config.get('meta_learner', {}))
self.performance_predictor = PerformancePredictor(config=self.config.get('performance_predictor', {}))

# Expert collaboration tracking
self.collaboration_matrix = np.zeros((len(self.experts), len(self.experts)))
self.collaboration_history = deque(maxlen=1000)

self.logger = logging.getLogger("AdvancedOnlineLearningRouter")

def _group_experts_by_relation_type(self) -> Dict[str, List[int]]:
"""Group experts by their relation types"""
groups = defaultdict(list)

for expert_id, expert in self.experts.items():
for relation_type in expert.relation_types:
groups[relation_type].append(expert_id)

return dict(groups)

def route_experts(self, data: np.ndarray, context: Dict[str, Any] = None) -> List[Tuple[int, float]]:
"""
Route data to appropriate experts using advanced online learning

Args:
data: Input data
context: Context information

Returns:
List of (expert_id, weight) tuples for selected experts
"""
context = context or {}

# Extract enhanced context features
context_features = self._extract_enhanced_context_features(data, context)

# Predict expert performance
performance_predictions = self.performance_predictor.predict_performance(
context_features, self.expert_performance_history
)

# Compute expert scores with multiple factors
expert_scores = self._compute_advanced_expert_scores(
context_features, performance_predictions
)

# Apply meta-learning adjustments
meta_adjustments = self.meta_learner.get_adjustments(context_features)
for expert_id, adjustment in meta_adjustments.items():
if expert_id in expert_scores:
expert_scores[expert_id] *= adjustment

# Select experts using advanced strategy
selected_experts = self._select_experts_advanced(expert_scores, context_features)

# Update collaboration tracking
self._update_collaboration_tracking(selected_experts)

# Record routing decision
self.routing_history.append({
'timestamp': time.time(),
'context_features': context_features,
'selected_experts': selected_experts,
'expert_scores': expert_scores,
'performance_predictions': performance_predictions
})

return selected_experts

def _extract_enhanced_context_features(self, data: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
"""Extract enhanced context features for routing decisions"""
features = super()._extract_context_features(data, context)

# Add advanced features
features.update({
'data_complexity': self._compute_data_complexity(data),
'temporal_patterns': self._detect_temporal_patterns(data),
'structural_patterns': self._detect_structural_patterns(data),
'statistical_patterns': self._detect_statistical_patterns(data),
'causal_potential': self._assess_causal_potential(data),
'semantic_richness': self._assess_semantic_richness(data),
'cognitive_load': self._estimate_cognitive_load(data)
})

return features

def _compute_data_complexity(self, data: np.ndarray) -> float:
"""Compute data complexity score"""
if data.size == 0:
return 0.0

complexity = 0.0

# Dimensional complexity
if data.ndim > 1:
complexity += min(1.0, data.shape[1] / 10.0)

# Size complexity
complexity += min(1.0, np.log10(data.size) / 5.0)

# Variance complexity
if data.size > 1:
complexity += min(1.0, np.std(data) / (np.mean(np.abs(data)) + 1e-10))

return complexity / 3.0

def _detect_temporal_patterns(self, data: np.ndarray) -> float:
"""Detect temporal patterns in data"""
if data.size < 10:
return 0.0

try:
# Simple temporal pattern detection
if data.ndim == 1:
# Check for trends
x = np.arange(len(data))
correlation = np.corrcoef(x, data)[0, 1]
return abs(correlation)
else:
# Check for temporal patterns across dimensions
temporal_scores = []
for i in range(min(5, data.shape[1])):
x = np.arange(len(data))
correlation = np.corrcoef(x, data[:, i])[0, 1]
temporal_scores.append(abs(correlation))
return np.mean(temporal_scores)
except:
return 0.0

def _detect_structural_patterns(self, data: np.ndarray) -> float:
"""Detect structural patterns in data"""
if data.size == 0:
return 0.0

structural_score = 0.0

# Check for hierarchical structure
if data.ndim > 1 and data.shape[1] > 1:
# Simple hierarchical detection
corr_matrix = np.corrcoef(data.T)
structural_score += np.mean(np.abs(corr_matrix))

# Check for clustering patterns
if data.size > 10:
try:
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=min(3, data.shape[0]//2), random_state=42)
labels = kmeans.fit_predict(data)
structural_score += len(set(labels)) / data.shape[0]
except:
pass

return min(1.0, structural_score)

def _detect_statistical_patterns(self, data: np.ndarray) -> float:
"""Detect statistical patterns in data"""
if data.size < 5:
return 0.0

statistical_score = 0.0

# Distribution patterns
if np.issubdtype(data.dtype, np.floating):
# Check for normality
from scipy import stats
try:
_, p_value = stats.normaltest(data.flatten())
statistical_score += 1.0 - p_value
except:
pass

# Check for outliers
q1, q3 = np.percentile(data.flatten(), [25, 75])
iqr = q3 - q1
if iqr > 0:
outliers = np.sum((data.flatten() < q1 - 1.5*iqr) | (data.flatten() > q3 + 1.5*iqr))
statistical_score += outliers / data.size

return min(1.0, statistical_score)

def _assess_causal_potential(self, data: np.ndarray) -> float:
"""Assess potential for causal relationships"""
if data.ndim < 2 or data.shape[1] < 2:
return 0.0

causal_potential = 0.0

# Check for correlation patterns
corr_matrix = np.corrcoef(data.T)
strong_correlations = np.sum(np.abs(corr_matrix) > 0.5)
causal_potential += strong_correlations / (data.shape[1] * (data.shape[1] - 1))

# Check for temporal ordering potential
if data.shape[0] > 10:
temporal_correlations = []
for i in range(data.shape[1]):
for j in range(data.shape[1]):
if i != j:
# Check lagged correlation
lagged_corr = np.corrcoef(data[:-1, i], data[1:, j])[0, 1]
temporal_correlations.append(abs(lagged_corr))

if temporal_correlations:
causal_potential += np.mean(temporal_correlations)

return min(1.0, causal_potential)

def _assess_semantic_richness(self, data: np.ndarray) -> float:
"""Assess semantic richness of data"""
semantic_score = 0.0

# Check for categorical data
if data.dtype.kind == 'O':
unique_ratio = len(np.unique(data)) / len(data)
semantic_score += unique_ratio

# Check for text-like patterns
if data.dtype.kind == 'O':
text_indicators = 0
sample_values = data[:min(10, len(data))]
for val in sample_values:
if isinstance(val, str) and len(str(val)) > 3:
text_indicators += 1
semantic_score += text_indicators / len(sample_values)

return min(1.0, semantic_score)

def _estimate_cognitive_load(self, data: np.ndarray) -> float:
"""Estimate cognitive load required for processing"""
cognitive_load = 0.0

# Data size factor
cognitive_load += min(1.0, np.log10(data.size) / 4.0)

# Dimensionality factor
if data.ndim > 1:
cognitive_load += min(1.0, data.shape[1] / 20.0)

# Complexity factor
cognitive_load += self._compute_data_complexity(data)

return min(1.0, cognitive_load / 3.0)

def _compute_advanced_expert_scores(self, context_features: Dict[str, Any], 
performance_predictions: Dict[int, float]) -> Dict[int, float]:
"""Compute advanced expert scores considering multiple factors"""
expert_scores = {}

for expert_id, expert in self.experts.items():
# Base score from expert weights
base_score = self.expert_weights[expert_id]

# Performance-based score
performance_score = performance_predictions.get(expert_id, 0.5)

# Context-based score
context_score = self._compute_enhanced_context_score(expert, context_features)

# Group-based score
group_score = self._compute_group_score(expert, context_features)

# Collaboration score
collaboration_score = self._compute_collaboration_score(expert_id)

# Exploration bonus
exploration_bonus = self._compute_exploration_bonus(expert_id)

# Combined score with weighted factors
expert_scores[expert_id] = (
base_score * 0.25 +
performance_score * 0.25 +
context_score * 0.20 +
group_score * 0.15 +
collaboration_score * 0.10 +
exploration_bonus * 0.05
)

return expert_scores

def _compute_enhanced_context_score(self, expert: BaseExpert, context_features: Dict[str, Any]) -> float:
"""Compute enhanced context score for expert"""
score = 0.0

# Relation type matching
for relation_type in expert.relation_types:
if relation_type in context_features:
score += context_features[relation_type] * 0.3

# Domain matching
if hasattr(expert, 'preferred_domains'):
domain = context_features.get('domain', 'unknown')
if domain in expert.preferred_domains:
score += 0.2

# Data type matching
if hasattr(expert, 'preferred_data_types'):
data_type = context_features.get('data_type', 'unknown')
if data_type in expert.preferred_data_types:
score += 0.2

# Task type matching
if hasattr(expert, 'preferred_tasks'):
task_type = context_features.get('task_type', 'unknown')
if task_type in expert.preferred_tasks:
score += 0.2

# Complexity matching
complexity = context_features.get('data_complexity', 0.5)
if hasattr(expert, 'complexity_preference'):
complexity_match = 1.0 - abs(complexity - expert.complexity_preference)
score += complexity_match * 0.1

return min(1.0, score)

def _compute_group_score(self, expert: BaseExpert, context_features: Dict[str, Any]) -> float:
"""Compute group-based score for expert"""
group_score = 0.0

for relation_type in expert.relation_types:
if relation_type in self.expert_groups:
group_performance = self.group_performance_history[relation_type]
if group_performance:
group_score += np.mean(list(group_performance))

return group_score / len(expert.relation_types) if expert.relation_types else 0.0

def _compute_collaboration_score(self, expert_id: int) -> float:
"""Compute collaboration score for expert"""
if expert_id not in self.experts:
return 0.0

# Get recent collaborations
recent_collaborations = list(self.collaboration_history)[-10:]

collaboration_score = 0.0
for collaboration in recent_collaborations:
if expert_id in collaboration.get('experts', []):
collaboration_score += collaboration.get('success_rate', 0.5)

return collaboration_score / len(recent_collaborations) if recent_collaborations else 0.5

def _select_experts_advanced(self, expert_scores: Dict[int, float], 
context_features: Dict[str, Any]) -> List[Tuple[int, float]]:
"""Select experts using advanced strategy"""
# Sort experts by score
sorted_experts = sorted(expert_scores.items(), key=lambda x: x[1], reverse=True)

# Apply softmax to get weights
scores = np.array([score for _, score in sorted_experts])
weights = F.softmax(torch.tensor(scores), dim=0).numpy()

# Select experts based on multiple criteria
selected_experts = []

# Top performers
top_count = min(5, len(sorted_experts))
for i in range(top_count):
expert_id, _ = sorted_experts[i]
if weights[i] > 0.05: # Threshold
selected_experts.append((expert_id, float(weights[i])))

# Diversity selection - ensure representation from different relation types
relation_type_counts = defaultdict(int)
for expert_id, weight in selected_experts:
expert = self.experts[expert_id]
for relation_type in expert.relation_types:
relation_type_counts[relation_type] += 1

# Add experts from underrepresented relation types
for relation_type, count in relation_type_counts.items():
if count == 0 and relation_type in self.expert_groups:
# Find best expert from this relation type
best_expert = None
best_score = 0.0
for expert_id in self.expert_groups[relation_type]:
if expert_id in expert_scores and expert_scores[expert_id] > best_score:
best_expert = expert_id
best_score = expert_scores[expert_id]

if best_expert and best_score > 0.3:
selected_experts.append((best_expert, float(best_score * 0.5)))

# Exploration - add some random experts occasionally
if np.random.random() < self.exploration_rate:
unexplored_experts = [expert_id for expert_id in self.experts.keys() 
if expert_id not in [e[0] for e in selected_experts]]
if unexplored_experts:
random_expert = np.random.choice(unexplored_experts)
selected_experts.append((random_expert, 0.1))

# Normalize weights
total_weight = sum(weight for _, weight in selected_experts)
if total_weight > 0:
selected_experts = [(expert_id, weight/total_weight) 
for expert_id, weight in selected_experts]

return selected_experts

def _update_collaboration_tracking(self, selected_experts: List[Tuple[int, float]]):
"""Update collaboration tracking"""
if len(selected_experts) > 1:
expert_ids = [expert_id for expert_id, _ in selected_experts]

# Update collaboration matrix
for i, expert_id1 in enumerate(expert_ids):
for j, expert_id2 in enumerate(expert_ids):
if i != j:
idx1 = list(self.experts.keys()).index(expert_id1)
idx2 = list(self.experts.keys()).index(expert_id2)
self.collaboration_matrix[idx1, idx2] += 1

# Record collaboration
self.collaboration_history.append({
'timestamp': time.time(),
'experts': expert_ids,
'success_rate': 0.5 # Will be updated based on feedback
})

def update_routing(self, expert_id: int, performance: float, feedback: Dict[str, Any] = None):
"""Update routing based on expert performance with advanced learning"""
# Update performance history
self.expert_performance_history[expert_id].append(performance)

# Update expert weights using adaptive learning
learning_rate = self.adaptation_rate * (1.0 + performance)
self.expert_weights[expert_id] += learning_rate * performance

# Keep weights in reasonable bounds
self.expert_weights[expert_id] = max(0.0, min(2.0, self.expert_weights[expert_id]))

# Update group performance
expert = self.experts[expert_id]
for relation_type in expert.relation_types:
self.group_performance_history[relation_type].append(performance)

# Update collaboration success rates
if feedback and 'collaboration_success' in feedback:
for collaboration in self.collaboration_history[-5:]: # Recent collaborations
if expert_id in collaboration.get('experts', []):
collaboration['success_rate'] = feedback['collaboration_success']

# Update context patterns
if feedback and 'context' in feedback:
self.context_patterns[expert_id].append(feedback['context'])

# Update meta-learner
self.meta_learner.update(expert_id, performance, feedback)

# Update performance predictor
self.performance_predictor.update(expert_id, performance, feedback)

def get_routing_stats(self) -> Dict[str, Any]:
"""Get comprehensive routing statistics"""
stats = super().get_routing_stats()

# Add advanced statistics
stats.update({
'expert_groups': {group: len(experts) for group, experts in self.expert_groups.items()},
'group_performance': {
group: np.mean(list(perf)) if perf else 0.0
for group, perf in self.group_performance_history.items()
},
'collaboration_matrix_sum': float(np.sum(self.collaboration_matrix)),
'context_patterns_count': len(self.context_patterns),
'meta_learner_stats': self.meta_learner.get_stats(),
'performance_predictor_stats': self.performance_predictor.get_stats()
})

return stats


class MetaLearner:
"""Meta-learning component for expert selection"""

def __init__(self, config: Dict[str, Any] = None):
self.config = config or {}
self.learning_rate = self.config.get('learning_rate', 0.01)
self.memory_size = self.config.get('memory_size', 1000)

self.pattern_memory = deque(maxlen=self.memory_size)
self.expert_patterns = defaultdict(list)

def get_adjustments(self, context_features: Dict[str, Any]) -> Dict[int, float]:
"""Get meta-learning adjustments for experts"""
adjustments = {}

# Find similar patterns in memory
similar_patterns = self._find_similar_patterns(context_features)

if similar_patterns:
# Compute adjustments based on historical performance
for pattern in similar_patterns:
expert_id = pattern['expert_id']
performance = pattern['performance']

if expert_id not in adjustments:
adjustments[expert_id] = []
adjustments[expert_id].append(performance)

# Average adjustments
for expert_id in adjustments:
adjustments[expert_id] = np.mean(adjustments[expert_id])

return adjustments

def _find_similar_patterns(self, context_features: Dict[str, Any]) -> List[Dict[str, Any]]:
"""Find similar patterns in memory"""
similar_patterns = []

for pattern in self.pattern_memory:
similarity = self._compute_pattern_similarity(context_features, pattern['context'])
if similarity > 0.7: # Threshold
similar_patterns.append(pattern)

return similar_patterns

def _compute_pattern_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
"""Compute similarity between context patterns"""
common_keys = set(context1.keys()) & set(context2.keys())
if not common_keys:
return 0.0

similarities = []
for key in common_keys:
if isinstance(context1[key], (int, float)) and isinstance(context2[key], (int, float)):
# Numerical similarity
diff = abs(context1[key] - context2[key])
max_val = max(abs(context1[key]), abs(context2[key]), 1.0)
similarity = 1.0 - (diff / max_val)
similarities.append(similarity)
elif context1[key] == context2[key]:
# Exact match
similarities.append(1.0)
else:
# No match
similarities.append(0.0)

return np.mean(similarities) if similarities else 0.0

def update(self, expert_id: int, performance: float, feedback: Dict[str, Any] = None):
"""Update meta-learner with new information"""
pattern = {
'expert_id': expert_id,
'performance': performance,
'context': feedback.get('context', {}) if feedback else {},
'timestamp': time.time()
}

self.pattern_memory.append(pattern)
self.expert_patterns[expert_id].append(pattern)

def get_stats(self) -> Dict[str, Any]:
"""Get meta-learner statistics"""
return {
'pattern_count': len(self.pattern_memory),
'expert_pattern_counts': {expert_id: len(patterns) 
for expert_id, patterns in self.expert_patterns.items()},
'memory_utilization': len(self.pattern_memory) / self.memory_size
}


class PerformancePredictor:
"""Performance prediction component for experts"""

def __init__(self, config: Dict[str, Any] = None):
self.config = config or {}
self.prediction_window = self.config.get('prediction_window', 20)

self.expert_models = {}
self.feature_history = deque(maxlen=1000)

def predict_performance(self, context_features: Dict[str, Any], 
performance_history: Dict[int, deque]) -> Dict[int, float]:
"""Predict expert performance based on context"""
predictions = {}

for expert_id, history in performance_history.items():
if len(history) > 5:
# Simple prediction based on recent performance trend
recent_performance = list(history)[-5:]
trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]

# Predict next performance
predicted_performance = recent_performance[-1] + trend
predictions[expert_id] = max(0.0, min(1.0, predicted_performance))
else:
# Default prediction for new experts
predictions[expert_id] = 0.5

return predictions

def update(self, expert_id: int, performance: float, feedback: Dict[str, Any] = None):
"""Update performance predictor with new information"""
# Store feature-performance pairs
if feedback and 'context' in feedback:
self.feature_history.append({
'expert_id': expert_id,
'performance': performance,
'features': feedback['context'],
'timestamp': time.time()
})

def get_stats(self) -> Dict[str, Any]:
"""Get performance predictor statistics"""
return {
'feature_history_count': len(self.feature_history),
'expert_model_count': len(self.expert_models),
'prediction_accuracy': 0.7 # Placeholder
}
