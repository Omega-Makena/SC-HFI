"""
Base Expert Class for Online Learning System
Implements the foundation for all 30 core experts in SCARCITY
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
from collections import defaultdict, deque
import json


class BaseExpert(ABC):
"""
Base class for all 30 SCARCITY experts implementing online learning capabilities

Each expert specializes in detecting, explaining, or predicting one or more 
of the six fundamental data relations:
- Structural Relations (data organization, shape, types)
- Statistical Relations (co-variation, distributions, variance)
- Temporal Relations (change over time, seasonality, drift)
- Relational/Interactional Relations (entity interactions, dependencies)
- Causal Relations (directional influence, cause-effect)
- Semantic/Contextual Relations (meaning, context, domain adaptation)
"""

def __init__(self, expert_id: int, name: str, relation_types: List[str], 
config: Dict[str, Any] = None):
"""
Initialize base expert

Args:
expert_id: Unique identifier for this expert
name: Human-readable name of the expert
relation_types: List of relation types this expert handles
config: Configuration dictionary
"""
self.expert_id = expert_id
self.name = name
self.relation_types = relation_types
self.config = config or {}

# Online learning state
self.is_online = True
self.learning_rate = self.config.get('learning_rate', 0.001)
self.memory_size = self.config.get('memory_size', 1000)
self.adaptation_rate = self.config.get('adaptation_rate', 0.01)

# Performance tracking
self.performance_history = deque(maxlen=100)
self.confidence_history = deque(maxlen=100)
self.activation_count = 0
self.last_activation_time = 0

# Online memory for continual learning
self.memory_buffer = deque(maxlen=self.memory_size)
self.meta_memory = defaultdict(list)

# Expert state
self.is_active = False
self.current_confidence = 0.0
self.last_update_time = time.time()

# Neural components (if needed)
self.neural_model = None
self.optimizer = None

# Logger
self.logger = logging.getLogger(f"Expert-{expert_id}-{name}")

# Initialize expert-specific components
self._initialize_expert()

@abstractmethod
def _initialize_expert(self):
"""Initialize expert-specific components (neural models, algorithms, etc.)"""
pass

@abstractmethod
def process_data(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
"""
Process data and return expert insights

Args:
data: Input data array
metadata: Additional context information

Returns:
Dictionary containing expert insights and confidence scores
"""
pass

@abstractmethod
def update_online(self, data: np.ndarray, feedback: Dict[str, Any] = None) -> bool:
"""
Update expert knowledge based on new data and feedback

Args:
data: New data for learning
feedback: Performance feedback from higher-level systems

Returns:
True if update was successful
"""
pass

def activate(self, context: Dict[str, Any] = None) -> bool:
"""
Activate expert based on context

Args:
context: Context information for activation decision

Returns:
True if expert should be activated
"""
activation_prob = self._compute_activation_probability(context)
self.is_active = activation_prob > 0.5

if self.is_active:
self.activation_count += 1
self.last_activation_time = time.time()
self.logger.debug(f"Expert {self.name} activated with probability {activation_prob:.3f}")

return self.is_active

def _compute_activation_probability(self, context: Dict[str, Any] = None) -> float:
"""
Compute probability of expert activation based on context

Args:
context: Context information

Returns:
Activation probability between 0 and 1
"""
# Base activation probability
base_prob = 0.1

# Context-based adjustments
if context:
# Adjust based on data characteristics
if 'data_shape' in context:
shape_factor = min(context['data_shape'][0] / 1000, 1.0)
base_prob += shape_factor * 0.2

# Adjust based on domain
if 'domain' in context and hasattr(self, 'preferred_domains'):
if context['domain'] in self.preferred_domains:
base_prob += 0.3

# Adjust based on recent performance
if self.performance_history:
recent_performance = np.mean(list(self.performance_history)[-10:])
base_prob += recent_performance * 0.2

# Ensure probability is in valid range
return max(0.0, min(1.0, base_prob))

def compute_confidence(self, data: np.ndarray, result: Dict[str, Any]) -> float:
"""
Compute confidence score for expert's analysis

Args:
data: Input data
result: Expert's analysis result

Returns:
Confidence score between 0 and 1
"""
# Base confidence from result quality
base_confidence = result.get('confidence', 0.5)

# Adjust based on data characteristics
data_quality = self._assess_data_quality(data)
confidence = base_confidence * data_quality

# Adjust based on expert's recent performance
if self.performance_history:
recent_performance = np.mean(list(self.performance_history)[-5:])
confidence = confidence * (0.5 + 0.5 * recent_performance)

self.current_confidence = max(0.0, min(1.0, confidence))
return self.current_confidence

def _assess_data_quality(self, data: np.ndarray) -> float:
"""
Assess quality of input data

Args:
data: Input data array

Returns:
Data quality score between 0 and 1
"""
if data.size == 0:
return 0.0

# Check for missing values
missing_ratio = np.isnan(data).sum() / data.size

# Check for constant values
if data.ndim == 1:
is_constant = np.all(data == data[0])
else:
is_constant = np.all(data == data[0, 0])

# Check for outliers (using IQR method)
outlier_ratio = 0.0
if data.size > 4:
q1, q3 = np.percentile(data.flatten(), [25, 75])
iqr = q3 - q1
if iqr > 0:
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = np.sum((data.flatten() < lower_bound) | (data.flatten() > upper_bound))
outlier_ratio = outliers / data.size

# Compute quality score
quality = 1.0 - missing_ratio - (0.5 if is_constant else 0.0) - min(outlier_ratio, 0.3)
return max(0.0, min(1.0, quality))

def store_memory(self, data: np.ndarray, result: Dict[str, Any], 
feedback: Dict[str, Any] = None):
"""
Store important information in expert's memory

Args:
data: Input data
result: Expert's analysis result
feedback: Performance feedback
"""
memory_entry = {
'timestamp': time.time(),
'data_hash': hash(data.tobytes()),
'result': result,
'feedback': feedback,
'confidence': self.current_confidence
}

self.memory_buffer.append(memory_entry)

# Store in meta-memory for specific relation types
for relation_type in self.relation_types:
self.meta_memory[relation_type].append(memory_entry)

def retrieve_memory(self, relation_type: str = None, 
max_entries: int = 10) -> List[Dict[str, Any]]:
"""
Retrieve relevant memories

Args:
relation_type: Specific relation type to retrieve
max_entries: Maximum number of entries to return

Returns:
List of relevant memory entries
"""
if relation_type and relation_type in self.meta_memory:
memories = self.meta_memory[relation_type]
else:
memories = list(self.memory_buffer)

# Sort by confidence and recency
memories.sort(key=lambda x: (x['confidence'], x['timestamp']), reverse=True)
return memories[:max_entries]

def adapt_learning_rate(self, performance_trend: float):
"""
Adapt learning rate based on performance trend

Args:
performance_trend: Recent performance trend (-1 to 1)
"""
if performance_trend > 0.1: # Improving performance
self.learning_rate *= 1.01
elif performance_trend < -0.1: # Declining performance
self.learning_rate *= 0.99

# Keep learning rate in reasonable bounds
self.learning_rate = max(1e-6, min(1e-2, self.learning_rate))

def get_expert_state(self) -> Dict[str, Any]:
"""
Get current state of the expert

Returns:
Dictionary containing expert state information
"""
return {
'expert_id': self.expert_id,
'name': self.name,
'relation_types': self.relation_types,
'is_active': self.is_active,
'is_online': self.is_online,
'current_confidence': self.current_confidence,
'activation_count': self.activation_count,
'learning_rate': self.learning_rate,
'memory_size': len(self.memory_buffer),
'performance_history': list(self.performance_history),
'last_update_time': self.last_update_time
}

def reset_expert(self):
"""Reset expert to initial state"""
self.is_active = False
self.current_confidence = 0.0
self.activation_count = 0
self.performance_history.clear()
self.confidence_history.clear()
self.memory_buffer.clear()
self.meta_memory.clear()
self.last_update_time = time.time()

# Reinitialize expert-specific components
self._initialize_expert()

def __str__(self):
return f"Expert({self.expert_id}, {self.name}, {self.relation_types})"

def __repr__(self):
return self.__str__()


class OnlineLearningRouter:
"""
Router for dynamically selecting and weighting experts based on online learning
"""

def __init__(self, experts: List[BaseExpert], config: Dict[str, Any] = None):
"""
Initialize online learning router

Args:
experts: List of all available experts
config: Configuration dictionary
"""
self.experts = {expert.expert_id: expert for expert in experts}
self.config = config or {}

# Routing parameters
self.exploration_rate = self.config.get('exploration_rate', 0.1)
self.exploitation_rate = self.config.get('exploitation_rate', 0.9)
self.routing_memory_size = self.config.get('routing_memory_size', 1000)

# Expert weights and performance tracking
self.expert_weights = {expert_id: 1.0 for expert_id in self.experts.keys()}
self.expert_performance = {expert_id: deque(maxlen=100) for expert_id in self.experts.keys()}
self.routing_history = deque(maxlen=self.routing_memory_size)

# Context-based routing
self.context_patterns = defaultdict(list)

self.logger = logging.getLogger("OnlineLearningRouter")

def route_experts(self, data: np.ndarray, context: Dict[str, Any] = None) -> List[Tuple[int, float]]:
"""
Route data to appropriate experts

Args:
data: Input data
context: Context information

Returns:
List of (expert_id, weight) tuples for selected experts
"""
context = context or {}

# Extract context features
context_features = self._extract_context_features(data, context)

# Compute expert scores
expert_scores = {}
for expert_id, expert in self.experts.items():
# Base score from expert weights
base_score = self.expert_weights[expert_id]

# Context-based score
context_score = self._compute_context_score(expert, context_features)

# Performance-based score
performance_score = self._compute_performance_score(expert_id)

# Exploration bonus
exploration_bonus = self._compute_exploration_bonus(expert_id)

# Combined score
expert_scores[expert_id] = (
base_score * 0.4 + 
context_score * 0.3 + 
performance_score * 0.2 + 
exploration_bonus * 0.1
)

# Select top experts
sorted_experts = sorted(expert_scores.items(), key=lambda x: x[1], reverse=True)

# Apply softmax to get weights
scores = np.array([score for _, score in sorted_experts])
weights = F.softmax(torch.tensor(scores), dim=0).numpy()

# Select experts with weights above threshold
threshold = 0.05
selected_experts = []
for i, (expert_id, _) in enumerate(sorted_experts):
if weights[i] > threshold:
selected_experts.append((expert_id, float(weights[i])))

# Record routing decision
self.routing_history.append({
'timestamp': time.time(),
'context_features': context_features,
'selected_experts': selected_experts,
'expert_scores': expert_scores
})

return selected_experts

def _extract_context_features(self, data: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
"""Extract features from data and context for routing decisions"""
features = {
'data_shape': data.shape,
'data_size': data.size,
'data_type': str(data.dtype),
'has_missing': np.isnan(data).any() if np.issubdtype(data.dtype, np.floating) else False,
'is_temporal': 'time' in context.get('domain', '').lower(),
'domain': context.get('domain', 'unknown'),
'task_type': context.get('task_type', 'unknown')
}

# Add statistical features
if data.size > 0:
features.update({
'mean': float(np.nanmean(data)),
'std': float(np.nanstd(data)),
'min': float(np.nanmin(data)),
'max': float(np.nanmax(data)),
'skewness': float(self._compute_skewness(data)),
'kurtosis': float(self._compute_kurtosis(data))
})

return features

def _compute_context_score(self, expert: BaseExpert, context_features: Dict[str, Any]) -> float:
"""Compute how well expert matches current context"""
score = 0.0

# Domain matching
if hasattr(expert, 'preferred_domains'):
if context_features['domain'] in expert.preferred_domains:
score += 0.5

# Data type matching
if hasattr(expert, 'preferred_data_types'):
if context_features['data_type'] in expert.preferred_data_types:
score += 0.3

# Task type matching
if hasattr(expert, 'preferred_tasks'):
if context_features['task_type'] in expert.preferred_tasks:
score += 0.2

return score

def _compute_performance_score(self, expert_id: int) -> float:
"""Compute performance-based score for expert"""
if not self.expert_performance[expert_id]:
return 0.5 # Neutral score for new experts

recent_performance = list(self.expert_performance[expert_id])[-10:]
return np.mean(recent_performance)

def _compute_exploration_bonus(self, expert_id: int) -> float:
"""Compute exploration bonus to encourage trying underused experts"""
total_activations = sum(len(perf) for perf in self.expert_performance.values())
expert_activations = len(self.expert_performance[expert_id])

if total_activations == 0:
return 1.0

usage_ratio = expert_activations / total_activations
return max(0.0, 1.0 - usage_ratio)

def update_routing(self, expert_id: int, performance: float, feedback: Dict[str, Any] = None):
"""Update routing based on expert performance"""
# Update performance history
self.expert_performance[expert_id].append(performance)

# Update expert weights using online learning
learning_rate = self.config.get('routing_learning_rate', 0.01)
self.expert_weights[expert_id] += learning_rate * performance

# Keep weights in reasonable bounds
self.expert_weights[expert_id] = max(0.0, min(2.0, self.expert_weights[expert_id]))

# Update context patterns
if feedback and 'context' in feedback:
self.context_patterns[expert_id].append(feedback['context'])

def _compute_skewness(self, data: np.ndarray) -> float:
"""Compute skewness of data"""
if data.size < 3:
return 0.0

data_flat = data.flatten()
mean = np.nanmean(data_flat)
std = np.nanstd(data_flat)

if std == 0:
return 0.0

skewness = np.nanmean(((data_flat - mean) / std) ** 3)
return skewness

def _compute_kurtosis(self, data: np.ndarray) -> float:
"""Compute kurtosis of data"""
if data.size < 4:
return 0.0

data_flat = data.flatten()
mean = np.nanmean(data_flat)
std = np.nanstd(data_flat)

if std == 0:
return 0.0

kurtosis = np.nanmean(((data_flat - mean) / std) ** 4) - 3
return kurtosis

def get_routing_stats(self) -> Dict[str, Any]:
"""Get routing statistics"""
return {
'total_experts': len(self.experts),
'active_experts': sum(1 for expert in self.experts.values() if expert.is_active),
'expert_weights': dict(self.expert_weights),
'routing_history_size': len(self.routing_history),
'avg_performance': {
expert_id: np.mean(list(perf)) if perf else 0.0
for expert_id, perf in self.expert_performance.items()
}
}
