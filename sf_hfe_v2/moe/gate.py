"""
Domain Router (Gate) - Entry Layer
The first intelligence that client data meets, responsible for domain routing and expert activation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import logging
from collections import defaultdict, Counter
import re

class DomainLearner:
"""
Dynamic domain learning system that discovers new domains from data patterns
"""

def __init__(self, config: Dict[str, Any] = None):
self.config = config or {}
self.learned_domains = {}
self.pattern_history = defaultdict(list)
self.min_samples_for_domain = self.config.get('min_samples_for_domain', 50)

def learn_from_data(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
"""
Learn new domain patterns from data

Args:
data: Input dataset
context: Extracted context

Returns:
dict: Learned domain information
"""
# Extract patterns from data
patterns = self._extract_data_patterns(data, context)

# Check if patterns match existing domains
domain_match = self._match_existing_domains(patterns)

if not domain_match:
# Learn new domain
new_domain = self._create_new_domain(patterns, context)
if new_domain:
self.learned_domains[new_domain['name']] = new_domain

return domain_match or new_domain

def _extract_data_patterns(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
"""Extract patterns from data for domain learning"""
patterns = {
'column_patterns': {},
'value_patterns': {},
'statistical_patterns': {},
'temporal_patterns': {}
}

# Analyze column names
for col in data.columns:
patterns['column_patterns'][col] = {
'length': len(col),
'has_numbers': bool(re.search(r'\d', col)),
'has_underscores': '_' in col,
'has_caps': any(c.isupper() for c in col),
'word_count': len(col.split())
}

# Analyze value patterns
numeric_cols = data.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
if len(data[col].dropna()) > 0:
patterns['value_patterns'][col] = {
'min': data[col].min(),
'max': data[col].max(),
'mean': data[col].mean(),
'std': data[col].std(),
'unique_count': data[col].nunique(),
'null_count': data[col].isnull().sum()
}

return patterns

def _match_existing_domains(self, patterns: Dict[str, Any]) -> Optional[Dict[str, Any]]:
"""Check if patterns match existing learned domains"""
# Simple pattern matching - can be enhanced with ML
for domain_name, domain_info in self.learned_domains.items():
similarity = self._calculate_pattern_similarity(patterns, domain_info['patterns'])
if similarity > 0.7:
return {
'name': domain_name,
'confidence': similarity,
'source': 'learned'
}
return None

def _create_new_domain(self, patterns: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
"""Create a new domain from patterns"""
# Generate domain name from context
domain_name = self._generate_domain_name(context)

if domain_name:
return {
'name': domain_name,
'patterns': patterns,
'features': self._extract_domain_features(patterns),
'threshold': 0.6,
'confidence': 0.5
}
return None

def _generate_domain_name(self, context: Dict[str, Any]) -> str:
"""Generate a domain name from context"""
column_names = context.get('column_names', [])
if not column_names:
return 'unknown_domain'

# Extract common words from column names
all_words = []
for col in column_names:
words = re.findall(r'\b[a-zA-Z]+\b', col.lower())
all_words.extend(words)

# Find most common words
word_counts = Counter(all_words)
common_words = [word for word, count in word_counts.most_common(3) if count > 1]

if common_words:
return '_'.join(common_words[:2])
else:
return 'data_domain'

def _extract_domain_features(self, patterns: Dict[str, Any]) -> List[str]:
"""Extract domain features from patterns"""
features = []

# Analyze value patterns to determine feature types
for col, pattern in patterns['value_patterns'].items():
if pattern['unique_count'] <= 2:
features.append('binary')
elif pattern['unique_count'] < 10:
features.append('categorical')
else:
features.append('numeric')

# Add temporal if datetime columns exist
if any('date' in col.lower() or 'time' in col.lower() for col in patterns['column_patterns']):
features.append('temporal')

return list(set(features))

def _calculate_pattern_similarity(self, patterns1: Dict[str, Any], patterns2: Dict[str, Any]) -> float:
"""Calculate similarity between two pattern sets"""
# Simple similarity calculation - can be enhanced
if not patterns1 or not patterns2:
return 0.0

# Compare column patterns
cols1 = set(patterns1['column_patterns'].keys())
cols2 = set(patterns2['column_patterns'].keys())

if not cols1 or not cols2:
return 0.0

intersection = len(cols1.intersection(cols2))
union = len(cols1.union(cols2))

return intersection / union if union > 0 else 0.0

class DomainRouter:
"""
Domain Router (Gate) - Entry layer for the Scarcity MoE system

Purpose: Recognize what kind of world the data belongs to and decide which experts to activate
"""

def __init__(self, config: Optional[Dict] = None):
"""
Initialize the Domain Router

Args:
config: Configuration dictionary for the router
"""
self.config = config or {}

# Dynamic domain baskets - configurable and extensible
self.domain_baskets = self.config.get('domain_baskets', self._get_default_domain_baskets())

# Domain learning system - learns new domains from data
self.domain_learner = DomainLearner(config=self.config.get('domain_learner', {}))

# Initialize routing models
self.context_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
self.domain_classifier = None
self.routing_history = []

# Performance tracking
self.routing_accuracy = {}
self.feedback_weights = {}

# Initialize logging
self.logger = logging.getLogger(__name__)

def _get_default_domain_baskets(self) -> Dict[str, Any]:
"""
Get default domain baskets - minimal generic domains
These can be overridden or extended through configuration
"""
return {
'numeric_domain': {
'features': ['numeric', 'temporal'],
'threshold': 0.6,
'description': 'Domain with primarily numeric data'
},
'categorical_domain': {
'features': ['categorical', 'temporal'],
'threshold': 0.6,
'description': 'Domain with primarily categorical data'
},
'mixed_domain': {
'features': ['numeric', 'categorical', 'temporal'],
'threshold': 0.5,
'description': 'Domain with mixed data types'
},
'temporal_domain': {
'features': ['temporal', 'numeric'],
'threshold': 0.7,
'description': 'Domain with strong temporal characteristics'
}
}

def route_data(self, data: pd.DataFrame, metadata: Optional[Dict] = None) -> Dict[str, Any]:
"""
Route incoming data to appropriate domain baskets

Args:
data: Input dataset
metadata: Optional metadata about the data source

Returns:
dict: Routing result with domain scores and confidence
"""
try:
# Step 1: Context Extraction
context_vector = self._extract_context(data, metadata)

# Step 2: Domain Learning (try to learn new domain from data)
learned_domain = self.domain_learner.learn_from_data(data, context_vector)

# Step 3: Domain Scoring (use learned domains + default domains)
domain_scores = self._score_domains(context_vector, data, learned_domain)

# Step 4: Expert Activation Decision
activation_plan = self._decide_activation(domain_scores)

# Step 5: Build route map
route_result = {
'route_map': activation_plan,
'basket_label': max(domain_scores.items(), key=lambda x: x[1])[0],
'confidence_score': max(domain_scores.values()),
'domain_scores': domain_scores,
'context_vector': context_vector,
'learned_domain': learned_domain,
'timestamp': pd.Timestamp.now(),
'data_shape': data.shape
}

# Store routing history for feedback
self.routing_history.append(route_result)

self.logger.info(f"Data routed to {route_result['basket_label']} with confidence {route_result['confidence_score']:.3f}")

return route_result

except Exception as e:
self.logger.error(f"Error in domain routing: {str(e)}")
# Return default routing to mixed domain
return {
'route_map': {'mixed_domain': 1.0},
'basket_label': 'mixed_domain',
'confidence_score': 0.5,
'domain_scores': {'mixed_domain': 0.5},
'context_vector': None,
'learned_domain': None,
'timestamp': pd.Timestamp.now(),
'data_shape': data.shape if hasattr(data, 'shape') else (0, 0)
}

def _extract_context(self, data: pd.DataFrame, metadata: Optional[Dict] = None) -> Dict[str, Any]:
"""
Extract context information from the data

Args:
data: Input dataset
metadata: Optional metadata

Returns:
dict: Context vector with extracted features
"""
context = {
'column_names': list(data.columns) if hasattr(data, 'columns') else [],
'data_types': data.dtypes.to_dict() if hasattr(data, 'dtypes') else {},
'shape': data.shape if hasattr(data, 'shape') else (0, 0),
'metadata': metadata or {}
}

# Extract statistical features
if len(data) > 0:
context['numeric_columns'] = data.select_dtypes(include=[np.number]).columns.tolist()
context['categorical_columns'] = data.select_dtypes(include=['object', 'category']).columns.tolist()
context['temporal_columns'] = data.select_dtypes(include=['datetime64']).columns.tolist()

# Basic statistics
if len(context['numeric_columns']) > 0:
context['numeric_stats'] = data[context['numeric_columns']].describe().to_dict()

# Text features from column names
context['text_features'] = ' '.join(context['column_names']).lower()

return context

def _score_domains(self, context: Dict[str, Any], data: pd.DataFrame, learned_domain: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
"""
Score each domain based on context and data characteristics

Args:
context: Extracted context vector
data: Input dataset
learned_domain: Optional learned domain from data

Returns:
dict: Domain scores
"""
domain_scores = {}

# Add learned domain to scoring if available
all_domains = self.domain_baskets.copy()
if learned_domain and learned_domain.get('name'):
all_domains[learned_domain['name']] = {
'features': learned_domain.get('features', []),
'threshold': learned_domain.get('threshold', 0.6),
'description': learned_domain.get('description', 'Learned domain')
}

for domain, config in all_domains.items():
score = 0.0

# Score based on data types (primary scoring method)
type_score = self._calculate_type_similarity(context, config['features'])
score += type_score * 0.6

# Score based on data patterns
pattern_score = self._calculate_pattern_similarity(data, domain)
score += pattern_score * 0.4

domain_scores[domain] = min(score, 1.0) # Cap at 1.0

return domain_scores

def _calculate_type_similarity(self, context: Dict[str, Any], expected_features: List[str]) -> float:
"""
Calculate similarity between data types and expected features

Args:
context: Context vector
expected_features: Expected feature types

Returns:
float: Type similarity score
"""
if not expected_features:
return 0.0

# Map context features to expected features
context_features = set()

if context.get('numeric_columns'):
context_features.add('numeric')
if context.get('categorical_columns'):
context_features.add('categorical')
if context.get('temporal_columns'):
context_features.add('temporal')

# Check for generic feature patterns
text_features = context.get('text_features', '').lower()

# Detect binary features
if any(word in text_features for word in ['binary', 'flag', 'indicator', 'status']):
context_features.add('binary')

# Detect ordinal features 
if any(word in text_features for word in ['rank', 'order', 'level', 'grade', 'score']):
context_features.add('ordinal')

# Detect continuous features
if any(word in text_features for word in ['continuous', 'measure', 'value', 'amount', 'rate']):
context_features.add('continuous')

expected_set = set(expected_features)
intersection = context_features.intersection(expected_set)

return len(intersection) / len(expected_set) if expected_set else 0.0

def _calculate_pattern_similarity(self, data: pd.DataFrame, domain: str) -> float:
"""
Calculate similarity based on generic data patterns

Args:
data: Input dataset
domain: Domain name

Returns:
float: Pattern similarity score
"""
if len(data) == 0:
return 0.0

score = 0.0

# Generic pattern analysis based on domain type
if 'numeric' in domain:
# Score based on numeric data characteristics
numeric_cols = data.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
score += len(numeric_cols) / len(data.columns) * 0.8

# Additional scoring based on numeric patterns
sample_data = data[numeric_cols].head(100)
if len(sample_data) > 0:
# Look for continuous values (many unique values)
continuous_cols = sample_data.columns[
sample_data.nunique() > 10
]
score += len(continuous_cols) / len(numeric_cols) * 0.2

elif 'categorical' in domain:
# Score based on categorical data characteristics
categorical_cols = data.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
score += len(categorical_cols) / len(data.columns) * 0.8

# Additional scoring based on categorical patterns
sample_data = data[categorical_cols].head(100)
if len(sample_data) > 0:
# Look for discrete categories (few unique values)
discrete_cols = sample_data.columns[
sample_data.nunique() <= 20
]
score += len(discrete_cols) / len(categorical_cols) * 0.2

elif 'temporal' in domain:
# Score based on temporal data characteristics
temporal_cols = data.select_dtypes(include=['datetime64']).columns
if len(temporal_cols) > 0:
score += len(temporal_cols) / len(data.columns) * 0.8

# Also check for temporal patterns in numeric data
numeric_cols = data.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
# Look for time-series like patterns (ordered data)
sample_data = data[numeric_cols].head(100)
if len(sample_data) > 0:
# Check for monotonic patterns (common in time series)
monotonic_cols = 0
for col in sample_data.columns:
if len(sample_data[col].dropna()) > 1:
diff = sample_data[col].diff().dropna()
if len(diff) > 0:
# Check if mostly increasing or decreasing
positive_ratio = (diff > 0).sum() / len(diff)
if positive_ratio > 0.7 or positive_ratio < 0.3:
monotonic_cols += 1

score += monotonic_cols / len(numeric_cols) * 0.2

elif 'mixed' in domain:
# Score based on mixed data characteristics
numeric_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(include=['object', 'category']).columns

# Balanced mix of data types
if len(numeric_cols) > 0 and len(categorical_cols) > 0:
numeric_ratio = len(numeric_cols) / len(data.columns)
categorical_ratio = len(categorical_cols) / len(data.columns)

# Score higher for balanced mix (not too skewed)
balance_score = 1.0 - abs(numeric_ratio - categorical_ratio)
score += balance_score * 0.8

# General data diversity
score += min(len(data.columns) / 10, 1.0) * 0.2

return min(score, 1.0)

def _decide_activation(self, domain_scores: Dict[str, float]) -> Dict[str, float]:
"""
Decide which experts to activate based on domain scores

Args:
domain_scores: Domain confidence scores

Returns:
dict: Activation plan with expert weights
"""
activation_plan = {}

# Activate experts for domains above threshold
threshold = self.config.get('activation_threshold', 0.5)

for domain, score in domain_scores.items():
if score >= threshold:
activation_plan[domain] = score

# Ensure at least one expert is activated
if not activation_plan:
best_domain = max(domain_scores.items(), key=lambda x: x[1])
activation_plan[best_domain[0]] = best_domain[1]

return activation_plan

def update_routing_feedback(self, routing_result: Dict[str, Any], success: bool):
"""
Update routing model based on feedback

Args:
routing_result: Previous routing result
success: Whether the routing was successful
"""
domain = routing_result['basket_label']

# Update routing accuracy
if domain not in self.routing_accuracy:
self.routing_accuracy[domain] = []

self.routing_accuracy[domain].append(success)

# Keep only recent history
if len(self.routing_accuracy[domain]) > 100:
self.routing_accuracy[domain] = self.routing_accuracy[domain][-100:]

# Update feedback weights
if domain not in self.feedback_weights:
self.feedback_weights[domain] = 1.0

if success:
self.feedback_weights[domain] = min(self.feedback_weights[domain] * 1.1, 2.0)
else:
self.feedback_weights[domain] = max(self.feedback_weights[domain] * 0.9, 0.1)

def get_routing_statistics(self) -> Dict[str, Any]:
"""
Get routing statistics and performance metrics

Returns:
dict: Routing statistics
"""
stats = {
'total_routings': len(self.routing_history),
'domain_accuracy': {},
'feedback_weights': self.feedback_weights.copy(),
'recent_routings': self.routing_history[-10:] if self.routing_history else []
}

# Calculate accuracy for each domain
for domain, history in self.routing_accuracy.items():
if history:
stats['domain_accuracy'][domain] = sum(history) / len(history)

return stats

def get_weights(self) -> Dict[str, Any]:
"""
Get current weights/parameters for Domain Router

Returns:
dict: Domain Router weights and parameters
"""
return {
'domain_weights': self.feedback_weights.copy(),
'routing_accuracy': self.routing_accuracy.copy(),
'domain_baskets': self.domain_baskets.copy(),
'router_status': self.get_status()
}

def update_parameters(self, params: Dict[str, Any]):
"""
Update parameters for Domain Router

Args:
params: New parameters
"""
if 'domain_weights' in params:
self.feedback_weights.update(params['domain_weights'])
if 'domain_baskets' in params:
self.domain_baskets.update(params['domain_baskets'])