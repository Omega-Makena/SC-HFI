"""
Statistical Experts - Quantitative Essence Layer
Implements 4 core experts for understanding numerical co-variation, distributions, and statistical patterns
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict, Counter
from scipy import stats
from scipy.stats import entropy, kurtosis, skew
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import time

from .base_expert import BaseExpert


class DescriptiveExpert(BaseExpert):
"""
Expert 5: Descriptive Expert
Computes distributions, skew, kurtosis, entropy, and statistical summaries
"""

def __init__(self, expert_id: int = 5, config: Dict[str, Any] = None):
super().__init__(
expert_id=expert_id,
name="DescriptiveExpert",
relation_types=["statistical"],
config=config
)

self.preferred_domains = ["numerical", "statistical", "analytical"]
self.preferred_data_types = ["float64", "int64"]
self.preferred_tasks = ["descriptive_analysis", "distribution_analysis", "statistical_summary"]

# Statistical parameters
self.confidence_level = self.config.get('confidence_level', 0.95)
self.outlier_method = self.config.get('outlier_method', 'iqr')
self.distribution_tests = self.config.get('distribution_tests', ['normal', 'uniform', 'exponential'])

# Pattern storage
self.distribution_patterns = defaultdict(list)
self.statistical_patterns = defaultdict(list)

def _initialize_expert(self):
"""Initialize descriptive analysis components"""
self.distribution_analyzer = DistributionAnalyzer()
self.statistical_calculator = StatisticalCalculator()
self.summary_generator = SummaryGenerator()

def process_data(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
"""Process data to compute descriptive statistics"""
metadata = metadata or {}

# Compute basic statistics
basic_stats = self.statistical_calculator.compute_basic_stats(data)

# Analyze distributions
distribution_info = self.distribution_analyzer.analyze_distributions(data)

# Generate summary
summary_info = self.summary_generator.generate_summary(data, basic_stats, distribution_info)

# Compute confidence
confidence = self._compute_descriptive_confidence(basic_stats, distribution_info, summary_info)

result = {
'expert_id': self.expert_id,
'expert_name': self.name,
'basic_stats': basic_stats,
'distribution_info': distribution_info,
'summary_info': summary_info,
'confidence': confidence,
'relation_types': self.relation_types,
'timestamp': time.time()
}

return result

def update_online(self, data: np.ndarray, feedback: Dict[str, Any] = None) -> bool:
"""Update descriptive patterns based on new data"""
try:
# Extract statistical features
stat_features = self.statistical_calculator.extract_features(data)
dist_features = self.distribution_analyzer.extract_features(data)

# Update patterns
self.statistical_patterns['recent'].append(stat_features)
self.distribution_patterns['recent'].append(dist_features)

# Adapt parameters based on feedback
if feedback and 'statistical_accuracy' in feedback:
accuracy = feedback['statistical_accuracy']
if accuracy > 0.9:
self.confidence_level = min(0.99, self.confidence_level + 0.01)
elif accuracy < 0.7:
self.confidence_level = max(0.90, self.confidence_level - 0.01)

# Store in memory
self.store_memory(data, {
'stat_features': stat_features,
'dist_features': dist_features
}, feedback)

return True

except Exception as e:
self.logger.error(f"Error updating DescriptiveExpert: {e}")
return False

def _compute_descriptive_confidence(self, basic_stats: Dict, distribution_info: Dict, summary_info: Dict) -> float:
"""Compute confidence in descriptive analysis"""
confidence = 0.8 # Base confidence for descriptive stats

# Distribution clarity
dist_clarity = distribution_info.get('clarity_score', 0)
confidence += dist_clarity * 0.1

# Summary completeness
summary_completeness = summary_info.get('completeness_score', 0)
confidence += summary_completeness * 0.1

return min(1.0, confidence)


class CorrelationExpert(BaseExpert):
"""
Expert 6: Correlation Expert
Learns linear and nonlinear dependencies between variables
"""

def __init__(self, expert_id: int = 6, config: Dict[str, Any] = None):
super().__init__(
expert_id=expert_id,
name="CorrelationExpert",
relation_types=["statistical"],
config=config
)

self.preferred_domains = ["multivariate", "relational", "correlational"]
self.preferred_data_types = ["float64", "int64"]
self.preferred_tasks = ["correlation_analysis", "dependency_detection", "relationship_analysis"]

# Correlation parameters
self.correlation_threshold = self.config.get('correlation_threshold', 0.3)
self.nonlinear_threshold = self.config.get('nonlinear_threshold', 0.2)
self.mutual_info_threshold = self.config.get('mutual_info_threshold', 0.1)

# Pattern storage
self.correlation_patterns = defaultdict(list)
self.dependency_patterns = defaultdict(list)

def _initialize_expert(self):
"""Initialize correlation analysis components"""
self.linear_correlator = LinearCorrelator()
self.nonlinear_correlator = NonlinearCorrelator()
self.dependency_detector = DependencyDetector()

def process_data(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
"""Process data to detect correlations and dependencies"""
metadata = metadata or {}

# Detect linear correlations
linear_corr = self.linear_correlator.detect_linear_correlations(data)

# Detect nonlinear correlations
nonlinear_corr = self.nonlinear_correlator.detect_nonlinear_correlations(data)

# Detect dependencies
dependencies = self.dependency_detector.detect_dependencies(data)

# Compute confidence
confidence = self._compute_correlation_confidence(linear_corr, nonlinear_corr, dependencies)

result = {
'expert_id': self.expert_id,
'expert_name': self.name,
'linear_correlations': linear_corr,
'nonlinear_correlations': nonlinear_corr,
'dependencies': dependencies,
'confidence': confidence,
'relation_types': self.relation_types,
'timestamp': time.time()
}

return result

def update_online(self, data: np.ndarray, feedback: Dict[str, Any] = None) -> bool:
"""Update correlation patterns based on new data"""
try:
# Extract correlation features
linear_features = self.linear_correlator.extract_features(data)
nonlinear_features = self.nonlinear_correlator.extract_features(data)

# Update patterns
self.correlation_patterns['recent'].append(linear_features)
self.dependency_patterns['recent'].append(nonlinear_features)

# Adapt thresholds based on feedback
if feedback and 'correlation_accuracy' in feedback:
accuracy = feedback['correlation_accuracy']
if accuracy > 0.8:
self.correlation_threshold *= 1.01
elif accuracy < 0.6:
self.correlation_threshold *= 0.99

self.correlation_threshold = max(0.1, min(0.8, self.correlation_threshold))

# Store in memory
self.store_memory(data, {
'linear_features': linear_features,
'nonlinear_features': nonlinear_features
}, feedback)

return True

except Exception as e:
self.logger.error(f"Error updating CorrelationExpert: {e}")
return False

def _compute_correlation_confidence(self, linear_corr: Dict, nonlinear_corr: Dict, dependencies: Dict) -> float:
"""Compute confidence in correlation analysis"""
confidence = 0.7 # Base confidence

# Linear correlation strength
linear_strength = linear_corr.get('strength_score', 0)
confidence += linear_strength * 0.15

# Nonlinear correlation strength
nonlinear_strength = nonlinear_corr.get('strength_score', 0)
confidence += nonlinear_strength * 0.15

return min(1.0, confidence)


class DensityExpert(BaseExpert):
"""
Expert 7: Density Expert
Estimates data manifold and latent clusters
"""

def __init__(self, expert_id: int = 7, config: Dict[str, Any] = None):
super().__init__(
expert_id=expert_id,
name="DensityExpert",
relation_types=["statistical"],
config=config
)

self.preferred_domains = ["clustering", "manifold", "density"]
self.preferred_data_types = ["float64", "int64"]
self.preferred_tasks = ["density_estimation", "clustering", "manifold_learning"]

# Density parameters
self.cluster_range = self.config.get('cluster_range', (2, 10))
self.density_method = self.config.get('density_method', 'gaussian_mixture')
self.manifold_dim = self.config.get('manifold_dim', 2)

# Pattern storage
self.density_patterns = defaultdict(list)
self.cluster_patterns = defaultdict(list)

def _initialize_expert(self):
"""Initialize density analysis components"""
self.density_estimator = DensityEstimator()
self.cluster_detector = ClusterDetector()
self.manifold_learner = ManifoldLearner()

def process_data(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
"""Process data to estimate density and detect clusters"""
metadata = metadata or {}

# Estimate density
density_info = self.density_estimator.estimate_density(data)

# Detect clusters
cluster_info = self.cluster_detector.detect_clusters(data)

# Learn manifold
manifold_info = self.manifold_learner.learn_manifold(data)

# Compute confidence
confidence = self._compute_density_confidence(density_info, cluster_info, manifold_info)

result = {
'expert_id': self.expert_id,
'expert_name': self.name,
'density_info': density_info,
'cluster_info': cluster_info,
'manifold_info': manifold_info,
'confidence': confidence,
'relation_types': self.relation_types,
'timestamp': time.time()
}

return result

def update_online(self, data: np.ndarray, feedback: Dict[str, Any] = None) -> bool:
"""Update density patterns based on new data"""
try:
# Extract density features
density_features = self.density_estimator.extract_features(data)
cluster_features = self.cluster_detector.extract_features(data)

# Update patterns
self.density_patterns['recent'].append(density_features)
self.cluster_patterns['recent'].append(cluster_features)

# Adapt parameters based on feedback
if feedback and 'cluster_quality' in feedback:
quality = feedback['cluster_quality']
if quality > 0.8:
# Increase cluster range
self.cluster_range = (self.cluster_range[0], min(20, self.cluster_range[1] + 1))
elif quality < 0.6:
# Decrease cluster range
self.cluster_range = (max(2, self.cluster_range[0]), self.cluster_range[1])

# Store in memory
self.store_memory(data, {
'density_features': density_features,
'cluster_features': cluster_features
}, feedback)

return True

except Exception as e:
self.logger.error(f"Error updating DensityExpert: {e}")
return False

def _compute_density_confidence(self, density_info: Dict, cluster_info: Dict, manifold_info: Dict) -> float:
"""Compute confidence in density analysis"""
confidence = 0.6 # Base confidence

# Density estimation quality
density_quality = density_info.get('quality_score', 0)
confidence += density_quality * 0.2

# Cluster quality
cluster_quality = cluster_info.get('quality_score', 0)
confidence += cluster_quality * 0.2

return min(1.0, confidence)


class AnomalyExpert(BaseExpert):
"""
Expert 8: Anomaly Expert
Identifies outliers and regime changes statistically
"""

def __init__(self, expert_id: int = 8, config: Dict[str, Any] = None):
super().__init__(
expert_id=expert_id,
name="AnomalyExpert",
relation_types=["statistical"],
config=config
)

self.preferred_domains = ["anomaly_detection", "outlier_detection", "change_detection"]
self.preferred_data_types = ["float64", "int64"]
self.preferred_tasks = ["anomaly_detection", "outlier_detection", "regime_change_detection"]

# Anomaly detection parameters
self.anomaly_threshold = self.config.get('anomaly_threshold', 0.05)
self.outlier_method = self.config.get('outlier_method', 'isolation_forest')
self.regime_threshold = self.config.get('regime_threshold', 0.1)

# Pattern storage
self.anomaly_patterns = defaultdict(list)
self.regime_patterns = defaultdict(list)

def _initialize_expert(self):
"""Initialize anomaly detection components"""
self.outlier_detector = OutlierDetector()
self.regime_detector = RegimeDetector()
self.change_detector = ChangeDetector()

def process_data(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
"""Process data to detect anomalies and regime changes"""
metadata = metadata or {}

# Detect outliers
outlier_info = self.outlier_detector.detect_outliers(data)

# Detect regime changes
regime_info = self.regime_detector.detect_regime_changes(data)

# Detect changes
change_info = self.change_detector.detect_changes(data)

# Compute confidence
confidence = self._compute_anomaly_confidence(outlier_info, regime_info, change_info)

result = {
'expert_id': self.expert_id,
'expert_name': self.name,
'outlier_info': outlier_info,
'regime_info': regime_info,
'change_info': change_info,
'confidence': confidence,
'relation_types': self.relation_types,
'timestamp': time.time()
}

return result

def update_online(self, data: np.ndarray, feedback: Dict[str, Any] = None) -> bool:
"""Update anomaly patterns based on new data"""
try:
# Extract anomaly features
outlier_features = self.outlier_detector.extract_features(data)
regime_features = self.regime_detector.extract_features(data)

# Update patterns
self.anomaly_patterns['recent'].append(outlier_features)
self.regime_patterns['recent'].append(regime_features)

# Adapt thresholds based on feedback
if feedback and 'anomaly_accuracy' in feedback:
accuracy = feedback['anomaly_accuracy']
if accuracy > 0.8:
self.anomaly_threshold *= 1.01
elif accuracy < 0.6:
self.anomaly_threshold *= 0.99

self.anomaly_threshold = max(0.01, min(0.2, self.anomaly_threshold))

# Store in memory
self.store_memory(data, {
'outlier_features': outlier_features,
'regime_features': regime_features
}, feedback)

return True

except Exception as e:
self.logger.error(f"Error updating AnomalyExpert: {e}")
return False

def _compute_anomaly_confidence(self, outlier_info: Dict, regime_info: Dict, change_info: Dict) -> float:
"""Compute confidence in anomaly detection"""
confidence = 0.7 # Base confidence

# Outlier detection strength
outlier_strength = outlier_info.get('detection_strength', 0)
confidence += outlier_strength * 0.15

# Regime change detection strength
regime_strength = regime_info.get('detection_strength', 0)
confidence += regime_strength * 0.15

return min(1.0, confidence)


# Helper classes for statistical experts

class DistributionAnalyzer:
"""Analyzes data distributions"""

def analyze_distributions(self, data: np.ndarray) -> Dict[str, Any]:
"""Analyze distributions in data"""
dist_info = {
'distributions': [],
'clarity_score': 0.0,
'distribution_tests': {}
}

if data.size > 10:
# Test for common distributions
distributions = ['normal', 'uniform', 'exponential', 'lognormal']
test_results = {}

for dist_name in distributions:
try:
if dist_name == 'normal':
_, p_value = stats.normaltest(data.flatten())
elif dist_name == 'uniform':
_, p_value = stats.kstest(data.flatten(), 'uniform')
elif dist_name == 'exponential':
_, p_value = stats.kstest(data.flatten(), 'exponential')
elif dist_name == 'lognormal':
_, p_value = stats.kstest(data.flatten(), 'lognorm')

test_results[dist_name] = p_value
except:
test_results[dist_name] = 0.0

dist_info['distribution_tests'] = test_results

# Find best fitting distribution
best_dist = max(test_results.items(), key=lambda x: x[1])
dist_info['distributions'] = [best_dist[0]]
dist_info['clarity_score'] = best_dist[1]

return dist_info

def extract_features(self, data: np.ndarray) -> Dict[str, Any]:
"""Extract distribution features"""
if data.size > 4:
return {
'skewness': float(skew(data.flatten())),
'kurtosis': float(kurtosis(data.flatten())),
'entropy': float(entropy(np.histogram(data.flatten(), bins=10)[0] + 1e-10))
}
else:
return {
'skewness': 0.0,
'kurtosis': 0.0,
'entropy': 0.0
}


class StatisticalCalculator:
"""Calculates statistical measures"""

def compute_basic_stats(self, data: np.ndarray) -> Dict[str, Any]:
"""Compute basic statistical measures"""
if data.size == 0:
return {}

stats_dict = {
'count': data.size,
'mean': float(np.mean(data)),
'std': float(np.std(data)),
'min': float(np.min(data)),
'max': float(np.max(data)),
'median': float(np.median(data)),
'q25': float(np.percentile(data, 25)),
'q75': float(np.percentile(data, 75)),
'iqr': float(np.percentile(data, 75) - np.percentile(data, 25))
}

# Additional measures
if data.size > 1:
stats_dict['variance'] = float(np.var(data))
stats_dict['range'] = float(np.max(data) - np.min(data))
stats_dict['cv'] = float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else 0.0

return stats_dict

def extract_features(self, data: np.ndarray) -> Dict[str, Any]:
"""Extract statistical features"""
return self.compute_basic_stats(data)


class SummaryGenerator:
"""Generates statistical summaries"""

def generate_summary(self, data: np.ndarray, basic_stats: Dict, distribution_info: Dict) -> Dict[str, Any]:
"""Generate statistical summary"""
summary = {
'completeness_score': 0.0,
'summary_text': '',
'key_insights': []
}

# Compute completeness
completeness = 0.0
if basic_stats:
completeness += 0.5
if distribution_info.get('distributions'):
completeness += 0.3
if distribution_info.get('clarity_score', 0) > 0.5:
completeness += 0.2

summary['completeness_score'] = completeness

# Generate key insights
insights = []
if basic_stats.get('cv', 0) > 1.0:
insights.append("High variability detected")
if distribution_info.get('clarity_score', 0) > 0.7:
insights.append(f"Clear {distribution_info['distributions'][0]} distribution")

summary['key_insights'] = insights

return summary


class LinearCorrelator:
"""Detects linear correlations"""

def detect_linear_correlations(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect linear correlations"""
corr_info = {
'correlation_matrix': None,
'strong_correlations': [],
'strength_score': 0.0
}

if data.ndim > 1 and data.shape[1] > 1:
try:
# Compute correlation matrix
corr_matrix = np.corrcoef(data.T)
corr_info['correlation_matrix'] = corr_matrix.tolist()

# Find strong correlations
strong_corr = []
for i in range(corr_matrix.shape[0]):
for j in range(i+1, corr_matrix.shape[1]):
corr_val = abs(corr_matrix[i, j])
if corr_val > 0.5: # Threshold for strong correlation
strong_corr.append({
'var1': i,
'var2': j,
'correlation': float(corr_val)
})

corr_info['strong_correlations'] = strong_corr
corr_info['strength_score'] = len(strong_corr) / (data.shape[1] * (data.shape[1] - 1) / 2)

except:
pass

return corr_info

def extract_features(self, data: np.ndarray) -> Dict[str, Any]:
"""Extract correlation features"""
if data.ndim > 1 and data.shape[1] > 1:
try:
corr_matrix = np.corrcoef(data.T)
return {
'max_correlation': float(np.max(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))),
'mean_correlation': float(np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))),
'correlation_count': int(np.sum(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]) > 0.5))
}
except:
return {'max_correlation': 0.0, 'mean_correlation': 0.0, 'correlation_count': 0}
else:
return {'max_correlation': 0.0, 'mean_correlation': 0.0, 'correlation_count': 0}


class NonlinearCorrelator:
"""Detects nonlinear correlations"""

def detect_nonlinear_correlations(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect nonlinear correlations using mutual information"""
corr_info = {
'mutual_info_matrix': None,
'strong_dependencies': [],
'strength_score': 0.0
}

if data.ndim > 1 and data.shape[1] > 1:
try:
# Compute mutual information matrix
mi_matrix = np.zeros((data.shape[1], data.shape[1]))

for i in range(data.shape[1]):
for j in range(data.shape[1]):
if i != j:
# Discretize for mutual information
data_i = self._discretize(data[:, i])
data_j = self._discretize(data[:, j])
mi_matrix[i, j] = mutual_info_score(data_i, data_j)

corr_info['mutual_info_matrix'] = mi_matrix.tolist()

# Find strong dependencies
strong_deps = []
for i in range(mi_matrix.shape[0]):
for j in range(i+1, mi_matrix.shape[1]):
mi_val = mi_matrix[i, j]
if mi_val > 0.1: # Threshold for strong dependency
strong_deps.append({
'var1': i,
'var2': j,
'mutual_info': float(mi_val)
})

corr_info['strong_dependencies'] = strong_deps
corr_info['strength_score'] = len(strong_deps) / (data.shape[1] * (data.shape[1] - 1) / 2)

except:
pass

return corr_info

def _discretize(self, data: np.ndarray, bins: int = 10) -> np.ndarray:
"""Discretize continuous data for mutual information"""
try:
_, bin_edges = np.histogram(data, bins=bins)
return np.digitize(data, bin_edges[1:-1])
except:
return np.zeros_like(data, dtype=int)

def extract_features(self, data: np.ndarray) -> Dict[str, Any]:
"""Extract nonlinear correlation features"""
if data.ndim > 1 and data.shape[1] > 1:
try:
mi_matrix = np.zeros((data.shape[1], data.shape[1]))
for i in range(data.shape[1]):
for j in range(data.shape[1]):
if i != j:
data_i = self._discretize(data[:, i])
data_j = self._discretize(data[:, j])
mi_matrix[i, j] = mutual_info_score(data_i, data_j)

return {
'max_mutual_info': float(np.max(mi_matrix)),
'mean_mutual_info': float(np.mean(mi_matrix[mi_matrix > 0])),
'dependency_count': int(np.sum(mi_matrix > 0.1))
}
except:
return {'max_mutual_info': 0.0, 'mean_mutual_info': 0.0, 'dependency_count': 0}
else:
return {'max_mutual_info': 0.0, 'mean_mutual_info': 0.0, 'dependency_count': 0}


class DependencyDetector:
"""Detects dependencies between variables"""

def detect_dependencies(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect dependencies"""
dep_info = {
'dependencies': [],
'dependency_strength': 0.0,
'dependency_types': []
}

if data.ndim > 1 and data.shape[1] > 1:
# Simple dependency detection based on variance
dependencies = []
for i in range(data.shape[1]):
for j in range(i+1, data.shape[1]):
# Check if variables co-vary
corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
if abs(corr) > 0.3:
dependencies.append({
'var1': i,
'var2': j,
'strength': abs(corr),
'type': 'linear' if corr > 0 else 'negative_linear'
})

dep_info['dependencies'] = dependencies
dep_info['dependency_strength'] = len(dependencies) / (data.shape[1] * (data.shape[1] - 1) / 2)
dep_info['dependency_types'] = list(set([dep['type'] for dep in dependencies]))

return dep_info


class DensityEstimator:
"""Estimates data density"""

def estimate_density(self, data: np.ndarray) -> Dict[str, Any]:
"""Estimate data density"""
density_info = {
'density_method': 'histogram',
'density_score': 0.0,
'quality_score': 0.0
}

if data.size > 10:
try:
# Simple density estimation using histogram
hist, bin_edges = np.histogram(data.flatten(), bins=20)
density_score = np.sum(hist > 0) / len(hist) # Proportion of non-empty bins

density_info['density_score'] = density_score
density_info['quality_score'] = min(1.0, density_score * 2)

except:
pass

return density_info

def extract_features(self, data: np.ndarray) -> Dict[str, Any]:
"""Extract density features"""
if data.size > 10:
try:
hist, _ = np.histogram(data.flatten(), bins=20)
return {
'density_variance': float(np.var(hist)),
'density_entropy': float(entropy(hist + 1e-10)),
'peak_count': int(np.sum(hist > np.mean(hist)))
}
except:
return {'density_variance': 0.0, 'density_entropy': 0.0, 'peak_count': 0}
else:
return {'density_variance': 0.0, 'density_entropy': 0.0, 'peak_count': 0}


class ClusterDetector:
"""Detects clusters in data"""

def detect_clusters(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect clusters in data"""
cluster_info = {
'cluster_count': 0,
'cluster_labels': None,
'quality_score': 0.0,
'cluster_method': 'kmeans'
}

if data.ndim > 1 and data.shape[1] > 1 and data.shape[0] > 5:
try:
# Use KMeans for clustering
n_clusters = min(5, data.shape[0] // 2)
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(data)

cluster_info['cluster_count'] = n_clusters
cluster_info['cluster_labels'] = labels.tolist()

# Compute silhouette score as quality measure
from sklearn.metrics import silhouette_score
if len(set(labels)) > 1:
silhouette_avg = silhouette_score(data, labels)
cluster_info['quality_score'] = max(0.0, silhouette_avg)

except:
pass

return cluster_info

def extract_features(self, data: np.ndarray) -> Dict[str, Any]:
"""Extract cluster features"""
if data.ndim > 1 and data.shape[1] > 1 and data.shape[0] > 5:
try:
n_clusters = min(5, data.shape[0] // 2)
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(data)

return {
'cluster_count': n_clusters,
'cluster_size_variance': float(np.var(np.bincount(labels))),
'cluster_separation': float(np.mean(kmeans.inertia_))
}
except:
return {'cluster_count': 0, 'cluster_size_variance': 0.0, 'cluster_separation': 0.0}
else:
return {'cluster_count': 0, 'cluster_size_variance': 0.0, 'cluster_separation': 0.0}


class ManifoldLearner:
"""Learns data manifolds"""

def learn_manifold(self, data: np.ndarray) -> Dict[str, Any]:
"""Learn data manifold"""
manifold_info = {
'manifold_dim': 2,
'embedding_quality': 0.0,
'manifold_method': 'pca'
}

if data.ndim > 1 and data.shape[1] > 1:
try:
# Use PCA for manifold learning
from sklearn.decomposition import PCA
pca = PCA(n_components=min(2, data.shape[1]))
pca.fit(data)

manifold_info['manifold_dim'] = pca.n_components_
manifold_info['embedding_quality'] = np.sum(pca.explained_variance_ratio_)

except:
pass

return manifold_info


class OutlierDetector:
"""Detects outliers in data"""

def detect_outliers(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect outliers"""
outlier_info = {
'outlier_count': 0,
'outlier_indices': [],
'detection_strength': 0.0,
'outlier_method': 'iqr'
}

if data.size > 4:
try:
# IQR method for outlier detection
q1, q3 = np.percentile(data.flatten(), [25, 75])
iqr = q3 - q1

if iqr > 0:
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = (data.flatten() < lower_bound) | (data.flatten() > upper_bound)
outlier_indices = np.where(outliers)[0].tolist()

outlier_info['outlier_count'] = len(outlier_indices)
outlier_info['outlier_indices'] = outlier_indices
outlier_info['detection_strength'] = len(outlier_indices) / data.size

except:
pass

return outlier_info

def extract_features(self, data: np.ndarray) -> Dict[str, Any]:
"""Extract outlier features"""
if data.size > 4:
try:
q1, q3 = np.percentile(data.flatten(), [25, 75])
iqr = q3 - q1

if iqr > 0:
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = (data.flatten() < lower_bound) | (data.flatten() > upper_bound)

return {
'outlier_ratio': float(np.sum(outliers) / data.size),
'iqr': float(iqr),
'outlier_severity': float(np.mean(np.abs(data.flatten()[outliers] - np.median(data))))
}
else:
return {'outlier_ratio': 0.0, 'iqr': 0.0, 'outlier_severity': 0.0}
except:
return {'outlier_ratio': 0.0, 'iqr': 0.0, 'outlier_severity': 0.0}
else:
return {'outlier_ratio': 0.0, 'iqr': 0.0, 'outlier_severity': 0.0}


class RegimeDetector:
"""Detects regime changes in data"""

def detect_regime_changes(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect regime changes"""
regime_info = {
'regime_count': 1,
'regime_changes': [],
'detection_strength': 0.0
}

if data.size > 10:
try:
# Simple regime detection using rolling statistics
window_size = max(5, data.size // 10)
rolling_mean = np.convolve(data.flatten(), np.ones(window_size)/window_size, mode='valid')

# Detect significant changes in rolling mean
changes = []
for i in range(1, len(rolling_mean)):
change_magnitude = abs(rolling_mean[i] - rolling_mean[i-1])
if change_magnitude > np.std(rolling_mean) * 2:
changes.append({
'position': i + window_size,
'magnitude': float(change_magnitude)
})

regime_info['regime_count'] = len(changes) + 1
regime_info['regime_changes'] = changes
regime_info['detection_strength'] = len(changes) / len(rolling_mean)

except:
pass

return regime_info

def extract_features(self, data: np.ndarray) -> Dict[str, Any]:
"""Extract regime features"""
if data.size > 10:
try:
window_size = max(5, data.size // 10)
rolling_mean = np.convolve(data.flatten(), np.ones(window_size)/window_size, mode='valid')

return {
'regime_variance': float(np.var(rolling_mean)),
'regime_trend': float(np.polyfit(range(len(rolling_mean)), rolling_mean, 1)[0]),
'regime_stability': float(1.0 / (1.0 + np.std(rolling_mean)))
}
except:
return {'regime_variance': 0.0, 'regime_trend': 0.0, 'regime_stability': 0.0}
else:
return {'regime_variance': 0.0, 'regime_trend': 0.0, 'regime_stability': 0.0}


class ChangeDetector:
"""Detects changes in data"""

def detect_changes(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect changes in data"""
change_info = {
'change_points': [],
'change_magnitude': 0.0,
'change_frequency': 0.0
}

if data.size > 5:
try:
# Simple change detection using differences
diffs = np.diff(data.flatten())
change_threshold = np.std(diffs) * 2

change_points = np.where(np.abs(diffs) > change_threshold)[0].tolist()

change_info['change_points'] = change_points
change_info['change_magnitude'] = float(np.mean(np.abs(diffs[diffs != 0]))) if len(diffs[diffs != 0]) > 0 else 0.0
change_info['change_frequency'] = len(change_points) / len(diffs)

except:
pass

return change_info
