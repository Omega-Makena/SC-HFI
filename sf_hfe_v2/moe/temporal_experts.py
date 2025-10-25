"""
Temporal Experts - Change & Memory Layer
Implements 4 core experts for understanding temporal patterns, seasonality, drift, and time-based causality
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict, deque
from scipy import stats
from scipy.signal import find_peaks, periodogram
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import time

from .base_expert import BaseExpert


class TrendExpert(BaseExpert):
"""
Expert 9: Trend Expert
Detects growth/decline patterns and long-term trends
"""

def __init__(self, expert_id: int = 9, config: Dict[str, Any] = None):
super().__init__(
expert_id=expert_id,
name="TrendExpert",
relation_types=["temporal"],
config=config
)

self.preferred_domains = ["time_series", "temporal", "trend_analysis"]
self.preferred_data_types = ["float64", "int64"]
self.preferred_tasks = ["trend_detection", "growth_analysis", "long_term_patterns"]

# Trend detection parameters
self.trend_threshold = self.config.get('trend_threshold', 0.1)
self.min_trend_length = self.config.get('min_trend_length', 10)
self.trend_methods = self.config.get('trend_methods', ['linear', 'polynomial', 'exponential'])

# Pattern storage
self.trend_patterns = defaultdict(list)
self.growth_patterns = defaultdict(list)

def _initialize_expert(self):
"""Initialize trend detection components"""
self.linear_trend_detector = LinearTrendDetector()
self.polynomial_trend_detector = PolynomialTrendDetector()
self.exponential_trend_detector = ExponentialTrendDetector()
self.trend_analyzer = TrendAnalyzer()

def process_data(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
"""Process data to detect trends and growth patterns"""
metadata = metadata or {}

# Detect linear trends
linear_trends = self.linear_trend_detector.detect_linear_trends(data)

# Detect polynomial trends
polynomial_trends = self.polynomial_trend_detector.detect_polynomial_trends(data)

# Detect exponential trends
exponential_trends = self.exponential_trend_detector.detect_exponential_trends(data)

# Analyze overall trend patterns
trend_analysis = self.trend_analyzer.analyze_trends(data, linear_trends, polynomial_trends, exponential_trends)

# Compute confidence
confidence = self._compute_trend_confidence(linear_trends, polynomial_trends, exponential_trends, trend_analysis)

result = {
'expert_id': self.expert_id,
'expert_name': self.name,
'linear_trends': linear_trends,
'polynomial_trends': polynomial_trends,
'exponential_trends': exponential_trends,
'trend_analysis': trend_analysis,
'confidence': confidence,
'relation_types': self.relation_types,
'timestamp': time.time()
}

return result

def update_online(self, data: np.ndarray, feedback: Dict[str, Any] = None) -> bool:
"""Update trend patterns based on new data"""
try:
# Extract trend features
trend_features = self.trend_analyzer.extract_features(data)

# Update patterns
self.trend_patterns['recent'].append(trend_features)

# Adapt thresholds based on feedback
if feedback and 'trend_accuracy' in feedback:
accuracy = feedback['trend_accuracy']
if accuracy > 0.8:
self.trend_threshold *= 1.01
elif accuracy < 0.6:
self.trend_threshold *= 0.99

self.trend_threshold = max(0.01, min(0.5, self.trend_threshold))

# Store in memory
self.store_memory(data, {'trend_features': trend_features}, feedback)

return True

except Exception as e:
self.logger.error(f"Error updating TrendExpert: {e}")
return False

def _compute_trend_confidence(self, linear_trends: Dict, polynomial_trends: Dict, 
exponential_trends: Dict, trend_analysis: Dict) -> float:
"""Compute confidence in trend detection"""
confidence = 0.7 # Base confidence

# Trend strength
trend_strength = trend_analysis.get('strength_score', 0)
confidence += trend_strength * 0.2

# Trend consistency
trend_consistency = trend_analysis.get('consistency_score', 0)
confidence += trend_consistency * 0.1

return min(1.0, confidence)


class DriftExpert(BaseExpert):
"""
Expert 10: Drift Expert
Detects distribution shift and concept drift over time
"""

def __init__(self, expert_id: int = 10, config: Dict[str, Any] = None):
super().__init__(
expert_id=expert_id,
name="DriftExpert",
relation_types=["temporal", "statistical"],
config=config
)

self.preferred_domains = ["concept_drift", "distribution_shift", "temporal"]
self.preferred_data_types = ["float64", "int64"]
self.preferred_tasks = ["drift_detection", "concept_drift", "distribution_shift"]

# Drift detection parameters
self.drift_threshold = self.config.get('drift_threshold', 0.05)
self.window_size = self.config.get('window_size', 100)
self.drift_methods = self.config.get('drift_methods', ['ks_test', 'wasserstein', 'kl_divergence'])

# Pattern storage
self.drift_patterns = defaultdict(list)
self.shift_patterns = defaultdict(list)

def _initialize_expert(self):
"""Initialize drift detection components"""
self.ks_drift_detector = KSDriftDetector()
self.wasserstein_drift_detector = WassersteinDriftDetector()
self.kl_drift_detector = KLDriftDetector()
self.drift_analyzer = DriftAnalyzer()

def process_data(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
"""Process data to detect drift and distribution shifts"""
metadata = metadata or {}

# Detect drift using KS test
ks_drift = self.ks_drift_detector.detect_drift(data)

# Detect drift using Wasserstein distance
wasserstein_drift = self.wasserstein_drift_detector.detect_drift(data)

# Detect drift using KL divergence
kl_drift = self.kl_drift_detector.detect_drift(data)

# Analyze overall drift patterns
drift_analysis = self.drift_analyzer.analyze_drift(data, ks_drift, wasserstein_drift, kl_drift)

# Compute confidence
confidence = self._compute_drift_confidence(ks_drift, wasserstein_drift, kl_drift, drift_analysis)

result = {
'expert_id': self.expert_id,
'expert_name': self.name,
'ks_drift': ks_drift,
'wasserstein_drift': wasserstein_drift,
'kl_drift': kl_drift,
'drift_analysis': drift_analysis,
'confidence': confidence,
'relation_types': self.relation_types,
'timestamp': time.time()
}

return result

def update_online(self, data: np.ndarray, feedback: Dict[str, Any] = None) -> bool:
"""Update drift patterns based on new data"""
try:
# Extract drift features
drift_features = self.drift_analyzer.extract_features(data)

# Update patterns
self.drift_patterns['recent'].append(drift_features)

# Adapt thresholds based on feedback
if feedback and 'drift_accuracy' in feedback:
accuracy = feedback['drift_accuracy']
if accuracy > 0.8:
self.drift_threshold *= 1.01
elif accuracy < 0.6:
self.drift_threshold *= 0.99

self.drift_threshold = max(0.01, min(0.2, self.drift_threshold))

# Store in memory
self.store_memory(data, {'drift_features': drift_features}, feedback)

return True

except Exception as e:
self.logger.error(f"Error updating DriftExpert: {e}")
return False

def _compute_drift_confidence(self, ks_drift: Dict, wasserstein_drift: Dict, 
kl_drift: Dict, drift_analysis: Dict) -> float:
"""Compute confidence in drift detection"""
confidence = 0.6 # Base confidence

# Drift detection strength
drift_strength = drift_analysis.get('strength_score', 0)
confidence += drift_strength * 0.2

# Drift consistency across methods
drift_consistency = drift_analysis.get('consistency_score', 0)
confidence += drift_consistency * 0.2

return min(1.0, confidence)


class CyclicExpert(BaseExpert):
"""
Expert 11: Cyclic Expert
Detects seasonality and recurrence patterns
"""

def __init__(self, expert_id: int = 11, config: Dict[str, Any] = None):
super().__init__(
expert_id=expert_id,
name="CyclicExpert",
relation_types=["temporal"],
config=config
)

self.preferred_domains = ["seasonal", "cyclical", "periodic", "temporal"]
self.preferred_data_types = ["float64", "int64"]
self.preferred_tasks = ["seasonality_detection", "cyclical_patterns", "periodicity_analysis"]

# Cyclic detection parameters
self.min_period = self.config.get('min_period', 2)
self.max_period = self.config.get('max_period', 100)
self.cyclic_threshold = self.config.get('cyclic_threshold', 0.3)

# Pattern storage
self.cyclic_patterns = defaultdict(list)
self.seasonal_patterns = defaultdict(list)

def _initialize_expert(self):
"""Initialize cyclic detection components"""
self.periodogram_analyzer = PeriodogramAnalyzer()
self.autocorr_analyzer = AutocorrAnalyzer()
self.seasonal_decomposer = SeasonalDecomposer()
self.cyclic_analyzer = CyclicAnalyzer()

def process_data(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
"""Process data to detect cyclic and seasonal patterns"""
metadata = metadata or {}

# Analyze periodogram
periodogram_info = self.periodogram_analyzer.analyze_periodogram(data)

# Analyze autocorrelation
autocorr_info = self.autocorr_analyzer.analyze_autocorrelation(data)

# Decompose seasonal components
seasonal_info = self.seasonal_decomposer.decompose_seasonal(data)

# Analyze overall cyclic patterns
cyclic_analysis = self.cyclic_analyzer.analyze_cyclic_patterns(data, periodogram_info, autocorr_info, seasonal_info)

# Compute confidence
confidence = self._compute_cyclic_confidence(periodogram_info, autocorr_info, seasonal_info, cyclic_analysis)

result = {
'expert_id': self.expert_id,
'expert_name': self.name,
'periodogram_info': periodogram_info,
'autocorr_info': autocorr_info,
'seasonal_info': seasonal_info,
'cyclic_analysis': cyclic_analysis,
'confidence': confidence,
'relation_types': self.relation_types,
'timestamp': time.time()
}

return result

def update_online(self, data: np.ndarray, feedback: Dict[str, Any] = None) -> bool:
"""Update cyclic patterns based on new data"""
try:
# Extract cyclic features
cyclic_features = self.cyclic_analyzer.extract_features(data)

# Update patterns
self.cyclic_patterns['recent'].append(cyclic_features)

# Adapt thresholds based on feedback
if feedback and 'cyclic_accuracy' in feedback:
accuracy = feedback['cyclic_accuracy']
if accuracy > 0.8:
self.cyclic_threshold *= 1.01
elif accuracy < 0.6:
self.cyclic_threshold *= 0.99

self.cyclic_threshold = max(0.1, min(0.8, self.cyclic_threshold))

# Store in memory
self.store_memory(data, {'cyclic_features': cyclic_features}, feedback)

return True

except Exception as e:
self.logger.error(f"Error updating CyclicExpert: {e}")
return False

def _compute_cyclic_confidence(self, periodogram_info: Dict, autocorr_info: Dict, 
seasonal_info: Dict, cyclic_analysis: Dict) -> float:
"""Compute confidence in cyclic detection"""
confidence = 0.6 # Base confidence

# Periodogram strength
periodogram_strength = periodogram_info.get('strength_score', 0)
confidence += periodogram_strength * 0.2

# Autocorrelation strength
autocorr_strength = autocorr_info.get('strength_score', 0)
confidence += autocorr_strength * 0.1

# Seasonal strength
seasonal_strength = seasonal_info.get('strength_score', 0)
confidence += seasonal_strength * 0.1

return min(1.0, confidence)


class TemporalCausalityExpert(BaseExpert):
"""
Expert 12: Temporal Causality Expert
Performs lagged causal reasoning and temporal dependency analysis
"""

def __init__(self, expert_id: int = 12, config: Dict[str, Any] = None):
super().__init__(
expert_id=expert_id,
name="TemporalCausalityExpert",
relation_types=["temporal", "causal"],
config=config
)

self.preferred_domains = ["causal", "temporal", "lagged", "time_series"]
self.preferred_data_types = ["float64", "int64"]
self.preferred_tasks = ["temporal_causality", "lagged_analysis", "causal_inference"]

# Causality parameters
self.max_lag = self.config.get('max_lag', 10)
self.causality_threshold = self.config.get('causality_threshold', 0.1)
self.causality_methods = self.config.get('causality_methods', ['granger', 'transfer_entropy', 'cross_correlation'])

# Pattern storage
self.causality_patterns = defaultdict(list)
self.lag_patterns = defaultdict(list)

def _initialize_expert(self):
"""Initialize temporal causality components"""
self.granger_causality = GrangerCausality()
self.transfer_entropy = TransferEntropy()
self.cross_correlation = CrossCorrelation()
self.temporal_causality_analyzer = TemporalCausalityAnalyzer()

def process_data(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
"""Process data to detect temporal causality and lagged dependencies"""
metadata = metadata or {}

# Detect Granger causality
granger_causality = self.granger_causality.detect_causality(data)

# Detect transfer entropy
transfer_entropy = self.transfer_entropy.detect_causality(data)

# Detect cross-correlation
cross_corr = self.cross_correlation.detect_causality(data)

# Analyze overall temporal causality
causality_analysis = self.temporal_causality_analyzer.analyze_causality(data, granger_causality, transfer_entropy, cross_corr)

# Compute confidence
confidence = self._compute_causality_confidence(granger_causality, transfer_entropy, cross_corr, causality_analysis)

result = {
'expert_id': self.expert_id,
'expert_name': self.name,
'granger_causality': granger_causality,
'transfer_entropy': transfer_entropy,
'cross_correlation': cross_corr,
'causality_analysis': causality_analysis,
'confidence': confidence,
'relation_types': self.relation_types,
'timestamp': time.time()
}

return result

def update_online(self, data: np.ndarray, feedback: Dict[str, Any] = None) -> bool:
"""Update temporal causality patterns based on new data"""
try:
# Extract causality features
causality_features = self.temporal_causality_analyzer.extract_features(data)

# Update patterns
self.causality_patterns['recent'].append(causality_features)

# Adapt thresholds based on feedback
if feedback and 'causality_accuracy' in feedback:
accuracy = feedback['causality_accuracy']
if accuracy > 0.8:
self.causality_threshold *= 1.01
elif accuracy < 0.6:
self.causality_threshold *= 0.99

self.causality_threshold = max(0.01, min(0.3, self.causality_threshold))

# Store in memory
self.store_memory(data, {'causality_features': causality_features}, feedback)

return True

except Exception as e:
self.logger.error(f"Error updating TemporalCausalityExpert: {e}")
return False

def _compute_causality_confidence(self, granger_causality: Dict, transfer_entropy: Dict, 
cross_corr: Dict, causality_analysis: Dict) -> float:
"""Compute confidence in temporal causality detection"""
confidence = 0.5 # Base confidence for causality

# Causality strength
causality_strength = causality_analysis.get('strength_score', 0)
confidence += causality_strength * 0.3

# Method consistency
method_consistency = causality_analysis.get('consistency_score', 0)
confidence += method_consistency * 0.2

return min(1.0, confidence)


# Helper classes for temporal experts

class LinearTrendDetector:
"""Detects linear trends in data"""

def detect_linear_trends(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect linear trends"""
trend_info = {
'has_trend': False,
'trend_slope': 0.0,
'trend_strength': 0.0,
'r_squared': 0.0
}

if data.size > 2:
try:
# Fit linear regression
x = np.arange(len(data.flatten()))
y = data.flatten()

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

trend_info['trend_slope'] = slope
trend_info['r_squared'] = r_value ** 2
trend_info['trend_strength'] = abs(slope) / np.std(y) if np.std(y) > 0 else 0.0
trend_info['has_trend'] = abs(slope) > 0.01 and p_value < 0.05

except:
pass

return trend_info


class PolynomialTrendDetector:
"""Detects polynomial trends in data"""

def detect_polynomial_trends(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect polynomial trends"""
trend_info = {
'has_trend': False,
'polynomial_degree': 0,
'trend_strength': 0.0,
'r_squared': 0.0
}

if data.size > 5:
try:
x = np.arange(len(data.flatten()))
y = data.flatten()

# Try different polynomial degrees
best_r_squared = 0.0
best_degree = 0

for degree in range(1, min(4, len(x) - 1)):
coeffs = np.polyfit(x, y, degree)
poly_func = np.poly1d(coeffs)
y_pred = poly_func(x)
r_squared = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

if r_squared > best_r_squared:
best_r_squared = r_squared
best_degree = degree

trend_info['polynomial_degree'] = best_degree
trend_info['r_squared'] = best_r_squared
trend_info['trend_strength'] = best_r_squared
trend_info['has_trend'] = best_r_squared > 0.5

except:
pass

return trend_info


class ExponentialTrendDetector:
"""Detects exponential trends in data"""

def detect_exponential_trends(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect exponential trends"""
trend_info = {
'has_trend': False,
'exponential_rate': 0.0,
'trend_strength': 0.0,
'r_squared': 0.0
}

if data.size > 3:
try:
x = np.arange(len(data.flatten()))
y = data.flatten()

# Check for exponential trend (log-linear)
if np.all(y > 0):
log_y = np.log(y)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_y)

trend_info['exponential_rate'] = slope
trend_info['r_squared'] = r_value ** 2
trend_info['trend_strength'] = abs(slope)
trend_info['has_trend'] = abs(slope) > 0.01 and p_value < 0.05

except:
pass

return trend_info


class TrendAnalyzer:
"""Analyzes overall trend patterns"""

def analyze_trends(self, data: np.ndarray, linear_trends: Dict, polynomial_trends: Dict, exponential_trends: Dict) -> Dict[str, Any]:
"""Analyze overall trend patterns"""
analysis = {
'strength_score': 0.0,
'consistency_score': 0.0,
'trend_type': 'none',
'trend_direction': 'stable'
}

# Compute strength score
strengths = []
if linear_trends.get('has_trend'):
strengths.append(linear_trends.get('trend_strength', 0))
if polynomial_trends.get('has_trend'):
strengths.append(polynomial_trends.get('trend_strength', 0))
if exponential_trends.get('has_trend'):
strengths.append(exponential_trends.get('trend_strength', 0))

if strengths:
analysis['strength_score'] = np.mean(strengths)

# Determine trend type
if exponential_trends.get('has_trend'):
analysis['trend_type'] = 'exponential'
elif polynomial_trends.get('has_trend'):
analysis['trend_type'] = 'polynomial'
elif linear_trends.get('has_trend'):
analysis['trend_type'] = 'linear'

# Determine trend direction
if linear_trends.get('trend_slope', 0) > 0:
analysis['trend_direction'] = 'increasing'
elif linear_trends.get('trend_slope', 0) < 0:
analysis['trend_direction'] = 'decreasing'

# Compute consistency score
trend_counts = sum([
linear_trends.get('has_trend', False),
polynomial_trends.get('has_trend', False),
exponential_trends.get('has_trend', False)
])
analysis['consistency_score'] = trend_counts / 3.0

return analysis

def extract_features(self, data: np.ndarray) -> Dict[str, Any]:
"""Extract trend features"""
if data.size > 2:
try:
x = np.arange(len(data.flatten()))
y = data.flatten()

# Linear trend features
slope, _, r_value, _, _ = stats.linregress(x, y)

return {
'linear_slope': float(slope),
'linear_r_squared': float(r_value ** 2),
'data_range': float(np.max(y) - np.min(y)),
'data_variance': float(np.var(y))
}
except:
return {'linear_slope': 0.0, 'linear_r_squared': 0.0, 'data_range': 0.0, 'data_variance': 0.0}
else:
return {'linear_slope': 0.0, 'linear_r_squared': 0.0, 'data_range': 0.0, 'data_variance': 0.0}


class KSDriftDetector:
"""Detects drift using Kolmogorov-Smirnov test"""

def detect_drift(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect drift using KS test"""
drift_info = {
'has_drift': False,
'drift_strength': 0.0,
'p_value': 1.0,
'ks_statistic': 0.0
}

if data.size > 20:
try:
# Split data into two halves
mid_point = data.size // 2
first_half = data.flatten()[:mid_point]
second_half = data.flatten()[mid_point:]

# Perform KS test
ks_statistic, p_value = stats.ks_2samp(first_half, second_half)

drift_info['ks_statistic'] = ks_statistic
drift_info['p_value'] = p_value
drift_info['drift_strength'] = ks_statistic
drift_info['has_drift'] = p_value < 0.05

except:
pass

return drift_info


class WassersteinDriftDetector:
"""Detects drift using Wasserstein distance"""

def detect_drift(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect drift using Wasserstein distance"""
drift_info = {
'has_drift': False,
'drift_strength': 0.0,
'wasserstein_distance': 0.0
}

if data.size > 20:
try:
# Split data into two halves
mid_point = data.size // 2
first_half = data.flatten()[:mid_point]
second_half = data.flatten()[mid_point:]

# Compute Wasserstein distance
from scipy.stats import wasserstein_distance
wd = wasserstein_distance(first_half, second_half)

drift_info['wasserstein_distance'] = wd
drift_info['drift_strength'] = wd / np.std(data.flatten()) if np.std(data.flatten()) > 0 else 0.0
drift_info['has_drift'] = wd > np.std(data.flatten()) * 0.1

except:
pass

return drift_info


class KLDriftDetector:
"""Detects drift using KL divergence"""

def detect_drift(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect drift using KL divergence"""
drift_info = {
'has_drift': False,
'drift_strength': 0.0,
'kl_divergence': 0.0
}

if data.size > 20:
try:
# Split data into two halves
mid_point = data.size // 2
first_half = data.flatten()[:mid_point]
second_half = data.flatten()[mid_point:]

# Compute KL divergence
from scipy.stats import entropy

# Create histograms
bins = np.linspace(np.min(data.flatten()), np.max(data.flatten()), 20)
hist1, _ = np.histogram(first_half, bins=bins)
hist2, _ = np.histogram(second_half, bins=bins)

# Normalize histograms
hist1 = hist1 / np.sum(hist1)
hist2 = hist2 / np.sum(hist2)

# Compute KL divergence
kl_div = entropy(hist1 + 1e-10, hist2 + 1e-10)

drift_info['kl_divergence'] = kl_div
drift_info['drift_strength'] = kl_div
drift_info['has_drift'] = kl_div > 0.1

except:
pass

return drift_info


class DriftAnalyzer:
"""Analyzes overall drift patterns"""

def analyze_drift(self, data: np.ndarray, ks_drift: Dict, wasserstein_drift: Dict, kl_drift: Dict) -> Dict[str, Any]:
"""Analyze overall drift patterns"""
analysis = {
'strength_score': 0.0,
'consistency_score': 0.0,
'drift_methods': []
}

# Collect drift detection results
drift_results = []
if ks_drift.get('has_drift'):
drift_results.append(ks_drift.get('drift_strength', 0))
analysis['drift_methods'].append('ks_test')
if wasserstein_drift.get('has_drift'):
drift_results.append(wasserstein_drift.get('drift_strength', 0))
analysis['drift_methods'].append('wasserstein')
if kl_drift.get('has_drift'):
drift_results.append(kl_drift.get('drift_strength', 0))
analysis['drift_methods'].append('kl_divergence')

if drift_results:
analysis['strength_score'] = np.mean(drift_results)
analysis['consistency_score'] = len(drift_results) / 3.0

return analysis

def extract_features(self, data: np.ndarray) -> Dict[str, Any]:
"""Extract drift features"""
if data.size > 20:
try:
mid_point = data.size // 2
first_half = data.flatten()[:mid_point]
second_half = data.flatten()[mid_point:]

return {
'mean_difference': float(np.mean(second_half) - np.mean(first_half)),
'std_difference': float(np.std(second_half) - np.std(first_half)),
'variance_ratio': float(np.var(second_half) / np.var(first_half)) if np.var(first_half) > 0 else 1.0
}
except:
return {'mean_difference': 0.0, 'std_difference': 0.0, 'variance_ratio': 1.0}
else:
return {'mean_difference': 0.0, 'std_difference': 0.0, 'variance_ratio': 1.0}


class PeriodogramAnalyzer:
"""Analyzes periodogram for cyclic patterns"""

def analyze_periodogram(self, data: np.ndarray) -> Dict[str, Any]:
"""Analyze periodogram"""
periodogram_info = {
'dominant_frequencies': [],
'strength_score': 0.0,
'periodogram': None
}

if data.size > 10:
try:
# Compute periodogram
frequencies, power = periodogram(data.flatten())

# Find dominant frequencies
peaks, _ = find_peaks(power, height=np.mean(power))

if len(peaks) > 0:
# Get top frequencies
top_peaks = peaks[np.argsort(power[peaks])[-3:]]
dominant_freqs = frequencies[top_peaks]
periodogram_info['dominant_frequencies'] = dominant_freqs.tolist()
periodogram_info['strength_score'] = np.max(power) / np.mean(power)

periodogram_info['periodogram'] = {
'frequencies': frequencies.tolist(),
'power': power.tolist()
}

except:
pass

return periodogram_info


class AutocorrAnalyzer:
"""Analyzes autocorrelation for cyclic patterns"""

def analyze_autocorrelation(self, data: np.ndarray) -> Dict[str, Any]:
"""Analyze autocorrelation"""
autocorr_info = {
'autocorr_lags': [],
'strength_score': 0.0,
'max_autocorr': 0.0
}

if data.size > 10:
try:
# Compute autocorrelation
autocorr = np.correlate(data.flatten(), data.flatten(), mode='full')
autocorr = autocorr[autocorr.size // 2:]
autocorr = autocorr / autocorr[0] # Normalize

# Find significant lags
threshold = 0.3
significant_lags = np.where(autocorr > threshold)[0]

autocorr_info['autocorr_lags'] = significant_lags.tolist()
autocorr_info['max_autocorr'] = float(np.max(autocorr[1:])) # Exclude lag 0
autocorr_info['strength_score'] = len(significant_lags) / len(autocorr)

except:
pass

return autocorr_info


class SeasonalDecomposer:
"""Decomposes seasonal components"""

def decompose_seasonal(self, data: np.ndarray) -> Dict[str, Any]:
"""Decompose seasonal components"""
seasonal_info = {
'has_seasonality': False,
'seasonal_strength': 0.0,
'seasonal_period': 0,
'strength_score': 0.0
}

if data.size > 20:
try:
# Simple seasonal decomposition
# Assume potential seasonal periods
periods = [2, 3, 4, 5, 7, 12, 24] # Common seasonal periods

best_period = 0
best_strength = 0.0

for period in periods:
if data.size >= period * 2:
# Compute seasonal strength
seasonal_strength = self._compute_seasonal_strength(data.flatten(), period)
if seasonal_strength > best_strength:
best_strength = seasonal_strength
best_period = period

seasonal_info['seasonal_period'] = best_period
seasonal_info['seasonal_strength'] = best_strength
seasonal_info['strength_score'] = best_strength
seasonal_info['has_seasonality'] = best_strength > 0.3

except:
pass

return seasonal_info

def _compute_seasonal_strength(self, data: np.ndarray, period: int) -> float:
"""Compute seasonal strength for given period"""
try:
# Reshape data into seasonal periods
n_periods = len(data) // period
if n_periods < 2:
return 0.0

seasonal_data = data[:n_periods * period].reshape(n_periods, period)

# Compute variance within seasons vs between seasons
within_season_var = np.mean(np.var(seasonal_data, axis=1))
between_season_var = np.var(np.mean(seasonal_data, axis=1))

if within_season_var > 0:
seasonal_strength = between_season_var / (within_season_var + between_season_var)
else:
seasonal_strength = 0.0

return seasonal_strength
except:
return 0.0


class CyclicAnalyzer:
"""Analyzes overall cyclic patterns"""

def analyze_cyclic_patterns(self, data: np.ndarray, periodogram_info: Dict, autocorr_info: Dict, seasonal_info: Dict) -> Dict[str, Any]:
"""Analyze overall cyclic patterns"""
analysis = {
'has_cyclicity': False,
'cyclic_strength': 0.0,
'dominant_period': 0,
'cyclic_methods': []
}

# Collect cyclic detection results
cyclic_strengths = []

if periodogram_info.get('strength_score', 0) > 0.5:
cyclic_strengths.append(periodogram_info.get('strength_score', 0))
analysis['cyclic_methods'].append('periodogram')

if autocorr_info.get('strength_score', 0) > 0.3:
cyclic_strengths.append(autocorr_info.get('strength_score', 0))
analysis['cyclic_methods'].append('autocorrelation')

if seasonal_info.get('has_seasonality'):
cyclic_strengths.append(seasonal_info.get('strength_score', 0))
analysis['cyclic_methods'].append('seasonal')
analysis['dominant_period'] = seasonal_info.get('seasonal_period', 0)

if cyclic_strengths:
analysis['cyclic_strength'] = np.mean(cyclic_strengths)
analysis['has_cyclicity'] = analysis['cyclic_strength'] > 0.3

return analysis

def extract_features(self, data: np.ndarray) -> Dict[str, Any]:
"""Extract cyclic features"""
if data.size > 10:
try:
# Compute basic cyclic features
autocorr = np.correlate(data.flatten(), data.flatten(), mode='full')
autocorr = autocorr[autocorr.size // 2:]
autocorr = autocorr / autocorr[0]

return {
'max_autocorr': float(np.max(autocorr[1:])),
'autocorr_decay': float(np.mean(np.diff(autocorr[:10]))),
'data_entropy': float(-np.sum(autocorr * np.log(autocorr + 1e-10)))
}
except:
return {'max_autocorr': 0.0, 'autocorr_decay': 0.0, 'data_entropy': 0.0}
else:
return {'max_autocorr': 0.0, 'autocorr_decay': 0.0, 'data_entropy': 0.0}


class GrangerCausality:
"""Detects Granger causality"""

def detect_causality(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect Granger causality"""
causality_info = {
'has_causality': False,
'causality_strength': 0.0,
'causal_pairs': [],
'lag_structure': {}
}

if data.ndim > 1 and data.shape[1] > 1 and data.shape[0] > 10:
try:
# Simple Granger causality test
causal_pairs = []

for i in range(data.shape[1]):
for j in range(data.shape[1]):
if i != j:
# Test if variable j Granger-causes variable i
causality_strength = self._test_granger_causality(data[:, i], data[:, j])

if causality_strength > 0.1: # Threshold
causal_pairs.append({
'cause': j,
'effect': i,
'strength': causality_strength
})

causality_info['causal_pairs'] = causal_pairs
causality_info['has_causality'] = len(causal_pairs) > 0
causality_info['causality_strength'] = np.mean([pair['strength'] for pair in causal_pairs]) if causal_pairs else 0.0

except:
pass

return causality_info

def _test_granger_causality(self, y: np.ndarray, x: np.ndarray, max_lag: int = 5) -> float:
"""Test Granger causality between two variables"""
try:
from sklearn.linear_model import LinearRegression

if len(y) <= max_lag + 1:
return 0.0

# Restricted model (only y's own lags)
restricted_model = LinearRegression()
restricted_X = np.column_stack([y[i:-max_lag+i] for i in range(1, max_lag+1)])
restricted_y = y[max_lag:]
restricted_model.fit(restricted_X, restricted_y)
restricted_rss = np.sum((restricted_y - restricted_model.predict(restricted_X)) ** 2)

# Unrestricted model (y's lags + x's lags)
unrestricted_model = LinearRegression()
unrestricted_X = np.column_stack([y[i:-max_lag+i] for i in range(1, max_lag+1)] + 
[x[i:-max_lag+i] for i in range(1, max_lag+1)])
unrestricted_y = y[max_lag:]
unrestricted_model.fit(unrestricted_X, unrestricted_y)
unrestricted_rss = np.sum((unrestricted_y - unrestricted_model.predict(unrestricted_X)) ** 2)

# F-test statistic
if restricted_rss > 0:
f_stat = ((restricted_rss - unrestricted_rss) / max_lag) / (unrestricted_rss / (len(unrestricted_y) - 2 * max_lag))
return min(1.0, f_stat / 10.0) # Normalize to [0, 1]
else:
return 0.0

except:
return 0.0


class TransferEntropy:
"""Detects causality using transfer entropy"""

def detect_causality(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect causality using transfer entropy"""
causality_info = {
'has_causality': False,
'causality_strength': 0.0,
'causal_pairs': [],
'transfer_entropy_matrix': None
}

if data.ndim > 1 and data.shape[1] > 1 and data.shape[0] > 10:
try:
# Compute transfer entropy matrix
te_matrix = np.zeros((data.shape[1], data.shape[1]))
causal_pairs = []

for i in range(data.shape[1]):
for j in range(data.shape[1]):
if i != j:
te_value = self._compute_transfer_entropy(data[:, i], data[:, j])
te_matrix[i, j] = te_value

if te_value > 0.1: # Threshold
causal_pairs.append({
'cause': j,
'effect': i,
'strength': te_value
})

causality_info['transfer_entropy_matrix'] = te_matrix.tolist()
causality_info['causal_pairs'] = causal_pairs
causality_info['has_causality'] = len(causal_pairs) > 0
causality_info['causality_strength'] = np.mean([pair['strength'] for pair in causal_pairs]) if causal_pairs else 0.0

except:
pass

return causality_info

def _compute_transfer_entropy(self, y: np.ndarray, x: np.ndarray, lag: int = 1) -> float:
"""Compute transfer entropy from x to y"""
try:
from sklearn.metrics import mutual_info_score

if len(y) <= lag:
return 0.0

# Discretize data
y_disc = self._discretize(y)
x_disc = self._discretize(x)

# Compute transfer entropy
y_lag = y_disc[lag:]
x_lag = x_disc[:-lag]
y_current = y_disc[lag:]

# TE(X->Y) = MI(Y_t, X_{t-lag} | Y_{t-lag})
# Simplified version: MI(Y_t, X_{t-lag})
te_value = mutual_info_score(y_current, x_lag)

return te_value
except:
return 0.0

def _discretize(self, data: np.ndarray, bins: int = 10) -> np.ndarray:
"""Discretize continuous data"""
try:
_, bin_edges = np.histogram(data, bins=bins)
return np.digitize(data, bin_edges[1:-1])
except:
return np.zeros_like(data, dtype=int)


class CrossCorrelation:
"""Detects causality using cross-correlation"""

def detect_causality(self, data: np.ndarray) -> Dict[str, Any]:
"""Detect causality using cross-correlation"""
causality_info = {
'has_causality': False,
'causality_strength': 0.0,
'causal_pairs': [],
'cross_corr_matrix': None
}

if data.ndim > 1 and data.shape[1] > 1:
try:
# Compute cross-correlation matrix
corr_matrix = np.corrcoef(data.T)
causal_pairs = []

for i in range(data.shape[1]):
for j in range(i+1, data.shape[1]):
corr_val = corr_matrix[i, j]
if abs(corr_val) > 0.3: # Threshold
causal_pairs.append({
'var1': i,
'var2': j,
'strength': abs(corr_val),
'direction': 'positive' if corr_val > 0 else 'negative'
})

causality_info['cross_corr_matrix'] = corr_matrix.tolist()
causality_info['causal_pairs'] = causal_pairs
causality_info['has_causality'] = len(causal_pairs) > 0
causality_info['causality_strength'] = np.mean([pair['strength'] for pair in causal_pairs]) if causal_pairs else 0.0

except:
pass

return causality_info


class TemporalCausalityAnalyzer:
"""Analyzes overall temporal causality patterns"""

def analyze_causality(self, data: np.ndarray, granger_causality: Dict, transfer_entropy: Dict, cross_corr: Dict) -> Dict[str, Any]:
"""Analyze overall temporal causality patterns"""
analysis = {
'strength_score': 0.0,
'consistency_score': 0.0,
'causality_methods': [],
'causal_strength': 0.0
}

# Collect causality detection results
causality_strengths = []

if granger_causality.get('has_causality'):
causality_strengths.append(granger_causality.get('causality_strength', 0))
analysis['causality_methods'].append('granger')

if transfer_entropy.get('has_causality'):
causality_strengths.append(transfer_entropy.get('causality_strength', 0))
analysis['causality_methods'].append('transfer_entropy')

if cross_corr.get('has_causality'):
causality_strengths.append(cross_corr.get('causality_strength', 0))
analysis['causality_methods'].append('cross_correlation')

if causality_strengths:
analysis['strength_score'] = np.mean(causality_strengths)
analysis['causal_strength'] = analysis['strength_score']
analysis['consistency_score'] = len(causality_strengths) / 3.0

return analysis

def extract_features(self, data: np.ndarray) -> Dict[str, Any]:
"""Extract temporal causality features"""
if data.ndim > 1 and data.shape[1] > 1:
try:
# Compute cross-correlation features
corr_matrix = np.corrcoef(data.T)
upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

return {
'max_cross_corr': float(np.max(np.abs(upper_tri))),
'mean_cross_corr': float(np.mean(np.abs(upper_tri))),
'corr_variance': float(np.var(upper_tri)),
'strong_corr_count': int(np.sum(np.abs(upper_tri) > 0.5))
}
except:
return {'max_cross_corr': 0.0, 'mean_cross_corr': 0.0, 'corr_variance': 0.0, 'strong_corr_count': 0}
else:
return {'max_cross_corr': 0.0, 'mean_cross_corr': 0.0, 'corr_variance': 0.0, 'strong_corr_count': 0}
