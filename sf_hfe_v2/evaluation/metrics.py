"""
Fairness and Evaluation Metrics - P1 Essential

Implements Healthcare Fairness Index (HFI) and per-client metrics
Reference: Arafat et al. - fairness metrics for federated learning
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import logging


def per_client_evaluation(
model,
client_test_loaders: List,
criterion=None
) -> Dict[str, any]:
"""
Evaluate model on each client's test set separately

Critical for fairness: Shows if any client is disadvantaged

Args:
model: Model to evaluate
client_test_loaders: List of test DataLoaders (one per client)
criterion: Loss function (default: MSELoss)

Returns:
Dictionary with per-client metrics
"""
if criterion is None:
criterion = torch.nn.MSELoss()

model.eval()

client_losses = []
client_accuracies = []
client_samples = []

with torch.no_grad():
for client_id, test_loader in enumerate(client_test_loaders):
total_loss = 0.0
total_samples = 0

for batch_x, batch_y in test_loader:
outputs = model(batch_x)
loss = criterion(outputs, batch_y)

total_loss += loss.item() * len(batch_x)
total_samples += len(batch_x)

avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')

client_losses.append(avg_loss)
client_samples.append(total_samples)

return {
"client_losses": client_losses,
"client_samples": client_samples,
"avg_loss": np.mean(client_losses),
"std_loss": np.std(client_losses),
"min_loss": np.min(client_losses),
"max_loss": np.max(client_losses),
"range_loss": np.max(client_losses) - np.min(client_losses),
}


def healthcare_fairness_index(client_losses: List[float], baseline_loss: float = 1.0) -> float:
"""
Compute Healthcare Fairness Index (HFI)

HFI measures fairness across clients (institutions/users)
Lower variance = more fair

Reference: Arafat et al. - Healthcare Fairness in Federated Learning

Formula: HFI = 1 - (normalized_variance_of_client_losses)

Args:
client_losses: List of loss values (one per client)
baseline_loss: Baseline loss for normalization

Returns:
HFI score (0-1, higher is more fair)
"""
if len(client_losses) == 0:
return 0.0

# Normalize losses
normalized_losses = [loss / baseline_loss for loss in client_losses]

# Compute variance
variance = np.var(normalized_losses)

# HFI: 1 means perfect fairness (all clients same loss)
# 0 means very unfair (high variance)
hfi = 1.0 / (1.0 + variance)

return float(hfi)


def compute_fairness_metrics(client_metrics: List[Dict]) -> Dict:
"""
Compute comprehensive fairness metrics

Args:
client_metrics: List of metric dicts (one per client)

Returns:
Aggregated fairness metrics
"""
# Extract losses
client_losses = [m.get("loss", float('inf')) for m in client_metrics]
client_samples = [m.get("samples", 0) for m in client_metrics]

# Basic statistics
mean_loss = np.mean(client_losses)
std_loss = np.std(client_losses)
min_loss = np.min(client_losses)
max_loss = np.max(client_losses)

# Healthcare Fairness Index
hfi = healthcare_fairness_index(client_losses, baseline_loss=mean_loss)

# Worst-case fairness (ratio of worst to average)
worst_case_ratio = max_loss / mean_loss if mean_loss > 0 else float('inf')

# Coefficient of variation (relative dispersion)
cv = std_loss / mean_loss if mean_loss > 0 else float('inf')

# Sample-weighted metrics
total_samples = sum(client_samples)
if total_samples > 0:
weighted_loss = sum(
client_losses[i] * client_samples[i] / total_samples
for i in range(len(client_losses))
)
else:
weighted_loss = mean_loss

return {
# Central tendency
"mean_loss": float(mean_loss),
"median_loss": float(np.median(client_losses)),
"weighted_loss": float(weighted_loss),

# Dispersion
"std_loss": float(std_loss),
"min_loss": float(min_loss),
"max_loss": float(max_loss),
"range_loss": float(max_loss - min_loss),

# Fairness indices
"hfi": float(hfi),
"worst_case_ratio": float(worst_case_ratio),
"coefficient_variation": float(cv),

# Distribution
"client_losses": client_losses,
"num_clients": len(client_losses),
}


def compare_to_baseline(
sc_hfi_metrics: Dict,
fedavg_metrics: Dict
) -> Dict:
"""
Compare SC-HFI performance to FedAvg baseline

Args:
sc_hfi_metrics: Metrics from SC-HFI
fedavg_metrics: Metrics from FedAvg baseline

Returns:
Comparison statistics
"""
comparison = {
"mean_loss_improvement": (
(fedavg_metrics["mean_loss"] - sc_hfi_metrics["mean_loss"]) / 
fedavg_metrics["mean_loss"] * 100
),
"fairness_improvement": (
sc_hfi_metrics["hfi"] - fedavg_metrics["hfi"]
) * 100,
"worst_case_improvement": (
(fedavg_metrics["worst_case_ratio"] - sc_hfi_metrics["worst_case_ratio"]) /
fedavg_metrics["worst_case_ratio"] * 100
),
"variance_reduction": (
(fedavg_metrics["std_loss"] - sc_hfi_metrics["std_loss"]) /
fedavg_metrics["std_loss"] * 100
),
}

return comparison

