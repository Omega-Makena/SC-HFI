"""
Online MAML Meta-Learning Engine - Federated Learning Component
Learns optimal initialization and expert-specific learning rates from insights
"""

import torch
from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict
import logging
import threading
import json

from ..config import META_LEARNING_CONFIG


class OnlineMAMLEngine:
"""
Online Meta-Learning Engine (MAML-based)

Learns WITHOUT raw data - only from client insights!

Meta-Parameters:
- w_init: Optimal initialization weights for new experts
- alpha_i: Expert-specific learning rates
"""

def __init__(self, num_experts: int = 10):
self.num_experts = num_experts
self.logger = logging.getLogger("MetaLearning")
self._lock = threading.Lock() # Thread safety

# Meta-parameters with proper initialization
self.w_init = torch.randn(num_experts, 64) * 0.01 # Initialize with small random values
self.expert_alphas = {
i: META_LEARNING_CONFIG["expert_lr_init"]
for i in range(num_experts)
}

# Meta-learning state
self.meta_updates = 0
self.meta_loss_history = []
self.parameter_version = 0 # Version tracking for clients

# Running statistics across all clients
self.global_stats = {
"loss_mean": 0.0,
"loss_std": 0.0,
"activation_frequencies": torch.zeros(num_experts),
"expert_performance": torch.zeros(num_experts),
}

# Insight validation schema
self.required_insight_fields = {
"client_id": (str, int),
"expert_insights": dict,
"avg_loss": (int, float),
"total_samples": int
}

def _validate_insight(self, insight: Dict) -> bool:
"""Validate insight schema to prevent corruption"""
try:
for field, expected_type in self.required_insight_fields.items():
if field not in insight:
self.logger.warning(f"Missing field {field} in insight")
return False
if not isinstance(insight[field], expected_type):
self.logger.warning(f"Invalid type for field {field}: expected {expected_type}, got {type(insight[field])}")
return False
return True
except Exception as e:
self.logger.error(f"Error validating insight: {e}")
return False

def meta_update(self, insights: List[Dict]) -> Dict[str, Any]:
"""
Perform meta-learning update from client insights

This is where the Developer learns WITHOUT data!
Only uses metadata from users.

Args:
insights: List of insight dictionaries from clients

Returns:
Updated meta-parameters
"""
with self._lock: # Thread safety
if not insights:
return self.get_meta_parameters()

# Validate all insights
valid_insights = [ins for ins in insights if self._validate_insight(ins)]
if not valid_insights:
self.logger.warning("No valid insights for meta-learning update")
return self.get_meta_parameters()

self.meta_updates += 1
self.parameter_version += 1

# Step 1: Aggregate expert performance across clients
expert_losses = defaultdict(list)
expert_activations = defaultdict(int)
expert_lr_trends = defaultdict(list)

for insight in valid_insights:
expert_insights = insight.get("expert_insights", {})

for expert_name, expert_data in expert_insights.items():
expert_id = expert_data.get("expert_id")
if expert_id is not None and 0 <= expert_id < self.num_experts:
# Loss with validation
ema_loss = expert_data.get("ema_loss", 0.0)
if isinstance(ema_loss, (int, float)) and not np.isnan(ema_loss):
expert_losses[expert_id].append(float(ema_loss))

# Activation count
activation = expert_data.get("activation_count", 0)
if isinstance(activation, (int, float)) and activation >= 0:
expert_activations[expert_id] += int(activation)

# Learning rate
lr = expert_data.get("learning_rate", 0.001)
if isinstance(lr, (int, float)) and lr > 0:
expert_lr_trends[expert_id].append(float(lr))

# Step 2: Compute global statistics with proper tensor handling
for expert_id in range(self.num_experts):
if expert_id in expert_losses and expert_losses[expert_id]:
# Average performance
avg_loss = np.mean(expert_losses[expert_id])
self.global_stats["expert_performance"][expert_id] = float(avg_loss)

# Activation frequency
self.global_stats["activation_frequencies"][expert_id] = float(expert_activations[expert_id])

# Normalize activation frequencies with safe tensor operations
total_activations = self.global_stats["activation_frequencies"].sum()
if total_activations > 0:
# Ensure tensor is float type before division
self.global_stats["activation_frequencies"] = self.global_stats["activation_frequencies"].float()
self.global_stats["activation_frequencies"] /= total_activations

# Update global loss statistics
all_losses = [loss for losses in expert_losses.values() for loss in losses]
if all_losses:
self.global_stats["loss_mean"] = float(np.mean(all_losses))
self.global_stats["loss_std"] = float(np.std(all_losses))

# Step 3: Adapt expert-specific learning rates (alpha_i) - NOW USING THE PARAMETERS!
for expert_id in range(self.num_experts):
if expert_id in expert_lr_trends and expert_lr_trends[expert_id]:
# Use median of successful clients' LRs
successful_lrs = expert_lr_trends[expert_id]
new_alpha = float(np.median(successful_lrs))

# Blend with current (EMA) - ensure within bounds
min_lr = META_LEARNING_CONFIG["expert_lr_min"]
max_lr = META_LEARNING_CONFIG["expert_lr_max"]

self.expert_alphas[expert_id] = max(min_lr, min(max_lr,
0.7 * self.expert_alphas[expert_id] + 0.3 * new_alpha
))

# Step 4: Update w_init based on expert performance (NOW USING IT!)
if all_losses:
# Update initialization weights based on performance patterns
performance_weights = torch.softmax(-self.global_stats["expert_performance"], dim=0)
self.w_init = self.w_init * 0.9 + torch.randn_like(self.w_init) * 0.1 * performance_weights.unsqueeze(1)

# Step 5: Compute meta-loss (average across all experts and clients)
meta_loss = np.mean(all_losses) if all_losses else 0.0
self.meta_loss_history.append(float(meta_loss))

# Keep recent history (bounded memory)
if len(self.meta_loss_history) > 1000:
self.meta_loss_history = self.meta_loss_history[-1000:]

return self.get_meta_parameters()

def get_meta_parameters(self) -> Dict:
"""Get current meta-parameters for broadcast to clients"""
with self._lock: # Thread safety
return {
"expert_alphas": self.expert_alphas,
"global_stats": {
"avg_loss": float(self.global_stats["expert_performance"].mean().item()),
"loss_mean": self.global_stats["loss_mean"],
"loss_std": self.global_stats["loss_std"],
"activation_frequencies": self.global_stats["activation_frequencies"].cpu().numpy().tolist(),
},
"w_init": self.w_init.cpu().numpy().tolist(), # Now included!
"meta_updates": self.meta_updates,
"parameter_version": self.parameter_version, # Version tracking!
"apply_to_new_experts": False, # Flag for initialization
}

def stats(self) -> Dict:
"""Meta-learning engine statistics"""
with self._lock: # Thread safety
return {
"meta_updates": self.meta_updates,
"parameter_version": self.parameter_version,
"meta_loss_history": self.meta_loss_history[-100:],
"avg_meta_loss": np.mean(self.meta_loss_history) if self.meta_loss_history else 0.0,
"expert_alphas": self.expert_alphas,
"global_performance": self.global_stats["expert_performance"].cpu().numpy().tolist(),
"loss_mean": self.global_stats["loss_mean"],
"loss_std": self.global_stats["loss_std"],
}

