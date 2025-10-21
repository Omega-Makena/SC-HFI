"""
Drift Detection Expert - Intelligence Expert #2
Specializes in detecting distribution shifts using KL divergence
"""

import torch
import torch.nn as nn
from typing import Dict
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_expert import BaseExpert


class DriftDetectionExpert(BaseExpert):
    """
    Expert specializing in concept drift detection
    
    Capabilities:
    - KL divergence monitoring
    - Distribution shift detection
    - Adaptive threshold setting
    - Drift magnitude estimation
    """
    
    def __init__(self, input_dim: int, output_dim: int, window_size: int = 100, **kwargs):
        super().__init__(
            expert_id=4,
            input_dim=input_dim,
            output_dim=output_dim,
            expert_type="intelligence",
            **kwargs
        )
        
        self.window_size = window_size
        
        # Reference distribution (from initial data)
        self.reference_mean = None
        self.reference_std = None
        self.reference_samples = []
        
        # Current window
        self.current_window = []
        
        # Drift tracking
        self.drift_scores = []
        self.drift_detected = False
        self.drift_timestamps = []
        self.last_drift_batch = 0
        
        # Adaptive threshold
        self.drift_threshold = 0.05  # KL divergence threshold
        self.threshold_history = []
        
    def _task_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Standard prediction loss
        Drift detection happens in monitoring, not in loss
        """
        return nn.functional.mse_loss(outputs, targets)
    
    def online_update(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        replay: bool = True
    ) -> Dict[str, float]:
        """
        Update with drift monitoring
        """
        # Standard update
        metrics = super().online_update(batch_x, batch_y, replay)
        
        # Monitor drift
        drift_score = self._monitor_drift(batch_x)
        
        # Check if drift detected
        if drift_score > self.drift_threshold:
            self._handle_drift()
        
        metrics.update({
            "drift_score": drift_score,
            "drift_detected": self.drift_detected,
            "batches_since_drift": self.batch_count - self.last_drift_batch,
        })
        
        return metrics
    
    def _monitor_drift(self, batch_x: torch.Tensor) -> float:
        """
        Monitor distribution drift using KL divergence
        """
        # Initialize reference distribution
        if self.reference_mean is None:
            self._initialize_reference(batch_x)
            return 0.0
        
        # Add to current window
        self.current_window.append(batch_x)
        if len(self.current_window) > self.window_size:
            self.current_window.pop(0)
        
        # Compute drift every 10 batches
        if len(self.current_window) < 10:
            return 0.0
        
        # Current distribution statistics
        current_data = torch.cat(self.current_window, dim=0)
        current_mean = current_data.mean(dim=0)
        current_std = current_data.std(dim=0) + 1e-8
        
        # Approximate KL divergence (Gaussian assumption)
        # KL(P||Q) = log(σ_Q/σ_P) + (σ_P² + (μ_P - μ_Q)²) / (2σ_Q²) - 1/2
        kl_div = (
            torch.log(current_std / self.reference_std) +
            (self.reference_std ** 2 + (self.reference_mean - current_mean) ** 2) / (2 * current_std ** 2) -
            0.5
        )
        
        # Average KL divergence across dimensions
        drift_score = float(kl_div.mean().item())
        
        # Track
        self.drift_scores.append(drift_score)
        if len(self.drift_scores) > 1000:
            self.drift_scores.pop(0)
        
        # Update adaptive threshold (95th percentile of recent scores)
        if len(self.drift_scores) > 50:
            self.drift_threshold = float(np.percentile(self.drift_scores[-50:], 95))
        
        return drift_score
    
    def _initialize_reference(self, batch_x: torch.Tensor):
        """
        Initialize reference distribution from first batch
        """
        self.reference_mean = batch_x.mean(dim=0)
        # Use unbiased=False to avoid division by zero for single sample
        self.reference_std = batch_x.std(dim=0, unbiased=False) + 1e-8
        self.reference_samples = [batch_x.clone()]
    
    def _handle_drift(self):
        """
        Handle detected drift
        """
        self.drift_detected = True
        self.drift_timestamps.append(self.batch_count)
        self.last_drift_batch = self.batch_count
        
        # Update reference distribution to current
        if len(self.current_window) > 0:
            current_data = torch.cat(self.current_window, dim=0)
            self.reference_mean = current_data.mean(dim=0)
            self.reference_std = current_data.std(dim=0) + 1e-8
        
        # Signal to other experts (could trigger memory consolidation, reset temporal state, etc.)
        self.drift_detected = True
    
    def reset_drift_flag(self):
        """Reset drift detection flag (called after handling)"""
        self.drift_detected = False
    
    def generate_insight(self) -> Dict:
        """
        Drift-specific insights for FL
        """
        insight = super().generate_insight()
        
        recent_drift_score = self.drift_scores[-1] if self.drift_scores else 0.0
        avg_drift_score = sum(self.drift_scores) / len(self.drift_scores) if self.drift_scores else 0.0
        
        insight.update({
            "specialization": "drift_detection",
            "current_drift_score": float(recent_drift_score),
            "avg_drift_score": float(avg_drift_score),
            "drift_threshold": float(self.drift_threshold),
            "drift_detected": self.drift_detected,
            "num_drifts_detected": len(self.drift_timestamps),
            "batches_since_last_drift": self.batch_count - self.last_drift_batch,
        })
        
        return insight

