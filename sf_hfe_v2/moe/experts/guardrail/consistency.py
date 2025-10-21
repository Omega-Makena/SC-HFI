"""
Statistical Consistency Expert - Guardrail Expert #2
Specializes in outlier detection and statistical validation
"""

import torch
import torch.nn as nn
from typing import Dict
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_expert import BaseExpert


class StatisticalConsistencyExpert(BaseExpert):
    """
    Expert specializing in statistical consistency and outlier detection
    
    Capabilities:
    - Outlier detection (Z-score, IQR)
    - Distribution consistency checking
    - Statistical anomaly detection
    - Data quality validation
    """
    
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__(
            expert_id=6,
            input_dim=input_dim,
            output_dim=output_dim,
            expert_type="guardrail",
            **kwargs
        )
        
        # Running statistics
        self.running_mean = torch.zeros(input_dim)
        self.running_var = torch.ones(input_dim)
        self.n_samples = 0
        
        # Outlier tracking
        self.outlier_scores = []
        self.outlier_flags = []
        
        # Thresholds
        self.z_score_threshold = 3.0  # Standard z-score threshold
        self.iqr_multiplier = 1.5
        
    def _task_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Consistency-aware loss
        """
        # Standard loss
        pred_loss = nn.functional.mse_loss(outputs, targets)
        
        # Consistency penalty (penalize predictions far from mean)
        if self.n_samples > 100:
            z_scores = torch.abs((outputs - self.running_mean) / torch.sqrt(self.running_var + 1e-8))
            consistency_penalty = torch.mean(torch.relu(z_scores - self.z_score_threshold))
            return pred_loss + 0.2 * consistency_penalty
        
        return pred_loss
    
    def online_update(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        replay: bool = True
    ) -> Dict[str, float]:
        """
        Update with outlier detection
        """
        # Detect outliers in batch
        outlier_mask = self._detect_outliers(batch_x)
        num_outliers = outlier_mask.sum().item()
        
        # Standard update
        metrics = super().online_update(batch_x, batch_y, replay)
        
        # Update running statistics
        self._update_running_stats(batch_x)
        
        metrics.update({
            "num_outliers": num_outliers,
            "outlier_rate": num_outliers / len(batch_x),
            "avg_outlier_score": float(self.outlier_scores[-1]) if self.outlier_scores else 0.0,
        })
        
        return metrics
    
    def _update_running_stats(self, batch_x: torch.Tensor):
        """
        Update running mean and variance (Welford's method)
        """
        for x in batch_x:
            self.n_samples += 1
            delta = x - self.running_mean
            self.running_mean += delta / self.n_samples
            delta2 = x - self.running_mean
            self.running_var += delta * delta2
    
    def _detect_outliers(self, batch_x: torch.Tensor) -> torch.Tensor:
        """
        Detect outliers using Z-score method
        
        Returns:
            Boolean mask where True indicates outlier
        """
        if self.n_samples < 10:
            # Not enough data yet
            self.outlier_flags.extend([False] * len(batch_x))
            self.outlier_scores.append(0.0)
            return torch.zeros(len(batch_x), dtype=torch.bool)
        
        # Compute Z-scores
        std = torch.sqrt(self.running_var / self.n_samples + 1e-8)
        z_scores = torch.abs((batch_x - self.running_mean) / std)
        
        # Outliers are points with Z-score > threshold
        outlier_mask = (z_scores.max(dim=1)[0] > self.z_score_threshold)
        
        # Track
        self.outlier_flags.extend(outlier_mask.tolist())
        avg_z_score = float(z_scores.mean().item())
        self.outlier_scores.append(avg_z_score)
        
        # Keep recent
        if len(self.outlier_flags) > 10000:
            self.outlier_flags = self.outlier_flags[-10000:]
        if len(self.outlier_scores) > 1000:
            self.outlier_scores = self.outlier_scores[-1000:]
        
        return outlier_mask
    
    def compute_consistency_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency score for data
        
        Returns:
            Score between 0 (outlier) and 1 (consistent)
        """
        if self.n_samples < 10:
            return torch.ones(len(x))
        
        # Z-score based consistency
        std = torch.sqrt(self.running_var / self.n_samples + 1e-8)
        z_scores = torch.abs((x - self.running_mean) / std)
        max_z_scores = z_scores.max(dim=1)[0]
        
        # Convert to score (higher z-score = lower consistency)
        consistency = torch.exp(-max_z_scores / self.z_score_threshold)
        
        return consistency
    
    def validate_batch(self, batch_x: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, int]:
        """
        Validate a batch for statistical consistency
        
        Returns:
            (mask of valid samples, count of invalid)
        """
        consistency_scores = self.compute_consistency_score(batch_x)
        valid_mask = consistency_scores > threshold
        invalid_count = (~valid_mask).sum().item()
        
        return valid_mask, invalid_count
    
    def generate_insight(self) -> Dict:
        """
        Consistency-specific insights for FL
        """
        insight = super().generate_insight()
        
        recent_outlier_rate = sum(self.outlier_flags[-1000:]) / len(self.outlier_flags[-1000:]) if len(self.outlier_flags) >= 1000 else 0.0
        avg_outlier_score = sum(self.outlier_scores) / len(self.outlier_scores) if self.outlier_scores else 0.0
        
        insight.update({
            "specialization": "statistical_consistency",
            "outlier_rate": float(recent_outlier_rate),
            "avg_z_score": float(avg_outlier_score),
            "samples_analyzed": self.n_samples,
            "z_score_threshold": self.z_score_threshold,
        })
        
        return insight

