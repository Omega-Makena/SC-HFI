"""
Meta-Adaptation Expert - Control Expert
Specializes in dynamic learning rate scheduling for all experts
"""

import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_expert import BaseExpert


class MetaAdaptationExpert(BaseExpert):
    """
    Expert specializing in meta-level adaptation and LR scheduling
    
    Capabilities:
    - Dynamic learning rate adjustment per expert
    - Convergence monitoring
    - Performance-based LR scheduling
    - Freeze/unfreeze decisions
    """
    
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__(
            expert_id=8,
            input_dim=input_dim,
            output_dim=output_dim,
            expert_type="control",
            **kwargs
        )
        
        # Track all experts' performance
        self.expert_loss_history = {}  # expert_id -> loss history
        self.expert_gradient_norms = {}  # expert_id -> gradient norms
        self.expert_lr_recommendations = {}  # expert_id -> recommended LR
        
        # Adaptation parameters
        self.lr_increase_factor = 1.05
        self.lr_decrease_factor = 0.9
        self.convergence_threshold = 0.001
        self.freeze_gradient_threshold = 0.0001
        
        # Monitoring
        self.adaptation_events = []
        
    def _task_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Standard prediction loss
        Meta-adaptation logic happens in monitoring, not loss
        """
        return nn.functional.mse_loss(outputs, targets)
    
    def monitor_expert(
        self,
        expert_id: int,
        loss: float,
        gradient_norm: float,
        current_lr: float
    ):
        """
        Monitor an expert's performance and recommend LR adjustment
        
        Args:
            expert_id: ID of the expert
            loss: Current loss value
            gradient_norm: Gradient norm
            current_lr: Current learning rate
        """
        # Initialize tracking for new expert
        if expert_id not in self.expert_loss_history:
            self.expert_loss_history[expert_id] = []
            self.expert_gradient_norms[expert_id] = []
            self.expert_lr_recommendations[expert_id] = current_lr
        
        # Store metrics
        self.expert_loss_history[expert_id].append(loss)
        self.expert_gradient_norms[expert_id].append(gradient_norm)
        
        # Keep only recent history
        if len(self.expert_loss_history[expert_id]) > 100:
            self.expert_loss_history[expert_id].pop(0)
            self.expert_gradient_norms[expert_id].pop(0)
        
        # Adapt LR if we have enough history
        if len(self.expert_loss_history[expert_id]) >= 10:
            new_lr = self._compute_adaptive_lr(expert_id, current_lr)
            self.expert_lr_recommendations[expert_id] = new_lr
    
    def _compute_adaptive_lr(self, expert_id: int, current_lr: float) -> float:
        """
        Compute adaptive learning rate based on loss trends
        """
        loss_history = self.expert_loss_history[expert_id]
        grad_history = self.expert_gradient_norms[expert_id]
        
        # Check if converged (loss stable)
        recent_losses = loss_history[-10:]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        
        # Check if improving
        if len(loss_history) >= 20:
            recent_mean = np.mean(loss_history[-10:])
            past_mean = np.mean(loss_history[-20:-10])
            improving = recent_mean < past_mean
        else:
            improving = False
        
        # Check gradient magnitude
        recent_grad_norm = np.mean(grad_history[-10:])
        
        # Decision logic
        new_lr = current_lr
        action = "maintain"
        
        # Case 1: Converged (small gradients) -> freeze or reduce LR
        if recent_grad_norm < self.freeze_gradient_threshold:
            action = "freeze"
            new_lr = 0.0  # Signals freeze
        
        # Case 2: Loss increasing -> reduce LR
        elif not improving and loss_mean > 0.1:
            new_lr = current_lr * self.lr_decrease_factor
            action = "decrease"
        
        # Case 3: Loss decreasing steadily -> slight increase in LR
        elif improving and loss_std < self.convergence_threshold:
            new_lr = current_lr * self.lr_increase_factor
            action = "increase"
        
        # Clip to safe range
        new_lr = np.clip(new_lr, 0.0, 0.01)
        
        # Log adaptation event
        if action != "maintain":
            self.adaptation_events.append({
                "expert_id": expert_id,
                "batch": self.batch_count,
                "action": action,
                "old_lr": current_lr,
                "new_lr": new_lr,
                "loss_mean": loss_mean,
                "grad_norm": recent_grad_norm,
            })
            
            # Keep only recent events
            if len(self.adaptation_events) > 1000:
                self.adaptation_events.pop(0)
        
        return new_lr
    
    def get_lr_recommendation(self, expert_id: int) -> float:
        """
        Get recommended learning rate for an expert
        """
        if expert_id in self.expert_lr_recommendations:
            return self.expert_lr_recommendations[expert_id]
        return 0.001  # Default
    
    def should_freeze(self, expert_id: int) -> bool:
        """
        Check if expert should be frozen
        """
        if expert_id not in self.expert_gradient_norms:
            return False
        
        recent_grads = self.expert_gradient_norms[expert_id][-10:]
        if len(recent_grads) < 10:
            return False
        
        avg_grad = np.mean(recent_grads)
        return avg_grad < self.freeze_gradient_threshold
    
    def should_unfreeze(self, expert_id: int, current_loss: float) -> bool:
        """
        Check if frozen expert should be unfrozen
        (e.g., if overall performance degrades)
        """
        if expert_id not in self.expert_loss_history:
            return False
        
        # Unfreeze if recent system loss is high
        return current_loss > 0.5
    
    def get_adaptation_summary(self) -> Dict:
        """
        Get summary of all adaptation events
        """
        if not self.adaptation_events:
            return {
                "total_adaptations": 0,
                "lr_increases": 0,
                "lr_decreases": 0,
                "freezes": 0,
            }
        
        return {
            "total_adaptations": len(self.adaptation_events),
            "lr_increases": sum(1 for e in self.adaptation_events if e["action"] == "increase"),
            "lr_decreases": sum(1 for e in self.adaptation_events if e["action"] == "decrease"),
            "freezes": sum(1 for e in self.adaptation_events if e["action"] == "freeze"),
            "recent_events": self.adaptation_events[-5:],
        }
    
    def generate_insight(self) -> Dict:
        """
        Meta-adaptation-specific insights for FL
        """
        insight = super().generate_insight()
        
        adaptation_summary = self.get_adaptation_summary()
        num_experts_monitored = len(self.expert_loss_history)
        
        insight.update({
            "specialization": "meta_adaptation",
            "experts_monitored": num_experts_monitored,
            **adaptation_summary,
        })
        
        return insight

