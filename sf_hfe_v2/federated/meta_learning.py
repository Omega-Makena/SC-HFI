"""
Online MAML Meta-Learning Engine - Federated Learning Component
Learns optimal initialization and expert-specific learning rates from insights
"""

import torch
from typing import Dict, List
import numpy as np
from collections import defaultdict
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import META_LEARNING_CONFIG


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
        
        # Meta-parameters
        self.w_init = None  # Will be initialized from first insights
        self.expert_alphas = {
            i: META_LEARNING_CONFIG["expert_lr_init"]
            for i in range(num_experts)
        }
        
        # Meta-learning state
        self.meta_updates = 0
        self.meta_loss_history = []
        
        # Running statistics across all clients
        self.global_stats = {
            "loss_mean": 0.0,
            "loss_std": 0.0,
            "activation_frequencies": torch.zeros(num_experts),
            "expert_performance": torch.zeros(num_experts),
        }
        
    def meta_update(self, insights: List[Dict]) -> Dict[str, any]:
        """
        Perform meta-learning update from client insights
        
        This is where the Developer learns WITHOUT data!
        Only uses metadata from users.
        
        Args:
            insights: List of insight dictionaries from clients
        
        Returns:
            Updated meta-parameters
        """
        if not insights:
            return self.get_meta_parameters()
        
        self.meta_updates += 1
        
        # Step 1: Aggregate expert performance across clients
        expert_losses = defaultdict(list)
        expert_activations = defaultdict(int)
        expert_lr_trends = defaultdict(list)
        
        for insight in insights:
            expert_insights = insight.get("expert_insights", {})
            
            for expert_name, expert_data in expert_insights.items():
                expert_id = expert_data.get("expert_id")
                if expert_id is not None:
                    # Loss
                    ema_loss = expert_data.get("ema_loss", 0.0)
                    expert_losses[expert_id].append(ema_loss)
                    
                    # Activation count
                    activation = expert_data.get("activation_count", 0)
                    expert_activations[expert_id] += activation
                    
                    # Learning rate
                    lr = expert_data.get("learning_rate", 0.001)
                    expert_lr_trends[expert_id].append(lr)
        
        # Step 2: Compute global statistics
        for expert_id in range(self.num_experts):
            if expert_id in expert_losses and expert_losses[expert_id]:
                # Average performance
                avg_loss = np.mean(expert_losses[expert_id])
                self.global_stats["expert_performance"][expert_id] = avg_loss
                
                # Activation frequency
                self.global_stats["activation_frequencies"][expert_id] = expert_activations[expert_id]
        
        # Normalize activation frequencies
        total_activations = self.global_stats["activation_frequencies"].sum()
        if total_activations > 0:
            self.global_stats["activation_frequencies"] /= total_activations
        
        # Step 3: Adapt expert-specific learning rates (alpha_i)
        for expert_id in range(self.num_experts):
            if expert_id in expert_lr_trends and expert_lr_trends[expert_id]:
                # Use median of successful clients' LRs
                successful_lrs = expert_lr_trends[expert_id]
                new_alpha = float(np.median(successful_lrs))
                
                # Blend with current (EMA)
                self.expert_alphas[expert_id] = (
                    0.7 * self.expert_alphas[expert_id] +
                    0.3 * new_alpha
                )
        
        # Step 4: Compute meta-loss (average across all experts and clients)
        all_losses = [loss for losses in expert_losses.values() for loss in losses]
        meta_loss = np.mean(all_losses) if all_losses else 0.0
        self.meta_loss_history.append(meta_loss)
        
        # Keep recent history
        if len(self.meta_loss_history) > 1000:
            self.meta_loss_history = self.meta_loss_history[-1000:]
        
        return self.get_meta_parameters()
    
    def get_meta_parameters(self) -> Dict:
        """Get current meta-parameters for broadcast to clients"""
        return {
            "expert_alphas": self.expert_alphas,
            "global_stats": {
                "avg_loss": float(self.global_stats["expert_performance"].mean().item()),
                "activation_frequencies": self.global_stats["activation_frequencies"].cpu().numpy().tolist(),
            },
            "meta_updates": self.meta_updates,
            "apply_to_new_experts": False,  # Flag for initialization
        }
    
    def stats(self) -> Dict:
        """Meta-learning engine statistics"""
        return {
            "meta_updates": self.meta_updates,
            "meta_loss_history": self.meta_loss_history[-100:],
            "avg_meta_loss": np.mean(self.meta_loss_history) if self.meta_loss_history else 0.0,
            "expert_alphas": self.expert_alphas,
            "global_performance": self.global_stats["expert_performance"].cpu().numpy().tolist(),
        }

