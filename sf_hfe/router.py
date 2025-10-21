"""
Router (Cross-Dimension Expert)
Contextual bandit-based expert selection with UCB algorithm
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import sys

from config import ROUTER_CONFIG, EXPERT_CONFIG


class ContextualBanditRouter(nn.Module):
    """
    Router that selects which experts to activate using UCB (Upper Confidence Bound)
    
    This is the "Cross-Dimension Expert" that learns which experts work best
    for different types of input data.
    
    Capabilities:
    - Contextual bandit with UCB exploration
    - Learned context representation
    - EMA-smoothed routing decisions
    - Entropy regularization
    - Top-K expert selection
    """
    
    def __init__(self, input_dim: int, num_experts: int = 10):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = ROUTER_CONFIG["top_k_active"]
        self.exploration_bonus = ROUTER_CONFIG["exploration_bonus"]
        self.ema_alpha = ROUTER_CONFIG["ema_alpha"]
        self.entropy_weight = ROUTER_CONFIG["entropy_regularization"]
        
        # Context network: learns to represent input in "routing space"
        self.context_dim = ROUTER_CONFIG["context_dims"]
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.context_dim),
            nn.Tanh()  # Bounded context
        )
        
        # Expert preference matrix: context -> expert scores
        self.expert_scorer = nn.Sequential(
            nn.Linear(self.context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
        )
        
        # UCB statistics per expert
        self.expert_pulls = torch.zeros(num_experts)  # How many times selected
        self.expert_rewards = torch.zeros(num_experts)  # Cumulative rewards (negative loss)
        self.expert_avg_reward = torch.zeros(num_experts)  # Average reward
        
        # EMA-smoothed routing probabilities
        self.routing_probs_ema = torch.ones(num_experts) / num_experts
        
        # Router statistics
        self.total_selections = 0
        self.selection_history = []
        self.entropy_history = []
        
        # Optimizer for router
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
    def encode_context(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to context representation
        
        Args:
            x: Input data [batch_size, input_dim]
        
        Returns:
            Context vectors [batch_size, context_dim]
        """
        return self.context_encoder(x)
    
    def compute_ucb_scores(self, context: torch.Tensor) -> torch.Tensor:
        """
        Compute UCB scores for each expert
        
        UCB = mean_reward + exploration_bonus * sqrt(log(total_pulls) / expert_pulls)
        
        Args:
            context: Context representation [batch_size, context_dim]
        
        Returns:
            UCB scores [batch_size, num_experts]
        """
        batch_size = context.size(0)
        
        # Learned expert preferences from context
        expert_scores = self.expert_scorer(context)  # [batch_size, num_experts]
        
        # UCB exploration bonus
        if self.total_selections > 0:
            # Avoid division by zero
            safe_pulls = self.expert_pulls.clamp(min=1.0)
            exploration_term = torch.sqrt(
                torch.log(torch.tensor(float(self.total_selections))) / safe_pulls
            )
            # Expand to batch
            exploration_bonus = self.exploration_bonus * exploration_term.unsqueeze(0).expand(batch_size, -1)
        else:
            exploration_bonus = torch.ones(batch_size, self.num_experts) * self.exploration_bonus
        
        # Average rewards (expand to batch)
        avg_rewards = self.expert_avg_reward.unsqueeze(0).expand(batch_size, -1)
        
        # UCB = learned preference + historical performance + exploration
        ucb_scores = expert_scores + avg_rewards + exploration_bonus
        
        return ucb_scores
    
    def select_experts(
        self,
        x: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-K experts based on UCB scores
        
        Args:
            x: Input data [batch_size, input_dim]
            deterministic: If True, always select top-K (no sampling)
        
        Returns:
            expert_indices: Selected expert IDs [batch_size, top_k]
            routing_weights: Weights for selected experts [batch_size, top_k]
        """
        # Encode context
        context = self.encode_context(x)
        
        # Compute UCB scores
        ucb_scores = self.compute_ucb_scores(context)
        
        # Select top-K experts
        top_k_values, top_k_indices = torch.topk(ucb_scores, self.top_k, dim=1)
        
        # Compute routing weights (softmax over top-K)
        routing_weights = torch.softmax(top_k_values, dim=1)
        
        # Apply EMA smoothing to probabilities (per batch average)
        with torch.no_grad():
            current_probs = torch.zeros(self.num_experts)
            for i in range(len(top_k_indices)):
                for j, expert_idx in enumerate(top_k_indices[i]):
                    current_probs[expert_idx] += routing_weights[i, j].item()
            current_probs /= len(x)
            
            # EMA update
            self.routing_probs_ema = (
                self.ema_alpha * self.routing_probs_ema +
                (1 - self.ema_alpha) * current_probs
            )
        
        return top_k_indices, routing_weights
    
    def update_statistics(
        self,
        selected_experts: torch.Tensor,
        losses: torch.Tensor
    ):
        """
        Update UCB statistics after observing expert performance
        
        Args:
            selected_experts: Expert IDs that were used [batch_size, top_k]
            losses: Losses from each expert [batch_size, top_k]
        """
        # Convert losses to rewards (negative loss)
        rewards = -losses
        
        # Update statistics for each selected expert
        for i in range(len(selected_experts)):
            for j in range(len(selected_experts[i])):
                expert_id = selected_experts[i, j].item()
                reward = rewards[i, j].item()
                
                # Update pulls
                self.expert_pulls[expert_id] += 1
                
                # Update cumulative reward
                self.expert_rewards[expert_id] += reward
                
                # Update average reward
                self.expert_avg_reward[expert_id] = (
                    self.expert_rewards[expert_id] / self.expert_pulls[expert_id]
                )
        
        self.total_selections += len(selected_experts)
        
        # Track selection history
        expert_counts = torch.bincount(
            selected_experts.flatten().long(),
            minlength=self.num_experts
        )
        self.selection_history.append(expert_counts.tolist())
        
        # Keep only recent history
        if len(self.selection_history) > 1000:
            self.selection_history.pop(0)
    
    def compute_entropy(self) -> float:
        """
        Compute entropy of routing distribution
        
        Higher entropy = more uniform routing (good for exploration)
        Lower entropy = concentrated routing (potential collapse)
        """
        probs = self.routing_probs_ema / (self.routing_probs_ema.sum() + 1e-8)
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        return entropy.item()
    
    def forward(
        self,
        x: torch.Tensor,
        expert_modules: List[nn.Module],
        targets: torch.Tensor = None,
        train: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Full forward pass: select experts and get predictions
        
        Args:
            x: Input data [batch_size, input_dim]
            expert_modules: List of expert networks
            targets: Target values (for training) [batch_size, output_dim]
            train: Whether in training mode
        
        Returns:
            predictions: Weighted predictions [batch_size, output_dim]
            info: Dictionary with routing information
        """
        batch_size = x.size(0)
        
        # Select experts
        selected_indices, routing_weights = self.select_experts(x, deterministic=not train)
        
        # Get predictions from selected experts
        predictions = []
        expert_losses = []
        
        for i in range(batch_size):
            sample_preds = []
            sample_losses = []
            
            for j in range(self.top_k):
                expert_idx = selected_indices[i, j].item()
                expert = expert_modules[expert_idx]
                
                # Forward through expert
                with torch.set_grad_enabled(train):
                    pred = expert(x[i:i+1])
                    sample_preds.append(pred * routing_weights[i, j])
                    
                    # Compute loss if targets provided
                    if targets is not None:
                        loss = nn.functional.mse_loss(pred, targets[i:i+1])
                        sample_losses.append(loss)
            
            # Weighted prediction
            weighted_pred = torch.stack(sample_preds).sum(dim=0)
            predictions.append(weighted_pred)
            
            # Update statistics if training
            if train and sample_losses:
                expert_losses.append(torch.stack(sample_losses))
        
        predictions = torch.cat(predictions, dim=0)
        
        # Update UCB statistics
        if train and expert_losses:
            expert_losses_tensor = torch.stack(expert_losses)
            self.update_statistics(selected_indices, expert_losses_tensor)
        
        # Compute routing entropy
        entropy = self.compute_entropy()
        self.entropy_history.append(entropy)
        if len(self.entropy_history) > 1000:
            self.entropy_history.pop(0)
        
        # Info dict
        info = {
            "selected_experts": selected_indices.cpu().numpy(),
            "routing_weights": routing_weights.cpu().detach().numpy(),
            "routing_entropy": entropy,
            "expert_pulls": self.expert_pulls.cpu().numpy(),
            "expert_avg_rewards": self.expert_avg_reward.cpu().numpy(),
        }
        
        return predictions, info
    
    def compute_router_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        entropy: float
    ) -> torch.Tensor:
        """
        Compute router training loss
        
        Loss = prediction_loss - entropy_weight * entropy
        
        (Negative entropy encourages exploration)
        """
        # Prediction loss
        pred_loss = nn.functional.mse_loss(predictions, targets)
        
        # Entropy regularization (negative to encourage higher entropy)
        entropy_penalty = -self.entropy_weight * entropy
        
        total_loss = pred_loss + entropy_penalty
        
        return total_loss
    
    def train_router(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        entropy: float
    ):
        """
        Update router parameters
        """
        loss = self.compute_router_loss(predictions, targets, entropy)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def get_expert_statistics(self) -> Dict:
        """
        Get routing statistics for all experts
        """
        return {
            "expert_pulls": self.expert_pulls.cpu().numpy().tolist(),
            "expert_avg_rewards": self.expert_avg_reward.cpu().numpy().tolist(),
            "expert_total_rewards": self.expert_rewards.cpu().numpy().tolist(),
            "routing_probs_ema": self.routing_probs_ema.cpu().numpy().tolist(),
            "total_selections": self.total_selections,
            "current_entropy": self.compute_entropy(),
            "avg_entropy": np.mean(self.entropy_history) if self.entropy_history else 0.0,
        }
    
    def get_most_active_experts(self, k: int = 3) -> List[int]:
        """
        Get IDs of k most frequently selected experts
        """
        top_k_indices = torch.topk(self.expert_pulls, k).indices
        return top_k_indices.tolist()
    
    def generate_insight(self) -> Dict:
        """
        Generate routing insights for FL
        """
        stats = self.get_expert_statistics()
        most_active = self.get_most_active_experts()
        
        # Diversity score (how evenly distributed are selections)
        probs = self.routing_probs_ema / (self.routing_probs_ema.sum() + 1e-8)
        diversity = torch.exp(-(probs * torch.log(probs + 1e-8)).sum()).item()
        
        return {
            "component": "router",
            "most_active_experts": most_active,
            "routing_diversity": diversity,
            "current_entropy": stats["current_entropy"],
            "avg_entropy": stats["avg_entropy"],
            "total_selections": stats["total_selections"],
            "expert_utilization": {
                i: float(stats["expert_pulls"][i] / max(stats["total_selections"], 1))
                for i in range(self.num_experts)
            },
        }

