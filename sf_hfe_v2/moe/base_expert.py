"""
Base Expert Class for SF-HFE
All 10 experts inherit from this base with online learning capabilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional
import numpy as np
from abc import ABC, abstractmethod

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory import HierarchicalMemory
from config import (
    EXPERT_CONFIG, MEMORY_CONFIG, LEARNING_CONFIG,
    STABILITY_CONFIG, META_LEARNING_CONFIG
)


class BaseExpert(nn.Module, ABC):
    """
    Base class for all SF-HFE experts
    
    Implements:
    - Online learning (per mini-batch updates)
    - Hierarchical memory with replay
    - Elastic Weight Consolidation (anti-forgetting)
    - Adaptive learning rate
    - Loss tracking and statistics
    """
    
    def __init__(
        self,
        expert_id: int,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = None,
        expert_type: str = "general"
    ):
        super(BaseExpert, self).__init__()
        
        self.expert_id = expert_id
        self.expert_name = EXPERT_CONFIG["experts"][expert_id]["name"]
        self.expert_type = expert_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Hidden dimensions
        if hidden_dims is None:
            hidden_dims = EXPERT_CONFIG["expert_hidden_dims"]
        self.hidden_dims = hidden_dims
        
        # Build network
        built_network = self._build_network()
        if built_network is not None:
            self.network = built_network
        else:
            # Subclass didn't override or returned None - build default
            self.network = self._build_default_network()
        
        # Optimizer (adaptive per expert)
        self.lr = META_LEARNING_CONFIG["expert_lr_init"]
        # Delay optimizer creation if no parameters yet (subclass will create)
        try:
            self.optimizer = self._create_optimizer()
        except ValueError:
            # No parameters yet - subclass will handle
            self.optimizer = None
        
        # Memory system
        self.memory = HierarchicalMemory(
            recent_size=MEMORY_CONFIG["recent_buffer_size"],
            compressed_size=MEMORY_CONFIG["compressed_size"],
            critical_size=MEMORY_CONFIG["critical_anchors_size"],
            latent_dim=MEMORY_CONFIG["compression_dim"]
        )
        
        # Online learning state
        self.batch_count = 0
        self.total_samples_seen = 0
        self.activation_count = 0  # How many times this expert was used
        
        # Loss tracking (EMA smoothed)
        self.current_loss = 0.0
        self.ema_loss = 0.0
        self.loss_history = []
        
        # EWC (Elastic Weight Consolidation) for anti-forgetting
        self.ewc_enabled = STABILITY_CONFIG["elastic_weight_consolidation"]
        self.ewc_lambda = STABILITY_CONFIG["ewc_lambda"]
        self.fisher_information = {}
        self.optimal_params = {}
        self._init_ewc()
        
        # Frozen state
        self.is_frozen = False
    
    def _build_default_network(self) -> nn.Module:
        """Build default network (for when subclass doesn't override)"""
        layers = []
        
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(EXPERT_CONFIG["dropout"]))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        return nn.Sequential(*layers)
        
    def _build_network(self) -> nn.Module:
        """
        Build the expert's neural network
        
        Subclasses can override this to return their own architecture
        or return None to use the default builder
        """
        # Check if subclass has custom architecture
        if hasattr(self, 'lstm') or hasattr(self, 'encoder') or hasattr(self, 'decoder'):
            return nn.Identity()  # Subclass handles it
        
        # Return None to signal using default builder
        return None
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for this expert"""
        if LEARNING_CONFIG["optimizer"] == "adam":
            return optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=LEARNING_CONFIG["weight_decay"]
            )
        elif LEARNING_CONFIG["optimizer"] == "sgd":
            return optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=LEARNING_CONFIG["momentum"],
                weight_decay=LEARNING_CONFIG["weight_decay"]
            )
        else:
            raise ValueError(f"Unknown optimizer: {LEARNING_CONFIG['optimizer']}")
    
    def _init_ewc(self):
        """Initialize EWC (Elastic Weight Consolidation)"""
        for name, param in self.named_parameters():
            self.fisher_information[name] = torch.zeros_like(param.data)
            self.optimal_params[name] = param.data.clone()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert network"""
        return self.network(x)
    
    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        include_ewc: bool = True
    ) -> torch.Tensor:
        """
        Compute loss with optional EWC regularization
        
        Args:
            outputs: Model predictions
            targets: Ground truth
            include_ewc: Whether to include EWC penalty
        """
        # Task loss (can be overridden by subclasses)
        task_loss = self._task_loss(outputs, targets)
        
        # EWC penalty (anti-forgetting)
        if include_ewc and self.ewc_enabled and self.batch_count > 100:
            ewc_loss = self._ewc_penalty()
            total_loss = task_loss + self.ewc_lambda * ewc_loss
        else:
            total_loss = task_loss
        
        return total_loss
    
    @abstractmethod
    def _task_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Task-specific loss (must be implemented by subclasses)"""
        pass
    
    def _ewc_penalty(self) -> torch.Tensor:
        """Compute EWC regularization penalty"""
        penalty = 0.0
        for name, param in self.named_parameters():
            if name in self.fisher_information:
                penalty += (
                    self.fisher_information[name] * 
                    (param - self.optimal_params[name]).pow(2)
                ).sum()
        return penalty
    
    def update_fisher_information(self, dataloader):
        """Update Fisher Information Matrix for EWC"""
        self.eval()
        
        for name, param in self.named_parameters():
            self.fisher_information[name].zero_()
        
        # Compute Fisher Information
        for xs, ys in dataloader:
            self.zero_grad()
            outputs = self.forward(xs)
            loss = self._task_loss(outputs, ys)
            loss.backward()
            
            for name, param in self.named_parameters():
                if param.grad is not None:
                    self.fisher_information[name] += param.grad.data.pow(2)
        
        # Average over batches
        num_batches = len(dataloader)
        for name in self.fisher_information:
            self.fisher_information[name] /= num_batches
        
        # Update optimal parameters
        for name, param in self.named_parameters():
            self.optimal_params[name] = param.data.clone()
        
        self.train()
    
    def online_update(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        replay: bool = True
    ) -> Dict[str, float]:
        """
        Single online update step
        
        Args:
            batch_x: Input mini-batch
            batch_y: Target mini-batch
            replay: Whether to include replay samples
        
        Returns:
            Dictionary with loss and metrics
        """
        if self.is_frozen:
            return {"loss": 0.0, "status": "frozen"}
        
        # Ensure optimizer exists (create if needed)
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
        
        self.train()
        self.activation_count += 1
        self.batch_count += 1
        self.total_samples_seen += len(batch_x)
        
        # 1. Current batch loss
        outputs = self.forward(batch_x)
        current_loss = self.compute_loss(outputs, batch_y)
        
        # 2. Replay loss (anti-forgetting)
        replay_loss = 0.0
        if replay and self.memory.total_size() > 0:
            replay_x, replay_y = self.memory.replay_batch(
                batch_size=MEMORY_CONFIG["replay_batch_size"]
            )
            if replay_x is not None:
                replay_outputs = self.forward(replay_x)
                replay_loss = self.compute_loss(replay_outputs, replay_y, include_ewc=False)
        
        # 3. Total loss
        if replay_loss > 0:
            total_loss = 0.7 * current_loss + 0.3 * replay_loss
        else:
            total_loss = current_loss
        
        # 4. Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            LEARNING_CONFIG["gradient_clip"]
        )
        
        # 5. Update weights
        self.optimizer.step()
        
        # 6. Store in memory (with uncertainty for critical anchors)
        uncertainties = self._compute_uncertainty(batch_x)
        self.memory.add_batch(batch_x, batch_y, uncertainties)
        
        # 7. Update loss tracking
        loss_value = total_loss.item()
        self.current_loss = loss_value
        self.ema_loss = (
            LEARNING_CONFIG["loss_smoothing_ema"] * self.ema_loss +
            (1 - LEARNING_CONFIG["loss_smoothing_ema"]) * loss_value
        )
        self.loss_history.append(loss_value)
        
        # Metrics
        metrics = {
            "loss": loss_value,
            "ema_loss": self.ema_loss,
            "replay_loss": replay_loss.item() if isinstance(replay_loss, torch.Tensor) else replay_loss,
            "memory_size": self.memory.total_size(),
            "batch_count": self.batch_count,
            "activation_count": self.activation_count,
        }
        
        return metrics
    
    def _compute_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction uncertainty (for critical anchor selection)
        Using entropy of predictions as proxy
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probs = torch.softmax(outputs, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        self.train()
        return entropy
    
    def adapt_learning_rate(self, new_lr: float):
        """Adapt learning rate (called by Meta-Adaptation Expert)"""
        # Ensure optimizer exists
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
        
        # Clip to safe range
        new_lr = np.clip(
            new_lr,
            META_LEARNING_CONFIG["expert_lr_min"],
            META_LEARNING_CONFIG["expert_lr_max"]
        )
        
        self.lr = new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def freeze(self):
        """Freeze expert (stop learning)"""
        self.is_frozen = True
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze expert (resume learning)"""
        self.is_frozen = False
        for param in self.parameters():
            param.requires_grad = True
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get expert weights (for P2P exchange)"""
        return {name: param.data.clone() for name, param in self.named_parameters()}
    
    def set_weights(self, weights: Dict[str, torch.Tensor], blend_factor: float = 0.5):
        """
        Set expert weights (from P2P gossip)
        
        Args:
            weights: New weights from peer
            blend_factor: How much to blend (0=keep all, 1=replace all)
        """
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in weights:
                    # Blend: interpolate between current and new
                    param.data = (
                        (1 - blend_factor) * param.data +
                        blend_factor * weights[name]
                    )
    
    def generate_insight(self) -> Dict:
        """
        Generate insight for FL (NOT raw weights/data)
        
        Returns metadata about expert's learning experience
        """
        return {
            "expert_id": self.expert_id,
            "expert_name": self.expert_name,
            "expert_type": self.expert_type,
            "activation_count": self.activation_count,
            "total_samples": self.total_samples_seen,
            "current_loss": self.current_loss,
            "ema_loss": self.ema_loss,
            "loss_trend": "improving" if len(self.loss_history) > 10 and 
                         self.loss_history[-1] < self.loss_history[-10] else "stable",
            "memory_utilization": self.memory.total_size() / (
                MEMORY_CONFIG["recent_buffer_size"] +
                MEMORY_CONFIG["compressed_size"] +
                MEMORY_CONFIG["critical_anchors_size"]
            ),
            "learning_rate": self.lr,
            "is_frozen": self.is_frozen,
        }
    
    def stats(self) -> Dict:
        """Detailed statistics for monitoring"""
        return {
            **self.generate_insight(),
            "memory_stats": self.memory.stats(),
            "loss_history_len": len(self.loss_history),
            "recent_loss_mean": np.mean(self.loss_history[-100:]) if len(self.loss_history) >= 100 else self.ema_loss,
            "recent_loss_std": np.std(self.loss_history[-100:]) if len(self.loss_history) >= 100 else 0.0,
        }

