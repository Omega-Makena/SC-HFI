"""
Memory Consolidation Expert - Memory Expert
Specializes in orchestrating memory replay and preventing forgetting
"""

import torch
import torch.nn as nn
from typing import Dict
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_expert import BaseExpert


class MemoryConsolidationExpert(BaseExpert):
    """
    Expert specializing in memory consolidation and replay orchestration
    
    Capabilities:
    - Memory replay scheduling
    - Forgetting prevention
    - Important sample identification
    - Cross-expert memory coordination
    """
    
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__(
            expert_id=9,
            input_dim=input_dim,
            output_dim=output_dim,
            expert_type="memory",
            **kwargs
        )
        
        # Store input_dim for later use
        self.stored_input_dim = input_dim
        
        # Memory importance scorer (uses input features, not outputs)
        self.importance_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Importance score 0-1
        )
        
        # Consolidation metrics
        self.consolidation_events = []
        self.forgetting_scores = []
        self.memory_pressure = 0.0  # How full is memory
        
    # Don't override _build_network - let parent build it
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward: predict and compute importance
        """
        # Use the network built by parent class
        return self.network(x)
    
    def _task_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Memory-aware loss
        Note: We need access to input features for importance network
        For now, use simplified loss
        """
        # Standard prediction loss
        pred_loss = nn.functional.mse_loss(outputs, targets)
        
        # Note: Importance training happens separately in online_update
        # where we have access to input features
        
        return pred_loss
    
    def compute_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute importance scores for samples
        
        Returns:
            Importance scores [0, 1] for each sample
        """
        self.eval()
        with torch.no_grad():
            importance = self.importance_net(x)
        self.train()
        return importance.squeeze()
    
    def should_consolidate(self, batch_count: int) -> bool:
        """
        Decide if memory consolidation should happen
        
        Triggers:
        - Every N batches
        - When memory pressure is high
        - After drift detection
        """
        # Periodic consolidation
        if batch_count % 100 == 0:
            return True
        
        # High memory pressure
        if self.memory_pressure > 0.8:
            return True
        
        return False
    
    def consolidate_memory(self, memory_system) -> Dict:
        """
        Perform memory consolidation
        
        This involves:
        1. Identifying most important samples
        2. Moving them to critical anchors
        3. Compressing less important to latent space
        
        Args:
            memory_system: The HierarchicalMemory instance
        
        Returns:
            Consolidation metrics
        """
        # Sample from recent buffer
        xs, ys = memory_system.tier1.sample(memory_system.tier1.size())
        if xs is None or len(xs) == 0:
            return {"consolidated": 0, "avg_importance": 0.0}
        
        # Compute importance
        importance_scores = self.compute_importance(xs)
        
        # Ensure it's 1-d
        if importance_scores.dim() == 0:
            importance_scores = importance_scores.unsqueeze(0)
        
        # Identify top-K important samples
        k = min(10, importance_scores.size(0))
        if k == 0:
            return {"consolidated": 0, "avg_importance": 0.0}
            
        top_k_indices = torch.topk(importance_scores, k).indices
        
        # Move important samples to critical anchors
        consolidated_count = 0
        for idx in top_k_indices:
            memory_system.tier3.add(
                xs[idx],
                ys[idx],
                importance_scores[idx].item()
            )
            consolidated_count += 1
        
        # Log consolidation event
        self.consolidation_events.append({
            "batch": self.batch_count,
            "samples_consolidated": consolidated_count,
            "avg_importance": float(importance_scores.mean().item()),
            "max_importance": float(importance_scores.max().item()),
        })
        
        # Keep recent events
        if len(self.consolidation_events) > 100:
            self.consolidation_events.pop(0)
        
        return {
            "consolidated": consolidated_count,
            "avg_importance": float(importance_scores.mean().item()),
        }
    
    def estimate_forgetting_risk(self) -> float:
        """
        Estimate risk of catastrophic forgetting
        
        Based on:
        - Memory utilization
        - Time since last consolidation
        - Loss instability
        """
        # Base risk from memory pressure
        risk = self.memory_pressure
        
        # Increase risk if no recent consolidation
        if self.consolidation_events:
            batches_since_consolidation = self.batch_count - self.consolidation_events[-1]["batch"]
            risk += min(batches_since_consolidation / 1000, 0.5)
        
        # Increase risk if loss is unstable
        if len(self.loss_history) > 20:
            recent_std = torch.std(torch.tensor(self.loss_history[-20:])).item()
            risk += min(recent_std, 0.3)
        
        return min(risk, 1.0)
    
    def update_memory_pressure(self, memory_system):
        """
        Update memory pressure metric
        """
        total_capacity = (
            memory_system.tier1.max_size +
            memory_system.tier2.max_size +
            memory_system.tier3.max_size
        )
        total_used = memory_system.total_size()
        
        self.memory_pressure = total_used / total_capacity
    
    def generate_insight(self) -> Dict:
        """
        Memory-consolidation-specific insights for FL
        """
        insight = super().generate_insight()
        
        num_consolidations = len(self.consolidation_events)
        avg_consolidated = sum(e["samples_consolidated"] for e in self.consolidation_events) / num_consolidations if num_consolidations > 0 else 0
        forgetting_risk = self.estimate_forgetting_risk()
        
        insight.update({
            "specialization": "memory_consolidation",
            "num_consolidations": num_consolidations,
            "avg_samples_consolidated": float(avg_consolidated),
            "memory_pressure": float(self.memory_pressure),
            "forgetting_risk": float(forgetting_risk),
        })
        
        return insight

