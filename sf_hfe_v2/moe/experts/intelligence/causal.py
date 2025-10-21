"""
Causal Inference Expert - Intelligence Expert #1
Specializes in discovering causal relationships using neural approaches
"""

import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_expert import BaseExpert


class CausalInferenceExpert(BaseExpert):
    """
    Expert specializing in causal relationship discovery
    
    Capabilities:
    - Neural causal discovery
    - DAG structure learning
    - Intervention prediction
    - Causal effect estimation
    """
    
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__(
            expert_id=3,
            input_dim=input_dim,
            output_dim=output_dim,
            expert_type="intelligence",
            **kwargs
        )
        
        # Causal adjacency matrix (learned)
        self.causal_graph = nn.Parameter(torch.zeros(input_dim, input_dim))
        
        # Causal mechanism network
        self.causal_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
        # Recreate optimizer now that we have all parameters
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
        
        # DAG constraint penalty weight
        self.dag_penalty_weight = 0.1
        
        # Causal metrics
        self.causal_scores = []
        self.dag_violations = []
        
    def _build_network(self) -> nn.Module:
        """Override base network"""
        return nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with causal graph
        """
        # Apply learned causal graph
        causal_mask = torch.sigmoid(self.causal_graph)
        x_causal = torch.matmul(x, causal_mask)
        
        # Predict through causal network
        output = self.causal_net(x_causal)
        return output
    
    def _task_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Causal loss: prediction + DAG constraint
        """
        # Prediction loss
        pred_loss = nn.functional.mse_loss(outputs, targets)
        
        # DAG constraint (acyclicity)
        # Ensures causal graph is a Directed Acyclic Graph
        dag_loss = self._dag_penalty()
        
        # Sparsity penalty (encourage sparse causal graphs)
        sparsity_loss = torch.mean(torch.abs(self.causal_graph))
        
        # Track
        self.dag_violations.append(dag_loss.item())
        if len(self.dag_violations) > 100:
            self.dag_violations.pop(0)
        
        return pred_loss + self.dag_penalty_weight * dag_loss + 0.01 * sparsity_loss
    
    def _dag_penalty(self) -> torch.Tensor:
        """
        DAG constraint penalty
        Penalizes cycles in the causal graph
        
        Uses trace of matrix exponential: Tr(exp(A ∘ A)) - d
        """
        # Apply sigmoid to get probabilities
        A = torch.sigmoid(self.causal_graph)
        
        # Matrix power series (approximation of exp)
        # exp(A) ≈ I + A + A²/2 + A³/6 + ...
        d = A.size(0)
        I = torch.eye(d, device=A.device)
        
        # Compute first few terms
        A_sq = torch.matmul(A, A)
        A_cube = torch.matmul(A_sq, A)
        
        exp_A = I + A + 0.5 * A_sq + (1/6) * A_cube
        
        # DAG constraint: trace should equal dimension if acyclic
        trace = torch.trace(exp_A)
        penalty = torch.relu(trace - d)  # Penalty if trace > d
        
        return penalty
    
    def get_causal_graph(self) -> np.ndarray:
        """
        Get learned causal adjacency matrix
        """
        with torch.no_grad():
            causal_probs = torch.sigmoid(self.causal_graph)
            # Threshold at 0.5
            causal_adj = (causal_probs > 0.5).float()
        return causal_adj.cpu().numpy()
    
    def generate_insight(self) -> Dict:
        """
        Causal-specific insights for FL
        """
        insight = super().generate_insight()
        
        causal_graph = self.get_causal_graph()
        num_causal_edges = int(causal_graph.sum())
        avg_dag_violation = sum(self.dag_violations) / len(self.dag_violations) if self.dag_violations else 0.0
        
        insight.update({
            "specialization": "causal_inference",
            "num_causal_edges": num_causal_edges,
            "causal_density": float(num_causal_edges / (self.input_dim ** 2)),
            "avg_dag_violation": float(avg_dag_violation),
            "is_acyclic": avg_dag_violation < 0.01,
        })
        
        return insight

