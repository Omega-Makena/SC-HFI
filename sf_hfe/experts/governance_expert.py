"""
Governance Expert - Guardrail Expert #1  
Specializes in constraint validation and rule enforcement
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_expert import BaseExpert


class GovernanceExpert(BaseExpert):
    """
    Expert specializing in governance and constraint enforcement
    
    Capabilities:
    - Constraint validation
    - Boundary enforcement  
    - Rule compliance checking
    - Safety guardrails
    """
    
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__(
            expert_id=5,
            input_dim=input_dim,
            output_dim=output_dim,
            expert_type="guardrail",
            **kwargs
        )
        
        # Learned constraints
        self.input_bounds_min = None
        self.input_bounds_max = None
        self.output_bounds_min = None
        self.output_bounds_max = None
        
        # Violation tracking
        self.input_violations = []
        self.output_violations = []
        self.constraint_violations = []
        
        # Safety thresholds (learned adaptively)
        self.safety_margin = 0.1
        
    def _task_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Governance loss: prediction + constraint penalties
        """
        # Standard prediction loss
        pred_loss = nn.functional.mse_loss(outputs, targets)
        
        # Constraint penalties
        constraint_penalty = self._compute_constraint_penalty(outputs)
        
        return pred_loss + 0.5 * constraint_penalty
    
    def _compute_constraint_penalty(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Penalize outputs that violate learned constraints
        """
        if self.output_bounds_min is None:
            return torch.tensor(0.0)
        
        # Penalty for outputs outside bounds
        lower_violation = torch.relu(self.output_bounds_min - outputs)
        upper_violation = torch.relu(outputs - self.output_bounds_max)
        
        penalty = torch.mean(lower_violation + upper_violation)
        
        # Track violations
        if penalty.item() > 0:
            self.constraint_violations.append(penalty.item())
            if len(self.constraint_violations) > 100:
                self.constraint_violations.pop(0)
        
        return penalty
    
    def online_update(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        replay: bool = True
    ) -> Dict[str, float]:
        """
        Update with constraint learning
        """
        # Check input constraints
        self._check_input_constraints(batch_x)
        
        # Standard update
        metrics = super().online_update(batch_x, batch_y, replay)
        
        # Update learned bounds
        self._update_bounds(batch_x, batch_y)
        
        # Check output constraints
        with torch.no_grad():
            outputs = self.forward(batch_x)
            self._check_output_constraints(outputs)
        
        metrics.update({
            "input_violations": len([v for v in self.input_violations if v]),
            "output_violations": len([v for v in self.output_violations if v]),
            "constraint_penalty": self.constraint_violations[-1] if self.constraint_violations else 0.0,
        })
        
        return metrics
    
    def _update_bounds(self, batch_x: torch.Tensor, batch_y: torch.Tensor):
        """
        Update learned bounds (moving window)
        """
        # Initialize bounds
        if self.input_bounds_min is None:
            self.input_bounds_min = batch_x.min(dim=0)[0]
            self.input_bounds_max = batch_x.max(dim=0)[0]
            self.output_bounds_min = batch_y.min(dim=0)[0]
            self.output_bounds_max = batch_y.max(dim=0)[0]
        else:
            # EMA update
            alpha = 0.05
            self.input_bounds_min = (1 - alpha) * self.input_bounds_min + alpha * batch_x.min(dim=0)[0]
            self.input_bounds_max = (1 - alpha) * self.input_bounds_max + alpha * batch_x.max(dim=0)[0]
            self.output_bounds_min = (1 - alpha) * self.output_bounds_min + alpha * batch_y.min(dim=0)[0]
            self.output_bounds_max = (1 - alpha) * self.output_bounds_max + alpha * batch_y.max(dim=0)[0]
        
        # Add safety margin
        margin = (self.input_bounds_max - self.input_bounds_min) * self.safety_margin
        self.input_bounds_min -= margin
        self.input_bounds_max += margin
        
        margin = (self.output_bounds_max - self.output_bounds_min) * self.safety_margin
        self.output_bounds_min -= margin
        self.output_bounds_max += margin
    
    def _check_input_constraints(self, batch_x: torch.Tensor):
        """
        Check if inputs satisfy constraints
        """
        if self.input_bounds_min is None:
            self.input_violations.append(False)
            return
        
        # Check bounds
        below_min = (batch_x < self.input_bounds_min).any()
        above_max = (batch_x > self.input_bounds_max).any()
        
        violated = below_min or above_max
        self.input_violations.append(violated.item() if isinstance(violated, torch.Tensor) else violated)
        
        if len(self.input_violations) > 1000:
            self.input_violations.pop(0)
    
    def _check_output_constraints(self, outputs: torch.Tensor):
        """
        Check if outputs satisfy constraints
        """
        if self.output_bounds_min is None:
            self.output_violations.append(False)
            return
        
        # Check bounds
        below_min = (outputs < self.output_bounds_min).any()
        above_max = (outputs > self.output_bounds_max).any()
        
        violated = below_min or above_max
        self.output_violations.append(violated.item() if isinstance(violated, torch.Tensor) else violated)
        
        if len(self.output_violations) > 1000:
            self.output_violations.pop(0)
    
    def validate_sample(self, x: torch.Tensor, y: torch.Tensor = None) -> Tuple[bool, str]:
        """
        Validate if a sample satisfies all constraints
        
        Returns:
            (is_valid, reason)
        """
        if self.input_bounds_min is None:
            return True, "No constraints learned yet"
        
        # Check input
        if (x < self.input_bounds_min).any():
            return False, "Input below minimum bound"
        if (x > self.input_bounds_max).any():
            return False, "Input above maximum bound"
        
        # Check output if provided
        if y is not None:
            if (y < self.output_bounds_min).any():
                return False, "Output below minimum bound"
            if (y > self.output_bounds_max).any():
                return False, "Output above maximum bound"
        
        return True, "All constraints satisfied"
    
    def generate_insight(self) -> Dict:
        """
        Governance-specific insights for FL
        """
        insight = super().generate_insight()
        
        recent_input_violations = sum(self.input_violations[-100:]) if len(self.input_violations) >= 100 else sum(self.input_violations)
        recent_output_violations = sum(self.output_violations[-100:]) if len(self.output_violations) >= 100 else sum(self.output_violations)
        
        insight.update({
            "specialization": "governance",
            "input_violation_rate": float(recent_input_violations / min(100, len(self.input_violations))) if self.input_violations else 0.0,
            "output_violation_rate": float(recent_output_violations / min(100, len(self.output_violations))) if self.output_violations else 0.0,
            "constraints_learned": self.input_bounds_min is not None,
            "safety_margin": float(self.safety_margin),
        })
        
        return insight

