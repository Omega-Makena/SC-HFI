"""
Temporal Expert - Core Structure Expert #2
Specializes in sequential patterns and temporal dynamics using LSTM
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_expert import BaseExpert


class TemporalExpert(BaseExpert):
    """
    Expert specializing in temporal/sequential patterns
    
    Capabilities:
    - LSTM-based sequence modeling
    - Temporal dependency learning
    - Time-series prediction
    - Recurrent pattern recognition
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, **kwargs):
        self.hidden_dim = hidden_dim
        
        super().__init__(
            expert_id=1,
            input_dim=input_dim,
            output_dim=output_dim,
            expert_type="structure",
            **kwargs
        )
        
        # Now create LSTM architecture (after super() init)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Create optimizer now that we have parameters
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
        
        # Hidden state (maintained across batches for temporal continuity)
        self.hidden_state = None
        self.cell_state = None
        
        # Temporal metrics
        self.sequence_lengths = []
        self.temporal_correlations = []
        
    def _build_network(self) -> nn.Module:
        """Override base network (we use LSTM instead)"""
        return nn.Identity()  # Placeholder
    
    def forward(self, x: torch.Tensor, reset_hidden: bool = False) -> torch.Tensor:
        """
        Forward pass through LSTM
        
        Args:
            x: Input tensor [batch_size, input_dim] or [batch_size, seq_len, input_dim]
            reset_hidden: Whether to reset hidden state
        """
        # Ensure 3D input [batch, seq, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        batch_size = x.size(0)
        
        # Initialize or reset hidden state
        if reset_hidden or self.hidden_state is None or self.hidden_state.size(1) != batch_size:
            self.hidden_state = torch.zeros(2, batch_size, self.hidden_dim, device=x.device)
            self.cell_state = torch.zeros(2, batch_size, self.hidden_dim, device=x.device)
        
        # LSTM forward
        lstm_out, (self.hidden_state, self.cell_state) = self.lstm(
            x, (self.hidden_state.detach(), self.cell_state.detach())
        )
        
        # Take last time step
        last_output = lstm_out[:, -1, :]
        
        # Final prediction
        output = self.fc(last_output)
        
        return output
    
    def _task_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Temporal prediction loss
        """
        # Standard MSE loss
        pred_loss = nn.functional.mse_loss(outputs, targets)
        
        # Temporal smoothness regularization (encourage smooth predictions)
        if self.hidden_state is not None and len(self.loss_history) > 1:
            # Penalize large jumps in predictions
            smoothness_loss = torch.mean(torch.abs(outputs[1:] - outputs[:-1]))
            return pred_loss + 0.01 * smoothness_loss
        
        return pred_loss
    
    def online_update(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        replay: bool = True
    ) -> Dict[str, float]:
        """
        Update with temporal awareness
        """
        # Standard update but maintain hidden state
        self.train()
        self.activation_count += 1
        self.batch_count += 1
        self.total_samples_seen += len(batch_x)
        
        # Forward pass (maintains hidden state)
        outputs = self.forward(batch_x, reset_hidden=False)
        current_loss = self.compute_loss(outputs, batch_y)
        
        # Replay
        replay_loss = 0.0
        if replay and self.memory.total_size() > 0:
            replay_x, replay_y = self.memory.replay_batch(batch_size=32)
            if replay_x is not None:
                replay_outputs = self.forward(replay_x, reset_hidden=True)
                replay_loss = self.compute_loss(replay_outputs, replay_y, include_ewc=False)
        
        # Combined loss
        total_loss = 0.7 * current_loss + 0.3 * replay_loss if replay_loss > 0 else current_loss
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        # Store in memory
        uncertainties = self._compute_uncertainty(batch_x)
        self.memory.add_batch(batch_x, batch_y, uncertainties)
        
        # Update loss tracking
        loss_value = total_loss.item()
        self.current_loss = loss_value
        self.ema_loss = 0.95 * self.ema_loss + 0.05 * loss_value
        self.loss_history.append(loss_value)
        
        # Temporal metrics
        self._update_temporal_metrics(batch_x)
        
        metrics = {
            "loss": loss_value,
            "ema_loss": self.ema_loss,
            "replay_loss": replay_loss.item() if isinstance(replay_loss, torch.Tensor) else replay_loss,
            "hidden_state_norm": self.hidden_state.norm().item() if self.hidden_state is not None else 0.0,
            "temporal_correlation": self.temporal_correlations[-1] if self.temporal_correlations else 0.0,
        }
        
        return metrics
    
    def _update_temporal_metrics(self, batch_x: torch.Tensor):
        """
        Track temporal statistics
        """
        # Compute autocorrelation (simple version)
        if len(batch_x) > 1:
            x_mean = batch_x.mean()
            x_std = batch_x.std()
            if x_std > 1e-6:
                # Lag-1 autocorrelation
                corr = ((batch_x[:-1] - x_mean) * (batch_x[1:] - x_mean)).mean() / (x_std ** 2)
                self.temporal_correlations.append(corr.item())
                
                # Keep only recent
                if len(self.temporal_correlations) > 100:
                    self.temporal_correlations.pop(0)
    
    def reset_temporal_state(self):
        """
        Reset hidden state (e.g., when concept drift detected)
        """
        self.hidden_state = None
        self.cell_state = None
    
    def generate_insight(self) -> Dict:
        """
        Temporal-specific insights for FL
        """
        insight = super().generate_insight()
        
        avg_corr = sum(self.temporal_correlations) / len(self.temporal_correlations) if self.temporal_correlations else 0.0
        
        insight.update({
            "specialization": "temporal",
            "temporal_correlation": float(avg_corr),
            "hidden_state_active": self.hidden_state is not None,
            "sequence_memory_depth": 2,  # LSTM layers
        })
        
        return insight

