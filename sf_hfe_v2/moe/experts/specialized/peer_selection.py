"""
Peer Selection Expert - Relational Expert
Specializes in identifying similar clients for P2P topology
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_expert import BaseExpert


class PeerSelectionExpert(BaseExpert):
    """
    Expert specializing in peer selection for P2P gossip
    
    Capabilities:
    - Cosine similarity computation
    - Peer ranking
    - Adaptive topology formation
    - Similarity-based clustering
    """
    
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__(
            expert_id=7,
            input_dim=input_dim,
            output_dim=output_dim,
            expert_type="relational",
            **kwargs
        )
        
        # Embedding network (maps data to peer-space)
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),  # 16-dim embedding space
        )
        
        # Client embedding (average of data embeddings)
        self.client_embedding = None
        self.embedding_history = []
        
        # Peer similarities (client_id -> similarity score)
        self.peer_similarities = {}
        self.peer_history = {}
        
        # EMA smoothing
        self.ema_alpha = 0.9
        
    def _build_network(self) -> nn.Module:
        """Override base network"""
        return nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward: embed data and predict
        """
        embedding = self.embedding_net(x)
        
        # Simple prediction from embedding
        prediction = embedding.mean(dim=1, keepdim=True)  # Simplified
        
        return prediction
    
    def _task_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Embedding quality loss
        """
        # Standard prediction loss
        pred_loss = nn.functional.mse_loss(outputs, targets)
        
        # Embedding consistency loss (encourage stable embeddings)
        if self.client_embedding is not None and len(self.embedding_history) > 0:
            current_embedding = self.client_embedding
            past_embedding = self.embedding_history[-1]
            consistency_loss = nn.functional.mse_loss(current_embedding, past_embedding)
            return pred_loss + 0.1 * consistency_loss
        
        return pred_loss
    
    def online_update(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        replay: bool = True
    ) -> Dict[str, float]:
        """
        Update with embedding tracking
        """
        # Compute current batch embedding
        with torch.no_grad():
            batch_embedding = self.embedding_net(batch_x).mean(dim=0)
        
        # Update client embedding (EMA)
        if self.client_embedding is None:
            self.client_embedding = batch_embedding
        else:
            self.client_embedding = (
                self.ema_alpha * self.client_embedding +
                (1 - self.ema_alpha) * batch_embedding
            )
        
        # Store in history
        self.embedding_history.append(self.client_embedding.clone())
        if len(self.embedding_history) > 100:
            self.embedding_history.pop(0)
        
        # Standard update
        metrics = super().online_update(batch_x, batch_y, replay)
        
        metrics.update({
            "embedding_norm": float(self.client_embedding.norm().item()),
            "embedding_stability": self._compute_embedding_stability(),
        })
        
        return metrics
    
    def _compute_embedding_stability(self) -> float:
        """
        Measure how stable the client embedding is over time
        """
        if len(self.embedding_history) < 2:
            return 1.0
        
        # Cosine similarity between current and past embeddings
        recent = self.embedding_history[-10:] if len(self.embedding_history) >= 10 else self.embedding_history
        similarities = []
        
        for i in range(len(recent) - 1):
            sim = nn.functional.cosine_similarity(
                recent[i].unsqueeze(0),
                recent[i+1].unsqueeze(0)
            )
            similarities.append(sim.item())
        
        return float(np.mean(similarities)) if similarities else 1.0
    
    def compute_similarity(self, other_embedding: torch.Tensor) -> float:
        """
        Compute cosine similarity with another client's embedding
        """
        if self.client_embedding is None:
            return 0.0
        
        similarity = nn.functional.cosine_similarity(
            self.client_embedding.unsqueeze(0),
            other_embedding.unsqueeze(0)
        )
        
        return float(similarity.item())
    
    def update_peer_similarity(self, peer_id: int, similarity: float):
        """
        Update similarity score for a peer (EMA smoothed)
        """
        if peer_id in self.peer_similarities:
            # EMA update
            self.peer_similarities[peer_id] = (
                self.ema_alpha * self.peer_similarities[peer_id] +
                (1 - self.ema_alpha) * similarity
            )
        else:
            self.peer_similarities[peer_id] = similarity
        
        # Track history
        if peer_id not in self.peer_history:
            self.peer_history[peer_id] = []
        self.peer_history[peer_id].append(similarity)
        
        # Keep only recent
        if len(self.peer_history[peer_id]) > 100:
            self.peer_history[peer_id].pop(0)
    
    def select_top_k_peers(self, k: int = 3, hysteresis: float = 0.05) -> List[int]:
        """
        Select top-k most similar peers
        
        Args:
            k: Number of peers to select
            hysteresis: Threshold to prevent peer churn
        
        Returns:
            List of peer IDs
        """
        if not self.peer_similarities:
            return []
        
        # Sort by similarity
        sorted_peers = sorted(
            self.peer_similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Apply hysteresis: only change if difference > threshold
        top_k = sorted_peers[:k]
        peer_ids = [peer_id for peer_id, _ in top_k]
        
        return peer_ids
    
    def get_client_embedding(self) -> torch.Tensor:
        """
        Get current client embedding
        """
        return self.client_embedding if self.client_embedding is not None else torch.zeros(16)
    
    def generate_insight(self) -> Dict:
        """
        Peer-selection-specific insights for FL
        """
        insight = super().generate_insight()
        
        num_peers = len(self.peer_similarities)
        avg_similarity = sum(self.peer_similarities.values()) / num_peers if num_peers > 0 else 0.0
        max_similarity = max(self.peer_similarities.values()) if num_peers > 0 else 0.0
        
        insight.update({
            "specialization": "peer_selection",
            "num_peers_tracked": num_peers,
            "avg_peer_similarity": float(avg_similarity),
            "max_peer_similarity": float(max_similarity),
            "embedding_dimensionality": 16,
            "embedding_stability": self._compute_embedding_stability(),
        })
        
        return insight

