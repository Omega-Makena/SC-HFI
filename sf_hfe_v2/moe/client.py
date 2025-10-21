"""
SF-HFE Client
User device with 10 experts + router (MoE architecture)
Performs online continual learning on local data stream
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

from config import (
    EXPERT_CONFIG, MEMORY_CONFIG, STREAM_CONFIG,
    LEARNING_CONFIG, P2P_CONFIG, MONITORING_CONFIG
)
from router import ContextualBanditRouter
from experts import (
    GeometryExpert, TemporalExpert, ReconstructionExpert,
    CausalInferenceExpert, DriftDetectionExpert,
    GovernanceExpert, StatisticalConsistencyExpert,
    PeerSelectionExpert, MetaAdaptationExpert,
    MemoryConsolidationExpert
)


class SFHFEClient:
    """
    SF-HFE Client Node
    
    Represents a user device that:
    - Owns local private data
    - Trains 10 specialized experts online
    - Uses router to select active experts
    - Generates insights for FL
    - Exchanges weights via P2P gossip
    - Learns continuously from data stream
    """
    
    def __init__(
        self,
        client_id: int,
        input_dim: int = 20,
        output_dim: int = 1,
        has_data: bool = True
    ):
        self.client_id = client_id
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.has_data = has_data  # True for User, False for Developer
        
        # Logger
        self.logger = logging.getLogger(f"Client{client_id}")
        self.logger.info(f"Initializing SF-HFE Client {client_id} ({'User with data' if has_data else 'Developer with NO data'})")
        
        # Initialize all 10 experts
        self.experts = self._initialize_experts()
        
        # Initialize router (Cross-Dimension Expert)
        self.router = ContextualBanditRouter(input_dim=input_dim, num_experts=10)
        
        self.logger.info(f"Client {client_id}: Initialized {len(self.experts)} experts + router")
        
        # Local data stream buffer
        self.stream_buffer = []
        self.batch_size = STREAM_CONFIG["mini_batch_size"]
        
        # Training state
        self.batch_count = 0
        self.total_samples_processed = 0
        
        # Meta-parameters (received from server)
        self.meta_params = None
        
        # P2P state
        self.connected_peers = []  # List of peer client IDs
        self.peer_embeddings = {}  # peer_id -> embedding
        
        # Performance tracking
        self.loss_history = []
        self.drift_events = []
        
    def _initialize_experts(self) -> List[nn.Module]:
        """
        Initialize all 10 specialized experts
        
        Returns:
            List of expert modules
        """
        experts = [
            GeometryExpert(self.input_dim, self.output_dim),              # 0
            TemporalExpert(self.input_dim, self.output_dim),              # 1
            ReconstructionExpert(self.input_dim, self.output_dim),        # 2
            CausalInferenceExpert(self.input_dim, self.output_dim),       # 3
            DriftDetectionExpert(self.input_dim, self.output_dim),        # 4
            GovernanceExpert(self.input_dim, self.output_dim),            # 5
            StatisticalConsistencyExpert(self.input_dim, self.output_dim),# 6
            PeerSelectionExpert(self.input_dim, self.output_dim),         # 7
            MetaAdaptationExpert(self.input_dim, self.output_dim),        # 8
            MemoryConsolidationExpert(self.input_dim, self.output_dim),   # 9
        ]
        
        return experts
    
    def process_stream_batch(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> Dict:
        """
        Process a mini-batch from the data stream
        
        This is the MAIN LEARNING FUNCTION
        
        Args:
            batch_x: Input mini-batch [batch_size, input_dim]
            batch_y: Target mini-batch [batch_size, output_dim]
        
        Returns:
            Metrics dictionary
        """
        if not self.has_data:
            self.logger.warning(f"Client {self.client_id}: Developer has no data - cannot train")
            return {}
        
        self.batch_count += 1
        self.total_samples_processed += len(batch_x)
        
        # Step 1: Router selects top-K experts
        selected_indices, routing_weights = self.router.select_experts(batch_x, deterministic=False)
        
        # Step 2: Each selected expert performs online update
        expert_metrics = {}
        expert_losses = []
        
        for i in range(self.batch_size):
            batch_expert_losses = []
            
            for j in range(self.router.top_k):
                expert_idx = selected_indices[i, j].item()
                expert = self.experts[expert_idx]
                
                # Expert performs online update
                metrics = expert.online_update(
                    batch_x[i:i+1],
                    batch_y[i:i+1],
                    replay=(self.batch_count % MEMORY_CONFIG["replay_frequency"] == 0)
                )
                
                if expert_idx not in expert_metrics:
                    expert_metrics[expert_idx] = []
                expert_metrics[expert_idx].append(metrics)
                
                batch_expert_losses.append(metrics["loss"])
            
            expert_losses.append(torch.tensor(batch_expert_losses))
        
        # Step 3: Update router statistics
        expert_losses_tensor = torch.stack(expert_losses)
        self.router.update_statistics(selected_indices, expert_losses_tensor)
        
        # Step 4: Get final predictions for router training
        predictions, router_info = self.router.forward(
            batch_x,
            self.experts,
            targets=batch_y,
            train=True
        )
        
        # Step 5: Train router
        router_loss = self.router.train_router(
            predictions,
            batch_y,
            router_info["routing_entropy"]
        )
        
        # Step 6: Meta-Adaptation Expert monitors all experts
        meta_expert = self.experts[8]  # MetaAdaptationExpert
        for expert_idx, metrics_list in expert_metrics.items():
            if expert_idx != 8:  # Don't monitor itself
                avg_loss = np.mean([m["loss"] for m in metrics_list])
                # Get gradient norm (simplified - use loss as proxy)
                grad_norm = avg_loss  # Simplified
                current_lr = self.experts[expert_idx].lr
                
                meta_expert.monitor_expert(expert_idx, avg_loss, grad_norm, current_lr)
        
        # Step 7: Apply LR recommendations from Meta-Adaptation Expert
        for expert_idx in range(10):
            if expert_idx != 8:
                recommended_lr = meta_expert.get_lr_recommendation(expert_idx)
                if recommended_lr == 0.0:
                    # Freeze signal
                    if not self.experts[expert_idx].is_frozen:
                        self.experts[expert_idx].freeze()
                        self.logger.info(f"Client {self.client_id}: Froze Expert {expert_idx}")
                else:
                    # Apply LR
                    self.experts[expert_idx].adapt_learning_rate(recommended_lr)
        
        # Step 8: Check for drift and handle
        drift_expert = self.experts[4]  # DriftDetectionExpert
        if drift_expert.drift_detected:
            self._handle_drift()
            drift_expert.reset_drift_flag()
        
        # Step 9: Memory consolidation if needed
        memory_expert = self.experts[9]  # MemoryConsolidationExpert
        if memory_expert.should_consolidate(self.batch_count):
            for expert in self.experts:
                memory_expert.update_memory_pressure(expert.memory)
                consolidation_info = memory_expert.consolidate_memory(expert.memory)
                if consolidation_info["consolidated"] > 0:
                    self.logger.debug(
                        f"Client {self.client_id}: Consolidated {consolidation_info['consolidated']} "
                        f"samples for Expert {expert.expert_id}"
                    )
        
        # Aggregate metrics
        avg_loss = np.mean([m["loss"] for metrics_list in expert_metrics.values() for m in metrics_list])
        self.loss_history.append(avg_loss)
        
        summary = {
            "batch": self.batch_count,
            "samples_processed": self.total_samples_processed,
            "avg_loss": avg_loss,
            "router_loss": router_loss,
            "routing_entropy": router_info["routing_entropy"],
            "selected_experts": router_info["selected_experts"],
            "num_active_experts": len(expert_metrics),
        }
        
        return summary
    
    def _handle_drift(self):
        """
        Handle concept drift detection
        """
        self.logger.info(f"Client {self.client_id}: Concept drift detected! Triggering adaptations...")
        
        # Reset temporal expert's hidden state
        temporal_expert = self.experts[1]
        temporal_expert.reset_temporal_state()
        
        # Trigger memory consolidation
        memory_expert = self.experts[9]
        for expert in self.experts:
            memory_expert.consolidate_memory(expert.memory)
        
        # Log event
        self.drift_events.append(self.batch_count)
    
    def generate_insights(self) -> Dict:
        """
        Generate insights for Federated Learning
        
        Returns metadata (NOT raw data or weights) for Meta-Learner
        """
        insights = {
            "client_id": self.client_id,
            "has_data": self.has_data,
            "total_samples": self.total_samples_processed,
            "batch_count": self.batch_count,
            "avg_loss": np.mean(self.loss_history[-100:]) if len(self.loss_history) >= 100 else 0.0,
        }
        
        # Collect insights from all experts
        expert_insights = {}
        for expert in self.experts:
            expert_insight = expert.generate_insight()
            expert_insights[expert.expert_name] = expert_insight
        
        insights["expert_insights"] = expert_insights
        
        # Router insights
        insights["router"] = self.router.generate_insight()
        
        # Drift information
        insights["drift_events_count"] = len(self.drift_events)
        insights["batches_since_last_drift"] = (
            self.batch_count - self.drift_events[-1]
            if self.drift_events else self.batch_count
        )
        
        return insights
    
    def receive_meta_parameters(self, meta_params: Dict):
        """
        Receive global meta-parameters from server
        
        Args:
            meta_params: Dictionary containing w_init and alpha_i
        """
        self.meta_params = meta_params
        
        # Apply meta-learned initialization (if we're restarting experts)
        if "w_init" in meta_params and "apply_to_new_experts" in meta_params:
            # This would be used when adding new experts or reinitializing
            pass
        
        # Apply expert-specific learning rates
        if "expert_alphas" in meta_params:
            for expert_idx, alpha in meta_params["expert_alphas"].items():
                if 0 <= expert_idx < len(self.experts):
                    self.experts[expert_idx].adapt_learning_rate(alpha)
        
        self.logger.info(f"Client {self.client_id}: Received meta-parameters from server")
    
    def get_expert_weights(self, expert_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get weights from a specific expert (for P2P exchange)
        """
        if 0 <= expert_idx < len(self.experts):
            return self.experts[expert_idx].get_weights()
        return {}
    
    def set_expert_weights(
        self,
        expert_idx: int,
        weights: Dict[str, torch.Tensor],
        blend_factor: float = 0.5
    ):
        """
        Set weights for a specific expert (from P2P peer)
        """
        if 0 <= expert_idx < len(self.experts):
            self.experts[expert_idx].set_weights(weights, blend_factor)
    
    def sync_with_peer(self, peer_id: int, peer_client: 'SFHFEClient'):
        """
        P2P gossip: exchange weights with a peer
        
        Exchanges only the most active experts' weights
        """
        if not P2P_CONFIG["enabled"]:
            return
        
        # Get most active experts from router
        most_active = self.router.get_most_active_experts(k=P2P_CONFIG["exchange_top_n"])
        
        # Exchange weights for each active expert
        for expert_idx in most_active:
            # Get peer's weights
            peer_weights = peer_client.get_expert_weights(expert_idx)
            
            # Update our weights (blend)
            self.set_expert_weights(
                expert_idx,
                peer_weights,
                blend_factor=P2P_CONFIG["aggregation_weight"]
            )
            
            # Give our weights to peer
            our_weights = self.get_expert_weights(expert_idx)
            peer_client.set_expert_weights(
                expert_idx,
                our_weights,
                blend_factor=P2P_CONFIG["aggregation_weight"]
            )
        
        self.logger.info(
            f"Client {self.client_id} <-> Client {peer_id}: "
            f"Synced {len(most_active)} experts ({most_active})"
        )
    
    def compute_peer_similarity(self, peer_client: 'SFHFEClient') -> float:
        """
        Compute similarity with another client
        Uses PeerSelectionExpert (expert #7)
        """
        peer_expert = self.experts[7]  # PeerSelectionExpert
        
        # Get our embedding
        our_embedding = peer_expert.get_client_embedding()
        
        # Get peer's embedding
        peer_peer_expert = peer_client.experts[7]
        peer_embedding = peer_peer_expert.get_client_embedding()
        
        # Compute similarity
        similarity = peer_expert.compute_similarity(peer_embedding)
        
        # Update similarity tracking
        peer_expert.update_peer_similarity(peer_client.client_id, similarity)
        
        return similarity
    
    def select_peers(self, all_clients: List['SFHFEClient']) -> List[int]:
        """
        Select top-K similar peers for P2P exchange
        
        Args:
            all_clients: List of all clients in the system
        
        Returns:
            List of selected peer IDs
        """
        peer_expert = self.experts[7]  # PeerSelectionExpert
        
        # Compute similarity with all other clients
        for client in all_clients:
            if client.client_id != self.client_id:
                self.compute_peer_similarity(client)
        
        # Select top-K
        selected_peers = peer_expert.select_top_k_peers(
            k=P2P_CONFIG["top_k_peers"],
            hysteresis=P2P_CONFIG["hysteresis_threshold"]
        )
        
        self.connected_peers = selected_peers
        
        return selected_peers
    
    def get_stats(self) -> Dict:
        """
        Get comprehensive client statistics
        """
        return {
            "client_id": self.client_id,
            "has_data": self.has_data,
            "total_samples": self.total_samples_processed,
            "batch_count": self.batch_count,
            "loss_history": self.loss_history[-100:],
            "avg_loss": np.mean(self.loss_history[-100:]) if len(self.loss_history) >= 100 else 0.0,
            "drift_events": len(self.drift_events),
            "router_stats": self.router.get_expert_statistics(),
            "expert_stats": [expert.stats() for expert in self.experts],
            "connected_peers": self.connected_peers,
        }

