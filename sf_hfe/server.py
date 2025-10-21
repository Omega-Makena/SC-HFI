"""
SF-HFE Server
Central Meta-Learning Engine with Global Memory
Developer operates this (with ZERO initial training data)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np
import logging
from collections import defaultdict

from config import FL_CONFIG, META_LEARNING_CONFIG, MONITORING_CONFIG


class GlobalMemory:
    """
    Stores aggregated insights from all clients
    Organizes by domain partitions
    """
    
    def __init__(self):
        self.insights = []  # All insights ever received
        self.domain_partitions = defaultdict(list)  # domain -> insights
        self.client_insights = defaultdict(list)  # client_id -> insights
        
        # Statistics
        self.total_insights = 0
        self.unique_clients = set()
        
    def add_insight(self, insight: Dict):
        """
        Add client insight to global memory
        """
        self.insights.append(insight)
        self.total_insights += 1
        
        client_id = insight.get("client_id")
        if client_id is not None:
            self.unique_clients.add(client_id)
            self.client_insights[client_id].append(insight)
        
        # Partition by domain (if specified)
        domain = insight.get("domain", "general")
        self.domain_partitions[domain].append(insight)
    
    def get_recent_insights(self, n: int = 100) -> List[Dict]:
        """Get n most recent insights"""
        return self.insights[-n:] if len(self.insights) >= n else self.insights
    
    def get_insights_by_domain(self, domain: str) -> List[Dict]:
        """Get all insights for a specific domain"""
        return self.domain_partitions.get(domain, [])
    
    def get_insights_by_client(self, client_id: int) -> List[Dict]:
        """Get all insights from a specific client"""
        return self.client_insights.get(client_id, [])
    
    def stats(self) -> Dict:
        """Global memory statistics"""
        return {
            "total_insights": self.total_insights,
            "unique_clients": len(self.unique_clients),
            "domains": list(self.domain_partitions.keys()),
            "insights_per_domain": {
                domain: len(insights)
                for domain, insights in self.domain_partitions.items()
            },
        }


class OnlineMAMLEngine:
    """
    Online Meta-Learning Engine (MAML-based)
    
    Learns:
    - w_init: Optimal initialization weights for new experts
    - alpha_i: Expert-specific learning rates
    """
    
    def __init__(self, num_experts: int = 10):
        self.num_experts = num_experts
        
        # Meta-parameters
        self.w_init = None  # Will be initialized from first insights
        self.expert_alphas = {i: META_LEARNING_CONFIG["expert_lr_init"] for i in range(num_experts)}
        
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
        """
        Get current meta-parameters for broadcast to clients
        """
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


class SFHFEServer:
    """
    SF-HFE Central Server
    
    Operated by Developer (who has ZERO training data)
    
    Responsibilities:
    - Collect insights from clients
    - Store in Global Memory
    - Run Meta-Learning Engine
    - Broadcast meta-parameters
    """
    
    def __init__(self, num_experts: int = 10):
        self.num_experts = num_experts
        
        # Logger
        self.logger = logging.getLogger("Server")
        self.logger.info("Initializing SF-HFE Server (Developer with ZERO data)")
        
        # Global Memory
        self.global_memory = GlobalMemory()
        
        # Meta-Learning Engine
        self.meta_engine = OnlineMAMLEngine(num_experts=num_experts)
        
        # Communication round tracking
        self.round_count = 0
        self.clients_seen = set()
        
        # Meta-learning trigger state
        self.samples_since_meta = 0
        self.time_since_meta = 0
        self.last_meta_loss = 0.0
        
    def receive_insights(self, insights: List[Dict]) -> Dict:
        """
        Receive insights from clients
        
        Args:
            insights: List of insight dictionaries from clients
        
        Returns:
            Acknowledgment with current meta-parameters
        """
        self.round_count += 1
        
        # Add to global memory
        for insight in insights:
            self.global_memory.add_insight(insight)
            
            client_id = insight.get("client_id")
            if client_id is not None:
                self.clients_seen.add(client_id)
            
            # Track samples
            samples = insight.get("total_samples", 0)
            self.samples_since_meta += samples
        
        self.logger.info(
            f"Server: Round {self.round_count} - "
            f"Received {len(insights)} insights from {len(self.clients_seen)} unique clients"
        )
        
        # Check if meta-learning should trigger
        should_trigger = self._check_meta_trigger(insights)
        
        if should_trigger:
            meta_params = self._trigger_meta_learning(insights)
        else:
            meta_params = self.meta_engine.get_meta_parameters()
        
        return {
            "status": "received",
            "round": self.round_count,
            "meta_params": meta_params,
            "meta_learning_triggered": should_trigger,
        }
    
    def _check_meta_trigger(self, insights: List[Dict]) -> bool:
        """
        Check if meta-learning should be triggered
        
        Conditions (OR logic):
        - Sample count threshold
        - Time threshold
        - Drift detected across clients
        - Performance drop
        """
        triggers = META_LEARNING_CONFIG["triggers"]
        
        # Sample count trigger
        if self.samples_since_meta >= triggers["sample_count"]:
            self.logger.info(f"Server: Meta-learning triggered by sample count ({self.samples_since_meta})")
            return True
        
        # Drift trigger (if multiple clients report drift)
        drift_reports = sum(1 for ins in insights if ins.get("drift_events_count", 0) > 0)
        if drift_reports >= len(insights) * 0.3:  # 30% of clients report drift
            self.logger.info(f"Server: Meta-learning triggered by drift ({drift_reports}/{len(insights)} clients)")
            return True
        
        # Performance drop trigger
        avg_loss = np.mean([ins.get("avg_loss", 0.0) for ins in insights])
        if self.last_meta_loss > 0 and avg_loss > self.last_meta_loss * (1 + triggers["performance_drop"]):
            self.logger.info(f"Server: Meta-learning triggered by performance drop ({avg_loss:.4f} vs {self.last_meta_loss:.4f})")
            return True
        
        return False
    
    def _trigger_meta_learning(self, insights: List[Dict]) -> Dict:
        """
        Trigger meta-learning update
        """
        self.logger.info(f"Server: Running meta-learning update #{self.meta_engine.meta_updates + 1}")
        
        # Get recent insights for meta-learning
        recent_insights = self.global_memory.get_recent_insights(n=100)
        
        # Meta-update
        meta_params = self.meta_engine.meta_update(recent_insights)
        
        # Reset trigger counters
        self.samples_since_meta = 0
        self.last_meta_loss = np.mean([ins.get("avg_loss", 0.0) for ins in insights])
        
        self.logger.info(
            f"Server: Meta-learning complete - "
            f"Updated alphas for {len(meta_params['expert_alphas'])} experts"
        )
        
        return meta_params
    
    def broadcast_meta_parameters(self, clients: List) -> int:
        """
        Broadcast meta-parameters to all clients
        
        Args:
            clients: List of SFHFEClient instances
        
        Returns:
            Number of clients updated
        """
        meta_params = self.meta_engine.get_meta_parameters()
        
        updated = 0
        for client in clients:
            client.receive_meta_parameters(meta_params)
            updated += 1
        
        self.logger.info(f"Server: Broadcast meta-parameters to {updated} clients")
        
        return updated
    
    def get_stats(self) -> Dict:
        """
        Get comprehensive server statistics
        """
        return {
            "round_count": self.round_count,
            "total_insights": self.global_memory.total_insights,
            "unique_clients": len(self.clients_seen),
            "memory_stats": self.global_memory.stats(),
            "meta_engine_stats": self.meta_engine.stats(),
            "samples_since_meta": self.samples_since_meta,
        }

