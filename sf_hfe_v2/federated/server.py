"""
SF-HFE Server - Federated Learning Component
Central coordinator operated by Developer (with ZERO training data)
"""

import logging
from typing import Dict, List
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FL_CONFIG, META_LEARNING_CONFIG
from federated.global_memory import GlobalMemory
from federated.meta_learning import OnlineMAMLEngine


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
        """Check if meta-learning should be triggered"""
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
        """Trigger meta-learning update"""
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
        """Broadcast meta-parameters to all clients"""
        meta_params = self.meta_engine.get_meta_parameters()
        
        updated = 0
        for client in clients:
            client.receive_meta_parameters(meta_params)
            updated += 1
        
        self.logger.info(f"Server: Broadcast meta-parameters to {updated} clients")
        
        return updated
    
    def get_stats(self) -> Dict:
        """Get comprehensive server statistics"""
        return {
            "round_count": self.round_count,
            "total_insights": self.global_memory.total_insights,
            "unique_clients": len(self.clients_seen),
            "memory_stats": self.global_memory.stats(),
            "meta_engine_stats": self.meta_engine.stats(),
            "samples_since_meta": self.samples_since_meta,
        }

