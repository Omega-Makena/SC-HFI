"""
SF-HFE Server - Federated Learning Component
Central coordinator operated by Developer (with ZERO training data)
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
import threading
import time
import json
from datetime import datetime

from ..config import FL_CONFIG, META_LEARNING_CONFIG, SYSTEM_CONFIG
from .global_memory import GlobalMemory
from .meta_learning import OnlineMAMLEngine
from .initialization import setup_reproducibility, get_client_id_type, safe_get_config


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
        
        # Ensure reproducibility
        setup_reproducibility()
        
        # Logger with structured format
        self.logger = logging.getLogger("Server")
        self.logger.info("Initializing SF-HFE Server (Developer with ZERO data)")
        
        # Global Memory with bounded storage
        max_insights = safe_get_config(FL_CONFIG, "max_insights", 10000)
        self.global_memory = GlobalMemory(max_insights=max_insights)
        
        # Meta-Learning Engine
        self.meta_engine = OnlineMAMLEngine(num_experts=num_experts)
        
        # Communication round tracking
        self.round_count = 0
        self.clients_seen = set()
        self._lock = threading.Lock()  # Thread safety
        
        # Meta-learning trigger state with robust defaults
        self.samples_since_meta = 0
        self.time_since_meta = time.time()
        self.last_meta_loss = 0.0
        
        # Rate limiting for backpressure
        self.request_count = 0
        self.last_rate_reset = time.time()
        self.max_requests_per_minute = safe_get_config(FL_CONFIG, "max_requests_per_minute", 100)
        
        # Standardized client ID type
        self.client_id_type = get_client_id_type()
        
    def _check_rate_limit(self) -> bool:
        """Check if request rate is within limits (backpressure)"""
        current_time = time.time()
        if current_time - self.last_rate_reset > 60:  # Reset every minute
            self.request_count = 0
            self.last_rate_reset = current_time
        
        if self.request_count >= self.max_requests_per_minute:
            self.logger.warning("Rate limit exceeded, rejecting request")
            return False
        
        self.request_count += 1
        return True
    
    def receive_insights(self, insights: List[Dict]) -> Dict:
        """
        Receive insights from clients with robust error handling
        
        Args:
            insights: List of insight dictionaries from clients
        
        Returns:
            Acknowledgment with current meta-parameters
        """
        # Rate limiting check
        if not self._check_rate_limit():
            return {
                "status": "rate_limited",
                "message": "Server is overloaded, please try again later"
            }
        
        with self._lock:  # Thread safety
            if not insights:
                self.logger.warning("Received empty insights list")
                return {
                    "status": "error",
                    "message": "No insights provided"
                }
            
            self.round_count += 1
            
            # Add to global memory with validation
            valid_insights = 0
            for insight in insights:
                try:
                    self.global_memory.add_insight(insight)
                    valid_insights += 1
                    
                    client_id = insight.get("client_id")
                    if client_id is not None:
                        self.clients_seen.add(client_id)
                    
                    # Track samples
                    samples = insight.get("total_samples", 0)
                    if isinstance(samples, (int, float)) and samples >= 0:
                        self.samples_since_meta += int(samples)
                except Exception as e:
                    self.logger.error(f"Error processing insight: {e}")
            
            self.logger.info(
                f"Server: Round {self.round_count} - "
                f"Received {valid_insights}/{len(insights)} valid insights from {len(self.clients_seen)} unique clients"
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
                "valid_insights": valid_insights,
            }
    
    def _check_meta_trigger(self, insights: List[Dict]) -> bool:
        """Check if meta-learning should be triggered with robust edge case handling"""
        try:
            triggers = META_LEARNING_CONFIG.get("triggers", {})
            
            # Sample count trigger with validation
            sample_count_threshold = triggers.get("sample_count", 1000)
            if self.samples_since_meta >= sample_count_threshold:
                self.logger.info(f"Server: Meta-learning triggered by sample count ({self.samples_since_meta})")
                return True
            
            # Time-based trigger with validation
            time_threshold = triggers.get("time_seconds", 300)
            if time.time() - self.time_since_meta >= time_threshold:
                self.logger.info(f"Server: Meta-learning triggered by time threshold")
                return True
            
            # Drift trigger with robust validation
            if insights:
                drift_reports = 0
                valid_insights = 0
                
                for ins in insights:
                    if isinstance(ins, dict):
                        valid_insights += 1
                        drift_count = ins.get("drift_events_count", 0)
                        if isinstance(drift_count, (int, float)) and drift_count > 0:
                            drift_reports += 1
                
                if valid_insights > 0 and drift_reports >= valid_insights * 0.3:  # 30% of clients report drift
                    self.logger.info(f"Server: Meta-learning triggered by drift ({drift_reports}/{valid_insights} clients)")
                    return True
            
            # Performance drop trigger with NaN/infinity protection
            if insights:
                losses = []
                for ins in insights:
                    if isinstance(ins, dict):
                        loss = ins.get("avg_loss", 0.0)
                        if isinstance(loss, (int, float)) and not (np.isnan(loss) or np.isinf(loss)):
                            losses.append(float(loss))
                
                if losses and self.last_meta_loss > 0:
                    avg_loss = np.mean(losses)
                    performance_drop_threshold = triggers.get("performance_drop", 0.15)
                    
                    if avg_loss > self.last_meta_loss * (1 + performance_drop_threshold):
                        self.logger.info(f"Server: Meta-learning triggered by performance drop ({avg_loss:.4f} vs {self.last_meta_loss:.4f})")
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in meta-learning trigger check: {e}")
            return False  # Fail safe
    
    def _trigger_meta_learning(self, insights: List[Dict]) -> Dict:
        """Trigger meta-learning update with robust error handling"""
        try:
            self.logger.info(f"Server: Running meta-learning update #{self.meta_engine.meta_updates + 1}")
            
            # Get recent insights for meta-learning
            recent_insights = self.global_memory.get_recent_insights(n=100)
            
            # Meta-update
            meta_params = self.meta_engine.meta_update(recent_insights)
            
            # Reset trigger counters
            self.samples_since_meta = 0
            self.time_since_meta = time.time()
            
            # Update last meta loss with validation
            if insights:
                losses = []
                for ins in insights:
                    if isinstance(ins, dict):
                        loss = ins.get("avg_loss", 0.0)
                        if isinstance(loss, (int, float)) and not (np.isnan(loss) or np.isinf(loss)):
                            losses.append(float(loss))
                
                if losses:
                    self.last_meta_loss = float(np.mean(losses))
            
            self.logger.info(
                f"Server: Meta-learning complete - "
                f"Updated alphas for {len(meta_params.get('expert_alphas', {}))} experts"
            )
            
            return meta_params
            
        except Exception as e:
            self.logger.error(f"Error in meta-learning update: {e}")
            # Return current parameters as fallback
            return self.meta_engine.get_meta_parameters()
    
    def broadcast_meta_parameters(self, clients: List) -> int:
        """Broadcast meta-parameters to all clients"""
        with self._lock:  # Thread safety
            meta_params = self.meta_engine.get_meta_parameters()
            
            updated = 0
            for client in clients:
                try:
                    client.receive_meta_parameters(meta_params)
                    updated += 1
                except Exception as e:
                    self.logger.error(f"Error broadcasting to client: {e}")
            
            self.logger.info(f"Server: Broadcast meta-parameters to {updated} clients")
            
            return updated
    
    def get_stats(self) -> Dict:
        """Get comprehensive server statistics"""
        with self._lock:  # Thread safety
            return {
                "round_count": self.round_count,
                "total_insights": self.global_memory.total_insights,
                "unique_clients": len(self.clients_seen),
                "memory_stats": self.global_memory.stats(),
                "meta_engine_stats": self.meta_engine.stats(),
                "samples_since_meta": self.samples_since_meta,
                "rate_limit_stats": {
                    "requests_this_minute": self.request_count,
                    "max_requests_per_minute": self.max_requests_per_minute,
                },
            }

