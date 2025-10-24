"""
Global Memory - Federated Learning Component
Stores and organizes insights from all clients (Developer's knowledge base)
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict
import threading
import json
import logging
from datetime import datetime


class GlobalMemory:
    """
    Stores aggregated insights from all clients
    Organizes by domain partitions
    
    This is the ONLY thing the Developer learns from (no raw data!)
    """
    
    def __init__(self, max_insights: int = 10000):
        self.max_insights = max_insights
        self.logger = logging.getLogger("GlobalMemory")
        self._lock = threading.Lock()  # Thread safety
        
        # Bounded storage to prevent unlimited growth
        self.insights = []  # All insights ever received (bounded)
        self.domain_partitions = defaultdict(list)  # domain -> insights
        self.client_insights = defaultdict(list)  # client_id -> insights
        
        # Statistics
        self.total_insights = 0
        self.unique_clients = set()
        
        # Insight validation schema
        self.required_fields = {
            "client_id": (str, int),
            "expert_insights": dict,
            "avg_loss": (int, float),
            "total_samples": int
        }
        
    def _validate_insight(self, insight: Dict) -> bool:
        """Validate insight schema"""
        try:
            for field, expected_type in self.required_fields.items():
                if field not in insight:
                    self.logger.warning(f"Missing field {field} in insight")
                    return False
                if not isinstance(insight[field], expected_type):
                    self.logger.warning(f"Invalid type for field {field}")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Error validating insight: {e}")
            return False
    
    def _trim_memory(self):
        """Trim memory to prevent unlimited growth"""
        if len(self.insights) > self.max_insights:
            # Remove oldest insights (FIFO)
            to_remove = len(self.insights) - self.max_insights
            removed_insights = self.insights[:to_remove]
            self.insights = self.insights[to_remove:]
            
            # Clean up domain partitions and client insights
            for insight in removed_insights:
                client_id = insight.get("client_id")
                domain = insight.get("domain", "general")
                
                if client_id in self.client_insights:
                    try:
                        self.client_insights[client_id].remove(insight)
                    except ValueError:
                        pass  # Already removed
                
                if domain in self.domain_partitions:
                    try:
                        self.domain_partitions[domain].remove(insight)
                    except ValueError:
                        pass  # Already removed
            
            self.logger.info(f"Trimmed memory: removed {to_remove} old insights")
    
    def add_insight(self, insight: Dict):
        """Add client insight to global memory with validation and bounded storage"""
        with self._lock:  # Thread safety
            if not self._validate_insight(insight):
                self.logger.warning("Skipping invalid insight")
                return
            
            self.insights.append(insight)
            self.total_insights += 1
            
            client_id = insight.get("client_id")
            if client_id is not None:
                self.unique_clients.add(client_id)
                self.client_insights[client_id].append(insight)
            
            # Partition by domain (if specified)
            domain = insight.get("domain", "general")
            self.domain_partitions[domain].append(insight)
            
            # Trim memory if needed
            self._trim_memory()
    
    def get_recent_insights(self, n: int = 100) -> List[Dict]:
        """Get n most recent insights"""
        with self._lock:  # Thread safety
            return self.insights[-n:] if len(self.insights) >= n else self.insights
    
    def get_insights_by_domain(self, domain: str) -> List[Dict]:
        """Get all insights for a specific domain"""
        with self._lock:  # Thread safety
            return self.domain_partitions.get(domain, [])
    
    def get_insights_by_client(self, client_id: int) -> List[Dict]:
        """Get all insights from a specific client"""
        with self._lock:  # Thread safety
            return self.client_insights.get(client_id, [])
    
    def stats(self) -> Dict:
        """Global memory statistics"""
        with self._lock:  # Thread safety
            return {
                "total_insights": self.total_insights,
                "current_insights": len(self.insights),
                "max_insights": self.max_insights,
                "unique_clients": len(self.unique_clients),
                "domains": list(self.domain_partitions.keys()),
                "insights_per_domain": {
                    domain: len(insights)
                    for domain, insights in self.domain_partitions.items()
                },
                "memory_utilization": len(self.insights) / self.max_insights,
            }

