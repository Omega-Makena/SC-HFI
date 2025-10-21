"""
Global Memory - Federated Learning Component
Stores and organizes insights from all clients (Developer's knowledge base)
"""

from typing import Dict, List
from collections import defaultdict


class GlobalMemory:
    """
    Stores aggregated insights from all clients
    Organizes by domain partitions
    
    This is the ONLY thing the Developer learns from (no raw data!)
    """
    
    def __init__(self):
        self.insights = []  # All insights ever received
        self.domain_partitions = defaultdict(list)  # domain -> insights
        self.client_insights = defaultdict(list)  # client_id -> insights
        
        # Statistics
        self.total_insights = 0
        self.unique_clients = set()
        
    def add_insight(self, insight: Dict):
        """Add client insight to global memory"""
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

