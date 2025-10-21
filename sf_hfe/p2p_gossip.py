"""
P2P Gossip Protocol
Decentralized expert weight exchange between similar clients
"""

import logging
from typing import List, Dict, Tuple
import time
import numpy as np

from config import P2P_CONFIG


class P2PGossipManager:
    """
    Manages P2P gossip communication between clients
    
    Responsibilities:
    - Topology formation (using PeerSelectionExpert)
    - Asynchronous weight exchange
    - Gossip scheduling
    - Hysteresis-based stability
    """
    
    def __init__(self, clients: List):
        self.clients = clients
        self.num_clients = len(clients)
        
        # Logger
        self.logger = logging.getLogger("P2PGossip")
        self.logger.info(f"Initializing P2P Gossip for {self.num_clients} clients")
        
        # Topology
        self.topology = {}  # client_id -> [peer_ids]
        self.last_topology_update = {}
        
        # Exchange tracking
        self.exchange_count = 0
        self.exchange_history = []
        
        # Timing
        self.last_exchange_time = time.time()
        self.exchange_frequency = P2P_CONFIG["exchange_frequency"]
        
    def update_topology(self):
        """
        Update P2P topology based on client similarities
        Uses each client's PeerSelectionExpert
        """
        self.logger.info("P2P: Updating topology based on client similarities")
        
        for client in self.clients:
            # Check cooldown
            last_update = self.last_topology_update.get(client.client_id, 0)
            cooldown = STABILITY_CONFIG.get("peer_update_cooldown", 60) if "STABILITY_CONFIG" in dir() else 60
            
            if time.time() - last_update < cooldown:
                continue  # Skip if updated recently
            
            # Select peers
            selected_peers = client.select_peers(self.clients)
            self.topology[client.client_id] = selected_peers
            self.last_topology_update[client.client_id] = time.time()
            
            self.logger.debug(
                f"Client {client.client_id} -> Peers {selected_peers}"
            )
    
    def should_exchange(self) -> bool:
        """
        Check if gossip exchange should happen
        
        Based on time since last exchange
        """
        elapsed = time.time() - self.last_exchange_time
        return elapsed >= self.exchange_frequency
    
    def perform_gossip_round(self):
        """
        Execute one round of P2P gossip exchanges
        
        Each client exchanges weights with its selected peers
        """
        if not P2P_CONFIG["enabled"]:
            return
        
        if not self.should_exchange():
            return
        
        self.logger.info(f"P2P: Starting gossip round #{self.exchange_count + 1}")
        
        # Update topology first
        self.update_topology()
        
        # Track exchanges in this round
        exchanges_done = []
        
        # Each client exchanges with its peers
        for client in self.clients:
            client_id = client.client_id
            peers = self.topology.get(client_id, [])
            
            for peer_id in peers:
                # Avoid duplicate exchanges (A->B and B->A)
                if (peer_id, client_id) in exchanges_done:
                    continue
                
                # Find peer client
                peer_client = next((c for c in self.clients if c.client_id == peer_id), None)
                
                if peer_client is not None:
                    # Perform bidirectional exchange
                    client.sync_with_peer(peer_id, peer_client)
                    
                    exchanges_done.append((client_id, peer_id))
                    
                    self.exchange_history.append({
                        "round": self.exchange_count,
                        "client_a": client_id,
                        "client_b": peer_id,
                        "timestamp": time.time(),
                    })
        
        self.exchange_count += 1
        self.last_exchange_time = time.time()
        
        self.logger.info(f"P2P: Completed {len(exchanges_done)} exchanges")
    
    def get_topology_stats(self) -> Dict:
        """
        Get statistics about P2P topology
        """
        # Compute topology metrics
        if not self.topology:
            return {
                "num_clients": self.num_clients,
                "avg_connections": 0,
                "total_edges": 0,
                "topology": {},
            }
        
        num_connections = [len(peers) for peers in self.topology.values()]
        
        return {
            "num_clients": self.num_clients,
            "avg_connections": np.mean(num_connections) if num_connections else 0,
            "min_connections": min(num_connections) if num_connections else 0,
            "max_connections": max(num_connections) if num_connections else 0,
            "total_edges": sum(num_connections) // 2,  # Undirected edges
            "topology": self.topology,
        }
    
    def get_stats(self) -> Dict:
        """
        Get P2P gossip statistics
        """
        return {
            "exchange_count": self.exchange_count,
            "total_exchanges": len(self.exchange_history),
            "topology_stats": self.get_topology_stats(),
            "recent_exchanges": self.exchange_history[-10:],
        }


# Import STABILITY_CONFIG
try:
    from config import STABILITY_CONFIG
except ImportError:
    # Fallback if not available
    STABILITY_CONFIG = {"peer_update_cooldown": 60}

