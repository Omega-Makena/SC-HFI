"""
P2P Gossip Protocol
Decentralized expert weight exchange between similar clients
"""

import logging
from typing import List, Dict, Tuple, Set, Any, Optional
import time
import numpy as np
import threading
from collections import defaultdict

from ..config import P2P_CONFIG, STABILITY_CONFIG


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
        
        # Topology with thread safety
        self.topology = {}  # client_id -> [peer_ids]
        self.last_topology_update = {}
        self._lock = threading.Lock()  # Thread safety
        
        # Exchange tracking with duplicate prevention
        self.exchange_count = 0
        self.exchange_history = []
        self.active_exchanges = set()  # Track ongoing exchanges to prevent duplicates
        
        # Timing
        self.last_exchange_time = time.time()
        self.exchange_frequency = P2P_CONFIG["exchange_frequency"]
        
        # Client health tracking
        self.client_health = {client.client_id: {"last_seen": time.time(), "active": True} for client in clients}
        
    def _is_client_healthy(self, client_id: int) -> bool:
        """Check if client is healthy and active"""
        health_info = self.client_health.get(client_id, {"active": False})
        return health_info.get("active", False) and (time.time() - health_info.get("last_seen", 0)) < 300  # 5 min timeout
    
    def _get_healthy_clients(self) -> List:
        """Get list of healthy clients only"""
        return [client for client in self.clients if self._is_client_healthy(client.client_id)]
    
    def update_topology(self):
        """
        Update P2P topology based on client similarities
        Uses each client's PeerSelectionExpert
        """
        with self._lock:  # Thread safety
            self.logger.info("P2P: Updating topology based on client similarities")
            
            healthy_clients = self._get_healthy_clients()
            
            for client in healthy_clients:
                # Check cooldown with proper configuration access
                last_update = self.last_topology_update.get(client.client_id, 0)
                cooldown = STABILITY_CONFIG.get("peer_update_cooldown", 60)
                
                if time.time() - last_update < cooldown:
                    self.logger.debug(f"Client {client.client_id} topology update on cooldown")
                    continue  # Skip if updated recently
                
                # Select peers with self-exclusion and uniqueness
                try:
                    selected_peers = client.select_peers(healthy_clients)
                    # Ensure self-exclusion and uniqueness
                    selected_peers = list(set(selected_peers))
                    if client.client_id in selected_peers:
                        selected_peers.remove(client.client_id)
                    
                    self.topology[client.client_id] = selected_peers
                    self.last_topology_update[client.client_id] = time.time()
                    
                    self.logger.debug(
                        f"Client {client.client_id} -> Peers {selected_peers}"
                    )
                except Exception as e:
                    self.logger.error(f"Error updating topology for client {client.client_id}: {e}")
    
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
        
        with self._lock:  # Thread safety
            self.logger.info(f"P2P: Starting gossip round #{self.exchange_count + 1}")
            
            # Update topology first
            self.update_topology()
            
            # Track exchanges in this round with proper duplicate prevention
            exchanges_done = set()  # Use set for O(1) lookup
            
            # Each client exchanges with its peers
            healthy_clients = self._get_healthy_clients()
            for client in healthy_clients:
                client_id = client.client_id
                peers = self.topology.get(client_id, [])
                
                for peer_id in peers:
                    # Create exchange key for bidirectional prevention
                    exchange_key = tuple(sorted([client_id, peer_id]))
                    
                    # Avoid duplicate exchanges (A->B and B->A)
                    if exchange_key in exchanges_done or exchange_key in self.active_exchanges:
                        continue
                    
                    # Find peer client
                    peer_client = next((c for c in healthy_clients if c.client_id == peer_id), None)
                    
                    if peer_client is not None:
                        # Check if both clients are healthy
                        if not (self._is_client_healthy(client_id) and self._is_client_healthy(peer_id)):
                            continue
                        
                        # Mark exchange as active
                        self.active_exchanges.add(exchange_key)
                        
                        try:
                            # Perform bidirectional exchange
                            client.sync_with_peer(peer_id, peer_client)
                            
                            exchanges_done.add(exchange_key)
                            
                            self.exchange_history.append({
                                "round": self.exchange_count,
                                "client_a": client_id,
                                "client_b": peer_id,
                                "timestamp": time.time(),
                            })
                            
                            # Update client health
                            self.client_health[client_id]["last_seen"] = time.time()
                            self.client_health[peer_id]["last_seen"] = time.time()
                            
                        except Exception as e:
                            self.logger.error(f"Error in exchange between {client_id} and {peer_id}: {e}")
                        finally:
                            # Remove from active exchanges
                            self.active_exchanges.discard(exchange_key)
            
            self.exchange_count += 1
            self.last_exchange_time = time.time()
            
            self.logger.info(f"P2P: Completed {len(exchanges_done)} exchanges")
    
    def get_topology_stats(self) -> Dict:
        """
        Get statistics about P2P topology with correct edge counting
        """
        with self._lock:  # Thread safety
            # Compute topology metrics
            if not self.topology:
                return {
                    "num_clients": self.num_clients,
                    "avg_connections": 0,
                    "total_edges": 0,
                    "topology": {},
                }
            
            # Count unique bidirectional edges (corrected edge counting)
            unique_edges = set()
            num_connections = []
            
            for client_id, peers in self.topology.items():
                num_connections.append(len(peers))
                for peer_id in peers:
                    # Create bidirectional edge representation
                    edge = tuple(sorted([client_id, peer_id]))
                    unique_edges.add(edge)
            
            return {
                "num_clients": self.num_clients,
                "avg_connections": np.mean(num_connections) if num_connections else 0,
                "min_connections": min(num_connections) if num_connections else 0,
                "max_connections": max(num_connections) if num_connections else 0,
                "total_edges": len(unique_edges),  # Correct edge count
                "topology": self.topology,
                "healthy_clients": len(self._get_healthy_clients()),
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


# Configuration already imported at top

