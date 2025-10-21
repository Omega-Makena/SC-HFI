"""
SF-HFE Main Orchestrator
Simulates online continual learning with:
- Developer (zero data) operating server
- Users (with data) as clients
- Continuous data streaming
- Online expert training
- P2P gossip
- Meta-learning triggers
"""

import torch
import logging
import time
from typing import List, Dict
import numpy as np

from config import (
    STREAM_CONFIG, FL_CONFIG, P2P_CONFIG,
    META_LEARNING_CONFIG, MONITORING_CONFIG
)
from client import SFHFEClient
from server import SFHFEServer
from data_stream import MultiClientStreamGenerator
from p2p_gossip import P2PGossipManager


def setup_logging(level=logging.INFO):
    """Configure logging for the system"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('sf_hfe_online.log')
        ]
    )
    return logging.getLogger("SF-HFE")


class OnlineTrainingOrchestrator:
    """
    Orchestrates the complete online training system
    
    Simulates:
    1. Developer starts with ZERO data (only server)
    2. Users join with their local data streams
    3. Continuous online learning
    4. P2P gossip exchanges
    5. Meta-learning triggers
    """
    
    def __init__(
        self,
        num_clients: int = 5,
        input_dim: int = 20,
        output_dim: int = 1,
        stream_length: int = 10000,
        developer_participates: bool = False
    ):
        self.num_clients = num_clients
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.stream_length = stream_length
        self.developer_participates = developer_participates
        
        # Logger
        self.logger = setup_logging(level=logging.INFO)
        
        self.logger.info("=" * 80)
        self.logger.info("SF-HFE: Online Continual Learning System")
        self.logger.info("=" * 80)
        self.logger.info(f"Scenario: Developer (ZERO data) + {num_clients} Users (with data)")
        self.logger.info("=" * 80)
        
        # Initialize components
        self._initialize_system()
        
    def _initialize_system(self):
        """Initialize all system components"""
        
        # 1. SERVER (Developer with ZERO data)
        self.logger.info("\n[DEVELOPER] Initializing server with ZERO training data...")
        self.server = SFHFEServer(num_experts=10)
        self.logger.info("[DEVELOPER] Server ready - waiting for user insights")
        
        # 2. CLIENTS (Users with data)
        self.logger.info(f"\n[USERS] Initializing {self.num_clients} clients with local data...")
        self.clients = []
        
        for client_id in range(self.num_clients):
            client = SFHFEClient(
                client_id=client_id,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                has_data=True  # Users have data
            )
            self.clients.append(client)
        
        self.logger.info(f"[USERS] {self.num_clients} clients initialized")
        
        # 3. DATA STREAMS (Each user has their own stream)
        self.logger.info("\n[DATA] Generating online data streams...")
        self.stream_generator = MultiClientStreamGenerator(
            num_clients=self.num_clients,
            num_features=self.input_dim,
            stream_length=self.stream_length,
            heterogeneous=True  # Different distributions per client
        )
        self.logger.info("[DATA] Data streams ready")
        
        # 4. P2P GOSSIP
        self.logger.info("\n[P2P] Initializing gossip protocol...")
        self.p2p_manager = P2PGossipManager(self.clients)
        self.logger.info("[P2P] Gossip manager ready")
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("System initialization complete!")
        self.logger.info("=" * 80 + "\n")
        
    def run_online_training(
        self,
        num_batches: int = 300,
        log_frequency: int = 50
    ):
        """
        Run online continual learning
        
        Args:
            num_batches: Number of mini-batches to process
            log_frequency: Log progress every N batches
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Starting Online Training: {num_batches} mini-batches")
        self.logger.info("Mode: PURE ONLINE CONTINUAL LEARNING")
        self.logger.info("Developer has ZERO data - Learning from user insights only")
        self.logger.info("=" * 80 + "\n")
        
        batch_size = STREAM_CONFIG["mini_batch_size"]
        
        # Metrics tracking
        all_losses = []
        meta_learning_triggers = []
        p2p_exchanges = []
        
        start_time = time.time()
        
        for batch_idx in range(num_batches):
            batch_start = time.time()
            
            # Phase 1: Each client processes a mini-batch from their stream
            client_metrics = []
            
            for client in self.clients:
                # Get next batch from client's stream
                batch_x, batch_y = self.stream_generator.get_batch(
                    client.client_id,
                    batch_size=batch_size
                )
                
                # Client processes batch (online learning)
                metrics = client.process_stream_batch(batch_x, batch_y)
                client_metrics.append(metrics)
            
            # Phase 2: P2P Gossip (asynchronous - happens periodically)
            if self.p2p_manager.should_exchange():
                self.p2p_manager.perform_gossip_round()
                p2p_exchanges.append(batch_idx)
            
            # Phase 3: Clients send insights to server (every N batches)
            if batch_idx % FL_CONFIG.get("insight_frequency", 50) == 0 and batch_idx > 0:
                # Generate insights
                insights = [client.generate_insights() for client in self.clients]
                
                # Server receives and processes
                response = self.server.receive_insights(insights)
                
                # If meta-learning triggered, broadcast new parameters
                if response.get("meta_learning_triggered", False):
                    self.server.broadcast_meta_parameters(self.clients)
                    meta_learning_triggers.append(batch_idx)
            
            # Logging
            if batch_idx % log_frequency == 0 or batch_idx == num_batches - 1:
                avg_loss = np.mean([m.get("avg_loss", 0.0) for m in client_metrics if m])
                all_losses.append(avg_loss)
                
                self.logger.info(
                    f"Batch {batch_idx}/{num_batches} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Time: {(time.time() - batch_start)*1000:.1f}ms | "
                    f"Samples: {batch_idx * batch_size * self.num_clients}"
                )
                
                # Show router entropy
                avg_entropy = np.mean([
                    client.router.compute_entropy()
                    for client in self.clients
                ])
                self.logger.info(f"  Router Entropy: {avg_entropy:.3f}")
                
                # Show most active experts
                most_active_per_client = []
                for client in self.clients:
                    most_active = client.router.get_most_active_experts(k=3)
                    most_active_per_client.append(most_active)
                
                self.logger.info(f"  Most Active Experts: {most_active_per_client[0]} (Client 0)")
        
        # Training complete
        elapsed = time.time() - start_time
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Online Training Complete!")
        self.logger.info("=" * 80)
        self.logger.info(f"Total Time: {elapsed:.2f}s")
        self.logger.info(f"Batches Processed: {num_batches}")
        self.logger.info(f"Samples Processed: {num_batches * batch_size * self.num_clients}")
        self.logger.info(f"Meta-Learning Triggers: {len(meta_learning_triggers)}")
        self.logger.info(f"P2P Exchanges: {len(p2p_exchanges)}")
        
        # Final statistics
        self._print_final_statistics()
    
    def _print_final_statistics(self):
        """Print comprehensive final statistics"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("FINAL STATISTICS")
        self.logger.info("=" * 80)
        
        # Server stats
        server_stats = self.server.get_stats()
        self.logger.info(f"\n[SERVER]")
        self.logger.info(f"  Total Insights Received: {server_stats['total_insights']}")
        self.logger.info(f"  Unique Clients: {server_stats['unique_clients']}")
        self.logger.info(f"  Meta-Learning Updates: {server_stats['meta_engine_stats']['meta_updates']}")
        
        # Client stats
        self.logger.info(f"\n[CLIENTS]")
        for client in self.clients:
            stats = client.get_stats()
            self.logger.info(
                f"  Client {client.client_id}: "
                f"Samples={stats['total_samples']}, "
                f"Loss={stats['avg_loss']:.4f}, "
                f"Drifts={stats['drift_events']}, "
                f"Peers={len(stats['connected_peers'])}"
            )
        
        # Expert utilization across all clients
        self.logger.info(f"\n[EXPERT UTILIZATION]")
        for expert_id in range(10):
            total_activations = sum(
                client.experts[expert_id].activation_count
                for client in self.clients
            )
            expert_name = client.experts[expert_id].expert_name
            self.logger.info(f"  Expert {expert_id} ({expert_name}): {total_activations} activations")
        
        # P2P stats
        p2p_stats = self.p2p_manager.get_stats()
        self.logger.info(f"\n[P2P GOSSIP]")
        self.logger.info(f"  Total Exchanges: {p2p_stats['total_exchanges']}")
        self.logger.info(f"  Topology Edges: {p2p_stats['topology_stats']['total_edges']}")
        self.logger.info(f"  Avg Connections: {p2p_stats['topology_stats']['avg_connections']:.1f}")
        
        self.logger.info("\n" + "=" * 80)


def main():
    """
    Main entry point for SF-HFE online training
    """
    # Configuration
    NUM_CLIENTS = 5
    INPUT_DIM = 20
    OUTPUT_DIM = 1
    NUM_BATCHES = 300  # 300 batches * 32 samples = 9600 samples per client
    
    # Create orchestrator
    orchestrator = OnlineTrainingOrchestrator(
        num_clients=NUM_CLIENTS,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        stream_length=10000,
        developer_participates=False  # Developer has NO data
    )
    
    # Run online training
    orchestrator.run_online_training(
        num_batches=NUM_BATCHES,
        log_frequency=50
    )


if __name__ == "__main__":
    main()

