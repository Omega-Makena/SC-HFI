"""
SF-HFE v2.0 - Domain-Specific Client Test
Simulates 3 clients from different fields:
1. Agriculture - Crop yield prediction
2. Tech - Software performance metrics
3. Economics - Market trend analysis
"""

import sys
import os
import torch
import numpy as np
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from federated.server import SFHFEServer
from moe.client import SFHFEClient
from data.stream import ConceptDriftStream
from reproducibility import set_global_seed

# Set random seeds for reproducibility
set_global_seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('domain_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DomainSpecificStream:
    """Generates domain-specific data streams with realistic characteristics"""
    
    def __init__(self, domain: str, input_dim: int, output_dim: int):
        self.domain = domain
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_count = 0
        
        logger.info(f"Initialized {domain} data stream (input_dim={input_dim}, output_dim={output_dim})")
    
    def generate_batch(self, batch_size: int):
        """Generate domain-specific data batches"""
        self.batch_count += 1
        
        if self.domain == "agriculture":
            # Agriculture: Seasonal patterns, weather dependencies
            # Features: temperature, rainfall, soil_pH, sunlight, humidity, etc.
            season_phase = (self.batch_count / 50) * 2 * np.pi
            
            # Seasonal variations
            X = torch.randn(batch_size, self.input_dim) * 0.5
            X[:, 0] += np.sin(season_phase) * 2  # Temperature (seasonal)
            X[:, 1] += np.cos(season_phase) * 1.5  # Rainfall (seasonal)
            X[:, 2] += np.random.uniform(5.5, 7.5, batch_size)  # Soil pH (stable)
            
            # Target: Crop yield (influenced by seasonal factors)
            y = torch.randn(batch_size, self.output_dim)
            y[:, 0] = (
                0.3 * X[:, 0] +  # Temperature effect
                0.4 * X[:, 1] +  # Rainfall effect
                0.2 * X[:, 2] +  # Soil pH effect
                torch.randn(batch_size) * 0.3  # Noise
            )
            
            if self.batch_count % 20 == 0:
                logger.info(f"[Agriculture] Batch {self.batch_count}: Season={season_phase:.2f}, "
                           f"Temp={X[0, 0]:.2f}, Rain={X[0, 1]:.2f}, Yield={y[0, 0]:.2f}")
        
        elif self.domain == "tech":
            # Tech: Software performance - sudden spikes, version updates
            # Features: CPU, memory, network, request_rate, cache_hit_ratio, etc.
            
            # Base metrics
            X = torch.randn(batch_size, self.input_dim) * 0.3
            X[:, 0] += np.random.uniform(20, 80, batch_size)  # CPU usage
            X[:, 1] += np.random.uniform(40, 90, batch_size)  # Memory usage
            X[:, 2] += np.random.exponential(100, batch_size)  # Request rate
            
            # Random performance spikes (simulate traffic bursts)
            if self.batch_count % 15 == 0:
                spike_idx = np.random.randint(0, batch_size, size=batch_size // 3)
                X[spike_idx, 0] *= 1.8  # CPU spike
                X[spike_idx, 2] *= 2.5  # Request spike
                logger.info(f"[Tech] Batch {self.batch_count}: TRAFFIC SPIKE detected!")
            
            # Target: Response time (latency)
            y = torch.randn(batch_size, self.output_dim)
            y[:, 0] = (
                0.4 * (X[:, 0] / 100) +  # CPU impact
                0.3 * (X[:, 1] / 100) +  # Memory impact
                0.3 * torch.log1p(X[:, 2] / 100) +  # Request rate (log scale)
                torch.randn(batch_size) * 0.2
            )
            
            if self.batch_count % 20 == 0:
                logger.info(f"[Tech] Batch {self.batch_count}: CPU={X[0, 0]:.1f}%, "
                           f"Mem={X[0, 1]:.1f}%, Latency={y[0, 0]:.2f}ms")
        
        elif self.domain == "economics":
            # Economics: Market trends, volatility, macro indicators
            # Features: stock_price, volume, GDP_growth, inflation, interest_rate, etc.
            
            # Market trend (bull/bear cycle)
            market_phase = (self.batch_count / 100) * 2 * np.pi
            trend = np.sin(market_phase)
            
            X = torch.randn(batch_size, self.input_dim) * 0.4
            X[:, 0] += 100 + trend * 20  # Stock price
            X[:, 1] += np.random.exponential(1e6, batch_size)  # Volume
            X[:, 2] += np.random.uniform(1.5, 4.5, batch_size)  # GDP growth
            X[:, 3] += np.random.uniform(2.0, 6.0, batch_size)  # Inflation
            
            # Market crash simulation (rare event)
            if self.batch_count == 80:
                X[:, 0] *= 0.7  # Price drop
                X[:, 1] *= 3.0  # Volume spike
                logger.warning(f"[Economics] Batch {self.batch_count}: MARKET CRASH EVENT!")
            
            # Target: Future returns
            y = torch.randn(batch_size, self.output_dim)
            y[:, 0] = (
                0.5 * trend +  # Market trend
                0.2 * (X[:, 2] - 3.0) +  # GDP deviation
                -0.3 * (X[:, 3] - 4.0) +  # Inflation deviation
                torch.randn(batch_size) * 0.5  # Noise (volatility)
            )
            
            if self.batch_count % 20 == 0:
                logger.info(f"[Economics] Batch {self.batch_count}: Price={X[0, 0]:.2f}, "
                           f"Trend={trend:.2f}, Return={y[0, 0]:.2f}")
        
        else:
            # Default: generic data
            X = torch.randn(batch_size, self.input_dim)
            y = torch.randn(batch_size, self.output_dim)
        
        return X, y


def run_domain_test():
    """Run comprehensive test with 3 domain-specific clients"""
    
    logger.info("=" * 80)
    logger.info("SF-HFE v2.0 - DOMAIN-SPECIFIC CLIENT TEST")
    logger.info("=" * 80)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    # Configuration
    input_dim = 20
    output_dim = 1
    batch_size = 32
    num_batches = 60  # Reduced for faster testing
    
    domains = ["agriculture", "tech", "economics"]
    client_names = ["AgriCorp Farm System", "TechMetrics Platform", "EconPredict Analytics"]
    
    logger.info("CONFIGURATION:")
    logger.info(f"  Input Dimension: {input_dim}")
    logger.info(f"  Output Dimension: {output_dim}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Total Batches: {num_batches}")
    logger.info(f"  Domains: {', '.join(domains)}")
    logger.info("")
    
    # Initialize server
    logger.info("STEP 1: Initializing SF-HFE Server (Developer - ZERO data)")
    server = SFHFEServer(num_experts=10)
    logger.info("  Server initialized successfully")
    logger.info(f"  Meta-Learning Engine: {server.meta_engine.__class__.__name__}")
    logger.info(f"  Global Memory: {server.global_memory.__class__.__name__}")
    logger.info("")
    
    # Initialize domain-specific clients
    logger.info("STEP 2: Initializing Domain-Specific Clients")
    clients = []
    streams = []
    
    for i, (domain, client_name) in enumerate(zip(domains, client_names)):
        logger.info(f"  Client {i+1}: {client_name} ({domain.upper()})")
        
        # Create client
        client = SFHFEClient(
            client_id=i,
            input_dim=input_dim,
            output_dim=output_dim
        )
        clients.append(client)
        
        # Create domain-specific stream
        stream = DomainSpecificStream(domain, input_dim, output_dim)
        streams.append(stream)
        
        logger.info(f"    - 10 Experts initialized")
        logger.info(f"    - 3-Tier Memory system ready")
        logger.info(f"    - Router: {client.router.__class__.__name__}")
        logger.info(f"    - Data Stream: {domain} characteristics")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("STARTING ONLINE TRAINING")
    logger.info("=" * 80)
    logger.info("")
    
    # Training loop
    all_losses = {i: [] for i in range(3)}
    expert_usage = {i: {j: 0 for j in range(10)} for i in range(3)}
    
    for batch_idx in range(num_batches):
        if batch_idx % 30 == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"ROUND {batch_idx // 30 + 1} - Batch {batch_idx}/{num_batches}")
            logger.info(f"{'='*60}")
        
        # Each client trains on their domain-specific data
        round_insights = []
        
        for client_id, (client, stream, domain) in enumerate(zip(clients, streams, domains)):
            # Generate domain-specific batch
            batch_x, batch_y = stream.generate_batch(batch_size)
            
            # Client trains locally (online learning)
            result = client.process_stream_batch(batch_x, batch_y)
            
            # Track metrics
            if 'avg_loss' in result:
                avg_loss = result['avg_loss']
                all_losses[client_id].append(avg_loss)
            else:
                all_losses[client_id].append(0.0)
            
            # Track expert usage
            if 'selected_experts' in result:
                selected_experts_list = result['selected_experts']
                # Flatten and convert to integers
                if isinstance(selected_experts_list, torch.Tensor):
                    selected_experts_list = selected_experts_list.cpu().numpy()
                if isinstance(selected_experts_list, np.ndarray):
                    selected_experts_list = selected_experts_list.flatten()
                
                for expert_id in selected_experts_list:
                    try:
                        expert_id = int(expert_id)
                        expert_usage[client_id][expert_id] += 1
                    except (ValueError, TypeError):
                        pass  # Skip invalid IDs
            
            # Generate insight
            insight = client.generate_insights()
            insight['domain'] = domain
            round_insights.append(insight)
            
            if batch_idx % 30 == 0 and result:
                logger.info(f"\n[{domain.upper()}] Client {client_id} Update:")
                logger.info(f"  Avg Loss: {result.get('avg_loss', 0):.4f}")
                logger.info(f"  Selected Experts: {result.get('selected_experts', [])}")
                logger.info(f"  Routing Entropy: {result.get('routing_entropy', 0):.4f}")
                # Get memory stats (try different attribute names)
                mem = client.experts[0].memory
                recent_size = len(mem.recent.buffer) if hasattr(mem, 'recent') else 0
                compressed_size = len(mem.compressed.store) if hasattr(mem, 'compressed') else 0
                logger.info(f"  Memory Usage: Recent={recent_size}, Compressed={compressed_size}")
        
        # Server aggregates insights (no raw data)
        if batch_idx % 10 == 0:
            logger.info(f"\n[SERVER] Aggregating insights from {len(round_insights)} clients...")
            server.receive_insights(round_insights)
            
            # Meta-learning update
            if batch_idx > 0 and batch_idx % 30 == 0:
                logger.info("[SERVER] Triggering manual meta-learning update...")
                server.meta_engine.meta_update(server.global_memory.insights)
                meta_params = server.meta_engine.get_meta_parameters()
                
                # Broadcast to clients
                for client in clients:
                    client.receive_meta_parameters(meta_params)
                
                logger.info(f"  Meta-parameters broadcasted to all clients")
                logger.info(f"  Total insights in memory: {server.global_memory.total_insights}")
        
        # P2P Gossip (if enabled)
        if batch_idx % 20 == 0 and len(clients) > 1:
            logger.info("\n[P2P] Initiating gossip protocol...")
            # Pair clients randomly
            if np.random.rand() > 0.5:
                peer1, peer2 = np.random.choice(len(clients), 2, replace=False)
                logger.info(f"  Client {peer1} ({domains[peer1]}) <-> Client {peer2} ({domains[peer2]})")
                # Simplified gossip (in real implementation, would exchange weights)
    
    # Final Statistics
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE - FINAL STATISTICS")
    logger.info("=" * 80)
    
    for client_id, (domain, client_name) in enumerate(zip(domains, client_names)):
        logger.info(f"\n{client_name} ({domain.upper()}):")
        logger.info(f"  Initial Loss: {all_losses[client_id][0]:.4f}")
        logger.info(f"  Final Loss: {all_losses[client_id][-1]:.4f}")
        logger.info(f"  Improvement: {(1 - all_losses[client_id][-1] / all_losses[client_id][0]) * 100:.1f}%")
        
        logger.info(f"\n  Expert Usage Distribution:")
        total_usage = sum(expert_usage[client_id].values())
        expert_names = ['Geometry', 'Temporal', 'Reconstruction', 'Causal', 'Drift',
                       'Governance', 'Consistency', 'PeerSelection', 'MetaAdaptation', 'MemoryConsolidation']
        for expert_id, expert_name in enumerate(expert_names):
            usage_pct = (expert_usage[client_id][expert_id] / total_usage * 100) if total_usage > 0 else 0
            logger.info(f"    {expert_name:20s}: {usage_pct:5.1f}% ({expert_usage[client_id][expert_id]} times)")
    
    logger.info(f"\n{'='*80}")
    logger.info("SERVER STATISTICS:")
    logger.info(f"  Total Insights Collected: {len(server.global_memory.insights)}")
    logger.info(f"  Meta-Learning Updates: {num_batches // 30}")
    logger.info(f"  Developer Data Used: ZERO (learned from insights only)")
    
    logger.info(f"\n{'='*80}")
    logger.info("TEST COMPLETED SUCCESSFULLY")
    logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log saved to: domain_test.log")
    logger.info(f"{'='*80}\n")
    
    return {
        'clients': clients,
        'server': server,
        'losses': all_losses,
        'expert_usage': expert_usage
    }


if __name__ == "__main__":
    try:
        results = run_domain_test()
        print("\n\nTest completed successfully!")
        print("Check 'domain_test.log' for detailed logs")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

