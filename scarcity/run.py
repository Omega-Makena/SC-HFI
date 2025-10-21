"""
Scarcity Framework - Entry Point

This script serves as the main entry point for the Scarcity Framework (SF-HFE).
It initializes logging and orchestrates the system components.
"""

import logging
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scarcity.core import Expert, StructureExpert, DriftExpert, Router, Client, Server, MetaLearner
import torch


def setup_logging(level=logging.INFO):
    """
    Configure logging for the entire framework.
    
    Args:
        level: Logging level (default: INFO)
    """
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # File handler
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / 'scarcity.log')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return root_logger


def run_stage1_simulation(num_clients=3):
    """
    Stage 1: Simple simulation of local training and insight generation.
    
    Each client has fake numeric data, trains locally using experts,
    and generates insights. Server collects and aggregates insights.
    
    Args:
        num_clients: Number of clients (default 3 for simplicity)
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("STAGE 1: Local Training and Insight Generation")
    logger.info("=" * 80)
    
    # Create clients with simple fake data
    logger.info(f"\nCreating {num_clients} clients with fake NumPy data...")
    clients = []
    
    for client_id in range(num_clients):
        # Generate fake data (NumPy random arrays)
        np.random.seed(client_id * 10)
        fake_data = np.random.randn(50, 10)  # 50 samples, 10 features
        fake_labels = np.random.randn(50, 1)
        
        logger.info(f"  Client {client_id}: Generated {fake_data.shape[0]} samples")
        clients.append({
            "id": client_id,
            "data": torch.tensor(fake_data, dtype=torch.float32),
            "labels": torch.tensor(fake_labels, dtype=torch.float32),
            "experts": [
                StructureExpert(expert_id=0, input_dim=10, output_dim=1),
                DriftExpert(expert_id=1, input_dim=10, output_dim=1)
            ]
        })
    
    # Server collects insights
    server_insights = []
    
    logger.info("\n" + "=" * 80)
    logger.info("Local Training Phase")
    logger.info("=" * 80 + "\n")
    
    # Each client trains locally
    for client in clients:
        logger.info(f"Client {client['id']}: Starting local training...")
        
        client_insights = {}
        
        # Train each expert
        for expert in client['experts']:
            expert_name = type(expert).__name__
            logger.info(f"  Training {expert_name}...")
            
            # Train expert on fake data
            metrics = expert.train(
                X_train=client['data'],
                y_train=client['labels'],
                epochs=3,
                lr=0.01
            )
            
            # Get summary statistics
            summary = expert.summarize(client['data'])
            
            client_insights[expert_name] = {
                "training_metrics": metrics,
                "summary": summary
            }
            
            logger.info(f"    {expert_name} trained: Loss {metrics['initial_loss']:.4f} -> {metrics['final_loss']:.4f}")
            
            if expert_name == "StructureExpert":
                logger.info(f"    Statistics: mean={summary['mean']:.4f}, std={summary['std']:.4f}")
            elif expert_name == "DriftExpert":
                logger.info(f"    Drift: current_mean={summary['current_mean']:.4f}, drift={summary['drift']:.4f}")
        
        # Generate insight
        insight = {
            "client_id": client['id'],
            "insights": client_insights,
            "num_experts": len(client['experts']),
            "data_shape": list(client['data'].shape)
        }
        
        logger.info(f"  Client {client['id']}: Generated insight with {len(client_insights)} expert summaries\n")
        
        # Send insight to server
        server_insights.append(insight)
    
    logger.info("=" * 80)
    logger.info("Server Aggregation Phase")
    logger.info("=" * 80 + "\n")
    
    # Server aggregates insights
    logger.info(f"Server: Received insights from {len(server_insights)} clients")
    
    for insight in server_insights:
        logger.info(f"  Client {insight['client_id']}: {insight['num_experts']} experts, data shape {insight['data_shape']}")
    
    # Aggregate statistics across all clients
    all_structure_means = []
    all_structure_stds = []
    all_drift_values = []
    
    for insight in server_insights:
        if "StructureExpert" in insight['insights']:
            struct_summary = insight['insights']['StructureExpert']['summary']
            all_structure_means.append(struct_summary['mean'])
            all_structure_stds.append(struct_summary['std'])
        
        if "DriftExpert" in insight['insights']:
            drift_summary = insight['insights']['DriftExpert']['summary']
            all_drift_values.append(drift_summary['drift'])
    
    logger.info("\nAggregated Statistics:")
    logger.info(f"  Average data mean across clients: {np.mean(all_structure_means):.4f}")
    logger.info(f"  Average data std across clients: {np.mean(all_structure_stds):.4f}")
    logger.info(f"  Average drift across clients: {np.mean(all_drift_values):.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Stage 1 Complete!")
    logger.info("=" * 80)
    
    return server_insights


def run_federated_learning(num_clients=5, num_rounds=5, input_dim=10, output_dim=1):
    """
    Run the federated learning training loop.
    
    Args:
        num_clients: Number of federated learning clients
        num_rounds: Number of global training rounds
        input_dim: Input dimension for models
        output_dim: Output dimension for models
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("STAGE 2: Federated Learning Training Loop")
    logger.info("=" * 80)
    
    # Create clients with local data (no centralized data!)
    logger.info(f"Creating {num_clients} clients with local data...")
    clients = [
        Client(client_id=i, input_dim=input_dim, output_dim=output_dim, data_size=100) 
        for i in range(num_clients)
    ]
    
    # Create server
    logger.info("Creating federated learning server...")
    server = Server(clients=clients, input_dim=input_dim, output_dim=output_dim)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Starting Federated Learning: {num_rounds} global rounds")
    logger.info("=" * 80 + "\n")
    
    # Run federated learning rounds
    for round_num in range(1, num_rounds + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"GLOBAL ROUND {round_num}/{num_rounds}")
        logger.info(f"{'='*80}")
        
        metrics = server.run_round(round_num)
        
        logger.info(f"Round {round_num} Summary: Global Loss = {metrics['avg_loss']:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Federated Learning Complete!")
    logger.info("=" * 80)
    
    # Final evaluation
    logger.info("\nFinal Global Model Evaluation:")
    final_metrics = server.evaluate_global_model()
    logger.info(f"Final Global Loss: {final_metrics['avg_loss']:.4f}")
    
    return server, clients


def run_insight_exchange(num_clients=5, num_rounds=5, input_dim=10, output_dim=1):
    """
    Run the Scarcity-style Insight Exchange loop.
    
    Instead of sharing raw model weights, clients generate and share
    structured insights about their local learning dynamics.
    
    Args:
        num_clients: Number of federated learning clients
        num_rounds: Number of global training rounds
        input_dim: Input dimension for models
        output_dim: Output dimension for models
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("STAGE 3: Scarcity-Style Insight Exchange")
    logger.info("=" * 80)
    
    # Create meta-learner
    logger.info("Creating MetaLearner for insight aggregation...")
    meta_learner = MetaLearner()
    
    # Create clients with local data
    logger.info(f"Creating {num_clients} clients with local data...")
    clients = [
        Client(client_id=i, input_dim=input_dim, output_dim=output_dim, data_size=100) 
        for i in range(num_clients)
    ]
    
    # Create server with meta-learner
    logger.info("Creating server with MetaLearner integration...")
    server = Server(clients=clients, meta_learner=meta_learner, 
                   input_dim=input_dim, output_dim=output_dim)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Starting Insight Exchange: {num_rounds} global rounds")
    logger.info("Key Innovation: Sharing insights, not raw weights!")
    logger.info("=" * 80 + "\n")
    
    # Run insight exchange rounds
    for round_num in range(1, num_rounds + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"INSIGHT EXCHANGE ROUND {round_num}/{num_rounds}")
        logger.info(f"{'='*80}")
        
        metrics = server.run_insight_round(round_num)
        knowledge = metrics["aggregated_knowledge"]
        
        logger.info(
            f"\nRound {round_num} Summary:\n"
            f"  Global Loss: {metrics['avg_loss']:.4f}\n"
            f"  Avg Uncertainty: {knowledge['avg_uncertainty']:.4f} (±{knowledge['std_uncertainty']:.4f})\n"
            f"  Avg Gradient: {knowledge['avg_mean_grad']:.4f}\n"
            f"  Insights Collected: {metrics['num_insights']}\n"
            f"  Total Memory Size: {len(server.memory)} insights\n"
            f"  High Uncertainty Clients: {knowledge['high_uncertainty_clients']}\n"
            f"  Low Uncertainty Clients: {knowledge['low_uncertainty_clients']}"
        )
    
    logger.info("\n" + "=" * 80)
    logger.info("Insight Exchange Complete!")
    logger.info("=" * 80)
    
    # Summary statistics
    logger.info("\n" + "Final System State:")
    logger.info(f"  Total Insights Stored: {len(server.memory)}")
    logger.info(f"  MetaLearner History Size: {len(meta_learner.insight_history)}")
    
    # Analyze insight patterns
    if server.memory:
        all_uncertainties = [i["uncertainty"] for i in server.memory]
        all_improvements = [i["loss_improvement"] for i in server.memory]
        logger.info(f"  Overall Avg Uncertainty: {np.mean(all_uncertainties):.4f}")
        logger.info(f"  Overall Avg Loss Improvement: {np.mean(all_improvements):.4f}")
    
    return server, clients, meta_learner


def run_expert_routing(num_clients=5, num_rounds=5, input_dim=10, output_dim=1,
                      router_strategy="variance"):
    """
    Run the Expert Routing system with specialized experts and adaptive routing.
    
    Each client has multiple experts (StructureExpert, DriftExpert), and a Router
    selects the most appropriate expert based on data characteristics.
    
    Args:
        num_clients: Number of federated learning clients
        num_rounds: Number of global training rounds
        input_dim: Input dimension for models
        output_dim: Output dimension for models
        router_strategy: Routing strategy ("variance", "random", "round_robin")
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("STAGE 4: Expert Routing Architecture")
    logger.info("=" * 80)
    
    # Create meta-learner
    logger.info("Creating MetaLearner for insight aggregation...")
    meta_learner = MetaLearner()
    
    # Create clients with expert routing enabled
    logger.info(f"Creating {num_clients} clients with Expert Routing (strategy={router_strategy})...")
    clients = [
        Client(client_id=i, input_dim=input_dim, output_dim=output_dim, 
               data_size=100, use_experts=True, router_strategy=router_strategy) 
        for i in range(num_clients)
    ]
    
    # Create server with meta-learner
    logger.info("Creating server with MetaLearner integration...")
    server = Server(clients=clients, meta_learner=meta_learner, 
                   input_dim=input_dim, output_dim=output_dim)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Starting Expert Routing: {num_rounds} global rounds")
    logger.info(f"Each client has: StructureExpert + DriftExpert")
    logger.info(f"Router Strategy: {router_strategy}")
    logger.info("=" * 80 + "\n")
    
    # Track expert usage statistics
    expert_usage = {
        "StructureExpert": 0,
        "DriftExpert": 0
    }
    
    # Run expert routing rounds
    for round_num in range(1, num_rounds + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"EXPERT ROUTING ROUND {round_num}/{num_rounds}")
        logger.info(f"{'='*80}")
        
        metrics = server.run_insight_round(round_num)
        knowledge = metrics["aggregated_knowledge"]
        
        # Display expert selection for each client
        logger.info("\n" + "-" * 80)
        logger.info("Expert Selection Summary:")
        logger.info("-" * 80)
        
        for client in clients:
            if hasattr(client, 'selected_expert') and client.selected_expert:
                expert_name = type(client.selected_expert).__name__
                expert_usage[expert_name] += 1
                
                # Get the expert summaries from the insight
                insights = [i for i in server.memory if i.get("client_id") == client.client_id]
                if insights:
                    last_insight = insights[-1]
                    if "expert_summaries" in last_insight:
                        summaries = last_insight["expert_summaries"]
                        logger.info(
                            f"  Client {client.client_id}: {expert_name} | "
                            f"Loss: {last_insight['final_loss']:.4f}"
                        )
                        for summary in summaries:
                            if summary["expert_type"] == "StructureExpert":
                                logger.info(
                                    f"     -> Structure: mean={summary['mean']:.4f}, "
                                    f"std={summary['std']:.4f}, var={summary['variance']:.4f}"
                                )
                            elif summary["expert_type"] == "DriftExpert":
                                logger.info(
                                    f"     -> Drift: current={summary['current_mean']:.4f}, "
                                    f"drift={summary['drift']:.4f}"
                                )
        
        logger.info("-" * 80)
        
        logger.info(
            f"\nRound {round_num} Summary:\n"
            f"  StructureExpert used: {sum(1 for c in clients if hasattr(c, 'selected_expert') and type(c.selected_expert).__name__ == 'StructureExpert')} times\n"
            f"  DriftExpert used: {sum(1 for c in clients if hasattr(c, 'selected_expert') and type(c.selected_expert).__name__ == 'DriftExpert')} times\n"
            f"  Total Insights: {len(server.memory)}"
        )
    
    logger.info("\n" + "=" * 80)
    logger.info("Expert Routing Complete!")
    logger.info("=" * 80)
    
    # Final statistics
    logger.info("\n" + "Final Expert Usage Statistics:")
    total_selections = sum(expert_usage.values())
    for expert_name, count in expert_usage.items():
        percentage = (count / total_selections * 100) if total_selections > 0 else 0
        logger.info(f"  {expert_name}: {count} times ({percentage:.1f}%)")
    
    logger.info(f"\n  Total Insights Stored: {len(server.memory)}")
    logger.info(f"  MetaLearner History Size: {len(meta_learner.insight_history)}")
    
    return server, clients, meta_learner


def run_meta_learning(num_clients=5, num_rounds=5, input_dim=10, output_dim=1):
    """
    Stage 5: Reptile-style meta-learning with adaptive global parameters.
    
    The MetaLearner tracks statistics across rounds and provides better
    initialization parameters to clients, inspired by Reptile meta-learning.
    
    Args:
        num_clients: Number of federated learning clients
        num_rounds: Number of global training rounds
        input_dim: Input dimension for models
        output_dim: Output dimension for models
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("STAGE 5: Reptile-Style Meta-Learning")
    logger.info("=" * 80)
    
    # Create meta-learner
    logger.info("Creating MetaLearner with Reptile-style approach...")
    meta_learner = MetaLearner()
    
    # Create clients with expert routing enabled
    logger.info(f"Creating {num_clients} clients with Expert Routing...")
    clients = [
        Client(client_id=i, input_dim=input_dim, output_dim=output_dim, 
               data_size=100, use_experts=True, router_strategy="variance") 
        for i in range(num_clients)
    ]
    
    # Create server with meta-learner
    logger.info("Creating server with MetaLearner integration...")
    server = Server(clients=clients, meta_learner=meta_learner, 
                   input_dim=input_dim, output_dim=output_dim)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Starting Meta-Learning: {num_rounds} global rounds")
    logger.info("Meta-Learner will adapt global parameters across rounds")
    logger.info("=" * 80 + "\n")
    
    # Track meta-parameter evolution
    meta_param_history = []
    
    # Run meta-learning rounds
    for round_num in range(1, num_rounds + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"META-LEARNING ROUND {round_num}/{num_rounds}")
        logger.info(f"{'='*80}")
        
        # Display current meta-parameters
        current_meta_params = meta_learner.broadcast_params()
        meta_param_history.append(current_meta_params.copy())
        
        logger.info(
            f"\nCurrent Meta-Parameters (Round {round_num}):\n"
            f"  Meta-Mean: {current_meta_params['meta_mean']:.4f}\n"
            f"  Meta-Std: {current_meta_params['meta_std']:.4f}\n"
            f"  Meta-LR: {current_meta_params['meta_lr']:.4f}\n"
            f"  Updates: {current_meta_params['meta_updates']}"
        )
        
        # Run insight exchange round (with meta-learning)
        metrics = server.run_insight_round(round_num)
        knowledge = metrics["aggregated_knowledge"]
        
        logger.info(
            f"\nRound {round_num} Summary:\n"
            f"  Global Loss: {metrics['avg_loss']:.4f}\n"
            f"  Insights Collected: {metrics['num_insights']}\n"
            f"  Meta-Parameters Updated: Yes"
        )
    
    logger.info("\n" + "=" * 80)
    logger.info("Meta-Learning Complete!")
    logger.info("=" * 80)
    
    # Display meta-parameter evolution
    logger.info("\nMeta-Parameter Evolution Across Rounds:")
    for i, params in enumerate(meta_param_history, 1):
        logger.info(
            f"  Round {i}: mean={params['meta_mean']:.4f}, "
            f"std={params['meta_std']:.4f}, "
            f"lr={params['meta_lr']:.4f}"
        )
    
    # Final meta-learner summary
    logger.info("\nFinal Meta-Learner Summary:")
    summary = meta_learner.summarize()
    logger.info(f"  Total Insights Processed: {summary['total_insights']}")
    logger.info(f"  Meta-Updates: {summary['global_params']['num_updates']}")
    logger.info(f"  Final Meta-Mean: {summary['global_params']['mean']:.4f}")
    logger.info(f"  Final Meta-Std: {summary['global_params']['std']:.4f}")
    logger.info(f"  Final Meta-LR: {summary['global_params']['learning_rate']:.4f}")
    
    if 'avg_historical_loss' in summary:
        logger.info(f"  Avg Historical Loss: {summary['avg_historical_loss']:.4f}")
    
    return server, clients, meta_learner


def run_p2p_gossip(num_clients=5, num_rounds=5, input_dim=10, output_dim=1):
    """
    Stage 6: P2P Gossip mechanism with peer-to-peer expert weight exchange.
    
    After local training, clients randomly pair up and exchange expert weights
    directly (decentralized) without going through the server.
    
    Args:
        num_clients: Number of federated learning clients
        num_rounds: Number of global training rounds
        input_dim: Input dimension for models
        output_dim: Output dimension for models
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("STAGE 6: P2P Gossip Mechanism")
    logger.info("=" * 80)
    
    # Create meta-learner
    logger.info("Creating MetaLearner with meta-learning...")
    meta_learner = MetaLearner()
    
    # Create clients with expert routing enabled
    logger.info(f"Creating {num_clients} clients with Expert Routing...")
    clients = [
        Client(client_id=i, input_dim=input_dim, output_dim=output_dim, 
               data_size=100, use_experts=True, router_strategy="variance") 
        for i in range(num_clients)
    ]
    
    # Create server with meta-learner
    logger.info("Creating server with MetaLearner integration...")
    server = Server(clients=clients, meta_learner=meta_learner, 
                   input_dim=input_dim, output_dim=output_dim)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Starting P2P Gossip Training: {num_rounds} global rounds")
    logger.info("After training, clients pair up for P2P expert weight exchange")
    logger.info("=" * 80 + "\n")
    
    import random
    
    # Run P2P gossip rounds
    for round_num in range(1, num_rounds + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"P2P GOSSIP ROUND {round_num}/{num_rounds}")
        logger.info(f"{'='*80}")
        
        # Phase 1: Select clients and broadcast
        selected_clients = server.select_clients(round_num)
        server.broadcast_model(selected_clients)
        
        # Phase 2: Local training with expert routing
        logger.info("\n--- Local Training Phase ---")
        insights = []
        for client in selected_clients:
            insight = client.generate_insight(epochs=5)
            insights.append(insight)
            
            if 'selected_expert' in insight:
                logger.info(
                    f"  Client {client.client_id}: Trained with {insight['selected_expert']} - "
                    f"Loss: {insight['final_loss']:.4f}"
                )
        
        # Phase 3: P2P Gossip - Randomly pair clients for expert exchange
        logger.info("\n--- P2P Gossip Phase ---")
        
        # Shuffle clients for random pairing
        shuffled = selected_clients.copy()
        random.shuffle(shuffled)
        
        sync_operations = []
        
        # Pair clients (if odd number, last one doesn't sync this round)
        for i in range(0, len(shuffled) - 1, 2):
            client1 = shuffled[i]
            client2 = shuffled[i + 1]
            
            # Randomly select which expert to sync
            expert_idx = random.randint(0, len(client1.experts) - 1)
            
            # Sync expert weights between peers
            sync_info = client1.sync_with(client2, expert_idx=expert_idx)
            
            if sync_info:
                sync_operations.append(sync_info)
                logger.info(
                    f"  Client {client1.client_id} <-> Client {client2.client_id}: "
                    f"Synced {sync_info['expert_synced']}"
                )
        
        if len(shuffled) % 2 == 1:
            logger.info(f"  Client {shuffled[-1].client_id}: No peer this round (odd number)")
        
        logger.info(f"\n  Total P2P syncs: {len(sync_operations)}")
        
        # Phase 4: Server aggregates insights and updates meta-params
        logger.info("\n--- Server Aggregation Phase ---")
        server.memory.extend(insights)
        
        if server.meta_learner:
            aggregated_knowledge = server.meta_learner.aggregate(insights)
            updated_params = server.meta_learner.update_global_params(insights)
            meta_params = server.meta_learner.broadcast_params()
            
            for client in selected_clients:
                client.receive_meta_params(meta_params)
        
        # Phase 5: Evaluation
        metrics = server.evaluate_global_model()
        
        logger.info(
            f"\nRound {round_num} Summary:\n"
            f"  Global Loss: {metrics['avg_loss']:.4f}\n"
            f"  P2P Syncs: {len(sync_operations)}\n"
            f"  Insights Collected: {len(insights)}\n"
            f"  Total Memory: {len(server.memory)}"
        )
    
    logger.info("\n" + "=" * 80)
    logger.info("P2P Gossip Training Complete!")
    logger.info("=" * 80)
    
    # Summary
    logger.info("\nFinal System State:")
    logger.info(f"  Total Insights: {len(server.memory)}")
    logger.info(f"  MetaLearner Updates: {meta_learner.global_params['num_updates']}")
    
    summary = meta_learner.summarize()
    logger.info(f"  Final Meta-Mean: {summary['global_params']['mean']:.4f}")
    logger.info(f"  Final Meta-Std: {summary['global_params']['std']:.4f}")
    
    return server, clients, meta_learner


def main():
    """
    Main execution function for the Scarcity Framework.
    """
    # Setup logging
    logger = setup_logging(level=logging.INFO)
    logger.info("=" * 80)
    logger.info("Starting Scarcity Framework (SF-HFE)")
    logger.info("=" * 80)
    
    # Configuration
    NUM_CLIENTS = 5
    NUM_ROUNDS = 5
    INPUT_DIM = 10
    OUTPUT_DIM = 1
    
    # Choose which mode to run
    MODE = "p2p_gossip"  # Options: "stage1", "federated_learning", "insight_exchange", "expert_routing", "meta_learning", or "p2p_gossip"
    
    if MODE == "stage1":
        logger.info("\nRunning in STAGE 1 mode - Simple Training & Insights")
        logger.info("=" * 80)
        insights = run_stage1_simulation(num_clients=3)
        
    elif MODE == "federated_learning":
        logger.info("\nRunning in FEDERATED LEARNING mode (Stage 2)")
        logger.info("=" * 80)
        server, clients = run_federated_learning(
            num_clients=NUM_CLIENTS,
            num_rounds=NUM_ROUNDS,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM
        )
    elif MODE == "insight_exchange":
        logger.info("\nRunning in INSIGHT EXCHANGE mode (Stage 3)")
        logger.info("=" * 80)
        server, clients, meta_learner = run_insight_exchange(
            num_clients=NUM_CLIENTS,
            num_rounds=NUM_ROUNDS,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM
        )
    elif MODE == "expert_routing":
        logger.info("\nRunning in EXPERT ROUTING mode (Stage 4)")
        logger.info("=" * 80)
        server, clients, meta_learner = run_expert_routing(
            num_clients=NUM_CLIENTS,
            num_rounds=NUM_ROUNDS,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            router_strategy="variance"  # Options: "variance", "random", "round_robin"
        )
    
    elif MODE == "meta_learning":
        logger.info("\nRunning in META-LEARNING mode (Stage 5)")
        logger.info("=" * 80)
        server, clients, meta_learner = run_meta_learning(
            num_clients=NUM_CLIENTS,
            num_rounds=NUM_ROUNDS,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM
        )
    
    elif MODE == "p2p_gossip":
        logger.info("\nRunning in P2P GOSSIP mode (Stage 6)")
        logger.info("=" * 80)
        server, clients, meta_learner = run_p2p_gossip(
            num_clients=NUM_CLIENTS,
            num_rounds=NUM_ROUNDS,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM
        )
    
    logger.info("\n" + "=" * 80)
    logger.info("Framework execution completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

