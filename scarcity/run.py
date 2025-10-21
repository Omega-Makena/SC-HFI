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

from scarcity.core import Expert, Router, Client, Server, MetaLearner


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
    MODE = "insight_exchange"  # Options: "federated_learning" or "insight_exchange"
    
    if MODE == "federated_learning":
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
    
    logger.info("\n" + "=" * 80)
    logger.info("Framework execution completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

