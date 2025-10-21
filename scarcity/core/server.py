"""
Server Module

The Server class coordinates the federated learning process across multiple clients.
It aggregates client updates and manages the global model.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy


class Server:
    """
    Federated learning server in the Scarcity Framework.
    
    The server orchestrates the federated learning process by coordinating
    clients, aggregating their updates, and maintaining the global model(s).
    """
    
    def __init__(self, clients: list, meta_learner=None, input_dim: int = 10, 
                 output_dim: int = 1, client_fraction: float = 1.0, **kwargs):
        """
        Initialize the Server.
        
        Args:
            clients: List of Client objects in the federation
            meta_learner: MetaLearner instance for insight aggregation
            input_dim: Input dimension for the global model
            output_dim: Output dimension for the global model
            client_fraction: Fraction of clients to select per round
            **kwargs: Additional configuration parameters
        """
        self.clients = clients
        self.num_clients = len(clients)
        self.client_fraction = client_fraction
        self.meta_learner = meta_learner
        self.memory = []  # Store all insights for knowledge accumulation
        self.logger = logging.getLogger(f"{__name__}.Server")
        self.logger.info(f"Initializing Server with {self.num_clients} clients")
        
        # Initialize global model
        self.global_model = nn.Linear(input_dim, output_dim)
        self.logger.info(f"Global model initialized: Linear({input_dim}, {output_dim})")
        self.logger.info(f"Server memory initialized for insight storage")
        
    def aggregate_models(self, client_updates):
        """
        Aggregate model updates from multiple clients using FedAvg.
        
        Args:
            client_updates: List of model state_dicts from clients
            
        Returns:
            Aggregated global model weights
        """
        self.logger.info(f"Server: Aggregating models from {len(client_updates)} clients")
        
        if not client_updates:
            self.logger.warning("No client updates to aggregate")
            return self.global_model.state_dict()
        
        # FedAvg: Average all parameters
        aggregated_state = {}
        
        # Get keys from first client
        keys = client_updates[0].keys()
        
        for key in keys:
            # Average weights across all clients
            aggregated_state[key] = torch.stack([
                client_update[key].float() for client_update in client_updates
            ]).mean(dim=0)
        
        # Update global model
        self.global_model.load_state_dict(aggregated_state)
        
        self.logger.info("Server: New global weights aggregated using FedAvg")
        return deepcopy(aggregated_state)
    
    def select_clients(self, round_num: int):
        """
        Select which clients should participate in this round.
        
        Args:
            round_num: Current training round number
            
        Returns:
            List of selected Client objects
        """
        num_selected = max(1, int(self.num_clients * self.client_fraction))
        
        # For simplicity, select all clients or round-robin
        # In practice, this could be random sampling
        selected_clients = self.clients[:num_selected]
        
        selected_ids = [c.client_id for c in selected_clients]
        self.logger.info(f"Server: Selected {len(selected_clients)} clients for round {round_num}: {selected_ids}")
        
        return selected_clients
    
    def broadcast_model(self, clients):
        """
        Broadcast global model weights to selected clients.
        
        Args:
            clients: List of Client objects to broadcast to
        """
        global_weights = self.global_model.state_dict()
        
        for client in clients:
            client.receive_global_model(global_weights)
        
        self.logger.info(f"Server: Broadcasted global model to {len(clients)} clients")
    
    def evaluate_global_model(self, test_clients=None):
        """
        Evaluate the global model on client data.
        
        Args:
            test_clients: Clients to evaluate on (uses all clients if None)
            
        Returns:
            Average evaluation metrics
        """
        if test_clients is None:
            test_clients = self.clients
        
        total_loss = 0.0
        num_samples = 0
        
        for client in test_clients:
            # Temporarily set client model to global model
            original_weights = client.get_weights()
            client.receive_global_model(self.global_model.state_dict())
            
            # Evaluate
            metrics = client.evaluate()
            total_loss += metrics["loss"] * len(client.X_train)
            num_samples += len(client.X_train)
            
            # Restore original weights
            client.receive_global_model(original_weights)
        
        avg_loss = total_loss / num_samples
        self.logger.info(f"Server: Global model average loss = {avg_loss:.4f}")
        
        return {"avg_loss": avg_loss}
    
    def run_round(self, round_num: int):
        """
        Execute one round of federated learning.
        
        Args:
            round_num: Current round number
        """
        self.logger.info(f"Server: Starting round {round_num}")
        
        # 1. Select clients for this round
        selected_clients = self.select_clients(round_num)
        
        # 2. Broadcast current global model
        self.broadcast_model(selected_clients)
        
        # 3. Each client trains locally
        client_updates = []
        for client in selected_clients:
            updated_weights = client.local_train(epochs=5)
            client_updates.append(updated_weights)
        
        # 4. Aggregate client models
        self.aggregate_models(client_updates)
        
        # 5. Evaluate global model
        metrics = self.evaluate_global_model()
        
        self.logger.info(f"Server: Round {round_num} complete - Global loss: {metrics['avg_loss']:.4f}")
        return metrics
    
    def run_insight_round(self, round_num: int):
        """
        Execute one round of Scarcity-style Insight Exchange.
        
        Instead of exchanging raw weights, clients generate and share
        structured insights about their local learning.
        
        Args:
            round_num: Current round number
            
        Returns:
            Dictionary with aggregated metrics and insights
        """
        self.logger.info(f"Server: Starting INSIGHT EXCHANGE round {round_num}")
        
        # 1. Select clients for this round
        selected_clients = self.select_clients(round_num)
        
        # 2. Broadcast current global model (for local training baseline)
        self.broadcast_model(selected_clients)
        
        # 3. Each client generates insights (not raw weights!)
        insights = []
        for client in selected_clients:
            insight = client.generate_insight(epochs=5)
            insights.append(insight)
            
            # Log differently based on whether experts are used
            if 'selected_expert' in insight:
                self.logger.info(
                    f"Server: Received insight from Client {client.client_id} - "
                    f"expert={insight['selected_expert']}, loss={insight['final_loss']:.4f}"
                )
            elif 'uncertainty' in insight:
                self.logger.info(
                    f"Server: Received insight from Client {client.client_id} - "
                    f"uncertainty={insight['uncertainty']:.4f}"
                )
            else:
                self.logger.info(
                    f"Server: Received insight from Client {client.client_id}"
                )
        
        # 4. Store insights in memory
        self.memory.extend(insights)
        self.logger.info(f"Server: Stored {len(insights)} insights in memory (total: {len(self.memory)})")
        
        # 5. Use MetaLearner to aggregate insights
        if self.meta_learner:
            aggregated_knowledge = self.meta_learner.aggregate(insights)
            self.logger.info(f"Server: MetaLearner aggregated knowledge from {len(insights)} clients")
        else:
            self.logger.warning("Server: No MetaLearner available, skipping aggregation")
            aggregated_knowledge = {"avg_uncertainty": np.mean([i["uncertainty"] for i in insights])}
        
        # 6. Evaluate global model
        metrics = self.evaluate_global_model()
        metrics["aggregated_knowledge"] = aggregated_knowledge
        metrics["num_insights"] = len(insights)
        
        self.logger.info(
            f"Server: Round {round_num} complete - "
            f"Global loss: {metrics['avg_loss']:.4f}, "
            f"Avg uncertainty: {aggregated_knowledge.get('avg_uncertainty', 0):.4f}"
        )
        
        return metrics

