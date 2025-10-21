"""
Client Module

The Client class represents a participant in the federated learning process.
Each client holds local data and can train models locally.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
from .expert import StructureExpert, DriftExpert, MemoryConsolidationExpert, MetaAdaptationExpert
from .router import Router


class Client:
    """
    A federated learning client in the Scarcity Framework.
    
    Each client represents a data owner or edge device that participates
    in the federated learning process without sharing raw data.
    """
    
    def __init__(self, client_id: int, input_dim: int = 10, output_dim: int = 1, 
                 data_size: int = 100, use_experts: bool = False, 
                 router_strategy: str = "variance", **kwargs):
        """
        Initialize the Client.
        
        Args:
            client_id: Unique identifier for this client
            input_dim: Input dimension for the model
            output_dim: Output dimension for the model
            data_size: Number of local data samples
            use_experts: If True, use expert routing architecture
            router_strategy: Strategy for expert selection ("variance", "random", "round_robin")
            **kwargs: Additional configuration parameters
        """
        self.client_id = client_id
        self.logger = logging.getLogger(f"{__name__}.Client{client_id}")
        self.logger.info(f"Initializing Client {client_id}")
        
        # Create local model (simple linear model)
        self.model = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_experts = use_experts
        
        # Initialize experts and router if enabled
        if use_experts:
            self.experts = [
                StructureExpert(expert_id=0, input_dim=input_dim, output_dim=output_dim),
                DriftExpert(expert_id=1, input_dim=input_dim, output_dim=output_dim),
                MemoryConsolidationExpert(expert_id=2, input_dim=input_dim, output_dim=output_dim, memory_size=50),
                MetaAdaptationExpert(expert_id=3, input_dim=input_dim, output_dim=output_dim)
            ]
            self.router = Router(experts=self.experts, strategy=router_strategy)
            self.logger.info(f"Client {client_id}: Initialized with {len(self.experts)} experts and Router (strategy={router_strategy})")
            self.selected_expert = None  # Track which expert was used
        else:
            self.experts = None
            self.router = None
        
        # Generate synthetic local data (non-IID by adding client-specific bias)
        np.random.seed(client_id * 42)  # Different seed per client
        self.X_train = torch.randn(data_size, input_dim)
        # Add client-specific bias to create non-IID data
        self.X_train += torch.randn(1, input_dim) * 0.5 * client_id
        
        # Generate labels with some noise
        true_weights = torch.randn(input_dim, output_dim)
        self.y_train = self.X_train @ true_weights + torch.randn(data_size, output_dim) * 0.1
        
        self.logger.info(f"Client {client_id} created with {data_size} local samples")
        
    def local_train(self, epochs: int = 5, lr: float = 0.01):
        """
        Train the model locally on client data.
        
        Args:
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Local model weights
        """
        self.logger.info(f"Client {self.client_id}: Starting local training for {epochs} epochs")
        
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        
        initial_loss = None
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(self.X_train)
            loss = criterion(outputs, self.y_train)
            
            if epoch == 0:
                initial_loss = loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        final_loss = loss.item()
        self.logger.info(f"Client {self.client_id}: Trained locally - Loss: {initial_loss:.4f} -> {final_loss:.4f}")
        
        return self.get_weights()
    
    def generate_insight(self, epochs: int = 5, lr: float = 0.01):
        """
        Train locally and generate structured insight instead of raw weights.
        
        This is the Scarcity Framework's approach: share knowledge, not data or raw weights.
        
        Args:
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Structured insight dictionary with metadata about local learning
        """
        self.logger.info(f"Client {self.client_id}: Generating insight through local training")
        
        # If using experts, route and train with selected expert
        if self.use_experts and self.router:
            # Router selects expert based on data characteristics
            selected_expert = self.router.select_expert(self.X_train)
            self.selected_expert = selected_expert  # Track for logging
            
            self.logger.info(
                f"Client {self.client_id}: Router selected {type(selected_expert).__name__}"
            )
            
            # Train with selected expert
            metrics = selected_expert.train(self.X_train, self.y_train, epochs=epochs, lr=lr)
            
            # Collect summaries from all experts
            expert_summaries = []
            for expert in self.experts:
                summary = expert.summarize(self.X_train)
                expert_summaries.append(summary)
            
            # Create insight with expert information
            insight = {
                "client_id": self.client_id,
                "selected_expert": type(selected_expert).__name__,
                "selected_expert_id": selected_expert.expert_id,
                "loss_improvement": float(metrics["initial_loss"] - metrics["final_loss"]),
                "final_loss": float(metrics["final_loss"]),
                "num_samples": len(self.X_train),
                "epochs_trained": epochs,
                "expert_summaries": expert_summaries,
                "num_experts": len(self.experts)
            }
            
            self.logger.info(
                f"Client {self.client_id}: Generated expert insight - "
                f"expert={type(selected_expert).__name__}, "
                f"loss: {metrics['initial_loss']:.4f} -> {metrics['final_loss']:.4f}"
            )
            
            return insight
        
        else:
            # Original non-expert mode
            criterion = nn.MSELoss()
            optimizer = optim.SGD(self.model.parameters(), lr=lr)
            
            # Track gradients and losses
            gradient_norms = []
            losses = []
            
            for epoch in range(epochs):
                # Forward pass
                outputs = self.model(self.X_train)
                loss = criterion(outputs, self.y_train)
                losses.append(loss.item())
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Collect gradient information
                grad_norm = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                gradient_norms.append(grad_norm)
                
                optimizer.step()
            
            # Calculate insight metrics
            mean_grad = np.mean(gradient_norms)
            std_grad = np.std(gradient_norms)
            loss_improvement = losses[0] - losses[-1]
            final_loss = losses[-1]
            
            # Uncertainty score: based on gradient variance and loss stability
            # Higher uncertainty = more volatile training
            uncertainty_score = std_grad / (mean_grad + 1e-8) + abs(np.std(losses))
            
            insight = {
                "client_id": self.client_id,
                "mean_grad": float(mean_grad),
                "std_grad": float(std_grad),
                "uncertainty": float(uncertainty_score),
                "loss_improvement": float(loss_improvement),
                "final_loss": float(final_loss),
                "num_samples": len(self.X_train),
                "epochs_trained": epochs
            }
            
            self.logger.info(
                f"Client {self.client_id}: Generated insight - "
                f"mean_grad={mean_grad:.4f}, uncertainty={uncertainty_score:.4f}, "
                f"loss: {losses[0]:.4f} -> {final_loss:.4f}"
            )
            
            return insight
    
    def evaluate(self):
        """
        Evaluate the model on local data.
        
        Returns:
            Evaluation metrics (loss)
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_train)
            criterion = nn.MSELoss()
            loss = criterion(outputs, self.y_train)
        
        self.logger.info(f"Client {self.client_id}: Evaluation loss = {loss.item():.4f}")
        return {"loss": loss.item()}
    
    def get_data_summary(self):
        """
        Get a summary of the local data distribution.
        
        Returns:
            Summary statistics about local data
        """
        summary = {
            "num_samples": len(self.X_train),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "data_mean": self.X_train.mean().item(),
            "data_std": self.X_train.std().item()
        }
        self.logger.info(f"Client {self.client_id}: Data summary - {summary}")
        return summary
    
    def receive_global_model(self, model_weights):
        """
        Receive and apply global model weights from the server.
        
        Args:
            model_weights: Weights from the global model (state_dict)
        """
        self.model.load_state_dict(deepcopy(model_weights))
        self.logger.info(f"Client {self.client_id}: Received and loaded global model weights")
    
    def get_weights(self):
        """
        Get current model weights.
        
        Returns:
            Model state_dict
        """
        return deepcopy(self.model.state_dict())
    
    def send_update(self):
        """
        Send local model updates to the server.
        
        Returns:
            Model updates to send to server
        """
        weights = self.get_weights()
        self.logger.info(f"Client {self.client_id}: Sending model update to server")
        return weights
    
    def receive_meta_params(self, meta_params: dict):
        """
        Receive meta-learning parameters from the server.
        
        These parameters can be used to improve initialization and training.
        
        Args:
            meta_params: Dictionary of meta-learning parameters
        """
        self.meta_params = meta_params
        
        self.logger.info(
            f"Client {self.client_id}: Received meta-parameters - "
            f"mean={meta_params.get('meta_mean', 0):.4f}, "
            f"std={meta_params.get('meta_std', 1):.4f}, "
            f"lr={meta_params.get('meta_lr', 0.01):.4f}, "
            f"updates={meta_params.get('meta_updates', 0)}"
        )
    
    def get_expert_weights(self, expert_idx: int):
        """
        Get weights from a specific expert.
        
        Args:
            expert_idx: Index of the expert (0, 1, etc.)
            
        Returns:
            Expert model weights (state_dict)
        """
        if self.experts and expert_idx < len(self.experts):
            expert = self.experts[expert_idx]
            return deepcopy(expert.model.state_dict())
        return None
    
    def set_expert_weights(self, expert_idx: int, weights):
        """
        Set weights for a specific expert.
        
        Args:
            expert_idx: Index of the expert
            weights: Model weights (state_dict) to set
        """
        if self.experts and expert_idx < len(self.experts):
            expert = self.experts[expert_idx]
            expert.model.load_state_dict(deepcopy(weights))
    
    def sync_with(self, peer_client, expert_idx: int = None):
        """
        Peer-to-peer synchronization: average expert weights with another client.
        
        This implements a gossip-style P2P mechanism where clients can directly
        exchange knowledge without going through the central server.
        
        Args:
            peer_client: Another Client object to sync with
            expert_idx: Index of expert to sync (None = random selection)
            
        Returns:
            Dictionary with sync information
        """
        if not self.use_experts or not self.experts:
            self.logger.warning(f"Client {self.client_id}: Cannot sync - no experts available")
            return None
        
        if not peer_client.use_experts or not peer_client.experts:
            self.logger.warning(f"Client {self.client_id}: Cannot sync - peer has no experts")
            return None
        
        # Select which expert to sync (random if not specified)
        if expert_idx is None:
            import random
            expert_idx = random.randint(0, len(self.experts) - 1)
        
        if expert_idx >= len(self.experts) or expert_idx >= len(peer_client.experts):
            self.logger.warning(f"Client {self.client_id}: Invalid expert index {expert_idx}")
            return None
        
        # Get expert names
        my_expert = self.experts[expert_idx]
        peer_expert = peer_client.experts[expert_idx]
        expert_name = type(my_expert).__name__
        
        # Get current weights from both clients
        my_weights = my_expert.model.state_dict()
        peer_weights = peer_expert.model.state_dict()
        
        # Average the weights (gossip-style)
        averaged_weights = {}
        for key in my_weights.keys():
            averaged_weights[key] = (my_weights[key] + peer_weights[key]) / 2.0
        
        # Update both clients with averaged weights
        my_expert.model.load_state_dict(deepcopy(averaged_weights))
        peer_expert.model.load_state_dict(deepcopy(averaged_weights))
        
        sync_info = {
            "client_id": self.client_id,
            "peer_id": peer_client.client_id,
            "expert_synced": expert_name,
            "expert_idx": expert_idx
        }
        
        self.logger.info(
            f"Client {self.client_id}: Synced {expert_name} with Client {peer_client.client_id} (P2P gossip)"
        )
        
        return sync_info

