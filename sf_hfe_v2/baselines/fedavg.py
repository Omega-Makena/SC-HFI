"""
FedAvg Baseline - P0 Critical
Standard Federated Averaging for comparison

Reference: McMahan et al. (2017) - Communication-Efficient Learning of Deep Networks
"""

import torch
import torch.nn as nn
from typing import Dict, List
import logging
import copy


class FedAvgClient:
    """
    Standard FedAvg client
    
    Performs local SGD and returns model updates
    """
    
    def __init__(self, client_id: int, model: nn.Module, learning_rate: float = 0.01):
        self.client_id = client_id
        self.model = model
        self.lr = learning_rate
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.logger = logging.getLogger(f"FedAvgClient{client_id}")
        
    def local_train(self, data_loader, epochs: int = 1):
        """
        Perform local training (standard SGD)
        
        Args:
            data_loader: Local data
            epochs: Number of local epochs
            
        Returns:
            Number of samples trained on
        """
        self.model.train()
        total_samples = 0
        
        for epoch in range(epochs):
            for batch_x, batch_y in data_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_samples += len(batch_x)
        
        return total_samples
    
    def get_model_update(self) -> Dict[str, torch.Tensor]:
        """
        Get current model parameters
        
        Returns:
            Dictionary of model weights
        """
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_model_params(self, params: Dict[str, torch.Tensor]):
        """
        Set model parameters from server
        
        Args:
            params: Model parameters
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.data.copy_(params[name])


class FedAvgServer:
    """
    Standard FedAvg server
    
    Aggregates client updates using weighted averaging
    """
    
    def __init__(self, global_model: nn.Module):
        self.global_model = global_model
        self.round = 0
        
        self.logger = logging.getLogger("FedAvgServer")
        
    def aggregate(self, client_updates: List[Dict[str, torch.Tensor]], 
                  client_weights: List[int]) -> Dict[str, torch.Tensor]:
        """
        FedAvg aggregation: weighted average of client models
        
        Args:
            client_updates: List of client model parameters
            client_weights: List of number of samples per client
            
        Returns:
            Aggregated global model parameters
        """
        if not client_updates:
            return self.get_global_params()
        
        # Total samples across all clients
        total_samples = sum(client_weights)
        
        if total_samples == 0:
            self.logger.warning("No samples to aggregate!")
            return self.get_global_params()
        
        # Weighted averaging
        aggregated = {}
        
        for key in client_updates[0].keys():
            # Weighted sum
            aggregated[key] = sum(
                client_updates[i][key] * (client_weights[i] / total_samples)
                for i in range(len(client_updates))
            )
        
        # Update global model
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated:
                    param.data.copy_(aggregated[name])
        
        self.round += 1
        self.logger.info(f"Round {self.round}: Aggregated {len(client_updates)} clients ({total_samples} samples)")
        
        return aggregated
    
    def get_global_params(self) -> Dict[str, torch.Tensor]:
        """Get current global model parameters"""
        return {name: param.data.clone() for name, param in self.global_model.named_parameters()}
    
    def broadcast(self, clients: List[FedAvgClient]):
        """
        Broadcast global model to all clients
        
        Args:
            clients: List of FedAvgClient instances
        """
        global_params = self.get_global_params()
        
        for client in clients:
            client.set_model_params(global_params)
        
        self.logger.info(f"Broadcasted global model to {len(clients)} clients")

