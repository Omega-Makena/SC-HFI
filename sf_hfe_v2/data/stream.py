"""
Synthetic Data Stream Generator
Creates online streaming data with concept drift for testing SF-HFE
"""

import torch
import numpy as np
from typing import Iterator, Tuple, Dict
import logging

from config import DATA_CONFIG, STREAM_CONFIG


class ConceptDriftStream:
    """
    Generates synthetic data stream with concept drift
    
    Simulates real-world scenarios where data distribution changes over time
    """
    
    def __init__(
        self,
        num_features: int = 20,
        num_classes: int = 5,
        stream_length: int = 10000,
        drift_points: list = None,
        drift_magnitude: float = 0.3,
        noise_level: float = 0.1,
        seed: int = None
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        self.stream_length = stream_length
        self.noise_level = noise_level
        
        # Drift configuration
        if drift_points is None:
            drift_points = [2500, 5000, 7500]
        self.drift_points = sorted(drift_points)
        self.drift_magnitude = drift_magnitude
        
        # Random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Current distribution parameters
        self.current_concept = 0
        self.concept_means = self._generate_concept_means()
        self.concept_covariances = self._generate_concept_covariances()
        
        # Stream state
        self.sample_index = 0
        
        # Logger
        self.logger = logging.getLogger("DataStream")
        
    def _generate_concept_means(self) -> List[torch.Tensor]:
        """
        Generate mean vectors for each concept
        """
        num_concepts = len(self.drift_points) + 1
        means = []
        
        for i in range(num_concepts):
            # Random mean for each concept
            mean = torch.randn(self.num_features) * 2.0
            means.append(mean)
        
        return means
    
    def _generate_concept_covariances(self) -> List[torch.Tensor]:
        """
        Generate covariance matrices for each concept
        """
        num_concepts = len(self.drift_points) + 1
        covariances = []
        
        for i in range(num_concepts):
            # Random covariance (ensure positive definite)
            A = torch.randn(self.num_features, self.num_features)
            cov = torch.matmul(A, A.T) / self.num_features + torch.eye(self.num_features) * 0.5
            covariances.append(cov)
        
        return covariances
    
    def _get_current_concept(self) -> int:
        """
        Determine which concept we're in based on sample index
        """
        concept = 0
        for i, drift_point in enumerate(self.drift_points):
            if self.sample_index >= drift_point:
                concept = i + 1
        return concept
    
    def _sample_from_concept(self, concept_id: int, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample n examples from a specific concept
        """
        mean = self.concept_means[concept_id]
        cov = self.concept_covariances[concept_id]
        
        # Sample from multivariate Gaussian
        dist = torch.distributions.MultivariateNormal(mean, cov)
        x = dist.sample((n,))
        
        # Add noise
        x += torch.randn_like(x) * self.noise_level
        
        # Generate targets (simple linear relationship + concept-specific bias)
        # y = w^T x + b + noise
        w = torch.randn(self.num_features, 1)
        b = torch.tensor([concept_id * 0.5])  # Concept-specific bias
        
        y = torch.matmul(x, w) + b
        y += torch.randn(n, 1) * self.noise_level
        
        return x, y
    
    def generate_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a single mini-batch
        
        Returns:
            x: Input features [batch_size, num_features]
            y: Targets [batch_size, 1]
        """
        # Check for concept drift
        new_concept = self._get_current_concept()
        if new_concept != self.current_concept:
            self.logger.info(
                f"DataStream: Concept drift at sample {self.sample_index} "
                f"(Concept {self.current_concept} -> {new_concept})"
            )
            self.current_concept = new_concept
        
        # Sample from current concept
        x, y = self._sample_from_concept(self.current_concept, batch_size)
        
        self.sample_index += batch_size
        
        return x, y
    
    def stream_batches(self, batch_size: int = None) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate infinite stream of batches
        
        Yields:
            (x, y) tuples
        """
        if batch_size is None:
            batch_size = STREAM_CONFIG["mini_batch_size"]
        
        while self.sample_index < self.stream_length:
            yield self.generate_batch(batch_size)
    
    def reset(self):
        """Reset stream to beginning"""
        self.sample_index = 0
        self.current_concept = 0
    
    def get_stats(self) -> Dict:
        """Stream statistics"""
        return {
            "sample_index": self.sample_index,
            "current_concept": self.current_concept,
            "progress": self.sample_index / self.stream_length,
            "num_concepts": len(self.concept_means),
            "drift_points": self.drift_points,
        }


class MultiClientStreamGenerator:
    """
    Generates separate data streams for multiple clients
    
    Each client gets their own stream with potentially different distributions
    """
    
    def __init__(
        self,
        num_clients: int,
        num_features: int = 20,
        stream_length: int = 10000,
        heterogeneous: bool = True
    ):
        self.num_clients = num_clients
        self.num_features = num_features
        self.stream_length = stream_length
        self.heterogeneous = heterogeneous
        
        # Create stream for each client
        self.client_streams = {}
        
        for client_id in range(num_clients):
            # Different drift points for heterogeneous
            if heterogeneous:
                drift_points = [
                    2500 + np.random.randint(-500, 500),
                    5000 + np.random.randint(-500, 500),
                    7500 + np.random.randint(-500, 500),
                ]
            else:
                drift_points = [2500, 5000, 7500]
            
            # Different seeds for each client
            seed = client_id * 42
            
            stream = ConceptDriftStream(
                num_features=num_features,
                num_classes=5,
                stream_length=stream_length,
                drift_points=drift_points,
                drift_magnitude=0.3 if heterogeneous else 0.3,
                noise_level=0.1 + (0.05 * client_id if heterogeneous else 0),
                seed=seed
            )
            
            self.client_streams[client_id] = stream
        
        logging.getLogger("MultiClientStream").info(
            f"Initialized {num_clients} client streams ({'heterogeneous' if heterogeneous else 'homogeneous'})"
        )
    
    def get_batch(self, client_id: int, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get next batch for a specific client
        """
        if client_id not in self.client_streams:
            raise ValueError(f"Client {client_id} not found")
        
        return self.client_streams[client_id].generate_batch(batch_size)
    
    def get_stream_iterator(self, client_id: int, batch_size: int = 32) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get stream iterator for a specific client
        """
        if client_id not in self.client_streams:
            raise ValueError(f"Client {client_id} not found")
        
        return self.client_streams[client_id].stream_batches(batch_size)
    
    def reset_all(self):
        """Reset all client streams"""
        for stream in self.client_streams.values():
            stream.reset()
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all client streams"""
        return {
            client_id: stream.get_stats()
            for client_id, stream in self.client_streams.items()
        }

