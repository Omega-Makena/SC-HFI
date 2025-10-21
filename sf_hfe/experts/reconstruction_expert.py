"""
Reconstruction Expert - Core Structure Expert #3
Specializes in VAE-based reconstruction and data fidelity
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_expert import BaseExpert


class ReconstructionExpert(BaseExpert):
    """
    Expert specializing in reconstruction fidelity using VAE
    
    Capabilities:
    - Variational Autoencoder learning
    - Data reconstruction quality
    - Latent space representation
    - Anomaly detection via reconstruction error
    """
    
    def __init__(self, input_dim: int, output_dim: int, latent_dim: int = 16, **kwargs):
        self.latent_dim = latent_dim
        
        super().__init__(
            expert_id=2,
            input_dim=input_dim,
            output_dim=output_dim,
            expert_type="structure",
            **kwargs
        )
        
        # Build VAE architecture AFTER calling super()
        # VAE Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # VAE Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )
        
        # Prediction head (from latent)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
        
        # Create optimizer now
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
        
        # Reconstruction metrics
        self.reconstruction_errors = []
        self.kl_divergences = []
        
    def _build_network(self) -> nn.Module:
        """Override base network"""
        return nn.Identity()
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent space"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode to latent, then predict
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        prediction = self.predictor(z)
        return prediction
    
    def reconstruct(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE pass: encode, reparameterize, decode
        Returns: reconstruction, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def _task_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Simplified loss for now (full VAE loss requires access to inputs)
        """
        # Standard prediction loss
        pred_loss = nn.functional.mse_loss(outputs, targets)
        
        # Note: Full VAE reconstruction will be done in a separate training phase
        # For now, just use standard loss to get system working
        
        return pred_loss
    
    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Anomaly detection via reconstruction error
        """
        self.eval()
        with torch.no_grad():
            recon, _, _ = self.reconstruct(x)
            error = torch.mean((x - recon) ** 2, dim=1)
        self.train()
        return error
    
    def generate_insight(self) -> Dict:
        """
        Reconstruction-specific insights for FL
        """
        insight = super().generate_insight()
        
        avg_recon_error = sum(self.reconstruction_errors) / len(self.reconstruction_errors) if self.reconstruction_errors else 0.0
        avg_kl = sum(self.kl_divergences) / len(self.kl_divergences) if self.kl_divergences else 0.0
        
        insight.update({
            "specialization": "reconstruction",
            "avg_reconstruction_error": float(avg_recon_error),
            "avg_kl_divergence": float(avg_kl),
            "latent_dimensionality": self.latent_dim,
            "reconstruction_fidelity": 1.0 / (1.0 + avg_recon_error),  # Higher is better
        })
        
        return insight

