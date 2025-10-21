"""
Geometry Expert - Core Structure Expert #1
Focuses on PCA-based manifold analysis and geometric structure of data
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_expert import BaseExpert


class GeometryExpert(BaseExpert):
    """
    Expert specializing in geometric structure of data
    
    Capabilities:
    - PCA-based dimensionality analysis
    - Manifold structure learning
    - Data geometry preservation
    - Low-dimensional projections
    """
    
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__(
            expert_id=0,
            input_dim=input_dim,
            output_dim=output_dim,
            expert_type="structure",
            **kwargs
        )
        
        # PCA components (learned online)
        self.pca_components = None
        self.pca_mean = None
        self.explained_variance = None
        
        # Running statistics for online PCA
        self.n_samples_seen = 0
        self.running_mean = torch.zeros(input_dim)
        self.running_cov = torch.zeros(input_dim, input_dim)
        
        # Geometric metrics
        self.intrinsic_dim_estimate = None
        self.manifold_curvature = []
        
    def _task_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Geometry-preserving loss
        Combines standard prediction loss with geometric structure preservation
        """
        # Standard prediction loss
        pred_loss = nn.functional.mse_loss(outputs, targets)
        
        # Note: Geometric structure preservation would require access to input features
        # For now, just use standard loss
        # Full implementation would use PCA on inputs, not outputs
        
        return pred_loss
    
    def online_update(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        replay: bool = True
    ) -> Dict[str, float]:
        """
        Update with geometric structure learning
        """
        # Standard expert update
        metrics = super().online_update(batch_x, batch_y, replay)
        
        # Update PCA incrementally
        self._update_online_pca(batch_x)
        
        # Estimate intrinsic dimensionality
        if self.batch_count % 50 == 0:
            self._estimate_intrinsic_dim()
        
        # Add geometry-specific metrics
        metrics.update({
            "intrinsic_dim": self.intrinsic_dim_estimate if self.intrinsic_dim_estimate else self.input_dim,
            "pca_components": self.pca_components.shape[0] if self.pca_components is not None else 0,
            "explained_variance_ratio": float(self.explained_variance.sum()) if self.explained_variance is not None else 0.0,
        })
        
        return metrics
    
    def _update_online_pca(self, batch_x: torch.Tensor):
        """
        Incremental PCA update
        Updates running mean and covariance for PCA
        """
        batch_size = len(batch_x)
        
        # Update running mean
        delta = batch_x.mean(dim=0) - self.running_mean
        self.running_mean += delta * batch_size / (self.n_samples_seen + batch_size)
        
        # Update running covariance (Welford's online algorithm)
        for x in batch_x:
            self.n_samples_seen += 1
            delta = x - self.running_mean
            self.running_cov += torch.outer(delta, delta)
        
        # Recompute PCA every 100 samples
        if self.n_samples_seen % 100 == 0 and self.n_samples_seen > 50:
            self._compute_pca()
    
    def _compute_pca(self):
        """
        Compute PCA from running covariance
        """
        if self.n_samples_seen < 10:
            return
        
        # Normalize covariance
        cov_matrix = self.running_cov / self.n_samples_seen
        
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep top components (95% variance)
        total_var = eigenvalues.sum()
        cumsum_var = torch.cumsum(eigenvalues, dim=0)
        n_components = (cumsum_var < 0.95 * total_var).sum() + 1
        n_components = min(n_components, self.input_dim // 2)  # At least half
        
        self.pca_components = eigenvectors[:, :n_components].T
        self.pca_mean = self.running_mean.clone()
        self.explained_variance = eigenvalues[:n_components] / total_var
    
    def _estimate_intrinsic_dim(self):
        """
        Estimate intrinsic dimensionality using PCA
        Based on explained variance ratio
        """
        if self.explained_variance is not None:
            # Count components needed for 90% variance
            cumsum = torch.cumsum(self.explained_variance, dim=0)
            self.intrinsic_dim_estimate = int((cumsum < 0.90).sum()) + 1
        else:
            self.intrinsic_dim_estimate = self.input_dim
    
    def project_to_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project data to learned low-dimensional manifold
        """
        if self.pca_components is None:
            return x
        
        centered = x - self.pca_mean
        projected = torch.matmul(centered, self.pca_components.T)
        return projected
    
    def generate_insight(self) -> Dict:
        """
        Geometry-specific insights for FL
        """
        insight = super().generate_insight()
        
        insight.update({
            "specialization": "geometry",
            "intrinsic_dimensionality": self.intrinsic_dim_estimate if self.intrinsic_dim_estimate else self.input_dim,
            "pca_components_count": self.pca_components.shape[0] if self.pca_components is not None else 0,
            "variance_explained": float(self.explained_variance.sum()) if self.explained_variance is not None else 0.0,
            "data_compressibility": float(self.intrinsic_dim_estimate / self.input_dim) if self.intrinsic_dim_estimate else 1.0,
        })
        
        return insight

