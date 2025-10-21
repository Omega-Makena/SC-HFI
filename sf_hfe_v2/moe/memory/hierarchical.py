"""
3-Tier Hierarchical Memory System for SF-HFE
Prevents catastrophic forgetting in online continual learning
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import List, Tuple, Optional
import random


class RecentBuffer:
    """Tier 1: FIFO buffer for recent samples (raw data)"""
    
    def __init__(self, max_size: int = 500):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add(self, x: torch.Tensor, y: torch.Tensor):
        """Add new sample to buffer"""
        self.buffer.append((x.cpu(), y.cpu()))
    
    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n recent examples"""
        if len(self.buffer) == 0:
            return None, None
        
        n = min(n, len(self.buffer))
        samples = random.sample(list(self.buffer), n)
        
        xs = torch.stack([s[0] for s in samples])
        ys = torch.stack([s[1] for s in samples])
        
        return xs, ys
    
    def size(self) -> int:
        return len(self.buffer)


class CompressedMemory:
    """Tier 2: Reservoir sampling with compressed latent vectors"""
    
    def __init__(self, max_size: int = 2000, latent_dim: int = 16):
        self.max_size = max_size
        self.latent_dim = latent_dim
        self.memory = []
        self.count = 0
        
        # Simple VAE for compression (will be trained online)
        self.encoder = nn.Sequential(
            nn.Linear(20, 32),  # Input dim will be set dynamically
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 20)  # Output dim matches input
        )
        
        self.compression_trained = False
    
    def set_dimensions(self, input_dim: int):
        """Set actual input dimensions"""
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    
    def train_compression(self, samples: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Train VAE compression on recent samples"""
        if len(samples) < 10:
            return
        
        xs = torch.stack([s[0] for s in samples])
        
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=0.001
        )
        
        # Quick training (10 epochs)
        for _ in range(10):
            latent = self.encoder(xs)
            reconstructed = self.decoder(latent)
            loss = nn.MSELoss()(reconstructed, xs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        self.compression_trained = True
    
    def add(self, x: torch.Tensor, y: torch.Tensor):
        """Reservoir sampling: add with probability"""
        self.count += 1
        
        # Compress to latent
        with torch.no_grad():
            if self.compression_trained:
                latent = self.encoder(x.unsqueeze(0)).squeeze(0)
            else:
                latent = x  # Store raw if not trained yet
        
        if len(self.memory) < self.max_size:
            self.memory.append((latent.cpu(), y.cpu()))
        else:
            # Reservoir sampling: replace with probability
            idx = random.randint(0, self.count - 1)
            if idx < self.max_size:
                self.memory[idx] = (latent.cpu(), y.cpu())
    
    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n compressed examples and decode"""
        if len(self.memory) == 0:
            return None, None
        
        n = min(n, len(self.memory))
        samples = random.sample(self.memory, n)
        
        # Check if we have mixed dimensions (compressed vs raw)
        # This can happen during initial training before compression is trained
        sample_dims = [s[0].shape for s in samples]
        if len(set([tuple(d) for d in sample_dims])) > 1:
            # Mixed dimensions - only use samples with consistent size
            # Use the most common dimension
            from collections import Counter
            dim_counts = Counter([tuple(d) for d in sample_dims])
            target_dim = dim_counts.most_common(1)[0][0]
            samples = [s for s in samples if tuple(s[0].shape) == target_dim]
            
            if len(samples) == 0:
                return None, None
        
        latents = torch.stack([s[0] for s in samples])
        ys = torch.stack([s[1] for s in samples])
        
        # Decode latents back to input space
        with torch.no_grad():
            if self.compression_trained and latents.shape[1] == self.latent_dim:
                xs = self.decoder(latents)
            else:
                xs = latents  # Raw samples if not compressed yet
        
        return xs, ys
    
    def size(self) -> int:
        return len(self.memory)


class CriticalAnchors:
    """Tier 3: Priority queue for critical/boundary examples"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.anchors = []
        self.priorities = []  # Higher = more important
    
    def add(self, x: torch.Tensor, y: torch.Tensor, uncertainty: float):
        """Add example with priority based on uncertainty"""
        if len(self.anchors) < self.max_size:
            self.anchors.append((x.cpu(), y.cpu()))
            self.priorities.append(uncertainty)
        else:
            # Replace lowest priority if new one is more important
            min_idx = np.argmin(self.priorities)
            if uncertainty > self.priorities[min_idx]:
                self.anchors[min_idx] = (x.cpu(), y.cpu())
                self.priorities[min_idx] = uncertainty
    
    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n critical examples (prioritized)"""
        if len(self.anchors) == 0:
            return None, None
        
        n = min(n, len(self.anchors))
        
        # Sample with probability proportional to priority
        probs = np.array(self.priorities)
        probs = probs / probs.sum()
        
        indices = np.random.choice(len(self.anchors), size=n, replace=False, p=probs)
        samples = [self.anchors[i] for i in indices]
        
        xs = torch.stack([s[0] for s in samples])
        ys = torch.stack([s[1] for s in samples])
        
        return xs, ys
    
    def size(self) -> int:
        return len(self.anchors)


class HierarchicalMemory:
    """
    Complete 3-Tier Memory System
    Manages all tiers and provides unified replay interface
    """
    
    def __init__(
        self,
        recent_size: int = 500,
        compressed_size: int = 2000,
        critical_size: int = 100,
        latent_dim: int = 16
    ):
        self.tier1 = RecentBuffer(max_size=recent_size)
        self.tier2 = CompressedMemory(max_size=compressed_size, latent_dim=latent_dim)
        self.tier3 = CriticalAnchors(max_size=critical_size)
        
        self.batch_count = 0
        self.compression_update_frequency = 100  # Update compression every 100 batches
    
    def add_batch(self, xs: torch.Tensor, ys: torch.Tensor, uncertainties: Optional[torch.Tensor] = None):
        """Add a mini-batch to memory"""
        for i in range(len(xs)):
            x, y = xs[i], ys[i]
            
            # Tier 1: Always add to recent buffer
            self.tier1.add(x, y)
            
            # Tier 2: Reservoir sampling to compressed memory
            self.tier2.add(x, y)
            
            # Tier 3: Add to critical anchors if high uncertainty
            if uncertainties is not None:
                uncertainty = uncertainties[i].item()
                if uncertainty > 0.5:  # High uncertainty threshold
                    self.tier3.add(x, y, uncertainty)
        
        self.batch_count += 1
        
        # Periodically update compression
        if self.batch_count % self.compression_update_frequency == 0:
            self._update_compression()
    
    def _update_compression(self):
        """Update VAE compression using recent samples"""
        if self.tier1.size() < 50:
            return
        
        # Sample from recent buffer
        xs, ys = self.tier1.sample(min(200, self.tier1.size()))
        if xs is None:
            return
        
        samples = [(xs[i], ys[i]) for i in range(len(xs))]
        self.tier2.train_compression(samples)
    
    def replay_batch(self, batch_size: int = 32, mix_ratios: dict = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample replay batch from all tiers
        
        Args:
            batch_size: Total number of samples
            mix_ratios: Dict with keys 'recent', 'compressed', 'critical'
        """
        if mix_ratios is None:
            mix_ratios = {"recent": 16, "compressed": 8, "critical": 8}
        
        # Sample from each tier
        xs_list = []
        ys_list = []
        
        # Tier 1: Recent
        xs1, ys1 = self.tier1.sample(mix_ratios["recent"])
        if xs1 is not None:
            xs_list.append(xs1)
            ys_list.append(ys1)
        
        # Tier 2: Compressed
        xs2, ys2 = self.tier2.sample(mix_ratios["compressed"])
        if xs2 is not None:
            xs_list.append(xs2)
            ys_list.append(ys2)
        
        # Tier 3: Critical
        xs3, ys3 = self.tier3.sample(mix_ratios["critical"])
        if xs3 is not None:
            xs_list.append(xs3)
            ys_list.append(ys3)
        
        if len(xs_list) == 0:
            return None, None
        
        # Concatenate all
        xs_replay = torch.cat(xs_list, dim=0)
        ys_replay = torch.cat(ys_list, dim=0)
        
        return xs_replay, ys_replay
    
    def total_size(self) -> int:
        """Total samples stored across all tiers"""
        return self.tier1.size() + self.tier2.size() + self.tier3.size()
    
    def stats(self) -> dict:
        """Memory statistics"""
        return {
            "tier1_recent": self.tier1.size(),
            "tier2_compressed": self.tier2.size(),
            "tier3_critical": self.tier3.size(),
            "total": self.total_size(),
            "compression_trained": self.tier2.compression_trained,
        }

