"""
Reproducibility Module - P0 Critical Fix
Ensures all randomness is controlled for reproducible experiments
"""

import torch
import numpy as np
import random
import logging


def set_global_seed(seed: int = 42):
    """
    Set random seed for all libraries to ensure reproducibility
    
    Critical for research: Without this, experiments cannot be reproduced!
    
    Args:
        seed: Random seed value
    """
    logger = logging.getLogger("Reproducibility")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # PyTorch backends (deterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"✓ Global random seed set to {seed} (reproducible mode enabled)")
    
    return seed


def get_deterministic_config():
    """
    Get configuration for fully deterministic training
    
    Returns:
        Dictionary with deterministic training settings
    """
    return {
        "seed": 42,
        "deterministic": True,
        "benchmark": False,
        "worker_init_fn": lambda worker_id: np.random.seed(42 + worker_id),
    }

