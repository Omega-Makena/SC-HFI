"""
Federated Learning System Initialization
Handles reproducibility, logging setup, and system configuration
"""

import logging
import os
import random
import numpy as np
import torch
from typing import Optional

from ..config import SYSTEM_CONFIG


def setup_logging(log_level: Optional[str] = None, log_file: Optional[str] = None) -> None:
"""
Setup structured logging for the federated learning system

Args:
log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
log_file: Optional log file path
"""
# Get configuration
level = log_level or SYSTEM_CONFIG.get("log_level", "INFO")
log_format = SYSTEM_CONFIG.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_path = log_file or SYSTEM_CONFIG.get("log_file", "logs/federated_learning.log")

# Create logs directory if it doesn't exist
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# Configure logging
logging.basicConfig(
level=getattr(logging, level.upper()),
format=log_format,
handlers=[
logging.FileHandler(file_path),
logging.StreamHandler() # Console output
]
)

# Set specific loggers
loggers = ["Server", "MetaLearning", "GlobalMemory", "P2PGossip"]
for logger_name in loggers:
logger = logging.getLogger(logger_name)
logger.setLevel(getattr(logging, level.upper()))


def setup_reproducibility(seed: Optional[int] = None) -> None:
"""
Setup reproducible random seeds for deterministic behavior

Args:
seed: Random seed value
"""
seed = seed or SYSTEM_CONFIG.get("random_seed", 42)
deterministic = SYSTEM_CONFIG.get("deterministic", True)

# Set random seeds
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Enable deterministic behavior if requested
if deterministic:
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.info(f"Reproducibility setup complete with seed: {seed}")


def initialize_system(log_level: Optional[str] = None, seed: Optional[int] = None) -> None:
"""
Initialize the entire federated learning system

Args:
log_level: Logging level
seed: Random seed for reproducibility
"""
# Setup logging first
setup_logging(log_level)

# Setup reproducibility
setup_reproducibility(seed)

logging.info("Federated Learning System initialized successfully")


def get_client_id_type() -> type:
"""
Get the standardized client ID type

Returns:
Type to use for client IDs (int or str)
"""
return int # Standardize on int type


def validate_config(config_dict: dict, required_keys: list) -> bool:
"""
Validate configuration dictionary has required keys

Args:
config_dict: Configuration dictionary to validate
required_keys: List of required keys

Returns:
True if valid, False otherwise
"""
missing_keys = [key for key in required_keys if key not in config_dict]
if missing_keys:
logging.error(f"Missing required configuration keys: {missing_keys}")
return False
return True


def safe_get_config(config_dict: dict, key: str, default=None):
"""
Safely get configuration value with default

Args:
config_dict: Configuration dictionary
key: Key to retrieve
default: Default value if key not found

Returns:
Configuration value or default
"""
return config_dict.get(key, default)
