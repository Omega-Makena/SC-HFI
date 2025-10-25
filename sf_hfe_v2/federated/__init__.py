"""
Federated Learning Module
Server-side components (Developer with ZERO data)
Includes P2P gossip protocol for decentralized communication
"""

from .global_memory import GlobalMemory
from .meta_learning import OnlineMAMLEngine
from .server import SFHFEServer
from .gossip import P2PGossipManager

__all__ = [
'GlobalMemory',
'OnlineMAMLEngine',
'SFHFEServer',
'P2PGossipManager',
]

