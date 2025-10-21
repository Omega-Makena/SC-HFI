"""
Specialized Experts
Focus on system-level coordination and optimization
"""

from .peer_selection import PeerSelectionExpert
from .meta_adaptation import MetaAdaptationExpert
from .memory_consolidation import MemoryConsolidationExpert

__all__ = [
    'PeerSelectionExpert',
    'MetaAdaptationExpert',
    'MemoryConsolidationExpert',
]

