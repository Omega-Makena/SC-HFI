"""
Data Streaming and Validation Module
Generates synthetic data streams with concept drift and validates user data
"""

from .stream import ConceptDriftStream, MultiClientStreamGenerator
from .validation import DataValidator, DataEntryProcessor, quick_validate

__all__ = ['ConceptDriftStream', 'MultiClientStreamGenerator', 'DataValidator', 'DataEntryProcessor', 'quick_validate']

