"""
Pretraining module for AI-Frajerka.
Handles autoregressive pretraining on raw text and dialogue corpora.
"""

from .config import PretrainingConfig
from .dataset import PretrainingDataset
from .trainer import Pretrainer

__all__ = ["PretrainingConfig", "PretrainingDataset", "Pretrainer"]