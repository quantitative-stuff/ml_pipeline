"""
Pipeline Module
==============

Modular pipeline components for stock return prediction.
"""

from .data_processor import DataProcessor
from .factor_processor import FactorProcessor
from .model_trainer import ModelTrainer

__all__ = ['DataProcessor', 'FactorProcessor', 'ModelTrainer'] 