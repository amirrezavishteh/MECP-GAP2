"""
Training module for MECP-GAP

This module provides training utilities for the MECP-GAP model.
Contains training loop, data preparation, and evaluation utilities.
"""

from .trainer import (
    MECPTrainer,
    TrainingConfig,
    prepare_graph_data,
    training_step,
    train_mecp_gap,
)

from .training import (
    train_model,
    train_model_with_history,
    evaluate_model,
    save_model,
    load_model,
    quick_train,
    prepare_tensors,
)

__all__ = [
    # Trainer class interface
    'MECPTrainer',
    'TrainingConfig',
    'prepare_graph_data',
    'training_step',
    'train_mecp_gap',
    
    # Functional interface (simplified)
    'train_model',
    'train_model_with_history',
    'evaluate_model',
    'save_model',
    'load_model',
    'quick_train',
    'prepare_tensors',
]
