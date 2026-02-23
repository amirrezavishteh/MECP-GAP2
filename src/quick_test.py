"""Quick test script for MECP-GAP training pipeline."""

import numpy as np
import torch

# Test imports
print('Testing imports...')
from training.trainer import MECPTrainer, TrainingConfig, train_mecp_gap
from utils.visualization import visualize_results, compute_partition_metrics
from data_generation.graph_generator import CellTowerGraphGenerator, GraphConfig

print('All imports successful!')

# Generate small test data
print('\nGenerating test data...')
config = GraphConfig(num_nodes=50, area_size=10.0, seed=42)
generator = CellTowerGraphGenerator(config)
coords, G, W_matrix = generator.generate()
print(f'Generated: {len(coords)} nodes, {np.count_nonzero(W_matrix)//2} edges')

# Quick training test
print('\nRunning quick training test (20 epochs)...')
assignments, results = train_mecp_gap(
    coords, W_matrix,
    num_partitions=4,
    num_epochs=20,
    verbose=True
)

# Compute metrics
metrics = compute_partition_metrics(W_matrix, assignments)
print(f'\nFinal metrics:')
print(f'  Edge cut ratio: {metrics["edge_cut_ratio"]*100:.2f}%')
print(f'  Partition sizes: {metrics["partition_sizes"]}')
print(f'\nTest completed successfully!')
