"""
Utility functions for MECP-GAP

This module provides:
- Data generation utilities (Voronoi topology, Gravity model)
- Metric computation (Edge Cut, Load Balance)
- Visualization utilities
"""

from .visualization import (
    visualize_results,
    visualize_training_progress,
    visualize_comparison,
    compute_partition_metrics,
    print_partition_report,
    DEFAULT_COLORS
)

from .utils import (
    # Data Generation
    generate_synthetic_data,
    generate_synthetic_data_with_masses,
    coords_to_graph,
    
    # Metrics
    compute_metrics,
    compute_metrics_vectorized,
    compute_extended_metrics,
    compute_modularity,
    evaluate_partition,
    
    # Utilities
    normalize_weights,
    add_noise_to_weights,
    save_graph_data,
    load_graph_data,
    
    # Post-processing
    kl_refinement,
)

__all__ = [
    # Visualization
    'visualize_results',
    'visualize_training_progress',
    'visualize_comparison',
    'compute_partition_metrics',
    'print_partition_report',
    'DEFAULT_COLORS',
    
    # Data Generation
    'generate_synthetic_data',
    'generate_synthetic_data_with_masses',
    'coords_to_graph',
    
    # Metrics
    'compute_metrics',
    'compute_metrics_vectorized',
    'compute_extended_metrics',
    'compute_modularity',
    'evaluate_partition',
    
    # Utilities
    'normalize_weights',
    'add_noise_to_weights',
    'save_graph_data',
    'load_graph_data',
    
    # Post-processing
    'kl_refinement',
]
