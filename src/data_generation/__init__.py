"""Data generation modules for MECP-GAP

This module provides tools for generating synthetic 5G network data:

1. Graph Generation (graph_generator.py):
   - CellTowerGraphGenerator: Generates network topology with Voronoi diagrams
   - GraphConfig: Configuration for graph generation
   
2. Mobility Traces (mobility_generator.py):
   - MobilityTraceGenerator: Generates user mobility patterns (coordinate-based)
   - GraphMobilityGenerator: Graph-based mobility for handover simulation
   - MobilityConfig, GraphMobilityConfig: Configuration classes

3. Dataset Building (dataset_builder.py):
   - MECPDatasetBuilder: Creates PyTorch Geometric datasets
   - MECPDataset: InMemoryDataset for direct use with DataLoaders
   - DatasetConfig: Configuration for dataset building
   
4. Shanghai Data (shanghai_loader.py):
   - ShanghaiDataLoader: Load real cell tower data from OpenCellID
"""

from .mobility_generator import (
    MobilityTraceGenerator, 
    MobilityConfig,
    GraphMobilityGenerator,
    GraphMobilityConfig,
    generate_graph_mobility_traces
)
from .graph_generator import (
    CellTowerGraphGenerator, 
    GraphConfig,
    generate_network_graph
)
from .dataset_builder import (
    MECPDatasetBuilder, 
    MECPDataset,
    DatasetConfig,
    create_data_loaders
)

__all__ = [
    # Mobility generation
    'MobilityTraceGenerator',
    'MobilityConfig',
    
    # Graph generation
    'CellTowerGraphGenerator',
    'GraphConfig',
    'generate_network_graph',
    
    # Dataset building
    'MECPDatasetBuilder',
    'MECPDataset',
    'DatasetConfig',
    'create_data_loaders',
]
