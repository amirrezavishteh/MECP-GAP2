"""
Baseline Algorithms for Graph Partitioning

This module implements baseline algorithms for comparison with MECP-GAP:
- METIS: State-of-the-art graph partitioner
- Greedy: Simple heuristic-based partitioning
- Random: Random assignment baseline

Paper Reference: Section V.C - Comparison Methods
"""

from .metis_baseline import MetisPartitioner, run_metis
from .greedy_baseline import GreedyPartitioner, run_greedy
from .random_baseline import RandomPartitioner, run_random

__all__ = [
    'MetisPartitioner', 'run_metis',
    'GreedyPartitioner', 'run_greedy',
    'RandomPartitioner', 'run_random'
]
