"""
Random Baseline for Graph Partitioning

Implements random assignment as a lower-bound baseline.
Provides both uniform random and balanced random variants.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import time


class RandomPartitioner:
    """
    Random graph partitioner (baseline).
    
    Provides two modes:
    - uniform: Each node randomly assigned to a partition
    - balanced: Ensures equal partition sizes by round-robin shuffle
    """
    
    def __init__(self, num_partitions: int,
                 balanced: bool = True,
                 seed: Optional[int] = None):
        """
        Initialize random partitioner.
        
        Args:
            num_partitions: Number of partitions k
            balanced: If True, ensure equal partition sizes
            seed: Random seed for reproducibility
        """
        self.num_partitions = num_partitions
        self.balanced = balanced
        self.seed = seed
        
        self._assignments = None
        self._runtime = None
        
        if seed is not None:
            np.random.seed(seed)
    
    def fit(self, W_matrix: np.ndarray) -> np.ndarray:
        """
        Randomly partition the graph.
        
        Args:
            W_matrix: Weighted adjacency matrix (N, N) - ignored for random
            
        Returns:
            assignments: Partition assignments (N,)
        """
        start_time = time.time()
        
        num_nodes = W_matrix.shape[0]
        
        if self.balanced:
            # Create balanced assignment
            # Each partition gets floor(N/k) nodes, remainder distributed
            base_size = num_nodes // self.num_partitions
            remainder = num_nodes % self.num_partitions
            
            assignments = []
            for p in range(self.num_partitions):
                size = base_size + (1 if p < remainder else 0)
                assignments.extend([p] * size)
            
            # Shuffle to randomize which nodes get which partition
            assignments = np.array(assignments)
            np.random.shuffle(assignments)
        else:
            # Pure uniform random
            assignments = np.random.randint(0, self.num_partitions, size=num_nodes)
        
        self._assignments = assignments
        self._runtime = time.time() - start_time
        
        return assignments
    
    def get_assignments(self) -> np.ndarray:
        """Get partition assignments."""
        if self._assignments is None:
            raise ValueError("Must call fit() first")
        return self._assignments
    
    def get_runtime(self) -> float:
        """Get runtime in seconds."""
        if self._runtime is None:
            raise ValueError("Must call fit() first")
        return self._runtime
    
    def get_results(self) -> Dict[str, Any]:
        """Get all results."""
        return {
            'assignments': self._assignments,
            'runtime': self._runtime,
            'num_partitions': self.num_partitions,
            'balanced': self.balanced
        }


def run_random(
    W_matrix: np.ndarray,
    num_partitions: int,
    balanced: bool = True,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to run random partitioning.
    
    Args:
        W_matrix: Weighted adjacency matrix (N, N)
        num_partitions: Number of partitions k
        balanced: Whether to balance partition sizes
        seed: Random seed
        
    Returns:
        assignments: Partition assignments (N,)
        results: Dictionary with additional results
    """
    partitioner = RandomPartitioner(
        num_partitions=num_partitions,
        balanced=balanced,
        seed=seed
    )
    
    assignments = partitioner.fit(W_matrix)
    results = partitioner.get_results()
    
    return assignments, results


def run_random_multiple(
    W_matrix: np.ndarray,
    num_partitions: int,
    num_trials: int = 10,
    balanced: bool = True,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run multiple random partitionings and return the best one.
    
    Best is determined by lowest edge cut.
    
    Args:
        W_matrix: Weighted adjacency matrix
        num_partitions: Number of partitions
        num_trials: Number of random trials
        balanced: Whether to balance partition sizes
        seed: Random seed
        
    Returns:
        best_assignments: Best partition assignment found
        results: Dictionary with results from all trials
    """
    if seed is not None:
        np.random.seed(seed)
    
    best_assignments = None
    best_cut = np.inf
    all_cuts = []
    
    start_time = time.time()
    
    for trial in range(num_trials):
        partitioner = RandomPartitioner(
            num_partitions=num_partitions,
            balanced=balanced
        )
        assignments = partitioner.fit(W_matrix)
        
        # Calculate edge cut
        cut = 0
        num_nodes = len(assignments)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if W_matrix[i, j] > 0 and assignments[i] != assignments[j]:
                    cut += W_matrix[i, j]
        
        all_cuts.append(cut)
        
        if cut < best_cut:
            best_cut = cut
            best_assignments = assignments.copy()
    
    total_time = time.time() - start_time
    
    results = {
        'assignments': best_assignments,
        'runtime': total_time,
        'num_partitions': num_partitions,
        'num_trials': num_trials,
        'best_cut': best_cut,
        'avg_cut': np.mean(all_cuts),
        'std_cut': np.std(all_cuts)
    }
    
    return best_assignments, results
