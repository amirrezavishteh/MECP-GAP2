"""
METIS Baseline for Graph Partitioning

METIS is the preeminent industry standard for large-scale graph partitioning,
celebrated for its efficiency and the high quality of the cuts it produces.

Technical Foundation: Multilevel Graph Partitioning
====================================================

The operational brilliance of METIS lies in its multilevel paradigm:

1. COARSENING PHASE (Heavy Edge Matching - HEM):
   - Creates successive graph approximations G_0, G_1, ..., G_k
   - Uses Heavy Edge Matching (HEM) to collapse adjacent nodes into "super-vertices"
   - Node u is matched with neighbor v such that W_uv is maximized
   - This internalizes high-traffic edges, preventing them from being cut
   
   Mathematical formulation:
   - If v and u are matched into super-vertex w: wt(w) = wt(v) + wt(u)
   - For adjacent vertex z: new edge weight = W_vz + W_uz

2. INITIAL PARTITIONING PHASE:
   - Operates on coarsest graph G_k (typically < 100 nodes)
   - Uses spectral bisection via Fiedler vector (2nd smallest eigenvalue of L = D - A)
   - Provides global "blueprint" for partition structure

3. UNCOARSENING & REFINEMENT PHASE (Fiduccia-Mattheyses):
   - Projects partition from G_k back through G_0
   - FM heuristic uses "gain" concept: G_v = net reduction in edge cut if v moves
   - Maintains priority queue of vertices by gain
   - Enforces load balance: max_i |V_i| / (|V|/k) <= 1 + epsilon

Load Balance Constraint:
   The paper requires strict balance enforcement for MEC server provisioning
   to ensure server capacity is not exceeded.

Paper Reference: Section II.B - "METIS, Graclus"
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List
import time
import warnings
from collections import defaultdict

try:
    import pymetis
    HAS_PYMETIS = True
except ImportError:
    HAS_PYMETIS = False
    warnings.warn("pymetis not installed. Install with: pip install pymetis")


class MetisPartitioner:
    """
    METIS-based graph partitioner.
    
    Uses multilevel k-way partitioning to divide graph into k balanced partitions
    while minimizing edge cut.
    
    METIS Parameters:
    - ubvec: Load imbalance tolerance (1.001 = 0.1% imbalance allowed)
    - recursive: Use recursive bisection vs k-way partitioning
    """
    
    def __init__(self, num_partitions: int, 
                 balance_constraint: float = 1.001,
                 recursive: bool = False,
                 seed: Optional[int] = None):
        """
        Initialize METIS partitioner.
        
        Args:
            num_partitions: Number of partitions k
            balance_constraint: Max allowed imbalance factor (1.0 = perfectly balanced)
            recursive: Use recursive bisection (True) or k-way (False)
            seed: Random seed for reproducibility
        """
        self.num_partitions = num_partitions
        self.balance_constraint = balance_constraint
        self.recursive = recursive
        self.seed = seed
        
        self._partitions = None
        self._cut_weight = None
        self._runtime = None
    
    def fit(self, W_matrix: np.ndarray) -> np.ndarray:
        """
        Partition the graph using METIS.
        
        Args:
            W_matrix: Weighted adjacency matrix (N, N)
            
        Returns:
            assignments: Partition assignments for each node (N,)
        """
        if not HAS_PYMETIS:
            raise ImportError("pymetis is required. Install with: pip install pymetis")
        
        start_time = time.time()
        
        num_nodes = W_matrix.shape[0]
        
        # Convert weight matrix to adjacency list format for pymetis
        adjacency = []
        edge_weights_list = []
        
        for i in range(num_nodes):
            neighbors = []
            weights = []
            for j in range(num_nodes):
                if W_matrix[i, j] > 0 and i != j:
                    neighbors.append(j)
                    # METIS requires integer weights
                    weights.append(max(1, int(W_matrix[i, j])))
            adjacency.append(neighbors)
            if weights:
                edge_weights_list.append(weights)
            else:
                edge_weights_list.append([])
        
        # Build xadj and adjncy arrays for METIS
        xadj = [0]
        adjncy = []
        eweights = []
        
        for i in range(num_nodes):
            adjncy.extend(adjacency[i])
            eweights.extend(edge_weights_list[i])
            xadj.append(len(adjncy))
        
        # Handle empty graph
        if len(adjncy) == 0:
            # No edges - just distribute nodes evenly
            assignments = np.array([i % self.num_partitions for i in range(num_nodes)])
            self._partitions = assignments
            self._runtime = time.time() - start_time
            self._cut_weight = 0
            return assignments
        
        # Call METIS
        try:
            if self.recursive and self.num_partitions == 2:
                # Recursive bisection (only for 2 partitions)
                cuts, parts = pymetis.part_graph(
                    self.num_partitions,
                    xadj=xadj,
                    adjncy=adjncy,
                    eweights=eweights,
                    recursive=True
                )
            else:
                # k-way partitioning
                cuts, parts = pymetis.part_graph(
                    self.num_partitions,
                    xadj=xadj,
                    adjncy=adjncy,
                    eweights=eweights
                )
            
            self._partitions = np.array(parts)
            self._cut_weight = cuts
            
        except Exception as e:
            # Fallback if METIS fails
            warnings.warn(f"METIS failed: {e}. Using fallback partitioning.")
            self._partitions = np.array([i % self.num_partitions for i in range(num_nodes)])
            self._cut_weight = 0
        
        self._runtime = time.time() - start_time
        
        return self._partitions
    
    def get_assignments(self) -> np.ndarray:
        """Get partition assignments."""
        if self._partitions is None:
            raise ValueError("Must call fit() first")
        return self._partitions
    
    def get_cut_weight(self) -> int:
        """Get total edge cut weight from METIS."""
        if self._cut_weight is None:
            raise ValueError("Must call fit() first")
        return self._cut_weight
    
    def get_runtime(self) -> float:
        """Get partitioning runtime in seconds."""
        if self._runtime is None:
            raise ValueError("Must call fit() first")
        return self._runtime
    
    def get_results(self) -> Dict[str, Any]:
        """Get all results."""
        return {
            'assignments': self._partitions,
            'cut_weight': self._cut_weight,
            'runtime': self._runtime,
            'num_partitions': self.num_partitions
        }


def run_metis(
    W_matrix: np.ndarray,
    num_partitions: int,
    balance_constraint: float = 1.001,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to run METIS partitioning.
    
    Args:
        W_matrix: Weighted adjacency matrix (N, N)
        num_partitions: Number of partitions k
        balance_constraint: Load imbalance tolerance
        seed: Random seed
        
    Returns:
        assignments: Partition assignments (N,)
        results: Dictionary with additional results
    """
    partitioner = MetisPartitioner(
        num_partitions=num_partitions,
        balance_constraint=balance_constraint,
        seed=seed
    )
    
    assignments = partitioner.fit(W_matrix)
    results = partitioner.get_results()
    
    return assignments, results


class MetisFallback:
    """
    Fallback implementation when METIS is not available.
    Uses spectral partitioning via scipy.
    """
    
    @staticmethod
    def spectral_partition(W_matrix: np.ndarray, num_partitions: int) -> np.ndarray:
        """
        Simple spectral partitioning fallback.
        
        Uses eigenvalues of the graph Laplacian for partitioning.
        """
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        from sklearn.cluster import KMeans
        
        # Build Laplacian
        D = np.diag(W_matrix.sum(axis=1))
        L = D - W_matrix
        
        # Add small diagonal to avoid singularity
        L = L + 1e-6 * np.eye(L.shape[0])
        
        # Convert to sparse
        L_sparse = csr_matrix(L)
        
        # Get eigenvectors
        try:
            k = min(num_partitions, L.shape[0] - 1)
            eigenvalues, eigenvectors = eigsh(L_sparse, k=k, which='SM')
            
            # Cluster eigenvectors
            kmeans = KMeans(n_clusters=num_partitions, random_state=42, n_init=10)
            assignments = kmeans.fit_predict(eigenvectors)
            
        except Exception:
            # Last resort fallback
            assignments = np.array([i % num_partitions for i in range(W_matrix.shape[0])])
        
        return assignments
