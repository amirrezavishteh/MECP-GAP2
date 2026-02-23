"""
Greedy Baseline for Graph Partitioning - K-way Greedy Graph Growing Partitioning (KGGGP)

The KGGGP algorithm provides a localized approach to MEC server placement,
mimicking the "bubble" expansion of service areas from central server locations.

Technical Foundation: KGGGP Algorithm
======================================

1. SEED SELECTION AND INITIALIZATION:
   - Select P seed nodes where P = number of MEC servers
   - Seeds can be: random, density-aware (high traffic regions), or farthest-apart
   - Seeds serve as "nuclei" for the P partitions
   - Maintains a "frontier" of unassigned neighbors for each partition

2. ITERATIVE EXPANSION (Scoring Mechanism):
   - Best-first search strategy
   - Greedy score function:
     sc(v, i) = sum_{v' in V_i, (v,v') in E} W_vv' - v.unallocated_weight
   
   Where:
   - W_vv' = edge weight between candidate node and existing partition
   - v.unallocated_weight = connections v has with unassigned nodes

3. LOAD BALANCING:
   - Monitor total weight/traffic demand of each partition
   - Once partition reaches max capacity U, it is "closed"
   - Process repeats until all base stations are assigned

Limitations:
- Short-sighted decision-making: only considers local connectivity
- May inadvertently create boundaries that cut high-weight edges later
- No global structural awareness (unlike METIS or GNN)
- Can bisect major transit routes, causing frequent handovers

Complexity: O(E * log(N)) for priority queue based implementation

Paper Reference: Section on baseline comparisons
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List, Set
import time
from collections import defaultdict
import heapq


class GreedyPartitioner:
    """
    Greedy graph partitioner based on local decisions.
    
    Algorithm:
    1. Select k seed nodes (far apart in the graph)
    2. Grow partitions by adding nodes with strongest connections
    3. Balance partitions by moving boundary nodes
    """
    
    def __init__(self, num_partitions: int,
                 seed_method: str = 'farthest',
                 max_iterations: int = 100,
                 balance_tolerance: float = 0.1,
                 seed: Optional[int] = None):
        """
        Initialize greedy partitioner.
        
        Args:
            num_partitions: Number of partitions k
            seed_method: Method to select initial seeds ('farthest', 'random', 'degree')
            max_iterations: Max refinement iterations
            balance_tolerance: Acceptable size imbalance (0.1 = 10%)
            seed: Random seed
        """
        self.num_partitions = num_partitions
        self.seed_method = seed_method
        self.max_iterations = max_iterations
        self.balance_tolerance = balance_tolerance
        self.random_seed = seed
        
        self._assignments = None
        self._runtime = None
        
        if seed is not None:
            np.random.seed(seed)
    
    def fit(self, W_matrix: np.ndarray, coords: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Partition graph using greedy algorithm.
        
        Args:
            W_matrix: Weighted adjacency matrix (N, N)
            coords: Optional node coordinates for farthest-first seeding
            
        Returns:
            assignments: Partition assignments (N,)
        """
        start_time = time.time()
        
        num_nodes = W_matrix.shape[0]
        
        # Step 1: Select seed nodes
        seeds = self._select_seeds(W_matrix, coords)
        
        # Step 2: Initialize assignments (-1 = unassigned)
        assignments = np.full(num_nodes, -1, dtype=int)
        for i, seed in enumerate(seeds):
            assignments[seed] = i
        
        # Step 3: Grow partitions greedily
        assignments = self._grow_partitions(W_matrix, assignments, seeds)
        
        # Step 4: Balance refinement
        assignments = self._balance_partitions(W_matrix, assignments)
        
        self._assignments = assignments
        self._runtime = time.time() - start_time
        
        return assignments
    
    def _select_seeds(self, W_matrix: np.ndarray, 
                      coords: Optional[np.ndarray]) -> List[int]:
        """
        Select k seed nodes for initial partition centers.
        
        Args:
            W_matrix: Weight matrix
            coords: Optional coordinates
            
        Returns:
            seeds: List of k seed node indices
        """
        num_nodes = W_matrix.shape[0]
        
        if self.seed_method == 'farthest' and coords is not None:
            # Farthest-first selection in coordinate space
            seeds = []
            # Start with random node
            seeds.append(np.random.randint(num_nodes))
            
            for _ in range(1, self.num_partitions):
                # Find node farthest from all current seeds
                min_dists = np.full(num_nodes, np.inf)
                for seed in seeds:
                    dists = np.linalg.norm(coords - coords[seed], axis=1)
                    min_dists = np.minimum(min_dists, dists)
                
                # Select node with maximum minimum distance
                min_dists[seeds] = -np.inf  # Exclude existing seeds
                seeds.append(np.argmax(min_dists))
            
            return seeds
        
        elif self.seed_method == 'degree':
            # Select high-degree nodes spread across graph
            degrees = (W_matrix > 0).sum(axis=1)
            sorted_nodes = np.argsort(-degrees)
            
            seeds = [sorted_nodes[0]]
            used = {sorted_nodes[0]}
            
            for node in sorted_nodes:
                if len(seeds) >= self.num_partitions:
                    break
                    
                # Check if node is not a neighbor of any seed
                is_neighbor = any(W_matrix[node, seed] > 0 for seed in seeds)
                if node not in used and not is_neighbor:
                    seeds.append(node)
                    used.add(node)
            
            # Fill remaining with random if needed
            while len(seeds) < self.num_partitions:
                remaining = list(set(range(num_nodes)) - used)
                if not remaining:
                    remaining = list(range(num_nodes))
                new_seed = np.random.choice(remaining)
                seeds.append(new_seed)
                used.add(new_seed)
            
            return seeds
        
        else:  # random
            return np.random.choice(num_nodes, size=self.num_partitions, replace=False).tolist()
    
    def _grow_partitions(self, W_matrix: np.ndarray, 
                         assignments: np.ndarray,
                         seeds: List[int]) -> np.ndarray:
        """
        Grow partitions from seeds using KGGGP scoring mechanism.
        
        KGGGP Score Function:
            sc(v, i) = sum_{v' in V_i, (v,v') in E} W_vv' - v.unallocated_weight
        
        Where v.unallocated_weight = connections to unassigned nodes
        
        Uses frontier-based expansion with priority queue for efficiency.
        """
        num_nodes = len(assignments)
        unassigned = set(range(num_nodes)) - set(seeds)
        
        # Build adjacency list for efficient neighbor lookup
        adjacency = defaultdict(list)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if W_matrix[i, j] > 0 and i != j:
                    adjacency[i].append(j)
        
        # Initialize frontiers for each partition
        # frontier[p] = set of unassigned nodes adjacent to partition p
        frontiers = [set() for _ in range(self.num_partitions)]
        for p, seed in enumerate(seeds):
            for neighbor in adjacency[seed]:
                if neighbor in unassigned:
                    frontiers[p].add(neighbor)
        
        # Priority queue: (-score, node, partition) - use negative for max-heap simulation
        pq = []
        
        def compute_kgggp_score(node: int, partition: int) -> float:
            """
            Compute KGGGP score: connection to partition - connection to unassigned
            """
            partition_nodes = set(np.where(assignments == partition)[0])
            
            # Weight to partition (affinity)
            affinity = sum(W_matrix[node, v] for v in partition_nodes if W_matrix[node, v] > 0)
            
            # Weight to unallocated nodes (penalty for breaking potential clusters)
            unallocated_weight = sum(W_matrix[node, v] for v in unassigned - {node} if W_matrix[node, v] > 0)
            
            return affinity - 0.5 * unallocated_weight  # Paper suggests subtracting unallocated
        
        # Initialize priority queue with frontier nodes
        for p in range(self.num_partitions):
            for node in frontiers[p]:
                score = compute_kgggp_score(node, p)
                heapq.heappush(pq, (-score, node, p))  # Negative for max-heap
        
        # Track which nodes are in the queue to avoid duplicates
        in_queue = {(node, p) for p in range(self.num_partitions) for node in frontiers[p]}
        
        while unassigned and pq:
            # Get node with highest score
            neg_score, node, partition = heapq.heappop(pq)
            
            # Skip if already assigned
            if node not in unassigned:
                continue
            
            # Assign node to partition
            assignments[node] = partition
            unassigned.remove(node)
            
            # Update frontiers and add new candidates
            for neighbor in adjacency[node]:
                if neighbor in unassigned:
                    # Add to frontier of this partition
                    frontiers[partition].add(neighbor)
                    
                    # Compute score and add to priority queue
                    score = compute_kgggp_score(neighbor, partition)
                    if (neighbor, partition) not in in_queue:
                        heapq.heappush(pq, (-score, neighbor, partition))
                        in_queue.add((neighbor, partition))
        
        # Handle any remaining disconnected nodes
        if unassigned:
            for node in list(unassigned):
                # Assign to smallest partition
                sizes = [np.sum(assignments == p) for p in range(self.num_partitions)]
                assignments[node] = np.argmin(sizes)
        
        return assignments

    def _grow_partitions_original(self, W_matrix: np.ndarray, 
                         assignments: np.ndarray,
                         seeds: List[int]) -> np.ndarray:
        """
        Original simple grow partitions (kept for reference).
        
        Uses a best-first approach: always add the node with the strongest
        connection to an existing partition.
        """
        num_nodes = len(assignments)
        unassigned = set(range(num_nodes)) - set(seeds)
        
        # Priority queue: (weight, node, partition)
        # We'll use a simple approach with iteration
        while unassigned:
            best_node = None
            best_partition = -1
            best_weight = -np.inf
            
            for node in unassigned:
                # Find strongest connection to any partition
                for p in range(self.num_partitions):
                    # Sum of weights to nodes in partition p
                    partition_nodes = np.where(assignments == p)[0]
                    if len(partition_nodes) > 0:
                        weight = W_matrix[node, partition_nodes].sum()
                        if weight > best_weight:
                            best_weight = weight
                            best_node = node
                            best_partition = p
            
            if best_node is not None:
                assignments[best_node] = best_partition
                unassigned.remove(best_node)
            else:
                # Handle disconnected nodes
                node = unassigned.pop()
                # Assign to smallest partition
                sizes = [np.sum(assignments == p) for p in range(self.num_partitions)]
                assignments[node] = np.argmin(sizes)
        
        return assignments
    
    def _balance_partitions(self, W_matrix: np.ndarray, 
                            assignments: np.ndarray) -> np.ndarray:
        """
        Refine partitions to improve balance.
        
        Moves boundary nodes from large to small partitions.
        """
        num_nodes = len(assignments)
        ideal_size = num_nodes / self.num_partitions
        
        for iteration in range(self.max_iterations):
            # Check sizes
            sizes = np.array([np.sum(assignments == p) for p in range(self.num_partitions)])
            
            # Check if balanced
            max_imbalance = np.max(np.abs(sizes - ideal_size) / ideal_size)
            if max_imbalance <= self.balance_tolerance:
                break
            
            # Find largest and smallest partitions
            largest_p = np.argmax(sizes)
            smallest_p = np.argmin(sizes)
            
            # Find boundary nodes in largest partition
            largest_nodes = np.where(assignments == largest_p)[0]
            
            best_node = None
            best_gain = -np.inf
            
            for node in largest_nodes:
                # Check if this node has connections to smallest partition
                smallest_nodes = np.where(assignments == smallest_p)[0]
                weight_to_smallest = W_matrix[node, smallest_nodes].sum()
                weight_to_largest = W_matrix[node, largest_nodes[largest_nodes != node]].sum()
                
                # Moving this node should decrease cut
                gain = weight_to_smallest - weight_to_largest
                if weight_to_smallest > 0 and gain > best_gain:
                    best_gain = gain
                    best_node = node
            
            if best_node is not None:
                assignments[best_node] = smallest_p
            else:
                # No good candidate, pick any boundary node
                for node in largest_nodes:
                    smallest_nodes = np.where(assignments == smallest_p)[0]
                    if W_matrix[node, smallest_nodes].sum() > 0:
                        assignments[node] = smallest_p
                        break
                else:
                    # No boundary node, just move any node
                    assignments[largest_nodes[0]] = smallest_p
        
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
            'num_partitions': self.num_partitions
        }


def run_greedy(
    W_matrix: np.ndarray,
    num_partitions: int,
    coords: Optional[np.ndarray] = None,
    seed_method: str = 'farthest',
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to run greedy partitioning.
    
    Args:
        W_matrix: Weighted adjacency matrix (N, N)
        num_partitions: Number of partitions k
        coords: Optional node coordinates
        seed_method: Method for initial seed selection
        seed: Random seed
        
    Returns:
        assignments: Partition assignments (N,)
        results: Dictionary with additional results
    """
    partitioner = GreedyPartitioner(
        num_partitions=num_partitions,
        seed_method=seed_method,
        seed=seed
    )
    
    assignments = partitioner.fit(W_matrix, coords)
    results = partitioner.get_results()
    
    return assignments, results
