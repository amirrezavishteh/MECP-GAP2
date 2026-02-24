"""
MECP-GAP Utilities Module

This module provides two key functionalities:
1. Data Generation: Creating synthetic 5G network topology using Voronoi diagrams
   and Gravity Model for traffic simulation.
2. Metric Calculation: Computing Edge Cut and Load Balance metrics as defined
   in the paper for evaluation.

Paper Reference: Section V.A (Simulation) and Section IV.C (Metrics)
"""

import numpy as np
import networkx as nx
from scipy.spatial import Voronoi
from typing import Tuple, Dict, Optional


# =============================================================================
# Part 1: Data Generation (The Virtual City)
# =============================================================================

def generate_synthetic_data(
    num_nodes: int = 200,
    area_size: float = 10.0,
    gamma: float = 1.5,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates a 5G network using Voronoi topology and Gravity Model mobility.
    
    This function implements the data generation pipeline from the paper:
    1. Geometry: Random base station locations (Poisson Process approximation)
    2. Topology: Voronoi diagram to determine neighbor relationships
    3. Mobility: Gravity Model to simulate user traffic between cells
    
    Args:
        num_nodes: Number of base stations to generate (default: 200)
        area_size: Size of the square area in km (default: 10.0)
        gamma: Gravity model friction coefficient (default: 1.5)
        seed: Random seed for reproducibility (default: None)
        
    Returns:
        coords: (N, 2) array of base station coordinates (Features F)
        adj: (N, N) Adjacency matrix (Graph Structure)
        W: (N, N) Mobility Weight matrix (Traffic)
        
    Example:
        >>> coords, adj, W = generate_synthetic_data(num_nodes=100, seed=42)
        >>> print(f"Generated {coords.shape[0]} nodes with {np.sum(adj)/2:.0f} edges")
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 1. Geometry: Generate random base station locations (Poisson Process)
    coords = np.random.uniform(0, area_size, (num_nodes, 2))
    
    # 2. Topology: Build Voronoi Diagram to find neighbors
    # Add boundary points to handle edge cells properly
    boundary_pts = np.array([
        [-area_size, -area_size],
        [-area_size, 2 * area_size],
        [2 * area_size, -area_size],
        [2 * area_size, 2 * area_size]
    ])
    all_points = np.vstack([coords, boundary_pts])
    
    vor = Voronoi(all_points)
    adj = np.zeros((num_nodes, num_nodes))
    
    # Iterate through Voronoi ridges (boundaries between cells)
    for (p1, p2) in vor.ridge_points:
        # Only consider ridges between actual nodes (not boundary points)
        if p1 < num_nodes and p2 < num_nodes:
            adj[p1, p2] = 1
            adj[p2, p1] = 1  # Undirected graph
    
    # 3. Mobility: Gravity Model for Edge Weights (W)
    # Formula: W_ij = (Mass_i * Mass_j) / Distance^gamma
    # We approximate "Mass" as 1.0 for uniform synthetic data
    W = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj[i, j] == 1:  # Only calculate for neighbors
                dist = np.linalg.norm(coords[i] - coords[j])
                dist = max(dist, 0.01)  # Avoid division by zero
                
                # Gravity formula
                weight = (1.0 * 1.0) / (dist ** gamma)
                
                W[i, j] = weight
                W[j, i] = weight
    
    return coords, adj, W


def generate_synthetic_data_with_masses(
    num_nodes: int = 200,
    area_size: float = 10.0,
    gamma: float = 1.5,
    mass_range: Tuple[float, float] = (0.5, 2.0),
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extended version with variable node masses for more realistic traffic patterns.
    
    Nodes with higher "mass" (population, commercial activity) attract more traffic.
    
    Args:
        num_nodes: Number of base stations to generate
        area_size: Size of the square area in km
        gamma: Gravity model friction coefficient
        mass_range: (min_mass, max_mass) for random mass assignment
        seed: Random seed for reproducibility
        
    Returns:
        coords: (N, 2) array of base station coordinates
        adj: (N, N) Adjacency matrix
        W: (N, N) Mobility Weight matrix with mass-weighted traffic
        masses: (N,) array of node masses
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate base topology
    coords = np.random.uniform(0, area_size, (num_nodes, 2))
    
    # Generate random masses (representing population/activity level)
    masses = np.random.uniform(mass_range[0], mass_range[1], num_nodes)
    
    # Build Voronoi topology
    boundary_pts = np.array([
        [-area_size, -area_size],
        [-area_size, 2 * area_size],
        [2 * area_size, -area_size],
        [2 * area_size, 2 * area_size]
    ])
    all_points = np.vstack([coords, boundary_pts])
    vor = Voronoi(all_points)
    
    adj = np.zeros((num_nodes, num_nodes))
    for (p1, p2) in vor.ridge_points:
        if p1 < num_nodes and p2 < num_nodes:
            adj[p1, p2] = 1
            adj[p2, p1] = 1
    
    # Gravity model with variable masses
    W = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj[i, j] == 1:
                dist = np.linalg.norm(coords[i] - coords[j])
                dist = max(dist, 0.01)
                
                # Gravity formula with masses
                weight = (masses[i] * masses[j]) / (dist ** gamma)
                
                W[i, j] = weight
                W[j, i] = weight
    
    return coords, adj, W, masses


def coords_to_graph(coords: np.ndarray, adj: np.ndarray) -> nx.Graph:
    """
    Convert coordinate and adjacency arrays to NetworkX graph.
    
    Args:
        coords: (N, 2) array of node coordinates
        adj: (N, N) adjacency matrix
        
    Returns:
        G: NetworkX graph with position attributes
    """
    G = nx.from_numpy_array(adj)
    pos = {i: coords[i] for i in range(len(coords))}
    nx.set_node_attributes(G, pos, 'pos')
    return G


# =============================================================================
# Part 2: Metric Evaluation (The Scorecard)
# =============================================================================

def compute_metrics(
    assignments: np.ndarray,
    W: np.ndarray,
    num_partitions: int
) -> Tuple[float, float, np.ndarray]:
    """
    Calculates the exact metrics used in the paper for partition quality evaluation.
    
    This function computes:
    1. Edge Cut (Equation 7): Sum of mobility weights between different partitions
    2. Load Balance (Equation 8): Variance of partition sizes from ideal
    
    Args:
        assignments: (N,) array of cluster IDs (0 to P-1)
        W: (N, N) mobility weight matrix
        num_partitions: Number of partitions P
        
    Returns:
        edge_cut: Total edge cut value (lower is better)
        load_balance_var: Load balance variance (lower is better)
        partition_sizes: (P,) array of partition sizes
        
    Example:
        >>> edge_cut, balance_var, sizes = compute_metrics(assignments, W, num_partitions=4)
        >>> print(f"Edge Cut: {edge_cut:.2f}, Balance Variance: {balance_var:.2f}")
    """
    num_nodes = W.shape[0]
    
    # 1. Edge Cut Calculation
    # Sum of weights W_ij where i and j are in DIFFERENT partitions
    edge_cut = 0.0
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if W[i, j] > 0:  # If there is a connection
                if assignments[i] != assignments[j]:
                    edge_cut += W[i, j]
    
    # 2. Load Balance Calculation
    # Variance of partition sizes: Sum((Size_k - Ideal_Size)^2)
    partition_sizes = np.zeros(num_partitions)
    for k in range(num_partitions):
        partition_sizes[k] = np.sum(assignments == k)
    
    ideal_size = num_nodes / num_partitions
    load_balance_var = np.sum((partition_sizes - ideal_size) ** 2)
    
    return edge_cut, load_balance_var, partition_sizes


def compute_metrics_vectorized(
    assignments: np.ndarray,
    W: np.ndarray,
    num_partitions: int
) -> Tuple[float, float, np.ndarray]:
    """
    Vectorized (faster) version of metric computation for large graphs.
    
    Uses numpy operations instead of loops for better performance.
    
    Args:
        assignments: (N,) array of cluster IDs (0 to P-1)
        W: (N, N) mobility weight matrix
        num_partitions: Number of partitions P
        
    Returns:
        edge_cut: Total edge cut value
        load_balance_var: Load balance variance
        partition_sizes: (P,) array of partition sizes
    """
    num_nodes = W.shape[0]
    
    # Edge Cut: Vectorized computation
    # Create indicator matrix for different partitions
    diff_partition = assignments[:, None] != assignments[None, :]
    # Sum weights where nodes are in different partitions (upper triangle only)
    edge_cut = np.sum(np.triu(W * diff_partition))
    
    # Load Balance: Vectorized computation
    partition_sizes = np.array([np.sum(assignments == k) for k in range(num_partitions)])
    ideal_size = num_nodes / num_partitions
    load_balance_var = np.sum((partition_sizes - ideal_size) ** 2)
    
    return edge_cut, load_balance_var, partition_sizes


def compute_extended_metrics(
    assignments: np.ndarray,
    W: np.ndarray,
    num_partitions: int
) -> Dict[str, float]:
    """
    Compute extended set of metrics for comprehensive evaluation.
    
    Includes additional metrics beyond the paper's primary metrics:
    - Normalized Cut
    - Balance Ratio
    - Modularity
    
    Args:
        assignments: (N,) array of cluster IDs
        W: (N, N) mobility weight matrix
        num_partitions: Number of partitions
        
    Returns:
        Dictionary containing all metrics
    """
    edge_cut, balance_var, partition_sizes = compute_metrics(assignments, W, num_partitions)
    
    num_nodes = W.shape[0]
    total_weight = np.sum(W) / 2  # Undirected graph
    
    # Normalized Cut: cut(S, V\S) / min(vol(S), vol(V\S))
    degrees = np.sum(W, axis=1)
    volumes = np.array([np.sum(degrees[assignments == k]) for k in range(num_partitions)])
    
    normalized_cut = 0.0
    for k in range(num_partitions):
        mask_k = assignments == k
        cut_k = np.sum(W[mask_k][:, ~mask_k])
        if volumes[k] > 0:
            normalized_cut += cut_k / volumes[k]
    
    # Balance Ratio: max_size / min_size
    min_size = max(np.min(partition_sizes), 1)  # Avoid division by zero
    max_size = np.max(partition_sizes)
    balance_ratio = max_size / min_size
    
    # Modularity (graph clustering quality)
    modularity = compute_modularity(assignments, W)
    
    return {
        'edge_cut': edge_cut,
        'edge_cut_ratio': edge_cut / total_weight if total_weight > 0 else 0,
        'load_balance_var': balance_var,
        'balance_ratio': balance_ratio,
        'normalized_cut': normalized_cut,
        'modularity': modularity,
        'partition_sizes': partition_sizes.tolist(),
        'num_nodes': num_nodes,
        'num_partitions': num_partitions
    }


def compute_modularity(assignments: np.ndarray, W: np.ndarray) -> float:
    """
    Compute modularity score for the partition.
    
    Modularity measures the density of connections within partitions
    compared to random connections.
    
    Args:
        assignments: (N,) array of cluster IDs
        W: (N, N) weight matrix
        
    Returns:
        Modularity score (higher is better, max 1.0)
    """
    m = np.sum(W) / 2  # Total edge weight
    if m == 0:
        return 0.0
    
    degrees = np.sum(W, axis=1)
    num_partitions = len(np.unique(assignments))
    
    modularity = 0.0
    for k in range(num_partitions):
        mask = assignments == k
        internal_weight = np.sum(W[mask][:, mask]) / 2
        partition_degree_sum = np.sum(degrees[mask])
        modularity += internal_weight / m - (partition_degree_sum / (2 * m)) ** 2
    
    return modularity


def evaluate_partition(
    assignments: np.ndarray,
    W: np.ndarray,
    num_partitions: int,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Complete evaluation of a partition with optional printing.
    
    Args:
        assignments: Partition assignments
        W: Weight matrix
        num_partitions: Number of partitions
        verbose: Whether to print results
        
    Returns:
        Dictionary with all metrics
    """
    metrics = compute_extended_metrics(assignments, W, num_partitions)
    
    if verbose:
        print("\n" + "=" * 50)
        print("PARTITION EVALUATION RESULTS")
        print("=" * 50)
        print(f"Nodes: {metrics['num_nodes']} | Partitions: {metrics['num_partitions']}")
        print("-" * 50)
        print(f"Edge Cut:        {metrics['edge_cut']:.4f}")
        print(f"Edge Cut Ratio:  {metrics['edge_cut_ratio']:.4f}")
        print(f"Load Balance:    {metrics['load_balance_var']:.4f}")
        print(f"Balance Ratio:   {metrics['balance_ratio']:.2f}")
        print(f"Normalized Cut:  {metrics['normalized_cut']:.4f}")
        print(f"Modularity:      {metrics['modularity']:.4f}")
        print("-" * 50)
        print(f"Partition Sizes: {metrics['partition_sizes']}")
        print("=" * 50 + "\n")
    
    return metrics


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_weights(W: np.ndarray) -> np.ndarray:
    """
    Normalize weight matrix to [0, 1] range.
    
    Args:
        W: Weight matrix
        
    Returns:
        Normalized weight matrix
    """
    max_w = np.max(W)
    if max_w > 0:
        return W / max_w
    return W


def add_noise_to_weights(
    W: np.ndarray,
    noise_level: float = 0.1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add random noise to weight matrix for robustness testing.
    
    Args:
        W: Original weight matrix
        noise_level: Standard deviation of noise as fraction of mean weight
        seed: Random seed
        
    Returns:
        Noisy weight matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    W_noisy = W.copy()
    mask = W > 0
    mean_weight = np.mean(W[mask]) if np.any(mask) else 1.0
    
    noise = np.random.normal(0, noise_level * mean_weight, W.shape)
    noise = np.triu(noise, 1)
    noise = noise + noise.T  # Symmetric noise
    
    W_noisy = W + noise
    W_noisy = np.maximum(W_noisy, 0)  # Ensure non-negative weights
    W_noisy[~mask] = 0  # Keep zeros where original had zeros
    
    return W_noisy


def save_graph_data(
    filepath: str,
    coords: np.ndarray,
    adj: np.ndarray,
    W: np.ndarray,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save graph data to file.
    
    Args:
        filepath: Path to save (without extension)
        coords: Node coordinates
        adj: Adjacency matrix
        W: Weight matrix
        metadata: Optional metadata dictionary
    """
    import os
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    np.save(f"{filepath}_coords.npy", coords)
    np.save(f"{filepath}_adj.npy", adj)
    np.save(f"{filepath}_weights.npy", W)
    
    if metadata:
        import json
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


def load_graph_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict]]:
    """
    Load graph data from file.
    
    Args:
        filepath: Path to load from (without extension)
        
    Returns:
        coords, adj, W, metadata
    """
    import json
    
    coords = np.load(f"{filepath}_coords.npy")
    adj = np.load(f"{filepath}_adj.npy")
    W = np.load(f"{filepath}_weights.npy")
    
    metadata = None
    try:
        with open(f"{filepath}_metadata.json", 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        pass
    
    return coords, adj, W, metadata


# =============================================================================
# Part 3: Post-Processing Refinement (Kernighan-Lin style)
# =============================================================================

def kl_refinement(
    assignments: np.ndarray,
    W: np.ndarray,
    num_partitions: int,
    max_iterations: int = 100,
    balance_tolerance: float = 0.1,
    verbose: bool = False
) -> np.ndarray:
    """
    Kernighan-Lin (KL) style greedy refinement of GNN partition assignments.
    
    After the GNN produces soft-probability assignments, this post-processing
    step iteratively swaps boundary nodes between partitions to reduce the
    total edge cut while respecting a balance constraint.
    
    Algorithm:
    1. Identify boundary nodes (nodes with neighbors in a different partition).
    2. For each boundary node, compute the 'gain' of moving it to each
       neighboring partition: gain = (external edges to that partition) - 
       (internal edges lost from current partition).
    3. Execute the swap with the highest gain, subject to balance constraints.
    4. Repeat until no beneficial swap exists.
    
    This is similar to what METIS does internally, and typically reduces
    edge cut by 1-2% without any retraining.
    
    Args:
        assignments: (N,) array of partition IDs (0 to P-1) from GNN
        W: (N, N) mobility weight matrix
        num_partitions: Number of partitions P
        max_iterations: Maximum number of refinement passes
        balance_tolerance: Maximum allowed deviation from ideal size as a
                          fraction (e.g., 0.1 = 10% of ideal size)
        verbose: Whether to print progress
        
    Returns:
        refined_assignments: (N,) array of improved partition IDs
    """
    N = W.shape[0]
    refined = assignments.copy()
    ideal_size = N / num_partitions
    max_deviation = balance_tolerance * ideal_size
    
    # Pre-compute partition sizes
    partition_sizes = np.array([np.sum(refined == k) for k in range(num_partitions)])
    
    # Compute initial edge cut for tracking
    if verbose:
        init_cut = _compute_edge_cut_fast(refined, W)
        print(f"KL Refinement: initial edge cut = {init_cut:.4f}")
    
    total_swaps = 0
    for iteration in range(max_iterations):
        improved = False
        
        # Identify boundary nodes: nodes that have at least one neighbor in a different partition
        boundary_nodes = _get_boundary_nodes(refined, W, N)
        
        # Shuffle to avoid bias toward low-index nodes
        np.random.shuffle(boundary_nodes)
        
        for node in boundary_nodes:
            current_part = refined[node]
            
            # Compute gain for moving this node to each neighboring partition
            best_gain = 0.0
            best_target = -1
            
            # Get neighboring partitions for this node
            neighbors = np.where(W[node] > 0)[0]
            neighbor_parts = set(refined[neighbors]) - {current_part}
            
            for target_part in neighbor_parts:
                # Check balance constraint: target partition should not become too large
                # and source partition should not become too small
                if partition_sizes[target_part] + 1 > ideal_size + max_deviation:
                    continue
                if partition_sizes[current_part] - 1 < ideal_size - max_deviation:
                    continue
                
                # Compute gain = (weight from node to target_part) - (weight from node to current_part)
                # Moving the node to target_part removes edges to current_part and adds edges to target_part
                gain = 0.0
                for nb in neighbors:
                    w = W[node, nb]
                    if refined[nb] == target_part:
                        gain += w  # These edges become internal (no longer cut)
                    elif refined[nb] == current_part:
                        gain -= w  # These edges become cut
                
                if gain > best_gain:
                    best_gain = gain
                    best_target = target_part
            
            # Execute the best swap if beneficial
            if best_target >= 0 and best_gain > 1e-10:
                refined[node] = best_target
                partition_sizes[current_part] -= 1
                partition_sizes[best_target] += 1
                improved = True
                total_swaps += 1
        
        if not improved:
            if verbose:
                print(f"KL Refinement: converged after {iteration + 1} iterations, {total_swaps} total swaps")
            break
    else:
        if verbose:
            print(f"KL Refinement: reached max iterations ({max_iterations}), {total_swaps} total swaps")
    
    if verbose:
        final_cut = _compute_edge_cut_fast(refined, W)
        reduction = init_cut - final_cut
        pct = (reduction / init_cut * 100) if init_cut > 0 else 0.0
        print(f"KL Refinement: final edge cut = {final_cut:.4f} "
              f"(reduction: {reduction:.4f}, {pct:.2f}%)")
    
    return refined


def _get_boundary_nodes(assignments: np.ndarray, W: np.ndarray, N: int) -> np.ndarray:
    """
    Identify nodes that sit on partition boundaries.
    
    A boundary node has at least one neighbor in a different partition.
    
    Args:
        assignments: (N,) partition IDs
        W: (N, N) weight matrix
        N: number of nodes
        
    Returns:
        Array of boundary node indices
    """
    boundary = []
    for i in range(N):
        neighbors = np.where(W[i] > 0)[0]
        if len(neighbors) > 0 and np.any(assignments[neighbors] != assignments[i]):
            boundary.append(i)
    return np.array(boundary, dtype=np.int64)


def _compute_edge_cut_fast(assignments: np.ndarray, W: np.ndarray) -> float:
    """Fast vectorized edge cut computation."""
    diff = assignments[:, None] != assignments[None, :]
    return float(np.sum(np.triu(W * diff)))
