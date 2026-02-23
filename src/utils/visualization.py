"""
Visualization Utilities for MECP-GAP

This module provides functions for visualizing:
- Graph partitioning results (colored node assignments)
- Training progress (loss curves)
- Edge weights as line thickness (thick = high traffic)

Paper Reference: Figure 5 - Visual comparison of partitioning results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


# Default color palette for partitions (up to 10 distinct colors)
DEFAULT_COLORS = [
    '#FF9999',  # Light red
    '#66B2FF',  # Light blue
    '#99FF99',  # Light green
    '#FFCC99',  # Light orange
    '#D1C4E9',  # Light purple
    '#F0F4C3',  # Light yellow-green
    '#FFAB91',  # Light deep orange
    '#B2EBF2',  # Light cyan
    '#F8BBD9',  # Light pink
    '#C5E1A5',  # Light lime
]


def visualize_results(
    coords: np.ndarray,
    W_matrix: np.ndarray,
    assignments: np.ndarray,
    num_partitions: int,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    node_size: int = 80,
    edge_alpha: float = 0.3,
    show_legend: bool = True,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize the graph partitioning results.
    
    Creates a plot showing:
    - Nodes colored by partition assignment
    - Edges with thickness proportional to traffic weight
    - Thick edges = high traffic (should NOT be cut)
    - Thin edges = low traffic (good candidates for cutting)
    
    Args:
        coords: Node coordinates (N, 2)
        W_matrix: Mobility weight matrix (N, N)
        assignments: Partition assignments for each node (N,)
        num_partitions: Number of partitions
        title: Custom title (auto-generated if None)
        figsize: Figure size tuple
        node_size: Size of nodes in plot
        edge_alpha: Transparency of edges
        show_legend: Whether to show partition legend
        save_path: Path to save figure (None = don't save)
        show: Whether to display the plot
        
    Returns:
        fig: matplotlib Figure object
    """
    num_nodes = len(coords)
    
    # Build NetworkX graph for visualization
    G_vis = nx.Graph()
    G_vis.add_nodes_from(range(num_nodes))
    
    # Add edges only where traffic exists (W > 0)
    # Store weights for edge thickness
    rows, cols = np.nonzero(W_matrix)
    for u, v in zip(rows, cols):
        if u < v:  # Add each undirected edge once
            G_vis.add_edge(u, v, weight=W_matrix[u, v])
    
    # Create position dictionary for networkx
    pos = {i: coords[i] for i in range(num_nodes)}
    
    # Assign colors to nodes based on partition
    node_colors = [DEFAULT_COLORS[assignments[i] % len(DEFAULT_COLORS)] 
                   for i in range(num_nodes)]
    
    # Calculate edge widths based on traffic weight
    edges = list(G_vis.edges())
    if len(edges) > 0:
        weights = [G_vis[u][v]['weight'] for u, v in edges]
        max_w = max(weights) if weights else 1.0
        # Normalize weights to range [0.1, 2.5] for visibility
        edge_widths = [0.1 + (w / max_w) * 2.5 for w in weights]
    else:
        edge_widths = []
    
    # Identify cut edges (edges between different partitions)
    cut_edges = []
    internal_edges = []
    cut_edge_widths = []
    internal_edge_widths = []
    
    for i, (u, v) in enumerate(edges):
        if assignments[u] != assignments[v]:
            cut_edges.append((u, v))
            cut_edge_widths.append(edge_widths[i])
        else:
            internal_edges.append((u, v))
            internal_edge_widths.append(edge_widths[i])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw internal edges (same partition) in gray
    if internal_edges:
        nx.draw_networkx_edges(
            G_vis, pos, 
            edgelist=internal_edges,
            width=internal_edge_widths,
            edge_color='gray',
            alpha=edge_alpha,
            ax=ax
        )
    
    # Draw cut edges in red (to highlight partition boundaries)
    if cut_edges:
        nx.draw_networkx_edges(
            G_vis, pos,
            edgelist=cut_edges,
            width=cut_edge_widths,
            edge_color='red',
            alpha=0.5,
            style='dashed',
            ax=ax
        )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G_vis, pos,
        node_color=node_colors,
        node_size=node_size,
        edgecolors='black',
        linewidths=0.5,
        ax=ax
    )
    
    # Add legend
    if show_legend:
        legend_elements = []
        for p in range(num_partitions):
            count = np.sum(assignments == p)
            color = DEFAULT_COLORS[p % len(DEFAULT_COLORS)]
            element = plt.scatter([], [], c=color, s=100, 
                                  label=f'Partition {p} ({count} nodes)',
                                  edgecolors='black', linewidths=0.5)
            legend_elements.append(element)
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Set title
    if title is None:
        cut_weight = sum(W_matrix[u, v] for u, v in cut_edges) if cut_edges else 0
        total_weight = W_matrix.sum() / 2  # Divide by 2 for undirected
        title = (f"MECP-GAP Result: {num_partitions} Partitions\n"
                f"Cut Edges: {len(cut_edges)} | Cut Weight: {cut_weight:.1f} "
                f"({100*cut_weight/total_weight:.1f}% of total)")
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def visualize_training_progress(
    history: List[Dict[str, Any]],
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize training progress over epochs.
    
    Creates subplots showing:
    - Total loss over time
    - Cut loss vs balance loss
    - Partition size evolution
    
    Args:
        history: List of training history dictionaries
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        fig: matplotlib Figure object
    """
    epochs = [h['epoch'] for h in history]
    total_loss = [h['total_loss'] for h in history]
    cut_loss = [h.get('weighted_cut_loss', 0) for h in history]
    balance_loss = [h.get('weighted_balance_loss', 0) for h in history]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Total Loss
    ax1 = axes[0]
    ax1.plot(epochs, total_loss, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss Components
    ax2 = axes[1]
    ax2.plot(epochs, cut_loss, 'r-', label='Cut Loss', linewidth=2)
    ax2.plot(epochs, balance_loss, 'g-', label='Balance Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Partition Sizes
    ax3 = axes[2]
    if 'partition_sizes' in history[0]:
        partition_sizes = np.array([h['partition_sizes'] for h in history])
        num_partitions = partition_sizes.shape[1]
        for p in range(num_partitions):
            ax3.plot(epochs, partition_sizes[:, p], 
                    label=f'Partition {p}', linewidth=2,
                    color=DEFAULT_COLORS[p % len(DEFAULT_COLORS)])
        
        # Add ideal line
        ideal_size = partition_sizes[0].sum() / num_partitions
        ax3.axhline(y=ideal_size, color='black', linestyle='--', 
                   label=f'Ideal ({ideal_size:.0f})', alpha=0.7)
        
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Partition Size')
    ax3.set_title('Partition Size Balance')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def visualize_comparison(
    coords: np.ndarray,
    W_matrix: np.ndarray,
    assignments_list: List[np.ndarray],
    method_names: List[str],
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Compare multiple partitioning methods side by side.
    
    Useful for comparing MECP-GAP against baseline methods
    like METIS, Louvain, or random partitioning.
    
    Args:
        coords: Node coordinates (N, 2)
        W_matrix: Mobility weight matrix (N, N)
        assignments_list: List of partition assignments from different methods
        method_names: Names of the methods
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        fig: matplotlib Figure object
    """
    num_methods = len(assignments_list)
    fig, axes = plt.subplots(1, num_methods, figsize=figsize)
    
    if num_methods == 1:
        axes = [axes]
    
    num_nodes = len(coords)
    
    # Build graph once
    G_vis = nx.Graph()
    G_vis.add_nodes_from(range(num_nodes))
    rows, cols = np.nonzero(W_matrix)
    for u, v in zip(rows, cols):
        if u < v:
            G_vis.add_edge(u, v, weight=W_matrix[u, v])
    
    pos = {i: coords[i] for i in range(num_nodes)}
    
    # Calculate edge widths
    edges = list(G_vis.edges())
    if len(edges) > 0:
        weights = [G_vis[u][v]['weight'] for u, v in edges]
        max_w = max(weights)
        edge_widths = [0.1 + (w / max_w) * 2.0 for w in weights]
    else:
        edge_widths = []
    
    for idx, (assignments, method_name, ax) in enumerate(zip(assignments_list, method_names, axes)):
        num_partitions = len(np.unique(assignments))
        
        # Node colors
        node_colors = [DEFAULT_COLORS[assignments[i] % len(DEFAULT_COLORS)] 
                      for i in range(num_nodes)]
        
        # Calculate cut weight
        cut_weight = sum(W_matrix[u, v] for u, v in edges 
                        if assignments[u] != assignments[v])
        total_weight = W_matrix.sum() / 2
        
        # Draw
        nx.draw_networkx_edges(G_vis, pos, width=edge_widths,
                              edge_color='gray', alpha=0.3, ax=ax)
        nx.draw_networkx_nodes(G_vis, pos, node_color=node_colors,
                              node_size=50, edgecolors='black', 
                              linewidths=0.5, ax=ax)
        
        # Calculate partition balance (standard deviation)
        sizes = [np.sum(assignments == p) for p in range(num_partitions)]
        balance_std = np.std(sizes)
        
        ax.set_title(f'{method_name}\nCut: {cut_weight:.1f} | Balance σ: {balance_std:.1f}',
                    fontsize=11)
        ax.axis('off')
    
    plt.suptitle('Partitioning Method Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def compute_partition_metrics(
    W_matrix: np.ndarray,
    assignments: np.ndarray,
    num_partitions: int = None
) -> Dict[str, float]:
    """
    Compute quantitative metrics for partition quality.
    
    Args:
        W_matrix: Mobility weight matrix (N, N)
        assignments: Partition assignments (N,)
        num_partitions: Number of intended partitions (if None, inferred from assignments)
        
    Returns:
        Dictionary with metrics:
        - edge_cut: Total weight of cut edges
        - edge_cut_ratio: Cut weight / Total weight
        - balance_std: Std dev of partition sizes
        - balance_max_diff: Max deviation from ideal size
        - num_cut_edges: Number of edges cut
    """
    num_nodes = len(assignments)
    if num_partitions is None:
        num_partitions = len(np.unique(assignments))
    
    # Edge cut calculation
    rows, cols = np.nonzero(W_matrix)
    cut_weight = 0.0
    num_cut_edges = 0
    total_edges = 0
    
    for u, v in zip(rows, cols):
        if u < v:  # Count each edge once
            total_edges += 1
            if assignments[u] != assignments[v]:
                cut_weight += W_matrix[u, v]
                num_cut_edges += 1
    
    total_weight = W_matrix.sum() / 2
    
    # Balance calculation
    sizes = np.array([np.sum(assignments == p) for p in range(num_partitions)])
    ideal_size = num_nodes / num_partitions
    
    return {
        'edge_cut': cut_weight,
        'edge_cut_ratio': cut_weight / total_weight if total_weight > 0 else 0,
        'num_cut_edges': num_cut_edges,
        'total_edges': total_edges,
        'balance_std': float(np.std(sizes)),
        'balance_max_diff': float(np.max(np.abs(sizes - ideal_size))),
        'partition_sizes': sizes.tolist()
    }


def print_partition_report(
    W_matrix: np.ndarray,
    assignments: np.ndarray,
    method_name: str = "MECP-GAP",
    num_partitions: int = None
):
    """
    Print a formatted report of partition quality metrics.
    
    Args:
        W_matrix: Mobility weight matrix
        assignments: Partition assignments
        method_name: Name of the method for display
        num_partitions: Number of intended partitions
    """
    metrics = compute_partition_metrics(W_matrix, assignments, num_partitions=num_partitions)
    
    print(f"\n{'='*50}")
    print(f"Partition Quality Report: {method_name}")
    print(f"{'='*50}")
    print(f"Number of nodes: {len(assignments)}")
    print(f"Number of partitions: {len(np.unique(assignments))}")
    print(f"\nEdge Cut Metrics:")
    print(f"  - Cut Weight: {metrics['edge_cut']:.2f}")
    print(f"  - Cut Ratio: {metrics['edge_cut_ratio']*100:.2f}%")
    print(f"  - Cut Edges: {metrics['num_cut_edges']}/{metrics['total_edges']}")
    print(f"\nBalance Metrics:")
    print(f"  - Partition Sizes: {metrics['partition_sizes']}")
    print(f"  - Size Std Dev: {metrics['balance_std']:.2f}")
    print(f"  - Max Deviation from Ideal: {metrics['balance_max_diff']:.2f}")
    print(f"{'='*50}\n")
