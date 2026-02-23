"""
Cell Tower Graph Generator for MECP-GAP

This module implements the data generation pipeline as described in the paper:
1. Geometry: Base station location generation (Poisson Point Process or real data)
2. Topology: Neighbor relationships using Voronoi diagrams
3. Mobility: Edge weights using a Gravity Model

Paper Reference: Section V.A - "Simulation Part"
"""

import numpy as np
from scipy.spatial import Voronoi
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
import pickle


@dataclass
class GraphConfig:
    """Configuration for graph generation"""
    num_nodes: int = 200
    area_size: float = 10.0
    gamma: float = 1.5  # Gravity model friction coefficient
    weight_scale: float = 100.0  # Scale factor for mobility weights
    mode: str = 'synthetic'  # 'synthetic' or 'real'
    real_data_path: Optional[str] = None  # Path to real coordinate data
    seed: Optional[int] = None


class CellTowerGraphGenerator:
    """
    Generates 5G/cellular network graphs with realistic topology and mobility patterns.
    
    The generation follows three distinct layers:
    1. Geometry: Where are the base stations? (Poisson Process or Real Data)
    2. Topology: Which stations are neighbors? (Voronoi Diagram)
    3. Mobility (Weights): How many users move between them? (Gravity Model)
    
    Attributes:
        config: GraphConfig instance with generation parameters
        coords: numpy array of shape (num_nodes, 2) with base station coordinates
        G: networkx Graph representing the network topology
        W: numpy array of shape (num_nodes, num_nodes) with mobility weights
        node_masses: dict mapping node index to its "mass" (proxy for population/traffic)
    """
    
    def __init__(self, config: GraphConfig):
        """
        Initialize the graph generator.
        
        Args:
            config: GraphConfig instance with generation parameters
        """
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)
        
        self.coords: Optional[np.ndarray] = None
        self.G: Optional[nx.Graph] = None
        self.W: Optional[np.ndarray] = None
        self.node_masses: Optional[Dict[int, float]] = None
        self._voronoi: Optional[Voronoi] = None
    
    def generate(self) -> Tuple[np.ndarray, nx.Graph, np.ndarray]:
        """
        Execute the complete data generation pipeline.
        
        Returns:
            Tuple containing:
                - coords: Node feature matrix X (num_nodes, 2)
                - G: NetworkX graph with topology
                - W: Weight matrix (num_nodes, num_nodes)
        """
        # Step 1: Geometry
        self.generate_locations()
        
        # Step 2: Topology
        self.build_topology()
        
        # Step 3: Mobility Weights
        self.calculate_mobility_weights()
        
        return self.coords, self.G, self.W
    
    def generate_locations(self) -> np.ndarray:
        """
        Step 1: Generate Geometry (Base Station Locations)
        
        For synthetic mode: Uses Poisson Point Process approximation (uniform random)
        For real mode: Loads coordinates from file (e.g., OpenCellID data)
        
        Paper Reference: Section V.A - Poisson Point Process for small-scale,
        OpenCellID for Shanghai large-scale dataset
        
        Returns:
            coords: numpy array of shape (num_nodes, 2)
        """
        if self.config.mode == 'synthetic':
            # Poisson Point Process approximation - uniform random in bounded area
            self.coords = np.random.uniform(
                0, self.config.area_size, 
                (self.config.num_nodes, 2)
            )
        elif self.config.mode == 'real':
            self.coords = self._load_real_coordinates()
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}. Use 'synthetic' or 'real'.")
        
        return self.coords
    
    def _load_real_coordinates(self) -> np.ndarray:
        """
        Load real base station coordinates from file.
        
        Supports CSV and JSON formats with 'x', 'y' or 'lat', 'lon' columns.
        Coordinates are normalized to the specified area_size.
        
        Returns:
            coords: numpy array of shape (num_nodes, 2)
        """
        if self.config.real_data_path is None:
            raise ValueError("real_data_path must be specified for mode='real'")
        
        path = Path(self.config.real_data_path)
        
        if path.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(path)
            
            # Support both x/y and lat/lon naming conventions
            if 'x' in df.columns and 'y' in df.columns:
                coords = df[['x', 'y']].values
            elif 'lat' in df.columns and 'lon' in df.columns:
                coords = df[['lon', 'lat']].values  # lon=x, lat=y
            elif 'latitude' in df.columns and 'longitude' in df.columns:
                coords = df[['longitude', 'latitude']].values
            else:
                raise ValueError("CSV must have 'x','y' or 'lat','lon' columns")
                
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
            coords = np.array([[p['x'], p['y']] for p in data['coordinates']])
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Sample if more nodes than requested
        if len(coords) > self.config.num_nodes:
            indices = np.random.choice(len(coords), self.config.num_nodes, replace=False)
            coords = coords[indices]
        
        # Normalize coordinates to area_size
        coords = self._normalize_coordinates(coords)
        
        return coords
    
    def _normalize_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """
        Normalize coordinates to fit within [0, area_size] x [0, area_size].
        
        Args:
            coords: Raw coordinates array
            
        Returns:
            Normalized coordinates
        """
        # Shift to origin
        coords = coords - coords.min(axis=0)
        
        # Scale to area_size while preserving aspect ratio
        max_range = coords.max()
        if max_range > 0:
            coords = coords / max_range * self.config.area_size * 0.95  # Leave margin
            coords = coords + self.config.area_size * 0.025  # Center
        
        return coords
    
    def build_topology(self) -> Tuple[nx.Graph, Dict[int, float]]:
        """
        Step 2: Build Topology using Voronoi Diagram
        
        Creates network graph where edges exist between base stations that
        share a Voronoi cell boundary. This models the reality that users
        can only hand over between adjacent cells.
        
        Paper Reference: "create a Voronoi graph... to delineate service areas"
        
        Returns:
            Tuple containing:
                - G: NetworkX graph with adjacency structure
                - node_masses: Dict mapping node index to its "mass" proxy
        """
        if self.coords is None:
            raise ValueError("Must call generate_locations() first")
        
        # Create Voronoi Diagram
        self._voronoi = Voronoi(self.coords)
        
        # Initialize Graph
        self.G = nx.Graph()
        self.G.add_nodes_from(range(len(self.coords)))
        
        # Add node coordinates as attributes
        for i, coord in enumerate(self.coords):
            self.G.nodes[i]['pos'] = coord
            self.G.nodes[i]['x'] = coord[0]
            self.G.nodes[i]['y'] = coord[1]
        
        # Determine Neighbors (Adjacency) from Voronoi ridge_points
        # ridge_points contains pairs of point indices that share a ridge (boundary)
        for (u, v) in self._voronoi.ridge_points:
            # Skip invalid indices (shouldn't happen for interior points)
            if u >= 0 and v >= 0 and u < len(self.coords) and v < len(self.coords):
                dist = np.linalg.norm(self.coords[u] - self.coords[v])
                self.G.add_edge(u, v, distance=dist)
        
        # Calculate 'Mass' for each cell (proxy for population/traffic volume)
        # Using average distance to neighbors squared as area approximation
        self.node_masses = {}
        for n in self.G.nodes():
            neighbors = list(self.G.neighbors(n))
            if not neighbors:
                # Isolated nodes get minimal mass
                self.node_masses[n] = 0.1
            else:
                # Larger average distance to neighbors implies larger coverage area
                avg_dist = np.mean([self.G[n][nbr]['distance'] for nbr in neighbors])
                self.node_masses[n] = avg_dist ** 2  # Area ~ distance^2
        
        # Store mass as node attribute
        for n, mass in self.node_masses.items():
            self.G.nodes[n]['mass'] = mass
        
        return self.G, self.node_masses
    
    def calculate_mobility_weights(
        self, 
        gamma: Optional[float] = None,
        weight_scale: Optional[float] = None
    ) -> np.ndarray:
        """
        Step 3: Calculate Mobility Weights using Gravity Model
        
        The weight W[u,v] represents the average number of users moving 
        from gNB u to gNB v. Based on the Gravity Model from sociology:
        
        W[u,v] = scale * (Mass_u * Mass_v) / Distance^gamma
        
        Paper Reference: "input location and area... to calculate number of handover users"
        
        Args:
            gamma: Friction coefficient (1.0-2.0 typical). Higher = less long-distance traffic
            weight_scale: Scaling factor for weights
            
        Returns:
            W: Weight matrix of shape (num_nodes, num_nodes)
        """
        if self.G is None or self.node_masses is None:
            raise ValueError("Must call build_topology() first")
        
        gamma = gamma if gamma is not None else self.config.gamma
        weight_scale = weight_scale if weight_scale is not None else self.config.weight_scale
        
        num_nodes = len(self.coords)
        self.W = np.zeros((num_nodes, num_nodes))
        
        for u, v in self.G.edges():
            dist = self.G[u][v]['distance']
            
            # Get masses (proxy for population/traffic generation capacity)
            mass_u = self.node_masses[u]
            mass_v = self.node_masses[v]
            
            # Avoid division by zero
            dist = max(dist, 1e-6)
            
            # Gravity Model Formula
            # Traffic ~ (Mass_u * Mass_v) / Distance^gamma
            weight = (mass_u * mass_v) / (dist ** gamma)
            
            # Scale to reasonable values
            weight = weight * weight_scale
            
            # Symmetric traffic (undirected graph)
            self.W[u, v] = weight
            self.W[v, u] = weight
            
            # Store in graph edge
            self.G[u][v]['weight'] = weight
        
        return self.W
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get the adjacency matrix A from the graph.
        
        Returns:
            A: Binary adjacency matrix of shape (num_nodes, num_nodes)
        """
        if self.G is None:
            raise ValueError("Must call build_topology() first")
        
        return nx.adjacency_matrix(self.G).toarray()
    
    def get_normalized_features(self) -> np.ndarray:
        """
        Get normalized node features X for GNN input.
        
        Normalizes coordinates to [0, 1] range.
        
        Returns:
            X: Normalized feature matrix of shape (num_nodes, 2)
        """
        if self.coords is None:
            raise ValueError("Must call generate_locations() first")
        
        X = self.coords / self.config.area_size
        return X
    
    def get_edge_index(self) -> np.ndarray:
        """
        Get edge index in PyTorch Geometric format.
        
        Returns:
            edge_index: Array of shape (2, num_edges) with source/target indices
        """
        if self.G is None:
            raise ValueError("Must call build_topology() first")
        
        edges = list(self.G.edges())
        # Add reverse edges for undirected graph
        edge_index = np.array([[u, v] for u, v in edges] + [[v, u] for u, v in edges]).T
        return edge_index
    
    def get_edge_weights(self) -> np.ndarray:
        """
        Get edge weights in PyTorch Geometric format.
        
        Returns:
            edge_weights: Array of shape (num_edges,) with mobility weights
        """
        if self.W is None:
            raise ValueError("Must call calculate_mobility_weights() first")
        
        edges = list(self.G.edges())
        # Include both directions
        weights = [self.W[u, v] for u, v in edges] + [self.W[v, u] for u, v in edges]
        return np.array(weights)
    
    def visualize(
        self, 
        save_path: Optional[str] = None,
        show_weights: bool = True,
        figsize: Tuple[int, int] = (12, 10)
    ) -> None:
        """
        Visualize the generated graph.
        
        Edge width represents mobility traffic volume.
        
        Args:
            save_path: Optional path to save the figure
            show_weights: Whether to scale edge widths by weight
            figsize: Figure size tuple
        """
        if self.G is None or self.coords is None:
            raise ValueError("Must call generate() first")
        
        pos = {i: self.coords[i] for i in range(len(self.coords))}
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Calculate edge widths based on weights
        edges = list(self.G.edges())
        if show_weights and self.W is not None:
            weights = [self.W[u][v] for u, v in edges]
            max_w = max(weights) if weights else 1
            widths = [0.5 + (w / max_w) * 3.0 for w in weights]
        else:
            widths = [1.0] * len(edges)
        
        # Draw the network
        nx.draw_networkx_nodes(
            self.G, pos, 
            node_size=30, 
            node_color='royalblue',
            alpha=0.8,
            ax=ax
        )
        nx.draw_networkx_edges(
            self.G, pos, 
            width=widths, 
            alpha=0.4, 
            edge_color='gray',
            ax=ax
        )
        
        ax.set_title(
            f"5G Network Graph ({len(self.G.nodes())} nodes, {len(self.G.edges())} edges)\n"
            f"Edge Width = Mobility Traffic"
        )
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
    
    def save(self, output_dir: str) -> None:
        """
        Save generated data to files.
        
        Args:
            output_dir: Directory to save files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save coordinates
        np.save(output_path / 'coords.npy', self.coords)
        
        # Save weight matrix
        np.save(output_path / 'weights.npy', self.W)
        
        # Save adjacency matrix
        A = self.get_adjacency_matrix()
        np.save(output_path / 'adjacency.npy', A)
        
        # Save edge index for PyTorch Geometric
        edge_index = self.get_edge_index()
        np.save(output_path / 'edge_index.npy', edge_index)
        
        # Save graph using pickle (networkx gpickle functions are deprecated)
        with open(output_path / 'graph.pkl', 'wb') as f:
            pickle.dump(self.G, f)
        
        # Save metadata
        metadata = {
            'num_nodes': len(self.coords),
            'num_edges': len(self.G.edges()),
            'area_size': self.config.area_size,
            'gamma': self.config.gamma,
            'mode': self.config.mode
        }
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Data saved to: {output_path}")
    
    @classmethod
    def load(cls, input_dir: str) -> 'CellTowerGraphGenerator':
        """
        Load previously generated data.
        
        Args:
            input_dir: Directory containing saved files
            
        Returns:
            CellTowerGraphGenerator instance with loaded data
        """
        input_path = Path(input_dir)
        
        # Load metadata
        with open(input_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Create config
        config = GraphConfig(
            num_nodes=metadata['num_nodes'],
            area_size=metadata['area_size'],
            gamma=metadata['gamma'],
            mode=metadata['mode']
        )
        
        # Create instance
        generator = cls(config)
        
        # Load data
        generator.coords = np.load(input_path / 'coords.npy')
        generator.W = np.load(input_path / 'weights.npy')
        
        # Load graph using pickle
        with open(input_path / 'graph.pkl', 'rb') as f:
            generator.G = pickle.load(f)
        
        # Reconstruct node masses from graph
        generator.node_masses = {
            n: generator.G.nodes[n].get('mass', 1.0) 
            for n in generator.G.nodes()
        }
        
        return generator


# Convenience function for quick generation
def generate_network_graph(
    num_nodes: int = 200,
    area_size: float = 10.0,
    gamma: float = 1.5,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, nx.Graph, np.ndarray]:
    """
    Quick function to generate a network graph.
    
    Args:
        num_nodes: Number of base stations
        area_size: Size of the simulation area
        gamma: Gravity model friction coefficient
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (coords, graph, weight_matrix)
    """
    config = GraphConfig(
        num_nodes=num_nodes,
        area_size=area_size,
        gamma=gamma,
        seed=seed
    )
    generator = CellTowerGraphGenerator(config)
    return generator.generate()


if __name__ == '__main__':
    # Example usage - generate and visualize a network
    config = GraphConfig(
        num_nodes=200,
        area_size=10.0,
        gamma=1.5,
        seed=42
    )
    
    generator = CellTowerGraphGenerator(config)
    coords, graph, W = generator.generate()
    
    print(f"Graph Created: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    print(f"Weight Matrix Created. Max Traffic: {np.max(W):.2f}")
    print(f"Normalized Features Shape: {generator.get_normalized_features().shape}")
    print(f"Edge Index Shape: {generator.get_edge_index().shape}")
    
    generator.visualize()
