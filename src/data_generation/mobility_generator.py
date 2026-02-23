"""
Mobility Trace Generator for MECP-GAP
Generates synthetic user mobility patterns based on various mobility models

This module supports two types of mobility:
1. Coordinate-based traces (continuous 2D positions)
2. Graph-based traces (discrete node sequences for handover simulation)

Paper Reference: Section V.D - Handover scenario simulation
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class MobilityConfig:
    """Configuration for mobility trace generation"""
    num_users: int = 100
    num_timesteps: int = 50
    area_size: Tuple[float, float] = (1000.0, 1000.0)  # meters
    avg_speed: float = 5.0  # m/s (walking speed)
    speed_std: float = 2.0
    pause_prob: float = 0.2
    direction_change_prob: float = 0.1
    waypoint_model: bool = True
    seed: Optional[int] = None


@dataclass  
class GraphMobilityConfig:
    """Configuration for graph-based mobility trace generation"""
    num_users: int = 1000  # Paper uses 1000 users
    num_timesteps: int = 100  # Number of time steps
    stay_probability: float = 0.3  # Probability of staying at current node
    seed: Optional[int] = None


class GraphMobilityGenerator:
    """
    Generate graph-based mobility traces for handover simulation.
    
    Users move on a graph (cell tower network) with transition probabilities
    proportional to edge weights (traffic flow from Gravity model).
    
    Paper Reference: Section V.D
    - Start users at random nodes
    - At each timestep, move to neighbor based on probability P(u,v) = W(u,v) / sum(W(u,:))
    - Record sequence of base stations
    """
    
    def __init__(self, config: GraphMobilityConfig):
        """
        Initialize graph mobility generator.
        
        Args:
            config: GraphMobilityConfig with parameters
        """
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)
    
    def generate_traces(self, W_matrix: np.ndarray) -> np.ndarray:
        """
        Generate mobility traces on the graph.
        
        Users perform random walks on the graph with transition probabilities
        based on edge weights. Higher weights = more likely transitions.
        
        Args:
            W_matrix: Weighted adjacency matrix (N, N)
            
        Returns:
            traces: Node sequences (num_users, num_timesteps) - each entry is a node index
        """
        num_nodes = W_matrix.shape[0]
        num_users = self.config.num_users
        num_timesteps = self.config.num_timesteps
        
        # Initialize trace array
        traces = np.zeros((num_users, num_timesteps), dtype=int)
        
        # Start users at random nodes
        traces[:, 0] = np.random.randint(0, num_nodes, size=num_users)
        
        # Precompute transition probabilities for each node
        # P(u -> v) = W(u,v) / sum(W(u,:))
        transition_probs = W_matrix.copy()
        
        # Add self-loop weight to model staying in place
        for i in range(num_nodes):
            row_sum = transition_probs[i, :].sum()
            if row_sum > 0:
                # Add stay probability
                stay_weight = row_sum * self.config.stay_probability / (1 - self.config.stay_probability)
                transition_probs[i, i] += stay_weight
        
        # Normalize to get probabilities
        row_sums = transition_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_probs = transition_probs / row_sums
        
        # Simulate user movement
        for t in range(1, num_timesteps):
            for u in range(num_users):
                current_node = traces[u, t-1]
                probs = transition_probs[current_node, :]
                
                # If isolated node (no neighbors), stay in place
                if probs.sum() == 0:
                    traces[u, t] = current_node
                else:
                    # Sample next node based on transition probabilities
                    traces[u, t] = np.random.choice(num_nodes, p=probs)
        
        return traces
    
    def get_handover_events(self, traces: np.ndarray) -> Dict:
        """
        Extract handover events from traces.
        
        A handover occurs when a user moves from one node to a different node.
        
        Args:
            traces: Node sequences (num_users, num_timesteps)
            
        Returns:
            Dictionary with handover statistics
        """
        num_users, num_timesteps = traces.shape
        
        # Count handovers per user per timestep
        handovers_per_timestep = np.zeros(num_timesteps - 1)
        total_handovers = 0
        
        # Handover matrix: H[i,j] = number of handovers from node i to node j
        num_nodes = traces.max() + 1
        handover_matrix = np.zeros((num_nodes, num_nodes))
        
        for u in range(num_users):
            for t in range(1, num_timesteps):
                from_node = traces[u, t-1]
                to_node = traces[u, t]
                
                if from_node != to_node:
                    handovers_per_timestep[t-1] += 1
                    total_handovers += 1
                    handover_matrix[from_node, to_node] += 1
        
        return {
            'total_handovers': total_handovers,
            'handovers_per_timestep': handovers_per_timestep,
            'handover_matrix': handover_matrix,
            'avg_handovers_per_user': total_handovers / num_users,
            'avg_handovers_per_timestep': handovers_per_timestep.mean()
        }


def generate_graph_mobility_traces(
    W_matrix: np.ndarray,
    num_users: int = 1000,
    num_timesteps: int = 100,
    stay_probability: float = 0.3,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to generate graph-based mobility traces.
    
    Args:
        W_matrix: Weighted adjacency matrix
        num_users: Number of users to simulate
        num_timesteps: Number of time steps
        stay_probability: Probability of staying at current node
        seed: Random seed
        
    Returns:
        traces: Node sequences (num_users, num_timesteps)
        handover_stats: Dictionary with handover statistics
    """
    config = GraphMobilityConfig(
        num_users=num_users,
        num_timesteps=num_timesteps,
        stay_probability=stay_probability,
        seed=seed
    )
    
    generator = GraphMobilityGenerator(config)
    traces = generator.generate_traces(W_matrix)
    handover_stats = generator.get_handover_events(traces)
    
    return traces, handover_stats


class MobilityTraceGenerator:
    """
    Generates mobility traces for users in a 2D area.
    Supports multiple mobility models:
    - Random Waypoint Model
    - Random Walk Model
    - Gauss-Markov Model
    """
    
    def __init__(self, config: MobilityConfig):
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)
        
    def generate_random_waypoint_traces(self) -> np.ndarray:
        """
        Generate mobility traces using Random Waypoint Model.
        
        Returns:
            traces: shape (num_users, num_timesteps, 2) with (x, y) coordinates
        """
        traces = np.zeros((self.config.num_users, self.config.num_timesteps, 2))
        
        # Initialize random starting positions
        traces[:, 0, 0] = np.random.uniform(0, self.config.area_size[0], self.config.num_users)
        traces[:, 0, 1] = np.random.uniform(0, self.config.area_size[1], self.config.num_users)
        
        # For each user, generate waypoints and interpolate
        for user_idx in range(self.config.num_users):
            current_pos = traces[user_idx, 0].copy()
            current_time = 0
            
            while current_time < self.config.num_timesteps - 1:
                # Select random waypoint
                waypoint = np.array([
                    np.random.uniform(0, self.config.area_size[0]),
                    np.random.uniform(0, self.config.area_size[1])
                ])
                
                # Calculate distance and time to reach waypoint
                distance = np.linalg.norm(waypoint - current_pos)
                speed = max(0.1, np.random.normal(self.config.avg_speed, self.config.speed_std))
                time_to_reach = int(distance / speed) + 1
                
                # Interpolate positions
                steps = min(time_to_reach, self.config.num_timesteps - current_time)
                for step in range(steps):
                    t = (step + 1) / steps
                    traces[user_idx, current_time + step + 1] = (
                        current_pos * (1 - t) + waypoint * t
                    )
                
                current_pos = traces[user_idx, min(current_time + steps, self.config.num_timesteps - 1)]
                current_time += steps
                
                # Possible pause at waypoint
                if np.random.rand() < self.config.pause_prob and current_time < self.config.num_timesteps - 1:
                    pause_duration = min(5, self.config.num_timesteps - current_time - 1)
                    for p in range(pause_duration):
                        traces[user_idx, current_time + p + 1] = current_pos
                    current_time += pause_duration
        
        return traces
    
    def generate_random_walk_traces(self) -> np.ndarray:
        """
        Generate mobility traces using Random Walk Model.
        
        Returns:
            traces: shape (num_users, num_timesteps, 2) with (x, y) coordinates
        """
        traces = np.zeros((self.config.num_users, self.config.num_timesteps, 2))
        
        # Initialize random starting positions
        traces[:, 0, 0] = np.random.uniform(0, self.config.area_size[0], self.config.num_users)
        traces[:, 0, 1] = np.random.uniform(0, self.config.area_size[1], self.config.num_users)
        
        # Direction for each user (angle in radians)
        directions = np.random.uniform(0, 2 * np.pi, self.config.num_users)
        
        for t in range(1, self.config.num_timesteps):
            # Change direction with some probability
            change_dir = np.random.rand(self.config.num_users) < self.config.direction_change_prob
            directions[change_dir] = np.random.uniform(0, 2 * np.pi, change_dir.sum())
            
            # Calculate step size
            speeds = np.maximum(0.1, np.random.normal(
                self.config.avg_speed, 
                self.config.speed_std, 
                self.config.num_users
            ))
            
            # Update positions
            dx = speeds * np.cos(directions)
            dy = speeds * np.sin(directions)
            
            traces[:, t, 0] = traces[:, t-1, 0] + dx
            traces[:, t, 1] = traces[:, t-1, 1] + dy
            
            # Reflect at boundaries
            reflect_x = (traces[:, t, 0] < 0) | (traces[:, t, 0] > self.config.area_size[0])
            reflect_y = (traces[:, t, 1] < 0) | (traces[:, t, 1] > self.config.area_size[1])
            
            traces[:, t, 0] = np.clip(traces[:, t, 0], 0, self.config.area_size[0])
            traces[:, t, 1] = np.clip(traces[:, t, 1], 0, self.config.area_size[1])
            
            directions[reflect_x] = np.pi - directions[reflect_x]
            directions[reflect_y] = -directions[reflect_y]
        
        return traces
    
    def generate_gauss_markov_traces(self, alpha: float = 0.5) -> np.ndarray:
        """
        Generate mobility traces using Gauss-Markov Model.
        
        Args:
            alpha: Memory parameter (0 = random, 1 = linear motion)
            
        Returns:
            traces: shape (num_users, num_timesteps, 2) with (x, y) coordinates
        """
        traces = np.zeros((self.config.num_users, self.config.num_timesteps, 2))
        
        # Initialize random starting positions
        traces[:, 0, 0] = np.random.uniform(0, self.config.area_size[0], self.config.num_users)
        traces[:, 0, 1] = np.random.uniform(0, self.config.area_size[1], self.config.num_users)
        
        # Initialize velocity
        velocities = np.random.randn(self.config.num_users, 2) * self.config.avg_speed
        
        for t in range(1, self.config.num_timesteps):
            # Update velocity using Gauss-Markov model
            velocities = (alpha * velocities + 
                         (1 - alpha) * self.config.avg_speed * np.random.randn(self.config.num_users, 2))
            
            # Update positions
            traces[:, t] = traces[:, t-1] + velocities
            
            # Bounce at boundaries
            for dim in range(2):
                lower_bound = traces[:, t, dim] < 0
                upper_bound = traces[:, t, dim] > self.config.area_size[dim]
                
                traces[lower_bound, t, dim] = -traces[lower_bound, t, dim]
                traces[upper_bound, t, dim] = (2 * self.config.area_size[dim] - 
                                                traces[upper_bound, t, dim])
                
                velocities[lower_bound | upper_bound, dim] *= -1
        
        return traces
    
    def generate_traces(self, model: str = 'waypoint') -> np.ndarray:
        """
        Generate mobility traces using specified model.
        
        Args:
            model: 'waypoint', 'random_walk', or 'gauss_markov'
            
        Returns:
            traces: shape (num_users, num_timesteps, 2)
        """
        if model == 'waypoint':
            return self.generate_random_waypoint_traces()
        elif model == 'random_walk':
            return self.generate_random_walk_traces()
        elif model == 'gauss_markov':
            return self.generate_gauss_markov_traces()
        else:
            raise ValueError(f"Unknown mobility model: {model}")
    
    def compute_handover_events(self, traces: np.ndarray, 
                                cell_positions: np.ndarray,
                                cell_radius: float = 100.0) -> Dict:
        """
        Compute handover events based on mobility traces and cell tower positions.
        
        Args:
            traces: User mobility traces (num_users, num_timesteps, 2)
            cell_positions: Cell tower positions (num_cells, 2)
            cell_radius: Coverage radius of cell towers
            
        Returns:
            Dictionary with handover statistics
        """
        num_users, num_timesteps, _ = traces.shape
        num_cells = len(cell_positions)
        
        # Determine which cell each user is connected to at each timestep
        cell_assignments = np.zeros((num_users, num_timesteps), dtype=int)
        
        for t in range(num_timesteps):
            for u in range(num_users):
                distances = np.linalg.norm(cell_positions - traces[u, t], axis=1)
                cell_assignments[u, t] = np.argmin(distances)
        
        # Count handovers
        handovers = np.sum(cell_assignments[:, 1:] != cell_assignments[:, :-1])
        
        # Compute handover matrix (from cell i to cell j)
        handover_matrix = np.zeros((num_cells, num_cells))
        for u in range(num_users):
            for t in range(1, num_timesteps):
                if cell_assignments[u, t] != cell_assignments[u, t-1]:
                    from_cell = cell_assignments[u, t-1]
                    to_cell = cell_assignments[u, t]
                    handover_matrix[from_cell, to_cell] += 1
        
        return {
            'total_handovers': handovers,
            'handover_matrix': handover_matrix,
            'cell_assignments': cell_assignments,
            'avg_handovers_per_user': handovers / num_users
        }
    
    def save_traces(self, traces: np.ndarray, filepath: str):
        """Save mobility traces to file"""
        np.save(filepath, traces)
        print(f"Saved mobility traces to {filepath}")
        print(f"Shape: {traces.shape}")
    
    def load_traces(self, filepath: str) -> np.ndarray:
        """Load mobility traces from file"""
        traces = np.load(filepath)
        print(f"Loaded mobility traces from {filepath}")
        print(f"Shape: {traces.shape}")
        return traces


def visualize_traces(traces: np.ndarray, num_users: int = 5, save_path: Optional[str] = None):
    """
    Visualize mobility traces (requires matplotlib)
    
    Args:
        traces: Mobility traces (num_users, num_timesteps, 2)
        num_users: Number of users to visualize
        save_path: Path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        num_users = min(num_users, traces.shape[0])
        for i in range(num_users):
            ax.plot(traces[i, :, 0], traces[i, :, 1], '-o', 
                   alpha=0.6, markersize=2, label=f'User {i+1}')
            ax.plot(traces[i, 0, 0], traces[i, 0, 1], 'go', markersize=10)  # Start
            ax.plot(traces[i, -1, 0], traces[i, -1, 1], 'ro', markersize=10)  # End
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('User Mobility Traces')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("Matplotlib not available for visualization")
