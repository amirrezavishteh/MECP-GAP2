"""
Dataset Builder for MECP-GAP

Builds PyTorch Geometric compatible datasets from generated network graphs.
Creates multiple graph instances for training, validation, and testing.
"""

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import json
from tqdm import tqdm

from .graph_generator import CellTowerGraphGenerator, GraphConfig


@dataclass
class DatasetConfig:
    """Configuration for dataset building"""
    # Number of graph instances to generate
    num_train_graphs: int = 800
    num_val_graphs: int = 100
    num_test_graphs: int = 100
    
    # Graph generation parameters
    num_nodes: int = 200
    area_size: float = 10.0
    gamma: float = 1.5
    weight_scale: float = 100.0
    
    # Partitioning parameters
    num_partitions: int = 4  # Number of partitions (k)
    
    # Data paths
    output_dir: str = 'data/processed'
    
    # Random seed
    seed: Optional[int] = None


class MECPDatasetBuilder:
    """
    Builds datasets for MECP (Mobility-aware Edge Computing Partitioning).
    
    Creates multiple graph instances with:
    - Node features (X): Normalized coordinates
    - Edge index: Graph topology
    - Edge weights: Mobility traffic
    - Ground truth partitions (optional, from baselines)
    
    Attributes:
        config: DatasetConfig instance
        train_data: List of PyTorch Geometric Data objects for training
        val_data: List of Data objects for validation
        test_data: List of Data objects for testing
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize the dataset builder.
        
        Args:
            config: DatasetConfig instance
        """
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
        
        self.train_data: List[Data] = []
        self.val_data: List[Data] = []
        self.test_data: List[Data] = []
    
    def build(self) -> Tuple[List[Data], List[Data], List[Data]]:
        """
        Build the complete dataset.
        
        Generates train, validation, and test graph instances.
        
        Returns:
            Tuple of (train_data, val_data, test_data) lists
        """
        print("Building MECP Dataset...")
        
        # Generate training data
        print(f"\nGenerating {self.config.num_train_graphs} training graphs...")
        self.train_data = self._generate_graphs(self.config.num_train_graphs)
        
        # Generate validation data
        print(f"\nGenerating {self.config.num_val_graphs} validation graphs...")
        self.val_data = self._generate_graphs(self.config.num_val_graphs)
        
        # Generate test data
        print(f"\nGenerating {self.config.num_test_graphs} test graphs...")
        self.test_data = self._generate_graphs(self.config.num_test_graphs)
        
        print(f"\nDataset built successfully!")
        print(f"  Train: {len(self.train_data)} graphs")
        print(f"  Val:   {len(self.val_data)} graphs")
        print(f"  Test:  {len(self.test_data)} graphs")
        
        return self.train_data, self.val_data, self.test_data
    
    def _generate_graphs(self, num_graphs: int) -> List[Data]:
        """
        Generate a list of graph instances.
        
        Args:
            num_graphs: Number of graphs to generate
            
        Returns:
            List of PyTorch Geometric Data objects
        """
        data_list = []
        
        for i in tqdm(range(num_graphs), desc="Generating graphs"):
            data = self._generate_single_graph()
            data_list.append(data)
        
        return data_list
    
    def _generate_single_graph(self) -> Data:
        """
        Generate a single graph instance as PyTorch Geometric Data object.
        
        Returns:
            Data object with node features, edge index, and edge weights
        """
        # Create graph generator with random seed for variation
        graph_config = GraphConfig(
            num_nodes=self.config.num_nodes,
            area_size=self.config.area_size,
            gamma=self.config.gamma,
            weight_scale=self.config.weight_scale,
            mode='synthetic'
        )
        
        generator = CellTowerGraphGenerator(graph_config)
        coords, graph, W = generator.generate()
        
        # Get node features (normalized coordinates)
        x = generator.get_normalized_features()
        x = torch.tensor(x, dtype=torch.float32)
        
        # Get edge index
        edge_index = generator.get_edge_index()
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        # Get edge weights
        edge_weights = generator.get_edge_weights()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
        
        # Create full weight matrix as tensor
        weight_matrix = torch.tensor(W, dtype=torch.float32)
        
        # Create adjacency matrix
        adjacency = generator.get_adjacency_matrix()
        adjacency = torch.tensor(adjacency, dtype=torch.float32)
        
        # Store raw coordinates for visualization
        pos = torch.tensor(coords, dtype=torch.float32)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,                          # Node features (num_nodes, 2)
            edge_index=edge_index,        # Edge connectivity (2, num_edges)
            edge_attr=edge_weights,       # Edge weights (num_edges,)
            weight_matrix=weight_matrix,  # Full weight matrix (num_nodes, num_nodes)
            adjacency=adjacency,          # Adjacency matrix (num_nodes, num_nodes)
            pos=pos,                      # Raw coordinates for visualization
            num_nodes=self.config.num_nodes,
            num_partitions=self.config.num_partitions
        )
        
        return data
    
    def add_baseline_labels(
        self, 
        baseline_method: str = 'metis',
        data_list: Optional[List[Data]] = None
    ) -> List[Data]:
        """
        Add ground truth partition labels from baseline methods.
        
        This can be used for supervised pretraining or evaluation.
        
        Args:
            baseline_method: Method to use ('metis', 'louvain', 'spectral')
            data_list: List of Data objects to add labels to (defaults to all)
            
        Returns:
            List of Data objects with added labels
        """
        # Import baseline partitioners
        try:
            import metis
            from community import community_louvain
        except ImportError as e:
            print(f"Warning: Could not import baseline methods: {e}")
            print("Install with: pip install metis python-louvain")
            return data_list or (self.train_data + self.val_data + self.test_data)
        
        import networkx as nx
        
        if data_list is None:
            data_list = self.train_data + self.val_data + self.test_data
        
        print(f"Adding {baseline_method} labels to {len(data_list)} graphs...")
        
        for data in tqdm(data_list, desc=f"Computing {baseline_method} partitions"):
            # Reconstruct networkx graph from edge_index
            G = nx.Graph()
            G.add_nodes_from(range(data.num_nodes))
            
            edge_index = data.edge_index.numpy()
            edge_weights = data.edge_attr.numpy()
            
            for i in range(edge_index.shape[1] // 2):  # Only add each edge once
                u, v = edge_index[0, i], edge_index[1, i]
                w = edge_weights[i]
                G.add_edge(u, v, weight=w)
            
            # Compute partition using baseline method
            if baseline_method == 'metis':
                # METIS requires weighted graph
                _, labels = metis.part_graph(G, self.config.num_partitions)
            elif baseline_method == 'louvain':
                partition = community_louvain.best_partition(G)
                labels = [partition[i] for i in range(data.num_nodes)]
            else:
                raise ValueError(f"Unknown baseline method: {baseline_method}")
            
            # Add labels to data
            data.y = torch.tensor(labels, dtype=torch.long)
        
        return data_list
    
    def save(self, output_dir: Optional[str] = None) -> None:
        """
        Save the dataset to disk.
        
        Args:
            output_dir: Directory to save to (defaults to config.output_dir)
        """
        output_dir = output_dir or self.config.output_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each split
        torch.save(self.train_data, output_path / 'train_data.pt')
        torch.save(self.val_data, output_path / 'val_data.pt')
        torch.save(self.test_data, output_path / 'test_data.pt')
        
        # Save config
        config_dict = {
            'num_train_graphs': self.config.num_train_graphs,
            'num_val_graphs': self.config.num_val_graphs,
            'num_test_graphs': self.config.num_test_graphs,
            'num_nodes': self.config.num_nodes,
            'area_size': self.config.area_size,
            'gamma': self.config.gamma,
            'weight_scale': self.config.weight_scale,
            'num_partitions': self.config.num_partitions
        }
        with open(output_path / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Dataset saved to: {output_path}")
    
    @classmethod
    def load(cls, input_dir: str) -> 'MECPDatasetBuilder':
        """
        Load a saved dataset.
        
        Args:
            input_dir: Directory containing saved dataset
            
        Returns:
            MECPDatasetBuilder instance with loaded data
        """
        input_path = Path(input_dir)
        
        # Load config
        with open(input_path / 'config.json', 'r') as f:
            config_dict = json.load(f)
        
        config = DatasetConfig(**config_dict)
        builder = cls(config)
        
        # Load data
        builder.train_data = torch.load(input_path / 'train_data.pt')
        builder.val_data = torch.load(input_path / 'val_data.pt')
        builder.test_data = torch.load(input_path / 'test_data.pt')
        
        print(f"Dataset loaded from: {input_path}")
        print(f"  Train: {len(builder.train_data)} graphs")
        print(f"  Val:   {len(builder.val_data)} graphs")
        print(f"  Test:  {len(builder.test_data)} graphs")
        
        return builder


class MECPDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for MECP.
    
    Can be used directly with PyTorch Geometric DataLoaders.
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        config: Optional[DatasetConfig] = None,
        transform=None,
        pre_transform=None,
        force_reload: bool = False
    ):
        """
        Initialize MECP dataset.
        
        Args:
            root: Root directory for dataset storage
            split: One of 'train', 'val', 'test'
            config: DatasetConfig for generation (only needed if not already generated)
            transform: Optional transform to apply to each sample
            pre_transform: Optional transform to apply once during processing
            force_reload: Force regeneration of dataset
        """
        self.split = split
        self.config = config or DatasetConfig()
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[['train', 'val', 'test'].index(split)])
    
    @property
    def raw_file_names(self) -> List[str]:
        return []
    
    @property
    def processed_file_names(self) -> List[str]:
        return ['train_data.pt', 'val_data.pt', 'test_data.pt']
    
    def process(self) -> None:
        """Generate and save the dataset."""
        builder = MECPDatasetBuilder(self.config)
        train_data, val_data, test_data = builder.build()
        
        # Apply pre_transform if specified
        if self.pre_transform is not None:
            train_data = [self.pre_transform(d) for d in train_data]
            val_data = [self.pre_transform(d) for d in val_data]
            test_data = [self.pre_transform(d) for d in test_data]
        
        torch.save(train_data, self.processed_paths[0])
        torch.save(val_data, self.processed_paths[1])
        torch.save(test_data, self.processed_paths[2])


def create_data_loaders(
    train_data: List[Data],
    val_data: List[Data],
    test_data: List[Data],
    batch_size: int = 32,
    shuffle_train: bool = True
):
    """
    Create PyTorch Geometric DataLoaders.
    
    Args:
        train_data: Training data list
        val_data: Validation data list
        test_data: Test data list
        batch_size: Batch size
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch_geometric.loader import DataLoader
    
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=shuffle_train
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Example usage
    config = DatasetConfig(
        num_train_graphs=10,  # Small for demo
        num_val_graphs=2,
        num_test_graphs=2,
        num_nodes=100,
        seed=42
    )
    
    builder = MECPDatasetBuilder(config)
    train_data, val_data, test_data = builder.build()
    
    # Print sample info
    sample = train_data[0]
    print(f"\nSample graph:")
    print(f"  Node features shape: {sample.x.shape}")
    print(f"  Edge index shape: {sample.edge_index.shape}")
    print(f"  Edge weights shape: {sample.edge_attr.shape}")
    print(f"  Weight matrix shape: {sample.weight_matrix.shape}")
    
    # Save dataset
    builder.save('data/processed')
