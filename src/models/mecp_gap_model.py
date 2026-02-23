"""
MECP-GAP Model Architecture

This module implements the Graph Neural Network model for Mobility-aware
Edge Computing Partitioning (MECP-GAP).

Paper Reference:
- Section IV.B: The I-GAP (Improved Graph Partitioning) Model
- Uses GraphSAGE for weighted aggregation
- 2-layer GNN with 128-dimensional embeddings
- MLP for partition probability prediction

Key Innovation: Weighted aggregation based on mobility traffic (W matrix)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np

# Try to import both DGL and PyTorch Geometric for flexibility
try:
    import dgl
    from dgl.nn import SAGEConv as DGLSAGEConv
    HAS_DGL = True
except ImportError:
    HAS_DGL = False

try:
    from torch_geometric.nn import SAGEConv, GCNConv, GATConv
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class GraphSAGELayer(nn.Module):
    """
    GraphSAGE layer with edge weight support for weighted aggregation.
    
    The paper emphasizes that nodes should pay more attention to neighbors
    with higher mobility flow (larger weights in W matrix).
    """
    
    def __init__(self, in_feats: int, out_feats: int, 
                 aggregator: str = 'mean', 
                 bias: bool = True,
                 normalize: bool = True):
        """
        Args:
            in_feats: Input feature dimension
            out_feats: Output feature dimension
            aggregator: Aggregation type ('mean', 'pool', 'lstm')
            bias: Whether to add bias
            normalize: Whether to L2 normalize output
        """
        super().__init__()
        
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.aggregator = aggregator
        self.normalize = normalize
        
        # Linear transformation for self features
        self.linear_self = nn.Linear(in_feats, out_feats, bias=False)
        
        # Linear transformation for aggregated neighbor features
        self.linear_neigh = nn.Linear(in_feats, out_feats, bias=False)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_feats))
        else:
            self.register_parameter('bias', None)
        
        # For pooling aggregator
        if aggregator == 'pool':
            self.pool_linear = nn.Linear(in_feats, in_feats, bias=True)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier initialization"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.linear_neigh.weight, gain=gain)
        if hasattr(self, 'pool_linear'):
            nn.init.xavier_uniform_(self.pool_linear.weight, gain=gain)
            nn.init.zeros_(self.pool_linear.bias)
    
    def forward(self, x: torch.Tensor, 
                edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional edge weighting.
        
        Args:
            x: Node features (N, in_feats)
            edge_index: Edge indices (2, E)
            edge_weight: Edge weights (E,) - mobility weights
            
        Returns:
            Updated node features (N, out_feats)
        """
        row, col = edge_index  # row = source, col = target
        
        # Aggregate neighbor features with weighting
        if self.aggregator == 'mean':
            neigh_feats = self._weighted_mean_aggregation(x, row, col, edge_weight)
        elif self.aggregator == 'pool':
            neigh_feats = self._pool_aggregation(x, row, col, edge_weight)
        else:
            neigh_feats = self._weighted_mean_aggregation(x, row, col, edge_weight)
        
        # Combine self and neighbor representations
        out = self.linear_self(x) + self.linear_neigh(neigh_feats)
        
        if self.bias is not None:
            out = out + self.bias
        
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        
        return out
    
    def _weighted_mean_aggregation(self, x: torch.Tensor,
                                    row: torch.Tensor, col: torch.Tensor,
                                    edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted mean of neighbor features.
        
        This implements the key insight from the paper: edges with higher
        mobility (handover) traffic should have more influence.
        """
        num_nodes = x.size(0)
        
        if edge_weight is None:
            edge_weight = torch.ones(row.size(0), device=x.device)
        
        # Compute weighted neighbor features
        # For each target node, aggregate source node features weighted by edge weight
        weighted_feats = x[row] * edge_weight.unsqueeze(-1)
        
        # Sum aggregation
        out = torch.zeros(num_nodes, x.size(1), device=x.device)
        out.scatter_add_(0, col.unsqueeze(-1).expand_as(weighted_feats), weighted_feats)
        
        # Compute normalization (sum of weights per node)
        weight_sum = torch.zeros(num_nodes, device=x.device)
        weight_sum.scatter_add_(0, col, edge_weight)
        
        # Avoid division by zero
        weight_sum = weight_sum.clamp(min=1e-8)
        
        return out / weight_sum.unsqueeze(-1)
    
    def _pool_aggregation(self, x: torch.Tensor,
                          row: torch.Tensor, col: torch.Tensor,
                          edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        """Max-pooling aggregation with optional weighting"""
        num_nodes = x.size(0)
        
        # Apply MLP to neighbor features
        neigh_feats = F.relu(self.pool_linear(x[row]))
        
        if edge_weight is not None:
            neigh_feats = neigh_feats * edge_weight.unsqueeze(-1)
        
        # Max aggregation
        out = torch.full((num_nodes, x.size(1)), float('-inf'), device=x.device)
        out.scatter_reduce_(0, col.unsqueeze(-1).expand_as(neigh_feats), 
                           neigh_feats, reduce='amax', include_self=False)
        
        # Replace -inf with 0 for nodes with no neighbors
        out = torch.where(out == float('-inf'), torch.zeros_like(out), out)
        
        return out


class MECP_GAP_Model(nn.Module):
    """
    The main MECP-GAP model architecture.
    
    Architecture (from paper Section IV.B):
    1. Graph Embedding Module (2-layer GraphSAGE)
        - Input: Node features (coordinates) -> 128-dim embedding
        - Uses weighted aggregation based on mobility matrix W
    
    2. Graph Partitioning Module (MLP)
        - Input: 128-dim node embeddings
        - Output: Partition probability matrix X (N x P)
        - Final softmax for probability distribution
    
    Paper Parameters:
    - Hidden dimension: 128
    - Number of GNN layers: 2
    - Aggregator: mean (weighted)
    """
    
    def __init__(self, 
                 in_feats: int,
                 hidden_feats: int = 128,
                 out_feats: int = 128,
                 num_partitions: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 aggregator: str = 'mean',
                 use_batch_norm: bool = False):
        """
        Initialize MECP-GAP model.
        
        Args:
            in_feats: Input feature dimension (usually 2 for coordinates)
            hidden_feats: Hidden layer dimension (default 128 per paper)
            out_feats: Output embedding dimension (default 128 per paper)
            num_partitions: Number of partitions P (MEC servers)
            num_layers: Number of GNN layers (default 2 per paper)
            dropout: Dropout rate
            aggregator: Type of aggregation ('mean', 'pool')
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats
        self.num_partitions = num_partitions
        self.num_layers = num_layers
        self.dropout = dropout
        self.temperature = 1.0  # Temperature for softmax sharpening
        
        # ========================================
        # 1. Graph Embedding Module (GraphSAGE)
        # ========================================
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList() if use_batch_norm else None
        
        # First layer: in_feats -> hidden_feats
        self.gnn_layers.append(
            GraphSAGELayer(in_feats, hidden_feats, aggregator=aggregator)
        )
        if use_batch_norm:
            self.norms.append(nn.BatchNorm1d(hidden_feats))
        
        # Middle layers: hidden_feats -> hidden_feats
        for _ in range(num_layers - 2):
            self.gnn_layers.append(
                GraphSAGELayer(hidden_feats, hidden_feats, aggregator=aggregator)
            )
            if use_batch_norm:
                self.norms.append(nn.BatchNorm1d(hidden_feats))
        
        # Last layer: hidden_feats -> out_feats
        if num_layers > 1:
            self.gnn_layers.append(
                GraphSAGELayer(hidden_feats, out_feats, aggregator=aggregator)
            )
            if use_batch_norm:
                self.norms.append(nn.BatchNorm1d(out_feats))
        
        # ========================================
        # 2. Graph Partitioning Module (MLP)
        # ========================================
        # Paper: embedding (128) -> hidden (64) -> num_partitions (P)
        self.partition_mlp = nn.Sequential(
            nn.Linear(out_feats, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_partitions)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MLP weights"""
        for m in self.partition_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, 
                edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None,
                return_embeddings: bool = False) -> torch.Tensor:
        """
        Forward pass through MECP-GAP model.
        
        Args:
            x: Node features (N, in_feats)
            edge_index: Edge indices (2, E)
            edge_weight: Edge weights from mobility matrix W (E,)
            return_embeddings: If True, also return intermediate embeddings
            
        Returns:
            probs: Partition probability matrix X (N, num_partitions)
            embeddings: (optional) Node embeddings after GNN
        """
        h = x
        
        # Graph Embedding Module
        for i, gnn_layer in enumerate(self.gnn_layers):
            h = gnn_layer(h, edge_index, edge_weight)
            
            if self.norms is not None:
                h = self.norms[i](h)
            
            # ReLU activation for all but the last layer
            if i < len(self.gnn_layers) - 1:
                h = F.relu(h)
                if self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)
            else:
                # Final GNN layer: apply ReLU then normalize
                h = F.relu(h)
                h = F.normalize(h, p=2, dim=-1)
        
        embeddings = h
        
        # Graph Partitioning Module
        logits = self.partition_mlp(h)
        
        # Temperature-scaled softmax for probability distribution over partitions
        # Lower temperature = sharper assignments, higher = more uniform
        # Paper Equation: X_ip = probability that node i belongs to partition p
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        if return_embeddings:
            return probs, embeddings
        return probs
    
    def get_partition_assignments(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Convert soft partition probabilities to hard assignments.
        
        Args:
            probs: Partition probabilities (N, num_partitions)
            
        Returns:
            assignments: Partition indices (N,)
        """
        return torch.argmax(probs, dim=-1)


class MECP_GAP_Model_DGL(nn.Module):
    """
    Alternative implementation using DGL (Deep Graph Library).
    
    This version uses DGL's built-in SAGEConv for potentially
    better performance on large graphs.
    """
    
    def __init__(self,
                 in_feats: int,
                 hidden_feats: int = 128,
                 out_feats: int = 128,
                 num_partitions: int = 4,
                 aggregator: str = 'mean'):
        """
        Initialize DGL-based MECP-GAP model.
        """
        super().__init__()
        
        if not HAS_DGL:
            raise ImportError("DGL is not installed. Use MECP_GAP_Model instead.")
        
        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats
        self.num_partitions = num_partitions
        
        # GraphSAGE layers
        self.conv1 = DGLSAGEConv(in_feats, hidden_feats, aggregator_type=aggregator)
        self.conv2 = DGLSAGEConv(hidden_feats, out_feats, aggregator_type=aggregator)
        
        # Partition MLP
        self.mlp = nn.Sequential(
            nn.Linear(out_feats, 64),
            nn.ReLU(),
            nn.Linear(64, num_partitions)
        )
    
    def forward(self, g: 'dgl.DGLGraph', 
                inputs: torch.Tensor,
                edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass using DGL graph.
        
        Args:
            g: DGL graph
            inputs: Node features
            edge_weights: Edge weights for weighted aggregation
            
        Returns:
            Partition probabilities (N, num_partitions)
        """
        # Layer 1
        h = self.conv1(g, inputs, edge_weight=edge_weights)
        h = F.relu(h)
        
        # Layer 2
        h = self.conv2(g, h, edge_weight=edge_weights)
        h = F.relu(h)
        
        # Normalize embeddings
        h = F.normalize(h, p=2, dim=1)
        
        # Partition prediction
        logits = self.mlp(h)
        probs = F.softmax(logits, dim=1)
        
        return probs


class MECP_GAP_Model_PyG(nn.Module):
    """
    Alternative implementation using PyTorch Geometric.
    
    Provides flexibility to use different GNN architectures:
    - GraphSAGE (default, as in paper)
    - GCN
    - GAT (with attention)
    """
    
    def __init__(self,
                 in_feats: int,
                 hidden_feats: int = 128,
                 out_feats: int = 128,
                 num_partitions: int = 4,
                 gnn_type: str = 'sage'):
        """
        Initialize PyG-based MECP-GAP model.
        
        Args:
            gnn_type: Type of GNN ('sage', 'gcn', 'gat')
        """
        super().__init__()
        
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric not installed.")
        
        self.gnn_type = gnn_type
        
        # Select GNN layer type
        if gnn_type == 'sage':
            self.conv1 = SAGEConv(in_feats, hidden_feats)
            self.conv2 = SAGEConv(hidden_feats, out_feats)
        elif gnn_type == 'gcn':
            self.conv1 = GCNConv(in_feats, hidden_feats)
            self.conv2 = GCNConv(hidden_feats, out_feats)
        elif gnn_type == 'gat':
            self.conv1 = GATConv(in_feats, hidden_feats // 4, heads=4, concat=True)
            self.conv2 = GATConv(hidden_feats, out_feats, heads=1, concat=False)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Partition MLP
        self.mlp = nn.Sequential(
            nn.Linear(out_feats, 64),
            nn.ReLU(),
            nn.Linear(64, num_partitions)
        )
    
    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass using PyG conventions"""
        # Layer 1
        if self.gnn_type in ['sage', 'gcn']:
            h = self.conv1(x, edge_index, edge_weight)
        else:
            h = self.conv1(x, edge_index)
        h = F.relu(h)
        
        # Layer 2
        if self.gnn_type in ['sage', 'gcn']:
            h = self.conv2(h, edge_index, edge_weight)
        else:
            h = self.conv2(h, edge_index)
        h = F.relu(h)
        
        # Normalize
        h = F.normalize(h, p=2, dim=1)
        
        # Partition prediction
        logits = self.mlp(h)
        probs = F.softmax(logits, dim=1)
        
        return probs


def create_model(in_feats: int = 2,
                 hidden_feats: int = 128,
                 num_partitions: int = 4,
                 backend: str = 'native') -> nn.Module:
    """
    Factory function to create MECP-GAP model.
    
    Args:
        in_feats: Input feature dimension
        hidden_feats: Hidden dimension
        num_partitions: Number of partitions
        backend: 'native', 'dgl', or 'pyg'
        
    Returns:
        MECP-GAP model instance
    """
    if backend == 'native':
        return MECP_GAP_Model(
            in_feats=in_feats,
            hidden_feats=hidden_feats,
            out_feats=hidden_feats,
            num_partitions=num_partitions
        )
    elif backend == 'dgl':
        return MECP_GAP_Model_DGL(
            in_feats=in_feats,
            hidden_feats=hidden_feats,
            out_feats=hidden_feats,
            num_partitions=num_partitions
        )
    elif backend == 'pyg':
        return MECP_GAP_Model_PyG(
            in_feats=in_feats,
            hidden_feats=hidden_feats,
            out_feats=hidden_feats,
            num_partitions=num_partitions
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
