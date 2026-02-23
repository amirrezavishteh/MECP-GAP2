"""
MECP-GAP Trainer Module

This module implements the training loop for the MECP-GAP model as described
in the paper. It handles:
- Data preparation and tensor conversion
- Training loop with monitoring
- Model evaluation and checkpoint saving

Paper Reference: Section V - Experiments
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from tqdm import tqdm

# Import model and loss
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.mecp_gap_model import MECP_GAP_Model
from models.loss_functions import MECP_Loss


@dataclass
class TrainingConfig:
    """Configuration for training the MECP-GAP model.
    
    Default values are based on the paper's hyperparameters (Section V.B)
    """
    # Model hyperparameters
    in_feats: int = -1  # Auto-detect from data (-1 = N for weight_row features)
    hidden_feats: int = 128  # Paper: 128-dimensional embeddings
    out_feats: int = 128
    num_partitions: int = 4  # Number of MEC servers
    num_layers: int = 2  # Paper: 2-layer GNN
    dropout: float = 0.0
    aggregator: str = 'mean'
    feature_type: str = 'weight_row'  # 'weight_row' (paper) or 'coords'
    
    # Loss hyperparameters (from paper)
    alpha: float = -1.0  # Edge cut weight (-1 = auto-compute to match paper's Eq. 9 normalization)
    beta: float = 1.0  # Load balance weight
    gamma: float = 0.0  # Entropy regularization (0 = disabled per paper)
    
    # Training hyperparameters
    learning_rate: float = 0.01
    num_epochs: int = 200
    weight_decay: float = 0.0
    
    # Monitoring
    log_interval: int = 50  # Print every N epochs
    save_interval: int = 100  # Save checkpoint every N epochs
    
    # Paths
    checkpoint_dir: Optional[str] = None
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_graph_data(
    coords: np.ndarray,
    W_matrix: np.ndarray,
    device: str = 'cpu',
    feature_type: str = 'weight_row'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert numpy arrays to PyTorch tensors for training.
    
    This function prepares the data in the format expected by the model:
    - Node features (weight matrix row OR coordinates)
    - Edge index (connectivity)
    - Edge weights (mobility traffic)
    - Weight matrix (for loss computation)
    
    The paper states: 'It takes input features with a dimensionality of 200'
    for 200 nodes, indicating the weight matrix row W[i,:] is used as features.
    This gives each node a unique connectivity-based representation.
    
    Args:
        coords: Node coordinates of shape (N, 2)
        W_matrix: Mobility weight matrix of shape (N, N)
        device: Target device ('cpu' or 'cuda')
        feature_type: 'weight_row' (paper default, dim=N) or 'coords' (dim=2)
        
    Returns:
        features: Node features tensor (N, feat_dim)
        edge_index: Edge indices tensor (2, E)
        edge_weight: Edge weights tensor (E,)
        W_tensor: Weight matrix tensor (N, N)
    """
    if feature_type == 'weight_row':
        # Paper approach: use row-normalized W[i,:] as node features
        # Each node's feature = its connectivity pattern (dim = N)
        row_sums = W_matrix.sum(axis=1, keepdims=True) + 1e-8
        features_np = W_matrix / row_sums  # Row-normalized
        features = torch.tensor(features_np, dtype=torch.float32, device=device)
    else:
        # Fallback: use coordinates as features (dim = 2)
        features = torch.tensor(coords, dtype=torch.float32, device=device)
    
    # Convert weight matrix
    W_tensor = torch.tensor(W_matrix, dtype=torch.float32, device=device)
    
    # Build edge index from non-zero weights
    src, dst = np.nonzero(W_matrix)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=device)
    
    # Extract edge weights
    edge_weight = torch.tensor(W_matrix[src, dst], dtype=torch.float32, device=device)
    
    # Add self-loops (GNNs need self-loops for nodes to consider their own features)
    num_nodes = coords.shape[0]
    self_loop_src = torch.arange(num_nodes, dtype=torch.long, device=device)
    self_loop_edge = torch.stack([self_loop_src, self_loop_src])
    
    edge_index = torch.cat([edge_index, self_loop_edge], dim=1)
    
    # Add weights for self-loops (typically 1.0)
    self_loop_weights = torch.ones(num_nodes, dtype=torch.float32, device=device)
    edge_weight = torch.cat([edge_weight, self_loop_weights])
    
    return features, edge_index, edge_weight, W_tensor


def training_step(
    model: MECP_GAP_Model,
    optimizer: torch.optim.Optimizer,
    criterion: MECP_Loss,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    W_tensor: torch.Tensor,
    num_nodes: int,
    num_partitions: int
) -> Tuple[float, Dict[str, float], torch.Tensor]:
    """
    Execute a single training step.
    
    This implements one iteration of the training loop:
    1. Forward pass: Get partition probabilities
    2. Loss calculation: Compute edge cut + balance loss
    3. Backward pass: Compute gradients
    4. Optimizer step: Update model weights
    
    Args:
        model: The MECP-GAP model
        optimizer: The optimizer (Adam)
        criterion: The MECP loss function
        features: Node features (N, 2)
        edge_index: Edge indices (2, E)
        edge_weight: Edge weights (E,)
        W_tensor: Weight matrix (N, N)
        num_nodes: Number of nodes
        num_partitions: Number of partitions
        
    Returns:
        loss_value: Total loss as float
        loss_dict: Dictionary with loss components
        probs: Partition probabilities (N, P)
    """
    model.train()
    
    # A. Forward Pass - get partition probabilities
    probs = model(features, edge_index, edge_weight)
    
    # B. Calculate Loss
    total_loss, loss_dict = criterion(probs, W_tensor, num_nodes, num_partitions)
    
    # C. Backward Pass
    optimizer.zero_grad()  # Clear previous gradients
    total_loss.backward()  # Compute gradients
    optimizer.step()  # Update weights
    
    # Convert loss dict to float values
    loss_dict_float = {k: v.item() for k, v in loss_dict.items()}
    
    return total_loss.item(), loss_dict_float, probs.detach()


class MECPTrainer:
    """
    Main trainer class for MECP-GAP model.
    
    Handles the complete training pipeline including:
    - Model and optimizer initialization
    - Training loop execution
    - Progress monitoring and logging
    - Checkpoint saving
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.
        
        Args:
            config: TrainingConfig with all hyperparameters
        """
        self.config = config
        self.device = config.device
        
        # Model will be built in train() when in_feats=-1 (auto-detect)
        self.model = None
        self.optimizer = None
        
        # Initialize loss function (alpha may be overridden in train())
        self.criterion = MECP_Loss(
            alpha=config.alpha if config.alpha > 0 else 0.001,
            beta=config.beta,
            gamma=config.gamma
        )
        
        # Training history for tracking progress
        self.history: List[Dict[str, Any]] = []
        
        # Checkpoint directory
        if config.checkpoint_dir:
            self.checkpoint_dir = Path(config.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None
    
    def _build_model(self, in_feats: int):
        """Build model and optimizer with the correct input feature dimension."""
        self.model = MECP_GAP_Model(
            in_feats=in_feats,
            hidden_feats=self.config.hidden_feats,
            out_feats=self.config.out_feats,
            num_partitions=self.config.num_partitions,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            aggregator=self.config.aggregator
        ).to(self.device)
        
        # Initialize optimizer (Adam)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def prepare_data(
        self, 
        coords: np.ndarray, 
        W_matrix: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for training."""
        return prepare_graph_data(coords, W_matrix, self.device, self.config.feature_type)
    
    def train(
        self,
        coords: np.ndarray,
        W_matrix: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete training loop per Algorithm 1 from the paper.
        
        The paper describes a 2-phase training procedure:
        - Outer loop (e1 epochs): Updates GraphSAGE embeddings
        - Inner loop (e2 epochs): Trains MLP on mini-batches
        
        Args:
            coords: Node coordinates (N, 2)
            W_matrix: Mobility weight matrix (N, N)
            verbose: Whether to print progress
            
        Returns:
            results: Dictionary containing:
                - final_assignments: Partition assignments (N,)
                - final_probs: Partition probabilities (N, P)
                - history: Training history
                - final_loss: Final loss value
        """
        # Prepare data
        features, edge_index, edge_weight, W_tensor = self.prepare_data(coords, W_matrix)
        
        num_nodes = coords.shape[0]
        num_partitions = self.config.num_partitions
        
        # Auto-detect in_feats from prepared features
        actual_in_feats = features.shape[1]
        if self.config.in_feats <= 0:
            in_feats = actual_in_feats
        else:
            in_feats = self.config.in_feats
        
        # Build model with correct input dimension
        self._build_model(in_feats)
        
        # Auto-compute alpha if not explicitly set
        if self.config.alpha < 0:
            total_weight = float(W_tensor.sum())
            # Scale alpha so that cut_loss ~ balance_loss at initialization
            # At random assignment: cut ~ (1-1/P)*sum(W), balance ~ N^2/P
            # We want alpha * (1-1/P)*sum(W) ~ beta * N^2/P
            alpha = self.config.beta * (num_nodes ** 2 / num_partitions) / ((1 - 1.0/num_partitions) * total_weight + 1e-8)
            self.criterion = MECP_Loss(
                alpha=alpha,
                beta=self.config.beta,
                gamma=self.config.gamma,
                normalize_cut=False
            )
            if verbose:
                print(f"Auto-computed alpha = {alpha:.6f} (sum(W) = {total_weight:.0f})")
        
        if verbose:
            print(f"Starting Training on {num_nodes} nodes...")
            print(f"Model: {self.config.hidden_feats}-dim embeddings, {self.config.num_layers} GNN layers")
            print(f"Features: {self.config.feature_type} (dim={in_feats})")
            print(f"Target: {num_partitions} partitions")
            print(f"Device: {self.device}")
            print("-" * 60)
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(self.config.num_epochs + 1):
            # Single training step
            loss, loss_dict, probs = training_step(
                self.model,
                self.optimizer,
                self.criterion,
                features,
                edge_index,
                edge_weight,
                W_tensor,
                num_nodes,
                num_partitions
            )
            
            # Get hard assignments for monitoring
            assignments = torch.argmax(probs, dim=1)
            partition_sizes = torch.bincount(assignments, minlength=num_partitions)
            
            # Record history
            history_entry = {
                'epoch': epoch,
                'total_loss': loss,
                **loss_dict,
                'partition_sizes': partition_sizes.cpu().tolist()
            }
            self.history.append(history_entry)
            
            # Logging
            if verbose and epoch % self.config.log_interval == 0:
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
                print(f"   -> Cut Loss: {loss_dict['weighted_cut_loss']:.4f}, "
                      f"Balance Loss: {loss_dict['weighted_balance_loss']:.4f}")
                print(f"   -> Partition Sizes: {partition_sizes.tolist()}")
            
            # Save checkpoint
            if self.checkpoint_dir and epoch % self.config.save_interval == 0:
                self._save_checkpoint(epoch)
        
        training_time = time.time() - start_time
        
        # Final inference
        final_probs, final_assignments = self.inference(features, edge_index, edge_weight)
        
        if verbose:
            print("-" * 60)
            print(f"Training Complete in {training_time:.2f}s")
            final_sizes = torch.bincount(
                torch.tensor(final_assignments), 
                minlength=num_partitions
            ).tolist()
            print(f"Final Partition Sizes: {final_sizes}")
        
        return {
            'final_assignments': final_assignments,
            'final_probs': final_probs,
            'history': self.history,
            'final_loss': self.history[-1]['total_loss'],
            'training_time': training_time
        }
    
    def inference(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference to get final partition assignments.
        
        Args:
            features: Node features
            edge_index: Edge indices
            edge_weight: Edge weights
            
        Returns:
            probs: Partition probabilities as numpy array (N, P)
            assignments: Hard partition assignments as numpy array (N,)
        """
        self.model.eval()
        with torch.no_grad():
            probs = self.model(features, edge_index, edge_weight)
            assignments = torch.argmax(probs, dim=1)
        
        return probs.cpu().numpy(), assignments.cpu().numpy()
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'history': self.history
        }
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt'
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', [])
        return checkpoint.get('epoch', 0)


def train_mecp_gap(
    coords: np.ndarray,
    W_matrix: np.ndarray,
    num_partitions: int = 4,
    num_epochs: int = 200,
    learning_rate: float = 0.01,
    alpha: float = -1.0,
    beta: float = 1.0,
    gamma: float = 0.0,
    feature_type: str = 'weight_row',
    verbose: bool = True,
    device: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to train MECP-GAP model.
    
    This is a simple interface for quick training without
    creating config and trainer objects explicitly.
    
    Args:
        coords: Node coordinates (N, 2)
        W_matrix: Mobility weight matrix (N, N)
        num_partitions: Number of partitions (MEC servers)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        alpha: Weight for edge cut loss (-1 = auto)
        beta: Weight for balance loss
        gamma: Weight for entropy regularization
        feature_type: 'weight_row' (paper) or 'coords'
        verbose: Whether to print progress
        device: Target device ('cpu' or 'cuda')
        
    Returns:
        assignments: Final partition assignments (N,)
        results: Dictionary with training results
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = TrainingConfig(
        in_feats=-1,  # Auto-detect
        num_partitions=num_partitions,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        feature_type=feature_type,
        device=device
    )
    
    trainer = MECPTrainer(config)
    results = trainer.train(coords, W_matrix, verbose=verbose)
    
    return results['final_assignments'], results
