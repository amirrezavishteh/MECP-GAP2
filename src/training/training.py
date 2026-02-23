"""
MECP-GAP Training Module

This module manages the interaction between the MECP-GAP Model (the neural network)
and the MECP Loss function (the teacher). It provides a simplified training interface.

Paper Reference: Section V - Experiments
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.mecp_gap_model import MECP_GAP_Model
from models.loss_functions import MECP_Loss


def prepare_tensors(
    coords: np.ndarray,
    W: np.ndarray,
    device: str = 'cpu',
    feature_type: str = 'weight_row'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert numpy arrays to PyTorch tensors and build graph structure.
    
    This function prepares:
    1. Node features (weight-row connectivity or coordinates)
    2. Edge index (sparse adjacency)
    3. Edge weights (mobility traffic)
    4. Weight matrix (for loss computation)
    
    Args:
        coords: Node coordinates of shape (N, 2)
        W: Mobility weight matrix of shape (N, N)
        device: Target device ('cpu' or 'cuda')
        feature_type: 'weight_row' (paper default, dim=N) or 'coords' (dim=2)
        
    Returns:
        features: Node features tensor (N, feat_dim)
        edge_index: Edge indices tensor (2, E)
        edge_weights: Edge weights tensor (E,)
        W_tensor: Weight matrix tensor (N, N)
    """
    num_nodes = len(coords)
    
    # Convert to tensors
    if feature_type == 'weight_row':
        row_sums = W.sum(axis=1, keepdims=True) + 1e-8
        features_np = W / row_sums
        features = torch.tensor(features_np, dtype=torch.float32, device=device)
    else:
        features = torch.tensor(coords, dtype=torch.float32, device=device)
    W_tensor = torch.tensor(W, dtype=torch.float32, device=device)
    
    # Build edge index from non-zero weights
    src, dst = np.nonzero(W)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=device)
    
    # Extract edge weights for message passing
    weights = torch.tensor(W[src, dst], dtype=torch.float32, device=device)
    
    # Add self-loops (Required for GraphSAGE - nodes must aggregate their own features)
    self_loop_idx = torch.arange(num_nodes, dtype=torch.long, device=device)
    self_loop_edge = torch.stack([self_loop_idx, self_loop_idx])
    self_loop_weights = torch.ones(num_nodes, dtype=torch.float32, device=device)
    
    # Concatenate original edges with self-loops
    edge_index = torch.cat([edge_index, self_loop_edge], dim=1)
    edge_weights = torch.cat([weights, self_loop_weights])
    
    return features, edge_index, edge_weights, W_tensor


def train_model(
    coords: np.ndarray,
    W: np.ndarray,
    num_partitions: int = 4,
    epochs: int = 200,
    lr: float = 0.01,
    hidden_feats: int = 128,
    alpha: float = 0.001,
    beta: float = 1.0,
    device: Optional[str] = None,
    verbose: bool = True,
    log_interval: int = 50
) -> Tuple[np.ndarray, MECP_GAP_Model]:
    """
    Executes the training loop for MECP-GAP.
    
    This is the main entry point for training the model. It handles:
    1. Data Preparation: Converting numpy arrays to PyTorch tensors
    2. Optimization: Updating weights using Adam optimizer
    3. Monitoring: Printing loss and partition sizes for convergence check
    
    Args:
        coords: Node coordinates (N, 2) - the base station locations
        W: Mobility weight matrix (N, N) - traffic between cells
        num_partitions: Number of partitions P (MEC servers)
        epochs: Number of training epochs
        lr: Learning rate for Adam optimizer
        hidden_feats: Hidden dimension size (paper uses 128)
        alpha: Weight for edge cut loss (paper uses 0.001)
        beta: Weight for balance loss (paper uses 1.0)
        device: 'cpu' or 'cuda' (auto-detected if None)
        verbose: Whether to print progress
        log_interval: Print every N epochs
        
    Returns:
        final_assignments: (N,) array of partition IDs (0 to P-1)
        model: Trained MECP_GAP_Model instance
        
    Example:
        >>> from utils.utils import generate_synthetic_data
        >>> coords, adj, W = generate_synthetic_data(num_nodes=100, seed=42)
        >>> assignments, model = train_model(coords, W, num_partitions=4, epochs=200)
        >>> print(f"Partition sizes: {np.bincount(assignments, minlength=4)}")
    """
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- A. Setup Tensors & Graph ---
    num_nodes = len(coords)
    features, edge_index, edge_weights, W_tensor = prepare_tensors(coords, W, device)
    
    # --- B. Initialize Model ---
    in_feats = features.shape[1]  # Auto-detect from features
    model = MECP_GAP_Model(
        in_feats=in_feats,
        hidden_feats=hidden_feats,
        out_feats=hidden_feats,
        num_partitions=num_partitions,
        num_layers=2,  # Paper uses 2-layer GNN
        aggregator='mean'
    ).to(device)
    
    # Optimizer (Adam is standard for GNNs)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss Function (Parameters from paper Eq.9: Alpha=1/1000, Beta=1, no normalization)
    criterion = MECP_Loss(alpha=alpha, beta=beta, normalize_cut=False)
    
    # --- C. Training Loop ---
    if verbose:
        print(f"--- Starting Training ({epochs} Epochs) ---")
        print(f"Nodes: {num_nodes} | Partitions: {num_partitions} | Device: {device}")
        print("-" * 55)
    
    for epoch in range(epochs):
        model.train()
        
        # 1. Forward Pass - Returns Probability Matrix X (N x P)
        probs = model(features, edge_index, edge_weights)
        
        # 2. Loss Calculation
        loss, loss_dict = criterion(probs, W_tensor, num_nodes, num_partitions)
        cut_loss = loss_dict['cut_loss']
        bal_loss = loss_dict['balance_loss']
        
        # 3. Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 4. Logging (Every log_interval epochs)
        if verbose and (epoch % log_interval == 0 or epoch == epochs - 1):
            # Convert probs to hard assignments for visualization
            with torch.no_grad():
                assignments = torch.argmax(probs, dim=1)
                counts = torch.bincount(assignments, minlength=num_partitions).tolist()
            
            print(f"Epoch {epoch:03d} | Total Loss: {loss.item():.4f}")
            print(f"   -> Cut Loss: {cut_loss.item():.4f} | Bal Loss: {bal_loss.item():.4f}")
            print(f"   -> Partition Sizes: {counts}")
    
    # --- D. Final Inference ---
    model.eval()
    with torch.no_grad():
        final_probs = model(features, edge_index, edge_weights)
        final_assignments = torch.argmax(final_probs, dim=1).cpu().numpy()
    
    if verbose:
        print("-" * 55)
        print("Training Complete!")
        final_counts = np.bincount(final_assignments, minlength=num_partitions)
        print(f"Final Partition Distribution: {final_counts.tolist()}")
    
    return final_assignments, model


def train_model_with_history(
    coords: np.ndarray,
    W: np.ndarray,
    num_partitions: int = 4,
    epochs: int = 200,
    lr: float = 0.01,
    hidden_feats: int = 128,
    alpha: float = 0.001,
    beta: float = 1.0,
    device: Optional[str] = None,
    verbose: bool = True,
    log_interval: int = 50
) -> Dict[str, Any]:
    """
    Extended version of train_model that returns complete training history.
    
    Useful for plotting loss curves and analyzing convergence.
    
    Args:
        Same as train_model
        
    Returns:
        Dictionary containing:
            - 'assignments': Final partition assignments (N,)
            - 'model': Trained model
            - 'history': List of dicts with loss values per epoch
            - 'final_probs': Final partition probabilities (N, P)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    num_nodes = len(coords)
    features, edge_index, edge_weights, W_tensor = prepare_tensors(coords, W, device)
    
    in_feats = features.shape[1]
    model = MECP_GAP_Model(
        in_feats=in_feats,
        hidden_feats=hidden_feats,
        out_feats=hidden_feats,
        num_partitions=num_partitions,
        num_layers=2,
        aggregator='mean'
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = MECP_Loss(alpha=alpha, beta=beta, normalize_cut=False)
    
    history = []
    
    if verbose:
        print(f"--- Starting Training ({epochs} Epochs) ---")
    
    for epoch in range(epochs):
        model.train()
        
        probs = model(features, edge_index, edge_weights)
        loss, loss_dict = criterion(probs, W_tensor, num_nodes, num_partitions)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record history
        with torch.no_grad():
            assignments = torch.argmax(probs, dim=1)
            counts = torch.bincount(assignments, minlength=num_partitions)
        
        history.append({
            'epoch': epoch,
            'total_loss': loss.item(),
            'cut_loss': loss_dict['cut_loss'].item(),
            'balance_loss': loss_dict['balance_loss'].item(),
            'partition_sizes': counts.cpu().tolist()
        })
        
        if verbose and (epoch % log_interval == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Sizes: {counts.tolist()}")
    
    # Final inference
    model.eval()
    with torch.no_grad():
        final_probs = model(features, edge_index, edge_weights)
        final_assignments = torch.argmax(final_probs, dim=1).cpu().numpy()
    
    return {
        'assignments': final_assignments,
        'model': model,
        'history': history,
        'final_probs': final_probs.cpu().numpy()
    }


def evaluate_model(
    model: MECP_GAP_Model,
    coords: np.ndarray,
    W: np.ndarray,
    device: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a trained model.
    
    Args:
        model: Trained MECP_GAP_Model
        coords: Node coordinates (N, 2)
        W: Weight matrix (N, N)
        device: Target device
        
    Returns:
        assignments: Partition assignments (N,)
        probs: Partition probabilities (N, P)
    """
    if device is None:
        device = next(model.parameters()).device
    
    features, edge_index, edge_weights, _ = prepare_tensors(coords, W, str(device))
    
    model.eval()
    with torch.no_grad():
        probs = model(features, edge_index, edge_weights)
        assignments = torch.argmax(probs, dim=1).cpu().numpy()
    
    return assignments, probs.cpu().numpy()


def save_model(model: MECP_GAP_Model, filepath: str) -> None:
    """
    Save trained model to file.
    
    Args:
        model: Trained model
        filepath: Path to save model
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'in_feats': model.in_feats,
        'hidden_feats': model.hidden_feats,
        'out_feats': model.out_feats,
        'num_partitions': model.num_partitions,
        'num_layers': model.num_layers
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str, device: Optional[str] = None) -> MECP_GAP_Model:
    """
    Load a trained model from file.
    
    Args:
        filepath: Path to model file
        device: Target device
        
    Returns:
        Loaded model
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model = MECP_GAP_Model(
        in_feats=checkpoint['in_feats'],
        hidden_feats=checkpoint['hidden_feats'],
        out_feats=checkpoint['out_feats'],
        num_partitions=checkpoint['num_partitions'],
        num_layers=checkpoint['num_layers']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {filepath}")
    return model


# =============================================================================
# Quick Training Interface
# =============================================================================

def quick_train(
    num_nodes: int = 200,
    num_partitions: int = 4,
    epochs: int = 200,
    seed: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    All-in-one function: generate data and train model.
    
    Convenient for quick experiments and testing.
    
    Args:
        num_nodes: Number of nodes to generate
        num_partitions: Number of partitions
        epochs: Training epochs
        seed: Random seed for reproducibility
        **kwargs: Additional arguments passed to train_model
        
    Returns:
        Dictionary with results including coords, W, assignments, model
    """
    # Import data generation
    from utils.utils import generate_synthetic_data, compute_metrics
    
    # Generate data
    coords, adj, W = generate_synthetic_data(num_nodes=num_nodes, seed=seed)
    
    # Train model
    assignments, model = train_model(
        coords, W,
        num_partitions=num_partitions,
        epochs=epochs,
        **kwargs
    )
    
    # Compute metrics
    edge_cut, balance_var, sizes = compute_metrics(assignments, W, num_partitions)
    
    return {
        'coords': coords,
        'adj': adj,
        'W': W,
        'assignments': assignments,
        'model': model,
        'edge_cut': edge_cut,
        'balance_var': balance_var,
        'partition_sizes': sizes
    }


# =============================================================================
# Main Script Entry Point
# =============================================================================

if __name__ == "__main__":
    """
    Example usage when running as a script.
    """
    print("MECP-GAP Training Module")
    print("=" * 55)
    
    # Quick demonstration
    results = quick_train(
        num_nodes=100,
        num_partitions=4,
        epochs=100,
        seed=42,
        verbose=True
    )
    
    print("\n" + "=" * 55)
    print("RESULTS:")
    print(f"Edge Cut: {results['edge_cut']:.4f}")
    print(f"Load Balance Variance: {results['balance_var']:.4f}")
    print(f"Partition Sizes: {results['partition_sizes'].tolist()}")
