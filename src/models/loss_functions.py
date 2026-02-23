"""
Custom Loss Functions for MECP-GAP

This module implements the loss functions as described in the paper:
- Edge Cut Loss (Equation 7): Minimizes handover costs
- Load Balancing Loss (Equation 8): Ensures balanced partition sizes
- Total Loss (Equation 9): Combines both objectives

Paper Reference: Section IV.C - Loss Function Design
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class EdgeCutLoss(nn.Module):
    """
    Edge Cut Loss (Equation 7 from paper)
    
    Minimizes the total weight of edges cut by the partition.
    If nodes i and j have high mobility flow W[i,j] and are in different
    partitions, this incurs a high penalty.
    
    Formula: L_cut = sum_{i,j} W[i,j] * P(i and j in different partitions)
           = sum_{i,j} W[i,j] * (1 - sum_k X[i,k] * X[j,k])
    
    where X is the N x P probability matrix (output of model)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Compute edge cut loss.
        
        Args:
            X: Partition probability matrix (N, P) from model output
            W: Mobility weight matrix (N, N) - adjacency with mobility weights
            
        Returns:
            Edge cut loss value (scalar)
        """
        # Probability that nodes i and j are in the SAME partition
        # P_same[i,j] = sum_k (X[i,k] * X[j,k]) = X @ X.T
        P_same = torch.matmul(X, X.t())
        
        # Probability that nodes are in DIFFERENT partitions
        P_diff = 1 - P_same
        
        # Weight by mobility: penalize separating nodes with high traffic
        # Edge cut = sum of W[i,j] * P_diff[i,j] for all edges
        cut_loss = torch.sum(P_diff * W)
        
        return cut_loss


class LoadBalanceLoss(nn.Module):
    """
    Load Balance Loss (Equation 8 from paper)
    
    Ensures partitions have approximately equal sizes.
    Penalizes deviation from the ideal partition size N/P.
    
    Formula: L_balance = sum_k (|S_k| - N/P)^2
    
    where S_k is the expected size of partition k.
    Since X is soft, |S_k| = sum_i X[i,k]
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, X: torch.Tensor, 
                num_nodes: int, 
                num_partitions: int) -> torch.Tensor:
        """
        Compute load balance loss.
        
        Args:
            X: Partition probability matrix (N, P)
            num_nodes: Total number of nodes N
            num_partitions: Number of partitions P
            
        Returns:
            Load balance loss value (scalar)
        """
        # Expected size of each partition (soft count)
        # partition_sizes[k] = sum_i X[i,k]
        partition_sizes = torch.sum(X, dim=0)  # Shape: (P,)
        
        # Ideal size for balanced partitions
        ideal_size = num_nodes / num_partitions
        
        # Squared deviation from ideal
        balance_loss = torch.sum((partition_sizes - ideal_size) ** 2)
        
        return balance_loss


class MECP_Loss(nn.Module):
    """
    Combined MECP Loss Function (Equation 9 from paper)
    
    Total Loss = alpha * L_cut + beta * L_balance + gamma * L_entropy
    
    Where:
    - L_cut: Edge cut loss (handover minimization)
    - L_balance: Load balance loss (server load balancing)
    - L_entropy: Entropy regularization (encourages confident predictions)
    - alpha: Weight for edge cut (default 1/1000 = 0.001 per paper)
    - beta: Weight for balance (default 1.0 per paper)
    - gamma: Weight for entropy (default -0.1, negative to encourage low entropy)
    
    Paper Hyperparameters (Section V.B):
    - alpha = 1/1000 (accounts for scale difference)
    - beta = 1.0
    """
    
    def __init__(self, 
                 alpha: float = 0.001, 
                 beta: float = 1.0,
                 gamma: float = -0.1,
                 normalize_cut: bool = True):
        """
        Initialize MECP loss function.
        
        Args:
            alpha: Weight for edge cut loss (default 0.001 per paper)
            beta: Weight for load balance loss (default 1.0 per paper)
            gamma: Weight for entropy loss (negative = encourage confident predictions)
            normalize_cut: Whether to normalize cut loss by total edges
        """
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.normalize_cut = normalize_cut
        
        self.edge_cut_loss = EdgeCutLoss()
        self.balance_loss = LoadBalanceLoss()
    
    def forward(self, 
                X: torch.Tensor, 
                W: torch.Tensor,
                num_nodes: Optional[int] = None,
                num_partitions: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total MECP loss.
        
        Args:
            X: Partition probability matrix (N, P) - model output
            W: Mobility weight matrix (N, N)
            num_nodes: Number of nodes (inferred from X if not provided)
            num_partitions: Number of partitions (inferred from X if not provided)
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        if num_nodes is None:
            num_nodes = X.size(0)
        if num_partitions is None:
            num_partitions = X.size(1)
        
        # Compute individual losses
        cut_loss = self.edge_cut_loss(X, W)
        balance_loss = self.balance_loss(X, num_nodes, num_partitions)
        
        # Normalize cut loss if requested
        if self.normalize_cut:
            total_edge_weight = W.sum() + 1e-8
            cut_loss = cut_loss / total_edge_weight
        
        # Entropy regularization: encourages confident (peaky) predictions
        # H(X) = -sum(X * log(X)), we want low entropy so we subtract it
        entropy = -torch.sum(X * torch.log(X + 1e-8)) / num_nodes
        
        # Combined loss (Equation 9 + entropy)
        total_loss = self.alpha * cut_loss + self.beta * balance_loss + self.gamma * entropy
        
        # Return losses for logging
        loss_dict = {
            'total_loss': total_loss,
            'cut_loss': cut_loss,
            'balance_loss': balance_loss,
            'entropy': entropy,
            'weighted_cut_loss': self.alpha * cut_loss,
            'weighted_balance_loss': self.beta * balance_loss,
            'weighted_entropy': self.gamma * entropy
        }
        
        return total_loss, loss_dict


class NormalizedCutLoss(nn.Module):
    """
    Normalized Cut Loss (Alternative formulation)
    
    Normalized cut provides better theoretical guarantees
    by normalizing by partition volume:
    
    NCut = sum_k cut(S_k, V\S_k) / vol(S_k)
    
    where vol(S_k) = sum of degrees of nodes in S_k
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized cut loss.
        
        Args:
            X: Partition probability matrix (N, P)
            W: Weight matrix (N, N)
            
        Returns:
            Normalized cut loss
        """
        N, P = X.shape
        
        # Compute degrees
        degrees = W.sum(dim=1)  # (N,)
        
        # Partition volumes (weighted by degree)
        volumes = torch.matmul(degrees.unsqueeze(0), X).squeeze(0)  # (P,)
        
        # For numerical stability
        volumes = volumes.clamp(min=1e-8)
        
        # Compute cut for each partition
        # Association of partition k with all nodes
        # assoc[k] = sum_i sum_j X[i,k] * X[j,k] * W[i,j]
        # cut[k] = vol[k] - assoc[k]
        
        # Efficient computation using matrix operations
        WX = torch.matmul(W, X)  # (N, P)
        assoc = torch.sum(X * WX, dim=0)  # (P,)
        
        cuts = volumes - assoc
        
        # Normalized cut: sum of cut/volume for each partition
        ncut = torch.sum(cuts / volumes)
        
        return ncut


class RatioCutLoss(nn.Module):
    """
    Ratio Cut Loss (Alternative formulation)
    
    Ratio cut normalizes by partition size instead of volume:
    
    RCut = sum_k cut(S_k, V\S_k) / |S_k|
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Compute ratio cut loss.
        
        Args:
            X: Partition probability matrix (N, P)
            W: Weight matrix (N, N)
            
        Returns:
            Ratio cut loss
        """
        N, P = X.shape
        
        # Partition sizes
        sizes = X.sum(dim=0)  # (P,)
        sizes = sizes.clamp(min=1e-8)
        
        # Compute association
        WX = torch.matmul(W, X)  # (N, P)
        assoc = torch.sum(X * WX, dim=0)  # (P,)
        
        # Total weight going into each partition
        volumes = torch.matmul(W.sum(dim=1).unsqueeze(0), X).squeeze(0)  # (P,)
        
        cuts = volumes - 2 * assoc  # Factor of 2 because W is symmetric
        cuts = cuts.clamp(min=0)  # Numerical stability
        
        # Ratio cut
        rcut = torch.sum(cuts / sizes)
        
        return rcut


class ModularityLoss(nn.Module):
    """
    Modularity Loss (Alternative objective)
    
    Modularity measures the quality of a partition by comparing
    edge density within partitions to expected density in a null model.
    
    Q = (1/2m) * sum_{i,j} (W[i,j] - d_i*d_j/2m) * delta(c_i, c_j)
    
    where m = sum of all edge weights
    
    We want to MAXIMIZE modularity, so loss = -Q
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Compute negative modularity loss.
        
        Args:
            X: Partition probability matrix (N, P)
            W: Weight matrix (N, N)
            
        Returns:
            Negative modularity (to minimize)
        """
        # Total edge weight
        m = W.sum() / 2  # Divide by 2 for undirected graph
        m = m.clamp(min=1e-8)
        
        # Node degrees
        degrees = W.sum(dim=1)  # (N,)
        
        # Expected edges under null model
        # B[i,j] = W[i,j] - d_i * d_j / (2m)
        B = W - torch.outer(degrees, degrees) / (2 * m)
        
        # Modularity contribution from soft partition
        # sum_{k} sum_{i,j} X[i,k] * X[j,k] * B[i,j]
        XTX = torch.matmul(X.t(), X)  # (P, P) - for normalization
        
        # Compute BX
        BX = torch.matmul(B, X)  # (N, P)
        
        # Modularity Q = trace(X^T B X) / (2m)
        Q = torch.sum(X * BX) / (2 * m)
        
        # Return negative (we minimize loss, but want to maximize modularity)
        return -Q


class MECP_Loss_Extended(nn.Module):
    """
    Extended MECP Loss with additional regularization terms.
    
    Adds:
    - Entropy regularization: Encourages sharper partition assignments
    - Orthogonality regularization: Encourages diverse partition memberships
    """
    
    def __init__(self,
                 alpha: float = 0.001,
                 beta: float = 1.0,
                 entropy_weight: float = 0.0,
                 orthogonality_weight: float = 0.0):
        """
        Args:
            alpha: Edge cut weight
            beta: Balance weight
            entropy_weight: Weight for entropy regularization
            orthogonality_weight: Weight for orthogonality regularization
        """
        super().__init__()
        
        self.base_loss = MECP_Loss(alpha=alpha, beta=beta)
        self.entropy_weight = entropy_weight
        self.orthogonality_weight = orthogonality_weight
    
    def forward(self,
                X: torch.Tensor,
                W: torch.Tensor,
                num_nodes: Optional[int] = None,
                num_partitions: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute extended MECP loss.
        """
        # Base MECP loss
        base_loss, loss_dict = self.base_loss(X, W, num_nodes, num_partitions)
        
        # Entropy regularization: -sum(X * log(X))
        # Low entropy = sharper assignments
        if self.entropy_weight > 0:
            entropy = -torch.sum(X * torch.log(X.clamp(min=1e-8)))
            base_loss = base_loss + self.entropy_weight * entropy
            loss_dict['entropy'] = entropy
        
        # Orthogonality: ||X^T X - I||_F^2
        # Encourages orthogonal soft assignments
        if self.orthogonality_weight > 0:
            XTX = torch.matmul(X.t(), X)
            P = X.size(1)
            I = torch.eye(P, device=X.device) * (X.size(0) / P)  # Scaled identity
            ortho_loss = torch.norm(XTX - I, p='fro') ** 2
            base_loss = base_loss + self.orthogonality_weight * ortho_loss
            loss_dict['orthogonality'] = ortho_loss
        
        loss_dict['total_loss'] = base_loss
        
        return base_loss, loss_dict


def compute_hard_cut(assignments: torch.Tensor, W: torch.Tensor) -> float:
    """
    Compute actual edge cut from hard partition assignments.
    
    This is the true metric used for evaluation (not the soft loss).
    
    Args:
        assignments: Hard partition assignments (N,) with values in [0, P-1]
        W: Weight matrix (N, N)
        
    Returns:
        Total weight of cut edges
    """
    N = len(assignments)
    cut = 0.0
    
    for i in range(N):
        for j in range(i + 1, N):
            if assignments[i] != assignments[j]:
                cut += W[i, j].item()
    
    return cut


def compute_balance_ratio(assignments: torch.Tensor, 
                          num_partitions: int) -> float:
    """
    Compute balance ratio of partition.
    
    Balance ratio = max_partition_size / ideal_size
    
    Perfect balance = 1.0
    
    Args:
        assignments: Hard partition assignments (N,)
        num_partitions: Number of partitions P
        
    Returns:
        Balance ratio (>=1.0, lower is better)
    """
    N = len(assignments)
    ideal_size = N / num_partitions
    
    sizes = []
    for p in range(num_partitions):
        sizes.append((assignments == p).sum().item())
    
    max_size = max(sizes)
    
    return max_size / ideal_size if ideal_size > 0 else float('inf')


def compute_modularity(assignments: torch.Tensor, W: torch.Tensor) -> float:
    """
    Compute modularity of partition.
    
    Args:
        assignments: Hard partition assignments (N,)
        W: Weight matrix (N, N)
        
    Returns:
        Modularity value (higher is better)
    """
    m = W.sum().item() / 2
    if m == 0:
        return 0.0
    
    degrees = W.sum(dim=1)
    N = len(assignments)
    
    Q = 0.0
    for i in range(N):
        for j in range(N):
            if assignments[i] == assignments[j]:
                expected = degrees[i].item() * degrees[j].item() / (2 * m)
                Q += (W[i, j].item() - expected)
    
    return Q / (2 * m)
