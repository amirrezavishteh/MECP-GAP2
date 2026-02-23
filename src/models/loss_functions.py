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
        
        # Squared deviation from ideal (Equation 8 from paper)
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
                 gamma: float = 0.0,
                 normalize_cut: bool = False):
        """
        Initialize MECP loss function.
        
        Paper Equation 9: L = α·Edge_cut + β·Load_balancing
        
        Args:
            alpha: Weight for edge cut loss (default 0.001 = 1/1000 per paper)
            beta: Weight for load balance loss (default 1.0 per paper)
            gamma: Weight for entropy loss (default 0; paper does NOT use entropy)
            normalize_cut: Whether to normalize cut loss (default False per paper)
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
    r"""
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
    r"""
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


# =============================================================================
# SSC-Aware Loss Functions (5G Session and Service Continuity)
# =============================================================================

class SSCModeCost(nn.Module):
    """
    5G Session and Service Continuity (SSC) Mode Costs.
    
    In 5G core networks, there are three SSC modes that govern how
    service continuity is maintained during user mobility:
    
    SSC Mode 1: Server remains fixed for session duration. 
        - Cost: Increasing latency as user moves away from server
        - No migration cost, but latency penalty grows with distance
        
    SSC Mode 2: "Break before make" - release current server, allocate new
        - Cost: Service interruption during handover
        - Lower resource usage but higher latency spike
        
    SSC Mode 3: "Make before break" - establish new before releasing old
        - Cost: Requires geographically proximal servers for zero-downtime
        - Higher resource usage but seamless transition
        
    MECP-GAP's advantage: Explicitly models these costs in the loss function,
    providing more nuanced partitions than METIS (treats as single edge weight),
    Greedy (ignores entirely), or PBPA (can learn but complicates reward shaping).
    
    Paper Reference: Section on 5G Service Continuity in Placement
    """
    
    def __init__(self, 
                 mode1_weight: float = 0.3,
                 mode2_weight: float = 0.3,
                 mode3_weight: float = 0.4,
                 interruption_penalty: float = 2.0,
                 proximity_threshold: float = 0.5):
        """
        Initialize SSC mode costs.
        
        Args:
            mode1_weight: Fraction of sessions using SSC Mode 1
            mode2_weight: Fraction of sessions using SSC Mode 2
            mode3_weight: Fraction of sessions using SSC Mode 3
            interruption_penalty: Penalty multiplier for service interruption (Mode 2)
            proximity_threshold: Distance threshold for Mode 3 proximity requirement
        """
        super().__init__()
        
        assert abs(mode1_weight + mode2_weight + mode3_weight - 1.0) < 1e-6, \
            "SSC mode weights must sum to 1.0"
        
        self.mode1_weight = mode1_weight
        self.mode2_weight = mode2_weight
        self.mode3_weight = mode3_weight
        self.interruption_penalty = interruption_penalty
        self.proximity_threshold = proximity_threshold
    
    def forward(self, X: torch.Tensor, W: torch.Tensor,
                coords: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute SSC-aware handover costs.
        
        For each edge (i,j) that is cut (nodes in different partitions),
        the cost depends on the SSC mode:
        - Mode 1: Proportional to distance (latency increases with distance)
        - Mode 2: Fixed interruption penalty * traffic weight
        - Mode 3: Penalty if servers are not geographically proximal
        
        Args:
            X: Partition probability matrix (N, P)
            W: Mobility weight matrix (N, N)
            coords: Node coordinates (N, 2) for distance computation
            
        Returns:
            total_ssc_cost: Combined SSC cost
            cost_dict: Dictionary with per-mode costs
        """
        N = X.shape[0]
        
        # Probability that nodes i and j are in different partitions
        P_same = torch.matmul(X, X.t())
        P_diff = 1.0 - P_same
        
        # Mode 1 cost: latency penalty proportional to distance for cut edges
        if coords is not None:
            # Compute pairwise distances
            diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 2)
            distances = torch.norm(diff, dim=-1)  # (N, N)
            # Normalize distances
            max_dist = distances.max().clamp(min=1e-8)
            distances_norm = distances / max_dist
            
            # Mode 1: Latency grows with distance when session stays at original server
            mode1_cost = torch.sum(W * P_diff * distances_norm)
        else:
            # Without coordinates, use basic edge cut as proxy
            mode1_cost = torch.sum(W * P_diff)
        
        # Mode 2 cost: service interruption penalty for "break before make"
        # Every cut edge incurs a fixed interruption penalty
        mode2_cost = self.interruption_penalty * torch.sum(W * P_diff)
        
        # Mode 3 cost: proximity penalty for "make before break"
        # Requires geographically proximal servers - penalize if partition
        # centroids are far apart
        if coords is not None:
            # Compute soft partition centroids
            partition_weights = X.t()  # (P, N)
            weight_sums = partition_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            centroids = torch.matmul(partition_weights, coords) / weight_sums  # (P, 2)
            
            # For each cut edge, compute distance between partition centroids
            # Approximate: penalize when centroids of the two partitions are far
            # P_partition[i] = X[i, :] gives soft partition membership for node i
            # For node i assigned to partition p, and node j to partition q,
            # Mode 3 cost ~ distance(centroid_p, centroid_q) if p != q
            
            centroid_dists = torch.cdist(centroids, centroids)  # (P, P)
            max_centroid_dist = centroid_dists.max().clamp(min=1e-8)
            centroid_dists_norm = centroid_dists / max_centroid_dist
            
            # Expected centroid distance for each pair of nodes
            # E[dist(c_pi, c_pj)] = sum_{p,q} X[i,p] * X[j,q] * dist(c_p, c_q)
            # = X @ centroid_dists @ X.T
            expected_centroid_dist = torch.matmul(X, torch.matmul(centroid_dists_norm, X.t()))
            
            # Mode 3 cost: penalize large centroid distances for high-traffic edges
            mode3_cost = torch.sum(W * expected_centroid_dist)
        else:
            mode3_cost = torch.sum(W * P_diff) * 1.5  # Proxy without coordinates
        
        # Combined SSC cost
        total_ssc_cost = (self.mode1_weight * mode1_cost + 
                          self.mode2_weight * mode2_cost + 
                          self.mode3_weight * mode3_cost)
        
        cost_dict = {
            'ssc_mode1_cost': mode1_cost,
            'ssc_mode2_cost': mode2_cost,
            'ssc_mode3_cost': mode3_cost,
            'ssc_total_cost': total_ssc_cost
        }
        
        return total_ssc_cost, cost_dict


class MECP_Loss_SSC(nn.Module):
    """
    MECP Loss with SSC-Aware Cost Integration.
    
    This is the key innovation of MECP-GAP over baseline algorithms:
    - METIS: Treats SSC modes as a single edge weight
    - Greedy: Ignores SSC modes entirely (local decisions)
    - PBPA: Can learn SSC costs but complicates reward shaping and training
    - MECP-GAP (this): Explicitly models all three SSC modes in the loss function
    
    Total Loss = alpha * L_cut + beta * L_balance + gamma * L_entropy + delta * L_ssc
    
    Paper Reference: Section IV.C - Loss Function Design (specialized loss)
    """
    
    def __init__(self,
                 alpha: float = 0.001,
                 beta: float = 1.0,
                 gamma: float = -0.1,
                 delta: float = 0.0005,
                 mode1_weight: float = 0.3,
                 mode2_weight: float = 0.3,
                 mode3_weight: float = 0.4,
                 normalize_cut: bool = True):
        """
        Initialize MECP Loss with SSC awareness.
        
        Args:
            alpha: Weight for edge cut loss
            beta: Weight for load balance loss
            gamma: Weight for entropy (negative = encourage confident predictions)
            delta: Weight for SSC cost
            mode1_weight: Fraction of SSC Mode 1 sessions
            mode2_weight: Fraction of SSC Mode 2 sessions
            mode3_weight: Fraction of SSC Mode 3 sessions
            normalize_cut: Whether to normalize cut loss
        """
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.normalize_cut = normalize_cut
        
        self.edge_cut_loss = EdgeCutLoss()
        self.balance_loss = LoadBalanceLoss()
        self.ssc_cost = SSCModeCost(
            mode1_weight=mode1_weight,
            mode2_weight=mode2_weight,
            mode3_weight=mode3_weight
        )
    
    def forward(self, X: torch.Tensor, W: torch.Tensor,
                num_nodes: Optional[int] = None,
                num_partitions: Optional[int] = None,
                coords: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total MECP loss with SSC awareness.
        
        Args:
            X: Partition probability matrix (N, P)
            W: Mobility weight matrix (N, N)
            num_nodes: Number of nodes
            num_partitions: Number of partitions
            coords: Node coordinates (N, 2) for SSC distance computation
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with all loss components
        """
        if num_nodes is None:
            num_nodes = X.size(0)
        if num_partitions is None:
            num_partitions = X.size(1)
        
        # Standard losses
        cut_loss = self.edge_cut_loss(X, W)
        balance_loss = self.balance_loss(X, num_nodes, num_partitions)
        
        # Normalize cut loss
        if self.normalize_cut:
            total_edge_weight = W.sum() + 1e-8
            cut_loss = cut_loss / total_edge_weight
        
        # Entropy regularization
        entropy = -torch.sum(X * torch.log(X + 1e-8)) / num_nodes
        
        # SSC-aware cost
        ssc_loss, ssc_dict = self.ssc_cost(X, W, coords)
        
        # Normalize SSC loss
        total_edge_weight = W.sum() + 1e-8
        ssc_loss_norm = ssc_loss / total_edge_weight
        
        # Combined loss  
        total_loss = (self.alpha * cut_loss 
                      + self.beta * balance_loss 
                      + self.gamma * entropy
                      + self.delta * ssc_loss_norm)
        
        loss_dict = {
            'total_loss': total_loss,
            'cut_loss': cut_loss,
            'balance_loss': balance_loss,
            'entropy': entropy,
            'ssc_loss': ssc_loss_norm,
            'weighted_cut_loss': self.alpha * cut_loss,
            'weighted_balance_loss': self.beta * balance_loss,
            'weighted_entropy': self.gamma * entropy,
            'weighted_ssc_loss': self.delta * ssc_loss_norm,
            **{k: v for k, v in ssc_dict.items()}
        }
        
        return total_loss, loss_dict
