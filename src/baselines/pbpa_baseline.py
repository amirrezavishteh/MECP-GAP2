"""
Partition-Based Placement Algorithm (PBPA) using Deep Reinforcement Learning

PBPA represents the modern "AI competitor" in the MEC placement landscape.
It leverages Deep Reinforcement Learning (DRL) to learn an optimal placement
policy through interaction with a simulated network environment.

Technical Foundation: PPO-based DRL for MEC Server Placement
=============================================================

1. AGENT-ENVIRONMENT INTERACTION (MDP):
   - State Space (S): Current load on each MEC server, geographical
     distribution of users (BS-level aggregation), current server layout.
     Transformed into heat maps / grayscale maps for CNNs.
   - Action Space (A): Reconfiguration of MEC server layout. Move a BS
     from one partition to another. For N nodes, P partitions -> N*P actions.
   - Reward Function (R): R = alpha * R_L - beta * P_B - gamma * P_M
     where R_L = latency reward, P_B = load balance penalty,
     P_M = migration cost penalty.

2. PPO (Proximal Policy Optimization):
   - Robust policy gradient method
   - Clip objective: L^CLIP = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
   - Uses actor-critic architecture

3. KNOWN FAILURE MODES:
   - Oscillation / Jitter: Agent "chases" optimal placement, causing
     frequent service migrations
   - Scalability bottleneck: Training for large graphs (2000+ nodes)
     requires massive compute
   - Sensitivity to reward shaping

Paper Reference: Section on PBPA baseline comparisons
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
import time
import warnings
from collections import deque


@dataclass
class PBPAConfig:
    """Configuration for PBPA Deep RL agent.
    
    Attributes:
        num_partitions: Number of MEC servers (partitions)
        hidden_dim: Hidden layer dimension for actor-critic networks
        gamma_rl: RL discount factor (not related to gravity model gamma)
        lr: Learning rate for PPO
        ppo_epochs: Number of PPO update epochs per batch
        clip_epsilon: PPO clipping parameter
        value_coef: Coefficient for value loss in combined loss
        entropy_coef: Coefficient for entropy bonus (exploration)
        max_episodes: Maximum training episodes
        max_steps_per_episode: Max actions per episode
        batch_size: Mini-batch size for PPO updates
        alpha_reward: Weight for latency reward
        beta_reward: Weight for load balance penalty
        gamma_reward: Weight for migration cost penalty
    """
    num_partitions: int = 4
    hidden_dim: int = 256
    gamma_rl: float = 0.99
    lr: float = 3e-4
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_episodes: int = 500
    max_steps_per_episode: int = 50
    batch_size: int = 64
    alpha_reward: float = 1.0
    beta_reward: float = 0.5
    gamma_reward: float = 0.3
    seed: Optional[int] = None


class MECEnvironment:
    """
    MEC Server Placement Environment (MDP).
    
    Models the 5G network as an environment where the agent can
    reassign base stations to different MEC servers (partitions).
    
    State: Flattened representation of:
        - Current partition assignment (one-hot encoded per node)
        - Node features (coordinates)
        - Per-partition load statistics
        
    Action: (node_index, target_partition) - move a node to a new partition.
        Encoded as a single integer: action = node * num_partitions + partition
        
    Reward: R = alpha * R_L - beta * P_B - gamma * P_M
    """
    
    def __init__(self, W_matrix: np.ndarray, coords: np.ndarray,
                 num_partitions: int, config: PBPAConfig):
        """
        Initialize the MEC environment.
        
        Args:
            W_matrix: Weighted adjacency matrix (N, N)
            coords: Node coordinates (N, 2)
            num_partitions: Number of partitions
            config: PBPAConfig with reward weights
        """
        self.W_matrix = W_matrix
        self.coords = coords
        self.num_nodes = W_matrix.shape[0]
        self.num_partitions = num_partitions
        self.config = config
        
        # Precompute total edge weight for normalization
        self.total_weight = np.sum(W_matrix) / 2
        if self.total_weight == 0:
            self.total_weight = 1.0
        
        # Precompute degree for each node
        self.degrees = np.sum(W_matrix, axis=1)
        
        # State dimensions
        # Node features (coords) + partition assignment one-hot + partition loads
        self.state_dim = self.num_nodes * 2 + self.num_nodes * num_partitions + num_partitions
        
        # Action space: select a node to move and target partition
        # Simplified: select from num_nodes * num_partitions discrete actions
        self.action_dim = self.num_nodes * num_partitions
        
        # Current state
        self.assignments = None
        self.prev_assignments = None
        self.step_count = 0
        
    def reset(self, initial_assignments: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Args:
            initial_assignments: Optional initial partition assignments.
                If None, uses balanced random assignment.
                
        Returns:
            state: Initial state vector
        """
        if initial_assignments is not None:
            self.assignments = initial_assignments.copy()
        else:
            # Balanced random initialization
            self.assignments = np.zeros(self.num_nodes, dtype=int)
            indices = np.random.permutation(self.num_nodes)
            for i, idx in enumerate(indices):
                self.assignments[idx] = i % self.num_partitions
        
        self.prev_assignments = self.assignments.copy()
        self.step_count = 0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Construct state representation.
        
        State = [node_coords_flat, assignment_one_hot_flat, partition_loads]
        
        The paper suggests transforming network states into heat maps,
        but we use a flattened vector representation for simplicity
        with smaller graphs. For larger graphs (2000+ nodes), a CNN-based
        approach with heat maps would be more appropriate.
        
        Returns:
            state: Flattened state vector
        """
        # Node coordinates (N*2)
        coords_flat = self.coords.flatten()
        
        # One-hot encoded partition assignments (N*P)
        one_hot = np.zeros((self.num_nodes, self.num_partitions))
        one_hot[np.arange(self.num_nodes), self.assignments] = 1.0
        assignment_flat = one_hot.flatten()
        
        # Partition load statistics (P) - fraction of total degree in each partition
        partition_loads = np.zeros(self.num_partitions)
        for p in range(self.num_partitions):
            mask = self.assignments == p
            partition_loads[p] = np.sum(self.degrees[mask])
        total_degree = np.sum(self.degrees)
        if total_degree > 0:
            partition_loads /= total_degree
        
        state = np.concatenate([coords_flat, assignment_flat, partition_loads])
        return state.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action in the environment.
        
        Action = node_index * num_partitions + target_partition
        This moves the selected node to the target partition.
        
        Args:
            action: Encoded action (integer)
            
        Returns:
            next_state: New state after action
            reward: Reward signal
            done: Whether episode is over
            info: Additional information
        """
        # Decode action
        node_idx = action // self.num_partitions
        target_partition = action % self.num_partitions
        
        # Clamp node index
        node_idx = min(node_idx, self.num_nodes - 1)
        
        # Store previous assignment for migration cost
        self.prev_assignments = self.assignments.copy()
        old_partition = self.assignments[node_idx]
        
        # Execute action: move node to target partition
        self.assignments[node_idx] = target_partition
        
        # Compute reward
        reward = self._compute_reward(node_idx, old_partition, target_partition)
        
        self.step_count += 1
        done = self.step_count >= self.config.max_steps_per_episode
        
        # Check for degenerate partitions (all nodes in one partition)
        partition_sizes = np.bincount(self.assignments, minlength=self.num_partitions)
        if np.any(partition_sizes == 0):
            reward -= 10.0  # Heavy penalty for empty partitions
        
        next_state = self._get_state()
        
        info = {
            'edge_cut': self._compute_edge_cut(),
            'balance_var': self._compute_balance_var(),
            'migration_cost': 1 if old_partition != target_partition else 0,
            'partition_sizes': partition_sizes.tolist(),
            'node_moved': node_idx,
            'from_partition': old_partition,
            'to_partition': target_partition
        }
        
        return next_state, reward, done, info
    
    def _compute_reward(self, node_idx: int, old_partition: int, 
                        target_partition: int) -> float:
        """
        Compute reward: R = alpha * R_L - beta * P_B - gamma * P_M
        
        R_L (Latency Reward): Based on edge cut reduction
        P_B (Balance Penalty): Based on load balance variance
        P_M (Migration Penalty): Cost of moving a service
        
        Args:
            node_idx: Index of moved node
            old_partition: Previous partition of node
            target_partition: New partition of node
            
        Returns:
            reward: Combined reward signal
        """
        # R_L: Latency reward (negative edge cut, normalized)
        edge_cut = self._compute_edge_cut()
        latency_reward = -edge_cut / self.total_weight
        
        # P_B: Balance penalty (variance of partition sizes from ideal)
        balance_var = self._compute_balance_var()
        ideal_size = self.num_nodes / self.num_partitions
        balance_penalty = balance_var / (ideal_size ** 2 + 1e-8)
        
        # P_M: Migration cost penalty
        migration_cost = 1.0 if old_partition != target_partition else 0.0
        
        # Combined reward
        reward = (self.config.alpha_reward * latency_reward
                  - self.config.beta_reward * balance_penalty
                  - self.config.gamma_reward * migration_cost)
        
        return reward
    
    def _compute_edge_cut(self) -> float:
        """Compute total weight of edges between different partitions."""
        diff_partition = self.assignments[:, None] != self.assignments[None, :]
        return float(np.sum(np.triu(self.W_matrix * diff_partition)))
    
    def _compute_balance_var(self) -> float:
        """Compute variance of partition sizes from ideal."""
        partition_sizes = np.bincount(self.assignments, minlength=self.num_partitions)
        ideal_size = self.num_nodes / self.num_partitions
        return float(np.sum((partition_sizes - ideal_size) ** 2))


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network for PPO.
    
    The actor (policy network) outputs action probabilities.
    The critic (value network) estimates state values.
    
    For small graphs, we use MLPs. The paper suggests using CNNs with
    heat map representations for larger graphs (2000+ nodes).
    
    Architecture:
        Shared backbone -> Actor head (policy) + Critic head (value)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize Actor-Critic network.
        
        Args:
            state_dim: Dimension of state vector
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy): outputs action logits
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic head (value): outputs state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        
        # Smaller initial weights for actor output (more uniform initial policy)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: State tensor (batch_size, state_dim)
            
        Returns:
            action_logits: Logits for action distribution (batch_size, action_dim)
            value: State value estimate (batch_size, 1)
        """
        features = self.shared(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action(self, state: torch.Tensor) -> Tuple[int, float, float]:
        """
        Sample action from policy.
        
        Args:
            state: State tensor (1, state_dim) or (state_dim,)
            
        Returns:
            action: Sampled action index
            log_prob: Log probability of sampled action
            value: Estimated state value
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        action_logits, value = self.forward(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def evaluate_actions(self, states: torch.Tensor, 
                         actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Args:
            states: Batch of states (batch_size, state_dim)
            actions: Batch of actions (batch_size,)
            
        Returns:
            log_probs: Log probabilities of actions
            values: State values
            entropy: Policy entropy
        """
        action_logits, values = self.forward(states)
        dist = Categorical(logits=action_logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy


class PPOBuffer:
    """
    Experience buffer for PPO training.
    
    Stores trajectories and computes returns and advantages
    using Generalized Advantage Estimation (GAE).
    """
    
    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        """
        Args:
            gamma: Discount factor
            lam: GAE lambda parameter
        """
        self.gamma = gamma
        self.lam = lam
        self.clear()
    
    def clear(self):
        """Clear buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def add(self, state: np.ndarray, action: int, reward: float,
            log_prob: float, value: float, done: bool):
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns_and_advantages(self, last_value: float = 0.0):
        """
        Compute discounted returns and GAE advantages.
        
        Uses Generalized Advantage Estimation:
        A_t = sum_{l=0}^{T-t} (gamma * lam)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        
        Args:
            last_value: Value estimate of the state after the last step
            
        Returns:
            returns: Discounted returns
            advantages: GAE advantages 
        """
        n = len(self.rewards)
        returns = np.zeros(n)
        advantages = np.zeros(n)
        
        last_gae = 0.0
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            
            # TD error
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            
            # GAE
            last_gae = delta + self.gamma * self.lam * (1 - self.dones[t]) * last_gae
            advantages[t] = last_gae
        
        returns = advantages + np.array(self.values)
        
        return returns, advantages
    
    def get_batches(self, batch_size: int, device: str = 'cpu'):
        """
        Get mini-batches for PPO update.
        
        Returns:
            Iterator over mini-batches of (states, actions, old_log_probs,
                                           returns, advantages)
        """
        returns, advantages = self.compute_returns_and_advantages()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(self.actions), dtype=torch.long, device=device)
        old_log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        
        n = len(self.states)
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            
            yield (states[batch_idx], actions[batch_idx],
                   old_log_probs[batch_idx], returns_t[batch_idx],
                   advantages_t[batch_idx])


class PBPAPartitioner:
    """
    PBPA (Partition-Based Placement Algorithm) using PPO.
    
    The agent learns to reassign base stations to MEC servers by
    interacting with the MECEnvironment. The policy is trained using
    Proximal Policy Optimization (PPO).
    
    Known Issues (documented in paper):
    - Oscillation: Agent may constantly reconfigure placement
    - High training overhead: Slow compared to one-shot methods
    - Potential instability with sparse rewards
    
    Performance Characteristics:
        MetricValue
        Running Time (Inference): Fast
        Running Time (Training): Very Slow
        Edge Cut Quality: Fair
        Load Balance: Fair/Good
        Mobility Awareness: High (Learned)
        System Stability: Low (High Jitter risk)
    """
    
    def __init__(self, num_partitions: int, 
                 config: Optional[PBPAConfig] = None,
                 device: Optional[str] = None):
        """
        Initialize PBPA partitioner.
        
        Args:
            num_partitions: Number of partitions (MEC servers)
            config: PBPAConfig with hyperparameters
            device: 'cpu' or 'cuda'
        """
        self.num_partitions = num_partitions
        self.config = config or PBPAConfig(num_partitions=num_partitions)
        self.config.num_partitions = num_partitions
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
        
        self._model = None
        self._env = None
        self._assignments = None
        self._runtime = None
        self._training_history = []
    
    def fit(self, W_matrix: np.ndarray, 
            coords: Optional[np.ndarray] = None,
            verbose: bool = True) -> np.ndarray:
        """
        Train PBPA agent and produce partition assignments.
        
        This is the main entry point. It:
        1. Creates the MEC environment
        2. Initializes the actor-critic network
        3. Trains using PPO for max_episodes
        4. Returns the best partition found during training
        
        Args:
            W_matrix: Weighted adjacency matrix (N, N)
            coords: Node coordinates (N, 2). If None, uses random features.
            verbose: Whether to print training progress
            
        Returns:
            assignments: Best partition assignments found (N,)
        """
        start_time = time.time()
        
        num_nodes = W_matrix.shape[0]
        
        # Use coordinates or random features
        if coords is None:
            coords = np.random.randn(num_nodes, 2).astype(np.float32)
        
        # Normalize coordinates
        coords_norm = coords.copy().astype(np.float32)
        if coords_norm.max() > 0:
            coords_norm = coords_norm / coords_norm.max()
        
        # Create environment
        self._env = MECEnvironment(W_matrix, coords_norm, self.num_partitions, self.config)
        
        # Limit action and state space for tractability
        # For large graphs, we limit the action space by only considering
        # boundary nodes (nodes adjacent to other partitions)
        effective_action_dim = min(self._env.action_dim, num_nodes * self.num_partitions)
        
        # Initialize actor-critic network
        self._model = ActorCritic(
            state_dim=self._env.state_dim,
            action_dim=effective_action_dim,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)
        
        optimizer = optim.Adam(self._model.parameters(), lr=self.config.lr, eps=1e-5)
        
        # Training loop
        buffer = PPOBuffer(gamma=self.config.gamma_rl)
        
        best_assignments = None
        best_reward = float('-inf')
        best_edge_cut = float('inf')
        
        episode_rewards = deque(maxlen=100)
        
        if verbose:
            print(f"--- PBPA Training ({self.config.max_episodes} Episodes) ---")
            print(f"Nodes: {num_nodes} | Partitions: {self.num_partitions} | Device: {self.device}")
            print(f"State dim: {self._env.state_dim} | Action dim: {effective_action_dim}")
            print("-" * 55)
        
        for episode in range(self.config.max_episodes):
            state = self._env.reset()
            episode_reward = 0.0
            
            for step in range(self.config.max_steps_per_episode):
                # Get action from policy
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                
                with torch.no_grad():
                    action, log_prob, value = self._model.get_action(state_tensor)
                
                # Clamp action to valid range
                action = action % effective_action_dim
                
                # Step environment
                next_state, reward, done, info = self._env.step(action)
                
                # Store transition
                buffer.add(state, action, reward, log_prob, value, done)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            # Track best partition found
            current_cut = self._env._compute_edge_cut()
            current_balance = self._env._compute_balance_var()
            
            # Prefer lower edge cut with reasonable balance
            current_score = -current_cut - 0.1 * current_balance
            if current_score > best_reward:
                best_reward = current_score
                best_edge_cut = current_cut
                best_assignments = self._env.assignments.copy()
            
            # Record history
            self._training_history.append({
                'episode': episode,
                'reward': episode_reward,
                'edge_cut': current_cut,
                'balance_var': current_balance,
                'partition_sizes': np.bincount(
                    self._env.assignments, minlength=self.num_partitions
                ).tolist()
            })
            
            # PPO Update (every episode for simplicity)
            if len(buffer.states) >= self.config.batch_size:
                self._ppo_update(optimizer, buffer)
                buffer.clear()
            
            # Logging
            if verbose and (episode + 1) % max(1, self.config.max_episodes // 10) == 0:
                avg_reward = np.mean(episode_rewards)
                print(f"Episode {episode+1:4d} | Avg Reward: {avg_reward:.4f} | "
                      f"Edge Cut: {current_cut:.2f} | "
                      f"Balance: {current_balance:.2f} | "
                      f"Sizes: {np.bincount(self._env.assignments, minlength=self.num_partitions).tolist()}")
        
        self._assignments = best_assignments
        self._runtime = time.time() - start_time
        
        if verbose:
            print("-" * 55)
            print(f"Training complete in {self._runtime:.2f}s")
            print(f"Best edge cut: {best_edge_cut:.2f}")
            final_sizes = np.bincount(best_assignments, minlength=self.num_partitions)
            print(f"Best partition sizes: {final_sizes.tolist()}")
        
        return best_assignments
    
    def _ppo_update(self, optimizer: optim.Adam, buffer: PPOBuffer):
        """
        Perform PPO update on the actor-critic network.
        
        Implements the PPO-Clip objective:
        L^CLIP = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
        
        Combined loss = -L^CLIP + value_coef * L^VF - entropy_coef * H
        
        Args:
            optimizer: Adam optimizer
            buffer: PPOBuffer with collected trajectories
        """
        for _ in range(self.config.ppo_epochs):
            for batch in buffer.get_batches(self.config.batch_size, self.device):
                states, actions, old_log_probs, returns, advantages = batch
                
                # Evaluate current policy
                new_log_probs, values, entropy = self._model.evaluate_actions(states, actions)
                
                # PPO ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 
                                    1 - self.config.clip_epsilon, 
                                    1 + self.config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, returns)
                
                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()
                
                # Combined loss
                total_loss = (policy_loss 
                              + self.config.value_coef * value_loss 
                              + self.config.entropy_coef * entropy_loss)
                
                # Update
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 0.5)
                optimizer.step()
    
    def inference(self, W_matrix: np.ndarray,
                  coords: Optional[np.ndarray] = None,
                  num_steps: int = 50) -> np.ndarray:
        """
        Run inference with trained model (no training).
        
        Uses the learned policy to iteratively improve an initial
        partition assignment.
        
        Args:
            W_matrix: Weight matrix
            coords: Node coordinates
            num_steps: Number of inference steps
            
        Returns:
            assignments: Partition assignments
        """
        if self._model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        num_nodes = W_matrix.shape[0]
        if coords is None:
            coords = np.random.randn(num_nodes, 2).astype(np.float32)
        
        coords_norm = coords.copy().astype(np.float32)
        if coords_norm.max() > 0:
            coords_norm = coords_norm / coords_norm.max()
        
        env = MECEnvironment(W_matrix, coords_norm, self.num_partitions, self.config)
        state = env.reset()
        
        self._model.eval()
        with torch.no_grad():
            for _ in range(num_steps):
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                action, _, _ = self._model.get_action(state_tensor)
                action = action % env.action_dim
                state, _, done, _ = env.step(action)
                if done:
                    break
        
        return env.assignments
    
    def get_assignments(self) -> np.ndarray:
        """Get best partition assignments found during training."""
        if self._assignments is None:
            raise ValueError("Must call fit() first")
        return self._assignments
    
    def get_runtime(self) -> float:
        """Get total training runtime in seconds."""
        if self._runtime is None:
            raise ValueError("Must call fit() first")
        return self._runtime
    
    def get_results(self) -> Dict[str, Any]:
        """Get all results."""
        return {
            'assignments': self._assignments,
            'runtime': self._runtime,
            'num_partitions': self.num_partitions,
            'training_history': self._training_history
        }
    
    def get_training_history(self) -> List[Dict]:
        """Get training history for plotting."""
        return self._training_history


def run_pbpa(
    W_matrix: np.ndarray,
    num_partitions: int,
    coords: Optional[np.ndarray] = None,
    max_episodes: int = 200,
    max_steps: int = 30,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to run PBPA partitioning.
    
    Args:
        W_matrix: Weighted adjacency matrix (N, N)
        num_partitions: Number of partitions k
        coords: Optional node coordinates
        max_episodes: Training episodes
        max_steps: Max steps per episode
        seed: Random seed
        verbose: Print progress
        
    Returns:
        assignments: Partition assignments (N,)
        results: Dictionary with additional results
    """
    config = PBPAConfig(
        num_partitions=num_partitions,
        max_episodes=max_episodes,
        max_steps_per_episode=max_steps,
        seed=seed
    )
    
    partitioner = PBPAPartitioner(
        num_partitions=num_partitions,
        config=config
    )
    
    assignments = partitioner.fit(W_matrix, coords, verbose=verbose)
    results = partitioner.get_results()
    
    return assignments, results


def run_pbpa_light(
    W_matrix: np.ndarray,
    num_partitions: int,
    coords: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Lightweight version of PBPA with reduced training for quick benchmarking.
    
    Uses fewer episodes and smaller network for faster execution.
    
    Args:
        W_matrix: Weighted adjacency matrix (N, N)
        num_partitions: Number of partitions k
        coords: Optional node coordinates
        seed: Random seed
        verbose: Print progress
        
    Returns:
        assignments: Partition assignments (N,)
        results: Dictionary with results
    """
    config = PBPAConfig(
        num_partitions=num_partitions,
        hidden_dim=128,
        max_episodes=100,
        max_steps_per_episode=20,
        ppo_epochs=2,
        batch_size=32,
        seed=seed
    )
    
    partitioner = PBPAPartitioner(
        num_partitions=num_partitions,
        config=config
    )
    
    assignments = partitioner.fit(W_matrix, coords, verbose=verbose)
    results = partitioner.get_results()
    
    return assignments, results
