"""
MECP-GAP: Mobility-Aware MEC Planning with GNN-Based Graph Partitioning
========================================================================

Self-contained script that reproduces the paper's results:
  "Mobility-Aware MEC Planning With a GNN-Based Graph Partitioning Framework"
  IEEE Transactions on Network and Service Management, Vol. 21, No. 4, Aug 2024

Requirements: pip install torch networkx matplotlib scipy numpy

NO DGL required — uses a custom GraphSAGE with weighted aggregation.

Run:  python main.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import time

# ============================================================================
# PART 1: DATA GENERATION — Synthetic 5G Network (Paper Section V.A)
# ============================================================================
class NetworkGenerator:
    """
    Generates a realistic 5G network graph:
      1. Geometry  — Poisson Point Process (random base-station locations)
      2. Topology  — Voronoi diagram (neighbour relationships)
      3. Mobility  — Gravity Model (handover traffic weights)
    """
    def __init__(self, num_nodes=200, area_size=10.0, gamma=1.5,
                 weight_scale=100.0, seed=42):
        self.num_nodes = num_nodes
        self.area_size = area_size
        self.gamma = gamma
        self.weight_scale = weight_scale
        if seed is not None:
            np.random.seed(seed)

    def generate(self):
        N = self.num_nodes
        print(f"Generating synthetic 5G network with {N} nodes …")

        # 1. Random base-station locations (Poisson Point Process)
        coords = np.random.uniform(0, self.area_size, (N, 2))

        # 2. Voronoi topology — adjacent cells share a boundary
        vor = Voronoi(coords)
        adj = np.zeros((N, N))
        for p1, p2 in vor.ridge_points:
            if 0 <= p1 < N and 0 <= p2 < N:
                adj[p1, p2] = 1
                adj[p2, p1] = 1

        # Node "mass" ≈ coverage area (avg distance to neighbours squared)
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i, j]:
                    G.add_edge(i, j)
        masses = {}
        for n in G.nodes():
            nbrs = list(G.neighbors(n))
            if nbrs:
                avg_d = np.mean([np.linalg.norm(coords[n] - coords[nb])
                                 for nb in nbrs])
                masses[n] = avg_d ** 2
            else:
                masses[n] = 0.1

        # 3. Gravity Model weights: W_ij = scale * M_i * M_j / dist^gamma
        W = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i, j]:
                    dist = max(np.linalg.norm(coords[i] - coords[j]), 0.01)
                    w = self.weight_scale * masses[i] * masses[j] / (dist ** self.gamma)
                    W[i, j] = w
                    W[j, i] = w

        num_edges = int(adj.sum()) // 2
        print(f"  → {N} nodes, {num_edges} edges, "
              f"total weight = {W.sum():.0f}")
        return coords, W


# ============================================================================
# PART 2: THE MODEL — Custom GraphSAGE + MLP  (Paper Section IV.B)
# ============================================================================
class GraphSAGELayer(nn.Module):
    """GraphSAGE with edge-weight–aware mean aggregation (Paper Eq. 12-13)."""
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.W_self  = nn.Linear(in_feats, out_feats, bias=False)
        self.W_neigh = nn.Linear(in_feats, out_feats, bias=False)
        self.bias    = nn.Parameter(torch.zeros(out_feats))
        self._init()

    def _init(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.W_neigh.weight, gain=gain)

    def forward(self, x, edge_index, edge_weight):
        src, dst = edge_index                       # src→dst messages
        N, D = x.size()

        # weighted neighbour features
        w = edge_weight.unsqueeze(-1)               # (E, 1)
        msg = x[src] * w                            # (E, D)

        agg = torch.zeros(N, D, device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)

        w_sum = torch.zeros(N, device=x.device)
        w_sum.scatter_add_(0, dst, edge_weight)
        w_sum = w_sum.clamp(min=1e-8)

        neigh = agg / w_sum.unsqueeze(-1)           # weighted mean
        out = self.W_self(x) + self.W_neigh(neigh) + self.bias
        return F.normalize(out, p=2, dim=-1)


class MECP_GAP(nn.Module):
    """
    I-GAP model from the paper:
      2-layer GraphSAGE (→128-d embeddings)  +  MLP (128→64→P)  +  Softmax
    """
    def __init__(self, in_feats, hidden=128, num_partitions=4):
        super().__init__()
        self.sage1 = GraphSAGELayer(in_feats, hidden)
        self.sage2 = GraphSAGELayer(hidden, hidden)
        self.mlp   = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, num_partitions),
        )
        # Xavier init for MLP
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_weight):
        h = F.relu(self.sage1(x, edge_index, edge_weight))
        h = F.relu(self.sage2(h, edge_index, edge_weight))
        h = F.normalize(h, p=2, dim=-1)
        return F.softmax(self.mlp(h), dim=-1)


# ============================================================================
# PART 3: LOSS FUNCTION  (Paper Equations 7-9)
# ============================================================================
def mecp_loss(X, W, N, P, alpha, beta=1.0):
    """
    Eq. 9:  L = α · L_cut  +  β · L_balance

    L_cut     = Σ_{i,j} W_ij · (1 − X Xᵀ)_ij          (Eq. 7)
    L_balance = Σ_k (|S_k| − N/P)²                       (Eq. 8)
    """
    P_same    = X @ X.t()
    cut_loss  = torch.sum((1 - P_same) * W)

    sizes     = X.sum(dim=0)
    ideal     = N / P
    bal_loss  = ((sizes - ideal) ** 2).sum()

    total = alpha * cut_loss + beta * bal_loss
    return total, cut_loss, bal_loss


# ============================================================================
# PART 4: DATA PREPARATION
# ============================================================================
def prepare_data(coords, W_matrix, device='cpu'):
    """
    KEY FIX vs. naïve implementations:
    The paper states "input features with dimensionality of 200" for 200 nodes.
    → Node features = row-normalised W[i,:], NOT 2-D coordinates.
    This gives every node a unique connectivity fingerprint (dim = N),
    breaking the symmetry that traps 2-D-feature models in uniform softmax.
    """
    N = len(coords)

    # Row-normalised weight matrix as node features  (N × N)
    row_sums = W_matrix.sum(axis=1, keepdims=True) + 1e-8
    feat_np  = W_matrix / row_sums
    features = torch.tensor(feat_np, dtype=torch.float32, device=device)

    W_t = torch.tensor(W_matrix, dtype=torch.float32, device=device)

    # Sparse edge list + weights
    src, dst = np.nonzero(W_matrix)
    edge_index = torch.tensor(np.stack([src, dst]),
                              dtype=torch.long, device=device)
    edge_weight = torch.tensor(W_matrix[src, dst],
                               dtype=torch.float32, device=device)

    # Self-loops (GraphSAGE needs them)
    self_idx = torch.arange(N, dtype=torch.long, device=device)
    edge_index  = torch.cat([edge_index,
                             torch.stack([self_idx, self_idx])], dim=1)
    edge_weight = torch.cat([edge_weight,
                             torch.ones(N, device=device)])

    return features, edge_index, edge_weight, W_t


# ============================================================================
# PART 5: TRAINING LOOP + VISUALISATION
# ============================================================================
def run_experiment(num_nodes=200, num_partitions=4, epochs=500, seed=42):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)

    # ---- Data ----
    gen = NetworkGenerator(num_nodes, seed=seed)
    coords, W_matrix = gen.generate()
    features, edge_index, ew, W_t = prepare_data(coords, W_matrix, device)

    N, P = num_nodes, num_partitions
    in_feats = features.shape[1]           # = N (weight-row features)

    # ---- Auto-alpha  (scale cut ≈ balance at initialisation) ----
    total_w = float(W_t.sum())
    alpha = (N ** 2 / P) / ((1 - 1.0 / P) * total_w + 1e-8)
    print(f"Auto α = {alpha:.6f}  (Σ W = {total_w:.0f})")

    # ---- Model ----
    model = MECP_GAP(in_feats, hidden=128, num_partitions=P).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01)

    loss_hist, cut_hist, bal_hist, size_hist = [], [], [], []

    # ---- Training ----
    print(f"\nTraining I-GAP  |  {N} nodes → {P} partitions  |  "
          f"features dim={in_feats}  |  device={device}")
    print("-" * 65)
    t0 = time.time()

    for ep in range(epochs + 1):
        model.train()
        probs = model(features, edge_index, ew)
        loss, cut, bal = mecp_loss(probs, W_t, N, P, alpha)

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            assign = probs.argmax(dim=1)
            sizes  = torch.bincount(assign, minlength=P).tolist()

        loss_hist.append(loss.item())
        cut_hist.append(cut.item())
        bal_hist.append(bal.item())
        size_hist.append(sizes)

        if ep % 50 == 0:
            print(f"Epoch {ep:03d} | Loss {loss.item():10.1f} "
                  f"(Cut {cut.item():10.1f}  Bal {bal.item():7.1f}) "
                  f"| Sizes {sizes}")

    elapsed = time.time() - t0
    print("-" * 65)
    print(f"Done in {elapsed:.2f} s")

    # ---- Final inference ----
    model.eval()
    with torch.no_grad():
        final_probs = model(features, edge_index, ew)
        assignments = final_probs.argmax(dim=1).cpu().numpy()

    final_sizes = np.bincount(assignments, minlength=P).tolist()
    print(f"Final partition sizes: {final_sizes}")

    # ---- Metrics ----
    cut_w, total_edges, cut_edges = 0.0, 0, 0
    rows, cols = np.nonzero(W_matrix)
    for u, v in zip(rows, cols):
        if u < v:
            total_edges += 1
            if assignments[u] != assignments[v]:
                cut_w += W_matrix[u, v]
                cut_edges += 1
    total_w_half = W_matrix.sum() / 2
    print(f"Edge-cut ratio : {100 * cut_w / total_w_half:.2f}%  "
          f"({cut_edges}/{total_edges} edges)")
    print(f"Size std dev   : {np.std(final_sizes):.2f}")

    # ---- Plot ----
    plot_results(coords, W_matrix, assignments, loss_hist, cut_hist,
                 bal_hist, size_hist, P)

    return assignments


# ============================================================================
# PART 6: PLOTTING — Network Map + Convergence Curves
# ============================================================================
COLORS = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99',
          '#D1C4E9', '#F0F4C3', '#FFAB91', '#B2EBF2']


def plot_results(coords, W, assignments, loss_h, cut_h, bal_h, size_h, P):
    fig = plt.figure(figsize=(18, 10))

    # ---------- Top-left: Network map ----------
    ax1 = fig.add_subplot(2, 2, 1)
    G = nx.Graph()
    N = len(coords)
    rows, cols = np.nonzero(W)
    for u, v in zip(rows, cols):
        if u < v:
            G.add_edge(u, v, weight=W[u, v])
    pos = {i: coords[i] for i in range(N)}

    edges = list(G.edges())
    weights = [G[u][v]['weight'] for u, v in edges]
    max_w = max(weights) if weights else 1
    widths = [0.15 + (w / max_w) * 2.5 for w in weights]

    # Separate cut / internal edges
    cut_e, int_e, cut_ew, int_ew = [], [], [], []
    for (u, v), ew in zip(edges, widths):
        if assignments[u] != assignments[v]:
            cut_e.append((u, v)); cut_ew.append(ew)
        else:
            int_e.append((u, v)); int_ew.append(ew)

    node_c = [COLORS[assignments[i] % len(COLORS)] for i in range(N)]
    nx.draw_networkx_edges(G, pos, edgelist=int_e, width=int_ew,
                           edge_color='gray', alpha=0.35, ax=ax1)
    nx.draw_networkx_edges(G, pos, edgelist=cut_e, width=cut_ew,
                           edge_color='red', alpha=0.55, style='dashed', ax=ax1)
    nx.draw_networkx_nodes(G, pos, node_color=node_c, node_size=55,
                           edgecolors='black', linewidths=0.4, ax=ax1)

    for p in range(P):
        cnt = int(np.sum(assignments == p))
        ax1.scatter([], [], c=COLORS[p % len(COLORS)], s=80,
                    label=f'Partition {p} ({cnt})', edgecolors='k', linewidths=0.4)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_title(f"MECP-GAP Result: {P} MEC Clusters", fontsize=12)
    ax1.axis('off')

    # ---------- Top-right: Total loss ----------
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(loss_h, 'b-', lw=1.5)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Total Loss')
    ax2.set_title('Training Convergence'); ax2.grid(True, alpha=0.3)

    # ---------- Bottom-left: Cut vs Balance ----------
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(cut_h, 'r-', lw=1.5, label='Cut Loss (raw)')
    ax3b = ax3.twinx()
    ax3b.plot(bal_h, 'g-', lw=1.5, label='Balance Loss')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('Cut Loss', color='r')
    ax3b.set_ylabel('Balance Loss', color='g')
    ax3.set_title('Loss Components')
    ax3.grid(True, alpha=0.3)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    # ---------- Bottom-right: Partition sizes ----------
    ax4 = fig.add_subplot(2, 2, 4)
    sizes_arr = np.array(size_h)
    for p in range(P):
        ax4.plot(sizes_arr[:, p], lw=1.5, color=COLORS[p % len(COLORS)],
                 label=f'Part {p}')
    ideal = N / P
    ax4.axhline(ideal, ls='--', color='black', alpha=0.6,
                label=f'Ideal ({ideal:.0f})')
    ax4.set_xlabel('Epoch'); ax4.set_ylabel('# Nodes')
    ax4.set_title('Partition Size Balance'); ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mecp_gap_result.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to mecp_gap_result.png")
    plt.show()


# ============================================================================
if __name__ == "__main__":
    run_experiment(
        num_nodes=200,
        num_partitions=4,
        epochs=500,
        seed=42,
    )
