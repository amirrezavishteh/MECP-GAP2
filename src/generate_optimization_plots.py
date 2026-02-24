"""
Generate comparison plots: Baseline (old) vs Optimized (new) MECP-GAP results.

This script:
1. Runs the baseline model (beta=1.0, weight_row features, no entropy, no KL refinement)
2. Runs the optimized model (beta=2.0, hybrid features, entropy=-0.1, KL refinement)
3. Generates side-by-side comparison plots saved to results/plots/
"""

import sys
import os
from pathlib import Path
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from data_generation.graph_generator import CellTowerGraphGenerator, GraphConfig
from training.trainer import MECPTrainer, TrainingConfig
from utils.utils import compute_metrics_vectorized, kl_refinement
from utils.visualization import compute_partition_metrics, print_partition_report

import torch

# ── Configuration ──────────────────────────────────────────────────────────
NUM_NODES = 200
NUM_PARTITIONS = 4
AREA_SIZE = 10.0
GAMMA = 1.5
SEED = 42
EPOCHS = 500
RESULTS_DIR = project_root / 'results'
PLOTS_DIR = RESULTS_DIR / 'plots'
METRICS_DIR = RESULTS_DIR / 'metrics'


def generate_data():
    """Generate the 200-node synthetic graph."""
    np.random.seed(SEED)
    config = GraphConfig(num_nodes=NUM_NODES, area_size=AREA_SIZE, gamma=GAMMA,
                         mode='synthetic', seed=SEED)
    gen = CellTowerGraphGenerator(config)
    coords, G, W = gen.generate()
    return coords, W


def run_variant(coords, W, label, *, beta, feature_type, model_type, gamma_ent, refine):
    """
    Train a single MECP-GAP variant and return metrics dict.
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    cfg = TrainingConfig(
        in_feats=-1,
        hidden_feats=128,
        num_partitions=NUM_PARTITIONS,
        num_layers=2,
        num_epochs=EPOCHS,
        learning_rate=0.01,
        alpha=-1.0,
        beta=beta,
        gamma=gamma_ent,
        feature_type=feature_type,
        model_type=model_type,
        log_interval=100,
        checkpoint_dir=None,
    )
    trainer = MECPTrainer(cfg)
    results = trainer.train(coords, W, verbose=True)

    assignments = results['final_assignments']
    history = results['history']

    # Optional KL refinement
    if refine:
        assignments = kl_refinement(assignments, W, NUM_PARTITIONS,
                                    max_iterations=100, balance_tolerance=0.1,
                                    verbose=True)

    metrics = compute_partition_metrics(W, assignments, NUM_PARTITIONS)
    metrics['label'] = label
    metrics['training_time'] = results['training_time']
    metrics['final_loss'] = results['final_loss']
    metrics['history'] = history
    metrics['assignments'] = assignments
    return metrics


# ── Plotting helpers ───────────────────────────────────────────────────────

COLORS_OLD = '#3498db'   # blue
COLORS_NEW = '#e74c3c'   # red
COLORS_MID = '#2ecc71'   # green (for intermediate variants)
BAR_COLORS = ['#3498db', '#1abc9c', '#9b59b6', '#e67e22', '#e74c3c']


def plot_bar_comparison(variants, metric_key, ylabel, title, fname, pct=False):
    """Bar chart comparing a single metric across variants."""
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [v['label'] for v in variants]
    values = [v[metric_key] * (100 if pct else 1) for v in variants]
    bars = ax.bar(labels, values, color=BAR_COLORS[:len(variants)], edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, values):
        fmt = f'{val:.2f}%' if pct else f'{val:.2f}'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                fmt, ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_dual_bar(variants, fname):
    """Grouped bar chart: Edge Cut Ratio + Balance Std side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    labels = [v['label'] for v in variants]
    x = np.arange(len(labels))
    width = 0.55

    # Edge Cut Ratio
    ecr = [v['edge_cut_ratio'] * 100 for v in variants]
    bars1 = ax1.bar(x, ecr, width, color=BAR_COLORS[:len(variants)], edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars1, ecr):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Edge Cut Ratio (%)', fontsize=12)
    ax1.set_title('Edge Cut Ratio (lower is better)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Balance Std
    bstd = [v['balance_std'] for v in variants]
    bars2 = ax2.bar(x, bstd, width, color=BAR_COLORS[:len(variants)], edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars2, bstd):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Partition Size Std Dev (σ)', fontsize=12)
    ax2.set_title('Load Balance σ (lower is better)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('MECP-GAP: Baseline vs Optimized (P = 4)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_training_curves(variants, fname):
    """Overlay training loss curves for each variant."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    colors = BAR_COLORS[:len(variants)]

    for v, c in zip(variants, colors):
        h = v['history']
        epochs = [d['epoch'] for d in h]
        total = [d['total_loss'] for d in h]
        cut = [d.get('weighted_cut_loss', 0) for d in h]
        bal = [d.get('weighted_balance_loss', 0) for d in h]

        ax1.plot(epochs, total, color=c, label=v['label'], linewidth=2)
        ax2.plot(epochs, cut, color=c, label=v['label'], linewidth=2)
        ax3.plot(epochs, bal, color=c, label=v['label'], linewidth=2)

    for ax, t in [(ax1, 'Total Loss'), (ax2, 'Weighted Cut Loss'), (ax3, 'Weighted Balance Loss')]:
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title(t, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Training Curves: Baseline vs Optimized', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_partition_comparison(coords, W, variants, fname):
    """Side-by-side partition visualizations."""
    import networkx as nx
    from utils.visualization import DEFAULT_COLORS

    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    G_vis = nx.Graph()
    G_vis.add_nodes_from(range(len(coords)))
    rows, cols = np.nonzero(W)
    for u, v in zip(rows, cols):
        if u < v:
            G_vis.add_edge(u, v, weight=W[u, v])
    pos = {i: coords[i] for i in range(len(coords))}
    edges = list(G_vis.edges())
    if edges:
        ws = [G_vis[u][v]['weight'] for u, v in edges]
        mx = max(ws)
        ew = [0.1 + (w / mx) * 2.0 for w in ws]
    else:
        ew = []

    for ax, v in zip(axes, variants):
        a = v['assignments']
        nc = [DEFAULT_COLORS[a[i] % len(DEFAULT_COLORS)] for i in range(len(coords))]
        cut_e = [(u, vv) for u, vv in edges if a[u] != a[vv]]
        int_e = [(u, vv) for u, vv in edges if a[u] == a[vv]]
        cut_w = [ew[edges.index(e)] for e in cut_e] if cut_e else []
        int_w = [ew[edges.index(e)] for e in int_e] if int_e else []

        if int_e:
            nx.draw_networkx_edges(G_vis, pos, edgelist=int_e, width=int_w,
                                   edge_color='gray', alpha=0.3, ax=ax)
        if cut_e:
            nx.draw_networkx_edges(G_vis, pos, edgelist=cut_e, width=cut_w,
                                   edge_color='red', alpha=0.5, style='dashed', ax=ax)
        nx.draw_networkx_nodes(G_vis, pos, node_color=nc, node_size=50,
                              edgecolors='black', linewidths=0.5, ax=ax)

        ecr = v['edge_cut_ratio'] * 100
        bstd = v['balance_std']
        ax.set_title(f"{v['label']}\nCut: {ecr:.2f}%  |  σ: {bstd:.2f}", fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Partition Comparison: Baseline → Optimized (P = 4)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_improvement_waterfall(variants, fname):
    """Waterfall chart showing incremental improvement from each optimization."""
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [v['label'] for v in variants]
    ecr = [v['edge_cut_ratio'] * 100 for v in variants]
    
    # Each bar starts where the previous one ended
    n = len(ecr)
    x = np.arange(n)
    
    # Draw bars showing the absolute ECR for each variant
    bars = ax.bar(x, ecr, color=BAR_COLORS[:n], edgecolor='black', linewidth=0.8, width=0.6)
    
    # Annotate
    for i, (bar, val) in enumerate(zip(bars, ecr)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        if i > 0:
            delta = ecr[i] - ecr[0]
            color = 'green' if delta < 0 else 'red'
            ax.annotate(f'{delta:+.2f}%', xy=(bar.get_x() + bar.get_width()/2, val/2),
                       ha='center', fontsize=10, color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha='right')
    ax.set_ylabel('Edge Cut Ratio (%)', fontsize=12)
    ax.set_title('Optimization Impact on Edge Cut Ratio (P = 4)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_radar_comparison(baseline, optimized, fname):
    """Radar/spider chart comparing baseline vs optimized across multiple dimensions."""
    categories = ['Edge Cut\n(inverted)', 'Load Balance\n(inverted)', 'Training Speed\n(inverted)', 
                   'Confidence\n(low entropy)', 'Cut Edges\n(inverted)']
    N_cat = len(categories)
    
    # Normalize metrics: higher is better (so invert where needed)
    max_ecr = max(baseline['edge_cut_ratio'], optimized['edge_cut_ratio']) * 1.2
    max_bstd = max(baseline['balance_std'], optimized['balance_std']) * 1.2
    max_time = max(baseline['training_time'], optimized['training_time']) * 1.2
    max_ce = max(baseline['num_cut_edges'], optimized['num_cut_edges']) * 1.2

    def score(v):
        return [
            1 - v['edge_cut_ratio'] / max_ecr,
            1 - v['balance_std'] / max_bstd,
            1 - v['training_time'] / max_time,
            0.8 if 'entropy' not in v.get('label', '') else 0.5,  # proxy
            1 - v['num_cut_edges'] / max_ce,
        ]

    vals_b = score(baseline)
    vals_o = score(optimized)

    angles = np.linspace(0, 2 * np.pi, N_cat, endpoint=False).tolist()
    vals_b += vals_b[:1]
    vals_o += vals_o[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, vals_b, 'o-', color=COLORS_OLD, linewidth=2, label='Baseline (β=1.0)')
    ax.fill(angles, vals_b, alpha=0.15, color=COLORS_OLD)
    ax.plot(angles, vals_o, 's-', color=COLORS_NEW, linewidth=2, label='Optimized (all)')
    ax.fill(angles, vals_o, alpha=0.15, color=COLORS_NEW)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Dimensional Quality: Baseline vs Optimized', fontsize=13, fontweight='bold', y=1.1)
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MECP-GAP: Advanced Optimization Comparison")
    print("=" * 70)

    # 1. Generate data
    print("\n[1/6] Generating synthetic 200-node graph...")
    coords, W = generate_data()
    print(f"     Graph: {len(coords)} nodes, {np.count_nonzero(W)//2} edges")

    # 2. Run variants
    print("\n[2/6] Training Baseline (β=1.0, weight_row, no entropy, no KL)...")
    v_baseline = run_variant(coords, W, "Baseline\n(β=1.0)",
                             beta=1.0, feature_type='weight_row', model_type='sage',
                             gamma_ent=0.0, refine=False)
    print_partition_report(W, v_baseline['assignments'], "Baseline", NUM_PARTITIONS)

    print("\n[3/6] Training β=2.0 variant...")
    v_beta2 = run_variant(coords, W, "β=2.0",
                          beta=2.0, feature_type='weight_row', model_type='sage',
                          gamma_ent=0.0, refine=False)
    print_partition_report(W, v_beta2['assignments'], "β=2.0", NUM_PARTITIONS)

    print("\n[4/6] Training Hybrid Features + Entropy variant...")
    v_hybrid = run_variant(coords, W, "Hybrid+Entropy\n(β=2.0, γ=-0.1)",
                           beta=2.0, feature_type='hybrid', model_type='sage',
                           gamma_ent=-0.1, refine=False)
    print_partition_report(W, v_hybrid['assignments'], "Hybrid+Entropy", NUM_PARTITIONS)

    print("\n[5/6] Training Full Optimized (Hybrid + Entropy + KL Refinement)...")
    v_full = run_variant(coords, W, "Full Optimized\n(+KL Refine)",
                         beta=2.0, feature_type='hybrid', model_type='sage',
                         gamma_ent=-0.1, refine=True)
    print_partition_report(W, v_full['assignments'], "Full Optimized", NUM_PARTITIONS)

    variants = [v_baseline, v_beta2, v_hybrid, v_full]

    # 3. Generate plots
    print("\n[6/6] Generating comparison plots...")

    plot_dual_bar(variants, 'optimization_comparison_bar.png')
    plot_training_curves(variants, 'optimization_training_curves.png')
    plot_partition_comparison(coords, W, variants, 'optimization_partition_comparison.png')
    plot_improvement_waterfall(variants, 'optimization_improvement_waterfall.png')
    plot_radar_comparison(v_baseline, v_full, 'optimization_radar.png')

    # 4. Save metrics JSON
    summary = {}
    for v in variants:
        key = v['label'].replace('\n', ' ')
        summary[key] = {
            'edge_cut': v['edge_cut'],
            'edge_cut_ratio': v['edge_cut_ratio'],
            'edge_cut_ratio_pct': round(v['edge_cut_ratio'] * 100, 2),
            'balance_std': round(v['balance_std'], 2),
            'num_cut_edges': v['num_cut_edges'],
            'total_edges': v['total_edges'],
            'partition_sizes': v['partition_sizes'],
            'training_time': round(v['training_time'], 2),
        }
    with open(METRICS_DIR / 'optimization_comparison.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved optimization_comparison.json")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Variant':<30} {'ECR %':>8} {'σ':>8} {'Cut Edges':>10} {'Time':>8}")
    print("-" * 80)
    for v in variants:
        lbl = v['label'].replace('\n', ' ')
        print(f"{lbl:<30} {v['edge_cut_ratio']*100:>7.2f}% {v['balance_std']:>7.2f} "
              f"{v['num_cut_edges']:>10} {v['training_time']:>7.2f}s")
    print("=" * 80)
    print("\nAll plots saved to results/plots/")
    print("Done!")


if __name__ == '__main__':
    main()
