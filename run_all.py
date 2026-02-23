"""
MECP-GAP: Complete Pipeline Runner
===================================
Generates data, trains models, runs benchmarks, creates plots, and saves all results.

This script executes the full MECP-GAP evaluation pipeline:
1. Generate synthetic 5G network graph data
2. Train MECP-GAP model with default and SSC-aware variants  
3. Run all baseline comparisons (METIS, Greedy, PBPA, Random)
4. Run experiments across partition counts P=2,3,4,5,6
5. Generate all plots (partition visualizations, training curves, metrics bars)
6. Save comprehensive results to JSON
"""

import sys
import os
import time
import json
import numpy as np
import torch
import warnings
import traceback

# Force non-interactive matplotlib backend for saving plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).parent
SRC_DIR = ROOT_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

from data_generation.graph_generator import CellTowerGraphGenerator, GraphConfig
from training.trainer import MECPTrainer, TrainingConfig, prepare_graph_data
from utils.visualization import (
    visualize_results,
    visualize_training_progress,
    visualize_comparison,
    compute_partition_metrics,
    print_partition_report
)
from utils.utils import compute_extended_metrics
from baselines.greedy_baseline import GreedyPartitioner
from baselines.random_baseline import RandomPartitioner, run_random_multiple

# Try optional imports
try:
    from baselines.metis_baseline import MetisPartitioner, MetisFallback
    # Test if pymetis actually works
    try:
        _tmp = MetisPartitioner(2)
        HAS_PYMETIS = True
    except Exception:
        HAS_PYMETIS = False
    HAS_METIS = True  # MetisFallback always available
except ImportError:
    HAS_METIS = False
    HAS_PYMETIS = False

try:
    from baselines.pbpa_baseline import PBPAPartitioner, PBPAConfig
    HAS_PBPA = True
except ImportError:
    HAS_PBPA = False


# ============================================================================
# Configuration
# ============================================================================
SEED = 42
NUM_NODES = 200
AREA_SIZE = 10.0
GAMMA_GRAVITY = 1.5
NUM_EPOCHS_TRAIN = 500
NUM_EPOCHS_BENCHMARK = 300
PBPA_EPISODES = 200
PARTITION_COUNTS = [2, 3, 4, 5, 6]
DEFAULT_PARTITIONS = 4

# Output directories
RESULTS_DIR = ROOT_DIR / 'results'
PLOTS_DIR = RESULTS_DIR / 'plots'
METRICS_DIR = RESULTS_DIR / 'metrics'
DATA_DIR = ROOT_DIR / 'data'
GRAPHS_DIR = DATA_DIR / 'graphs'
PROCESSED_DIR = DATA_DIR / 'processed'


def ensure_dirs():
    """Create all output directories."""
    for d in [RESULTS_DIR, PLOTS_DIR, METRICS_DIR, GRAPHS_DIR, PROCESSED_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def set_seeds(seed=SEED):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# ============================================================================
# Step 1: Data Generation
# ============================================================================
def generate_data():
    """Generate synthetic 5G network graph data."""
    print("\n" + "=" * 70)
    print("STEP 1: DATA GENERATION")
    print("=" * 70)
    
    set_seeds()
    
    config = GraphConfig(
        num_nodes=NUM_NODES,
        area_size=AREA_SIZE,
        gamma=GAMMA_GRAVITY,
        mode='synthetic',
        seed=SEED
    )
    
    generator = CellTowerGraphGenerator(config)
    coords, G, W_matrix = generator.generate()
    
    num_edges = np.count_nonzero(W_matrix) // 2
    total_weight = W_matrix.sum() / 2
    avg_degree = 2 * num_edges / NUM_NODES
    
    print(f"  Nodes:         {NUM_NODES}")
    print(f"  Edges:         {num_edges}")
    print(f"  Total Weight:  {total_weight:.2f}")
    print(f"  Avg Degree:    {avg_degree:.1f}")
    print(f"  Area:          {AREA_SIZE} x {AREA_SIZE} km")
    print(f"  Gravity gamma: {GAMMA_GRAVITY}")
    
    # Save generated data
    save_dir = PROCESSED_DIR / 'main_graph'
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / 'coords.npy', coords)
    np.save(save_dir / 'weights.npy', W_matrix)
    
    # Save edge_index for reference
    rows, cols = np.nonzero(W_matrix)
    edge_index = np.array([rows, cols])
    np.save(save_dir / 'edge_index.npy', edge_index)
    
    # Adjacency (binary)
    adjacency = (W_matrix > 0).astype(np.float32)
    np.save(save_dir / 'adjacency.npy', adjacency)
    
    metadata = {
        'num_nodes': NUM_NODES,
        'num_edges': num_edges,
        'area_size': AREA_SIZE,
        'gamma': GAMMA_GRAVITY,
        'total_weight': float(total_weight),
        'avg_degree': float(avg_degree),
        'seed': SEED,
        'generation_method': 'Poisson Point Process + Voronoi + Gravity Model'
    }
    with open(save_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  Data saved to: {save_dir}")
    
    # Visualize raw graph
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(coords[:, 0], coords[:, 1], c='steelblue', s=40, 
               edgecolors='black', linewidths=0.5, zorder=5)
    for i in range(NUM_NODES):
        for j in range(i+1, NUM_NODES):
            if W_matrix[i, j] > 0:
                w = W_matrix[i, j]
                max_w = W_matrix.max()
                lw = 0.3 + (w / max_w) * 2.0
                alpha = 0.2 + (w / max_w) * 0.6
                ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                       'gray', linewidth=lw, alpha=alpha)
    ax.set_title(f'5G Network Graph: {NUM_NODES} Base Stations, {num_edges} Edges\n'
                f'(Edge thickness ∝ mobility traffic)', fontsize=13)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'raw_graph.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: results/plots/raw_graph.png")
    
    return coords, G, W_matrix, metadata


# ============================================================================
# Step 2: Train MECP-GAP Model
# ============================================================================
def train_model(coords, W_matrix, num_partitions=DEFAULT_PARTITIONS, num_epochs=NUM_EPOCHS_TRAIN):
    """Train MECP-GAP model and return results."""
    print("\n" + "=" * 70)
    print(f"STEP 2: MECP-GAP TRAINING (P={num_partitions}, {num_epochs} epochs)")
    print("=" * 70)
    
    set_seeds()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    config = TrainingConfig(
        in_feats=-1,
        hidden_feats=128,
        num_partitions=num_partitions,
        num_layers=2,
        num_epochs=num_epochs,
        learning_rate=0.01,
        alpha=-1.0,  # Auto-compute per paper
        beta=1.0,
        gamma=0.0,
        feature_type='weight_row',
        log_interval=100,
        device=device
    )
    
    trainer = MECPTrainer(config)
    start = time.time()
    results = trainer.train(coords, W_matrix, verbose=True)
    elapsed = time.time() - start
    
    assignments = results['final_assignments']
    
    # Print report
    print_partition_report(W_matrix, assignments, "MECP-GAP", num_partitions=num_partitions)
    print(f"  Training time: {elapsed:.2f}s")
    
    # Save partition visualization
    fig = visualize_results(
        coords, W_matrix, assignments,
        num_partitions=num_partitions,
        save_path=str(PLOTS_DIR / f'partition_P{num_partitions}.png'),
        show=False
    )
    plt.close()
    print(f"  Plot saved: results/plots/partition_P{num_partitions}.png")
    
    # Save training progress
    if 'history' in results and results['history']:
        fig = visualize_training_progress(
            results['history'],
            save_path=str(PLOTS_DIR / f'training_progress_P{num_partitions}.png'),
            show=False
        )
        plt.close()
        print(f"  Plot saved: results/plots/training_progress_P{num_partitions}.png")
    
    # Save assignments
    np.save(METRICS_DIR / f'assignments_mecp_gap_P{num_partitions}.npy', assignments)
    
    return assignments, results, elapsed


# ============================================================================
# Step 3: Run All Baselines
# ============================================================================
def run_baselines(coords, W_matrix, num_partitions=DEFAULT_PARTITIONS):
    """Run all baseline algorithms and return results."""
    print("\n" + "=" * 70)
    print(f"STEP 3: BASELINE COMPARISONS (P={num_partitions})")
    print("=" * 70)
    
    all_results = {}
    all_assignments = {}
    
    # --- Greedy (KGGGP) ---
    print(f"\n  [Greedy/KGGGP] Running...", end=" ", flush=True)
    try:
        set_seeds()
        partitioner = GreedyPartitioner(
            num_partitions=num_partitions,
            seed_method='farthest',
            seed=SEED
        )
        start = time.time()
        greedy_assign = partitioner.fit(W_matrix, coords)
        greedy_time = time.time() - start
        greedy_metrics = compute_partition_metrics(W_matrix, greedy_assign, num_partitions)
        all_results['Greedy (KGGGP)'] = {**greedy_metrics, 'time': greedy_time}
        all_assignments['Greedy (KGGGP)'] = greedy_assign
        print(f"Done ({greedy_time:.2f}s) - Cut ratio: {greedy_metrics['edge_cut_ratio']*100:.2f}%")
    except Exception as e:
        print(f"Failed: {e}")
    
    # --- METIS / Spectral ---
    print(f"  [METIS/Spectral] Running...", end=" ", flush=True)
    try:
        set_seeds()
        if HAS_METIS and HAS_PYMETIS:
            partitioner = MetisPartitioner(num_partitions=num_partitions, seed=SEED)
            start = time.time()
            metis_assign = partitioner.fit(W_matrix)
            metis_time = time.time() - start
            label = 'METIS'
        elif HAS_METIS:
            start = time.time()
            metis_assign = MetisFallback.spectral_partition(W_matrix, num_partitions)
            metis_time = time.time() - start
            label = 'Spectral'
        else:
            raise ImportError("No METIS or spectral fallback available")
        
        metis_metrics = compute_partition_metrics(W_matrix, metis_assign, num_partitions)
        all_results[label] = {**metis_metrics, 'time': metis_time}
        all_assignments[label] = metis_assign
        print(f"Done ({metis_time:.2f}s) - Cut ratio: {metis_metrics['edge_cut_ratio']*100:.2f}%")
    except Exception as e:
        print(f"Failed: {e}")
        traceback.print_exc()
    
    # --- PBPA (DRL) ---
    print(f"  [PBPA/DRL] Running...", end=" ", flush=True)
    try:
        if HAS_PBPA:
            set_seeds()
            pbpa_config = PBPAConfig(
                num_partitions=num_partitions,
                max_episodes=PBPA_EPISODES,
                max_steps_per_episode=30,
                hidden_dim=128,
                seed=SEED
            )
            partitioner = PBPAPartitioner(num_partitions=num_partitions, config=pbpa_config)
            start = time.time()
            pbpa_assign = partitioner.fit(W_matrix, coords, verbose=False)
            pbpa_time = time.time() - start
            pbpa_metrics = compute_partition_metrics(W_matrix, pbpa_assign, num_partitions)
            all_results['PBPA (DRL)'] = {**pbpa_metrics, 'time': pbpa_time}
            all_assignments['PBPA (DRL)'] = pbpa_assign
            print(f"Done ({pbpa_time:.2f}s) - Cut ratio: {pbpa_metrics['edge_cut_ratio']*100:.2f}%")
        else:
            print("Skipped (not available)")
    except Exception as e:
        print(f"Failed: {e}")
    
    # --- Random (Best of 10) ---
    print(f"  [Random Best-of-10] Running...", end=" ", flush=True)
    try:
        set_seeds()
        start = time.time()
        random_assign, random_info = run_random_multiple(
            W_matrix, num_partitions=num_partitions, num_trials=10,
            balanced=True, seed=SEED
        )
        random_time = time.time() - start
        random_metrics = compute_partition_metrics(W_matrix, random_assign, num_partitions)
        all_results['Random (Best of 10)'] = {**random_metrics, 'time': random_time}
        all_assignments['Random (Best of 10)'] = random_assign
        print(f"Done ({random_time:.2f}s) - Cut ratio: {random_metrics['edge_cut_ratio']*100:.2f}%")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Save baseline assignments
    for name, assign in all_assignments.items():
        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        np.save(METRICS_DIR / f'assignments_{safe_name}.npy', assign)
    
    return all_results, all_assignments


# ============================================================================
# Step 4: Run Experiments Across Partition Counts
# ============================================================================
def run_partition_sweep(coords, W_matrix):
    """Run MECP-GAP + baselines for P=2,3,4,5,6."""
    print("\n" + "=" * 70)
    print("STEP 4: PARTITION SWEEP (P=2,3,4,5,6)")
    print("=" * 70)
    
    sweep_results = {}
    
    for P in PARTITION_COUNTS:
        print(f"\n{'─' * 60}")
        print(f"  P = {P} Partitions")
        print(f"{'─' * 60}")
        
        results_P = {}
        
        # MECP-GAP
        print(f"    [MECP-GAP] Training...", end=" ", flush=True)
        try:
            set_seeds()
            config = TrainingConfig(
                in_feats=-1, hidden_feats=128, num_partitions=P,
                num_layers=2, num_epochs=NUM_EPOCHS_BENCHMARK,
                learning_rate=0.01, alpha=-1.0, beta=1.0, gamma=0.0,
                feature_type='weight_row',
                log_interval=NUM_EPOCHS_BENCHMARK + 1,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            trainer = MECPTrainer(config)
            start = time.time()
            res = trainer.train(coords, W_matrix, verbose=False)
            elapsed = time.time() - start
            igap_metrics = compute_partition_metrics(W_matrix, res['final_assignments'], P)
            results_P['MECP-GAP'] = {
                'edge_cut_ratio': igap_metrics['edge_cut_ratio'],
                'edge_cut': igap_metrics['edge_cut'],
                'partition_sizes': igap_metrics['partition_sizes'],
                'balance_std': igap_metrics['balance_std'],
                'time': elapsed
            }
            
            # Save partition plot for each P
            fig = visualize_results(
                coords, W_matrix, res['final_assignments'],
                num_partitions=P,
                save_path=str(PLOTS_DIR / f'partition_sweep_P{P}.png'),
                show=False
            )
            plt.close()
            
            print(f"Done ({elapsed:.2f}s) - Cut: {igap_metrics['edge_cut_ratio']*100:.2f}%")
        except Exception as e:
            print(f"Failed: {e}")
        
        # Greedy
        print(f"    [Greedy] Running...", end=" ", flush=True)
        try:
            set_seeds()
            partitioner = GreedyPartitioner(num_partitions=P, seed_method='farthest', seed=SEED)
            start = time.time()
            assign = partitioner.fit(W_matrix, coords)
            elapsed = time.time() - start
            m = compute_partition_metrics(W_matrix, assign, P)
            results_P['Greedy'] = {
                'edge_cut_ratio': m['edge_cut_ratio'], 'edge_cut': m['edge_cut'],
                'partition_sizes': m['partition_sizes'], 'balance_std': m['balance_std'],
                'time': elapsed
            }
            print(f"Done ({elapsed:.2f}s) - Cut: {m['edge_cut_ratio']*100:.2f}%")
        except Exception as e:
            print(f"Failed: {e}")
        
        # METIS / Spectral
        print(f"    [METIS/Spectral] Running...", end=" ", flush=True)
        try:
            set_seeds()
            if HAS_METIS and HAS_PYMETIS:
                partitioner = MetisPartitioner(num_partitions=P, seed=SEED)
                start = time.time()
                assign = partitioner.fit(W_matrix)
            elif HAS_METIS:
                start = time.time()
                assign = MetisFallback.spectral_partition(W_matrix, P)
            else:
                raise ImportError("No METIS available")
            elapsed = time.time() - start
            m = compute_partition_metrics(W_matrix, assign, P)
            method_label = 'METIS' if HAS_PYMETIS else 'Spectral'
            results_P[method_label] = {
                'edge_cut_ratio': m['edge_cut_ratio'], 'edge_cut': m['edge_cut'],
                'partition_sizes': m['partition_sizes'], 'balance_std': m['balance_std'],
                'time': elapsed
            }
            print(f"Done ({elapsed:.2f}s) - Cut: {m['edge_cut_ratio']*100:.2f}%")
        except Exception as e:
            print(f"Failed: {e}")
        
        # PBPA
        if HAS_PBPA:
            print(f"    [PBPA] Running...", end=" ", flush=True)
            try:
                set_seeds()
                pbpa_config = PBPAConfig(
                    num_partitions=P, max_episodes=PBPA_EPISODES,
                    max_steps_per_episode=30, hidden_dim=128, seed=SEED
                )
                partitioner = PBPAPartitioner(num_partitions=P, config=pbpa_config)
                start = time.time()
                assign = partitioner.fit(W_matrix, coords, verbose=False)
                elapsed = time.time() - start
                m = compute_partition_metrics(W_matrix, assign, P)
                results_P['PBPA'] = {
                    'edge_cut_ratio': m['edge_cut_ratio'], 'edge_cut': m['edge_cut'],
                    'partition_sizes': m['partition_sizes'], 'balance_std': m['balance_std'],
                    'time': elapsed
                }
                print(f"Done ({elapsed:.2f}s) - Cut: {m['edge_cut_ratio']*100:.2f}%")
            except Exception as e:
                print(f"Failed: {e}")
        
        # Random
        print(f"    [Random] Running...", end=" ", flush=True)
        try:
            set_seeds()
            start = time.time()
            assign, _ = run_random_multiple(W_matrix, num_partitions=P, num_trials=10,
                                            balanced=True, seed=SEED)
            elapsed = time.time() - start
            m = compute_partition_metrics(W_matrix, assign, P)
            results_P['Random'] = {
                'edge_cut_ratio': m['edge_cut_ratio'], 'edge_cut': m['edge_cut'],
                'partition_sizes': m['partition_sizes'], 'balance_std': m['balance_std'],
                'time': elapsed
            }
            print(f"Done ({elapsed:.2f}s) - Cut: {m['edge_cut_ratio']*100:.2f}%")
        except Exception as e:
            print(f"Failed: {e}")
        
        sweep_results[P] = results_P
    
    return sweep_results


# ============================================================================
# Step 5: Generate Summary Plots
# ============================================================================
def generate_summary_plots(sweep_results, coords, W_matrix, all_assignments, baseline_results):
    """Generate all summary plots."""
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING PLOTS")
    print("=" * 70)
    
    # --- Plot 1: Edge Cut Ratio vs P (line chart) ---
    print("  Generating edge cut ratio vs P plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = set()
    for P_results in sweep_results.values():
        methods.update(P_results.keys())
    
    method_colors = {
        'MECP-GAP': '#2196F3',
        'Greedy': '#FF9800',
        'METIS': '#4CAF50',
        'Spectral': '#4CAF50',
        'PBPA': '#9C27B0',
        'Random': '#F44336'
    }
    method_markers = {
        'MECP-GAP': 'o',
        'Greedy': 's',
        'METIS': '^',
        'Spectral': '^',
        'PBPA': 'D',
        'Random': 'v'
    }
    
    for method in ['MECP-GAP', 'Greedy', 'METIS', 'Spectral', 'PBPA', 'Random']:
        if method not in methods:
            continue
        Ps = []
        cuts = []
        for P in PARTITION_COUNTS:
            if P in sweep_results and method in sweep_results[P]:
                Ps.append(P)
                cuts.append(sweep_results[P][method]['edge_cut_ratio'] * 100)
        if Ps:
            ax.plot(Ps, cuts, 
                   color=method_colors.get(method, 'gray'),
                   marker=method_markers.get(method, 'o'),
                   linewidth=2, markersize=8, label=method)
    
    ax.set_xlabel('Number of Partitions (P)', fontsize=12)
    ax.set_ylabel('Edge Cut Ratio (%)', fontsize=12)
    ax.set_title('Edge Cut Ratio vs Number of Partitions', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(PARTITION_COUNTS)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'edge_cut_vs_partitions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: results/plots/edge_cut_vs_partitions.png")
    
    # --- Plot 2: Balance Std vs P ---
    print("  Generating balance std vs P plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in ['MECP-GAP', 'Greedy', 'METIS', 'Spectral', 'PBPA', 'Random']:
        if method not in methods:
            continue
        Ps = []
        stds = []
        for P in PARTITION_COUNTS:
            if P in sweep_results and method in sweep_results[P]:
                Ps.append(P)
                stds.append(sweep_results[P][method]['balance_std'])
        if Ps:
            ax.plot(Ps, stds,
                   color=method_colors.get(method, 'gray'),
                   marker=method_markers.get(method, 'o'),
                   linewidth=2, markersize=8, label=method)
    
    ax.set_xlabel('Number of Partitions (P)', fontsize=12)
    ax.set_ylabel('Partition Size Std Dev', fontsize=12)
    ax.set_title('Load Balance (Std Dev) vs Number of Partitions', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(PARTITION_COUNTS)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'balance_vs_partitions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: results/plots/balance_vs_partitions.png")
    
    # --- Plot 3: Runtime vs P ---
    print("  Generating runtime vs P plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in ['MECP-GAP', 'Greedy', 'METIS', 'Spectral', 'PBPA', 'Random']:
        if method not in methods:
            continue
        Ps = []
        times = []
        for P in PARTITION_COUNTS:
            if P in sweep_results and method in sweep_results[P]:
                Ps.append(P)
                times.append(sweep_results[P][method]['time'])
        if Ps:
            ax.plot(Ps, times,
                   color=method_colors.get(method, 'gray'),
                   marker=method_markers.get(method, 'o'),
                   linewidth=2, markersize=8, label=method)
    
    ax.set_xlabel('Number of Partitions (P)', fontsize=12)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('Execution Time vs Number of Partitions', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(PARTITION_COUNTS)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'runtime_vs_partitions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: results/plots/runtime_vs_partitions.png")
    
    # --- Plot 4: Comparison bar chart for P=4 ---
    print("  Generating metrics comparison bar chart...")
    if DEFAULT_PARTITIONS in sweep_results:
        p4 = sweep_results[DEFAULT_PARTITIONS]
        method_names = list(p4.keys())
        cut_ratios = [p4[m]['edge_cut_ratio'] for m in method_names]
        bal_stds = [p4[m]['balance_std'] for m in method_names]
        runtimes = [p4[m]['time'] for m in method_names]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        x = np.arange(len(method_names))
        colors = [method_colors.get(m, '#888888') for m in method_names]
        
        axes[0].bar(x, [c*100 for c in cut_ratios], color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        axes[0].set_ylabel('Edge Cut Ratio (%)', fontsize=11)
        axes[0].set_title(f'Edge Cut Ratio (P={DEFAULT_PARTITIONS})', fontsize=13)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(method_names, rotation=25, ha='right')
        axes[0].grid(axis='y', alpha=0.3)
        
        axes[1].bar(x, bal_stds, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        axes[1].set_ylabel('Partition Size Std Dev', fontsize=11)
        axes[1].set_title(f'Load Balance (P={DEFAULT_PARTITIONS})', fontsize=13)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(method_names, rotation=25, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        
        axes[2].bar(x, runtimes, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        axes[2].set_ylabel('Runtime (s)', fontsize=11)
        axes[2].set_title(f'Execution Time (P={DEFAULT_PARTITIONS})', fontsize=13)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(method_names, rotation=25, ha='right')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'Algorithm Comparison (N={NUM_NODES}, P={DEFAULT_PARTITIONS})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'metrics_comparison_bar.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: results/plots/metrics_comparison_bar.png")
    
    # --- Plot 5: Side-by-side partition comparison ---
    print("  Generating side-by-side partition comparison...")
    if all_assignments:
        # Get MECP-GAP assignment for P=4
        mecp_assign_path = METRICS_DIR / f'assignments_mecp_gap_P{DEFAULT_PARTITIONS}.npy'
        assignments_list = []
        method_names = []
        
        if mecp_assign_path.exists():
            assignments_list.append(np.load(mecp_assign_path))
            method_names.append('MECP-GAP')
        
        for name, assign in all_assignments.items():
            assignments_list.append(assign)
            method_names.append(name)
        
        if len(assignments_list) > 1:
            fig = visualize_comparison(
                coords, W_matrix,
                assignments_list, method_names,
                figsize=(6 * len(assignments_list), 6),
                save_path=str(PLOTS_DIR / 'partition_comparison.png'),
                show=False
            )
            plt.close()
            print(f"    Saved: results/plots/partition_comparison.png")
    
    # --- Plot 6: Heatmap of edge cut ratios ---
    print("  Generating edge cut heatmap...")
    all_methods_ordered = ['MECP-GAP', 'Greedy', 'METIS', 'Spectral', 'PBPA', 'Random']
    avail_methods = [m for m in all_methods_ordered if any(m in sweep_results.get(P, {}) for P in PARTITION_COUNTS)]
    
    if avail_methods:
        data_matrix = []
        for method in avail_methods:
            row = []
            for P in PARTITION_COUNTS:
                if P in sweep_results and method in sweep_results[P]:
                    row.append(sweep_results[P][method]['edge_cut_ratio'] * 100)
                else:
                    row.append(np.nan)
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(data_matrix, cmap='RdYlGn_r', aspect='auto')
        ax.set_xticks(range(len(PARTITION_COUNTS)))
        ax.set_xticklabels([f'P={p}' for p in PARTITION_COUNTS], fontsize=11)
        ax.set_yticks(range(len(avail_methods)))
        ax.set_yticklabels(avail_methods, fontsize=11)
        
        for i in range(len(avail_methods)):
            for j in range(len(PARTITION_COUNTS)):
                if not np.isnan(data_matrix[i, j]):
                    ax.text(j, i, f'{data_matrix[i, j]:.1f}%',
                           ha='center', va='center', fontsize=10, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Edge Cut Ratio (%)', fontsize=11)
        ax.set_title('Edge Cut Ratio Heatmap (Lower is Better)', fontsize=14)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'edge_cut_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: results/plots/edge_cut_heatmap.png")


# ============================================================================
# Step 6: Save Comprehensive Results
# ============================================================================
def save_all_results(graph_metadata, train_results, baseline_results, sweep_results):
    """Save all results to JSON."""
    print("\n" + "=" * 70)
    print("STEP 6: SAVING RESULTS")
    print("=" * 70)
    
    # Convert sweep results for JSON serialization
    sweep_json = {}
    for P, methods in sweep_results.items():
        sweep_json[f'P={P}'] = {}
        for method, metrics in methods.items():
            sweep_json[f'P={P}'][method] = {
                k: (v if not isinstance(v, (np.ndarray, np.integer, np.floating)) 
                    else v.tolist() if isinstance(v, np.ndarray) else float(v))
                for k, v in metrics.items()
            }
    
    # Convert baseline results
    baseline_json = {}
    for method, metrics in baseline_results.items():
        baseline_json[method] = {
            k: (v if not isinstance(v, (np.ndarray, np.integer, np.floating))
                else v.tolist() if isinstance(v, np.ndarray) else float(v))
            for k, v in metrics.items()
        }
    
    comprehensive = {
        'experiment_info': {
            'project': 'MECP-GAP',
            'description': 'Mobility-aware Edge Computing Partitioning using Graph Neural Networks',
            'paper': 'IEEE TNSM, Vol. 21, No. 4, August 2024',
            'num_nodes': NUM_NODES,
            'area_size': AREA_SIZE,
            'gravity_gamma': GAMMA_GRAVITY,
            'seed': SEED,
            'training_epochs': NUM_EPOCHS_TRAIN,
            'benchmark_epochs': NUM_EPOCHS_BENCHMARK,
            'pbpa_episodes': PBPA_EPISODES,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        },
        'graph_metadata': graph_metadata,
        'default_partition_results': {
            'num_partitions': DEFAULT_PARTITIONS,
            'mecp_gap': {
                'training_time': train_results.get('training_time', 0),
                'final_loss': train_results.get('final_loss', 0),
            },
            'baselines': baseline_json
        },
        'partition_sweep': sweep_json,
    }
    
    results_path = METRICS_DIR / 'comprehensive_results.json'
    with open(results_path, 'w') as f:
        json.dump(comprehensive, f, indent=2, default=str)
    print(f"  Saved: {results_path}")
    
    return comprehensive


# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    """Run the complete MECP-GAP pipeline."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#   MECP-GAP: Complete Evaluation Pipeline" + " " * 27 + "#")
    print("#   Mobility-aware MEC Planning with GNN-Based Graph Partitioning" + " " * 4 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print(f"\nConfiguration:")
    print(f"  Nodes: {NUM_NODES} | Area: {AREA_SIZE}km | Seed: {SEED}")
    print(f"  Training epochs: {NUM_EPOCHS_TRAIN} | Benchmark epochs: {NUM_EPOCHS_BENCHMARK}")
    print(f"  Partition sweep: {PARTITION_COUNTS}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    total_start = time.time()
    ensure_dirs()
    
    # Step 1: Generate Data
    coords, G, W_matrix, graph_metadata = generate_data()
    
    # Step 2: Train MECP-GAP
    assignments, train_results, train_time = train_model(coords, W_matrix)
    
    # Step 3: Run Baselines
    baseline_results, all_assignments = run_baselines(coords, W_matrix)
    
    # Step 4: Partition Sweep
    sweep_results = run_partition_sweep(coords, W_matrix)
    
    # Step 5: Generate Plots
    generate_summary_plots(sweep_results, coords, W_matrix, all_assignments, baseline_results)
    
    # Step 6: Save Results
    comprehensive = save_all_results(graph_metadata, train_results, baseline_results, sweep_results)
    
    total_time = time.time() - total_start
    
    # Final Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\n  Generated files:")
    
    # List all output files
    for d in [PLOTS_DIR, METRICS_DIR, PROCESSED_DIR / 'main_graph']:
        if d.exists():
            for f in sorted(d.iterdir()):
                rel = f.relative_to(ROOT_DIR)
                print(f"    {rel}")
    
    print(f"\n  Summary (P={DEFAULT_PARTITIONS}):")
    if DEFAULT_PARTITIONS in sweep_results:
        p4 = sweep_results[DEFAULT_PARTITIONS]
        print(f"  {'Method':<18} {'Cut Ratio':>10} {'Balance Std':>12} {'Time':>8}")
        print(f"  {'-'*52}")
        for method in ['MECP-GAP', 'Greedy', 'METIS', 'Spectral', 'PBPA', 'Random']:
            if method in p4:
                r = p4[method]
                print(f"  {method:<18} {r['edge_cut_ratio']*100:>9.2f}% {r['balance_std']:>12.2f} {r['time']:>7.2f}s")
    
    print("\n" + "=" * 70)
    return comprehensive


if __name__ == '__main__':
    main()
