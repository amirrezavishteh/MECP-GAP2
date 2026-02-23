"""
MECP-GAP Full Experiment Script

Reproduces the paper's experimental results (Figures 5-8):
- Small-scale: 200 nodes, Poisson Point Process
- Partition counts: 2, 3, 4, 5, 6
- Methods: I-GAP (ours), Greedy (KGGGP), METIS, Random

Paper: "Mobility-Aware MEC Planning With a GNN-Based Graph Partitioning Framework"
       IEEE TNSM, Vol. 21, No. 4, August 2024

Usage:
    python run_experiments.py                       # Full experiment
    python run_experiments.py --partitions 4        # Single partition count
    python run_experiments.py --num_nodes 50 --quick # Quick test
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
import json

src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from data_generation.graph_generator import CellTowerGraphGenerator, GraphConfig
from training.trainer import MECPTrainer, TrainingConfig
from baselines.greedy_baseline import GreedyPartitioner
from baselines.random_baseline import RandomPartitioner
from utils.visualization import compute_partition_metrics, print_partition_report

# Try METIS
try:
    from baselines.metis_baseline import MetisPartitioner
    HAS_METIS = True
except ImportError:
    HAS_METIS = False


def compute_edge_cut_weight(W, assignments, num_partitions):
    """Compute total weight of edges crossing partition boundaries."""
    N = len(assignments)
    cut_weight = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if assignments[i] != assignments[j] and W[i, j] > 0:
                cut_weight += W[i, j]
    return cut_weight


def run_igap(coords, W_matrix, num_partitions, num_epochs=500, seed=42):
    """Run I-GAP (our method)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    config = TrainingConfig(
        in_feats=-1,
        hidden_feats=128,
        num_partitions=num_partitions,
        num_layers=2,
        num_epochs=num_epochs,
        learning_rate=0.01,
        alpha=-1.0,
        beta=1.0,
        gamma=0.0,
        feature_type='weight_row',
        log_interval=num_epochs + 1,  # Suppress per-epoch output
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    trainer = MECPTrainer(config)
    start = time.time()
    results = trainer.train(coords, W_matrix, verbose=False)
    elapsed = time.time() - start
    
    return results['final_assignments'], elapsed


def run_greedy(coords, W_matrix, num_partitions, seed=42):
    """Run Greedy (KGGGP) baseline."""
    partitioner = GreedyPartitioner(
        num_partitions=num_partitions,
        seed_method='farthest',
        seed=seed
    )
    start = time.time()
    assignments = partitioner.fit(W_matrix, coords)
    elapsed = time.time() - start
    return assignments, elapsed


def run_metis(W_matrix, num_partitions, seed=42):
    """Run METIS baseline."""
    if not HAS_METIS:
        return None, 0.0
    try:
        partitioner = MetisPartitioner(
            num_partitions=num_partitions,
            seed=seed
        )
        start = time.time()
        assignments = partitioner.fit(W_matrix)
        elapsed = time.time() - start
        return assignments, elapsed
    except (ImportError, Exception) as e:
        print(f"METIS failed: {e}")
        return None, 0.0


def run_random(W_matrix, num_partitions, seed=42):
    """Run Random baseline."""
    partitioner = RandomPartitioner(
        num_partitions=num_partitions,
        balanced=True,
        seed=seed
    )
    start = time.time()
    assignments = partitioner.fit(W_matrix)
    elapsed = time.time() - start
    return assignments, elapsed


def main():
    parser = argparse.ArgumentParser(description='MECP-GAP Experiment Runner')
    parser.add_argument('--num_nodes', type=int, default=200, help='Number of nodes')
    parser.add_argument('--partitions', type=int, nargs='+', default=[2, 3, 4, 5, 6],
                        help='Partition counts to test')
    parser.add_argument('--num_epochs', type=int, default=500, help='Training epochs for I-GAP')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick', action='store_true', help='Quick test (fewer epochs)')
    parser.add_argument('--save_results', type=str, default=None, help='Save results to JSON')
    args = parser.parse_args()
    
    if args.quick:
        args.num_epochs = 200
    
    # Generate data
    print("=" * 70)
    print("MECP-GAP Experimental Evaluation")
    print("=" * 70)
    print(f"Nodes: {args.num_nodes} | Partitions: {args.partitions} | Epochs: {args.num_epochs}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    np.random.seed(args.seed)
    config = GraphConfig(num_nodes=args.num_nodes, area_size=10.0, seed=args.seed)
    generator = CellTowerGraphGenerator(config)
    coords, G, W_matrix = generator.generate()
    
    total_weight = W_matrix.sum()
    num_edges = np.count_nonzero(W_matrix) // 2
    print(f"Graph: {args.num_nodes} nodes, {num_edges} edges, total weight = {total_weight:.0f}")
    print()
    
    all_results = {}
    
    for P in args.partitions:
        print(f"\n{'='*70}")
        print(f"  P = {P} Partitions")
        print(f"{'='*70}")
        
        results_for_P = {}
        
        # I-GAP (ours)
        print(f"\n  [I-GAP] Training...", end=" ", flush=True)
        igap_assign, igap_time = run_igap(coords, W_matrix, P, args.num_epochs, args.seed)
        igap_metrics = compute_partition_metrics(W_matrix, igap_assign, num_partitions=P)
        print(f"Done ({igap_time:.2f}s)")
        results_for_P['I-GAP'] = {
            'edge_cut_ratio': igap_metrics['edge_cut_ratio'],
            'cut_weight': igap_metrics['edge_cut'],
            'partition_sizes': igap_metrics['partition_sizes'],
            'size_std': igap_metrics['balance_std'],
            'time': igap_time
        }
        
        # Greedy
        print(f"  [Greedy] Running...", end=" ", flush=True)
        greedy_assign, greedy_time = run_greedy(coords, W_matrix, P, args.seed)
        greedy_metrics = compute_partition_metrics(W_matrix, greedy_assign, num_partitions=P)
        print(f"Done ({greedy_time:.2f}s)")
        results_for_P['Greedy'] = {
            'edge_cut_ratio': greedy_metrics['edge_cut_ratio'],
            'cut_weight': greedy_metrics['edge_cut'],
            'partition_sizes': greedy_metrics['partition_sizes'],
            'size_std': greedy_metrics['balance_std'],
            'time': greedy_time
        }
        
        # METIS
        print(f"  [METIS] Running...", end=" ", flush=True)
        metis_assign, metis_time = run_metis(W_matrix, P, args.seed)
        if metis_assign is not None:
            metis_metrics = compute_partition_metrics(W_matrix, metis_assign, num_partitions=P)
            print(f"Done ({metis_time:.2f}s)")
            results_for_P['METIS'] = {
                'edge_cut_ratio': metis_metrics['edge_cut_ratio'],
                'cut_weight': metis_metrics['edge_cut'],
                'partition_sizes': metis_metrics['partition_sizes'],
                'size_std': metis_metrics['balance_std'],
                'time': metis_time
            }
        else:
            print("Skipped (pymetis not installed)")
        
        # Random
        print(f"  [Random] Running...", end=" ", flush=True)
        random_assign, random_time = run_random(W_matrix, P, args.seed)
        random_metrics = compute_partition_metrics(W_matrix, random_assign, num_partitions=P)
        print(f"Done ({random_time:.2f}s)")
        results_for_P['Random'] = {
            'edge_cut_ratio': random_metrics['edge_cut_ratio'],
            'cut_weight': random_metrics['edge_cut'],
            'partition_sizes': random_metrics['partition_sizes'],
            'size_std': random_metrics['balance_std'],
            'time': random_time
        }
        
        # Print comparison table
        print(f"\n  {'Method':<10} {'Cut Ratio':>10} {'Cut Weight':>12} {'Sizes':>25} {'Std Dev':>8} {'Time':>8}")
        print(f"  {'-'*78}")
        for method, r in results_for_P.items():
            sizes_str = str(r['partition_sizes'])
            print(f"  {method:<10} {r['edge_cut_ratio']*100:>9.2f}% {r['cut_weight']:>12.0f} {sizes_str:>25} {r['size_std']:>8.2f} {r['time']:>7.2f}s")
        
        all_results[f'P={P}'] = results_for_P
    
    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY: Edge Cut Ratio (%) by Method x Partitions")
    print(f"{'='*70}")
    
    methods = ['I-GAP', 'Greedy']
    if HAS_METIS:
        methods.append('METIS')
    methods.append('Random')
    
    header = f"{'P':>4}"
    for m in methods:
        header += f" {m:>10}"
    print(header)
    print("-" * (4 + 11 * len(methods)))
    
    for P in args.partitions:
        key = f'P={P}'
        row = f"{P:>4}"
        for m in methods:
            if m in all_results[key]:
                row += f" {all_results[key][m]['edge_cut_ratio']*100:>9.2f}%"
            else:
                row += f" {'N/A':>10}"
        print(row)
    
    # Save results
    if args.save_results:
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {save_path}")
    
    print(f"\n{'='*70}")
    print("Experiment Complete!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
