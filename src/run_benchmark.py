"""
MECP-GAP Benchmark Runner

Entry point for running comprehensive benchmark comparison of 
MEC server placement algorithms.

Usage:
    python run_benchmark.py                         # Default settings
    python run_benchmark.py --num_nodes 500 --num_partitions 8
    python run_benchmark.py --quick                 # Quick test mode
    python run_benchmark.py --scalability           # Run scalability test
    python run_benchmark.py --save_plots            # Save comparison figures
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from benchmark import (
    BenchmarkConfig, MECPBenchmark, 
    run_benchmark, run_scalability_benchmark
)


def main():
    parser = argparse.ArgumentParser(
        description='Run MEC placement algorithm benchmarks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--num_nodes', type=int, default=200,
                       help='Number of base stations')
    parser.add_argument('--num_partitions', type=int, default=4,
                       help='Number of MEC servers (partitions)')
    parser.add_argument('--area_size', type=float, default=10.0,
                       help='Simulation area size (km)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Training arguments
    parser.add_argument('--gap_epochs', type=int, default=200,
                       help='MECP-GAP training epochs')
    parser.add_argument('--pbpa_episodes', type=int, default=200,
                       help='PBPA training episodes')
    
    # Mode arguments
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with reduced settings')
    parser.add_argument('--scalability', action='store_true',
                       help='Run scalability benchmark across different graph sizes')
    
    # Method selection
    parser.add_argument('--no_mecp_gap', action='store_true',
                       help='Skip MECP-GAP')
    parser.add_argument('--no_ssc', action='store_true',
                       help='Skip MECP-GAP SSC variant')
    parser.add_argument('--no_metis', action='store_true',
                       help='Skip METIS')
    parser.add_argument('--no_greedy', action='store_true',
                       help='Skip Greedy')
    parser.add_argument('--no_pbpa', action='store_true',
                       help='Skip PBPA (DRL)')
    parser.add_argument('--no_random', action='store_true',
                       help='Skip Random')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save comparison plots')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Skip visualization')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 70)
    print("MECP-GAP Benchmark Suite")
    print("Comparative Analysis: MECP-GAP vs METIS vs Greedy vs PBPA vs Random")
    print("=" * 70)
    
    # Quick mode overrides
    if args.quick:
        args.num_nodes = 50
        args.gap_epochs = 50
        args.pbpa_episodes = 50
        print("\n[QUICK MODE] Using reduced settings for fast testing")
    
    # Scalability benchmark mode
    if args.scalability:
        print("\n[SCALABILITY MODE] Testing across different graph sizes")
        node_counts = [50, 100, 200]
        if not args.quick:
            node_counts.extend([500])
        
        results = run_scalability_benchmark(
            node_counts=node_counts,
            num_partitions=args.num_partitions,
            seed=args.seed,
            verbose=True
        )
        return
    
    # Standard benchmark
    config = BenchmarkConfig(
        num_nodes=args.num_nodes,
        num_partitions=args.num_partitions,
        area_size=args.area_size,
        seed=args.seed,
        gap_epochs=args.gap_epochs,
        pbpa_episodes=args.pbpa_episodes,
        run_mecp_gap=not args.no_mecp_gap,
        run_mecp_gap_ssc=not args.no_ssc,
        run_metis=not args.no_metis,
        run_greedy=not args.no_greedy,
        run_pbpa=not args.no_pbpa,
        run_random=not args.no_random,
        output_dir=args.output_dir,
        verbose=True
    )
    
    benchmark = MECPBenchmark(config)
    benchmark.prepare_data()
    results = benchmark.run_all()
    benchmark.print_comparison_table()
    
    # Save results
    benchmark.save_results()
    
    # Visualization
    if not args.no_visualize:
        save_path = None
        if args.save_plots:
            plots_dir = Path(args.output_dir) / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(plots_dir / 'benchmark_metrics.png')
        
        benchmark.visualize_metrics_bar(save_path=save_path, show=not args.save_plots)
        
        if args.save_plots:
            comp_path = str(plots_dir / 'benchmark_partitions.png')
            benchmark.visualize_comparison(save_path=comp_path, show=False)
    
    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
