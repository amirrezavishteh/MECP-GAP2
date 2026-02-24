"""
Comprehensive Benchmark Comparison for MEC Server Placement Algorithms

This module implements a rigorous comparative evaluation framework for:
1. MECP-GAP (GNN-based, our method)
2. METIS (Industry standard, multilevel graph partitioning)
3. Greedy / KGGGP (Localized heuristic)
4. PBPA (Deep Reinforcement Learning with PPO)
5. Random (Lower-bound baseline)

Evaluation Metrics:
- Edge Cut (total weight of cut edges) - lower is better
- Edge Cut Ratio (fraction of total edge weight cut) - lower is better  
- Load Balance Variance (deviation from ideal partition sizes) - lower is better
- Balance Ratio (max_size / ideal_size) - closer to 1.0 is better
- Normalized Cut - lower is better
- Modularity - higher is better
- Running Time (seconds) - lower is better

Summary of Expected Baseline Performance (from paper):
+-------------------+--------+--------+---------+--------+--------+
| Metric            | METIS  | Greedy | PBPA    | Random | GAP    |
+-------------------+--------+--------+---------+--------+--------+
| Edge Cut Quality  | Excl.  | Poor   | Fair    | Worst  | Good   |
| Load Balance      | Excl.  | Var.   | Fair    | Fair   | Good   |
| Running Time      | Med    | Fast   | Slow    | Fast   | Med    |
| Mobility Aware    | No     | No     | Yes     | No     | Yes    |
| Stability         | High   | High   | Low     | High   | High   |
+-------------------+--------+--------+---------+--------+--------+

Paper Reference: Section V - Experiments, Table comparing baselines
"""

import numpy as np
import time
import json
import warnings
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict

import sys
sys.path.append(str(Path(__file__).parent))

from utils.utils import (
    compute_metrics, compute_metrics_vectorized, 
    compute_extended_metrics, evaluate_partition
)
from data_generation.graph_generator import CellTowerGraphGenerator, GraphConfig


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""
    # Data settings
    num_nodes: int = 200
    area_size: float = 10.0
    gamma: float = 1.5  # Gravity model friction
    seed: int = 42
    
    # Partitioning settings
    num_partitions: int = 4
    
    # MECP-GAP settings
    gap_epochs: int = 500
    gap_lr: float = 0.01
    gap_hidden: int = 128
    gap_alpha: float = -1.0    # Auto-compute alpha per paper (Eq. 9 normalization)
    gap_beta: float = 1.0      # Balance loss weight (paper default)
    gap_gamma: float = 0.0     # Entropy regularization (disabled per paper)
    
    # PBPA settings
    pbpa_episodes: int = 200
    pbpa_steps: int = 30
    
    # Greedy settings
    greedy_seed_method: str = 'farthest'
    
    # Methods to run
    run_mecp_gap: bool = True
    run_mecp_gap_ssc: bool = True
    run_metis: bool = True
    run_greedy: bool = True
    run_pbpa: bool = True
    run_random: bool = True
    
    # Output settings
    output_dir: str = 'results'
    verbose: bool = True


@dataclass
class BenchmarkResult:
    """Result from a single algorithm run."""
    method_name: str
    assignments: np.ndarray
    edge_cut: float
    edge_cut_ratio: float
    load_balance_var: float
    balance_ratio: float
    normalized_cut: float
    modularity: float
    partition_sizes: List[int]
    runtime: float
    additional: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding numpy arrays)."""
        return {
            'method_name': self.method_name,
            'edge_cut': float(self.edge_cut),
            'edge_cut_ratio': float(self.edge_cut_ratio),
            'load_balance_var': float(self.load_balance_var),
            'balance_ratio': float(self.balance_ratio),
            'normalized_cut': float(self.normalized_cut),
            'modularity': float(self.modularity),
            'partition_sizes': [int(s) for s in self.partition_sizes],
            'runtime': float(self.runtime),
        }


class MECPBenchmark:
    """
    Comprehensive benchmarking framework for MEC placement algorithms.
    
    This class orchestrates the comparison of multiple partitioning
    algorithms on the same graph data, computing standardized metrics.
    """
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark.
        
        Args:
            config: BenchmarkConfig with experiment settings
        """
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.coords = None
        self.W_matrix = None
        self.adj_matrix = None
    
    def prepare_data(self, coords: Optional[np.ndarray] = None,
                     W_matrix: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare graph data for benchmarking.
        
        Args:
            coords: Pre-computed coordinates (None = generate synthetic)
            W_matrix: Pre-computed weight matrix (None = generate synthetic)
            
        Returns:
            coords, W_matrix
        """
        if coords is not None and W_matrix is not None:
            self.coords = coords
            self.W_matrix = W_matrix
        else:
            graph_config = GraphConfig(
                num_nodes=self.config.num_nodes,
                area_size=self.config.area_size,
                gamma=self.config.gamma,
                seed=self.config.seed
            )
            generator = CellTowerGraphGenerator(graph_config)
            self.coords, _, self.W_matrix = generator.generate()
        
        if self.config.verbose:
            num_edges = np.count_nonzero(self.W_matrix) // 2
            total_weight = np.sum(self.W_matrix) / 2
            print(f"\nGraph Data Prepared:")
            print(f"  Nodes: {len(self.coords)}")
            print(f"  Edges: {num_edges}")
            print(f"  Total Edge Weight: {total_weight:.2f}")
            print(f"  Target Partitions: {self.config.num_partitions}")
        
        return self.coords, self.W_matrix
    
    def _evaluate(self, assignments: np.ndarray, method_name: str,
                  runtime: float, additional: Dict = None) -> BenchmarkResult:
        """
        Evaluate a partition assignment and create BenchmarkResult.
        
        Args:
            assignments: Partition assignments (N,)
            method_name: Name of the algorithm
            runtime: Execution time in seconds
            additional: Any additional info to store
            
        Returns:
            BenchmarkResult with all computed metrics
        """
        metrics = compute_extended_metrics(
            assignments, self.W_matrix, self.config.num_partitions
        )
        
        result = BenchmarkResult(
            method_name=method_name,
            assignments=assignments,
            edge_cut=metrics['edge_cut'],
            edge_cut_ratio=metrics['edge_cut_ratio'],
            load_balance_var=metrics['load_balance_var'],
            balance_ratio=metrics['balance_ratio'],
            normalized_cut=metrics['normalized_cut'],
            modularity=metrics['modularity'],
            partition_sizes=metrics['partition_sizes'],
            runtime=runtime,
            additional=additional or {}
        )
        
        self.results.append(result)
        return result
    
    def run_all(self) -> List[BenchmarkResult]:
        """
        Run all configured benchmark algorithms.
        
        Returns:
            List of BenchmarkResult objects
        """
        if self.coords is None or self.W_matrix is None:
            self.prepare_data()
        
        self.results = []
        
        if self.config.verbose:
            print("\n" + "=" * 70)
            print("BENCHMARKING MEC PLACEMENT ALGORITHMS")
            print("=" * 70)
        
        # 1. MECP-GAP (Our Method)
        if self.config.run_mecp_gap:
            self._run_mecp_gap()
        
        # 2. MECP-GAP with SSC-aware loss
        if self.config.run_mecp_gap_ssc:
            self._run_mecp_gap_ssc()
        
        # 3. METIS
        if self.config.run_metis:
            self._run_metis()
        
        # 4. Greedy (KGGGP)
        if self.config.run_greedy:
            self._run_greedy()
        
        # 5. PBPA (DRL)
        if self.config.run_pbpa:
            self._run_pbpa()
        
        # 6. Random
        if self.config.run_random:
            self._run_random()
        
        return self.results
    
    def _run_mecp_gap(self):
        """Run MECP-GAP (standard loss)."""
        if self.config.verbose:
            print(f"\n[1/6] Running MECP-GAP (Standard)...")
        
        try:
            from training.trainer import MECPTrainer, TrainingConfig
            
            start_time = time.time()
            
            train_config = TrainingConfig(
                num_partitions=self.config.num_partitions,
                num_epochs=self.config.gap_epochs,
                learning_rate=self.config.gap_lr,
                hidden_feats=self.config.gap_hidden,
                alpha=self.config.gap_alpha,
                beta=self.config.gap_beta,
                gamma=self.config.gap_gamma,
                log_interval=max(1, self.config.gap_epochs)  # Suppress per-epoch logging
            )
            
            trainer = MECPTrainer(train_config)
            results = trainer.train(self.coords, self.W_matrix, verbose=False)
            
            runtime = time.time() - start_time
            assignments = results['final_assignments']
            
            result = self._evaluate(assignments, "MECP-GAP", runtime, {
                'final_loss': results['final_loss'],
                'training_epochs': self.config.gap_epochs
            })
            
            if self.config.verbose:
                self._print_result(result)
                
        except Exception as e:
            warnings.warn(f"MECP-GAP failed: {e}")
            if self.config.verbose:
                print(f"  FAILED: {e}")
    
    def _run_mecp_gap_ssc(self):
        """Run MECP-GAP with SSC-aware loss function."""
        if self.config.verbose:
            print(f"\n[2/6] Running MECP-GAP (SSC-Aware)...")
        
        try:
            import torch
            from models.mecp_gap_model import MECP_GAP_Model
            from models.loss_functions import MECP_Loss_SSC
            from training.trainer import prepare_graph_data
            
            start_time = time.time()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Prepare data (uses row-normalized W as features, dim=N)
            features, edge_index, edge_weight, W_tensor = prepare_graph_data(
                self.coords, self.W_matrix, device
            )
            coords_tensor = torch.tensor(self.coords, dtype=torch.float32, device=device)
            num_nodes = len(self.coords)
            in_feats = features.shape[1]  # N for weight_row features
            
            # Initialize model
            model = MECP_GAP_Model(
                in_feats=in_feats,
                hidden_feats=self.config.gap_hidden,
                out_feats=self.config.gap_hidden,
                num_partitions=self.config.num_partitions,
                num_layers=2,
                aggregator='mean'
            ).to(device)
            
            # Auto-compute alpha for SSC to balance cut vs balance losses
            if self.config.gap_alpha < 0:
                total_weight = float(W_tensor.sum())
                ssc_alpha = (num_nodes ** 2 / self.config.num_partitions) / (
                    (1 - 1.0 / self.config.num_partitions) * total_weight + 1e-8
                )
            else:
                ssc_alpha = self.config.gap_alpha
            
            # SSC-aware loss (normalize_cut=False to match auto-alpha scaling)
            criterion = MECP_Loss_SSC(
                alpha=ssc_alpha,
                beta=self.config.gap_beta,
                gamma=self.config.gap_gamma,
                delta=0.0005,
                mode1_weight=0.3,
                mode2_weight=0.3,
                mode3_weight=0.4,
                normalize_cut=False
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.gap_lr)
            
            # Training loop
            for epoch in range(self.config.gap_epochs):
                model.train()
                probs = model(features, edge_index, edge_weight)
                loss, loss_dict = criterion(
                    probs, W_tensor, num_nodes, self.config.num_partitions, coords_tensor
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Final inference
            model.eval()
            with torch.no_grad():
                final_probs = model(features, edge_index, edge_weight)
                assignments = torch.argmax(final_probs, dim=1).cpu().numpy()
            
            runtime = time.time() - start_time
            
            result = self._evaluate(assignments, "MECP-GAP (SSC)", runtime, {
                'final_loss': loss.item(),
                'ssc_loss': loss_dict['ssc_loss'].item() if 'ssc_loss' in loss_dict else 0,
                'training_epochs': self.config.gap_epochs
            })
            
            if self.config.verbose:
                self._print_result(result)
                
        except Exception as e:
            warnings.warn(f"MECP-GAP (SSC) failed: {e}")
            if self.config.verbose:
                print(f"  FAILED: {e}")
    
    def _run_metis(self):
        """Run METIS baseline."""
        if self.config.verbose:
            print(f"\n[3/6] Running METIS...")
        
        try:
            from baselines.metis_baseline import MetisPartitioner, MetisFallback
            
            start_time = time.time()
            
            try:
                partitioner = MetisPartitioner(
                    num_partitions=self.config.num_partitions,
                    seed=self.config.seed
                )
                assignments = partitioner.fit(self.W_matrix)
                runtime = time.time() - start_time
                additional = {'cut_weight_metis': partitioner.get_cut_weight()}
                
            except ImportError:
                # Fallback to spectral partitioning
                if self.config.verbose:
                    print("  pymetis not available, using spectral fallback...")
                assignments = MetisFallback.spectral_partition(
                    self.W_matrix, self.config.num_partitions
                )
                runtime = time.time() - start_time
                additional = {'fallback': 'spectral'}
            
            result = self._evaluate(assignments, "METIS", runtime, additional)
            
            if self.config.verbose:
                self._print_result(result)
                
        except Exception as e:
            warnings.warn(f"METIS failed: {e}")
            if self.config.verbose:
                print(f"  FAILED: {e}")
    
    def _run_greedy(self):
        """Run Greedy (KGGGP) baseline."""
        if self.config.verbose:
            print(f"\n[4/6] Running Greedy (KGGGP)...")
        
        try:
            from baselines.greedy_baseline import GreedyPartitioner
            
            start_time = time.time()
            
            partitioner = GreedyPartitioner(
                num_partitions=self.config.num_partitions,
                seed_method=self.config.greedy_seed_method,
                seed=self.config.seed
            )
            assignments = partitioner.fit(self.W_matrix, self.coords)
            
            runtime = time.time() - start_time
            
            result = self._evaluate(assignments, "Greedy (KGGGP)", runtime)
            
            if self.config.verbose:
                self._print_result(result)
                
        except Exception as e:
            warnings.warn(f"Greedy failed: {e}")
            if self.config.verbose:
                print(f"  FAILED: {e}")
    
    def _run_pbpa(self):
        """Run PBPA (DRL with PPO) baseline."""
        if self.config.verbose:
            print(f"\n[5/6] Running PBPA (DRL/PPO)...")
        
        try:
            from baselines.pbpa_baseline import PBPAPartitioner, PBPAConfig
            
            start_time = time.time()
            
            pbpa_config = PBPAConfig(
                num_partitions=self.config.num_partitions,
                max_episodes=self.config.pbpa_episodes,
                max_steps_per_episode=self.config.pbpa_steps,
                hidden_dim=128,
                seed=self.config.seed
            )
            
            partitioner = PBPAPartitioner(
                num_partitions=self.config.num_partitions,
                config=pbpa_config
            )
            assignments = partitioner.fit(self.W_matrix, self.coords, verbose=False)
            
            runtime = time.time() - start_time
            
            result = self._evaluate(assignments, "PBPA (DRL)", runtime, {
                'training_episodes': self.config.pbpa_episodes
            })
            
            if self.config.verbose:
                self._print_result(result)
                
        except Exception as e:
            warnings.warn(f"PBPA failed: {e}")
            if self.config.verbose:
                print(f"  FAILED: {e}")
    
    def _run_random(self):
        """Run Random baseline."""
        if self.config.verbose:
            print(f"\n[6/6] Running Random (Balanced)...")
        
        try:
            from baselines.random_baseline import run_random_multiple
            
            start_time = time.time()
            
            assignments, random_results = run_random_multiple(
                self.W_matrix,
                num_partitions=self.config.num_partitions,
                num_trials=10,
                balanced=True,
                seed=self.config.seed
            )
            
            runtime = time.time() - start_time
            
            result = self._evaluate(assignments, "Random (Best of 10)", runtime, {
                'avg_cut': random_results.get('avg_cut', 0),
                'std_cut': random_results.get('std_cut', 0)
            })
            
            if self.config.verbose:
                self._print_result(result)
                
        except Exception as e:
            warnings.warn(f"Random failed: {e}")
            if self.config.verbose:
                print(f"  FAILED: {e}")
    
    def _print_result(self, result: BenchmarkResult):
        """Print a single result."""
        print(f"  Edge Cut: {result.edge_cut:.4f} | "
              f"Cut Ratio: {result.edge_cut_ratio:.4f} | "
              f"Balance: {result.balance_ratio:.3f} | "
              f"Modularity: {result.modularity:.4f} | "
              f"Time: {result.runtime:.3f}s")
        print(f"  Partition Sizes: {result.partition_sizes}")
    
    def print_comparison_table(self):
        """
        Print a formatted comparison table of all results.
        
        Reproduces the paper's Table format comparing all baselines.
        """
        if not self.results:
            print("No results to display. Run run_all() first.")
            return
        
        print("\n" + "=" * 95)
        print("COMPARATIVE BENCHMARK RESULTS")
        print("=" * 95)
        
        # Header
        header = f"{'Method':<22} {'Edge Cut':>10} {'Cut Ratio':>10} " \
                 f"{'Balance':>9} {'Norm Cut':>10} {'Modul.':>8} {'Time(s)':>9}"
        print(header)
        print("-" * 95)
        
        # Sort results by edge cut for easy comparison
        sorted_results = sorted(self.results, key=lambda r: r.edge_cut)
        
        best_cut = sorted_results[0].edge_cut if sorted_results else float('inf')
        
        for result in sorted_results:
            # Mark best edge cut with *
            marker = " *" if result.edge_cut == best_cut else "  "
            
            row = f"{result.method_name:<22} " \
                  f"{result.edge_cut:>10.4f} " \
                  f"{result.edge_cut_ratio:>10.4f} " \
                  f"{result.balance_ratio:>9.3f} " \
                  f"{result.normalized_cut:>10.4f} " \
                  f"{result.modularity:>8.4f} " \
                  f"{result.runtime:>9.3f}" \
                  f"{marker}"
            print(row)
        
        print("-" * 95)
        print("* = Best edge cut. Balance closer to 1.0 is better. Higher modularity is better.")
        
        # Partition sizes
        print(f"\n{'Method':<22} {'Partition Sizes'}")
        print("-" * 60)
        for result in sorted_results:
            print(f"{result.method_name:<22} {result.partition_sizes}")
        
        print("=" * 95)
    
    def save_results(self, output_dir: Optional[str] = None):
        """
        Save benchmark results to JSON and numpy files.
        
        Args:
            output_dir: Output directory (uses config.output_dir if None)
        """
        output_dir = output_dir or self.config.output_dir
        output_path = Path(output_dir) / 'metrics'
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary JSON
        summary = {
            'config': {
                'num_nodes': self.config.num_nodes,
                'num_partitions': self.config.num_partitions,
                'area_size': self.config.area_size,
                'seed': self.config.seed
            },
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(output_path / 'benchmark_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save assignments
        for result in self.results:
            safe_name = result.method_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            np.save(output_path / f'assignments_{safe_name}.npy', result.assignments)
        
        if self.config.verbose:
            print(f"\nResults saved to: {output_path}")
    
    def visualize_comparison(self, save_path: Optional[str] = None, show: bool = True):
        """
        Create side-by-side visualization of all methods' partitions.
        
        Args:
            save_path: Path to save the figure
            show: Whether to display
        """
        if not self.results:
            print("No results to visualize.")
            return
        
        try:
            from utils.visualization import visualize_comparison as viz_compare
            
            assignments_list = [r.assignments for r in self.results]
            method_names = [r.method_name for r in self.results]
            
            viz_compare(
                self.coords, self.W_matrix,
                assignments_list, method_names,
                figsize=(6 * len(self.results), 6),
                save_path=save_path,
                show=show
            )
        except Exception as e:
            warnings.warn(f"Visualization failed: {e}")
    
    def visualize_metrics_bar(self, save_path: Optional[str] = None, show: bool = True):
        """
        Create bar chart comparison of key metrics across methods.
        
        Args:
            save_path: Path to save figure
            show: Whether to display
        """
        if not self.results:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            methods = [r.method_name for r in self.results]
            edge_cuts = [r.edge_cut_ratio for r in self.results]
            balance = [r.balance_ratio for r in self.results]
            modularity = [r.modularity for r in self.results]
            runtimes = [r.runtime for r in self.results]
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            x = np.arange(len(methods))
            bar_width = 0.6
            
            # Edge Cut Ratio
            ax1 = axes[0, 0]
            bars1 = ax1.bar(x, edge_cuts, bar_width, color='#FF6B6B', alpha=0.8)
            ax1.set_ylabel('Edge Cut Ratio')
            ax1.set_title('Edge Cut Ratio (Lower is Better)')
            ax1.set_xticks(x)
            ax1.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
            ax1.grid(axis='y', alpha=0.3)
            
            # Balance Ratio
            ax2 = axes[0, 1]
            bars2 = ax2.bar(x, balance, bar_width, color='#4ECDC4', alpha=0.8)
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Balance')
            ax2.set_ylabel('Balance Ratio')
            ax2.set_title('Load Balance Ratio (Closer to 1.0 is Better)')
            ax2.set_xticks(x)
            ax2.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            
            # Modularity
            ax3 = axes[1, 0]
            bars3 = ax3.bar(x, modularity, bar_width, color='#45B7D1', alpha=0.8)
            ax3.set_ylabel('Modularity')
            ax3.set_title('Modularity (Higher is Better)')
            ax3.set_xticks(x)
            ax3.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
            ax3.grid(axis='y', alpha=0.3)
            
            # Runtime
            ax4 = axes[1, 1]
            bars4 = ax4.bar(x, runtimes, bar_width, color='#96CEB4', alpha=0.8)
            ax4.set_ylabel('Runtime (s)')
            ax4.set_title('Execution Time (Lower is Better)')
            ax4.set_xticks(x)
            ax4.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
            ax4.grid(axis='y', alpha=0.3)
            
            plt.suptitle(f'MEC Placement Algorithm Comparison '
                        f'(N={self.config.num_nodes}, P={self.config.num_partitions})',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if show:
                plt.show()
            
            return fig
            
        except Exception as e:
            warnings.warn(f"Metrics visualization failed: {e}")


def run_benchmark(
    num_nodes: int = 200,
    num_partitions: int = 4,
    seed: int = 42,
    gap_epochs: int = 200,
    pbpa_episodes: int = 200,
    verbose: bool = True,
    save_results: bool = True,
    output_dir: str = 'results'
) -> List[BenchmarkResult]:
    """
    Convenience function to run the complete benchmark suite.
    
    Args:
        num_nodes: Number of base stations
        num_partitions: Number of MEC servers
        seed: Random seed
        gap_epochs: Training epochs for MECP-GAP
        pbpa_episodes: Training episodes for PBPA
        verbose: Print progress
        save_results: Save results to disk
        output_dir: Output directory
        
    Returns:
        List of BenchmarkResult objects
    """
    config = BenchmarkConfig(
        num_nodes=num_nodes,
        num_partitions=num_partitions,
        seed=seed,
        gap_epochs=gap_epochs,
        pbpa_episodes=pbpa_episodes,
        verbose=verbose,
        output_dir=output_dir
    )
    
    benchmark = MECPBenchmark(config)
    benchmark.prepare_data()
    results = benchmark.run_all()
    benchmark.print_comparison_table()
    
    if save_results:
        benchmark.save_results()
    
    return results


def run_scalability_benchmark(
    node_counts: List[int] = None,
    num_partitions: int = 4,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, List[Dict]]:
    """
    Run scalability benchmark across different graph sizes.
    
    Tests how each algorithm scales with increasing number of nodes.
    
    Args:
        node_counts: List of node counts to test
        num_partitions: Number of partitions
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Dictionary mapping method names to lists of metrics per size
    """
    if node_counts is None:
        node_counts = [50, 100, 200, 500]
    
    scalability_results = {}
    
    for n in node_counts:
        if verbose:
            print(f"\n{'='*70}")
            print(f"SCALABILITY TEST: {n} nodes")
            print(f"{'='*70}")
        
        config = BenchmarkConfig(
            num_nodes=n,
            num_partitions=num_partitions,
            seed=seed,
            gap_epochs=min(200, max(50, n)),
            pbpa_episodes=min(200, max(50, n)),
            verbose=verbose
        )
        
        benchmark = MECPBenchmark(config)
        benchmark.prepare_data()
        results = benchmark.run_all()
        
        for result in results:
            if result.method_name not in scalability_results:
                scalability_results[result.method_name] = []
            
            scalability_results[result.method_name].append({
                'num_nodes': n,
                **result.to_dict()
            })
    
    # Print scalability summary
    if verbose:
        print(f"\n{'='*70}")
        print("SCALABILITY SUMMARY")
        print(f"{'='*70}")
        
        for method, entries in scalability_results.items():
            print(f"\n{method}:")
            for entry in entries:
                print(f"  N={entry['num_nodes']:4d} | Cut: {entry['edge_cut']:.4f} | "
                      f"Balance: {entry['balance_ratio']:.3f} | "
                      f"Time: {entry['runtime']:.3f}s")
    
    return scalability_results
