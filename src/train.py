"""
MECP-GAP Training Script

This is the main entry point for training the MECP-GAP model.
Run this script to:
1. Generate synthetic 5G network data (or load existing)
2. Train the GNN-based partitioning model
3. Visualize and evaluate results

Usage:
    python train.py                     # Use default settings
    python train.py --num_nodes 200 --num_partitions 4
    python train.py --load_data data/processed/test_graph

Paper Reference: Section V - Experiments
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import json

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from data_generation.graph_generator import CellTowerGraphGenerator, GraphConfig
from training.trainer import MECPTrainer, TrainingConfig, train_mecp_gap
from utils.visualization import (
    visualize_results,
    visualize_training_progress,
    print_partition_report
)


def load_processed_data(data_dir: str):
    """
    Load pre-processed graph data from directory.
    
    Args:
        data_dir: Path to directory containing:
            - coords.npy: Node coordinates
            - weights.npy or adjacency.npy: Weight matrix
            - metadata.json: Graph metadata
            
    Returns:
        coords: Node coordinates (N, 2)
        W_matrix: Weight matrix (N, N)
        metadata: Metadata dictionary
    """
    data_path = Path(data_dir)
    
    coords = np.load(data_path / 'coords.npy')
    
    # Try different weight file names
    if (data_path / 'weights.npy').exists():
        W_matrix = np.load(data_path / 'weights.npy')
    elif (data_path / 'adjacency.npy').exists():
        W_matrix = np.load(data_path / 'adjacency.npy')
    else:
        raise FileNotFoundError("No weight matrix found (weights.npy or adjacency.npy)")
    
    # Load metadata if exists
    metadata = {}
    metadata_path = data_path / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return coords, W_matrix, metadata


def generate_synthetic_data(
    num_nodes: int = 200,
    area_size: float = 10.0,
    gamma: float = 1.5,
    seed: int = 42
):
    """
    Generate synthetic 5G network graph data.
    
    Args:
        num_nodes: Number of base stations
        area_size: Size of area (km x km)
        gamma: Gravity model distance decay
        seed: Random seed
        
    Returns:
        coords: Node coordinates (N, 2)
        W_matrix: Mobility weight matrix (N, N)
    """
    config = GraphConfig(
        num_nodes=num_nodes,
        area_size=area_size,
        gamma=gamma,
        mode='synthetic',
        seed=seed
    )
    
    generator = CellTowerGraphGenerator(config)
    coords, G, W_matrix = generator.generate()
    
    return coords, W_matrix


def main():
    parser = argparse.ArgumentParser(
        description='Train MECP-GAP model for 5G network partitioning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--load_data', type=str, default=None,
                           help='Path to pre-processed data directory')
    data_group.add_argument('--num_nodes', type=int, default=200,
                           help='Number of nodes (if generating synthetic)')
    data_group.add_argument('--area_size', type=float, default=10.0,
                           help='Area size in km')
    data_group.add_argument('--seed', type=int, default=42,
                           help='Random seed')
    
    # Model arguments
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--num_partitions', type=int, default=4,
                            help='Number of partitions (MEC servers)')
    model_group.add_argument('--hidden_feats', type=int, default=128,
                            help='Hidden layer dimension')
    model_group.add_argument('--num_layers', type=int, default=2,
                            help='Number of GNN layers')
    model_group.add_argument('--feature_type', type=str, default='weight_row',
                            choices=['weight_row', 'coords'],
                            help='Node feature type: weight_row (paper, dim=N) or coords (dim=2)')
    
    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--num_epochs', type=int, default=200,
                            help='Number of training epochs')
    train_group.add_argument('--learning_rate', type=float, default=0.01,
                            help='Learning rate')
    train_group.add_argument('--alpha', type=float, default=-1.0,
                            help='Weight for edge cut loss (-1 = auto-compute per paper)')
    train_group.add_argument('--beta', type=float, default=1.0,
                            help='Weight for balance loss')
    train_group.add_argument('--gamma', type=float, default=0.0,
                            help='Weight for entropy regularization (0 = disabled; -0.1 = confident predictions)')
    
    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--save_dir', type=str, default='results',
                             help='Directory to save results')
    output_group.add_argument('--no_visualize', action='store_true',
                             help='Skip visualization')
    output_group.add_argument('--save_plots', action='store_true',
                             help='Save plots to files')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 60)
    print("MECP-GAP Training")
    print("=" * 60)
    
    # Step 1: Load or generate data
    print("\n[Step 1] Preparing Data...")
    
    if args.load_data:
        print(f"Loading data from: {args.load_data}")
        coords, W_matrix, metadata = load_processed_data(args.load_data)
        print(f"Loaded graph with {len(coords)} nodes")
    else:
        print(f"Generating synthetic graph with {args.num_nodes} nodes...")
        coords, W_matrix = generate_synthetic_data(
            num_nodes=args.num_nodes,
            area_size=args.area_size,
            seed=args.seed
        )
        print(f"Generated graph: {len(coords)} nodes, {np.count_nonzero(W_matrix)//2} edges")
    
    # Step 2: Configure and train model
    print("\n[Step 2] Training Model...")
    
    config = TrainingConfig(
        in_feats=-1,  # Auto-detect from data (N for weight_row features)
        hidden_feats=args.hidden_feats,
        num_partitions=args.num_partitions,
        num_layers=args.num_layers,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        feature_type=getattr(args, 'feature_type', 'weight_row'),
        log_interval=50,
        checkpoint_dir=f"{args.save_dir}/checkpoints" if args.save_plots else None
    )
    
    trainer = MECPTrainer(config)
    results = trainer.train(coords, W_matrix, verbose=True)
    
    # Step 3: Evaluate results
    print("\n[Step 3] Evaluating Results...")
    
    final_assignments = results['final_assignments']
    print_partition_report(W_matrix, final_assignments, "MECP-GAP", num_partitions=args.num_partitions)
    
    # Step 4: Visualize
    if not args.no_visualize:
        print("\n[Step 4] Visualizing Results...")
        
        # Create save directory if saving
        if args.save_plots:
            save_dir = Path(args.save_dir) / 'plots'
            save_dir.mkdir(parents=True, exist_ok=True)
            partition_path = str(save_dir / 'partition_result.png')
            training_path = str(save_dir / 'training_progress.png')
        else:
            partition_path = None
            training_path = None
        
        # Visualize partition result
        visualize_results(
            coords, W_matrix, final_assignments,
            num_partitions=args.num_partitions,
            save_path=partition_path,
            show=True
        )
        
        # Visualize training progress
        visualize_training_progress(
            results['history'],
            save_path=training_path,
            show=True
        )
    
    # Save final assignments
    if args.save_plots:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(save_dir / 'assignments.npy', final_assignments)
        np.save(save_dir / 'partition_probs.npy', results['final_probs'])
        
        # Save summary
        summary = {
            'num_nodes': len(coords),
            'num_partitions': args.num_partitions,
            'final_loss': results['final_loss'],
            'training_time': results['training_time'],
            'partition_sizes': [int(np.sum(final_assignments == p)) 
                               for p in range(args.num_partitions)]
        }
        with open(save_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to: {save_dir}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return final_assignments, results


if __name__ == '__main__':
    main()
