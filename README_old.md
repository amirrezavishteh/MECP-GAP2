# MECP-GAP: Mobility-aware Edge Computing Partitioning using Graph Neural Networks

## Overview

MECP-GAP is a Graph Neural Network (GNN) based solution for partitioning 5G cellular networks into optimal service regions for Mobile Edge Computing (MEC) servers. The goal is to minimize user handover costs (when users move between cell towers served by different MEC servers) while maintaining balanced server loads.

### Problem Statement

In 5G networks, Mobile Edge Computing (MEC) servers provide low-latency services by processing data close to users. The challenge is:
- **How to assign base stations (gNBs) to MEC servers** such that:
  1. Users moving between cells cause minimal handovers between different MEC servers
  2. Each MEC server handles approximately equal traffic load

This is formulated as a **weighted graph partitioning problem** where:
- **Nodes** = Base stations (cell towers)
- **Edges** = Neighbor relationships (adjacent cells where users can move)
- **Edge Weights** = Mobility traffic (number of users moving between cells)

---

## Project Structure

```
MCGAP/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/
│   ├── graphs/              # Saved graph data
│   ├── processed/           # Pre-processed datasets
│   │   └── test_graph/      # Example test graph
│   └── raw/                 # Raw data files
├── results/
│   ├── metrics/             # Evaluation metrics
│   └── plots/               # Generated visualizations
└── src/
    ├── __init__.py
    ├── train.py             # Main training script (entry point)
    ├── quick_test.py        # Quick validation script
    ├── baselines/           # Baseline comparison methods
    │   ├── greedy_baseline.py
    │   ├── metis_baseline.py
    │   └── random_baseline.py
    ├── data_generation/     # Data generation pipeline
    │   ├── dataset_builder.py
    │   ├── graph_generator.py
    │   └── mobility_generator.py
    ├── models/              # GNN model architecture
    │   ├── mecp_gap_model.py
    │   └── loss_functions.py
    ├── training/            # Training loop and utilities
    │   ├── trainer.py
    │   └── training.py
    └── utils/               # Visualization and utilities
        ├── utils.py
        └── visualization.py
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA (optional, for GPU acceleration)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd MCGAP
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

#### Option A: Quick Install (CPU only)
```bash
# Install PyTorch (CPU)
pip install torch torchvision torchaudio

# Install PyTorch Geometric
pip install torch-geometric

# Install remaining dependencies
pip install -r requirements.txt
```

#### Option B: GPU Support (CUDA 11.8)
```bash
# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric with CUDA support
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric

# Install remaining dependencies
pip install -r requirements.txt
```

#### Option C: GPU Support (CUDA 12.1)
```bash
# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric with CUDA support
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu121.html
pip install torch-geometric

# Install remaining dependencies
pip install -r requirements.txt
```

#### Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

**Note:** For other CUDA versions, visit [PyTorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

---

## Quick Start

### Run Quick Test (Recommended First)

Verify the installation works correctly:

```bash
cd src
python quick_test.py
```

Expected output:
```
Testing imports...
All imports successful!

Generating test data...
Generated: 50 nodes, ~100 edges

Running quick training test (20 epochs)...
...
Final metrics:
  Edge cut ratio: ~15-25%
  Partition sizes: [~12, ~13, ~12, ~13]

Test completed successfully!
```

### Train with Default Settings

```bash
cd src
python train.py
```

This will:
1. Generate a synthetic 5G network with 200 base stations
2. Train the GNN model for 200 epochs
3. Display partition results and training progress

### Train with Custom Parameters

```bash
python train.py --num_nodes 500 --num_partitions 8 --num_epochs 300 --save_plots
```

### Load Pre-processed Data

```bash
python train.py --load_data ../data/processed/test_graph --num_partitions 4
```

---

## Detailed File Descriptions

### 1. Main Entry Point: `src/train.py`

**Purpose:** Main script to run the complete training pipeline.

**Key Functions:**
- `load_processed_data()`: Load pre-saved graph data from disk
- `generate_synthetic_data()`: Create new synthetic 5G network data
- `main()`: Parse arguments, run training, evaluate and visualize results

**Command Line Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--load_data` | None | Path to pre-processed data directory |
| `--num_nodes` | 200 | Number of base stations (if generating) |
| `--area_size` | 10.0 | Simulation area size in km |
| `--num_partitions` | 4 | Number of MEC servers/partitions |
| `--hidden_feats` | 128 | GNN hidden layer dimension |
| `--num_layers` | 2 | Number of GNN layers |
| `--num_epochs` | 200 | Training epochs |
| `--learning_rate` | 0.01 | Optimizer learning rate |
| `--alpha` | 0.001 | Edge cut loss weight |
| `--beta` | 1.0 | Load balance loss weight |
| `--gamma` | -0.1 | Entropy regularization weight |
| `--save_dir` | results | Directory for output files |
| `--save_plots` | False | Save visualization plots |
| `--no_visualize` | False | Skip visualization |

---

### 2. Model Architecture: `src/models/mecp_gap_model.py`

**Purpose:** Implements the Graph Neural Network model for partitioning.

**Architecture Overview:**

```
Input: Node Features (coordinates) [N x 2]
           │
           ▼
┌─────────────────────────────┐
│   Graph Embedding Module    │
│   (2-layer GraphSAGE)       │
│   ┌───────────────────┐     │
│   │ GraphSAGE Layer 1 │     │  2 → 128 dimensions
│   │ (weighted agg.)   │     │
│   └───────────────────┘     │
│           │ ReLU            │
│   ┌───────────────────┐     │
│   │ GraphSAGE Layer 2 │     │  128 → 128 dimensions
│   │ (weighted agg.)   │     │
│   └───────────────────┘     │
└─────────────────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Graph Partitioning Module  │
│         (MLP)               │
│   Linear(128 → 64) + ReLU   │
│   Linear(64 → P)            │
│   Softmax                   │
└─────────────────────────────┘
           │
           ▼
Output: Partition Probabilities [N x P]
```

**Key Classes:**

| Class | Description |
|-------|-------------|
| `GraphSAGELayer` | Custom GraphSAGE layer with edge weight support |
| `MECP_GAP_Model` | Main model (native PyTorch implementation) |
| `MECP_GAP_Model_DGL` | Alternative using DGL library |
| `MECP_GAP_Model_PyG` | Alternative using PyTorch Geometric |

**Key Innovation:** Weighted aggregation based on mobility traffic - nodes with higher edge weights (more user traffic) have more influence on neighbor embeddings.

---

### 3. Loss Functions: `src/models/loss_functions.py`

**Purpose:** Implements the training objectives to optimize.

**Main Loss (Equation 9 from paper):**

$$L_{total} = \alpha \cdot L_{cut} + \beta \cdot L_{balance} + \gamma \cdot L_{entropy}$$

**Loss Components:**

| Loss | Formula | Purpose |
|------|---------|---------|
| **Edge Cut Loss** | $L_{cut} = \sum_{i,j} W_{ij} \cdot (1 - \sum_k X_{ik} X_{jk})$ | Minimize cutting high-traffic edges |
| **Balance Loss** | $L_{balance} = \sum_k (\|S_k\| - N/P)^2$ | Keep partition sizes equal |
| **Entropy Loss** | $L_{entropy} = -\sum_{i,k} X_{ik} \log(X_{ik})$ | Encourage confident predictions |

**Default Hyperparameters:**
- α = 0.001 (edge cut weight - small due to scale)
- β = 1.0 (balance weight)
- γ = -0.1 (negative to encourage low entropy/confident predictions)

**Additional Loss Functions:**
- `NormalizedCutLoss`: Normalizes by partition volume
- `RatioCutLoss`: Normalizes by partition size
- `ModularityLoss`: Community detection objective

---

### 4. Data Generation: `src/data_generation/graph_generator.py`

**Purpose:** Generate realistic 5G network topology and mobility data.

**Three-Layer Generation Pipeline:**

#### Layer 1: Geometry (Base Station Locations)
- **Synthetic Mode:** Poisson Point Process (uniform random distribution)
- **Real Mode:** Load from CSV/JSON (e.g., OpenCellID data)

#### Layer 2: Topology (Neighbor Relationships)
- Uses **Voronoi Diagram** to determine cell boundaries
- Two stations are neighbors if their Voronoi cells share a boundary
- This models realistic handover possibilities

#### Layer 3: Mobility (Edge Weights using Gravity Model)

$$W_{uv} = scale \cdot \frac{Mass_u \cdot Mass_v}{Distance_{uv}^\gamma}$$

Where:
- $Mass$ = Proxy for population/traffic capacity (derived from cell area)
- $Distance$ = Euclidean distance between stations
- $\gamma$ = Friction coefficient (typical: 1.0-2.0, default: 1.5)

**Key Class: `CellTowerGraphGenerator`**

```python
config = GraphConfig(num_nodes=200, area_size=10.0, gamma=1.5, seed=42)
generator = CellTowerGraphGenerator(config)
coords, graph, W_matrix = generator.generate()
```

---

### 5. Training Loop: `src/training/trainer.py`

**Purpose:** Manages the complete training process.

**Key Components:**

| Component | Description |
|-----------|-------------|
| `TrainingConfig` | Dataclass with all hyperparameters |
| `prepare_graph_data()` | Convert numpy arrays to PyTorch tensors |
| `training_step()` | Single iteration: forward → loss → backward → update |
| `MECPTrainer` | Main trainer class with train/inference methods |
| `train_mecp_gap()` | Convenience function for quick training |

**Training Flow:**

```
1. Prepare Data
   - Convert coordinates to features [N x 2]
   - Build edge_index from weight matrix [2 x E]
   - Extract edge_weights [E]

2. Training Loop (200 epochs default)
   For each epoch:
   a. Forward Pass: probs = model(features, edge_index, edge_weight)
   b. Compute Loss: L = α*cut + β*balance + γ*entropy
   c. Backward Pass: loss.backward()
   d. Optimizer Step: optimizer.step()
   e. Log progress every 50 epochs

3. Inference
   - Get final partition probabilities
   - Convert to hard assignments: argmax(probs, dim=1)
```

---

### 6. Visualization: `src/utils/visualization.py`

**Purpose:** Visualize partitioning results and training progress.

**Key Functions:**

| Function | Output |
|----------|--------|
| `visualize_results()` | Graph plot with colored partitions, edge thickness = traffic |
| `visualize_training_progress()` | Loss curves and partition size evolution |
| `visualize_comparison()` | Side-by-side comparison of multiple methods |
| `compute_partition_metrics()` | Calculate edge cut, balance metrics |
| `print_partition_report()` | Formatted text report |

**Visualization Features:**
- **Node Colors:** Each partition has a distinct color
- **Edge Thickness:** Thicker = higher mobility traffic
- **Cut Edges:** Shown in red dashed lines (edges crossing partition boundaries)

---

### 7. Baseline Methods: `src/baselines/`

For comparison, three baseline partitioning methods are implemented:

#### METIS (`metis_baseline.py`)
Industry-standard multilevel graph partitioner:
1. **Coarsening:** Heavy Edge Matching to protect high-weight edges
2. **Initial Partitioning:** Spectral bisection on coarsest graph
3. **Refinement:** Fiduccia-Mattheyses move-based optimization

#### Greedy KGGGP (`greedy_baseline.py`)
K-way Greedy Graph Growing Partitioning:
1. Select P seed nodes (farthest-apart strategy)
2. Iteratively grow partitions by adding best-connected nodes
3. Balance by monitoring partition capacities

#### Random (`random_baseline.py`)
Lower-bound baseline:
- **Uniform:** Random assignment to partitions
- **Balanced:** Round-robin shuffle for equal sizes

---

## Usage Examples

### Example 1: Basic Training

```python
from training.trainer import train_mecp_gap
from data_generation.graph_generator import CellTowerGraphGenerator, GraphConfig

# Generate data
config = GraphConfig(num_nodes=200, seed=42)
generator = CellTowerGraphGenerator(config)
coords, _, W_matrix = generator.generate()

# Train model
assignments, results = train_mecp_gap(
    coords, W_matrix,
    num_partitions=4,
    num_epochs=200,
    verbose=True
)

print(f"Final partition sizes: {results['history'][-1]['partition_sizes']}")
```

### Example 2: Custom Model Configuration

```python
from models.mecp_gap_model import MECP_GAP_Model
from models.loss_functions import MECP_Loss
import torch

# Create model
model = MECP_GAP_Model(
    in_feats=2,
    hidden_feats=256,  # Larger embeddings
    num_partitions=8,
    num_layers=3,      # Deeper GNN
    dropout=0.1
)

# Custom loss weights
loss_fn = MECP_Loss(
    alpha=0.01,   # Higher edge cut weight
    beta=0.5,     # Lower balance weight
    gamma=-0.05   # Less aggressive entropy
)
```

### Example 3: Compare with Baselines

```python
from baselines.metis_baseline import MetisPartitioner
from baselines.greedy_baseline import GreedyPartitioner
from baselines.random_baseline import RandomPartitioner
from utils.visualization import visualize_comparison

# Get MECP-GAP result (from training)
mecp_assignments = results['final_assignments']

# METIS baseline
metis = MetisPartitioner(num_partitions=4)
metis_assignments = metis.fit(W_matrix)

# Greedy baseline
greedy = GreedyPartitioner(num_partitions=4)
greedy_assignments = greedy.fit(W_matrix, coords)

# Random baseline
random = RandomPartitioner(num_partitions=4, balanced=True)
random_assignments = random.fit(W_matrix)

# Visual comparison
visualize_comparison(
    coords, W_matrix,
    [mecp_assignments, metis_assignments, greedy_assignments, random_assignments],
    ['MECP-GAP', 'METIS', 'Greedy', 'Random']
)
```

---

## Key Concepts

### Soft vs Hard Partitioning

- **Soft (during training):** Model outputs probabilities $X_{ip}$ = probability node $i$ belongs to partition $p$
- **Hard (for inference):** Final assignment = $argmax_p(X_{ip})$

The soft formulation allows gradient-based optimization.

### Why GraphSAGE with Edge Weights?

Standard GNNs treat all neighbors equally. In MECP-GAP:
- Neighbors with **high mobility traffic** (large $W_{ij}$) should have **more influence**
- This encourages co-assignment of heavily connected nodes

### Loss Function Intuition

- **Edge Cut Loss:** If two nodes have high traffic between them ($W_{ij}$ large) and different partition probabilities, the loss is high
- **Balance Loss:** Penalizes deviation from ideal size $N/P$ per partition
- **Trade-off:** α and β control the balance between cut minimization and load balancing

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   python train.py --hidden_feats 64 --num_layers 1
   ```
   Reduce model size or use CPU.

2. **PyTorch Geometric Import Error**
   Follow official installation guide with correct CUDA version:
   ```bash
   pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
   ```

3. **No Visualization (matplotlib errors)**
   ```python
   import matplotlib
   matplotlib.use('Agg')  # Use non-interactive backend
   ```

4. **Poor Partition Quality**
   - Increase `num_epochs` (try 500+)
   - Adjust `alpha` and `beta` weights
   - Check if data is connected (isolated nodes cause issues)

---

## References

This implementation is based on research on Graph Neural Networks for graph partitioning and Mobile Edge Computing optimization. Key concepts include:

- **GraphSAGE:** Hamilton et al., "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)
- **Graph Partitioning:** Karypis & Kumar, "METIS: A Software Package for Partitioning Unstructured Graphs"
- **Gravity Model:** Wilson, "A Statistical Theory of Spatial Distribution Models"

---

## License

[Specify your license here]

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

---

## Contact

[Your contact information]
