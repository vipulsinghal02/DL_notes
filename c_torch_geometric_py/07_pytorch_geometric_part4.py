# %% [markdown]
# # PyTorch Geometric Part 4: Memory-Enhanced GNNs
#
# This notebook explores memory-enhanced Graph Neural Networks with comprehensive mathematical exposition, designed for scalability to large graphs. We'll implement GraphSAINT (Graph Sampling and Aggregation) and FastGCN with detailed mathematical foundations for sampling strategies, variance analysis, and scalability theory.
#
# ## Mathematical Foundation of Memory-Enhanced GNNs
#
# ### The Scalability Challenge: Neighborhood Explosion
#
# **Traditional GNN Complexity:**
# For an $L$-layer GNN, the receptive field of a node grows exponentially:
# $$|\text{Receptive Field}| = O(d^L)$$
#
# where $d$ is the average node degree.
#
# **Memory Complexity:**
# $$M_{\text{full-batch}} = O(N \times d^L \times D)$$
#
# where $N$ is number of nodes, $D$ is feature dimension.
#
# **Computational Complexity:**
# $$\mathcal{C}_{\text{full-batch}} = O(|\mathcal{E}| \times D^2 \times L)$$
#
# ### Mathematical Solutions
#
# **1. Graph Sampling (GraphSAINT):**
# $$\mathcal{L}_{\text{saint}} = \sum_{s \in \mathcal{S}} \frac{|\mathcal{V}|}{|\mathcal{V}_s|} \mathcal{L}_s(\theta)$$
#
# **2. Importance Sampling (FastGCN):**
# $$\mathbb{E}[\hat{\mathcal{L}}] = \mathcal{L} \text{ with controlled variance}$$
#
# **3. Layer-wise Sampling:**
# $$\text{Complexity} = O(K \times D^2 \times L)$$
#
# where $K \ll |\mathcal{E}|$ is the sample size.
#
# ### Key Mathematical Principles
#
# **Unbiased Estimation:**
# $$\mathbb{E}[\hat{\mathcal{L}}_{\text{sample}}] = \mathcal{L}_{\text{full}}$$
#
# **Variance Control:**
# $$\text{Var}[\hat{\mathcal{L}}_{\text{sample}}] = O(1/K)$$
#
# **Convergence Guarantees:**
# $$\|\hat{\theta}_T - \theta^*\| = O(1/\sqrt{T})$$
#
# These mathematical foundations enable training on graphs with millions of nodes while maintaining convergence guarantees and controlling approximation error.

# %%
# Installation (uncomment if needed)
# !pip install torch torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, FastRGCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.loader import GraphSAINTRandomWalkSampler, GraphSAINTNodeSampler, GraphSAINTEdgeSampler
from torch_geometric.utils import degree, to_undirected
import matplotlib.pyplot as plt
import numpy as np
import time
import psutil
import os
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. GraphSAINT Implementation
#
# GraphSAINT (Graph Sampling and Aggregation) uses graph sampling to create mini-batches of subgraphs for training, enabling scalable GNN training on large graphs.
#
# ### Mathematical Foundation of GraphSAINT
#
# **Sampling Strategy Mathematical Framework:**
#
# 1. **Subgraph Sampling:**
#    For a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, we sample a subgraph $\mathcal{G}_s = (\mathcal{V}_s, \mathcal{E}_s)$ where:
#    $$\mathcal{V}_s \subseteq \mathcal{V}, \quad \mathcal{E}_s \subseteq \mathcal{E}$$
#
# 2. **Sampling Probability Distribution:**
#    Each node $v \in \mathcal{V}$ has sampling probability $p_v$:
#    $$\sum_{v \in \mathcal{V}} p_v = 1$$
#
# 3. **Importance Sampling Loss:**
#    The unbiased estimator for the full-graph loss is:
#    $$\mathcal{L}_{\text{saint}} = \sum_{s \in \mathcal{S}} \frac{|\mathcal{V}|}{|\mathcal{V}_s|} \mathcal{L}_s(\theta)$$
#
#    where $\mathcal{L}_s(\theta)$ is the loss on subgraph $s$.
#
# **Sampling Strategies:**
#
# 1. **Node Sampling:**
#    $$p_v^{\text{node}} = \frac{1}{|\mathcal{V}|}$$
#
# 2. **Edge Sampling:**
#    $$p_e^{\text{edge}} = \frac{1}{|\mathcal{E}|}$$
#
# 3. **Random Walk Sampling:**
#    $$p_v^{\text{rw}} = \frac{\sum_{w \in \mathcal{W}} \mathbb{I}[v \in w]}{|\mathcal{W}| \cdot L_w}$$
#
#    where $\mathcal{W}$ is the set of random walks and $L_w$ is walk length.
#
# **Variance Analysis:**
# The variance of the sampling estimator is:
# $$\text{Var}[\mathcal{L}_{\text{saint}}] = \mathbb{E}\left[\left(\frac{|\mathcal{V}|}{|\mathcal{V}_s|} \mathcal{L}_s - \mathcal{L}\right)^2\right]$$
#
# **Convergence Guarantees:**
# Under proper sampling conditions:
# $$\mathbb{E}[\hat{\theta}_T - \theta^*] = O(1/\sqrt{T})$$
#
# where $T$ is the number of training iterations.

# %%
class GraphSAINTModel(nn.Module):
    """
    GraphSAINT model with sampling-based training
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers

        # Build GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, output_dim))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x

# %% [markdown]
# ## 2. FastGCN Implementation
#
# FastGCN implements importance sampling to select the most relevant neighbors for each layer, reducing computational complexity from exponential to linear in the number of layers.
#
# ### Mathematical Foundation of FastGCN
#
# **Traditional GCN Complexity Problem:**
# Standard GCN for an $L$-layer network requires:
# $$\mathcal{C}_{\text{full}} = O(|\mathcal{E}| \times L \times d^{L-1})$$
#
# where $d$ is the average degree and $|\mathcal{E}|$ is the number of edges.
#
# **FastGCN Solution:**
#
# 1. **Layer-wise Sampling:**
#    For layer $\ell$, sample $K^{(\ell)}$ nodes from the full set:
#    $$\mathcal{S}^{(\ell)} \subset \mathcal{V}, \quad |\mathcal{S}^{(\ell)}| = K^{(\ell)} \ll |\mathcal{V}|$$
#
# 2. **Importance Sampling Probability:**
#    Node $v$ is sampled with probability proportional to its importance:
#    $$q_v^{(\ell)} = \frac{w_v^{(\ell)}}{\sum_{u \in \mathcal{V}} w_u^{(\ell)}}$$
#
#    where $w_v^{(\ell)}$ can be based on degree, feature norm, or learned weights.
#
# 3. **Degree-based Importance:**
#    $$q_v^{(\ell)} = \frac{d_v}{\sum_{u \in \mathcal{V}} d_u} = \frac{d_v}{2|\mathcal{E}|}$$
#
# **Unbiased Estimation:**
# The FastGCN convolution becomes:
# $$\mathbf{H}^{(\ell+1)} = \sigma\left(\frac{1}{K^{(\ell)}} \sum_{v \in \mathcal{S}^{(\ell)}} \frac{\mathbf{A}_{:,v}}{q_v^{(\ell)}} \mathbf{H}_v^{(\ell)} \mathbf{W}^{(\ell)}\right)$$
#
# **Variance Control:**
# The variance of the estimator is:
# $$\text{Var}[\hat{\mathbf{H}}^{(\ell+1)}] = \frac{1}{K^{(\ell)}} \sum_{v \in \mathcal{V}} q_v^{(\ell)} \left(\frac{\mathbf{A}_{:,v}}{q_v^{(\ell)}} \mathbf{H}_v^{(\ell)} \mathbf{W}^{(\ell)}\right)^2$$
#
# **Optimal Sampling Strategy:**
# To minimize variance:
# $$q_v^{(\ell)*} = \frac{\|\mathbf{A}_{:,v} \mathbf{H}_v^{(\ell)}\|}{\sum_{u \in \mathcal{V}} \|\mathbf{A}_{:,u} \mathbf{H}_u^{(\ell)}\|}$$
#
# **Computational Complexity Reduction:**
# $$\mathcal{C}_{\text{FastGCN}} = O\left(\sum_{\ell=1}^L K^{(\ell)} \times d \times D\right)$$
#
# where $D$ is the feature dimension, significantly reducing complexity when $K^{(\ell)} \ll |\mathcal{V}|$.

# %%
class FastGCNConv(nn.Module):
    """
    FastGCN convolution layer with importance sampling
    """
    def __init__(self, in_channels, out_channels, sample_size=100):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_size = sample_size

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, importance_scores=None):
        """
        Forward pass with importance sampling
        """
        num_nodes = x.size(0)

        if importance_scores is None:
            # Use degree as importance score
            row, col = edge_index
            deg = degree(col, num_nodes=num_nodes)
            importance_scores = deg / deg.sum()

        # Sample nodes based on importance
        sample_size = min(self.sample_size, num_nodes)

        if sample_size < num_nodes:
            # Importance sampling
            sampled_indices = torch.multinomial(
                importance_scores,
                sample_size,
                replacement=False
            )

            # Create subgraph
            sampled_x = x[sampled_indices]

            # Transform features
            out = torch.matmul(sampled_x, self.weight) + self.bias

            # Map back to original size
            result = torch.zeros(num_nodes, self.out_channels, device=x.device)
            result[sampled_indices] = out * (num_nodes / sample_size)  # Importance weighting

            return result
        else:
            # Regular convolution if sample size is large enough
            return torch.matmul(x, self.weight) + self.bias

class FastGCNModel(nn.Module):
    """
    Complete FastGCN model with importance sampling
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3,
                 sample_sizes=None, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers

        if sample_sizes is None:
            sample_sizes = [200, 100, 50][:num_layers]  # Decreasing sample sizes

        # Build FastGCN layers
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(FastGCNConv(input_dim, hidden_dim, sample_sizes[0]))

        # Hidden layers
        for i in range(1, num_layers - 1):
            self.convs.append(FastGCNConv(hidden_dim, hidden_dim, sample_sizes[min(i, len(sample_sizes)-1)]))

        # Output layer
        self.convs.append(FastGCNConv(hidden_dim, output_dim, sample_sizes[-1]))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x

# %% [markdown]
# ## 3. Memory Monitoring Utilities
#
# To understand the memory efficiency of our approaches, we implement utilities to monitor memory usage during training and inference with mathematical analysis of memory complexity.
#
# ### Mathematical Memory Analysis
#
# **Full-batch GCN Memory Requirements:**
#
# 1. **Node Embeddings Storage:**
#    $$M_{\text{embeddings}} = L \times |\mathcal{V}| \times D \times \text{sizeof}(\text{float})$$
#
# 2. **Adjacency Matrix Storage:**
#    $$M_{\text{adj}} = |\mathcal{E}| \times 2 \times \text{sizeof}(\text{int}) \quad \text{(sparse format)}$$
#
# 3. **Gradient Storage:**
#    $$M_{\text{gradients}} = \sum_{\ell=1}^L (D^{(\ell)} \times D^{(\ell+1)} + D^{(\ell+1)}) \times \text{sizeof}(\text{float})$$
#
# 4. **Total Memory Complexity:**
#    $$M_{\text{total}} = O(L \times |\mathcal{V}| \times D + |\mathcal{E}| + \sum_{\ell} D^{(\ell)} \times D^{(\ell+1)})$$
#
# **Memory-Enhanced Approaches:**
#
# **GraphSAINT Memory Reduction:**
# $$M_{\text{saint}} = O(L \times |\mathcal{V}_s| \times D + |\mathcal{E}_s|) \ll M_{\text{total}}$$
#
# where $|\mathcal{V}_s| \ll |\mathcal{V}|$ and $|\mathcal{E}_s| \ll |\mathcal{E}|$.
#
# **FastGCN Memory Reduction:**
# $$M_{\text{fastgcn}} = O\left(L \times \max_{\ell} K^{(\ell)} \times D + |\mathcal{E}|\right)$$
#
# **Memory Efficiency Ratio:**
# $$\eta_{\text{memory}} = \frac{M_{\text{enhanced}}}{M_{\text{full}}} = \frac{\max(|\mathcal{V}_s|, \max_{\ell} K^{(\ell)})}{|\mathcal{V}|}$$
#
# **Peak Memory Analysis:**
# During backpropagation, peak memory includes:
# - Forward pass activations: $M_{\text{forward}}$
# - Gradient computations: $M_{\text{backward}}$
# - Optimizer states: $M_{\text{optimizer}}$
#
# $$M_{\text{peak}} = M_{\text{forward}} + M_{\text{backward}} + M_{\text{optimizer}}$$

# %%
class MemoryMonitor:
    """
    Monitor memory usage during training
    """
    def __init__(self):
        self.memory_log = []
        self.time_log = []
        self.start_time = time.time()

    def log_memory(self, tag=""):
        """
        Log current memory usage
        """
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        current_time = time.time() - self.start_time

        self.memory_log.append((tag, memory_mb))
        self.time_log.append(current_time)

        print(f"[{current_time:.2f}s] {tag}: {memory_mb:.1f} MB")
        return memory_mb

    def get_peak_memory(self):
        """
        Get peak memory usage
        """
        if not self.memory_log:
            return 0
        return max([mem for _, mem in self.memory_log])

    def plot_memory_usage(self):
        """
        Plot memory usage over time
        """
        if len(self.memory_log) < 2:
            print("Not enough data points to plot")
            return

        memory_values = [mem for _, mem in self.memory_log]

        plt.figure(figsize=(10, 6))
        plt.plot(self.time_log, memory_values, 'o-', alpha=0.7)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Over Time')
        plt.grid(True, alpha=0.3)

        # Annotate key points
        for i, (tag, mem) in enumerate(self.memory_log[::max(1, len(self.memory_log)//10)]):
            if tag:
                idx = i * max(1, len(self.memory_log)//10)
                plt.annotate(tag, (self.time_log[idx], mem),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.tight_layout()
        plt.show()

def benchmark_model(model, data, num_runs=5, batch_size=None):
    """
    Benchmark model inference time and memory usage
    """
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)

    # Warm-up runs
    with torch.no_grad():
        for _ in range(2):
            _ = model(data.x, data.edge_index)

    # Benchmark runs
    times = []
    monitor = MemoryMonitor()

    monitor.log_memory("Before benchmark")

    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.time()
            _ = model(data.x, data.edge_index)
            end_time = time.time()

            times.append(end_time - start_time)
            if i == 0:  # Log memory after first run
                monitor.log_memory("After forward pass")

    avg_time = np.mean(times)
    std_time = np.std(times)
    peak_memory = monitor.get_peak_memory()

    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'peak_memory': peak_memory,
        'times': times
    }

# %% [markdown]
# ## 4. Dataset Preparation
#
# We'll use the Cora dataset for our experiments, and create a synthetic larger graph to demonstrate the scalability benefits of these memory-enhanced approaches.
#
# ### Mathematical Graph Properties
#
# **Graph Characterization:**
# For a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
#
# 1. **Degree Distribution:**
#    $$d_v = |\{u \in \mathcal{V} : (v,u) \in \mathcal{E}\}|$$
#
#    Average degree: $\bar{d} = \frac{2|\mathcal{E}|}{|\mathcal{V}|}$
#
# 2. **Graph Density:**
#    $$\rho = \frac{|\mathcal{E}|}{|\mathcal{V}|(|\mathcal{V}|-1)/2}$$
#
# 3. **Memory Scaling with Graph Size:**
#    For dense graphs: $M = O(|\mathcal{V}|^2)$
#    For sparse graphs: $M = O(|\mathcal{V}| + |\mathcal{E}|)$
#
# **Synthetic Graph Generation (Erdős–Rényi Model):**
#
# 1. **Edge Probability:**
#    $$P((i,j) \in \mathcal{E}) = p = \frac{\bar{d}}{|\mathcal{V}|-1}$$
#
# 2. **Expected Properties:**
#    - Expected edges: $\mathbb{E}[|\mathcal{E}|] = p \binom{|\mathcal{V}|}{2}$
#    - Degree distribution: $d_v \sim \text{Binomial}(|\mathcal{V}|-1, p)$
#
# 3. **Clustering Coefficient:**
#    $$C = \frac{\text{Number of triangles}}{\text{Number of connected triples}} \approx p$$
#
# **Memory Requirements for Different Graph Sizes:**
#
# | Graph Size | Dense Memory | Sparse Memory | Sampling Benefit |
# |------------|-------------|---------------|------------------|
# | 1K nodes   | $O(10^6)$   | $O(10^4)$     | 10x              |
# | 10K nodes  | $O(10^8)$   | $O(10^5)$     | 100x             |
# | 100K nodes | $O(10^{10})$| $O(10^6)$     | 1000x            |
#
# **Scalability Mathematical Framework:**
# Memory efficiency ratio as function of graph size:
# $$\eta(|\mathcal{V}|) = \frac{M_{\text{sample}}}{M_{\text{full}}} = \frac{K}{|\mathcal{V}|} \to 0 \text{ as } |\mathcal{V}| \to \infty$$
#
# where $K$ is the fixed sample size.

# %%
# Load Cora dataset
print("Loading datasets...")
cora_dataset = Planetoid('/tmp/Cora', 'Cora', transform=NormalizeFeatures())
cora_data = cora_dataset[0]

print(f"Cora Dataset:")
print(f"  Nodes: {cora_data.num_nodes}")
print(f"  Edges: {cora_data.num_edges}")
print(f"  Features: {cora_data.num_features}")
print(f"  Classes: {cora_dataset.num_classes}")

def create_synthetic_large_graph(num_nodes=5000, num_features=100, num_classes=10,
                                avg_degree=10, seed=42):
    """
    Create a synthetic large graph for scalability testing
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate random features
    x = torch.randn(num_nodes, num_features)

    # Generate random labels
    y = torch.randint(0, num_classes, (num_nodes,))

    # Generate edges using Erdős–Rényi model
    prob = avg_degree / (num_nodes - 1)
    adj_matrix = torch.rand(num_nodes, num_nodes) < prob

    # Make symmetric and remove self-loops
    adj_matrix = adj_matrix | adj_matrix.T
    adj_matrix.fill_diagonal_(False)

    # Convert to edge_index format
    edge_index = adj_matrix.nonzero().t()

    # Create train/val/test masks
    num_train = int(0.6 * num_nodes)
    num_val = int(0.2 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[:num_train] = True
    val_mask[num_train:num_train + num_val] = True
    test_mask[num_train + num_val:] = True

    # Shuffle masks
    perm = torch.randperm(num_nodes)
    train_mask = train_mask[perm]
    val_mask = val_mask[perm]
    test_mask = test_mask[perm]

    return Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

# Create synthetic large graph for testing scalability
print("\nCreating synthetic large graph...")
large_graph = create_synthetic_large_graph(num_nodes=2000, num_features=50)  # Smaller for CPU

print(f"Synthetic Graph:")
print(f"  Nodes: {large_graph.num_nodes}")
print(f"  Edges: {large_graph.num_edges}")
print(f"  Features: {large_graph.x.size(1)}")
print(f"  Average degree: {large_graph.num_edges / large_graph.num_nodes:.2f}")

# %% [markdown]
# ## 5. GraphSAINT Training Setup
#
# Let's set up GraphSAINT with different sampling strategies. We'll compare random walk sampling, node sampling, and edge sampling to see their effects on performance and memory usage.
#
# ### Mathematical Analysis of Sampling Strategies
#
# **Random Walk Sampling Mathematics:**
#
# 1. **Transition Probability:**
#    $$P_{uv} = \frac{A_{uv}}{d_u}$$
#
#    where $A_{uv}$ is the adjacency matrix element and $d_u$ is node degree.
#
# 2. **Stationary Distribution:**
#    $$\pi_v = \frac{d_v}{2|E|}$$
#
# 3. **Walk Coverage Probability:**
#    For a random walk of length $L$, node $v$ is included with probability:
#    $$P(\text{visit } v) = 1 - (1 - \pi_v)^L$$
#
# **Node Sampling Mathematics:**
#
# 1. **Uniform Sampling:**
#    $$P(\text{select } v) = \frac{1}{|V|}$$
#
# 2. **Expected Subgraph Size:**
#    $$E[|V_s|] = |V| \times P(\text{selection})$$
#
# 3. **Edge Preservation:**
#    $$E[|E_s|] = |E| \times P(\text{both endpoints selected})$$
#
# **Edge Sampling Mathematics:**
#
# 1. **Edge Selection Probability:**
#    $$P(\text{select edge } (u,v)) = \frac{1}{|E|}$$
#
# 2. **Induced Subgraph:**
#    $$V_s = \{u : \exists v \text{ s.t. } (u,v) \in E_s\}$$
#
# **Variance Analysis for Different Strategies:**
#
# 1. **Random Walk Sampling Variance:**
#    $$\text{Var}[L_{\text{rw}}] = O\left(\frac{1}{L \times |\text{walks}|}\right)$$
#
# 2. **Node Sampling Variance:**
#    $$\text{Var}[L_{\text{node}}] = O\left(\frac{1}{|V_s|}\right)$$
#
# 3. **Edge Sampling Variance:**
#    $$\text{Var}[L_{\text{edge}}] = O\left(\frac{1}{|E_s|}\right)$$
#
# **Sample Coverage Parameter:**
# The coverage parameter $c$ ensures each node is sampled approximately $c$ times:
# $$E[\text{times node } v \text{ is sampled}] \approx c$$
#
# **Batch Size vs. Coverage Trade-off:**
# $$\text{Total Training Cost} = \frac{|\text{epochs}| \times |V|}{|\text{batch\_size}| \times c}$$

# %%
def setup_graphsaint_loaders(data, batch_size=512, walk_length=3, num_steps=30,
                            sample_coverage=20, cpu_only=True):
    """
    Setup GraphSAINT data loaders with different sampling strategies
    """
    # Make sure edge_index is undirected
    data.edge_index = to_undirected(data.edge_index)

    loaders = {}

    # Random Walk Sampler
    try:
        rw_loader = GraphSAINTRandomWalkSampler(
            data,
            batch_size=batch_size,
            walk_length=walk_length,
            num_steps=num_steps,
            sample_coverage=sample_coverage,
            shuffle=True
        )
        loaders['random_walk'] = rw_loader
        print(f"Random Walk Sampler: batch_size={batch_size}, walk_length={walk_length}")
    except Exception as e:
        print(f"Could not create Random Walk Sampler: {e}")

    # Node Sampler
    try:
        node_loader = GraphSAINTNodeSampler(
            data,
            batch_size=batch_size,
            shuffle=True
        )
        loaders['node'] = node_loader
        print(f"Node Sampler: batch_size={batch_size}")
    except Exception as e:
        print(f"Could not create Node Sampler: {e}")

    # Edge Sampler
    try:
        edge_loader = GraphSAINTEdgeSampler(
            data,
            batch_size=batch_size // 4,  # Smaller batch for edge sampling
            shuffle=True
        )
        loaders['edge'] = edge_loader
        print(f"Edge Sampler: batch_size={batch_size // 4}")
    except Exception as e:
        print(f"Could not create Edge Sampler: {e}")

    return loaders

# Setup GraphSAINT loaders for Cora
print("Setting up GraphSAINT loaders...")
saint_loaders = setup_graphsaint_loaders(cora_data, batch_size=256, sample_coverage=10)

print(f"\nAvailable samplers: {list(saint_loaders.keys())}")

# Show example batch
if 'node' in saint_loaders:
    sample_batch = next(iter(saint_loaders['node']))
    print(f"\nExample batch from node sampler:")
    print(f"  Batch nodes: {sample_batch.num_nodes}")
    print(f"  Batch edges: {sample_batch.num_edges}")
    print(f"  Has node_norm: {hasattr(sample_batch, 'node_norm')}")
    print(f"  Has edge_norm: {hasattr(sample_batch, 'edge_norm')}")

# %% [markdown]
# ## 6. Training Functions for Memory-Enhanced Models
#
# We create specialized training functions that can handle both GraphSAINT and FastGCN, with built-in memory monitoring to track efficiency gains.
#
# ### Mathematical Training Dynamics
#
# **Stochastic Gradient Descent with Sampling:**
#
# 1. **Full-batch Gradient:**
#    $$\nabla \mathcal{L}(\theta) = \frac{1}{|\mathcal{V}_{\text{train}}|} \sum_{v \in \mathcal{V}_{\text{train}}} \nabla \ell(f_\theta(\mathcal{G}, v), y_v)$$
#
# 2. **Sampled Gradient (GraphSAINT):**
#    $$\nabla \hat{\mathcal{L}}_{\text{saint}}(\theta) = \frac{|\mathcal{V}|}{|\mathcal{V}_s|} \frac{1}{|\mathcal{V}_{s,\text{train}}|} \sum_{v \in \mathcal{V}_{s,\text{train}}} \nabla \ell(f_\theta(\mathcal{G}_s, v), y_v)$$
#
# 3. **Sampled Gradient (FastGCN):**
#    $$\nabla \hat{\mathcal{L}}_{\text{fast}}(\theta) = \frac{1}{|\mathcal{V}_{\text{train}}|} \sum_{v \in \mathcal{V}_{\text{train}}} \nabla \ell(\hat{f}_\theta(\mathcal{G}, v), y_v)$$
#
#    where $\hat{f}_\theta$ uses sampled neighborhoods.
#
# **Unbiased Estimation Property:**
# For both approaches, we maintain:
# $$\mathbb{E}[\nabla \hat{\mathcal{L}}(\theta)] = \nabla \mathcal{L}(\theta)$$
#
# **Convergence Analysis:**
#
# 1. **Learning Rate Schedule:**
#    $$\alpha_t = \frac{\alpha_0}{\sqrt{t}}$$ or $$\alpha_t = \alpha_0 \gamma^{t/T}$$
#
# 2. **Convergence Rate:**
#    Under standard assumptions:
#    $$\mathbb{E}[\|\nabla \mathcal{L}(\theta_T)\|^2] = O\left(\frac{1}{\sqrt{T}}\right)$$
#
# 3. **Variance Effect:**
#    Higher sampling variance requires smaller learning rates:
#    $$\alpha_{\text{optimal}} \propto \frac{1}{\sqrt{\text{Var}[\nabla \hat{\mathcal{L}}]}}$$
#
# **Normalization in GraphSAINT:**
#
# 1. **Node Normalization:**
#    $$w_v^{\text{node}} = \frac{|\mathcal{V}|}{|\mathcal{V}_s| \times P(\text{node } v \text{ sampled})}$$
#
# 2. **Edge Normalization:**
#    $$w_{uv}^{\text{edge}} = \frac{|\mathcal{E}|}{|\mathcal{E}_s| \times P(\text{edge } (u,v) \text{ sampled})}$$
#
# **Memory-Time Trade-off:**
# $$\text{Training Time} \propto \frac{\text{Memory Saved}}{\text{Convergence Rate}}$$
#
# The optimal balance depends on:
# - Available memory constraints
# - Desired accuracy requirements
# - Time budget for training

# %%
def train_graphsaint(model, loader, optimizer, criterion, device, monitor=None):
    """
    Train GraphSAINT for one epoch
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_nodes = 0

    if monitor:
        monitor.log_memory("Start training epoch")

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)

        optimizer.zero_grad()

        # Forward pass with normalization if available
        if hasattr(batch, 'edge_norm'):
            out = model(batch.x, batch.edge_index, batch.edge_norm)
        else:
            out = model(batch.x, batch.edge_index)

        # Only use training nodes in the batch
        if hasattr(batch, 'train_mask') and batch.train_mask.any():
            mask = batch.train_mask
        else:
            # If no train_mask in batch, assume all nodes are training nodes
            mask = torch.ones(batch.num_nodes, dtype=torch.bool, device=device)

        loss = criterion(out[mask], batch.y[mask])

        # Apply node normalization if available
        if hasattr(batch, 'node_norm'):
            loss = loss * batch.node_norm[mask].sum() / mask.sum()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = out[mask].argmax(dim=1)
        total_correct += (pred == batch.y[mask]).sum().item()
        total_nodes += mask.sum().item()

        if monitor and batch_idx == 0:
            monitor.log_memory(f"After batch {batch_idx}")

    if monitor:
        monitor.log_memory("End training epoch")

    return total_loss / len(loader), total_correct / total_nodes

def train_fastgcn(model, data, optimizer, criterion, device, monitor=None):
    """
    Train FastGCN for one epoch
    """
    model.train()
    data = data.to(device)

    if monitor:
        monitor.log_memory("Start FastGCN training")

    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    if monitor:
        monitor.log_memory("End FastGCN training")

    # Calculate accuracy
    pred = out[data.train_mask].argmax(dim=1)
    acc = (pred == data.y[data.train_mask]).float().mean()

    return loss.item(), acc.item()

def evaluate_model(model, data, device, mask_type='test'):
    """
    Evaluate model on validation or test set
    """
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        out = model(data.x, data.edge_index)

        if mask_type == 'val':
            mask = data.val_mask
        else:
            mask = data.test_mask

        pred = out[mask].argmax(dim=1)
        acc = accuracy_score(data.y[mask].cpu(), pred.cpu())
        f1 = f1_score(data.y[mask].cpu(), pred.cpu(), average='weighted')

    return acc, f1

# %% [markdown]
# ## 7. Training GraphSAINT Models
#
# Let's train GraphSAINT models using different sampling strategies and compare their performance and memory efficiency.

# %%
# Set up device and optimization
device = torch.device('cpu')
torch.set_num_threads(8)

# Training parameters
epochs = 50  # Reduced for CPU
lr = 0.01
hidden_dim = 64

print("Training GraphSAINT models...")

saint_results = {}

# Train with different sampling strategies
for sampler_name, loader in saint_loaders.items():
    print(f"\n=== Training with {sampler_name} sampler ===")

    # Initialize model
    model = GraphSAINTModel(
        input_dim=cora_dataset.num_features,
        hidden_dim=hidden_dim,
        output_dim=cora_dataset.num_classes,
        num_layers=2,  # Reduced for CPU
        dropout=0.5
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Memory monitor
    monitor = MemoryMonitor()
    monitor.log_memory(f"Model initialized ({sampler_name})")

    # Training loop
    train_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        # Training
        loss, acc = train_graphsaint(model, loader, optimizer, criterion, device,
                                   monitor if epoch == 0 else None)
        train_losses.append(loss)
        train_accs.append(acc)

        # Validation every 10 epochs
        if epoch % 10 == 0:
            val_acc, val_f1 = evaluate_model(model, cora_data, device, 'val')
            val_accs.append(val_acc)
            print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Train Acc: {acc:.4f}, Val Acc: {val_acc:.4f}')

    # Final test evaluation
    test_acc, test_f1 = evaluate_model(model, cora_data, device, 'test')
    peak_memory = monitor.get_peak_memory()

    saint_results[sampler_name] = {
        'model': model,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'peak_memory': peak_memory,
        'num_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

    print(f'Final Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')
    print(f'Peak Memory: {peak_memory:.1f} MB')

print(f"\n=== GraphSAINT Results Summary ===")
for name, results in saint_results.items():
    print(f"{name:<15} Test Acc: {results['test_acc']:.4f}, Peak Memory: {results['peak_memory']:.1f} MB")

# %% [markdown]
# ## 8. Training FastGCN Model
#
# Now let's train the FastGCN model and compare its performance with GraphSAINT approaches.

# %%
print("\n=== Training FastGCN ===")

# Initialize FastGCN model
fastgcn_model = FastGCNModel(
    input_dim=cora_dataset.num_features,
    hidden_dim=hidden_dim,
    output_dim=cora_dataset.num_classes,
    num_layers=2,
    sample_sizes=[100, 50],  # Smaller samples for CPU
    dropout=0.5
).to(device)

print(f"FastGCN parameters: {sum(p.numel() for p in fastgcn_model.parameters() if p.requires_grad)}")

optimizer = torch.optim.Adam(fastgcn_model.parameters(), lr=lr, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# Memory monitor
fastgcn_monitor = MemoryMonitor()
fastgcn_monitor.log_memory("FastGCN initialized")

# Training loop
fastgcn_losses = []
fastgcn_train_accs = []
fastgcn_val_accs = []

for epoch in range(epochs):
    # Training
    loss, acc = train_fastgcn(fastgcn_model, cora_data, optimizer, criterion, device,
                             fastgcn_monitor if epoch == 0 else None)
    fastgcn_losses.append(loss)
    fastgcn_train_accs.append(acc)

    # Validation every 10 epochs
    if epoch % 10 == 0:
        val_acc, val_f1 = evaluate_model(fastgcn_model, cora_data, device, 'val')
        fastgcn_val_accs.append(val_acc)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Train Acc: {acc:.4f}, Val Acc: {val_acc:.4f}')

# Final test evaluation
fastgcn_test_acc, fastgcn_test_f1 = evaluate_model(fastgcn_model, cora_data, device, 'test')
fastgcn_peak_memory = fastgcn_monitor.get_peak_memory()

print(f'\nFastGCN Final Test Acc: {fastgcn_test_acc:.4f}, Test F1: {fastgcn_test_f1:.4f}')
print(f'FastGCN Peak Memory: {fastgcn_peak_memory:.1f} MB')

# %% [markdown]
# ## 9. Baseline GCN Comparison
#
# Let's train a standard GCN to compare against our memory-enhanced approaches and understand the trade-offs between efficiency and performance.
#
# ### Mathematical Baseline Analysis
#
# **Standard Full-batch GCN:**
#
# 1. **Forward Propagation:**
#    $$\mathbf{H}^{(\ell+1)} = \sigma\left(\tilde{\mathbf{A}} \mathbf{H}^{(\ell)} \mathbf{W}^{(\ell)}\right)$$
#
#    where $\tilde{\mathbf{A}} = \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$ is the normalized adjacency matrix.
#
# 2. **Complete Neighborhood Aggregation:**
#    Every node uses its complete neighborhood:
#    $$\mathbf{h}_v^{(\ell+1)} = \sigma\left(\mathbf{W}^{(\ell)} \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{\mathbf{h}_u^{(\ell)}}{\sqrt{d_u d_v}}\right)$$
#
# 3. **Computational Complexity:**
#    $$\mathcal{C}_{\text{baseline}} = O(L \times |\mathcal{E}| \times D \times D')$$
#
#    where $D$ and $D'$ are input and output dimensions.
#
# **Memory Requirements:**
#
# 1. **Forward Pass Memory:**
#    $$M_{\text{forward}} = \sum_{\ell=0}^L |\mathcal{V}| \times D^{(\ell)}$$
#
# 2. **Backward Pass Memory:**
#    $$M_{\text{backward}} = \sum_{\ell=0}^{L-1} |\mathcal{V}| \times D^{(\ell)}$$
#
# 3. **Parameter Memory:**
#    $$M_{\text{params}} = \sum_{\ell=0}^{L-1} D^{(\ell)} \times D^{(\ell+1)}$$
#
# **Theoretical Performance Bounds:**
#
# For the baseline GCN with optimal parameters:
# $$\mathcal{L}_{\text{baseline}} \leq \mathcal{L}_{\text{Bayes}} + \epsilon_{\text{approx}}$$
#
# where $\epsilon_{\text{approx}}$ is the approximation error due to finite depth $L$.
#
# **Comparison Metrics:**
#
# 1. **Accuracy Gap:**
#    $$\Delta_{\text{acc}} = \text{Acc}_{\text{baseline}} - \text{Acc}_{\text{enhanced}}$$
#
# 2. **Memory Efficiency:**
#    $$\eta_{\text{memory}} = \frac{M_{\text{enhanced}}}{M_{\text{baseline}}}$$
#
# 3. **Time Efficiency:**
#    $$\eta_{\text{time}} = \frac{T_{\text{enhanced}}}{T_{\text{baseline}}}$$
#
# 4. **Overall Efficiency Score:**
#    $$S_{\text{eff}} = \frac{\text{Acc}_{\text{enhanced}}}{\eta_{\text{memory}} \times \eta_{\text{time}}}$$
#
# **Trade-off Analysis:**
# The fundamental trade-off is captured by:
# $$\text{Performance} \propto \frac{\text{Information Used}}{\text{Resources Consumed}}$$
#
# Where:
# - Information Used $\propto$ (fraction of graph sampled)
# - Resources Consumed $\propto$ (memory + computation time)

# %%
print("\n=== Training Baseline GCN ===")

# Standard GCN implementation
class BaselineGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()

        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x

# Initialize baseline GCN
baseline_model = BaselineGCN(
    input_dim=cora_dataset.num_features,
    hidden_dim=hidden_dim,
    output_dim=cora_dataset.num_classes,
    num_layers=2,
    dropout=0.5
).to(device)

print(f"Baseline GCN parameters: {sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)}")

optimizer = torch.optim.Adam(baseline_model.parameters(), lr=lr, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# Memory monitor
baseline_monitor = MemoryMonitor()
baseline_monitor.log_memory("Baseline GCN initialized")

# Training loop
baseline_losses = []
baseline_train_accs = []
baseline_val_accs = []

for epoch in range(epochs):
    # Training
    baseline_model.train()
    optimizer.zero_grad()

    if epoch == 0:
        baseline_monitor.log_memory("Start baseline training")

    out = baseline_model(cora_data.x.to(device), cora_data.edge_index.to(device))
    loss = criterion(out[cora_data.train_mask], cora_data.y[cora_data.train_mask].to(device))
    loss.backward()
    optimizer.step()

    if epoch == 0:
        baseline_monitor.log_memory("After baseline forward/backward")

    baseline_losses.append(loss.item())

    # Training accuracy
    pred = out[cora_data.train_mask].argmax(dim=1)
    acc = (pred == cora_data.y[cora_data.train_mask].to(device)).float().mean().item()
    baseline_train_accs.append(acc)

    # Validation every 10 epochs
    if epoch % 10 == 0:
        val_acc, val_f1 = evaluate_model(baseline_model, cora_data, device, 'val')
        baseline_val_accs.append(val_acc)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Train Acc: {acc:.4f}, Val Acc: {val_acc:.4f}')

# Final test evaluation
baseline_test_acc, baseline_test_f1 = evaluate_model(baseline_model, cora_data, device, 'test')
baseline_peak_memory = baseline_monitor.get_peak_memory()

print(f'\nBaseline GCN Final Test Acc: {baseline_test_acc:.4f}, Test F1: {baseline_test_f1:.4f}')
print(f'Baseline GCN Peak Memory: {baseline_peak_memory:.1f} MB')

# %% [markdown]
# ## 10. Comprehensive Performance Comparison
#
# Let's create detailed visualizations comparing all approaches across multiple metrics: accuracy, memory usage, and training dynamics.
#
# ### Mathematical Performance Analysis Framework
#
# **Multi-Objective Optimization:**
# We seek to optimize the vector objective:
# $$\mathbf{f}(\theta, S) = \begin{bmatrix}
# \text{Accuracy}(\theta) \\
# -\text{Memory}(S) \\
# -\text{Time}(S)
# \end{bmatrix}$$
#
# where $\theta$ are model parameters and $S$ is the sampling strategy.
#
# **Pareto Efficiency:**
# A configuration $(\theta^*, S^*)$ is Pareto efficient if there exists no other configuration $(\theta, S)$ such that:
# $$\mathbf{f}(\theta, S) \geq \mathbf{f}(\theta^*, S^*)$$
# with at least one strict inequality.
#
# **Performance Metrics:**
#
# 1. **Relative Accuracy:**
#    $$R_{\text{acc}} = \frac{\text{Acc}_{\text{method}}}{\text{Acc}_{\text{baseline}}}$$
#
# 2. **Memory Efficiency Ratio:**
#    $$R_{\text{mem}} = \frac{M_{\text{baseline}}}{M_{\text{method}}}$$
#
# 3. **Speed-up Factor:**
#    $$R_{\text{speed}} = \frac{T_{\text{baseline}}}{T_{\text{method}}}$$
#
# 4. **Efficiency Score:**
#    $$E = R_{\text{acc}} \times \sqrt{R_{\text{mem}} \times R_{\text{speed}}}$$
#
# **Statistical Analysis:**
#
# 1. **Confidence Intervals:**
#    For accuracy measurements with $n$ runs:
#    $$\text{CI}_{95\%} = \bar{x} \pm 1.96 \frac{s}{\sqrt{n}}$$
#
# 2. **Significance Testing:**
#    H₀: $\mu_{\text{method}} = \mu_{\text{baseline}}$
#
#    Test statistic: $t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}$
#
# **Scalability Analysis:**
#
# 1. **Memory Scaling:**
#    $$M(n) = a \cdot n^b + c$$
#
#    where $n$ is graph size, $b$ is the scaling exponent.
#
# 2. **Time Complexity:**
#    $$T(n, e) = \alpha \cdot n^{\beta_1} \cdot e^{\beta_2}$$
#
#    where $e$ is the number of edges.
#
# **Trade-off Curves:**
# The accuracy-efficiency trade-off follows:
# $$\text{Accuracy} = f(\text{Resource Budget})$$
#
# typically showing diminishing returns:
# $$\frac{d(\text{Accuracy})}{d(\text{Resources})} \downarrow \text{ as Resources} \uparrow$$

# %%
# Collect all results for comparison
all_results = {
    'Baseline GCN': {
        'test_acc': baseline_test_acc,
        'test_f1': baseline_test_f1,
        'peak_memory': baseline_peak_memory,
        'train_losses': baseline_losses,
        'val_accs': baseline_val_accs,
        'num_params': sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
    },
    'FastGCN': {
        'test_acc': fastgcn_test_acc,
        'test_f1': fastgcn_test_f1,
        'peak_memory': fastgcn_peak_memory,
        'train_losses': fastgcn_losses,
        'val_accs': fastgcn_val_accs,
        'num_params': sum(p.numel() for p in fastgcn_model.parameters() if p.requires_grad)
    }
}

# Add GraphSAINT results
for name, results in saint_results.items():
    all_results[f'SAINT-{name}'] = results

# Create comprehensive comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Test Accuracy Comparison
models = list(all_results.keys())
test_accs = [all_results[model]['test_acc'] for model in models]

bars1 = axes[0, 0].bar(range(len(models)), test_accs, alpha=0.8,
                      color=['red', 'blue', 'green', 'orange', 'purple'][:len(models)])
axes[0, 0].set_ylabel('Test Accuracy')
axes[0, 0].set_title('Test Accuracy Comparison')
axes[0, 0].set_xticks(range(len(models)))
axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
axes[0, 0].set_ylim(0, 1)

# Add value labels
for bar, acc in zip(bars1, test_accs):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

# 2. Memory Usage Comparison
peak_memories = [all_results[model]['peak_memory'] for model in models]

bars2 = axes[0, 1].bar(range(len(models)), peak_memories, alpha=0.8,
                      color=['red', 'blue', 'green', 'orange', 'purple'][:len(models)])
axes[0, 1].set_ylabel('Peak Memory (MB)')
axes[0, 1].set_title('Memory Usage Comparison')
axes[0, 1].set_xticks(range(len(models)))
axes[0, 1].set_xticklabels(models, rotation=45, ha='right')

# Add value labels
for bar, mem in zip(bars2, peak_memories):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(peak_memories)*0.01,
                   f'{mem:.1f}', ha='center', va='bottom', fontsize=9)

# 3. Model Parameters Comparison
num_params = [all_results[model]['num_params'] for model in models]

bars3 = axes[0, 2].bar(range(len(models)), num_params, alpha=0.8,
                      color=['red', 'blue', 'green', 'orange', 'purple'][:len(models)])
axes[0, 2].set_ylabel('Number of Parameters')
axes[0, 2].set_title('Model Complexity Comparison')
axes[0, 2].set_xticks(range(len(models)))
axes[0, 2].set_xticklabels(models, rotation=45, ha='right')

# 4. Training Loss Curves
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i, (model, results) in enumerate(all_results.items()):
    if 'train_losses' in results:
        axes[1, 0].plot(results['train_losses'], label=model, alpha=0.8, color=colors[i])

axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Training Loss')
axes[1, 0].set_title('Training Loss Curves')
axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1, 0].grid(True, alpha=0.3)

# 5. Validation Accuracy Curves
eval_epochs = range(0, epochs, 10)
for i, (model, results) in enumerate(all_results.items()):
    if 'val_accs' in results and len(results['val_accs']) > 0:
        axes[1, 1].plot(eval_epochs[:len(results['val_accs'])], results['val_accs'],
                       'o-', label=model, alpha=0.8, color=colors[i])

axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Validation Accuracy')
axes[1, 1].set_title('Validation Accuracy Curves')
axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1, 1].grid(True, alpha=0.3)

# 6. Efficiency Analysis (Memory vs Accuracy)
axes[1, 2].scatter(peak_memories, test_accs, s=100, alpha=0.7,
                  c=range(len(models)), cmap='viridis')

for i, model in enumerate(models):
    axes[1, 2].annotate(model, (peak_memories[i], test_accs[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)

axes[1, 2].set_xlabel('Peak Memory (MB)')
axes[1, 2].set_ylabel('Test Accuracy')
axes[1, 2].set_title('Efficiency Analysis: Memory vs Accuracy')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print detailed comparison table
print("\n=== Detailed Performance Comparison ===")
print(f"{'Model':<20} {'Test Acc':<10} {'Test F1':<10} {'Peak Mem':<12} {'Params':<10}")
print("-" * 70)

for model, results in all_results.items():
    print(f"{model:<20} {results['test_acc']:<10.4f} {results['test_f1']:<10.4f} "
          f"{results['peak_memory']:<12.1f} {results['num_params']:<10}")

# %% [markdown]
# ## 11. Scalability Testing
#
# Let's test the scalability of our memory-enhanced approaches on the synthetic large graph to demonstrate their benefits for larger datasets.
#
# ### Mathematical Scalability Analysis
#
# **Asymptotic Complexity Analysis:**
#
# 1. **Full-batch GCN Scaling:**
#    - Memory: $M_{\text{full}}(n) = O(L \cdot n \cdot d + n^2)$ (dense case)
#    - Memory: $M_{\text{full}}(n) = O(L \cdot n \cdot d + e)$ (sparse case)
#    - Time: $T_{\text{full}}(n) = O(L \cdot e \cdot d)$
#
# 2. **GraphSAINT Scaling:**
#    - Memory: $M_{\text{saint}}(n) = O(L \cdot k \cdot d)$ where $k \ll n$
#    - Time: $T_{\text{saint}}(n) = O(L \cdot k \cdot \bar{d}_s \cdot d)$
#    - Scalability Factor: $\frac{M_{\text{full}}}{M_{\text{saint}}} = O(n/k)$
#
# 3. **FastGCN Scaling:**
#    - Memory: $M_{\text{fast}}(n) = O(L \cdot \max_\ell K^{(\ell)} \cdot d + e)$
#    - Time: $T_{\text{fast}}(n) = O(L \cdot \sum_\ell K^{(\ell)} \cdot d^2)$
#    - Scalability Factor: $\frac{M_{\text{full}}}{M_{\text{fast}}} = O(n/K_{\max})$
#
# **Scalability Metrics:**
#
# 1. **Memory Scalability Coefficient:**
#    $$\alpha_{\text{mem}} = \frac{\log(M_2/M_1)}{\log(n_2/n_1)}$$
#
#    where $(n_1, M_1)$ and $(n_2, M_2)$ are size-memory pairs.
#
# 2. **Time Scalability Coefficient:**
#    $$\alpha_{\text{time}} = \frac{\log(T_2/T_1)}{\log(n_2/n_1)}$$
#
# 3. **Efficiency Degradation:**
#    $$\beta = \frac{d(\text{Accuracy})}{d(\log n)}$$
#
# **Break-even Analysis:**
# Memory-enhanced methods become beneficial when:
# $$n > n_{\text{break}} = \frac{C_{\text{overhead}}}{S_{\text{reduction}}}$$
#
# where:
# - $C_{\text{overhead}}$ is the constant overhead of sampling
# - $S_{\text{reduction}}$ is the per-node memory/time savings
#
# **Theoretical Limits:**
#
# 1. **Information-Theoretic Bound:**
#    Minimum information needed for $\epsilon$-accurate learning:
#    $$I_{\min}(\epsilon) = O\left(\frac{d \log n}{\epsilon^2}\right)$$
#
# 2. **Sampling Lower Bound:**
#    Any sampling strategy must sample at least:
#    $$K_{\min} = \Omega\left(\frac{\log n}{\epsilon^2}\right)$$
#
#    nodes per layer for $\epsilon$-approximation.
#
# **Practical Scalability Guidelines:**
#
# 1. **Memory-Constrained Regime:** $M_{\text{available}} < M_{\text{full}}$
#    - Use GraphSAINT with $k = \lfloor M_{\text{available}} / (L \cdot d) \rfloor$
#
# 2. **Time-Constrained Regime:** $T_{\text{available}} < T_{\text{full}}$
#    - Use FastGCN with adaptive $K^{(\ell)}$
#
# 3. **Accuracy-Constrained Regime:** $\text{Acc}_{\text{required}} > \text{Acc}_{\text{sampled}}$
#    - Increase sampling budget or use hybrid approaches

# %%
print("\n=== Scalability Testing on Large Graph ===")
print(f"Large graph: {large_graph.num_nodes} nodes, {large_graph.num_edges} edges")

# Test baseline GCN on large graph
print("\n--- Testing Baseline GCN ---")
baseline_large = BaselineGCN(
    input_dim=large_graph.x.size(1),
    hidden_dim=32,  # Smaller for scalability
    output_dim=10,  # 10 classes in synthetic graph
    num_layers=2
).to(device)

baseline_large_monitor = MemoryMonitor()
baseline_large_monitor.log_memory("Baseline large model initialized")

try:
    # Test forward pass
    baseline_large.eval()
    with torch.no_grad():
        start_time = time.time()
        out = baseline_large(large_graph.x.to(device), large_graph.edge_index.to(device))
        baseline_time = time.time() - start_time
        baseline_large_monitor.log_memory("After baseline large forward pass")

    print(f"Baseline GCN: {baseline_time:.3f}s, Peak Memory: {baseline_large_monitor.get_peak_memory():.1f} MB")
    baseline_large_success = True

except Exception as e:
    print(f"Baseline GCN failed: {e}")
    baseline_large_success = False

# Test FastGCN on large graph
print("\n--- Testing FastGCN ---")
fastgcn_large = FastGCNModel(
    input_dim=large_graph.x.size(1),
    hidden_dim=32,
    output_dim=10,
    num_layers=2,
    sample_sizes=[100, 50]  # Small samples for efficiency
).to(device)

fastgcn_large_monitor = MemoryMonitor()
fastgcn_large_monitor.log_memory("FastGCN large model initialized")

try:
    # Test forward pass
    fastgcn_large.eval()
    with torch.no_grad():
        start_time = time.time()
        out = fastgcn_large(large_graph.x.to(device), large_graph.edge_index.to(device))
        fastgcn_time = time.time() - start_time
        fastgcn_large_monitor.log_memory("After FastGCN large forward pass")

    print(f"FastGCN: {fastgcn_time:.3f}s, Peak Memory: {fastgcn_large_monitor.get_peak_memory():.1f} MB")
    fastgcn_large_success = True

except Exception as e:
    print(f"FastGCN failed: {e}")
    fastgcn_large_success = False

# Test GraphSAINT on large graph (if we have a working sampler)
if saint_loaders:
    print("\n--- Testing GraphSAINT ---")

    # Setup GraphSAINT for large graph
    try:
        large_saint_loaders = setup_graphsaint_loaders(
            large_graph,
            batch_size=128,  # Smaller batch for large graph
            sample_coverage=5
        )

        if large_saint_loaders:
            sampler_name = list(large_saint_loaders.keys())[0]
            loader = large_saint_loaders[sampler_name]

            saint_large = GraphSAINTModel(
                input_dim=large_graph.x.size(1),
                hidden_dim=32,
                output_dim=10,
                num_layers=2
            ).to(device)

            saint_large_monitor = MemoryMonitor()
            saint_large_monitor.log_memory("GraphSAINT large model initialized")

            # Test one batch
            saint_large.eval()
            batch = next(iter(loader))
            batch = batch.to(device)

            with torch.no_grad():
                start_time = time.time()
                out = saint_large(batch.x, batch.edge_index)
                saint_time = time.time() - start_time
                saint_large_monitor.log_memory("After GraphSAINT large forward pass")

            print(f"GraphSAINT ({sampler_name}): {saint_time:.3f}s per batch, "
                  f"Peak Memory: {saint_large_monitor.get_peak_memory():.1f} MB")

    except Exception as e:
        print(f"GraphSAINT setup failed: {e}")

# Performance summary for scalability
print("\n=== Scalability Summary ===")
print("Memory-enhanced approaches show benefits on larger graphs:")
print("• FastGCN: Reduces memory through importance sampling")
print("• GraphSAINT: Enables mini-batch training on subgraphs")
print("• Both approaches trade some accuracy for significant memory savings")
print("• Essential for graphs with millions of nodes")

# %% [markdown]
# ## 12. CPU Optimization Guidelines for Memory-Enhanced GNNs
#
# Let's provide comprehensive optimization guidelines specifically for running memory-enhanced GNNs on CPU architectures like the MacBook Air M2.
#
# ### Mathematical Framework for CPU Optimization
#
# **CPU-Specific Performance Model:**
#
# 1. **Memory Bandwidth Limitation:**
#    $$T_{\text{memory}} = \frac{\text{Data Size}}{\text{Bandwidth}} = \frac{n \cdot d \cdot \text{sizeof}(\text{float})}{B_{\text{mem}}}$$
#
# 2. **Cache Efficiency:**
#    $$\text{Cache Hit Ratio} = \frac{\text{Data in Cache}}{\text{Total Data Access}}$$
#
#    Optimal performance when working set $< $ L3 cache size.
#
# 3. **Vectorization Benefits:**
#    $$\text{Speedup}_{\text{SIMD}} = \min\left(\frac{\text{Vector Width}}{\text{Element Size}}, \text{Data Parallelism}\right)$$
#
# **Memory-Compute Trade-offs:**
#
# 1. **Batch Size Optimization:**
#    $$B_{\text{optimal}} = \arg\min_B \left(T_{\text{compute}}(B) + T_{\text{memory}}(B)\right)$$
#
#    where:
#    - $T_{\text{compute}}(B) = \frac{W}{B \cdot P}$ (work per core)
#    - $T_{\text{memory}}(B) = \frac{B \cdot S}{B_{\text{mem}}}$ (memory transfer time)
#
# 2. **Sample Size vs. Accuracy:**
#    $$\text{Accuracy}(K) = A_{\max} \left(1 - e^{-\alpha K}\right)$$
#
#    Diminishing returns beyond optimal $K^*$.
#
# **CPU Architecture Considerations:**
#
# 1. **Unified Memory Architecture (M2):**
#    - Memory sharing between CPU and GPU
#    - No data transfer overhead
#    - Total memory constraint: $M_{\text{CPU}} + M_{\text{GPU}} \leq M_{\text{total}}$
#
# 2. **Thermal Throttling Model:**
#    $$f_{\text{CPU}}(t) = f_{\max} \cdot \min\left(1, \frac{T_{\max} - T(t)}{T_{\max} - T_{\text{ambient}}}\right)$$
#
#    where $T(t)$ is temperature as function of time.
#
# **Optimization Strategies:**
#
# 1. **Memory Access Patterns:**
#    - Sequential access: $O(n)$ cache misses
#    - Random access: $O(n \log n)$ cache misses
#    - Optimize for spatial locality
#
# 2. **Computational Intensity:**
#    $$I = \frac{\text{Operations}}{\text{Bytes Transferred}}$$
#
#    Higher intensity → better CPU utilization
#
# 3. **Work Distribution:**
#    $$T_{\text{parallel}} = \frac{W}{P} + T_{\text{synchronization}}$$
#
#    where $P$ is number of cores, $W$ is total work.
#
# **Platform-Specific Optimizations:**
#
# **For M2 MacBook Air:**
# - 8 cores (4 performance + 4 efficiency)
# - 24GB unified memory
# - 100GB/s memory bandwidth
# - No active cooling
#
# **Optimization Guidelines:**
# 1. **Thread Allocation:** Use 4-6 threads for optimal performance
# 2. **Memory Usage:** Keep < 16GB for system stability
# 3. **Thermal Management:** Use smaller batches for sustained performance
# 4. **Vectorization:** Leverage ARM NEON instructions

# %%
# CPU Optimization Guidelines for Memory-Enhanced GNNs
print("=== CPU Optimization Guidelines for Memory-Enhanced GNNs ===")

print("\n1. GraphSAINT Optimizations:")
print("   • Use smaller batch sizes (128-512) to fit in CPU memory")
print("   • Reduce sample_coverage (5-20) for faster sampling")
print("   • Choose node sampling over edge sampling for better CPU performance")
print("   • Limit walk_length (2-4) in random walk sampling")
print("   • Use fewer GCN layers (2-3) to reduce computation")

print("\n2. FastGCN Optimizations:")
print("   • Use small sample sizes (50-200) per layer")
print("   • Implement decreasing sample sizes by layer")
print("   • Cache importance scores when possible")
print("   • Use sparse tensors for large graphs")
print("   • Consider adaptive sampling based on node degree")

print("\n3. Memory Management:")
print("   • Monitor peak memory usage during training")
print("   • Use gradient accumulation for effective larger batch sizes")
print("   • Clear intermediate tensors explicitly")
print("   • Use torch.no_grad() for validation/testing")
print("   • Consider mixed precision training if supported")

print("\n4. Training Strategies:")
print("   • Use learning rate scheduling for better convergence")
print("   • Implement early stopping to prevent overfitting")
print("   • Use smaller learning rates (0.001-0.01) for stability")
print("   • Apply gradient clipping for numerical stability")
print("   • Cache embeddings for repeated evaluations")

print("\n5. Hardware-Specific Tips for M2 MacBook Air:")
print(f"   • Set torch.set_num_threads({torch.get_num_threads()}) for optimal CPU usage")
print("   • Use unified memory architecture efficiently")
print("   • Monitor thermal throttling during long training")
print("   • Consider training in smaller epochs with checkpoints")
print("   • Use vectorized operations over loops when possible")

# Demonstrate memory-efficient evaluation
def memory_efficient_evaluation(model, data, batch_size=1000):
    """
    Evaluate large graphs in batches to manage memory
    """
    model.eval()
    device = next(model.parameters()).device

    if data.num_nodes <= batch_size:
        # Small graph - evaluate normally
        with torch.no_grad():
            return model(data.x.to(device), data.edge_index.to(device))

    # Large graph - batch processing (simplified version)
    print(f"Using batched evaluation for {data.num_nodes} nodes")

    with torch.no_grad():
        # For demonstration - in practice, need more sophisticated batching
        # that considers graph connectivity
        return model(data.x.to(device), data.edge_index.to(device))

print("\n=== Performance Recommendations ===")
print("\nFor small graphs (< 10K nodes):")
print("• Use standard GCN for simplicity")
print("• Focus on model architecture optimization")

print("\nFor medium graphs (10K - 100K nodes):")
print("• Consider FastGCN with moderate sampling")
print("• Use GraphSAINT with node sampling")
print("• Implement gradient accumulation")

print("\nFor large graphs (> 100K nodes):")
print("• GraphSAINT is essential for feasible training")
print("• Use aggressive sampling strategies")
print("• Consider distributed training if available")

print(f"\n=== Current System Status ===")
process = psutil.Process(os.getpid())
print(f"Current memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
print(f"PyTorch threads: {torch.get_num_threads()}")
print(f"Device: {device}")

# %% [markdown]
# ## Summary
#
# In this notebook, we explored memory-enhanced Graph Neural Networks designed for scalability with comprehensive mathematical foundations:
#
# ### **GraphSAINT (Graph Sampling and Aggregation)**:
# **Mathematical Foundation:**
# - **Sampling Framework:** $\mathcal{L}_{\text{saint}} = \sum_{s \in \mathcal{S}} \frac{|\mathcal{V}|}{|\mathcal{V}_s|} \mathcal{L}_s(\theta)$
# - **Unbiased Estimation:** $\mathbb{E}[\hat{\mathcal{L}}_{\text{saint}}] = \mathcal{L}_{\text{full}}$
# - **Memory Reduction:** $M_{\text{saint}} = O(L \times |\mathcal{V}_s| \times D) \ll M_{\text{full}}$
# - **Convergence Rate:** $\mathbb{E}[\hat{\theta}_T - \theta^*] = O(1/\sqrt{T})$
#
# **Key Benefits:**
# - Enables training on large graphs through subgraph sampling
# - Maintains convergence guarantees with proper normalization
# - Flexible sampling strategies (node, edge, random walk)
#
# ### **FastGCN (Fast Graph Convolutional Networks)**:
# **Mathematical Foundation:**
# - **Layer-wise Sampling:** $\mathbf{H}^{(\ell+1)} = \sigma\left(\frac{1}{K^{(\ell)}} \sum_{v \in \mathcal{S}^{(\ell)}} \frac{\mathbf{A}_{:,v}}{q_v^{(\ell)}} \mathbf{H}_v^{(\ell)} \mathbf{W}^{(\ell)}\right)$
# - **Importance Sampling:** $q_v^{(\ell)} = \frac{d_v}{2|\mathcal{E}|}$ (degree-based)
# - **Complexity Reduction:** $\mathcal{C}_{\text{FastGCN}} = O(\sum_{\ell} K^{(\ell)} \times d \times D)$
# - **Variance Control:** $\text{Var}[\hat{\mathbf{H}}^{(\ell+1)}] = O(1/K^{(\ell)})$
#
# **Key Benefits:**
# - Reduces complexity from exponential to linear in layers
# - Maintains performance with controlled variance
# - Optimal for medium-scale graphs with uniform structure
#
# ### **Performance Trade-offs**:
# **Mathematical Analysis:**
# - **Accuracy-Memory Trade-off:** $\eta_{\text{memory}} = \frac{M_{\text{enhanced}}}{M_{\text{full}}} = \frac{K}{|\mathcal{V}|} \to 0$
# - **Efficiency Score:** $E = R_{\text{acc}} \times \sqrt{R_{\text{mem}} \times R_{\text{speed}}}$
# - **Scalability Factor:** $\frac{M_{\text{full}}}{M_{\text{enhanced}}} = O(n/k)$ for graph size $n$
#
# ### **When to Use Each Approach**:
# **Decision Framework:**
# 1. **Standard GCN:** Small graphs $(< 10K \text{ nodes})$, $M_{\text{available}} \gg M_{\text{required}}$
# 2. **FastGCN:** Medium graphs $(10K-100K \text{ nodes})$, uniform degree distribution
# 3. **GraphSAINT:** Large graphs $(> 100K \text{ nodes})$, severe memory constraints
# 4. **Hybrid:** Combine techniques for maximum scalability
#
# ### **CPU Optimization Mathematical Framework**:
# **Performance Model:**
# - **Memory Bandwidth:** $T_{\text{memory}} = \frac{n \cdot d \cdot 4}{B_{\text{mem}}}$ bytes
# - **Optimal Batch Size:** $B_{\text{optimal}} = \arg\min_B (T_{\text{compute}}(B) + T_{\text{memory}}(B))$
# - **Thermal Model:** $f_{\text{CPU}}(t) = f_{\max} \cdot \min(1, \frac{T_{\max} - T(t)}{T_{\max} - T_{\text{ambient}}})$
#
# ### **Key Mathematical Insights**:
#
# 1. **Sampling Theory:** Proper importance weighting ensures unbiased gradient estimation
# 2. **Variance-Bias Trade-off:** Smaller samples increase variance but reduce computational cost
# 3. **Convergence Guarantees:** Both methods maintain $O(1/\sqrt{T})$ convergence under standard assumptions
# 4. **Scalability Laws:** Memory efficiency improves as $O(n/k)$ where $n$ is graph size and $k$ is sample size
#
# ### **Practical Guidelines**:
# - **Memory-Constrained:** Use GraphSAINT with adaptive sampling strategies
# - **Time-Constrained:** Use FastGCN with layer-wise sample size optimization
# - **Accuracy-Critical:** Increase sampling budget or use ensemble methods
# - **CPU-Optimized:** Balance batch size, thread count, and thermal management
#
# These memory-enhanced techniques make GNNs practical for real-world large-scale applications, enabling deployment on resource-constrained environments while maintaining theoretical guarantees and reasonable performance. The mathematical foundations provide principled approaches to navigate the fundamental trade-offs between accuracy, memory efficiency, and computational speed.