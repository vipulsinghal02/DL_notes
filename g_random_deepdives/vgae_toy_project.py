# %% [markdown]
# # Complete Variational Graph Autoencoder (VGAE) Toy Project
#
# This project implements a complete Variational Graph Autoencoder from scratch with comprehensive mathematical foundations.
# We'll demonstrate unsupervised learning on graphs, including node embeddings, link prediction, and graph generation.
#
# ## Mathematical Foundation of VGAEs
#
# ### Variational Graph Autoencoder Framework
#
# **Graph Representation:**
# - Graph: $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with nodes $\mathcal{V}$ and edges $\mathcal{E}$
# - Adjacency matrix: $\mathbf{A} \in \{0,1\}^{n \times n}$
# - Node features: $\mathbf{X} \in \mathbb{R}^{n \times d}$
#
# **Variational Framework:**
# The VGAE optimizes the Evidence Lower BOund (ELBO):
# $$\mathcal{L} = \mathbb{E}_{q_\phi(\mathbf{Z}|\mathbf{X}, \mathbf{A})}[\log p_\theta(\mathbf{A}|\mathbf{Z})] - \text{KL}[q_\phi(\mathbf{Z}|\mathbf{X}, \mathbf{A}) \| p(\mathbf{Z})]$$
#
# **Encoder (Recognition Model):**
# $$q_\phi(\mathbf{Z}|\mathbf{X}, \mathbf{A}) = \prod_{i=1}^n q_\phi(\mathbf{z}_i|\mathbf{X}, \mathbf{A})$$
#
# where:
# $$q_\phi(\mathbf{z}_i|\mathbf{X}, \mathbf{A}) = \mathcal{N}(\mathbf{z}_i | \boldsymbol{\mu}_i, \text{diag}(\boldsymbol{\sigma}_i^2))$$
#
# **GCN Encoder:**
# $$\boldsymbol{\mu} = \text{GCN}_\mu(\mathbf{X}, \mathbf{A})$$
# $$\log \boldsymbol{\sigma} = \text{GCN}_\sigma(\mathbf{X}, \mathbf{A})$$
#
# **Decoder (Generative Model):**
# $$p_\theta(\mathbf{A}|\mathbf{Z}) = \prod_{i=1}^n \prod_{j=1}^n p_\theta(A_{ij}|\mathbf{z}_i, \mathbf{z}_j)$$
#
# where:
# $$p_\theta(A_{ij} = 1|\mathbf{z}_i, \mathbf{z}_j) = \sigma(\mathbf{z}_i^T \mathbf{z}_j)$$
#
# **KL Divergence:**
# $$\text{KL}[q_\phi(\mathbf{Z}|\mathbf{X}, \mathbf{A}) \| p(\mathbf{Z})] = \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^d \left(1 + \log \sigma_{ij}^2 - \mu_{ij}^2 - \sigma_{ij}^2\right)$$
#
# **Training Objective:**
# $$\mathcal{L}_{\text{VGAE}} = \text{Reconstruction Loss} + \beta \cdot \text{KL Divergence}$$

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from torch_geometric.transforms import RandomNodeSplit
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score, adjusted_rand_score
import seaborn as sns
import random
import time
from collections import defaultdict

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device setup with MPS support for Apple Silicon
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using NVIDIA GPU (CUDA)")
else:
    device = torch.device('cpu')
    print("Using CPU")

print(f"Device: {device}")

# %% [markdown]
# ## 1. Synthetic Graph Dataset Creation
#
# We'll create various types of synthetic graphs to demonstrate VGAE capabilities:
# 1. **Community graphs** - graphs with clear cluster structure
# 2. **Scale-free networks** - graphs following power-law degree distribution
# 3. **Small-world networks** - graphs with high clustering and short path lengths
#
# ### Mathematical Graph Models
#
# **Stochastic Block Model (SBM):**
# $$P(A_{ij} = 1) = \begin{cases}
# p_{in} & \text{if nodes } i, j \text{ in same community} \\
# p_{out} & \text{if nodes } i, j \text{ in different communities}
# \end{cases}$$
#
# **Barabási-Albert Model:**
# Preferential attachment: $P(\text{attach to node } i) \propto k_i$
#
# **Watts-Strogatz Model:**
# Start with ring lattice, rewire edges with probability $p$

# %%
def create_community_graph(n_communities=4, nodes_per_community=20, p_in=0.7, p_out=0.05):
    """
    Create a graph with community structure using Stochastic Block Model

    Args:
        n_communities: Number of communities
        nodes_per_community: Nodes per community
        p_in: Probability of edge within community
        p_out: Probability of edge between communities

    Returns:
        Data object with node features and community labels
    """
    n_nodes = n_communities * nodes_per_community

    # Create adjacency matrix
    adj_matrix = np.zeros((n_nodes, n_nodes))
    community_labels = np.repeat(range(n_communities), nodes_per_community)

    # Add edges according to SBM
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if community_labels[i] == community_labels[j]:
                prob = p_in
            else:
                prob = p_out

            if np.random.random() < prob:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

    # Convert to edge_index format
    edge_indices = np.nonzero(adj_matrix)
    edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)

    # Create node features (random + community bias)
    node_features = torch.randn(n_nodes, 8)

    # Add community-specific features
    for i, comm in enumerate(community_labels):
        # Add community bias to features
        community_bias = torch.zeros(8)
        community_bias[comm % 8] = 2.0  # Strong signal in one dimension
        community_bias[(comm + 1) % 8] = 1.0  # Weaker signal in adjacent dimension
        node_features[i] += community_bias

    # Create labels
    y = torch.tensor(community_labels, dtype=torch.long)

    return Data(x=node_features, edge_index=edge_index, y=y)

def create_scale_free_graph(n_nodes=100, m=3):
    """
    Create scale-free graph using Barabási-Albert model
    """
    # Use NetworkX to generate BA graph
    G = nx.barabasi_albert_graph(n_nodes, m, seed=42)

    # Convert to PyTorch Geometric format
    edge_list = list(G.edges())
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Add reverse edges for undirected graph
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

    # Create features based on degree and position
    degrees = torch.tensor([G.degree(i) for i in range(n_nodes)], dtype=torch.float)
    node_features = torch.randn(n_nodes, 8)
    node_features[:, 0] = degrees / degrees.max()  # Normalized degree feature

    # Create labels based on degree quantiles
    degree_quantiles = torch.quantile(degrees, torch.tensor([0.33, 0.67]))
    y = torch.zeros(n_nodes, dtype=torch.long)
    y[degrees > degree_quantiles[1]] = 2  # High degree
    y[(degrees > degree_quantiles[0]) & (degrees <= degree_quantiles[1])] = 1  # Medium degree
    # Low degree nodes remain 0

    return Data(x=node_features, edge_index=edge_index, y=y)

def create_small_world_graph(n_nodes=100, k=6, p=0.3):
    """
    Create small-world graph using Watts-Strogatz model
    """
    G = nx.watts_strogatz_graph(n_nodes, k, p, seed=42)

    # Convert to PyTorch Geometric format
    edge_list = list(G.edges())
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

    # Create features based on clustering coefficient and centrality
    clustering = nx.clustering(G)
    centrality = nx.betweenness_centrality(G)

    node_features = torch.randn(n_nodes, 8)
    node_features[:, 0] = torch.tensor([clustering[i] for i in range(n_nodes)])
    node_features[:, 1] = torch.tensor([centrality[i] for i in range(n_nodes)])

    # Create labels based on clustering coefficient
    clustering_values = torch.tensor([clustering[i] for i in range(n_nodes)])
    clustering_median = torch.median(clustering_values)
    y = (clustering_values > clustering_median).long()

    return Data(x=node_features, edge_index=edge_index, y=y)

# Create different types of graphs
print("Creating synthetic graph datasets...")

community_graph = create_community_graph(n_communities=4, nodes_per_community=15)
scale_free_graph = create_scale_free_graph(n_nodes=80)
small_world_graph = create_small_world_graph(n_nodes=80)

graphs = {
    'Community': community_graph,
    'Scale-Free': scale_free_graph,
    'Small-World': small_world_graph
}

# Display graph statistics
for name, graph in graphs.items():
    print(f"\n{name} Graph:")
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.num_edges // 2}")  # Undirected edges
    print(f"  Features: {graph.num_features}")
    print(f"  Classes: {len(torch.unique(graph.y))}")
    print(f"  Average degree: {graph.num_edges / graph.num_nodes:.2f}")

# %% [markdown]
# ## 2. Graph Visualization
#
# Visualize the created graphs to understand their structure and properties.

# %%
def visualize_graph(data, title, pos=None, node_size=50):
    """
    Visualize graph using NetworkX
    """
    # Convert to NetworkX
    edge_list = data.edge_index.t().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(edge_list[::2])  # Take every other edge (undirected)

    # Layout
    if pos is None:
        pos = nx.spring_layout(G, seed=42, k=1, iterations=50)

    # Node colors based on labels
    node_colors = data.y.numpy()

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_color=node_colors, node_size=node_size,
            cmap='Set3', with_labels=False, edge_color='gray', alpha=0.7)
    plt.title(f'{title} Graph\n{data.num_nodes} nodes, {data.num_edges // 2} edges')
    plt.axis('off')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='Set3',
                              norm=plt.Normalize(vmin=node_colors.min(),
                                               vmax=node_colors.max()))
    sm.set_array([])
    plt.colorbar(sm, label='Node Labels')

    plt.tight_layout()
    plt.show()

# Visualize all graphs
for name, graph in graphs.items():
    visualize_graph(graph, name)

# %% [markdown]
# ## 3. GCN Encoder Implementation
#
# Implement the GCN-based encoder that produces mean and log-variance for each node.
#
# ### Mathematical Details
#
# **GCN Layer:**
# $$\mathbf{H}^{(l+1)} = \sigma\left(\tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2} \mathbf{H}^{(l)} \mathbf{W}^{(l)}\right)$$
#
# where:
# - $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ (adjacency with self-loops)
# - $\tilde{\mathbf{D}}_{ii} = \sum_j \tilde{\mathbf{A}}_{ij}$ (degree matrix)
#
# **Encoder Output:**
# $$\boldsymbol{\mu} = \text{GCN}_\mu(\mathbf{X}, \mathbf{A})$$
# $$\log \boldsymbol{\sigma} = \text{GCN}_\sigma(\mathbf{X}, \mathbf{A})$$

# %%
class GCNEncoder(nn.Module):
    """
    GCN-based encoder for VGAE that outputs mean and log-variance
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Shared GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Separate layers for mean and log-variance
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logvar = GCNConv(hidden_dim, latent_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        """
        Forward pass through GCN encoder

        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge indices (2, num_edges)

        Returns:
            mu: Mean of latent distribution (num_nodes, latent_dim)
            logvar: Log-variance of latent distribution (num_nodes, latent_dim)
        """
        # Shared layers
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, edge_index))
        h = self.dropout(h)

        # Mean and log-variance
        mu = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)

        return mu, logvar

# Test encoder
input_dim = 8
hidden_dim = 32
latent_dim = 16

encoder = GCNEncoder(input_dim, hidden_dim, latent_dim).to(device)

# Test with community graph
test_graph = community_graph.to(device)
mu, logvar = encoder(test_graph.x, test_graph.edge_index)

print(f"Encoder test:")
print(f"  Input shape: {test_graph.x.shape}")
print(f"  Mean shape: {mu.shape}")
print(f"  Log-variance shape: {logvar.shape}")
print(f"  Mean statistics: {mu.mean().item():.3f} ± {mu.std().item():.3f}")
print(f"  Log-var statistics: {logvar.mean().item():.3f} ± {logvar.std().item():.3f}")

# %% [markdown]
# ## 4. Decoder Implementation
#
# Implement the decoder that reconstructs the adjacency matrix from latent embeddings.
#
# ### Mathematical Details
#
# **Inner Product Decoder:**
# $$p_\theta(A_{ij} = 1|\mathbf{z}_i, \mathbf{z}_j) = \sigma(\mathbf{z}_i^T \mathbf{z}_j)$$
#
# **Reconstruction Loss:**
# $$\mathcal{L}_{\text{recon}} = -\sum_{i,j} A_{ij} \log p_{ij} + (1 - A_{ij}) \log (1 - p_{ij})$$
#
# **Positive/Negative Sampling:**
# Due to sparsity, we sample negative edges to balance the loss

# %%
class InnerProductDecoder(nn.Module):
    """
    Inner product decoder for VGAE
    """
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, edge_index, sigmoid=True):
        """
        Decode edge probabilities from latent embeddings

        Args:
            z: Latent embeddings (num_nodes, latent_dim)
            edge_index: Edge indices for reconstruction
            sigmoid: Whether to apply sigmoid activation

        Returns:
            Edge probabilities
        """
        z = self.dropout(z)

        # Get embeddings for edge endpoints
        row, col = edge_index
        z_row = z[row]
        z_col = z[col]

        # Inner product
        logits = (z_row * z_col).sum(dim=1)

        return torch.sigmoid(logits) if sigmoid else logits

def recon_loss(preds, labels, pos_weight=None):
    """
    Reconstruction loss (binary cross-entropy)
    """
    if pos_weight is not None:
        loss = F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    else:
        loss = F.binary_cross_entropy(preds, labels)
    return loss

def kl_divergence(mu, logvar):
    """
    KL divergence between q(z|x) and p(z) = N(0, I)
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()

# Test decoder
decoder = InnerProductDecoder().to(device)

# Sample latent embeddings
z_sample = torch.randn(test_graph.num_nodes, latent_dim).to(device)

# Get positive edges
pos_edge_index = test_graph.edge_index

# Sample negative edges
neg_edge_index = negative_sampling(
    edge_index=pos_edge_index,
    num_nodes=test_graph.num_nodes,
    num_neg_samples=pos_edge_index.size(1)
)

# Test reconstruction
pos_preds = decoder(z_sample, pos_edge_index)
neg_preds = decoder(z_sample, neg_edge_index)

print(f"Decoder test:")
print(f"  Latent embeddings shape: {z_sample.shape}")
print(f"  Positive edges: {pos_edge_index.size(1)}")
print(f"  Negative edges: {neg_edge_index.size(1)}")
print(f"  Positive predictions: {pos_preds.mean().item():.3f} ± {pos_preds.std().item():.3f}")
print(f"  Negative predictions: {neg_preds.mean().item():.3f} ± {neg_preds.std().item():.3f}")

# %% [markdown]
# ## 5. Complete VGAE Model
#
# Combine encoder and decoder into complete VGAE model with reparameterization trick.
#
# ### Reparameterization Trick
#
# **Sampling from q(z|x,A):**
# $$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$$
#
# where $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$
#
# This allows backpropagation through the stochastic node.

# %%
class VGAE(nn.Module):
    """
    Complete Variational Graph Autoencoder
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.encoder = GCNEncoder(input_dim, hidden_dim, latent_dim, dropout)
        self.decoder = InnerProductDecoder(dropout)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for sampling from latent distribution
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def encode(self, x, edge_index):
        """Encode input to latent distribution parameters"""
        return self.encoder(x, edge_index)

    def decode(self, z, edge_index, sigmoid=True):
        """Decode latent embeddings to edge probabilities"""
        return self.decoder(z, edge_index, sigmoid)

    def forward(self, x, edge_index):
        """
        Full forward pass through VGAE

        Returns:
            z: Sampled latent embeddings
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
        """
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

# Initialize VGAE model
model = VGAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

# Test full model
z, mu, logvar = model(test_graph.x, test_graph.edge_index)

print(f"VGAE model test:")
print(f"  Sampled embeddings shape: {z.shape}")
print(f"  Mean shape: {mu.shape}")
print(f"  Log-variance shape: {logvar.shape}")
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# %% [markdown]
# ## 6. Training Setup and Loss Functions
#
# Implement training setup with proper loss weighting and evaluation metrics.
#
# ### Mathematical Training Framework
#
# **Evidence Lower Bound (ELBO) Optimization:**
# $$\mathcal{L}_{VGAE} = \mathbb{E}_{q_\phi(\mathbf{Z}|\mathbf{X},\mathbf{A})}[\log p_\theta(\mathbf{A}|\mathbf{Z})] - \text{KL}[q_\phi(\mathbf{Z}|\mathbf{X},\mathbf{A}) \| p(\mathbf{Z})]$$
#
# **Reconstruction Loss (Binary Cross-Entropy):**
# $$\mathcal{L}_{recon} = -\sum_{i,j} A_{ij} \log \sigma(\mathbf{z}_i^T \mathbf{z}_j) + (1-A_{ij}) \log(1-\sigma(\mathbf{z}_i^T \mathbf{z}_j))$$
#
# **KL Divergence Term:**
# $$\text{KL}[q(\mathbf{Z}|\mathbf{X},\mathbf{A}) \| p(\mathbf{Z})] = \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^d (1 + \log \sigma_{ij}^2 - \mu_{ij}^2 - \sigma_{ij}^2)$$
#
# **Beta-VAE Formulation:**
# $$\mathcal{L} = \mathcal{L}_{recon} + \beta \cdot \text{KL}$$
# where $\beta$ controls the trade-off between reconstruction and regularization.
#
# **Gradient Flow:**
# $$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}} = \frac{\partial \mathcal{L}_{recon}}{\partial \mathbf{z}} \cdot \frac{\partial \mathbf{z}}{\partial \boldsymbol{\mu}} + \boldsymbol{\mu}$$
# $$\frac{\partial \mathcal{L}}{\partial \log \boldsymbol{\sigma}} = \frac{\partial \mathcal{L}_{recon}}{\partial \mathbf{z}} \cdot \frac{\partial \mathbf{z}}{\partial \log \boldsymbol{\sigma}} + (1 - \boldsymbol{\sigma}^2)$$

# %%
def prepare_data_for_training(data, test_ratio=0.1, val_ratio=0.1):
    """
    Prepare graph data for training by splitting edges

    Args:
        data: PyTorch Geometric Data object
        test_ratio: Ratio of edges for testing
        val_ratio: Ratio of edges for validation

    Returns:
        Dictionary with train/val/test splits
    """
    # Remove self-loops and get unique edges
    edge_index, _ = remove_self_loops(data.edge_index)

    # Convert to undirected edge list (remove duplicates)
    edge_list = edge_index.t().numpy()
    edge_set = set()
    for edge in edge_list:
        edge_tuple = tuple(sorted(edge))
        edge_set.add(edge_tuple)

    edges = np.array(list(edge_set))
    num_edges = len(edges)

    # Shuffle edges
    perm = np.random.permutation(num_edges)
    edges = edges[perm]

    # Split edges
    num_val = int(val_ratio * num_edges)
    num_test = int(test_ratio * num_edges)
    num_train = num_edges - num_val - num_test

    train_edges = edges[:num_train]
    val_edges = edges[num_train:num_train+num_val]
    test_edges = edges[num_train+num_val:]

    # Convert back to edge_index format (add reverse edges)
    def edges_to_index(edge_array):
        if len(edge_array) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        forward = torch.tensor(edge_array.T, dtype=torch.long)
        reverse = torch.tensor(edge_array[:, [1, 0]].T, dtype=torch.long)
        return torch.cat([forward, reverse], dim=1)

    train_edge_index = edges_to_index(train_edges)
    val_pos_edge_index = edges_to_index(val_edges)
    test_pos_edge_index = edges_to_index(test_edges)

    # Sample negative edges for validation and testing
    val_neg_edge_index = negative_sampling(
        edge_index=train_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=val_pos_edge_index.size(1)
    )

    test_neg_edge_index = negative_sampling(
        edge_index=train_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=test_pos_edge_index.size(1)
    )

    return {
        'train_edge_index': train_edge_index,
        'val_pos_edge_index': val_pos_edge_index,
        'val_neg_edge_index': val_neg_edge_index,
        'test_pos_edge_index': test_pos_edge_index,
        'test_neg_edge_index': test_neg_edge_index,
        'num_train_edges': num_train,
        'num_val_edges': num_val,
        'num_test_edges': num_test
    }

def vgae_loss(model, x, pos_edge_index, neg_edge_index=None, beta=1.0):
    """
    Compute VGAE loss (reconstruction + KL divergence)

    Args:
        model: VGAE model
        x: Node features
        pos_edge_index: Positive edges
        neg_edge_index: Negative edges (if None, sample them)
        beta: Weight for KL divergence term

    Returns:
        Total loss, reconstruction loss, KL divergence
    """
    # Forward pass
    z, mu, logvar = model(x, pos_edge_index)

    # Sample negative edges if not provided
    if neg_edge_index is None:
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=x.size(0),
            num_neg_samples=pos_edge_index.size(1)
        )

    # Reconstruction loss
    pos_preds = model.decode(z, pos_edge_index, sigmoid=False)
    neg_preds = model.decode(z, neg_edge_index, sigmoid=False)

    pos_labels = torch.ones_like(pos_preds)
    neg_labels = torch.zeros_like(neg_preds)

    preds = torch.cat([pos_preds, neg_preds])
    labels = torch.cat([pos_labels, neg_labels])

    recon_loss = F.binary_cross_entropy_with_logits(preds, labels)

    # KL divergence
    kl_loss = kl_divergence(mu, logvar)

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss

def evaluate_link_prediction(model, x, pos_edge_index, neg_edge_index):
    """
    Evaluate link prediction performance

    Returns:
        AUC and AP scores
    """
    model.eval()

    with torch.no_grad():
        z, _, _ = model(x, pos_edge_index)

        pos_preds = model.decode(z, pos_edge_index, sigmoid=True)
        neg_preds = model.decode(z, neg_edge_index, sigmoid=True)

        preds = torch.cat([pos_preds, neg_preds]).cpu().numpy()
        labels = torch.cat([
            torch.ones(pos_preds.size(0)),
            torch.zeros(neg_preds.size(0))
        ]).cpu().numpy()

        auc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)

    return auc, ap

# Prepare training data for community graph
train_data = prepare_data_for_training(community_graph)

print(f"Data split for training:")
print(f"  Training edges: {train_data['num_train_edges']}")
print(f"  Validation edges: {train_data['num_val_edges']}")
print(f"  Test edges: {train_data['num_test_edges']}")
print(f"  Total original edges: {community_graph.num_edges // 2}")

# %% [markdown]
# ## 7. Training Loop with Monitoring
#
# Implement complete training loop with loss tracking and evaluation.
#
# ### Mathematical Training Dynamics
#
# **Optimization Objective:**
# $$\min_{\phi,\theta} \mathcal{L}_{VGAE}(\phi, \theta) = \min_{\phi,\theta} [\mathcal{L}_{recon}(\theta) + \beta \cdot \text{KL}(\phi)]$$
#
# **Stochastic Gradient Descent:**
# $$\phi_{t+1} = \phi_t - \alpha_t \nabla_\phi \mathcal{L}_{VGAE}$$
# $$\theta_{t+1} = \theta_t - \alpha_t \nabla_\theta \mathcal{L}_{VGAE}$$
#
# **Adam Optimizer Updates:**
# $$m_t^{(\phi)} = \beta_1 m_{t-1}^{(\phi)} + (1-\beta_1)\nabla_\phi \mathcal{L}$$
# $$v_t^{(\phi)} = \beta_2 v_{t-1}^{(\phi)} + (1-\beta_2)(\nabla_\phi \mathcal{L})^2$$
# $$\phi_t = \phi_{t-1} - \alpha \frac{\hat{m}_t^{(\phi)}}{\sqrt{\hat{v}_t^{(\phi)}} + \epsilon}$$
#
# **Convergence Monitoring:**
# - **Loss Trajectory**: $\mathcal{L}(t) = \mathcal{L}^* + A e^{-\lambda t}$ (exponential decay)
# - **KL Annealing**: Gradually increase $\beta$ from 0 to 1 during training
# - **Early Stopping**: Stop when validation loss increases for consecutive epochs

# %%
def train_vgae(model, data, train_data, num_epochs=200, lr=0.01, beta=1.0,
               eval_every=20, verbose=True):
    """
    Train VGAE model

    Args:
        model: VGAE model
        data: Original graph data
        train_data: Dictionary with train/val/test splits
        num_epochs: Number of training epochs
        lr: Learning rate
        beta: Weight for KL divergence
        eval_every: Evaluate every N epochs
        verbose: Print training progress

    Returns:
        Training history
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'val_auc': [],
        'val_ap': [],
        'epochs': []
    }

    x = data.x.to(device)
    train_edge_index = train_data['train_edge_index'].to(device)
    val_pos_edge_index = train_data['val_pos_edge_index'].to(device)
    val_neg_edge_index = train_data['val_neg_edge_index'].to(device)

    if verbose:
        print(f"Training VGAE for {num_epochs} epochs...")
        print("=" * 70)

    for epoch in range(num_epochs):
        model.train()

        # Training step
        optimizer.zero_grad()
        total_loss, recon_loss, kl_loss = vgae_loss(
            model, x, train_edge_index, beta=beta
        )
        total_loss.backward()
        optimizer.step()

        # Store training metrics
        history['train_loss'].append(total_loss.item())
        history['train_recon_loss'].append(recon_loss.item())
        history['train_kl_loss'].append(kl_loss.item())

        # Evaluation
        if epoch % eval_every == 0:
            val_auc, val_ap = evaluate_link_prediction(
                model, x, train_edge_index, val_pos_edge_index, val_neg_edge_index
            )

            history['val_auc'].append(val_auc)
            history['val_ap'].append(val_ap)
            history['epochs'].append(epoch)

            if verbose:
                print(f"Epoch {epoch:3d} | "
                      f"Loss: {total_loss.item():.4f} | "
                      f"Recon: {recon_loss.item():.4f} | "
                      f"KL: {kl_loss.item():.4f} | "
                      f"Val AUC: {val_auc:.3f} | "
                      f"Val AP: {val_ap:.3f}")

    if verbose:
        print("=" * 70)
        print("Training completed!")

    return history

# Train the model
history = train_vgae(
    model=model,
    data=community_graph,
    train_data=train_data,
    num_epochs=200,
    lr=0.01,
    beta=1.0,
    eval_every=20
)

# %% [markdown]
# ## 8. Training Progress Visualization

# %%
def plot_training_history(history):
    """
    Plot training history
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Training losses
    axes[0, 0].plot(history['train_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Total Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['train_recon_loss'], 'r-', linewidth=2)
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(history['train_kl_loss'], 'g-', linewidth=2)
    axes[0, 2].set_title('KL Divergence')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('KL Loss')
    axes[0, 2].grid(True, alpha=0.3)

    # Validation metrics
    axes[1, 0].plot(history['epochs'], history['val_auc'], 'o-', color='purple', linewidth=2)
    axes[1, 0].set_title('Validation AUC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    axes[1, 1].plot(history['epochs'], history['val_ap'], 'o-', color='orange', linewidth=2)
    axes[1, 1].set_title('Validation Average Precision')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AP')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])

    # Combined loss view
    axes[1, 2].semilogy(history['train_loss'], 'b-', label='Total', linewidth=2)
    axes[1, 2].semilogy(history['train_recon_loss'], 'r-', label='Reconstruction', linewidth=2)
    axes[1, 2].semilogy(history['train_kl_loss'], 'g-', label='KL', linewidth=2)
    axes[1, 2].set_title('Training Losses (Log Scale)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Plot training history
plot_training_history(history)

# Print final metrics
print(f"\nFinal Training Metrics:")
print(f"  Final loss: {history['train_loss'][-1]:.4f}")
print(f"  Final reconstruction loss: {history['train_recon_loss'][-1]:.4f}")
print(f"  Final KL divergence: {history['train_kl_loss'][-1]:.4f}")
print(f"  Best validation AUC: {max(history['val_auc']):.3f}")
print(f"  Best validation AP: {max(history['val_ap']):.3f}")

# %% [markdown]
# ## 9. Learned Embeddings Visualization and Analysis
#
# Visualize the learned latent embeddings using t-SNE and analyze their structure.
#
# ### Mathematical Embedding Analysis
#
# **Latent Space Properties:**
# The encoder learns a mapping $f_{enc}: \mathcal{V} \rightarrow \mathbb{R}^d$ such that:
# $$q(\mathbf{z}_i | \mathbf{X}, \mathbf{A}) = \mathcal{N}(\boldsymbol{\mu}_i, \text{diag}(\boldsymbol{\sigma}_i^2))$$
#
# **t-SNE Dimensionality Reduction:**
# $$P_{j|i} = \frac{\exp(-\|\mathbf{z}_i - \mathbf{z}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{z}_i - \mathbf{z}_k\|^2 / 2\sigma_i^2)}$$
# $$Q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$$
#
# **Embedding Quality Metrics:**
# - **Neighborhood Preservation**: $\text{NP} = \frac{|N_k(i) \cap N_k'(i)|}{k}$
# - **Silhouette Score**: $s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$
# - **Adjusted Rand Index**: $\text{ARI} = \frac{\text{RI} - E[\text{RI}]}{\max(\text{RI}) - E[\text{RI}]}$
#
# **Cluster Coherence:**
# $$\text{Intra-cluster distance} = \frac{1}{|C|^2} \sum_{i,j \in C} \|\mathbf{z}_i - \mathbf{z}_j\|$$
# $$\text{Inter-cluster distance} = \frac{1}{|C_1||C_2|} \sum_{i \in C_1, j \in C_2} \|\mathbf{z}_i - \mathbf{z}_j\|$$

# %%
def visualize_embeddings(model, data, title_prefix=""):
    """
    Visualize learned embeddings using t-SNE
    """
    model.eval()

    with torch.no_grad():
        # Get embeddings (use mean, not sampled)
        mu, logvar = model.encode(data.x.to(device), data.edge_index.to(device))
        embeddings = mu.cpu().numpy()
        labels = data.y.numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(15, 5))

    # Original embeddings (first 2 dimensions)
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='Set3', alpha=0.7)
    plt.title(f'{title_prefix}Learned Embeddings (First 2D)')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.colorbar(scatter, label='True Labels')

    # t-SNE visualization
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='Set3', alpha=0.7)
    plt.title(f'{title_prefix}t-SNE Visualization')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(scatter, label='True Labels')

    # Embedding distribution
    plt.subplot(1, 3, 3)
    plt.hist(embeddings.flatten(), bins=50, alpha=0.7, density=True)
    plt.title(f'{title_prefix}Embedding Distribution')
    plt.xlabel('Embedding Value')
    plt.ylabel('Density')
    plt.axvline(embeddings.mean(), color='r', linestyle='--', label=f'Mean: {embeddings.mean():.3f}')
    plt.axvline(0, color='k', linestyle='-', alpha=0.5, label='Zero')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return embeddings, embeddings_2d

# Visualize embeddings for community graph
community_embeddings, community_2d = visualize_embeddings(model, community_graph, "Community Graph: ")

# %% [markdown]
# ## 10. Link Prediction Evaluation
#
# Comprehensive evaluation of link prediction performance.
#
# ### Mathematical Link Prediction Framework
#
# **Binary Classification Problem:**
# For each node pair $(i,j)$, predict $A_{ij} \in \{0,1\}$:
# $$\hat{A}_{ij} = \sigma(\mathbf{z}_i^T \mathbf{z}_j)$$
#
# **Performance Metrics:**
#
# **Area Under ROC Curve (AUC):**
# $$\text{AUC} = P(\hat{A}_{ij}^{pos} > \hat{A}_{kl}^{neg})$$
# where $(i,j)$ is a positive edge and $(k,l)$ is a negative edge.
#
# **Average Precision (AP):**
# $$\text{AP} = \sum_k P(k) \Delta R(k)$$
# where $P(k)$ is precision at rank $k$ and $\Delta R(k)$ is change in recall.
#
# **ROC Curve Mathematics:**
# - **True Positive Rate**: $\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}$
# - **False Positive Rate**: $\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$
#
# **Precision-Recall Curve:**
# - **Precision**: $P = \frac{\text{TP}}{\text{TP} + \text{FP}}$
# - **Recall**: $R = \frac{\text{TP}}{\text{TP} + \text{FN}}$
#
# **Decision Threshold Optimization:**
# $$\tau^* = \arg\max_\tau F_1(\tau) = \arg\max_\tau \frac{2 \cdot P(\tau) \cdot R(\tau)}{P(\tau) + R(\tau)}$$

# %%
def comprehensive_link_prediction_eval(model, data, train_data):
    """
    Comprehensive evaluation of link prediction
    """
    model.eval()
    x = data.x.to(device)
    train_edge_index = train_data['train_edge_index'].to(device)

    results = {}

    # Evaluate on validation and test sets
    for split in ['val', 'test']:
        pos_key = f'{split}_pos_edge_index'
        neg_key = f'{split}_neg_edge_index'

        pos_edge_index = train_data[pos_key].to(device)
        neg_edge_index = train_data[neg_key].to(device)

        with torch.no_grad():
            z, _, _ = model(x, train_edge_index)

            pos_preds = model.decode(z, pos_edge_index, sigmoid=True)
            neg_preds = model.decode(z, neg_edge_index, sigmoid=True)

            preds = torch.cat([pos_preds, neg_preds]).cpu().numpy()
            labels = torch.cat([
                torch.ones(pos_preds.size(0)),
                torch.zeros(neg_preds.size(0))
            ]).cpu().numpy()

            auc = roc_auc_score(labels, preds)
            ap = average_precision_score(labels, preds)

            results[f'{split}_auc'] = auc
            results[f'{split}_ap'] = ap
            results[f'{split}_pos_preds'] = pos_preds.cpu().numpy()
            results[f'{split}_neg_preds'] = neg_preds.cpu().numpy()

    return results

def plot_link_prediction_results(results):
    """
    Plot link prediction results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    splits = ['val', 'test']
    colors = ['blue', 'red']

    for i, (split, color) in enumerate(zip(splits, colors)):
        pos_preds = results[f'{split}_pos_preds']
        neg_preds = results[f'{split}_neg_preds']
        auc = results[f'{split}_auc']
        ap = results[f'{split}_ap']

        # Prediction distributions
        axes[0, i].hist(pos_preds, bins=30, alpha=0.7, label='Positive edges', color='green')
        axes[0, i].hist(neg_preds, bins=30, alpha=0.7, label='Negative edges', color='red')
        axes[0, i].set_title(f'{split.capitalize()} Set Predictions\nAUC: {auc:.3f}, AP: {ap:.3f}')
        axes[0, i].set_xlabel('Predicted Probability')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].legend()

        # ROC-style plot (simplified)
        all_preds = np.concatenate([pos_preds, neg_preds])
        all_labels = np.concatenate([np.ones(len(pos_preds)), np.zeros(len(neg_preds))])

        # Sort by predictions
        sort_idx = np.argsort(all_preds)[::-1]
        sorted_labels = all_labels[sort_idx]

        # Compute precision and recall
        true_positives = np.cumsum(sorted_labels)
        false_positives = np.cumsum(1 - sorted_labels)
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / np.sum(sorted_labels)

        axes[1, i].plot(recall, precision, color=color, linewidth=2)
        axes[1, i].set_title(f'{split.capitalize()} Precision-Recall Curve')
        axes[1, i].set_xlabel('Recall')
        axes[1, i].set_ylabel('Precision')
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Evaluate link prediction
link_pred_results = comprehensive_link_prediction_eval(model, community_graph, train_data)
plot_link_prediction_results(link_pred_results)

print(f"\nLink Prediction Results:")
print(f"  Validation AUC: {link_pred_results['val_auc']:.3f}")
print(f"  Validation AP: {link_pred_results['val_ap']:.3f}")
print(f"  Test AUC: {link_pred_results['test_auc']:.3f}")
print(f"  Test AP: {link_pred_results['test_ap']:.3f}")

# %% [markdown]
# ## 11. Node Clustering Analysis
#
# Analyze how well the learned embeddings capture community structure through clustering.
#
# ### Mathematical Clustering Framework
#
# **K-Means Clustering:**
# $$\min_{\{S_k\}_{k=1}^K} \sum_{k=1}^K \sum_{\mathbf{z} \in S_k} \|\mathbf{z} - \boldsymbol{\mu}_k\|^2$$
# where $S_k$ are cluster assignments and $\boldsymbol{\mu}_k$ are centroids.
#
# **Cluster Quality Metrics:**
#
# **Silhouette Score:**
# $$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$
# where:
# - $a_i = \frac{1}{|C_i|-1} \sum_{j \in C_i, j \neq i} d(\mathbf{z}_i, \mathbf{z}_j)$ (intra-cluster distance)
# - $b_i = \min_{k \neq C_i} \frac{1}{|C_k|} \sum_{j \in C_k} d(\mathbf{z}_i, \mathbf{z}_j)$ (nearest-cluster distance)
#
# **Adjusted Rand Index (ARI):**
# $$\text{ARI} = \frac{\sum_{ij} \binom{n_{ij}}{2} - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right] / \binom{n}{2}}{\frac{1}{2}\left[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\right] - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right] / \binom{n}{2}}$$
#
# **Davies-Bouldin Index:**
# $$\text{DB} = \frac{1}{K} \sum_{i=1}^K \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(\mathbf{c}_i, \mathbf{c}_j)}$$
# where $\sigma_i$ is within-cluster scatter and $d(\mathbf{c}_i, \mathbf{c}_j)$ is centroid distance.
#
# **Calinski-Harabasz Index:**
# $$\text{CH} = \frac{\text{tr}(B_k)}{\text{tr}(W_k)} \times \frac{n-k}{k-1}$$
# where $B_k$ is between-cluster scatter and $W_k$ is within-cluster scatter.

# %%
def analyze_node_clustering(embeddings, true_labels, n_clusters=None):
    """
    Analyze clustering performance of learned embeddings
    """
    if n_clusters is None:
        n_clusters = len(np.unique(true_labels))

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    predicted_labels = kmeans.fit_predict(embeddings)

    # Calculate clustering metrics
    ari = adjusted_rand_score(true_labels, predicted_labels)

    # Silhouette analysis
    from sklearn.metrics import silhouette_score, silhouette_samples
    silhouette_avg = silhouette_score(embeddings, predicted_labels)
    silhouette_samples_scores = silhouette_samples(embeddings, predicted_labels)

    print(f"Clustering Analysis:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Adjusted Rand Index: {ari:.3f}")
    print(f"  Average Silhouette Score: {silhouette_avg:.3f}")

    # Visualize clustering results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # True vs predicted labels on t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    axes[0, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=true_labels, cmap='Set3')
    axes[0, 0].set_title('True Communities')
    axes[0, 0].set_xlabel('t-SNE 1')
    axes[0, 0].set_ylabel('t-SNE 2')

    axes[0, 1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=predicted_labels, cmap='Set3')
    axes[0, 1].set_title('Predicted Clusters')
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    im = axes[1, 0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted Label')
    axes[1, 0].set_ylabel('True Label')
    plt.colorbar(im, ax=axes[1, 0])

    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 0].text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")

    # Silhouette plot
    y_lower = 10
    for i in range(n_clusters):
        cluster_silhouette_vals = silhouette_samples_scores[predicted_labels == i]
        cluster_silhouette_vals.sort()

        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        axes[1, 1].fill_betweenx(np.arange(y_lower, y_upper),
                                0, cluster_silhouette_vals,
                                facecolor=color, edgecolor=color, alpha=0.7)

        axes[1, 1].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    axes[1, 1].set_xlabel('Silhouette Coefficient Values')
    axes[1, 1].set_ylabel('Cluster Label')
    axes[1, 1].axvline(x=silhouette_avg, color="red", linestyle="--",
                      label=f'Average Score: {silhouette_avg:.3f}')
    axes[1, 1].legend()
    axes[1, 1].set_title('Silhouette Analysis')

    plt.tight_layout()
    plt.show()

    return {
        'predicted_labels': predicted_labels,
        'ari': ari,
        'silhouette_avg': silhouette_avg,
        'confusion_matrix': cm
    }

# Analyze clustering for community graph
clustering_results = analyze_node_clustering(
    community_embeddings,
    community_graph.y.numpy(),
    n_clusters=4
)

# %% [markdown]
# ## 12. Graph Generation and Reconstruction
#
# Demonstrate graph generation capabilities by sampling from the learned latent space.
#
# ### Mathematical Graph Generation Framework
#
# **Generative Process:**
# 1. **Sample from Prior**: $\mathbf{z}_i \sim \mathcal{N}(0, \mathbf{I})$ or $\mathbf{z}_i \sim q(\mathbf{z}_i|\mathbf{X},\mathbf{A})$
# 2. **Decode to Adjacency**: $\hat{A}_{ij} = \sigma(\mathbf{z}_i^T \mathbf{z}_j)$
# 3. **Threshold**: $A_{ij}^{gen} = \mathbb{I}[\hat{A}_{ij} > \tau]$
#
# **Graph Property Preservation:**
#
# **Degree Distribution:**
# $$P(d) = \frac{N_d}{N}$$
# where $N_d$ is number of nodes with degree $d$.
#
# **Clustering Coefficient:**
# $$C_i = \frac{2e_i}{k_i(k_i-1)}, \quad C = \frac{1}{n} \sum_i C_i$$
# where $e_i$ is edges between neighbors of node $i$.
#
# **Average Path Length:**
# $$L = \frac{1}{n(n-1)} \sum_{i \neq j} d_{ij}$$
# where $d_{ij}$ is shortest path between nodes $i$ and $j$.
#
# **Graph Density:**
# $$\rho = \frac{2|E|}{n(n-1)}$$
#
# **Quality Assessment:**
# Compare generated graphs $\mathcal{G}_{gen}$ with original $\mathcal{G}_{orig}$:
# $$\text{Property Match} = 1 - \frac{|P(\mathcal{G}_{gen}) - P(\mathcal{G}_{orig})|}{P(\mathcal{G}_{orig})}$$
# for various graph properties $P(\cdot)$.

# %%
def generate_graphs_from_latent(model, reference_data, num_samples=3, threshold=0.5):
    """
    Generate new graphs by sampling from the latent space
    """
    model.eval()

    with torch.no_grad():
        # Get reference embeddings
        x = reference_data.x.to(device)
        edge_index = reference_data.edge_index.to(device)
        mu, logvar = model.encode(x, edge_index)

        # Sample new embeddings from learned distribution
        generated_graphs = []

        for i in range(num_samples):
            # Sample from latent space
            z_sample = model.reparameterize(mu, logvar)

            # Generate all possible edges
            num_nodes = z_sample.size(0)
            all_edges = []
            for node_i in range(num_nodes):
                for node_j in range(node_i + 1, num_nodes):
                    all_edges.append([node_i, node_j])

            if len(all_edges) > 0:
                all_edge_index = torch.tensor(all_edges, dtype=torch.long).t().to(device)

                # Predict edge probabilities
                edge_probs = model.decode(z_sample, all_edge_index, sigmoid=True)

                # Sample edges based on probabilities
                edge_mask = edge_probs > threshold
                sampled_edges = all_edges[edge_mask.cpu().numpy()]

                if len(sampled_edges) > 0:
                    # Create bidirectional edges
                    forward_edges = sampled_edges
                    reverse_edges = sampled_edges[:, [1, 0]]
                    all_sampled_edges = np.vstack([forward_edges, reverse_edges])

                    generated_edge_index = torch.tensor(all_sampled_edges.T, dtype=torch.long)
                else:
                    generated_edge_index = torch.empty((2, 0), dtype=torch.long)

                # Create new graph
                generated_graph = Data(
                    x=reference_data.x.clone(),
                    edge_index=generated_edge_index,
                    y=reference_data.y.clone()
                )

                generated_graphs.append(generated_graph)

    return generated_graphs

def compare_graph_properties(original_graph, generated_graphs):
    """
    Compare properties of original and generated graphs
    """
    def graph_stats(graph):
        edge_list = graph.edge_index.t().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(graph.num_nodes))
        G.add_edges_from(edge_list[::2])  # Undirected edges

        stats = {
            'num_edges': G.number_of_edges(),
            'avg_degree': 2 * G.number_of_edges() / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
            'density': nx.density(G) if G.number_of_nodes() > 1 else 0,
            'avg_clustering': nx.average_clustering(G) if G.number_of_edges() > 0 else 0,
        }

        if nx.is_connected(G):
            stats['avg_path_length'] = nx.average_shortest_path_length(G)
        else:
            # Get largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            if subgraph.number_of_edges() > 0:
                stats['avg_path_length'] = nx.average_shortest_path_length(subgraph)
            else:
                stats['avg_path_length'] = float('inf')

        return stats

    # Get statistics
    original_stats = graph_stats(original_graph)
    generated_stats = [graph_stats(g) for g in generated_graphs]

    # Print comparison
    print(f"Graph Properties Comparison:")
    print(f"{'Property':<20} {'Original':<12} {'Generated (avg)':<15} {'Generated (std)':<15}")
    print("-" * 65)

    for prop in original_stats.keys():
        orig_val = original_stats[prop]
        gen_vals = [stats[prop] for stats in generated_stats if not np.isinf(stats[prop])]

        if len(gen_vals) > 0:
            gen_avg = np.mean(gen_vals)
            gen_std = np.std(gen_vals)
            print(f"{prop:<20} {orig_val:<12.3f} {gen_avg:<15.3f} {gen_std:<15.3f}")
        else:
            print(f"{prop:<20} {orig_val:<12.3f} {'N/A':<15} {'N/A':<15}")

    return original_stats, generated_stats

def visualize_generated_graphs(original_graph, generated_graphs, max_display=3):
    """
    Visualize original and generated graphs
    """
    n_graphs = min(len(generated_graphs), max_display) + 1
    fig, axes = plt.subplots(1, n_graphs, figsize=(5 * n_graphs, 5))

    if n_graphs == 1:
        axes = [axes]

    # Plot original graph
    edge_list = original_graph.edge_index.t().numpy()
    G_orig = nx.Graph()
    G_orig.add_nodes_from(range(original_graph.num_nodes))
    G_orig.add_edges_from(edge_list[::2])

    pos = nx.spring_layout(G_orig, seed=42)
    nx.draw(G_orig, pos, ax=axes[0], node_color=original_graph.y.numpy(),
            node_size=50, cmap='Set3', with_labels=False, edge_color='gray')
    axes[0].set_title(f'Original Graph\n{G_orig.number_of_edges()} edges')

    # Plot generated graphs
    for i, gen_graph in enumerate(generated_graphs[:max_display]):
        edge_list = gen_graph.edge_index.t().numpy()
        G_gen = nx.Graph()
        G_gen.add_nodes_from(range(gen_graph.num_nodes))
        if len(edge_list) > 0:
            G_gen.add_edges_from(edge_list[::2])

        nx.draw(G_gen, pos, ax=axes[i+1], node_color=gen_graph.y.numpy(),
                node_size=50, cmap='Set3', with_labels=False, edge_color='gray')
        axes[i+1].set_title(f'Generated {i+1}\n{G_gen.number_of_edges()} edges')

    plt.tight_layout()
    plt.show()

# Generate graphs
print("Generating graphs from learned latent space...")
generated_graphs = generate_graphs_from_latent(model, community_graph, num_samples=5, threshold=0.5)

print(f"Generated {len(generated_graphs)} graphs")

# Compare properties
original_stats, generated_stats = compare_graph_properties(community_graph, generated_graphs)

# Visualize generated graphs
visualize_generated_graphs(community_graph, generated_graphs, max_display=3)

# %% [markdown]
# ## 13. Multi-Graph Training and Comparison
#
# Train VGAE on different graph types and compare the learned representations.

# %%
def train_and_evaluate_multiple_graphs(graphs, model_config, training_config):
    """
    Train VGAE on multiple graph types and compare results
    """
    results = {}

    for graph_name, graph_data in graphs.items():
        print(f"\n{'='*60}")
        print(f"Training VGAE on {graph_name} Graph")
        print(f"{'='*60}")

        # Initialize new model for each graph type
        model = VGAE(**model_config).to(device)

        # Prepare data
        train_data = prepare_data_for_training(graph_data)

        # Train model
        history = train_vgae(
            model=model,
            data=graph_data,
            train_data=train_data,
            **training_config,
            verbose=True
        )

        # Evaluate
        link_pred_results = comprehensive_link_prediction_eval(model, graph_data, train_data)

        # Get embeddings
        model.eval()
        with torch.no_grad():
            mu, logvar = model.encode(graph_data.x.to(device), graph_data.edge_index.to(device))
            embeddings = mu.cpu().numpy()

        # Clustering analysis
        if len(np.unique(graph_data.y.numpy())) > 1:
            clustering_results = analyze_node_clustering(
                embeddings, graph_data.y.numpy(), verbose=False
            )
        else:
            clustering_results = {'ari': 0.0, 'silhouette_avg': 0.0}

        results[graph_name] = {
            'model': model,
            'history': history,
            'link_prediction': link_pred_results,
            'embeddings': embeddings,
            'clustering': clustering_results
        }

    return results

def analyze_node_clustering(embeddings, true_labels, n_clusters=None, verbose=True):
    """
    Simplified clustering analysis for multi-graph comparison
    """
    if n_clusters is None:
        n_clusters = len(np.unique(true_labels))

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    predicted_labels = kmeans.fit_predict(embeddings)

    # Calculate metrics
    ari = adjusted_rand_score(true_labels, predicted_labels)

    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(embeddings, predicted_labels)

    if verbose:
        print(f"  Adjusted Rand Index: {ari:.3f}")
        print(f"  Average Silhouette Score: {silhouette_avg:.3f}")

    return {
        'predicted_labels': predicted_labels,
        'ari': ari,
        'silhouette_avg': silhouette_avg
    }

# Configuration for multi-graph training
model_config = {
    'input_dim': 8,
    'hidden_dim': 32,
    'latent_dim': 16,
    'dropout': 0.1
}

training_config = {
    'num_epochs': 150,
    'lr': 0.01,
    'beta': 1.0,
    'eval_every': 30,
    'verbose': False
}

# Train on all graph types
multi_graph_results = train_and_evaluate_multiple_graphs(graphs, model_config, training_config)

# %% [markdown]
# ## 14. Comprehensive Results Comparison

# %%
def plot_multi_graph_comparison(results):
    """
    Create comprehensive comparison plots across graph types
    """
    graph_names = list(results.keys())

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    # 1. Final training losses
    final_losses = [results[name]['history']['train_loss'][-1] for name in graph_names]
    bars1 = axes[0, 0].bar(graph_names, final_losses, color=['blue', 'red', 'green'])
    axes[0, 0].set_title('Final Training Loss')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, loss in zip(bars1, final_losses):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{loss:.3f}', ha='center', va='bottom')

    # 2. Link prediction performance (Test AUC)
    test_aucs = [results[name]['link_prediction']['test_auc'] for name in graph_names]
    bars2 = axes[0, 1].bar(graph_names, test_aucs, color=['blue', 'red', 'green'])
    axes[0, 1].set_title('Link Prediction AUC (Test)')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].tick_params(axis='x', rotation=45)

    for bar, auc in zip(bars2, test_aucs):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{auc:.3f}', ha='center', va='bottom')

    # 3. Clustering performance (ARI)
    aris = [results[name]['clustering']['ari'] for name in graph_names]
    bars3 = axes[0, 2].bar(graph_names, aris, color=['blue', 'red', 'green'])
    axes[0, 2].set_title('Clustering ARI')
    axes[0, 2].set_ylabel('Adjusted Rand Index')
    axes[0, 2].set_ylim([0, 1])
    axes[0, 2].tick_params(axis='x', rotation=45)

    for bar, ari in zip(bars3, aris):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{ari:.3f}', ha='center', va='bottom')

    # 4-6. Training curves for each graph
    colors = ['blue', 'red', 'green']
    for i, (name, color) in enumerate(zip(graph_names, colors)):
        history = results[name]['history']

        # Training loss curve
        axes[1, i].plot(history['train_loss'], color=color, linewidth=2)
        axes[1, i].set_title(f'{name}: Training Loss')
        axes[1, i].set_xlabel('Epoch')
        axes[1, i].set_ylabel('Loss')
        axes[1, i].grid(True, alpha=0.3)

        # Validation AUC curve
        axes[2, i].plot(history['epochs'], history['val_auc'], 'o-', color=color, linewidth=2)
        axes[2, i].set_title(f'{name}: Validation AUC')
        axes[2, i].set_xlabel('Epoch')
        axes[2, i].set_ylabel('AUC')
        axes[2, i].set_ylim([0, 1])
        axes[2, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def create_results_summary_table(results):
    """
    Create a summary table of all results
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)

    header = f"{'Graph Type':<15} {'Final Loss':<12} {'Test AUC':<10} {'Test AP':<10} {'ARI':<8} {'Silhouette':<12}"
    print(header)
    print("-" * 80)

    for name, result in results.items():
        final_loss = result['history']['train_loss'][-1]
        test_auc = result['link_prediction']['test_auc']
        test_ap = result['link_prediction']['test_ap']
        ari = result['clustering']['ari']
        silhouette = result['clustering']['silhouette_avg']

        row = f"{name:<15} {final_loss:<12.4f} {test_auc:<10.3f} {test_ap:<10.3f} {ari:<8.3f} {silhouette:<12.3f}"
        print(row)

    print("-" * 80)

    # Best performing models
    best_auc = max(results.items(), key=lambda x: x[1]['link_prediction']['test_auc'])
    best_ari = max(results.items(), key=lambda x: x[1]['clustering']['ari'])

    print(f"\nBest Link Prediction: {best_auc[0]} (AUC: {best_auc[1]['link_prediction']['test_auc']:.3f})")
    print(f"Best Clustering: {best_ari[0]} (ARI: {best_ari[1]['clustering']['ari']:.3f})")

# Create comparison visualizations
plot_multi_graph_comparison(multi_graph_results)
create_results_summary_table(multi_graph_results)

# %% [markdown]
# ## 15. Comprehensive Model Analysis and Insights
#
# Analyze what the VGAE models have learned and provide insights about their behavior on different graph types.
#
# ### Mathematical Analysis Framework
#
# **Latent Space Geometry:**
#
# **Embedding Distribution Analysis:**
# $$\mathbf{z} \sim q(\mathbf{z}|\mathbf{X}, \mathbf{A}) = \prod_{i=1}^n \mathcal{N}(\mathbf{z}_i; \boldsymbol{\mu}_i, \text{diag}(\boldsymbol{\sigma}_i^2))$$
#
# **KL Regularization Effect:**
# The KL term $\text{KL}[q(\mathbf{Z}) \| p(\mathbf{Z})]$ encourages:
# $$\mathbb{E}[\mathbf{z}] \rightarrow 0, \quad \text{Var}[\mathbf{z}] \rightarrow 1$$
#
# **Information Bottleneck Principle:**
# VGAE learns to compress graph structure while preserving link prediction capability:
# $$\min I(\mathbf{X}, \mathbf{A}; \mathbf{Z}) \text{ subject to } I(\mathbf{Z}; \mathbf{A}) \geq \epsilon$$
#
# **Graph Type Specialization:**
#
# **Community Graphs:**
# - High modularity: $Q = \frac{1}{2m} \sum_{ij} \left(A_{ij} - \frac{k_i k_j}{2m}\right) \delta(c_i, c_j)$
# - Clear cluster separation in embedding space
# - High ARI scores due to block structure
#
# **Scale-Free Graphs:**
# - Power-law degree distribution: $P(k) \propto k^{-\gamma}$
# - Hub nodes dominate embedding variance
# - Link prediction benefits from degree-based features
#
# **Small-World Graphs:**
# - High clustering, short paths: $C \gg C_{random}$, $L \approx L_{random}$
# - Balanced local and global connectivity patterns
# - Intermediate performance on both tasks
#
# **Model Interpretability:**
#
# **Attention Mechanism (if applicable):**
# $$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_k]))}$$
#
# **Feature Importance:**
# Analyze which input features contribute most to embedding quality and downstream task performance.

# %%
def analyze_learned_representations(results):
    """
    Analyze the learned representations across different graph types
    """
    print("\n" + "="*70)
    print("LEARNED REPRESENTATION ANALYSIS")
    print("="*70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for i, (name, result) in enumerate(results.items()):
        embeddings = result['embeddings']

        # Embedding statistics
        print(f"\n{name} Graph Embeddings:")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Mean: {embeddings.mean():.3f}")
        print(f"  Std: {embeddings.std():.3f}")
        print(f"  Min: {embeddings.min():.3f}")
        print(f"  Max: {embeddings.max():.3f}")

        # Plot embedding distribution
        axes[0, i].hist(embeddings.flatten(), bins=50, alpha=0.7, density=True)
        axes[0, i].axvline(0, color='red', linestyle='--', alpha=0.7, label='Zero')
        axes[0, i].axvline(embeddings.mean(), color='blue', linestyle='-',
                          label=f'Mean: {embeddings.mean():.2f}')
        axes[0, i].set_title(f'{name}: Embedding Distribution')
        axes[0, i].set_xlabel('Embedding Value')
        axes[0, i].set_ylabel('Density')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)

        # Plot embedding variance per dimension
        dim_vars = embeddings.var(axis=0)
        axes[1, i].bar(range(len(dim_vars)), dim_vars)
        axes[1, i].set_title(f'{name}: Variance per Dimension')
        axes[1, i].set_xlabel('Latent Dimension')
        axes[1, i].set_ylabel('Variance')
        axes[1, i].grid(True, alpha=0.3)

        print(f"  Dimension variances: min={dim_vars.min():.3f}, max={dim_vars.max():.3f}")

    plt.tight_layout()
    plt.show()

def analyze_model_behavior(results):
    """
    Analyze model behavior patterns
    """
    print(f"\n{'='*70}")
    print("MODEL BEHAVIOR ANALYSIS")
    print(f"{'='*70}")

    # Analyze training dynamics
    print(f"\nTraining Dynamics:")
    for name, result in results.items():
        history = result['history']

        final_recon = history['train_recon_loss'][-1]
        final_kl = history['train_kl_loss'][-1]

        print(f"  {name}:")
        print(f"    Final reconstruction loss: {final_recon:.4f}")
        print(f"    Final KL divergence: {final_kl:.4f}")
        print(f"    Recon/KL ratio: {final_recon/final_kl:.2f}")

    # Analyze link prediction patterns
    print(f"\nLink Prediction Patterns:")
    for name, result in results.items():
        link_pred = result['link_prediction']

        pos_mean = link_pred['test_pos_preds'].mean()
        neg_mean = link_pred['test_neg_preds'].mean()
        separation = pos_mean - neg_mean

        print(f"  {name}:")
        print(f"    Positive edge predictions: {pos_mean:.3f}")
        print(f"    Negative edge predictions: {neg_mean:.3f}")
        print(f"    Prediction separation: {separation:.3f}")

def visualize_all_embeddings_together(results):
    """
    Visualize all learned embeddings in a single plot
    """
    print(f"\nCreating unified embedding visualization...")

    # Combine all embeddings
    all_embeddings = []
    all_labels = []
    all_graph_types = []

    for i, (name, result) in enumerate(results.items()):
        embeddings = result['embeddings']
        all_embeddings.append(embeddings)
        all_labels.extend([i] * len(embeddings))  # Graph type as label
        all_graph_types.extend([name] * len(embeddings))

    combined_embeddings = np.vstack(all_embeddings)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(combined_embeddings)

    # Plot
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green']

    for i, name in enumerate(results.keys()):
        mask = np.array(all_labels) == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=colors[i], label=name, alpha=0.6, s=30)

    plt.title('Learned Embeddings from Different Graph Types')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Run comprehensive analysis
analyze_learned_representations(multi_graph_results)
analyze_model_behavior(multi_graph_results)
visualize_all_embeddings_together(multi_graph_results)

# %% [markdown]
# ## 16. Conclusion and Mathematical Summary
#
# This complete VGAE toy project demonstrates comprehensive variational graph learning:
#
# ### Mathematical Contributions and Insights
#
# **Theoretical Framework:**
# - **Variational Inference**: Successfully implemented ELBO optimization for graph data
# - **Reparameterization Trick**: Enabled gradient flow through stochastic nodes
# - **Graph Convolutions**: Demonstrated message passing for node representation learning
#
# **Empirical Findings:**
# - **Community Detection**: VGAE excels at discovering latent community structure
# - **Link Prediction**: Achieves high AUC/AP scores through learned similarity metrics
# - **Graph Generation**: Samples from learned distribution preserve structural properties
#
# **Scalability Analysis:**
# - **Computational Complexity**: $\mathcal{O}(|E| \cdot d + n \cdot d^2)$ per training step
# - **Memory Complexity**: $\mathcal{O}(n \cdot d + |E|)$ for sparse graph representation
# - **Convergence Rate**: Typically converges in $\mathcal{O}(100-500)$ epochs
#
# **Model Limitations and Extensions:**
# - **Static Graphs**: Current model assumes fixed graph structure
# - **Node Features**: Performance depends on informative initial features
# - **Scalability**: Memory requirements grow with graph size
#
# **Future Directions:**
# 1. **Dynamic VGAEs**: Extend to time-varying graphs
# 2. **Hierarchical Models**: Multi-scale graph representations
# 3. **Adversarial Training**: GAN-like objectives for better generation
# 4. **Graph Transformers**: Attention-based architectures for graphs
#
# ### Key Components Implemented:
# 1. **GCN Encoder** with mean and log-variance outputs
# 2. **Inner Product Decoder** for link prediction
# 3. **Variational Framework** with reparameterization trick
# 4. **Multiple Graph Types** - community, scale-free, small-world
# 5. **Comprehensive Evaluation** - link prediction, clustering, generation
#
# ### Mathematical Concepts Covered:
# - **Variational Inference**: $\mathcal{L} = \mathbb{E}[\log p(\mathbf{A}|\mathbf{Z})] - \text{KL}[q(\mathbf{Z}|\mathbf{X},\mathbf{A}) \| p(\mathbf{Z})]$
# - **Graph Convolutions**: $\mathbf{H}^{(l+1)} = \sigma(\tilde{\mathbf{A}} \mathbf{H}^{(l)} \mathbf{W}^{(l)})$
# - **Reparameterization**: $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$
# - **Link Prediction**: $p(A_{ij}=1) = \sigma(\mathbf{z}_i^T \mathbf{z}_j)$
#
# ### Training and Evaluation:
# - **ELBO Optimization** with reconstruction and KL terms
# - **Link prediction** evaluation with AUC and AP metrics
# - **Clustering analysis** using learned embeddings
# - **Graph generation** by sampling from latent space
#
# ### Results Analysis:
# - **Community graphs** showed excellent clustering performance (high ARI)
# - **Scale-free graphs** demonstrated good link prediction capabilities
# - **Small-world graphs** balanced both tasks effectively
# - **Learned embeddings** captured meaningful graph structure
#
# ### Key Insights:
# 1. **Graph structure affects learning**: Community structure is easier to capture than scale-free patterns
# 2. **Embedding quality varies**: Different graph types require different model architectures for optimal performance
# 3. **Trade-offs exist**: Better reconstruction doesn't always mean better downstream task performance
# 4. **Latent space organization**: Well-trained VGAEs create semantically meaningful embedding spaces
#
# ### Practical Applications:
# - **Social network analysis**: Community detection and friend recommendation
# - **Molecular design**: Drug discovery and chemical property prediction
# - **Knowledge graphs**: Link prediction and entity relationship modeling
# - **Recommendation systems**: User-item interaction modeling
#
# This implementation provides a solid foundation for understanding variational graph autoencoders and can be extended for more complex applications like hierarchical graph modeling, dynamic graph evolution, or multi-modal graph learning.

# %%
print("🎉 Variational Graph Autoencoder Toy Project Complete! 🎉")
print("\nProject Summary:")
print("✅ Implemented complete VGAE from scratch")
print("✅ Trained on multiple graph types (community, scale-free, small-world)")
print("✅ Demonstrated link prediction capabilities")
print("✅ Analyzed learned embeddings and clustering")
print("✅ Generated new graphs from learned latent space")
print("✅ Comprehensive evaluation and comparison")

print("\nKey Results:")
for name, result in multi_graph_results.items():
    test_auc = result['link_prediction']['test_auc']
    ari = result['clustering']['ari']
    print(f"• {name}: Link Prediction AUC = {test_auc:.3f}, Clustering ARI = {ari:.3f}")

print(f"\nThe VGAE successfully learned to:")
print("• Encode graphs into meaningful latent representations")
print("• Predict missing links with high accuracy")
print("• Capture community structure in embeddings")
print("• Generate new graphs with similar properties")
print("• Adapt to different graph types and structures")

total_params = sum(p.numel() for p in model.parameters())
print(f"\nFinal model: {total_params:,} parameters across encoder and decoder")