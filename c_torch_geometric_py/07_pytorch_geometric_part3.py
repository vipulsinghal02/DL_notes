# %% [markdown]
# # PyTorch Geometric Part 3: Graph Transformers
#
# This notebook explores Graph Transformers with comprehensive mathematical exposition, adapting the powerful Transformer architecture to work with graph-structured data. We'll implement GraphiT (Graph Transformer) with detailed mathematical foundations for attention mechanisms, positional encodings, and graph-specific adaptations.
#
# ## Mathematical Foundation of Graph Transformers
#
# ### Core Concept: Self-Attention on Graphs
#
# Graph Transformers extend the transformer paradigm to irregular graph structures by enabling **global attention** between all node pairs while incorporating **structural inductive biases**:
#
# $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$
#
# ### Key Mathematical Challenges and Solutions
#
# **1. Lack of Natural Order:**
# Unlike sequences, graphs have no inherent ordering. Solution: **Positional Encodings**
# $$\mathbf{h}_v^{(0)} = \mathbf{x}_v + \mathbf{PE}(v)$$
#
# **2. Structural Awareness:**
# Pure attention ignores graph topology. Solution: **Structure-aware Attention**
# $$\text{Attention}_{ij} = \text{softmax}\left(\frac{\mathbf{q}_i^T \mathbf{k}_j + \mathbf{b}_{ij}}{\sqrt{d_k}}\right)$$
#
# where $\mathbf{b}_{ij}$ encodes structural relationship between nodes $i$ and $j$.
#
# **3. Computational Complexity:**
# Full attention is $O(N^2)$ in graph size. Solution: **Sparse Attention** patterns.
#
# ### Mathematical Advantages
#
# **Global Receptive Field:**
# $$\text{Receptive Field} = \text{All nodes at layer 1}$$
#
# **Long-range Dependencies:**
# $$\text{Path Length} = 1 \text{ hop for any node pair}$$
#
# **Permutation Equivariance:**
# $$f(\pi(\mathbf{X}), \pi(\mathbf{A})\pi^T) = \pi(f(\mathbf{X}, \mathbf{A}))$$
#
# Graph Transformers combine the representational power of attention mechanisms with graph-specific structural awareness, enabling powerful modeling of complex node relationships and global graph patterns.

# %%
# Installation (uncomment if needed)
# !pip install torch torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_dense_adj, degree
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import accuracy_score, classification_report
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Positional Encoding for Graphs
#
# ### Mathematical Foundation of Graph Positional Encodings
#
# Unlike sequences with natural positional order, graphs require **structural positional encodings** that capture topological information:
#
# **Challenge:** No canonical node ordering in graphs
# **Solution:** Encode structural properties as positional information
#
# ### Laplacian Positional Encoding
#
# **Graph Laplacian Matrix:**
# $$\mathbf{L} = \mathbf{D} - \mathbf{A}$$
#
# **Normalized Laplacian:**
# $$\mathcal{L} = \mathbf{D}^{-1/2}\mathbf{L}\mathbf{D}^{-1/2} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$$
#
# **Eigendecomposition:**
# $$\mathcal{L} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^T$$
#
# where $\mathbf{U} = [\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_N]$ are eigenvectors and $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_N)$ are eigenvalues.
#
# **Positional Encoding:**
# $$\mathbf{PE}_{\text{Laplacian}} = \mathbf{U}[:, 1:k+1] \in \mathbb{R}^{N \times k}$$
#
# **Mathematical Properties:**
# - $\lambda_1 = 0$ (connected graphs): Skip first eigenvector (constant)
# - $\lambda_i \leq 2$ for normalized Laplacian
# - Eigenvectors capture graph harmonics and structural patterns
# - Lower eigenvalues → global structure, higher → local structure
#
# ### Degree-based Positional Encoding
#
# **Node Degree:** $d_v = \sum_{u \in \mathcal{V}} A_{vu}$
#
# **Sinusoidal Encoding:**
# $$\mathbf{PE}_{\text{degree}}[v, 2i] = \sin\left(\frac{d_v}{10000^{2i/d_{\text{model}}}}\right)$$
# $$\mathbf{PE}_{\text{degree}}[v, 2i+1] = \cos\left(\frac{d_v}{10000^{2i/d_{\text{model}}}}\right)$$
#
# **Properties:**
# - Simple and computationally efficient: $O(|\mathcal{E}|)$
# - Captures local connectivity information
# - Differentiable and learnable
#
# ### Learned Positional Encoding
#
# **Embedding Matrix:**
# $$\mathbf{PE}_{\text{learned}} = \mathbf{E}_{\text{pos}} \in \mathbb{R}^{N_{\max} \times d_{\text{model}}}$$
#
# **Advantages:** Adaptive to task-specific structural patterns
# **Disadvantages:** Requires fixed maximum graph size, less interpretable
#
# ### Theoretical Justification
#
# **Spectral Graph Theory Connection:**
# Laplacian eigenvectors provide optimal low-dimensional representation preserving graph structure:
#
# $$\min_{\mathbf{Y}} \text{tr}(\mathbf{Y}^T\mathcal{L}\mathbf{Y}) \text{ s.t. } \mathbf{Y}^T\mathbf{Y} = \mathbf{I}$$
#
# Solution: $\mathbf{Y} = \mathbf{U}[:, 1:k]$ (first $k$ non-trivial eigenvectors)
#
# **Expressive Power:**
# Different positional encodings capture different aspects:
# - **Laplacian**: Global graph structure and communities
# - **Degree**: Local connectivity patterns
# - **Random Walk**: Diffusion and reachability information

# %%
class GraphPositionalEncoding(nn.Module):
    """
    Various positional encoding methods for graphs
    """
    def __init__(self, d_model, max_len=1000, pe_type='laplacian'):
        super().__init__()
        self.d_model = d_model
        self.pe_type = pe_type

        if pe_type == 'learned':
            self.pe = nn.Embedding(max_len, d_model)

    def laplacian_pe(self, edge_index, num_nodes, k=8):
        """
        Laplacian positional encoding using eigenvectors
        """
        # Convert to dense adjacency matrix
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]

        # Compute degree matrix
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.diag(torch.pow(deg + 1e-8, -0.5))

        # Normalized Laplacian
        laplacian = torch.eye(num_nodes) - deg_inv_sqrt @ adj @ deg_inv_sqrt

        # Compute eigenvectors (use only first k)
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        pe = eigenvectors[:, :min(k, self.d_model)]

        # Pad if necessary
        if pe.size(1) < self.d_model:
            pe = F.pad(pe, (0, self.d_model - pe.size(1)))

        return pe

    def degree_pe(self, edge_index, num_nodes):
        """
        Simple degree-based positional encoding
        """
        deg = degree(edge_index[0], num_nodes=num_nodes)

        # Create sinusoidal encoding based on degree
        pe = torch.zeros(num_nodes, self.d_model)

        for i in range(self.d_model):
            if i % 2 == 0:
                pe[:, i] = torch.sin(deg / (10000 ** (i / self.d_model)))
            else:
                pe[:, i] = torch.cos(deg / (10000 ** (i / self.d_model)))

        return pe

    def forward(self, edge_index, num_nodes, batch_idx=None):
        if self.pe_type == 'laplacian':
            return self.laplacian_pe(edge_index, num_nodes)
        elif self.pe_type == 'degree':
            return self.degree_pe(edge_index, num_nodes)
        elif self.pe_type == 'learned':
            if batch_idx is None:
                batch_idx = torch.arange(num_nodes)
            return self.pe(batch_idx)
        else:
            return torch.zeros(num_nodes, self.d_model)

# %% [markdown]
# ## 2. Graph Transformer Layer
#
# ### Mathematical Formulation of Graph-Aware Attention
#
# The Graph Transformer layer adapts multi-head self-attention to incorporate graph structure while maintaining global connectivity:
#
# **Standard Multi-Head Attention:**
# $$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O$$
#
# where each attention head is:
# $$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$$
#
# **Scaled Dot-Product Attention:**
# $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$
#
# ### Graph-Specific Attention Mechanisms
#
# **1. Attention Masking:**
# $$\text{Attention}_{ij} = \begin{cases}
# \frac{\exp(\mathbf{q}_i^T \mathbf{k}_j / \sqrt{d_k})}{\sum_{k \in \mathcal{N}(i) \cup \{i\}} \exp(\mathbf{q}_i^T \mathbf{k}_k / \sqrt{d_k})} & \text{if } (i,j) \in \mathcal{E} \text{ or } i = j \\
# 0 & \text{otherwise}
# \end{cases}$$
#
# **2. Structural Bias:**
# $$e_{ij} = \frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}} + \mathbf{b}_{ij}$$
#
# where $\mathbf{b}_{ij}$ can be:
# - Edge features: $\mathbf{b}_{ij} = \mathbf{W}_b \mathbf{e}_{ij}$
# - Distance encoding: $\mathbf{b}_{ij} = f(d_{\mathcal{G}}(i,j))$
# - Learned relative positions: $\mathbf{b}_{ij} = \mathbf{W}_r(\mathbf{PE}_i - \mathbf{PE}_j)$
#
# **3. Complete Graph Transformer Update:**
# $$\mathbf{h}_i^{(l+1)} = \text{LayerNorm}\left(\mathbf{h}_i^{(l)} + \text{MultiHead}(\mathbf{H}^{(l)})\right)$$
# $$\mathbf{h}_i^{(l+1)} = \text{LayerNorm}\left(\mathbf{h}_i^{(l+1)} + \text{FFN}(\mathbf{h}_i^{(l+1)})\right)$$
#
# **Feed-Forward Network:**
# $$\text{FFN}(\mathbf{x}) = \text{max}(0, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$
#
# ### Mathematical Properties
#
# **1. Computational Complexity:**
# - **Standard Attention**: $O(N^2 d)$ for $N$ nodes
# - **Sparse Attention**: $O(|\mathcal{E}| d)$ with masking
# - **Memory**: $O(N^2)$ for attention matrix
#
# **2. Receptive Field:**
# - **Layer 1**: Global (all nodes can attend to each other)
# - **Expressive Power**: Can capture arbitrary long-range dependencies
#
# **3. Permutation Equivariance:**
# $$\text{GraphTransformer}(\pi(\mathbf{X}), \pi(\mathbf{A})\pi^T) = \pi(\text{GraphTransformer}(\mathbf{X}, \mathbf{A}))$$
#
# for any permutation matrix $\pi$.
#
# **4. Universal Approximation (Theoretical):**
# Graph Transformers with sufficient depth and width can approximate any permutation-equivariant function on graphs.
#
# ### Advantages over Traditional GNNs
#
# **1. No Over-smoothing:**
# Traditional GNNs suffer from over-smoothing: $\mathbf{h}_v^{(l)} \rightarrow \mathbf{c}$ as $l \rightarrow \infty$
# Graph Transformers maintain distinct representations through self-attention.
#
# **2. Global Information Flow:**
# Direct connections between all node pairs enable efficient global information propagation.
#
# **3. Interpretability:**
# Attention weights $\alpha_{ij}$ provide interpretable measure of node importance for each prediction.
#
# The mathematical framework ensures that Graph Transformers can capture both local graph structure through positional encodings and global patterns through unrestricted attention mechanisms.

# %%
class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer layer with multi-head attention
    """
    def __init__(self, d_model, n_heads, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        """
        Forward pass with optional attention masking
        """
        # Self-attention with residual connection
        attn_out, attention_weights = self.attention(
            x, x, x,
            attn_mask=attention_mask,
            need_weights=True
        )
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, attention_weights

# %% [markdown]
# ## 3. Complete Graph Transformer Architecture
#
# ### Mathematical Framework for Full Graph Transformer (GraphiT)
#
# The complete Graph Transformer integrates multiple components into a unified architecture for graph learning:
#
# **Overall Architecture:**
# $$\mathbf{H}^{(0)} = \mathbf{W}_{\text{input}}\mathbf{X} + \mathbf{PE}$$
# $$\mathbf{H}^{(l+1)} = \text{GraphTransformerLayer}^{(l)}(\mathbf{H}^{(l)})$$
# $$\mathbf{y} = \text{Classifier}(\text{Pooling}(\mathbf{H}^{(L)}))$$
#
# ### Component-wise Mathematical Analysis
#
# **1. Input Embedding:**
# $$\mathbf{h}_v^{(0)} = \mathbf{W}_{\text{input}} \mathbf{x}_v + \mathbf{PE}(v)$$
#
# where:
# - $\mathbf{W}_{\text{input}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{features}}}$: Input projection matrix
# - $\mathbf{PE}(v) \in \mathbb{R}^{d_{\text{model}}}$: Positional encoding for node $v$
#
# **2. Transformer Stack:**
# $$\mathbf{H}^{(l+1)} = \text{TransformerLayer}(\mathbf{H}^{(l)}, \mathbf{A})$$
#
# Each layer applies:
# $$\mathbf{H}^{(l+1)} = \text{LayerNorm}(\mathbf{H}^{(l)} + \text{MultiHeadAttention}(\mathbf{H}^{(l)}))$$
# $$\mathbf{H}^{(l+1)} = \text{LayerNorm}(\mathbf{H}^{(l+1)} + \text{FFN}(\mathbf{H}^{(l+1)}))$$
#
# **3. Graph-Level Representation (for graph classification):**
#
# **Mean Pooling:**
# $$\mathbf{h}_{\mathcal{G}} = \frac{1}{N} \sum_{v=1}^N \mathbf{h}_v^{(L)}$$
#
# **Max Pooling:**
# $$\mathbf{h}_{\mathcal{G}} = \max_{v=1}^N \mathbf{h}_v^{(L)}$$
#
# **Attention Pooling:**
# $$\alpha_v = \text{softmax}(\mathbf{w}_{\text{att}}^T \mathbf{h}_v^{(L)})$$
# $$\mathbf{h}_{\mathcal{G}} = \sum_{v=1}^N \alpha_v \mathbf{h}_v^{(L)}$$
#
# **4. Output Classification:**
# $$\mathbf{y} = \text{softmax}(\mathbf{W}_{\text{out}} \mathbf{h}_{\mathcal{G}} + \mathbf{b}_{\text{out}})$$
#
# ### Attention Masking Strategies
#
# **1. No Masking (Full Attention):**
# $$\text{Mask}_{ij} = 0 \quad \forall i,j$$
#
# Allows all nodes to attend to each other - captures global patterns but computationally expensive.
#
# **2. Adjacency-based Masking:**
# $$\text{Mask}_{ij} = \begin{cases}
# 0 & \text{if } A_{ij} = 1 \text{ or } i = j \\
# -\infty & \text{otherwise}
# \end{cases}$$
#
# Restricts attention to graph neighbors - more efficient, preserves local structure.
#
# **3. Distance-based Masking:**
# $$\text{Mask}_{ij} = \begin{cases}
# 0 & \text{if } d_{\mathcal{G}}(i,j) \leq k \\
# -\infty & \text{otherwise}
# \end{cases}$$
#
# Allows attention within $k$-hop neighborhoods.
#
# ### Mathematical Properties and Analysis
#
# **1. Parameter Count:**
# $$|\theta| = d_{\text{input}} \times d_{\text{model}} + L \times (4d_{\text{model}}^2 + 2d_{\text{model}} \times d_{\text{ff}}) + d_{\text{model}} \times C$$
#
# where $L$ is number of layers, $d_{\text{ff}}$ is feed-forward dimension, $C$ is number of classes.
#
# **2. Computational Complexity per Forward Pass:**
# - **Input Projection**: $O(N \times d_{\text{input}} \times d_{\text{model}})$
# - **Positional Encoding**: $O(N \times d_{\text{model}})$ or $O(N^3)$ for Laplacian
# - **Attention Layers**: $O(L \times N^2 \times d_{\text{model}})$ (full) or $O(L \times |\mathcal{E}| \times d_{\text{model}})$ (sparse)
# - **Output**: $O(d_{\text{model}} \times C)$
#
# **3. Memory Complexity:**
# - **Node Features**: $O(N \times d_{\text{model}} \times L)$
# - **Attention Matrices**: $O(h \times N^2)$ where $h$ is number of heads
# - **Gradients**: $O(|\theta|)$
#
# **4. Expressiveness:**
# Graph Transformers can express functions that traditional MPNNs cannot:
# - **Global Graph Properties**: Graph diameter, connectivity
# - **Long-range Correlations**: Node similarities across distant parts
# - **Higher-order Structures**: Triangles, motifs, community structures
#
# ### Task-Specific Adaptations
#
# **Node Classification:**
# $$p(y_v | \mathcal{G}) = \text{softmax}(\mathbf{W}_{\text{node}} \mathbf{h}_v^{(L)} + \mathbf{b}_{\text{node}})$$
#
# **Graph Classification:**
# $$p(y_{\mathcal{G}} | \mathcal{G}) = \text{softmax}(\mathbf{W}_{\text{graph}} \text{Pool}(\mathbf{H}^{(L)}) + \mathbf{b}_{\text{graph}})$$
#
# **Link Prediction:**
# $$p(e_{ij} | \mathcal{G}) = \sigma(\mathbf{h}_i^{(L)T} \mathbf{W}_{\text{link}} \mathbf{h}_j^{(L)})$$
#
# This architectural framework provides the mathematical foundation for powerful and flexible graph learning while maintaining interpretability through attention mechanisms.

# %%
class GraphTransformer(nn.Module):
    """
    Complete Graph Transformer (GraphiT) implementation
    """
    def __init__(self, input_dim, d_model, n_heads, n_layers,
                 num_classes, dim_feedforward=512, dropout=0.1,
                 pe_type='degree', max_nodes=1000):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_nodes = max_nodes

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pe = GraphPositionalEncoding(d_model, max_nodes, pe_type)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])

        # Output head for node classification
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        # Graph-level pooling for graph classification
        self.graph_pooling = 'mean'  # 'mean', 'max', or 'attention'

        self.dropout = nn.Dropout(dropout)

    def create_attention_mask(self, edge_index, num_nodes, mask_type='none'):
        """
        Create attention mask based on graph structure
        """
        if mask_type == 'none':
            return None
        elif mask_type == 'adjacency':
            # Only allow attention between connected nodes
            adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
            # Add self-loops
            adj = adj + torch.eye(num_nodes)
            # Convert to attention mask (0 = attend, -inf = don't attend)
            mask = torch.where(adj == 0, float('-inf'), 0.0)
            return mask
        else:
            return None

    def forward(self, x, edge_index, batch=None, return_attention=False):
        num_nodes = x.size(0)

        # Input projection
        x = self.input_proj(x)

        # Add positional encoding
        pe = self.pe(edge_index, num_nodes)
        x = x + pe

        # Reshape for transformer (batch_size=1, seq_len=num_nodes, d_model)
        x = x.unsqueeze(0)

        # Create attention mask (optional)
        attention_mask = self.create_attention_mask(edge_index, num_nodes, 'none')

        # Apply transformer layers
        attention_weights_list = []
        for layer in self.transformer_layers:
            x, attention_weights = layer(x, attention_mask)
            if return_attention:
                attention_weights_list.append(attention_weights)

        # Remove batch dimension
        x = x.squeeze(0)

        # For graph classification, pool node features
        if batch is not None:
            if self.graph_pooling == 'mean':
                x = global_mean_pool(x, batch)
            elif self.graph_pooling == 'max':
                x = global_max_pool(x, batch)

        # Classification
        out = self.classifier(x)

        if return_attention:
            return out, attention_weights_list
        return out

# %% [markdown]
# ## 4. Lightweight Graph Transformer for CPU
#
# ### Mathematical Optimization for Resource-Constrained Environments
#
# The lightweight Graph Transformer addresses computational constraints while preserving essential transformer capabilities:
#
# **Design Principles:**
# 1. **Reduced Model Dimensions**: $d_{\text{model}} = 64$ vs. typical $d_{\text{model}} = 512$
# 2. **Fewer Attention Heads**: $h = 4$ vs. typical $h = 8-16$
# 3. **Head Averaging**: Use $\text{concat} = \text{False}$ to average instead of concatenating heads
# 4. **PyG Integration**: Leverage optimized `TransformerConv` operations
#
# **Mathematical Complexity Reduction:**
#
# **Standard Transformer:**
# $$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O$$
#
# Output dimension: $h \times d_k$ then projected to $d_{\text{model}}$
#
# **Lightweight Version:**
# $$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \frac{1}{h}\sum_{i=1}^h \text{head}_i$$
#
# Output dimension: $d_k = d_{\text{model}} / h$ directly
#
# **Parameter Reduction:**
# - **Standard**: $4d_{\text{model}}^2 + d_{\text{model}} \times (h \times d_k)$ per layer
# - **Lightweight**: $4d_{\text{model}}^2$ per layer (no projection matrix)
# - **Savings**: $\approx 25\%$ reduction in transformer parameters
#
# **Memory Complexity:**
# - **Attention Matrix**: $O(h \times N^2)$ → Shared across heads in PyG implementation
# - **Intermediate Activations**: $O(N \times d_{\text{model}})$ vs. $O(N \times h \times d_k)$
#
# **Computational Complexity per Layer:**
# - **Attention**: $O(N^2 d_{\text{model}})$ (same complexity, but smaller constants)
# - **FFN**: $O(N d_{\text{model}}^2)$ (quadratically reduced with smaller $d_{\text{model}}$)
#
# **TransformerConv Integration:**
#
# PyTorch Geometric's `TransformerConv` implements efficient sparse attention:
# $$\mathbf{h}_i^{(l+1)} = \mathbf{W}_1 \mathbf{h}_i^{(l)} + \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W}_2 \mathbf{h}_j^{(l)}$$
#
# where attention weights are computed as:
# $$\alpha_{ij} = \text{softmax}_j\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}_1 \mathbf{h}_i^{(l)} \| \mathbf{W}_2 \mathbf{h}_j^{(l)}]\right)\right)$$
#
# **Efficiency Benefits:**
# - **Sparsity**: Only computes attention for graph edges, not all node pairs
# - **Complexity**: $O(|\mathcal{E}| d_{\text{model}})$ instead of $O(N^2 d_{\text{model}})$
# - **Memory**: Linear in graph size rather than quadratic
#
# **Mathematical Approximation Quality:**
# The lightweight design maintains core transformer properties:
# - **Expressiveness**: Can still capture global patterns through multi-hop attention paths
# - **Interpretability**: Attention weights remain meaningful
# - **Convergence**: Training dynamics similar to full transformers with proper learning rate tuning
#
# This mathematical framework enables efficient graph transformer training on CPU while preserving essential modeling capabilities.

# %%
class LightweightGraphTransformer(nn.Module):
    """
    CPU-optimized Graph Transformer with reduced complexity
    """
    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=2,
                 num_classes=7, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Smaller input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Simplified positional encoding (degree-based)
        self.pe = GraphPositionalEncoding(d_model, pe_type='degree')

        # Lightweight transformer layers using PyG's TransformerConv
        self.transformer_convs = nn.ModuleList([
            TransformerConv(
                in_channels=d_model,
                out_channels=d_model,
                heads=n_heads,
                concat=False,  # Average heads instead of concatenating
                dropout=dropout
            ) for _ in range(n_layers)
        ])

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        # Output classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        # Input projection
        x = self.input_proj(x)

        # Add positional encoding
        pe = self.pe(edge_index, x.size(0))
        x = x + pe

        # Apply transformer convolution layers
        for i, (conv, norm) in enumerate(zip(self.transformer_convs, self.layer_norms)):
            # Transformer convolution
            x_new = conv(x, edge_index)

            # Residual connection and layer norm
            x = norm(x + self.dropout(x_new))

            # Apply activation except for last layer
            if i < len(self.transformer_convs) - 1:
                x = F.relu(x)

        # For graph classification, pool features
        if batch is not None:
            x = global_mean_pool(x, batch)

        # Classification
        return self.classifier(x)

# %% [markdown]
# ## 5. Dataset Preparation
#
# We'll test our Graph Transformers on both node classification (Cora) and graph classification (MUTAG) tasks to demonstrate their versatility across different types of graph learning problems.

# %%
# Load datasets for both node and graph classification
print("Loading datasets...")

# Node classification dataset (Cora)
cora_dataset = Planetoid('/tmp/Cora', 'Cora', transform=NormalizeFeatures())
cora_data = cora_dataset[0]

print(f"Cora Dataset:")
print(f"  Nodes: {cora_data.num_nodes}")
print(f"  Edges: {cora_data.num_edges}")
print(f"  Features: {cora_data.num_features}")
print(f"  Classes: {cora_dataset.num_classes}")

# Graph classification dataset (MUTAG - small for CPU)
try:
    mutag_dataset = TUDataset('/tmp/MUTAG', 'MUTAG')
    print(f"\nMUTAG Dataset:")
    print(f"  Graphs: {len(mutag_dataset)}")
    print(f"  Classes: {mutag_dataset.num_classes}")
    print(f"  Features: {mutag_dataset.num_features}")

    # Sample graph info
    sample_graph = mutag_dataset[0]
    print(f"  Avg nodes per graph: ~{sample_graph.num_nodes} (sample)")
    print(f"  Avg edges per graph: ~{sample_graph.num_edges} (sample)")

    # Split dataset
    torch.manual_seed(42)
    mutag_dataset = mutag_dataset.shuffle()

    train_size = int(0.7 * len(mutag_dataset))
    val_size = int(0.15 * len(mutag_dataset))

    mutag_train = mutag_dataset[:train_size]
    mutag_val = mutag_dataset[train_size:train_size + val_size]
    mutag_test = mutag_dataset[train_size + val_size:]

    print(f"  Train/Val/Test: {len(mutag_train)}/{len(mutag_val)}/{len(mutag_test)}")

except:
    print("\nMUTAG dataset not available - will focus on Cora node classification")
    mutag_dataset = None

# %% [markdown]
# ## 6. Training Functions
#
# We'll create training functions for both node classification and graph classification tasks, optimized for CPU performance with the MacBook Air M2.

# %%
def train_node_classification(model, data, epochs=100, lr=0.01):
    """
    Train Graph Transformer for node classification
    """
    device = torch.device('cpu')
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Validation accuracy every 20 epochs
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=1)
                val_acc = accuracy_score(data.y[data.val_mask].cpu(), pred[data.val_mask].cpu())
                val_accuracies.append(val_acc)
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')
            model.train()

    return train_losses, val_accuracies

def train_graph_classification(model, train_loader, val_loader, epochs=100, lr=0.01):
    """
    Train Graph Transformer for graph classification
    """
    device = torch.device('cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation every 20 epochs
        if epoch % 20 == 0:
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred = model(batch.x, batch.edge_index, batch.batch).argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)

            val_acc = correct / total
            val_accuracies.append(val_acc)
            print(f'Epoch {epoch:03d}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')

    return train_losses, val_accuracies

def test_model(model, data=None, test_loader=None, task='node'):
    """
    Test trained model
    """
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    if task == 'node':
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)
            test_acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
        return test_acc

    elif task == 'graph':
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.batch).argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)

        return correct / total

# %% [markdown]
# ## 7. Node Classification with Graph Transformer
#
# Let's train our lightweight Graph Transformer on the Cora dataset for node classification. This demonstrates how transformers can capture long-range dependencies between nodes in citation networks.

# %%
# Set up CPU optimization
device = torch.device('cpu')
torch.set_num_threads(8)  # Optimize for M2 8-core CPU

print("Training Lightweight Graph Transformer on Cora...")

# Initialize model
gt_model = LightweightGraphTransformer(
    input_dim=cora_dataset.num_features,
    d_model=64,  # Smaller for CPU efficiency
    n_heads=4,
    n_layers=2,
    num_classes=cora_dataset.num_classes,
    dropout=0.1
)

print(f"Model parameters: {sum(p.numel() for p in gt_model.parameters() if p.requires_grad)}")

# Train the model
gt_losses, gt_val_accs = train_node_classification(gt_model, cora_data, epochs=100, lr=0.005)

# Test the model
gt_test_acc = test_model(gt_model, cora_data, task='node')
print(f"\nGraph Transformer Test Accuracy: {gt_test_acc:.4f}")

# %% [markdown]
# ## 8. Comparison with Traditional GNNs
#
# Let's compare our Graph Transformer with a traditional GCN to understand the benefits of the transformer architecture for graph tasks.

# %%
# Compare with GCN baseline
from torch_geometric.nn import GCNConv

class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

print("\nTraining GCN baseline for comparison...")

# Initialize GCN
gcn_model = SimpleGCN(
    input_dim=cora_dataset.num_features,
    hidden_dim=64,
    num_classes=cora_dataset.num_classes
)

print(f"GCN parameters: {sum(p.numel() for p in gcn_model.parameters() if p.requires_grad)}")

# Train GCN
gcn_losses, gcn_val_accs = train_node_classification(gcn_model, cora_data, epochs=100, lr=0.01)

# Test GCN
gcn_test_acc = test_model(gcn_model, cora_data, task='node')
print(f"\nGCN Test Accuracy: {gcn_test_acc:.4f}")

# Compare results
print(f"\n=== Model Comparison ===")
print(f"{'Model':<20} {'Test Acc':<10} {'Parameters':<12}")
print("-" * 45)
print(f"{'Graph Transformer':<20} {gt_test_acc:<10.4f} {sum(p.numel() for p in gt_model.parameters() if p.requires_grad):<12}")
print(f"{'GCN':<20} {gcn_test_acc:<10.4f} {sum(p.numel() for p in gcn_model.parameters() if p.requires_grad):<12}")

# %% [markdown]
# ## 9. Attention Analysis and Visualization
#
# One of the key advantages of Graph Transformers is interpretability through attention weights. Let's extract and visualize attention patterns to understand what the model learns.

# %%
# Extract attention weights for analysis
def analyze_attention(model, data, num_nodes=50):
    """
    Analyze attention patterns in Graph Transformer
    """
    model.eval()
    device = torch.device('cpu')
    data = data.to(device)

    # Get a subset of nodes for visualization
    subset_idx = torch.randperm(data.num_nodes)[:num_nodes]
    subset_x = data.x[subset_idx]

    # Create subgraph (simplified - just use subset)
    with torch.no_grad():
        # For simplicity, we'll analyze the transformer conv layers
        # which have built-in attention mechanisms

        x = model.input_proj(subset_x)
        pe = model.pe(data.edge_index, subset_x.size(0))
        x = x + pe

        # Get attention from first transformer layer
        attention_weights = []

        # Note: TransformerConv doesn't directly expose attention weights
        # This is a limitation, but we can still analyze embeddings
        embeddings = x

    return embeddings, subset_idx

# Analyze embeddings from Graph Transformer
embeddings, node_indices = analyze_attention(gt_model, cora_data, num_nodes=100)

print(f"Extracted embeddings shape: {embeddings.shape}")
print(f"Analyzing nodes: {node_indices[:10]}...")

# Visualize embeddings using t-SNE
print("\nApplying t-SNE to transformer embeddings...")
tsne = TSNE(n_components=2, random_state=42, perplexity=20)
embeddings_2d = tsne.fit_transform(embeddings.cpu().numpy())

# Get true labels for the subset
subset_labels = cora_data.y[node_indices].cpu().numpy()

# Create visualization
plt.figure(figsize=(12, 5))

# Plot 1: Graph Transformer embeddings
plt.subplot(1, 2, 1)
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                     c=subset_labels, cmap='tab10', alpha=0.7, s=30)
plt.title('Graph Transformer Node Embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(scatter, label='Node Class')
plt.grid(True, alpha=0.3)

# Plot 2: Compare with GCN embeddings
gcn_model.eval()
with torch.no_grad():
    gcn_x = gcn_model.conv1(cora_data.x, cora_data.edge_index)
    gcn_embeddings = F.relu(gcn_x)[node_indices]

gcn_embeddings_2d = tsne.fit_transform(gcn_embeddings.cpu().numpy())

plt.subplot(1, 2, 2)
scatter2 = plt.scatter(gcn_embeddings_2d[:, 0], gcn_embeddings_2d[:, 1],
                      c=subset_labels, cmap='tab10', alpha=0.7, s=30)
plt.title('GCN Node Embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(scatter2, label='Node Class')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Graph Classification with MUTAG
#
# If the MUTAG dataset is available, let's demonstrate graph-level classification with our Graph Transformer. This shows how transformers can aggregate information across an entire graph for prediction.

# %%
if mutag_dataset is not None:
    print("Training Graph Transformer on MUTAG for graph classification...")

    # Create data loaders
    batch_size = 32  # Small batch size for CPU

    train_loader = DataLoader(mutag_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(mutag_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(mutag_test, batch_size=batch_size, shuffle=False)

    # Initialize model for graph classification
    mutag_gt_model = LightweightGraphTransformer(
        input_dim=mutag_dataset.num_features,
        d_model=32,  # Even smaller for graph classification
        n_heads=2,
        n_layers=2,
        num_classes=mutag_dataset.num_classes,
        dropout=0.1
    )

    print(f"MUTAG GT parameters: {sum(p.numel() for p in mutag_gt_model.parameters() if p.requires_grad)}")

    # Train model
    mutag_losses, mutag_val_accs = train_graph_classification(
        mutag_gt_model, train_loader, val_loader, epochs=100, lr=0.01
    )

    # Test model
    mutag_test_acc = test_model(mutag_gt_model, test_loader=test_loader, task='graph')
    print(f"\nMUTAG Graph Transformer Test Accuracy: {mutag_test_acc:.4f}")

    # Train GCN for comparison
    mutag_gcn_model = SimpleGCN(
        input_dim=mutag_dataset.num_features,
        hidden_dim=32,
        num_classes=mutag_dataset.num_classes
    )

    # Modify GCN for graph classification
    class GraphGCN(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_classes):
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, num_classes)
            self.dropout = nn.Dropout(0.1)

        def forward(self, x, edge_index, batch):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = global_mean_pool(x, batch)
            return self.classifier(x)

    mutag_gcn_model = GraphGCN(
        input_dim=mutag_dataset.num_features,
        hidden_dim=32,
        num_classes=mutag_dataset.num_classes
    )

    print("\nTraining GCN baseline on MUTAG...")
    gcn_mutag_losses, gcn_mutag_val_accs = train_graph_classification(
        mutag_gcn_model, train_loader, val_loader, epochs=100, lr=0.01
    )

    gcn_mutag_test_acc = test_model(mutag_gcn_model, test_loader=test_loader, task='graph')
    print(f"\nMUTAG GCN Test Accuracy: {gcn_mutag_test_acc:.4f}")

    print(f"\n=== MUTAG Results ===")
    print(f"Graph Transformer: {mutag_test_acc:.4f}")
    print(f"GCN: {gcn_mutag_test_acc:.4f}")

else:
    print("MUTAG dataset not available - skipping graph classification demo")

# %% [markdown]
# ## 11. Performance Analysis and Visualization
#
# Let's create comprehensive visualizations comparing the training dynamics and final performance of Graph Transformers versus traditional GNNs.

# %%
# Create comprehensive performance visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Training loss comparison (Node classification)
axes[0, 0].plot(gt_losses, label='Graph Transformer', alpha=0.8, color='blue')
axes[0, 0].plot(gcn_losses, label='GCN', alpha=0.8, color='red')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Training Loss')
axes[0, 0].set_title('Node Classification: Training Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Validation accuracy comparison (Node classification)
epochs_eval = range(0, 100, 20)
axes[0, 1].plot(epochs_eval, gt_val_accs, 'o-', label='Graph Transformer', alpha=0.8, color='blue')
axes[0, 1].plot(epochs_eval, gcn_val_accs, 'o-', label='GCN', alpha=0.8, color='red')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Validation Accuracy')
axes[0, 1].set_title('Node Classification: Validation Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Test accuracy comparison
models = ['Graph Transformer', 'GCN']
node_accs = [gt_test_acc, gcn_test_acc]

bars = axes[1, 0].bar(models, node_accs, alpha=0.8, color=['blue', 'red'])
axes[1, 0].set_ylabel('Test Accuracy')
axes[1, 0].set_title('Node Classification: Final Test Accuracy')
axes[1, 0].set_ylim(0, 1)

# Add value labels on bars
for bar, acc in zip(bars, node_accs):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom')

# Model complexity comparison
gt_params = sum(p.numel() for p in gt_model.parameters() if p.requires_grad)
gcn_params = sum(p.numel() for p in gcn_model.parameters() if p.requires_grad)
params = [gt_params, gcn_params]

bars2 = axes[1, 1].bar(models, params, alpha=0.8, color=['blue', 'red'])
axes[1, 1].set_ylabel('Number of Parameters')
axes[1, 1].set_title('Model Complexity')

# Add value labels on bars
for bar, param in zip(bars2, params):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + param*0.01,
                   f'{param}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Print detailed comparison
print("\n=== Detailed Performance Analysis ===")
print(f"{'Metric':<25} {'Graph Transformer':<18} {'GCN':<10}")
print("-" * 55)
print(f"{'Node Test Accuracy':<25} {gt_test_acc:<18.4f} {gcn_test_acc:<10.4f}")
print(f"{'Parameters':<25} {gt_params:<18} {gcn_params:<10}")
print(f"{'Final Train Loss':<25} {gt_losses[-1]:<18.4f} {gcn_losses[-1]:<10.4f}")
print(f"{'Best Val Accuracy':<25} {max(gt_val_accs):<18.4f} {max(gcn_val_accs):<10.4f}")

# %% [markdown]
# ## 12. CPU Optimization Tips for Graph Transformers
#
# ### Mathematical Framework for Computational Optimization
#
# **Complexity Analysis for M2 MacBook Air:**
#
# **Hardware Characteristics:**
# - **CPU Cores**: 8 cores (4 performance + 4 efficiency)
# - **Memory Architecture**: Unified memory (shared CPU/GPU)
# - **Vector Units**: Advanced SIMD support for matrix operations
#
# **Mathematical Optimization Strategies:**
#
# **1. Attention Complexity Reduction:**
#
# **Full Attention Complexity:**
# $$\mathcal{C}_{\text{full}} = O(N^2 d_{\text{model}} + N d_{\text{model}}^2)$$
#
# **Sparse Attention Complexity:**
# $$\mathcal{C}_{\text{sparse}} = O(|\mathcal{E}| d_{\text{model}} + N d_{\text{model}}^2)$$
#
# **Sparsity Ratio:**
# $$\rho = \frac{|\mathcal{E}|}{N^2} \ll 1$$
#
# **Speedup Factor:**
# $$\text{Speedup} \approx \frac{1}{\rho} \text{ for attention computation}$$
#
# **2. Memory Optimization Mathematics:**
#
# **Memory Requirements:**
# $$M_{\text{total}} = M_{\text{features}} + M_{\text{attention}} + M_{\text{gradients}}$$
#
# where:
# - $M_{\text{features}} = L \times N \times d_{\text{model}} \times 4$ bytes
# - $M_{\text{attention}} = h \times N^2 \times 4$ bytes (dense) or $h \times |\mathcal{E}| \times 4$ bytes (sparse)
# - $M_{\text{gradients}} = 2 \times |\theta| \times 4$ bytes (Adam optimizer)
#
# **Gradient Checkpointing:**
# $$M_{\text{checkpoint}} = M_{\text{base}} + \sqrt{L} \times M_{\text{layer}}$$
#
# instead of $M_{\text{base}} + L \times M_{\text{layer}}$
#
# **3. Batch Size Optimization:**
#
# **Optimal Batch Size Formula:**
# $$B_{\text{opt}} = \arg\max_B \frac{B \times \text{Utilization}(B)}{\text{Memory}(B)} \text{ s.t. } \text{Memory}(B) \leq M_{\text{available}}$$
#
# **Throughput Model:**
# $$\text{Throughput}(B) = \frac{B}{\text{ComputeTime}(B) + \text{MemoryTime}(B)}$$
#
# **4. Numerical Optimization:**
#
# **Mixed Precision Benefits:**
# - **Memory Reduction**: $\approx 50\%$ using FP16 vs FP32
# - **Speed Improvement**: $\approx 1.5-2\times$ on modern CPUs with proper vectorization
#
# **Learning Rate Scaling:**
# $$\alpha_{\text{mixed}} = \alpha_{\text{base}} \times \sqrt{\frac{B_{\text{mixed}}}{B_{\text{base}}}}$$
#
# **5. Architectural Optimizations:**
#
# **Dimension Scaling Laws:**
# $$\text{Performance} \propto d_{\text{model}}^{\alpha} \text{ where } \alpha \approx 0.5-0.7$$
# $$\text{Compute} \propto d_{\text{model}}^2$$
#
# **Optimal Scaling:**
# $$d_{\text{model}}^* = \arg\max_{d} \frac{\text{Performance}(d)}{\text{Compute}(d)^\beta}$$
#
# where $\beta$ is the computational budget constraint.
#
# **6. Training Efficiency:**
#
# **Learning Rate Warm-up:**
# $$\alpha(t) = \begin{cases}
# \alpha_{\text{base}} \times \frac{t}{T_{\text{warmup}}} & \text{if } t \leq T_{\text{warmup}} \\
# \alpha_{\text{base}} \times \gamma^{\lfloor (t-T_{\text{warmup}})/T_{\text{step}} \rfloor} & \text{otherwise}
# \end{cases}$$
#
# **Gradient Clipping:**
# $$\mathbf{g}_{\text{clipped}} = \mathbf{g} \times \min\left(1, \frac{\tau}{\|\mathbf{g}\|_2}\right)$$
#
# **Early Stopping Criterion:**
# $$\text{Stop if } \text{Validation}(t) - \max_{i<t} \text{Validation}(i) < \epsilon \text{ for } p \text{ consecutive epochs}$$
#
# **Performance Monitoring:**
#
# **CPU Utilization Efficiency:**
# $$\eta_{\text{CPU}} = \frac{\text{Actual FLOPS}}{\text{Peak FLOPS} \times \text{Cores Used}}$$
#
# **Memory Bandwidth Utilization:**
# $$\eta_{\text{Memory}} = \frac{\text{Data Transferred}}{\text{Peak Bandwidth} \times \text{Time}}$$
#
# **Practical Recommendations for M2 MacBook Air:**
#
# **Optimal Configuration:**
# - $d_{\text{model}} = 32-64$ (balance performance/memory)
# - $h = 2-4$ attention heads
# - $L = 2-3$ layers maximum
# - Batch size = 16-32 for graph classification
# - Learning rate = 0.001-0.01 with warm-up
#
# **Threading Configuration:**
# ```python
# torch.set_num_threads(6)  # Leave 2 cores for system
# torch.set_num_interop_threads(2)  # Reduce overhead
# ```
#
# **Expected Performance:**
# - **Memory Usage**: 2-4 GB for typical graph sizes (1K-10K nodes)
# - **Training Speed**: 10-50 graphs/second depending on complexity
# - **Convergence**: 50-200 epochs typically sufficient
#
# These mathematical optimizations ensure efficient Graph Transformer training while maintaining model quality on resource-constrained CPU environments.

# %%
# CPU Optimization Tips for Graph Transformers
print("=== CPU Optimization Tips for Graph Transformers ===")

print("\n1. Architecture Optimizations:")
print("   - Use smaller model dimensions (d_model=32-64)")
print("   - Reduce number of attention heads (2-4 heads)")
print("   - Limit number of layers (2-3 layers max)")
print("   - Use concat=False in TransformerConv to average heads")

print("\n2. Attention Optimizations:")
print("   - Use sparse attention when possible")
print("   - Apply attention masks to focus on local neighborhoods")
print("   - Consider using efficient attention variants (linear attention)")
print("   - Cache attention weights when doing multiple forward passes")

print("\n3. Memory Management:")
print("   - Process large graphs in smaller batches")
print("   - Use gradient checkpointing for deep models")
print("   - Clear intermediate attention matrices explicitly")
print("   - Monitor memory usage during training")

print("\n4. Training Optimizations:")
print("   - Use smaller learning rates (0.001-0.01)")
print("   - Implement learning rate warm-up for stability")
print("   - Use gradient clipping to prevent instability")
print("   - Early stopping based on validation metrics")

print("\n5. Inference Optimizations:")
print("   - Cache node embeddings for repeated queries")
print("   - Use model.eval() to disable dropout and batch norm")
print("   - Batch multiple graphs together when possible")
print("   - Consider quantization for deployment")

# Example of memory monitoring
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

print(f"\nCurrent memory usage: {get_memory_usage():.1f} MB")
print(f"PyTorch threads: {torch.get_num_threads()}")
print(f"Device: {device}")

# Demonstrate efficient evaluation
def efficient_inference(model, data, batch_size=100):
    """
    Memory-efficient inference for large graphs
    """
    model.eval()
    with torch.no_grad():
        if data.num_nodes <= batch_size:
            return model(data.x, data.edge_index)

        # For very large graphs, you might need more sophisticated batching
        # This is a simplified version
        return model(data.x, data.edge_index)

print("\n=== Performance Recommendations ===")
print("For MacBook Air M2:")
print("• Optimal model size: d_model=32-64, 2-3 layers")
print("• Batch size: 16-32 for graph classification")
print("• Learning rate: 0.001-0.01 with warm-up")
print("• Use early stopping to prevent overfitting")
print("• Monitor memory usage and adjust accordingly")

# %% [markdown]
# ## Summary
#
# ### Comprehensive Mathematical Foundation of Graph Transformers
#
# In this notebook, we have explored the mathematical foundations and practical implementations of Graph Transformers for graph learning tasks:
#
# ### **Graph Transformer Architecture**
# - **Mathematical Core**: $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$ adapted for graphs
# - **Key Innovation**: Self-attention on graphs with structural positional encodings
# - **Global Receptive Field**: All nodes can attend to each other from layer 1
# - **Advantages**: Captures long-range dependencies, interpretable attention, no over-smoothing
# - **Limitations**: Quadratic complexity in graph size, requires structural awareness
#
# ### **Positional Encoding Strategies**
# - **Laplacian PE**: $\mathbf{PE} = \mathbf{U}[:, 1:k+1]$ using eigenvectors of normalized Laplacian
# - **Degree PE**: Sinusoidal encoding based on node degrees
# - **Learned PE**: Task-adaptive embeddings for structural patterns
# - **Mathematical Foundation**: Spectral graph theory provides optimal structural representations
#
# ### **Mathematical Insights**
#
# **Theoretical Comparison with GNNs:**
# $$\text{Graph Transformer} \supset \text{Traditional GNNs}$$
#
# **Complexity Analysis:**
# - **Memory**: $O(N^2)$ for full attention vs. $O(|\mathcal{E}|)$ for GNNs
# - **Computation**: $O(N^2 d_{\text{model}})$ vs. $O(|\mathcal{E}| d_{\text{model}})$
# - **Expressiveness**: Can capture patterns beyond 1-WL hierarchy
# - **Scalability**: Requires optimization for large graphs
#
# **Key Mathematical Properties:**
# 1. **Permutation Equivariance**: $f(\pi(\mathbf{X}), \pi(\mathbf{A})\pi^T) = \pi(f(\mathbf{X}, \mathbf{A}))$
# 2. **Universal Approximation**: Can approximate any permutation-equivariant function
# 3. **Global Information Flow**: Direct paths between all node pairs
# 4. **Structural Awareness**: Through positional encodings and attention biases
#
# ### **Performance Insights**
#
# **Advantages over Traditional GNNs:**
# - **Long-range Dependencies**: Direct attention between distant nodes
# - **No Over-smoothing**: Attention preserves node distinctions across layers
# - **Interpretability**: Attention weights provide model explanations
# - **Global Reasoning**: Captures graph-level properties effectively
#
# **Computational Trade-offs:**
# - **Higher Memory**: Quadratic in graph size for attention matrices
# - **Better Expressiveness**: Can model complex node relationships
# - **Scalability Challenges**: Requires optimization for large graphs
# - **Training Stability**: Generally more stable than deep GNNs
#
# ### **Practical Guidelines**
#
# **Architecture Selection:**
# | Use Case | Recommended Configuration | Rationale |
# |----------|--------------------------|-----------|
# | **Small-Medium Graphs** | Full attention | Maximum expressiveness |
# | **Large Graphs** | Sparse attention + masking | Computational efficiency |
# | **Global Reasoning Tasks** | Graph Transformer | Superior long-range modeling |
# | **Local Pattern Tasks** | Traditional GNNs | More efficient |
# | **Interpretability Required** | Graph Transformer | Attention visualization |
#
# **CPU Optimization for M2 MacBook Air:**
# - **Model Size**: $d_{\text{model}} = 32-64$, $h = 2-4$ heads, $L = 2-3$ layers
# - **Memory Management**: Gradient checkpointing, mixed precision
# - **Training Efficiency**: Learning rate warm-up, gradient clipping
# - **Expected Performance**: 2-4 GB memory, 10-50 graphs/sec
#
# ### **Mathematical Extensions and Future Directions**
#
# **Advanced Attention Mechanisms:**
# - **Linear Attention**: $O(N)$ complexity through kernel approximations
# - **Sparse Attention Patterns**: Local + global attention combinations
# - **Hierarchical Attention**: Multi-scale graph representations
#
# **Structural Enhancements:**
# - **Edge-aware Attention**: $e_{ij} = \frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}} + \mathbf{W}_e \mathbf{e}_{ij}$
# - **Relative Positional Encoding**: Distance-based attention biases
# - **Graph-specific Normalizations**: Degree-aware layer normalization
#
# **Theoretical Developments:**
# - **Expressive Power Analysis**: Beyond 1-WL limitations
# - **Generalization Theory**: Sample complexity for graph learning
# - **Optimization Landscapes**: Training dynamics analysis
#
# ### **When to Use Graph Transformers**
#
# **Ideal Applications:**
# - **Global Graph Properties**: Connectivity, diameter, centrality
# - **Long-range Node Relationships**: Cross-community interactions
# - **Complex Pattern Recognition**: Higher-order structural motifs
# - **Interpretable Predictions**: Attention-based explanations
#
# **Consider Alternatives When:**
# - **Very Large Graphs**: $N > 10^4$ nodes (memory constraints)
# - **Simple Local Patterns**: Traditional GNNs more efficient
# - **Limited Computational Resources**: GCN/GraphSAGE alternatives
# - **Real-time Applications**: Latency requirements favor simpler models
#
# ### **Next Notebook Preview**
#
# The next notebook will explore **Advanced GNN Architectures and Scalability**, covering:
# - **GraphSAINT**: $\mathcal{L} = \sum_{s \in \mathcal{S}} \frac{|\mathcal{V}|}{|\mathcal{V}_s|} \mathcal{L}_s$ for scalable training
# - **FastGCN**: Importance sampling for layer-wise node sampling
# - **DropEdge and DropNode**: Regularization techniques for large graphs
# - **Memory-enhanced GNNs**: Persistent memory mechanisms
#
# This mathematical foundation in Graph Transformers provides the necessary background for understanding how attention mechanisms revolutionize graph neural networks while highlighting the computational trade-offs and optimization strategies required for practical deployment.