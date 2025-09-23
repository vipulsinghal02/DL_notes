# %% [markdown]
# # PyTorch Geometric Part 2: Graph Autoencoders
#
# This notebook explores graph autoencoders for unsupervised learning on graphs with comprehensive mathematical exposition. We'll implement Graph Autoencoder (GAE) and Variational Graph Autoencoder (VGAE) architectures, deriving their theoretical foundations and practical implementations.
#
# ## Mathematical Foundation of Graph Autoencoders
#
# ### Core Concept: Latent Graph Representation Learning
#
# Graph autoencoders learn to encode graph structure $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ and node features $\mathbf{X}$ into low-dimensional latent representations $\mathbf{Z}$, then reconstruct the original graph from these embeddings.
#
# **Fundamental Framework:**
# $$\mathcal{G} \xrightarrow{\text{Encoder}} \mathbf{Z} \xrightarrow{\text{Decoder}} \hat{\mathcal{G}}$$
#
# where:
# - **Input**: Graph $\mathcal{G}$ with adjacency matrix $\mathbf{A} \in \{0,1\}^{N \times N}$ and features $\mathbf{X} \in \mathbb{R}^{N \times d}$
# - **Latent Space**: Node embeddings $\mathbf{Z} \in \mathbb{R}^{N \times k}$ where $k \ll d$
# - **Reconstruction**: Predicted adjacency $\hat{\mathbf{A}} \in [0,1]^{N \times N}$
#
# ### Applications with Mathematical Formulations
#
# **1. Link Prediction:**
# $$P(A_{ij} = 1) = \sigma(\mathbf{z}_i^T \mathbf{z}_j)$$
#
# **2. Node Clustering:**
# $$\mathbf{z}_i \approx \mathbf{z}_j \text{ if nodes } i,j \text{ belong to same community}$$
#
# **3. Graph Generation:**
# $$\mathbf{Z} \sim p(\mathbf{Z}) \rightarrow \hat{\mathbf{A}} = \text{Decode}(\mathbf{Z})$$
#
# **4. Anomaly Detection:**
# $$\text{Anomaly Score} = \|\mathbf{A} - \hat{\mathbf{A}}\|_F^2$$
#
# The mathematical foundation enables principled unsupervised learning on graph-structured data through dimensionality reduction and reconstruction objectives.

# %%
# Installation (uncomment if needed)
# !pip install torch torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures, RandomLinkSplit
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Graph Autoencoder (GAE) Architecture
#
# ### Mathematical Formulation
#
# GAE is a **deterministic autoencoder** that learns node embeddings through reconstruction of the adjacency matrix:
#
# **Encoder Function:**
# $$\mathbf{Z} = \text{GCN}(\mathbf{X}, \mathbf{A}) = f_{\theta}(\mathbf{X}, \mathbf{A})$$
#
# where the encoder $f_{\theta}$ is typically a multi-layer GCN:
# $$\mathbf{H}^{(l+1)} = \sigma\left(\tilde{\mathbf{D}}^{-\frac{1}{2}}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-\frac{1}{2}}\mathbf{H}^{(l)}\mathbf{W}^{(l)}\right)$$
#
# **Decoder Function:**
# $$\hat{A}_{ij} = \sigma(\mathbf{z}_i^T \mathbf{z}_j)$$
#
# This **inner product decoder** assumes that edge probability depends on embedding similarity.
#
# **Complete GAE Pipeline:**
# 1. **Input**: $\mathbf{X} \in \mathbb{R}^{N \times d}$, $\mathbf{A} \in \{0,1\}^{N \times N}$
# 2. **Encoding**: $\mathbf{Z} = \text{GCN}_{\theta}(\mathbf{X}, \mathbf{A}) \in \mathbb{R}^{N \times k}$
# 3. **Decoding**: $\hat{\mathbf{A}} = \sigma(\mathbf{Z}\mathbf{Z}^T) \in [0,1]^{N \times N}$
#
# **Loss Function:**
# $$\mathcal{L}_{\text{GAE}} = -\mathbb{E}_{(i,j) \sim \mathcal{E}^+}[\log \hat{A}_{ij}] - \mathbb{E}_{(i,j) \sim \mathcal{E}^-}[\log(1 - \hat{A}_{ij})]$$
#
# where:
# - $\mathcal{E}^+$: Set of positive edges (existing edges)
# - $\mathcal{E}^-$: Set of negative edges (non-existing edges, sampled)
# - $\sigma(\cdot)$: Sigmoid activation function
#
# **Key Properties:**
# - **Deterministic**: Same input always produces same embedding
# - **Efficient**: Single forward pass for encoding
# - **Scalable**: Linear complexity in number of edges
# - **Interpretable**: Inner product captures node similarity
#
# The reconstruction objective forces the model to learn meaningful node representations that preserve graph structure.

# %%
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

# %% [markdown]
# ## 2. Variational Graph Autoencoder (VGAE) Architecture
#
# ### Mathematical Formulation
#
# VGAE extends GAE by introducing a **probabilistic latent space** with Bayesian inference:
#
# **Probabilistic Encoder:**
# $$q_{\phi}(\mathbf{Z}|\mathbf{X}, \mathbf{A}) = \prod_{i=1}^N q_{\phi}(\mathbf{z}_i|\mathbf{X}, \mathbf{A})$$
#
# where each node embedding follows a multivariate Gaussian:
# $$q_{\phi}(\mathbf{z}_i|\mathbf{X}, \mathbf{A}) = \mathcal{N}(\boldsymbol{\mu}_i, \text{diag}(\boldsymbol{\sigma}_i^2))$$
#
# **Encoder Network Architecture:**
# $$\boldsymbol{\mu} = \text{GCN}_{\mu}(\mathbf{X}, \mathbf{A}) = \mathbf{H}^{(L)}\mathbf{W}_{\mu}$$
# $$\log \boldsymbol{\sigma} = \text{GCN}_{\sigma}(\mathbf{X}, \mathbf{A}) = \mathbf{H}^{(L)}\mathbf{W}_{\sigma}$$
#
# where $\mathbf{H}^{(L)}$ is the shared hidden representation from GCN layers.
#
# **Reparameterization Trick:**
# $$\mathbf{z}_i = \boldsymbol{\mu}_i + \boldsymbol{\sigma}_i \odot \boldsymbol{\epsilon}_i$$
#
# where $\boldsymbol{\epsilon}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ enables gradient-based learning.
#
# **Probabilistic Decoder:**
# $$p_{\theta}(\mathbf{A}|\mathbf{Z}) = \prod_{i=1}^N \prod_{j=1}^N p_{\theta}(A_{ij}|\mathbf{z}_i, \mathbf{z}_j)$$
#
# with Bernoulli likelihood:
# $$p_{\theta}(A_{ij} = 1|\mathbf{z}_i, \mathbf{z}_j) = \sigma(\mathbf{z}_i^T \mathbf{z}_j)$$
#
# **Variational Lower Bound (ELBO):**
# $$\mathcal{L}_{\text{VGAE}} = \mathbb{E}_{q_{\phi}(\mathbf{Z}|\mathbf{X},\mathbf{A})}[\log p_{\theta}(\mathbf{A}|\mathbf{Z})] - D_{KL}[q_{\phi}(\mathbf{Z}|\mathbf{X},\mathbf{A}) \| p(\mathbf{Z})]$$
#
# **KL Divergence Term:**
# Assuming standard Gaussian prior $p(\mathbf{z}_i) = \mathcal{N}(\mathbf{0}, \mathbf{I})$:
# $$D_{KL}[q_{\phi}(\mathbf{z}_i) \| p(\mathbf{z}_i)] = \frac{1}{2}\sum_{j=1}^k \left(1 + \log(\sigma_{ij}^2) - \mu_{ij}^2 - \sigma_{ij}^2\right)$$
#
# **Total Loss:**
# $$\mathcal{L}_{\text{total}} = -\mathcal{L}_{\text{VGAE}} = \underbrace{-\mathbb{E}[\log p_{\theta}(\mathbf{A}|\mathbf{Z})]}_{\text{Reconstruction Loss}} + \underbrace{D_{KL}[q_{\phi}(\mathbf{Z}|\mathbf{X},\mathbf{A}) \| p(\mathbf{Z})]}_{\text{Regularization}}$$
#
# **Key Advantages:**
# - **Uncertainty Quantification**: Variance parameters capture embedding uncertainty
# - **Regularized Latent Space**: KL term prevents overfitting and encourages smooth embeddings
# - **Generative Capability**: Can sample new embeddings from learned distribution
# - **Better Representation**: Probabilistic framework often leads to more robust embeddings

# %%
class VariationalGCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv_mu = GCNConv(hidden_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(hidden_channels, out_channels, cached=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# %% [markdown]
# ## 3. Dataset Preparation and Link Prediction Setup
#
# ### Mathematical Framework for Link Prediction
#
# Link prediction is formulated as a **binary classification problem** on node pairs:
#
# **Problem Definition:**
# Given partial graph $\mathcal{G}_{\text{obs}} = (\mathcal{V}, \mathcal{E}_{\text{obs}})$ where $\mathcal{E}_{\text{obs}} \subset \mathcal{E}_{\text{true}}$, predict missing edges in $\mathcal{E}_{\text{missing}} = \mathcal{E}_{\text{true}} \setminus \mathcal{E}_{\text{obs}}$.
#
# **Data Splitting Strategy:**
# $$\mathcal{E}_{\text{true}} = \mathcal{E}_{\text{train}} \cup \mathcal{E}_{\text{val}} \cup \mathcal{E}_{\text{test}}$$
#
# where:
# - $\mathcal{E}_{\text{train}}$: Edges available during training (typically 70-80%)
# - $\mathcal{E}_{\text{val}}$: Edges for hyperparameter tuning (10-15%)
# - $\mathcal{E}_{\text{test}}$: Edges for final evaluation (10-20%)
#
# **Negative Sampling:**
# For each positive edge $(i,j) \in \mathcal{E}^+$, sample negative edge $(i,k) \notin \mathcal{E}$ such that:
# $$|\mathcal{E}^-| = |\mathcal{E}^+|$$
#
# Ensuring balanced classification problem.
#
# **Edge Prediction Probability:**
# $$P(\text{edge}_{ij} = 1) = \sigma(\mathbf{z}_i^T \mathbf{z}_j)$$
#
# **Decision Rule:**
# $$\hat{A}_{ij} = \begin{cases}
# 1 & \text{if } P(\text{edge}_{ij} = 1) > \tau \\
# 0 & \text{otherwise}
# \end{cases}$$
#
# where $\tau$ is the decision threshold (typically 0.5).
#
# **Evaluation Metrics:**
#
# **1. Area Under Curve (AUC):**
# $$\text{AUC} = \int_0^1 \text{TPR}(t) \, d\text{FPR}(t)$$
#
# where TPR = True Positive Rate, FPR = False Positive Rate.
#
# **2. Average Precision (AP):**
# $$\text{AP} = \sum_{k} \left(\text{Recall}(k) - \text{Recall}(k-1)\right) \times \text{Precision}(k)$$
#
# **Mathematical Properties:**
# - **AUC**: Measures ranking quality, range $[0,1]$, higher is better
# - **AP**: Emphasizes precision at high recall, suitable for imbalanced data
# - **Both metrics**: Threshold-independent, robust to class imbalance
#
# This framework enables rigorous evaluation of graph autoencoder performance on link prediction tasks.

# %%
# Load and prepare data for link prediction
dataset = Planetoid('/tmp/Cora', 'Cora', transform=NormalizeFeatures())
data = dataset[0]

# Remove self loops and add them back (standardization)
data.edge_index, _ = remove_self_loops(data.edge_index)
data.edge_index, _ = add_self_loops(data.edge_index)

print(f'Dataset: {dataset}')
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of features: {data.num_features}')
print(f'Number of classes: {dataset.num_classes}')

# Split edges for link prediction
transform = RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=True, split_labels=True)
train_data, val_data, test_data = transform(data)

print(f'\nAfter splitting:')
print(f'Training edges: {train_data.edge_index.size(1)}')
print(f'Validation edges: {val_data.edge_label_index.size(1)}')
print(f'Test edges: {test_data.edge_label_index.size(1)}')

# %% [markdown]
# ## 4. Training Functions
#
# ### Mathematical Framework for Training Graph Autoencoders
#
# **GAE Training Objective:**
# $$\mathcal{L}_{\text{GAE}} = -\sum_{(i,j) \in \mathcal{E}^+} \log \sigma(\mathbf{z}_i^T \mathbf{z}_j) - \sum_{(i,j) \in \mathcal{E}^-} \log(1 - \sigma(\mathbf{z}_i^T \mathbf{z}_j))$$
#
# This is the **binary cross-entropy loss** for link prediction.
#
# **VGAE Training Objective:**
# $$\mathcal{L}_{\text{VGAE}} = \mathcal{L}_{\text{reconstruction}} + \beta \cdot \mathcal{L}_{\text{KL}}$$
#
# where:
#
# **Reconstruction Loss:**
# $$\mathcal{L}_{\text{reconstruction}} = -\mathbb{E}_{q_{\phi}(\mathbf{Z}|\mathbf{X},\mathbf{A})}[\log p_{\theta}(\mathbf{A}|\mathbf{Z})]$$
#
# **KL Divergence Loss:**
# $$\mathcal{L}_{\text{KL}} = D_{KL}[q_{\phi}(\mathbf{Z}|\mathbf{X},\mathbf{A}) \| p(\mathbf{Z})]$$
#
# **KL Divergence Computation:**
# For each node $i$:
# $$D_{KL}[q_{\phi}(\mathbf{z}_i) \| p(\mathbf{z}_i)] = \frac{1}{2}\sum_{j=1}^k \left(\mu_{ij}^2 + \sigma_{ij}^2 - \log(\sigma_{ij}^2) - 1\right)$$
#
# **Total KL Loss:**
# $$\mathcal{L}_{\text{KL}} = \frac{1}{N} \sum_{i=1}^N D_{KL}[q_{\phi}(\mathbf{z}_i) \| p(\mathbf{z}_i)]$$
#
# **Gradient Computation:**
#
# **For GAE:** Standard backpropagation through GCN layers and inner product decoder.
#
# **For VGAE:** Uses reparameterization trick for gradients w.r.t. $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$:
# $$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}_i} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_i} + \frac{\partial \mathcal{L}_{\text{KL}}}{\partial \boldsymbol{\mu}_i}$$
# $$\frac{\partial \mathcal{L}}{\partial \log \boldsymbol{\sigma}_i} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_i} \odot \boldsymbol{\sigma}_i + \frac{\partial \mathcal{L}_{\text{KL}}}{\partial \log \boldsymbol{\sigma}_i}$$
#
# **Negative Sampling Strategy:**
# Sample negative edges uniformly from non-existing edges:
# $$\mathcal{E}^- = \{(i,j) : (i,j) \notin \mathcal{E}, \, i \neq j\}$$
#
# **Optimization Algorithm:**
# Both models use Adam optimizer with learning rate $\alpha$:
# $$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$
#
# **Convergence Criteria:**
# - Monitor validation AUC/AP scores
# - Early stopping when validation performance plateaus
# - Typical convergence in 100-200 epochs
#
# The mathematical framework ensures principled optimization of both deterministic and probabilistic graph autoencoders.

# %%
def train_gae(model, train_data, optimizer, device):
    """
    Train Graph Autoencoder for one epoch
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass
    z = model.encode(train_data.x, train_data.edge_index)

    # Compute loss (reconstruction + negative sampling)
    loss = model.recon_loss(z, train_data.edge_index)

    # Add negative sampling loss
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_index.size(1)
    )

    loss = loss + model.recon_loss(z, neg_edge_index, neg_edge_index=True)

    loss.backward()
    optimizer.step()

    return loss.item()

def train_vgae(model, train_data, optimizer, device):
    """
    Train Variational Graph Autoencoder for one epoch
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass
    z = model.encode(train_data.x, train_data.edge_index)

    # Compute reconstruction loss
    recon_loss = model.recon_loss(z, train_data.edge_index)

    # Add KL divergence loss
    kl_loss = model.kl_loss() / train_data.num_nodes

    # Total loss
    loss = recon_loss + kl_loss

    loss.backward()
    optimizer.step()

    return loss.item(), recon_loss.item(), kl_loss.item()

# %% [markdown]
# ## 5. Evaluation Functions
#
# ### Mathematical Framework for Link Prediction Evaluation
#
# **Performance Metrics for Binary Classification:**
#
# **1. Area Under Receiver Operating Characteristic Curve (AUC-ROC):**
#
# The ROC curve plots True Positive Rate vs. False Positive Rate:
# $$\text{TPR}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau) + \text{FN}(\tau)}$$
# $$\text{FPR}(\tau) = \frac{\text{FP}(\tau)}{\text{FP}(\tau) + \text{TN}(\tau)}$$
#
# where $\tau$ is the decision threshold.
#
# **AUC Computation:**
# $$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(x)) \, dx = P(\text{score}_{\text{pos}} > \text{score}_{\text{neg}})$$
#
# **Probabilistic Interpretation:**
# AUC equals the probability that a randomly chosen positive edge has higher score than a randomly chosen negative edge.
#
# **2. Average Precision (AP):**
#
# AP summarizes the precision-recall curve:
# $$\text{Precision}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau) + \text{FP}(\tau)}$$
# $$\text{Recall}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau) + \text{FN}(\tau)}$$
#
# **AP Computation:**
# $$\text{AP} = \sum_{k=1}^n \left[\text{Recall}(k) - \text{Recall}(k-1)\right] \times \text{Precision}(k)$$
#
# where $k$ indexes sorted predictions by decreasing score.
#
# **Evaluation Procedure:**
#
# **Step 1: Score Computation**
# For each edge candidate $(i,j)$:
# $$s_{ij} = \sigma(\mathbf{z}_i^T \mathbf{z}_j)$$
#
# **Step 2: Label Assignment**
# $$y_{ij} = \begin{cases}
# 1 & \text{if } (i,j) \in \mathcal{E}_{\text{true}} \\
# 0 & \text{if } (i,j) \in \mathcal{E}_{\text{negative}}
# \end{cases}$$
#
# **Step 3: Ranking and Evaluation**
# - Sort all edge candidates by score $s_{ij}$ in descending order
# - Compute TPR, FPR at each threshold
# - Calculate AUC and AP from these curves
#
# **Statistical Properties:**
# - **AUC Range**: $[0, 1]$, random classifier achieves 0.5
# - **AP Range**: $[0, 1]$, depends on positive class ratio
# - **Robustness**: Both metrics are threshold-independent
# - **Interpretation**: Higher values indicate better link prediction performance
#
# **Cross-Validation for Robust Evaluation:**
# $$\text{Score}_{\text{CV}} = \frac{1}{K} \sum_{k=1}^K \text{Score}_k$$
#
# where $K$ is the number of folds in cross-validation.
#
# This mathematical framework provides rigorous evaluation of graph autoencoder performance on link prediction tasks.

# %%
def evaluate_model(model, data, pos_edge_index, neg_edge_index):
    """
    Evaluate model on link prediction task
    """
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)

        # Get positive and negative edge scores
        pos_scores = model.decode(z, pos_edge_index).sigmoid()
        neg_scores = model.decode(z, neg_edge_index).sigmoid()

        # Combine scores and labels
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones(pos_scores.size(0)),
            torch.zeros(neg_scores.size(0))
        ])

        # Calculate metrics
        auc = roc_auc_score(labels.cpu(), scores.cpu())
        ap = average_precision_score(labels.cpu(), scores.cpu())

    return auc, ap

def generate_negative_edges(edge_index, num_nodes, num_neg_samples):
    """
    Generate negative edges for evaluation
    """
    return negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples
    )

# %% [markdown]
# ## 6. Training GAE Model
#
# ### Mathematical Analysis of GAE Training
#
# **Optimization Landscape:**
# The GAE loss function is non-convex due to the neural network encoder:
# $$\mathcal{L}_{\text{GAE}}(\boldsymbol{\theta}) = -\sum_{(i,j) \in \mathcal{E}^+} \log \sigma(\mathbf{z}_i^T(\boldsymbol{\theta}) \mathbf{z}_j(\boldsymbol{\theta})) - \sum_{(i,j) \in \mathcal{E}^-} \log(1 - \sigma(\mathbf{z}_i^T(\boldsymbol{\theta}) \mathbf{z}_j(\boldsymbol{\theta})))$$
#
# **Gradient Flow Analysis:**
# $$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}} = \sum_{(i,j)} \left(\hat{A}_{ij} - A_{ij}\right) \frac{\partial}{\partial \boldsymbol{\theta}}[\mathbf{z}_i^T \mathbf{z}_j]$$
#
# where $\hat{A}_{ij} = \sigma(\mathbf{z}_i^T \mathbf{z}_j)$ is the predicted edge probability.
#
# **Training Dynamics:**
# 1. **Early Stage**: Model learns to distinguish connected vs. disconnected node pairs
# 2. **Mid Stage**: Embeddings become more structured, capturing community information
# 3. **Late Stage**: Fine-tuning of embedding similarities for edge prediction
#
# **Convergence Properties:**
# - **Local Minima**: Multiple local optima exist due to permutation symmetry
# - **Saddle Points**: Abundant in high-dimensional parameter space
# - **Convergence Rate**: Typically $O(1/\sqrt{t})$ for Adam optimizer
#
# **Regularization Effects:**
# - **Dropout**: Prevents overfitting to specific edge patterns
# - **Weight Decay**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GAE}} + \lambda \|\boldsymbol{\theta}\|_2^2$
# - **Early Stopping**: Prevents overfitting to training edges
#
# **Model Architecture Impact:**
# - **Embedding Dimension $k$**: Larger $k$ increases capacity but may overfit
# - **Encoder Depth**: Deeper networks capture multi-hop relationships
# - **Hidden Dimensions**: Affect representational power and computational cost
#
# **CPU-Optimized Configuration:**
# - Embedding dimension: $k = 16$ (reduced from typical 32-64)
# - Hidden dimension: $d_h = 32$ (balanced performance/memory)
# - Dropout rate: $p = 0.1$ (light regularization)
# - Learning rate: $\alpha = 0.01$ (stable convergence)
#
# The mathematical framework ensures efficient training while maintaining model expressiveness for link prediction tasks.

# %%
# Initialize GAE model
device = torch.device('cpu')  # Using CPU for MacBook Air M2
torch.set_num_threads(8)  # Optimize for M2 8-core CPU

# Create encoder and GAE model
encoder = GCNEncoder(dataset.num_features, 32, 16)  # Smaller dimensions for CPU
gae_model = GAE(encoder).to(device)

# Move data to device
train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

# Training setup
optimizer_gae = torch.optim.Adam(gae_model.parameters(), lr=0.01)
epochs = 100

# Training loop
gae_losses = []
gae_val_aucs = []
gae_val_aps = []

print("Training GAE...")
for epoch in range(epochs):
    loss = train_gae(gae_model, train_data, optimizer_gae, device)
    gae_losses.append(loss)

    # Evaluate every 10 epochs
    if epoch % 20 == 0:
        # Generate negative edges for validation
        neg_edge_index = generate_negative_edges(
            val_data.edge_label_index[:, val_data.edge_label == 1],
            train_data.num_nodes,
            val_data.edge_label_index[:, val_data.edge_label == 1].size(1)
        )

        auc, ap = evaluate_model(
            gae_model, train_data,
            val_data.edge_label_index[:, val_data.edge_label == 1],
            neg_edge_index
        )

        gae_val_aucs.append(auc)
        gae_val_aps.append(ap)

        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val AUC: {auc:.4f}, Val AP: {ap:.4f}')

print("GAE training completed!")

# %% [markdown]
# ## 7. Training VGAE Model
#
# ### Mathematical Analysis of VGAE Training
#
# **Variational Inference Framework:**
# VGAE optimizes the Evidence Lower BOund (ELBO):
# $$\log p(\mathbf{A}|\mathbf{X}) \geq \mathbb{E}_{q_{\phi}(\mathbf{Z}|\mathbf{X},\mathbf{A})}[\log p_{\theta}(\mathbf{A}|\mathbf{Z})] - D_{KL}[q_{\phi}(\mathbf{Z}|\mathbf{X},\mathbf{A}) \| p(\mathbf{Z})]$$
#
# **Loss Function Decomposition:**
# $$\mathcal{L}_{\text{VGAE}} = \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{KL}}$$
#
# **Reconstruction Loss Gradients:**
# $$\frac{\partial \mathcal{L}_{\text{recon}}}{\partial \boldsymbol{\mu}_i} = \mathbb{E}_{\boldsymbol{\epsilon}}[\frac{\partial}{\partial \mathbf{z}_i} \log p(\mathbf{A}|\mathbf{Z})] \bigg|_{\mathbf{z}_i = \boldsymbol{\mu}_i + \boldsymbol{\sigma}_i \odot \boldsymbol{\epsilon}}$$
#
# $$\frac{\partial \mathcal{L}_{\text{recon}}}{\partial \log \boldsymbol{\sigma}_i} = \mathbb{E}_{\boldsymbol{\epsilon}}[\boldsymbol{\epsilon} \odot \frac{\partial}{\partial \mathbf{z}_i} \log p(\mathbf{A}|\mathbf{Z})] \bigg|_{\mathbf{z}_i = \boldsymbol{\mu}_i + \boldsymbol{\sigma}_i \odot \boldsymbol{\epsilon}}$$
#
# **KL Divergence Gradients:**
# $$\frac{\partial \mathcal{L}_{\text{KL}}}{\partial \boldsymbol{\mu}_i} = \boldsymbol{\mu}_i$$
#
# $$\frac{\partial \mathcal{L}_{\text{KL}}}{\partial \log \boldsymbol{\sigma}_i} = 1 - \boldsymbol{\sigma}_i^2$$
#
# **Training Dynamics Analysis:**
#
# **Phase 1: Reconstruction Focus** (Early epochs)
# - KL loss is small, reconstruction loss dominates
# - Model learns basic graph structure patterns
# - Embeddings may have high variance
#
# **Phase 2: Regularization Balance** (Mid epochs)
# - KL loss increases, providing regularization
# - Embeddings become more structured and smooth
# - Trade-off between reconstruction and regularization
#
# **Phase 3: Fine-tuning** (Late epochs)
# - Both losses stabilize
# - Embeddings capture detailed graph structure
# - Model achieves optimal reconstruction-regularization balance
#
# **Hyperparameter Effects:**
#
# **KL Weight $\beta$:**
# - $\beta = 0$: Reduces to standard autoencoder (no regularization)
# - $\beta = 1$: Standard VAE formulation
# - $\beta > 1$: Strong regularization, may underfit
# - $\beta < 1$: Weak regularization, may overfit
#
# **Embedding Dimension Trade-off:**
# $$\text{Model Capacity} \propto k \times \log k$$
# $$\text{KL Regularization} \propto k$$
#
# **Optimization Challenges:**
# 1. **Posterior Collapse**: $q_{\phi}(\mathbf{Z}) \approx p(\mathbf{Z})$, loss of information
# 2. **KL Vanishing**: Early training may ignore KL term
# 3. **Local Optima**: Multiple solutions due to symmetries
#
# **Mitigation Strategies:**
# - **KL Annealing**: Gradually increase $\beta$ during training
# - **Free Bits**: Prevent KL from dropping below threshold
# - **Architectural Constraints**: Limit encoder capacity
#
# The probabilistic framework provides more robust embeddings through Bayesian regularization while maintaining link prediction performance.

# %%
# Initialize VGAE model
vgae_encoder = VariationalGCNEncoder(dataset.num_features, 32, 16)
vgae_model = VGAE(vgae_encoder).to(device)

# Training setup
optimizer_vgae = torch.optim.Adam(vgae_model.parameters(), lr=0.01)

# Training loop
vgae_losses = []
vgae_recon_losses = []
vgae_kl_losses = []
vgae_val_aucs = []
vgae_val_aps = []

print("\nTraining VGAE...")
for epoch in range(epochs):
    loss, recon_loss, kl_loss = train_vgae(vgae_model, train_data, optimizer_vgae, device)
    vgae_losses.append(loss)
    vgae_recon_losses.append(recon_loss)
    vgae_kl_losses.append(kl_loss)

    # Evaluate every 10 epochs
    if epoch % 20 == 0:
        # Generate negative edges for validation
        neg_edge_index = generate_negative_edges(
            val_data.edge_label_index[:, val_data.edge_label == 1],
            train_data.num_nodes,
            val_data.edge_label_index[:, val_data.edge_label == 1].size(1)
        )

        auc, ap = evaluate_model(
            vgae_model, train_data,
            val_data.edge_label_index[:, val_data.edge_label == 1],
            neg_edge_index
        )

        vgae_val_aucs.append(auc)
        vgae_val_aps.append(ap)

        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Recon: {recon_loss:.4f}, '
              f'KL: {kl_loss:.4f}, Val AUC: {auc:.4f}, Val AP: {ap:.4f}')

print("VGAE training completed!")

# %% [markdown]
# ## 8. Model Comparison and Visualization
#
# ### Mathematical Analysis of Training Dynamics
#
# **Learning Curve Analysis:**
#
# **GAE Loss Evolution:**
# $$\mathcal{L}_{\text{GAE}}(t) = \mathcal{L}_{\text{GAE}}(\infty) + C_1 e^{-\alpha_1 t}$$
#
# where $t$ is training epoch, exhibiting exponential convergence to optimal loss.
#
# **VGAE Loss Components:**
# $$\mathcal{L}_{\text{VGAE}}(t) = \mathcal{L}_{\text{recon}}(t) + \beta \mathcal{L}_{\text{KL}}(t)$$
#
# **Reconstruction Loss**: Similar exponential decay as GAE
# **KL Loss**: Often increases initially, then stabilizes
# $$\mathcal{L}_{\text{KL}}(t) = \mathcal{L}_{\text{KL}}(\infty)(1 - e^{-\alpha_2 t})$$
#
# **Validation Performance Analysis:**
#
# **AUC Convergence:**
# $$\text{AUC}(t) = \text{AUC}_{\max} - (\text{AUC}_{\max} - \text{AUC}_0) e^{-\beta t}$$
#
# **Overfitting Detection:**
# Monitor validation vs. training performance gap:
# $$\text{Generalization Gap} = \text{AUC}_{\text{train}}(t) - \text{AUC}_{\text{val}}(t)$$
#
# **Model Complexity Comparison:**
#
# **Parameter Count Analysis:**
# - **GAE**: $|\theta_{\text{GAE}}| = (d \times h + h \times k) \times L$ where $L$ is number of layers
# - **VGAE**: $|\theta_{\text{VGAE}}| = |\theta_{\text{GAE}}| + h \times k$ (additional variance parameters)
#
# **Computational Complexity:**
# - **Forward Pass**: $O(|\mathcal{E}| \times h + N \times h \times k)$ for both models
# - **Backward Pass**: Additional KL gradient computation for VGAE: $O(N \times k)$
#
# **Statistical Analysis:**
#
# **Performance Comparison Framework:**
# Use paired t-test to compare AUC scores:
# $$t = \frac{\bar{d}}{\frac{s_d}{\sqrt{n}}}$$
#
# where $\bar{d}$ is mean difference, $s_d$ is standard deviation of differences.
#
# **Effect Size (Cohen's d):**
# $$d = \frac{\mu_{\text{VGAE}} - \mu_{\text{GAE}}}{\sigma_{\text{pooled}}}$$
#
# **Learning Rate Sensitivity:**
# Both models show robustness to learning rate in range $[0.001, 0.01]$:
# $$\text{Sensitivity} = \frac{\partial \text{AUC}}{\partial \alpha} \bigg|_{\alpha = \alpha_0}$$
#
# **Regularization Effects:**
# - **GAE**: Relies primarily on dropout and weight decay
# - **VGAE**: Inherent regularization through KL divergence provides better generalization
#
# **Convergence Diagnostics:**
# - **Loss Plateauing**: Indicates convergence or local minimum
# - **Validation Performance**: Monitor for early stopping
# - **Gradient Norms**: Track to detect vanishing/exploding gradients
#
# The mathematical analysis reveals trade-offs between model complexity, training stability, and final performance.

# %%
# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Training losses
axes[0, 0].plot(gae_losses, label='GAE', alpha=0.8, color='blue')
axes[0, 0].plot(vgae_losses, label='VGAE Total', alpha=0.8, color='red')
axes[0, 0].plot(vgae_recon_losses, label='VGAE Recon', alpha=0.8, color='orange')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# KL divergence for VGAE
axes[0, 1].plot(vgae_kl_losses, alpha=0.8, color='purple')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('KL Divergence Loss')
axes[0, 1].set_title('VGAE KL Divergence')
axes[0, 1].grid(True, alpha=0.3)

# Validation AUC
epochs_eval = range(0, epochs, 20)
axes[1, 0].plot(epochs_eval, gae_val_aucs, 'o-', label='GAE', alpha=0.8, color='blue')
axes[1, 0].plot(epochs_eval, vgae_val_aucs, 'o-', label='VGAE', alpha=0.8, color='red')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Validation AUC')
axes[1, 0].set_title('Validation AUC Comparison')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Validation AP
axes[1, 1].plot(epochs_eval, gae_val_aps, 'o-', label='GAE', alpha=0.8, color='blue')
axes[1, 1].plot(epochs_eval, vgae_val_aps, 'o-', label='VGAE', alpha=0.8, color='red')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Validation AP')
axes[1, 1].set_title('Validation AP Comparison')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Final Test Evaluation
#
# Let's evaluate both models on the test set to get their final performance scores. We'll compare their ability to predict held-out edges and distinguish them from negative samples.

# %%
# Final test evaluation
print("=== Final Test Evaluation ===")

# Generate negative test edges
test_neg_edge_index = generate_negative_edges(
    test_data.edge_label_index[:, test_data.edge_label == 1],
    train_data.num_nodes,
    test_data.edge_label_index[:, test_data.edge_label == 1].size(1)
)

# Evaluate GAE
gae_test_auc, gae_test_ap = evaluate_model(
    gae_model, train_data,
    test_data.edge_label_index[:, test_data.edge_label == 1],
    test_neg_edge_index
)

# Evaluate VGAE
vgae_test_auc, vgae_test_ap = evaluate_model(
    vgae_model, train_data,
    test_data.edge_label_index[:, test_data.edge_label == 1],
    test_neg_edge_index
)

print(f"{'Model':<8} {'Test AUC':<10} {'Test AP':<10}")
print("-" * 30)
print(f"{'GAE':<8} {gae_test_auc:<10.4f} {gae_test_ap:<10.4f}")
print(f"{'VGAE':<8} {vgae_test_auc:<10.4f} {vgae_test_ap:<10.4f}")

# Model complexity comparison
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

gae_params = count_parameters(gae_model)
vgae_params = count_parameters(vgae_model)

print(f"\n=== Model Complexity ===")
print(f"GAE parameters: {gae_params}")
print(f"VGAE parameters: {vgae_params}")

# %% [markdown]
# ## 10. Node Embeddings and Clustering Analysis
#
# ### Mathematical Framework for Embedding Quality Assessment
#
# **Embedding Space Analysis:**
#
# The learned embeddings $\mathbf{Z} \in \mathbb{R}^{N \times k}$ should exhibit desirable geometric properties:
#
# **1. Cluster Separability:**
# $$\text{Inter-cluster distance} > \text{Intra-cluster distance}$$
#
# Formally, for clusters $\mathcal{C}_1, \mathcal{C}_2$:
# $$\min_{i \in \mathcal{C}_1, j \in \mathcal{C}_2} \|\mathbf{z}_i - \mathbf{z}_j\|_2 > \max_{i,j \in \mathcal{C}_1} \|\mathbf{z}_i - \mathbf{z}_j\|_2$$
#
# **2. Graph Structure Preservation:**
# $$\|\mathbf{z}_i - \mathbf{z}_j\|_2 \approx f(d_{\mathcal{G}}(i,j))$$
#
# where $d_{\mathcal{G}}(i,j)$ is the shortest path distance in the graph.
#
# **Clustering Evaluation Metrics:**
#
# **1. Adjusted Rand Index (ARI):**
# $$\text{ARI} = \frac{\sum_{ij} \binom{n_{ij}}{2} - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right] / \binom{n}{2}}{\frac{1}{2}\left[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\right] - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right] / \binom{n}{2}}$$
#
# where:
# - $n_{ij}$: Number of objects in both cluster $i$ and class $j$
# - $a_i = \sum_j n_{ij}$: Total objects in cluster $i$
# - $b_j = \sum_i n_{ij}$: Total objects in class $j$
#
# **2. Normalized Mutual Information (NMI):**
# $$\text{NMI} = \frac{2 \times I(\mathcal{C}, \mathcal{Y})}{H(\mathcal{C}) + H(\mathcal{Y})}$$
#
# where:
# - $I(\mathcal{C}, \mathcal{Y}) = \sum_{c,y} P(c,y) \log \frac{P(c,y)}{P(c)P(y)}$: Mutual information
# - $H(\mathcal{C}) = -\sum_c P(c) \log P(c)$: Entropy of cluster assignments
# - $H(\mathcal{Y}) = -\sum_y P(y) \log P(y)$: Entropy of true labels
#
# **K-Means Clustering Objective:**
# $$\arg\min_{\mathcal{C}} \sum_{i=1}^k \sum_{\mathbf{z} \in \mathcal{C}_i} \|\mathbf{z} - \boldsymbol{\mu}_i\|_2^2$$
#
# where $\boldsymbol{\mu}_i = \frac{1}{|\mathcal{C}_i|} \sum_{\mathbf{z} \in \mathcal{C}_i} \mathbf{z}$ is the cluster centroid.
#
# **Embedding Quality Indicators:**
#
# **1. Silhouette Score:**
# $$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$
#
# where:
# - $a_i$: Average distance to other points in same cluster
# - $b_i$: Average distance to points in nearest different cluster
#
# **2. Davies-Bouldin Index:**
# $$\text{DB} = \frac{1}{k} \sum_{i=1}^k \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d_{ij}}$$
#
# where $\sigma_i$ is intra-cluster distance and $d_{ij}$ is inter-cluster distance.
#
# **Comparison Framework:**
#
# **Statistical Significance Testing:**
# Use Wilcoxon signed-rank test for comparing clustering performance:
# $$Z = \frac{W - \mu_W}{\sigma_W}$$
#
# where $W$ is the test statistic for paired differences.
#
# **Effect Size Analysis:**
# $$r = \frac{Z}{\sqrt{N}}$$
#
# where $N$ is the total number of observations.
#
# **Embedding Dimension Impact:**
# $$\text{Clustering Quality} \propto \log(k) \times \text{Signal-to-Noise Ratio}$$
#
# Higher dimensions can capture more information but may include noise that hurts clustering performance.
#
# The mathematical framework enables rigorous assessment of embedding quality for downstream clustering tasks.

# %%
# Extract embeddings from both models
gae_model.eval()
vgae_model.eval()

with torch.no_grad():
    # Get original data for full graph
    original_data = dataset[0].to(device)

    # GAE embeddings
    gae_embeddings = gae_model.encode(original_data.x, original_data.edge_index).cpu().numpy()

    # VGAE embeddings (mean of the distribution)
    vgae_embeddings = vgae_model.encode(original_data.x, original_data.edge_index).cpu().numpy()

    # True labels for clustering evaluation
    true_labels = original_data.y.cpu().numpy()

print(f"GAE embedding shape: {gae_embeddings.shape}")
print(f"VGAE embedding shape: {vgae_embeddings.shape}")
print(f"Number of true classes: {len(np.unique(true_labels))}")

# Clustering evaluation
def evaluate_clustering(embeddings, true_labels, n_clusters):
    """
    Evaluate clustering quality using embeddings
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)

    return ari, nmi, cluster_labels

# Evaluate clustering for both models
n_classes = len(np.unique(true_labels))

gae_ari, gae_nmi, gae_clusters = evaluate_clustering(gae_embeddings, true_labels, n_classes)
vgae_ari, vgae_nmi, vgae_clusters = evaluate_clustering(vgae_embeddings, true_labels, n_classes)

print(f"\n=== Clustering Performance ===")
print(f"{'Model':<8} {'ARI':<8} {'NMI':<8}")
print("-" * 25)
print(f"{'GAE':<8} {gae_ari:<8.4f} {gae_nmi:<8.4f}")
print(f"{'VGAE':<8} {vgae_ari:<8.4f} {vgae_nmi:<8.4f}")

# %% [markdown]
# ## 11. Embeddings Visualization
#
# ### Mathematical Foundation of Dimensionality Reduction
#
# **t-Distributed Stochastic Neighbor Embedding (t-SNE):**
#
# t-SNE preserves local neighborhood structure when reducing from high-dimensional embeddings $\mathbf{Z} \in \mathbb{R}^{N \times k}$ to 2D visualization space $\mathbf{Y} \in \mathbb{R}^{N \times 2}$.
#
# **Step 1: High-Dimensional Similarities**
# $$p_{j|i} = \frac{\exp(-\|\mathbf{z}_i - \mathbf{z}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{z}_i - \mathbf{z}_k\|^2 / 2\sigma_i^2)}$$
#
# $$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$$
#
# where $\sigma_i$ is chosen such that perplexity $\text{Perp}(P_i) = 2^{H(P_i)}$ equals a target value (typically 30).
#
# **Step 2: Low-Dimensional Similarities**
# $$q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$$
#
# Uses Student t-distribution with 1 degree of freedom to handle crowding problem.
#
# **Step 3: Optimization Objective**
# $$\mathcal{L} = \text{KL}(P||Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$$
#
# **Gradient Computation:**
# $$\frac{\partial \mathcal{L}}{\partial \mathbf{y}_i} = 4 \sum_j (p_{ij} - q_{ij})(\mathbf{y}_i - \mathbf{y}_j)(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}$$
#
# **Interpretation Guidelines:**
#
# **1. Cluster Cohesion:**
# Tight clusters in t-SNE indicate nodes with similar embeddings:
# $$\text{Cluster Quality} = \frac{\text{Inter-cluster distance}}{\text{Intra-cluster distance}}$$
#
# **2. Separation Assessment:**
# Well-separated clusters suggest good class discrimination:
# $$\text{Separation Score} = \min_{c_1 \neq c_2} \frac{d(\text{centroid}_{c_1}, \text{centroid}_{c_2})}{\max(\text{radius}_{c_1}, \text{radius}_{c_2})}$$
#
# **3. Neighborhood Preservation:**
# $$\text{Trustworthiness} = 1 - \frac{2}{NK(2N-3K-1)} \sum_{i=1}^N \sum_{j \in U_K(i)} (r(i,j) - K)$$
#
# where $U_K(i)$ are the $K$-nearest neighbors of point $i$ in the low-dimensional space that are not among its $K$-nearest neighbors in high-dimensional space.
#
# **Embedding Quality Indicators:**
#
# **1. Local Structure Preservation:**
# $$\text{Local Score} = \frac{1}{N} \sum_{i=1}^N \frac{|N_k^{\text{high}}(i) \cap N_k^{\text{low}}(i)|}{k}$$
#
# **2. Global Structure Preservation:**
# $$\text{Global Score} = \text{Correlation}(\text{dist}_{\text{high}}, \text{dist}_{\text{low}})$$
#
# **Visualization Diagnostics:**
#
# **1. Perplexity Sensitivity:**
# - Low perplexity (5-15): Emphasizes local structure
# - High perplexity (50-100): Emphasizes global structure
# - Optimal range: 20-50 for most applications
#
# **2. Convergence Assessment:**
# Monitor KL divergence convergence:
# $$\Delta \mathcal{L} = |\mathcal{L}(t) - \mathcal{L}(t-1)| < \epsilon$$
#
# **GAE vs. VGAE Visualization Comparison:**
#
# **Expected Differences:**
# - **GAE**: May show more scattered embeddings due to deterministic nature
# - **VGAE**: Often produces more structured, compact clusters due to KL regularization
#
# **Quantitative Assessment:**
# $$\text{Cluster Compactness} = \frac{1}{C} \sum_{c=1}^C \frac{1}{|c|} \sum_{i \in c} \|\mathbf{y}_i - \boldsymbol{\mu}_c\|_2$$
#
# The mathematical framework enables principled interpretation of embedding visualizations and comparison between different autoencoder architectures.

# %%
# Visualize embeddings using t-SNE
print("Applying t-SNE to embeddings...")

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
gae_embeddings_2d = tsne.fit_transform(gae_embeddings)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
vgae_embeddings_2d = tsne.fit_transform(vgae_embeddings)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# GAE embeddings
scatter1 = axes[0].scatter(gae_embeddings_2d[:, 0], gae_embeddings_2d[:, 1],
                          c=true_labels, cmap='tab10', alpha=0.7, s=20)
axes[0].set_title('GAE Embeddings (t-SNE)')
axes[0].set_xlabel('t-SNE Dimension 1')
axes[0].set_ylabel('t-SNE Dimension 2')
axes[0].grid(True, alpha=0.3)

# VGAE embeddings
scatter2 = axes[1].scatter(vgae_embeddings_2d[:, 0], vgae_embeddings_2d[:, 1],
                          c=true_labels, cmap='tab10', alpha=0.7, s=20)
axes[1].set_title('VGAE Embeddings (t-SNE)')
axes[1].set_xlabel('t-SNE Dimension 1')
axes[1].set_ylabel('t-SNE Dimension 2')
axes[1].grid(True, alpha=0.3)

# Add colorbar
plt.colorbar(scatter2, ax=axes[1], label='Node Class')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 12. Reconstruction Quality Analysis
#
# ### Mathematical Framework for Reconstruction Assessment
#
# **Reconstruction Score Distribution Analysis:**
#
# The reconstruction scores $s_{ij} = \sigma(\mathbf{z}_i^T \mathbf{z}_j)$ should exhibit different distributions for positive and negative edges:
#
# **For Positive Edges:** $s_{ij} \sim \mathcal{D}_{\text{pos}}$ with higher mean
# **For Negative Edges:** $s_{ij} \sim \mathcal{D}_{\text{neg}}$ with lower mean
#
# **Ideal Separation:**
# $$\mathbb{E}[s_{ij}^+] - \mathbb{E}[s_{ij}^-] > k \times \sqrt{\text{Var}[s_{ij}^+] + \text{Var}[s_{ij}^-]}$$
#
# where $k \geq 2$ for good separation.
#
# **Statistical Analysis of Reconstruction:**
#
# **1. Signal-to-Noise Ratio:**
# $$\text{SNR} = \frac{(\mu_{\text{pos}} - \mu_{\text{neg}})^2}{\sigma_{\text{pos}}^2 + \sigma_{\text{neg}}^2}$$
#
# Higher SNR indicates better discriminative power.
#
# **2. Overlap Coefficient:**
# $$\text{Overlap} = \frac{\int \min(\mathcal{D}_{\text{pos}}(x), \mathcal{D}_{\text{neg}}(x)) dx}{\int \mathcal{D}_{\text{pos}}(x) dx}$$
#
# Lower overlap indicates better separation.
#
# **3. KL Divergence Between Distributions:**
# $$D_{KL}(\mathcal{D}_{\text{pos}} \| \mathcal{D}_{\text{neg}}) = \int \mathcal{D}_{\text{pos}}(x) \log \frac{\mathcal{D}_{\text{pos}}(x)}{\mathcal{D}_{\text{neg}}(x)} dx$$
#
# **Reconstruction Error Analysis:**
#
# **Per-Edge Reconstruction Error:**
# $$\epsilon_{ij} = |A_{ij} - \hat{A}_{ij}|$$
#
# **Graph-Level Reconstruction Metrics:**
#
# **1. Frobenius Norm Error:**
# $$\text{Error}_F = \|\mathbf{A} - \hat{\mathbf{A}}\|_F = \sqrt{\sum_{i,j} (A_{ij} - \hat{A}_{ij})^2}$$
#
# **2. Normalized Reconstruction Error:**
# $$\text{Error}_{\text{norm}} = \frac{\|\mathbf{A} - \hat{\mathbf{A}}\|_F}{\|\mathbf{A}\|_F}$$
#
# **3. Spectral Reconstruction Quality:**
# Compare eigenvalues of original and reconstructed adjacency matrices:
# $$\text{Spectral Error} = \sum_{i=1}^k |\lambda_i(\mathbf{A}) - \lambda_i(\hat{\mathbf{A}})|$$
#
# **Threshold Analysis:**
#
# **Optimal Threshold Selection:**
# $$\tau^* = \arg\max_{\tau} \text{F1}(\tau) = \arg\max_{\tau} \frac{2 \times \text{Precision}(\tau) \times \text{Recall}(\tau)}{\text{Precision}(\tau) + \text{Recall}(\tau)}$$
#
# **ROC Analysis:**
# $$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) dt = P(\text{score}_+ > \text{score}_-)$$
#
# **Precision-Recall Analysis:**
# $$\text{AP} = \int_0^1 \text{Precision}(\text{Recall}^{-1}(r)) dr$$
#
# **Model-Specific Analysis:**
#
# **GAE Reconstruction Properties:**
# - Deterministic scores for same input
# - May exhibit mode collapse for similar nodes
# - Score distribution often more peaked
#
# **VGAE Reconstruction Properties:**
# - Stochastic sampling introduces variability
# - KL regularization may smooth score distributions
# - Often better calibrated probabilities
#
# **Calibration Assessment:**
# $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|$$
#
# where $B_m$ are bins of predictions with similar confidence scores.
#
# **Reconstruction Quality Diagnostics:**
#
# **1. Score Histogram Analysis:**
# - Bimodal distribution indicates good separation
# - Unimodal distribution suggests poor discrimination
# - High variance may indicate unstable embeddings
#
# **2. Calibration Plots:**
# Plot predicted probability vs. actual frequency of positive edges.
#
# **3. Confidence Intervals:**
# For VGAE, analyze uncertainty in reconstruction scores:
# $$\text{CI}_{95\%} = \hat{A}_{ij} \pm 1.96 \times \text{SE}(\hat{A}_{ij})$$
#
# The mathematical framework provides comprehensive assessment of reconstruction quality and model reliability.

# %%
# Analyze reconstruction quality
def analyze_reconstruction(model, data, num_samples=1000):
    """
    Analyze reconstruction scores for positive and negative edges
    """
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)

        # Sample some positive edges
        pos_edges = data.edge_index[:, :num_samples]
        pos_scores = model.decode(z, pos_edges).sigmoid().cpu().numpy()

        # Sample negative edges
        neg_edges = generate_negative_edges(data.edge_index, data.num_nodes, num_samples)
        neg_scores = model.decode(z, neg_edges).sigmoid().cpu().numpy()

    return pos_scores, neg_scores

# Get reconstruction scores
gae_pos_scores, gae_neg_scores = analyze_reconstruction(gae_model, original_data)
vgae_pos_scores, vgae_neg_scores = analyze_reconstruction(vgae_model, original_data)

# Plot reconstruction score distributions
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# GAE reconstruction scores
axes[0].hist(gae_pos_scores, bins=50, alpha=0.7, label='Positive Edges', color='green')
axes[0].hist(gae_neg_scores, bins=50, alpha=0.7, label='Negative Edges', color='red')
axes[0].set_xlabel('Reconstruction Score')
axes[0].set_ylabel('Frequency')
axes[0].set_title('GAE Reconstruction Scores')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# VGAE reconstruction scores
axes[1].hist(vgae_pos_scores, bins=50, alpha=0.7, label='Positive Edges', color='green')
axes[1].hist(vgae_neg_scores, bins=50, alpha=0.7, label='Negative Edges', color='red')
axes[1].set_xlabel('Reconstruction Score')
axes[1].set_ylabel('Frequency')
axes[1].set_title('VGAE Reconstruction Scores')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print statistics
print("=== Reconstruction Score Statistics ===")
print(f"{'Model':<6} {'Pos Mean':<10} {'Pos Std':<10} {'Neg Mean':<10} {'Neg Std':<10}")
print("-" * 55)
print(f"{'GAE':<6} {gae_pos_scores.mean():<10.4f} {gae_pos_scores.std():<10.4f} "
      f"{gae_neg_scores.mean():<10.4f} {gae_neg_scores.std():<10.4f}")
print(f"{'VGAE':<6} {vgae_pos_scores.mean():<10.4f} {vgae_pos_scores.std():<10.4f} "
      f"{vgae_neg_scores.mean():<10.4f} {vgae_neg_scores.std():<10.4f}")

# %% [markdown]
# ## 13. CPU Optimization Tips for Graph Autoencoders
#
# ### Mathematical Framework for Computational Optimization
#
# **Computational Complexity Analysis:**
#
# **Memory Complexity:**
# $$M_{\text{total}} = M_{\text{features}} + M_{\text{embeddings}} + M_{\text{gradients}} + M_{\text{adjacency}}$$
#
# where:
# - $M_{\text{features}} = N \times d \times \text{sizeof}(\text{float})$
# - $M_{\text{embeddings}} = N \times k \times \text{sizeof}(\text{float})$
# - $M_{\text{gradients}} = 2 \times |\theta| \times \text{sizeof}(\text{float})$ (for Adam)
# - $M_{\text{adjacency}} = N^2 \times \text{sizeof}(\text{bool})$ (dense) or $2|\mathcal{E}| \times \text{sizeof}(\text{int})$ (sparse)
#
# **Time Complexity per Training Step:**
#
# **Forward Pass:**
# - **GCN Layers**: $O(|\mathcal{E}| \times d_{\text{hidden}} + N \times d_{\text{hidden}}^2)$
# - **Decoding**: $O(|\mathcal{E}_{\text{batch}}| \times k)$ for sampled edges
# - **Total Forward**: $O(|\mathcal{E}| \times d + N \times d^2)$
#
# **Backward Pass:**
# - **Similar to forward but with gradient computation overhead**
# - **VGAE Additional**: $O(N \times k)$ for KL divergence gradients
#
# **Optimization Strategies:**
#
# **1. Architecture Scaling:**
# $$\text{Parameters} = d \times h + h \times k + \text{bias terms}$$
#
# **Optimal Scaling for M2 MacBook Air:**
# - Input dimension: $d = 1433$ (fixed by dataset)
# - Hidden dimension: $h = 32$ (reduced from typical 64-128)
# - Embedding dimension: $k = 16$ (reduced from typical 32-64)
#
# **Parameter Count:**
# $$|\theta| = 1433 \times 32 + 32 \times 16 + \text{biases} \approx 46,400$$
#
# **2. Memory Optimization:**
#
# **Gradient Checkpointing:**
# $$M_{\text{activations}} = \sqrt{L} \times M_{\text{layer}}$$
#
# instead of $L \times M_{\text{layer}}$ for $L$ layers.
#
# **Mixed Precision Training:**
# Use FP16 for forward pass, FP32 for gradients:
# $$M_{\text{reduction}} \approx 50\%$$
#
# **Sparse Adjacency Representation:**
# $$M_{\text{sparse}} = 2|\mathcal{E}| \times 4 \text{ bytes} \ll N^2 \times 1 \text{ byte}$$
#
# **3. Computational Optimization:**
#
# **Batch Size Scaling:**
# $$\text{Throughput} = \frac{B \times \text{Gradient Updates}}{\text{Wall Clock Time}}$$
#
# Optimal batch size for CPU: $B = 32-128$ edges per batch.
#
# **Negative Sampling Efficiency:**
# $$\text{Sampling Complexity} = O(|\mathcal{E}^+| \times \log(N^2 - |\mathcal{E}|))$$
#
# **4. PyTorch-Specific Optimizations:**
#
# **Thread Configuration:**
# ```python
# torch.set_num_threads(8)  # M2 has 8 cores
# torch.set_num_interop_threads(2)  # Reduce overhead
# ```
#
# **Memory Management:**
# ```python
# torch.backends.mkldnn.enabled = True  # Intel MKL-DNN
# torch.backends.cudnn.benchmark = False  # CPU mode
# ```
#
# **Autograd Optimization:**
# ```python
# with torch.autograd.detect_anomaly():  # Debug mode
#     # Training code
# ```
#
# **5. Training Efficiency:**
#
# **Learning Rate Scheduling:**
# $$\alpha(t) = \alpha_0 \times \gamma^{\lfloor t/\text{step\_size} \rfloor}$$
#
# **Early Stopping Criterion:**
# $$\text{Stop if } \text{AUC}_{\text{val}}(t) - \text{AUC}_{\text{val}}(t-p) < \epsilon$$
#
# for patience $p$ and threshold $\epsilon$.
#
# **6. Evaluation Optimization:**
#
# **Subsampled Evaluation:**
# Instead of evaluating on all possible edges, sample subset:
# $$|\mathcal{E}_{\text{eval}}| = \min(10^4, |\mathcal{E}_{\text{total}}|)$$
#
# **Performance Monitoring:**
#
# **CPU Utilization:**
# $$\text{Efficiency} = \frac{\text{CPU Time Used}}{\text{Wall Clock Time} \times \text{Number of Cores}}$$
#
# **Memory Usage:**
# $$\text{Peak Memory} = \max_t \sum_{\text{tensor}} \text{size}(\text{tensor}(t))$$
#
# **Throughput Measurement:**
# $$\text{Edges per Second} = \frac{|\mathcal{E}_{\text{processed}}|}{\text{Training Time}}$$
#
# These mathematical optimizations ensure efficient graph autoencoder training on resource-constrained CPU environments while maintaining model quality.

# %%
# CPU Optimization Tips for Graph Autoencoders
print("=== CPU Optimization Tips for Graph Autoencoders ===")

print("\n1. Model Architecture Optimizations:")
print("   - Use smaller embedding dimensions (16-32 instead of 64-128)")
print("   - Limit encoder to 2-3 layers max")
print("   - Use cached=True in GCN layers for repeated computations")
print("   - Consider using lighter activation functions (ReLU over ELU)")

print("\n2. Training Optimizations:")
print("   - Use fewer negative samples during training")
print("   - Implement gradient accumulation for large graphs")
print("   - Use learning rate scheduling for better convergence")
print("   - Early stopping based on validation metrics")

print("\n3. Memory Management:")
print("   - Process graphs in smaller subgraphs if possible")
print("   - Use in-place operations where applicable")
print("   - Clear intermediate tensors explicitly")
print("   - Monitor memory usage with torch.profiler")

print("\n4. Evaluation Efficiency:")
print("   - Sample edges for evaluation instead of using all edges")
print("   - Use batch processing for large-scale link prediction")
print("   - Cache embeddings for multiple evaluation rounds")

# Example of memory-efficient evaluation
def memory_efficient_evaluation(model, data, batch_size=1000):
    """
    Memory-efficient evaluation for large graphs
    """
    model.eval()
    with torch.no_grad():
        # Encode once and reuse
        z = model.encode(data.x, data.edge_index)

        # Process in batches
        all_scores = []
        for i in range(0, data.edge_index.size(1), batch_size):
            batch_edges = data.edge_index[:, i:i+batch_size]
            batch_scores = model.decode(z, batch_edges).sigmoid()
            all_scores.append(batch_scores)

        return torch.cat(all_scores)

print(f"\nCurrent setup optimized for M2 MacBook Air:")
print(f"- PyTorch threads: {torch.get_num_threads()}")
print(f"- Device: {device}")
print(f"- Embedding dimension: 16 (CPU-friendly size)")
print(f"- Hidden dimension: 32 (balanced performance/memory)")

# %% [markdown]
# ## Summary
#
# ### Comprehensive Mathematical Foundation of Graph Autoencoders
#
# In this notebook, we have explored the mathematical foundations and practical implementations of graph autoencoders for unsupervised learning:
#
# ### **Graph Autoencoder (GAE)**
# - **Mathematical Core**: $\hat{\mathbf{A}} = \sigma(\mathbf{Z}\mathbf{Z}^T)$ where $\mathbf{Z} = \text{GCN}(\mathbf{X}, \mathbf{A})$
# - **Loss Function**: Binary cross-entropy for link prediction
# - **Key Properties**: Deterministic, efficient, interpretable inner product decoder
# - **Advantages**: Simple implementation, stable training, good baseline performance
# - **Limitations**: No uncertainty quantification, potential overfitting to specific edge patterns
#
# ### **Variational Graph Autoencoder (VGAE)**
# - **Mathematical Core**: $q_{\phi}(\mathbf{Z}|\mathbf{X},\mathbf{A}) = \prod_{i=1}^N \mathcal{N}(\boldsymbol{\mu}_i, \text{diag}(\boldsymbol{\sigma}_i^2))$
# - **ELBO Objective**: $\mathcal{L} = \mathbb{E}[\log p(\mathbf{A}|\mathbf{Z})] - D_{KL}[q(\mathbf{Z}|\mathbf{X},\mathbf{A}) \| p(\mathbf{Z})]$
# - **Key Properties**: Probabilistic latent space, KL regularization, reparameterization trick
# - **Advantages**: Uncertainty quantification, robust embeddings, generative capability
# - **Limitations**: More complex training, additional hyperparameters (KL weight)
#
# ### **Mathematical Insights**
#
# **Theoretical Comparison:**
# $$\text{GAE} \subset \text{VGAE} \text{ when } \boldsymbol{\sigma} \rightarrow 0$$
#
# **Complexity Analysis:**
# - **GAE Parameters**: $O(d \times h + h \times k)$
# - **VGAE Parameters**: $O(d \times h + 2h \times k)$ (additional variance parameters)
# - **Computational Cost**: Both $O(|\mathcal{E}| \times d^2 + N \times d^2)$ per epoch
#
# **Performance Trade-offs:**
# - **Link Prediction**: VGAE often slightly better due to regularization
# - **Node Clustering**: VGAE typically superior due to structured embedding space
# - **Computational Efficiency**: GAE faster due to simpler objective
# - **Memory Usage**: VGAE requires ~25% more memory for variance parameters
#
# ### **Key Applications and Mathematical Formulations**
#
# **1. Link Prediction:**
# $$P(\text{edge}_{ij} = 1) = \sigma(\mathbf{z}_i^T \mathbf{z}_j)$$
# - Evaluation: AUC-ROC and Average Precision metrics
# - Challenge: Negative sampling strategy affects performance
#
# **2. Node Clustering:**
# $$\text{Cluster Assignment} = \arg\min_{\mathcal{C}} \sum_{i=1}^k \sum_{\mathbf{z} \in \mathcal{C}_i} \|\mathbf{z} - \boldsymbol{\mu}_i\|_2^2$$
# - Evaluation: ARI and NMI metrics
# - VGAE advantage: KL regularization promotes cluster-friendly embeddings
#
# **3. Graph Generation:**
# $$\mathbf{Z}_{\text{new}} \sim p(\mathbf{Z}) \rightarrow \mathbf{A}_{\text{new}} = \text{Decode}(\mathbf{Z}_{\text{new}})$$
# - Only feasible with VGAE due to probabilistic framework
# - Applications: Data augmentation, synthetic graph creation
#
# **4. Anomaly Detection:**
# $$\text{Anomaly Score} = \|\mathbf{A} - \hat{\mathbf{A}}\|_F^2$$
# - Both models applicable
# - VGAE provides uncertainty estimates for anomaly confidence
#
# ### **Practical Guidelines**
#
# **Architecture Selection:**
# | Use Case | Recommended Model | Rationale |
# |----------|------------------|-----------|
# | **Simple Link Prediction** | GAE | Faster, easier to tune |
# | **Node Clustering** | VGAE | Better regularization |
# | **Graph Generation** | VGAE | Probabilistic sampling |
# | **Uncertainty Quantification** | VGAE | Variance parameters |
# | **Large-Scale Graphs** | GAE | Lower computational cost |
#
# **Hyperparameter Guidelines:**
# - **Embedding Dimension**: $k = 16-32$ for most applications
# - **Hidden Dimension**: $h = 32-64$ for CPU, $h = 128-256$ for GPU
# - **Learning Rate**: $\alpha = 0.01$ with decay scheduling
# - **KL Weight (VGAE)**: $\beta = 1.0$ with optional annealing
#
# ### **Future Directions**
#
# **Mathematical Extensions:**
# - **Hierarchical Variational Autoencoders**: Multi-level latent representations
# - **Normalizing Flows**: More flexible posterior distributions
# - **Attention-based Decoders**: Beyond inner product similarity
# - **Dynamic Graph Autoencoders**: Temporal graph evolution
#
# **Next Notebook Preview:**
# The next notebook will explore **Graph Transformers**, covering:
# - **Self-Attention on Graphs**: $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$
# - **Positional Encodings**: Incorporating graph structural information
# - **Transformer Architectures**: Graph-specific modifications of transformer blocks
#
# This mathematical foundation in graph autoencoders provides the necessary background for understanding more advanced graph neural network architectures and their applications in unsupervised learning scenarios.