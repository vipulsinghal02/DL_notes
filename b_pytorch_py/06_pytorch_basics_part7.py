# %% [markdown]
# # PyTorch Basics Part 7: Transformers and Modern NLP
#
# Understanding the Transformer architecture, attention mechanisms, and pre-trained language models with mathematical foundations
#
# ## Mathematical Framework of Transformer Architecture
#
# **Transformers** revolutionized NLP by replacing recurrence with attention mechanisms, enabling parallel processing and better long-range dependency modeling.
#
# ### Core Mathematical Concepts
#
# **1. Self-Attention Mechanism:**
# For input sequence $\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n] \in \mathbb{R}^{n \times d}$:
#
# **Query, Key, Value Projections:**
# $$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$
#
# **Attention Computation:**
# $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$
#
# **2. Multi-Head Attention:**
# $$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}_O$$
#
# where $\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$
#
# **3. Positional Encoding:**
# $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
# $$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
#
# **4. Layer Architecture:**
# Each Transformer layer applies:
# $$\mathbf{Z} = \text{LayerNorm}(\mathbf{X} + \text{MultiHeadAttention}(\mathbf{X}))$$
# $$\mathbf{Y} = \text{LayerNorm}(\mathbf{Z} + \text{FFN}(\mathbf{Z}))$$
#
# **5. Computational Complexity:**
# - **Self-attention**: $O(n^2 \cdot d)$ time, $O(n^2 + n \cdot d)$ space
# - **Feed-forward**: $O(n \cdot d^2)$ time and space
# - **Total per layer**: $O(n^2 \cdot d + n \cdot d^2)$

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Understanding Attention Mechanisms
#
# ### Mathematical Foundation of Attention
#
# **Attention mechanisms** allow models to dynamically focus on relevant parts of the input sequence:
#
# **Information-Theoretic View:**
# Attention computes a probability distribution over input positions:
# $$\alpha_{i,j} = P(\text{position } j \text{ is relevant for position } i)$$
#
# **Attention as Soft Database Lookup:**
# - **Keys**: $\mathbf{K} = [\mathbf{k}_1, \mathbf{k}_2, \ldots, \mathbf{k}_n]$ (indexable content)
# - **Values**: $\mathbf{V} = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n]$ (retrievable content)
# - **Queries**: $\mathbf{Q} = [\mathbf{q}_1, \mathbf{q}_2, \ldots, \mathbf{q}_m]$ (what we search for)
#
# **Compatibility Function:**
# $$e_{ij} = f(\mathbf{q}_i, \mathbf{k}_j)$$
#
# Common choices:
# - **Dot-product**: $f(\mathbf{q}, \mathbf{k}) = \mathbf{q}^T\mathbf{k}$
# - **Scaled dot-product**: $f(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q}^T\mathbf{k}}{\sqrt{d}}$
# - **Additive**: $f(\mathbf{q}, \mathbf{k}) = \mathbf{v}^T\tanh(\mathbf{W}_q\mathbf{q} + \mathbf{W}_k\mathbf{k})$
#
# **Attention Weights:**
# $$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^n \exp(e_{ik})}$$
#
# **Weighted Combination:**
# $$\mathbf{c}_i = \sum_{j=1}^n \alpha_{ij} \mathbf{v}_j$$
#
# **Advantages over RNNs:**
# 1. **Parallel computation**: All positions computed simultaneously
# 2. **Direct connections**: Any position can attend to any other position
# 3. **Gradient flow**: No vanishing gradient problem for long sequences
# 4. **Interpretability**: Attention weights show model focus

# %%
# Simple attention mechanism implementation
class SimpleAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs):
        # encoder_outputs: (batch_size, seq_len, hidden_size)

        # Compute attention scores for each position
        attention_scores = self.attention(encoder_outputs)  # (batch_size, seq_len, 1)
        attention_scores = attention_scores.squeeze(-1)     # (batch_size, seq_len)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)

        # Compute weighted sum (context vector)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, hidden_size)
        context = context.squeeze(1)  # (batch_size, hidden_size)

        return context, attention_weights

# Demonstrate attention
batch_size, seq_len, hidden_size = 2, 5, 10
encoder_outputs = torch.randn(batch_size, seq_len, hidden_size)

attention_layer = SimpleAttention(hidden_size)
context, weights = attention_layer(encoder_outputs)

print(f"Encoder outputs shape: {encoder_outputs.shape}")
print(f"Context vector shape: {context.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"\nSample attention weights (should sum to 1):")
print(f"Batch 1: {weights[0]}")
print(f"Sum: {weights[0].sum():.4f}")
print(f"Batch 2: {weights[1]}")
print(f"Sum: {weights[1].sum():.4f}")

# %% [markdown]
# ## Scaled Dot-Product Attention
#
# ### Mathematical Analysis of Scaled Dot-Product Attention
#
# **Scaled dot-product attention** is the core mechanism in Transformers:
#
# **Mathematical Formula:**
# $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$
#
# **Why Scaling by $\sqrt{d_k}$?**
#
# **Variance Analysis:**
# If $\mathbf{q}$ and $\mathbf{k}$ have independent components with mean 0 and variance 1:
# $$\text{Var}[\mathbf{q}^T\mathbf{k}] = \sum_{i=1}^{d_k} \text{Var}[q_i k_i] = d_k$$
#
# **Without scaling**: As $d_k$ grows, dot products have larger variance
# **With scaling**: $\text{Var}[\frac{\mathbf{q}^T\mathbf{k}}{\sqrt{d_k}}] = 1$
#
# **Softmax Saturation:**
# Large magnitude inputs to softmax cause:
# $$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} \rightarrow \begin{cases} 1 & \text{if } x_i = \max_j x_j \\ 0 & \text{otherwise} \end{cases}$$
#
# This leads to vanishing gradients.
#
# **Matrix Form Computation:**
# For batch processing with $n$ queries, $m$ key-value pairs:
# - $\mathbf{Q} \in \mathbb{R}^{n \times d_k}$
# - $\mathbf{K} \in \mathbb{R}^{m \times d_k}$
# - $\mathbf{V} \in \mathbb{R}^{m \times d_v}$
# - Output: $\mathbb{R}^{n \times d_v}$
#
# **Computational Cost:**
# 1. **$\mathbf{Q}\mathbf{K}^T$**: $O(n \cdot m \cdot d_k)$ operations
# 2. **Softmax**: $O(n \cdot m)$ operations
# 3. **Attention × Values**: $O(n \cdot m \cdot d_v)$ operations
# 4. **Total**: $O(n \cdot m \cdot (d_k + d_v))$
#
# **Memory Requirements:**
# - **Attention matrix**: $O(n \cdot m)$
# - **Activations**: $O(n \cdot d_v + m \cdot d_k)$
# - **Peak memory**: $O(n \cdot m + n \cdot d_v + m \cdot d_k)$

# %%
# Scaled Dot-Product Attention implementation
def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    Compute scaled dot-product attention.

    Args:
        query: (batch_size, seq_len, d_k)
        key: (batch_size, seq_len, d_k)
        value: (batch_size, seq_len, d_v)
        mask: Optional mask to prevent attention to certain positions
        dropout: Optional dropout layer
    """
    d_k = query.size(-1)

    # Compute attention scores: Q * K^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask if provided (set masked positions to large negative value)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Apply dropout if provided
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Compute weighted sum of values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights

# Test scaled dot-product attention
batch_size, seq_len, d_model = 2, 4, 8

# For simplicity, use same tensor for Q, K, V (self-attention)
x = torch.randn(batch_size, seq_len, d_model)
query = key = value = x

output, attention_weights = scaled_dot_product_attention(query, key, value)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
print(f"\nAttention weights for first batch, first query:")
print(attention_weights[0, 0])  # How much each position attends to each position
print(f"Sum: {attention_weights[0, 0].sum():.4f}")

# %% [markdown]
# ## Multi-Head Attention
#
# ### Mathematical Framework for Multi-Head Attention
#
# **Multi-head attention** runs multiple attention functions in parallel to capture different types of relationships:
#
# **Mathematical Definition:**
# $$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O$$
#
# **Individual Attention Heads:**
# $$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$$
#
# **Projection Matrices:**
# - $\mathbf{W}_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$
# - $\mathbf{W}_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
# - $\mathbf{W}_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$
# - $\mathbf{W}^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$
#
# **Dimension Relationships:**
# Typically: $d_k = d_v = d_{\text{model}}/h$
#
# **Representational Capacity:**
# Each head can learn different attention patterns:
# - **Head 1**: Syntactic dependencies (subject-verb)
# - **Head 2**: Coreference resolution (pronoun-antecedent)
# - **Head 3**: Semantic relationships (cause-effect)
# - **Head 4**: Positional patterns (adjacent words)
#
# **Parameter Analysis:**
# For $h$ heads with $d_{\text{model}} = 512$, $h = 8$:
# - **Per head**: $3 \times (512 \times 64) = 98,304$ parameters (Q, K, V projections)
# - **All heads**: $8 \times 98,304 = 786,432$ parameters
# - **Output projection**: $512 \times 512 = 262,144$ parameters
# - **Total**: $1,048,576$ parameters per multi-head attention layer
#
# **Computational Complexity:**
# - **Sequential processing**: $h$ times single-head cost
# - **Parallel processing**: Same as single-head (with sufficient hardware)
# - **Memory overhead**: $h$ attention matrices vs. 1
#
# **Information Flow:**
# $$\mathbf{X} \rightarrow \{(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i)\}_{i=1}^h \rightarrow \{\text{head}_i\}_{i=1}^h \rightarrow \text{Concat} \rightarrow \mathbf{W}^O \rightarrow \text{Output}$$
#
# **Theoretical Benefits:**
# 1. **Subspace specialization**: Each head focuses on different representation subspaces
# 2. **Increased model capacity**: More parameters without increasing depth
# 3. **Attention diversity**: Reduces risk of attention collapse
# 4. **Parallel computation**: Heads computed independently

# %%
# Multi-Head Attention implementation
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. Linear projections
        Q = self.w_q(query)  # (batch_size, seq_len, d_model)
        K = self.w_k(key)
        V = self.w_v(value)

        # 2. Reshape for multi-head attention
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Apply attention
        attention_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )

        # 4. Concatenate heads
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # 5. Final linear projection
        output = self.w_o(attention_output)

        return output, attention_weights

# Test multi-head attention
d_model, num_heads = 64, 8
seq_len, batch_size = 10, 2

mha = MultiHeadAttention(d_model, num_heads)
x = torch.randn(batch_size, seq_len, d_model)

# Self-attention (query, key, value are all the same)
output, weights = mha(x, x, x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"Number of parameters: {sum(p.numel() for p in mha.parameters()):,}")

# Visualize attention pattern
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(weights[0, 0].detach().numpy(), cmap='Blues')
plt.title('Attention Head 1')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(weights[0, 1].detach().numpy(), cmap='Blues')
plt.title('Attention Head 2')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.colorbar()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Positional Encoding
#
# ### Mathematical Theory of Position Information
#
# **Positional encodings** inject sequence order information into the attention-based Transformer:
#
# **Why Positional Information is Needed:**
# Attention mechanism is **permutation invariant**:
# $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Attention}(\mathbf{Q}\mathbf{P}, \mathbf{K}\mathbf{P}, \mathbf{V}\mathbf{P})$$
# for any permutation matrix $\mathbf{P}$.
#
# **Sinusoidal Positional Encoding:**
# $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
# $$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
#
# **Mathematical Properties:**
#
# **1. Unique Position Encoding:**
# Each position $pos$ gets a unique encoding vector $\mathbf{pe}_{pos} \in \mathbb{R}^{d_{\text{model}}}$.
#
# **2. Relative Position Information:**
# For any fixed offset $k$:
# $$PE_{pos+k} = \mathbf{M}_k \cdot PE_{pos}$$
#
# where $\mathbf{M}_k$ is a linear transformation matrix.
#
# **3. Frequency Spectrum:**
# Different dimensions encode different frequency components:
# - **Low frequencies**: Capture long-range position differences
# - **High frequencies**: Capture fine-grained local position differences
#
# **Wavelength Analysis:**
# The wavelength for dimension $i$ is:
# $$\lambda_i = 2\pi \cdot 10000^{2i/d_{\text{model}}}$$
#
# **Range**: From $2\pi$ (highest frequency) to $2\pi \cdot 10000$ (lowest frequency).
#
# **Alternative Approaches:**
#
# **1. Learned Positional Embeddings:**
# $$\mathbf{PE} \in \mathbb{R}^{L_{\max} \times d_{\text{model}}}$$
# Trainable parameters for each position up to maximum length $L_{\max}$.
#
# **2. Relative Positional Encoding:**
# Attention modified to include relative positions:
# $$e_{ij} = \frac{(\mathbf{x}_i \mathbf{W}_Q)(\mathbf{x}_j \mathbf{W}_K + \mathbf{r}_{i-j})^T}{\sqrt{d_k}}$$
#
# **3. Rotary Position Embedding (RoPE):**
# $$\mathbf{q}_m = \mathbf{R}_{\Theta,m} \mathbf{W}_q \mathbf{x}_m$$
# $$\mathbf{k}_n = \mathbf{R}_{\Theta,n} \mathbf{W}_k \mathbf{x}_n$$
#
# where $\mathbf{R}_{\Theta,m}$ is rotation matrix encoding position $m$.
#
# **Addition vs Concatenation:**
# - **Addition**: $\mathbf{x}_{pos} = \mathbf{emb}_{token} + \mathbf{pe}_{pos}$
# - **Concatenation**: $\mathbf{x}_{pos} = [\mathbf{emb}_{token}; \mathbf{pe}_{pos}]$
#
# Addition preserves model dimension but risks information mixing.

# %%
# Positional Encoding implementation
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Create division term for sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)

        # Add positional encoding
        x = x + self.pe[:seq_len].transpose(0, 1)
        return self.dropout(x)

# Visualize positional encodings
d_model = 64
max_len = 100

pos_encoding = PositionalEncoding(d_model, max_len)

# Create dummy input to get positional encodings
dummy_input = torch.zeros(1, max_len, d_model)
pe_values = pos_encoding.pe[:max_len, 0, :].numpy()  # Get positional encodings

plt.figure(figsize=(15, 5))

# Plot positional encoding heatmap
plt.subplot(1, 3, 1)
plt.imshow(pe_values.T, cmap='RdBu', aspect='auto')
plt.title('Positional Encoding Heatmap')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.colorbar()

# Plot some dimensions over positions
plt.subplot(1, 3, 2)
for i in range(0, d_model, d_model//4):
    plt.plot(pe_values[:50, i], label=f'Dim {i}')
plt.title('PE Values for Different Dimensions')
plt.xlabel('Position')
plt.ylabel('PE Value')
plt.legend()

# Plot first few positions across all dimensions
plt.subplot(1, 3, 3)
for pos in [0, 5, 10, 20]:
    plt.plot(pe_values[pos, :20], label=f'Pos {pos}')
plt.title('PE Across Dimensions for Different Positions')
plt.xlabel('Dimension')
plt.ylabel('PE Value')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Positional encoding shape: {pe_values.shape}")
print(f"Range of PE values: [{pe_values.min():.3f}, {pe_values.max():.3f}]")

# %% [markdown]
# ## Feed-Forward Network and Layer Normalization
#
# ### Mathematical Components of Transformer Layers
#
# **Position-wise Feed-Forward Network:**
# Applied independently to each position in the sequence:
#
# **Mathematical Definition:**
# $$\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$
#
# **Dimension Transformation:**
# $$\mathbb{R}^{d_{\text{model}}} \rightarrow \mathbb{R}^{d_{ff}} \rightarrow \mathbb{R}^{d_{\text{model}}}$$
#
# Typically $d_{ff} = 4 \times d_{\text{model}}$ (expansion factor of 4).
#
# **Parameter Count:**
# For $d_{\text{model}} = 512$, $d_{ff} = 2048$:
# - $\mathbf{W}_1$: $512 \times 2048 = 1,048,576$ parameters
# - $\mathbf{W}_2$: $2048 \times 512 = 1,048,576$ parameters
# - Biases: $2048 + 512 = 2,560$ parameters
# - **Total**: $2,099,712$ parameters per layer
#
# **Layer Normalization Mathematics:**
#
# **Standard Layer Norm:**
# $$\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sigma} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$$
#
# **Statistics Computation:**
# $$\mu = \frac{1}{d} \sum_{i=1}^d x_i$$
# $$\sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$$
#
# **Gradient Properties:**
# Layer normalization provides:
# 1. **Gradient scaling**: $\frac{\partial \text{LayerNorm}(\mathbf{x})}{\partial \mathbf{x}} \propto \frac{1}{\sigma}$
# 2. **Mean-centering**: Removes activation mean shifts
# 3. **Variance stabilization**: Normalizes activation magnitudes
#
# **Pre-norm vs Post-norm:**
#
# **Post-norm (Original Transformer):**
# $$\mathbf{x}' = \text{LayerNorm}(\mathbf{x} + \text{Attention}(\mathbf{x}))$$
# $$\mathbf{y} = \text{LayerNorm}(\mathbf{x}' + \text{FFN}(\mathbf{x}'))$$
#
# **Pre-norm (More stable for deep models):**
# $$\mathbf{x}' = \mathbf{x} + \text{Attention}(\text{LayerNorm}(\mathbf{x}))$$
# $$\mathbf{y} = \mathbf{x}' + \text{FFN}(\text{LayerNorm}(\mathbf{x}'))$$
#
# **Residual Connection Mathematics:**
# $$\mathbf{y} = \mathbf{x} + F(\mathbf{x})$$
#
# **Benefits:**
# - **Gradient flow**: $\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \mathbf{I} + \frac{\partial F(\mathbf{x})}{\partial \mathbf{x}}$
# - **Identity mapping**: Network can learn identity if needed
# - **Training stability**: Mitigates vanishing gradients in deep networks

# %%
# Position-wise Feed-Forward Network
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply first linear transformation with ReLU
        output = F.relu(self.w_1(x))
        output = self.dropout(output)

        # Apply second linear transformation
        output = self.w_2(output)

        return output

# Layer Normalization (used in Transformer)
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# Test feed-forward network
d_model, d_ff = 64, 256
batch_size, seq_len = 2, 10

ffn = PositionwiseFeedForward(d_model, d_ff)
layer_norm = LayerNorm(d_model)

x = torch.randn(batch_size, seq_len, d_model)
ffn_output = ffn(x)
norm_output = layer_norm(x)

print(f"Input shape: {x.shape}")
print(f"FFN output shape: {ffn_output.shape}")
print(f"Layer norm output shape: {norm_output.shape}")
print(f"FFN parameters: {sum(p.numel() for p in ffn.parameters()):,}")

# Show effect of layer normalization
print(f"\nBefore layer norm - Mean: {x.mean():.4f}, Std: {x.std():.4f}")
print(f"After layer norm - Mean: {norm_output.mean():.4f}, Std: {norm_output.std():.4f}")

# %% [markdown]
# ## Transformer Encoder Layer
#
# ### Complete Mathematical Architecture
#
# **Transformer Encoder Layer** combines all components into a coherent processing unit:
#
# **Layer Computation Flow:**
# 1. **Input**: $\mathbf{X} \in \mathbb{R}^{n \times d_{\text{model}}}$
# 2. **Multi-Head Self-Attention**: $\mathbf{Z} = \text{MultiHead}(\mathbf{X}, \mathbf{X}, \mathbf{X})$
# 3. **Residual + LayerNorm**: $\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \mathbf{Z})$
# 4. **Feed-Forward**: $\mathbf{F} = \text{FFN}(\mathbf{X}')$
# 5. **Residual + LayerNorm**: $\mathbf{Y} = \text{LayerNorm}(\mathbf{X}' + \mathbf{F})$
#
# **Mathematical Expression:**
# $$\text{TransformerLayer}(\mathbf{X}) = \text{LayerNorm}(\mathbf{X}' + \text{FFN}(\mathbf{X}'))$$
#
# where $\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{MultiHead}(\mathbf{X}, \mathbf{X}, \mathbf{X}))$
#
# **Parameter Analysis per Layer:**
# For BERT-base configuration ($d_{\text{model}} = 768$, $h = 12$, $d_{ff} = 3072$):
#
# 1. **Multi-Head Attention**:
#    - Q, K, V projections: $3 \times 768^2 = 1,769,472$
#    - Output projection: $768^2 = 589,824$
#    - **Subtotal**: $2,359,296$ parameters
#
# 2. **Feed-Forward Network**:
#    - First layer: $768 \times 3072 = 2,359,296$
#    - Second layer: $3072 \times 768 = 2,359,296$
#    - Biases: $3072 + 768 = 3,840$
#    - **Subtotal**: $4,722,432$ parameters
#
# 3. **Layer Normalization** (2 layers):
#    - $\gamma, \beta$: $2 \times 768 = 1,536$
#
# **Total per layer**: $7,083,264$ parameters
#
# **Information Flow Analysis:**
#
# **Attention Information Mixing:**
# Each position can attend to all positions:
# $$\mathbf{h}_i = \sum_{j=1}^n \alpha_{ij} \mathbf{v}_j$$
#
# **Feed-Forward Information Processing:**
# Position-wise transformation preserves sequence structure while allowing complex non-linear mappings.
#
# **Residual Connections:**
# Enable gradient flow and allow network to learn incremental updates:
# $$\frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Y}} \left(\mathbf{I} + \frac{\partial (\text{Attention + FFN})}{\partial \mathbf{X}}\right)$$
#
# **Layer Stacking:**
# Deep Transformer with $L$ layers:
# $$\mathbf{Y} = \text{Layer}_L(\text{Layer}_{L-1}(\cdots \text{Layer}_1(\mathbf{X} + \mathbf{PE}) \cdots))$$
#
# **Computational Complexity per Layer:**
# - **Self-Attention**: $O(n^2 d + nd^2)$
# - **Feed-Forward**: $O(nd_{ff}d) = O(4nd^2)$
# - **Total**: $O(n^2 d + nd^2)$ (attention dominates for long sequences)

# %%
# Complete Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        # Multi-head attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-forward network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attention_weights

# Complete Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Token embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        attention_weights = []

        # Pass through each encoder layer
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)

        return x, attention_weights

# Create and test Transformer encoder
vocab_size = 1000
d_model = 128
num_heads = 8
d_ff = 512
num_layers = 6

transformer = TransformerEncoder(vocab_size, d_model, num_heads, d_ff, num_layers)

# Test input
batch_size, seq_len = 2, 20
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

output, attention_weights = transformer(input_ids)

print(f"Input shape: {input_ids.shape}")
print(f"Output shape: {output.shape}")
print(f"Number of attention weight tensors: {len(attention_weights)}")
print(f"Each attention weight shape: {attention_weights[0].shape}")
print(f"Total parameters: {sum(p.numel() for p in transformer.parameters()):,}")

# Parameter breakdown
embedding_params = transformer.embedding.weight.numel()
encoder_params = sum(p.numel() for layer in transformer.layers for p in layer.parameters())
print(f"\nParameter breakdown:")
print(f"Embedding: {embedding_params:,}")
print(f"Encoder layers: {encoder_params:,}")
print(f"Parameters per layer: {encoder_params // num_layers:,}")

# %% [markdown]
# ## Text Classification with Transformers
#
# ### Mathematical Framework for Transformer-Based Classification
#
# **Classification Objective:**
# Map variable-length sequences to discrete labels:
# $$f: \mathcal{V}^* \rightarrow \{1, 2, \ldots, C\}$$
#
# **Architecture Components:**
#
# **1. Token + Position Embeddings:**
# $$\mathbf{X} = \mathbf{E}_{\text{token}} + \mathbf{E}_{\text{pos}}$$
#
# where $\mathbf{E}_{\text{token}} \in \mathbb{R}^{n \times d}$ and $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{n \times d}$.
#
# **2. Transformer Encoding:**
# $$\mathbf{H} = \text{TransformerEncoder}(\mathbf{X})$$
#
# **3. Sequence Representation:**
# Multiple strategies for aggregating sequence information:
#
# **CLS Token Approach:**
# $$\mathbf{h}_{\text{cls}} = \mathbf{H}[0, :] \quad \text{(first position)}$$
#
# **Mean Pooling:**
# $$\mathbf{h}_{\text{mean}} = \frac{1}{n} \sum_{i=1}^n \mathbf{H}[i, :]$$
#
# **Attention Pooling:**
# $$\alpha_i = \frac{\exp(\mathbf{w}^T \mathbf{H}[i, :])}{\sum_{j=1}^n \exp(\mathbf{w}^T \mathbf{H}[j, :])}$$
# $$\mathbf{h}_{\text{att}} = \sum_{i=1}^n \alpha_i \mathbf{H}[i, :]$$
#
# **4. Classification Head:**
# $$\mathbf{y} = \text{softmax}(\mathbf{W}_{\text{cls}} \mathbf{h} + \mathbf{b}_{\text{cls}})$$
#
# **Loss Function:**
# $$L = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log \hat{y}_{i,c}$$
#
# **Fine-tuning Mathematics:**
#
# **Parameter Optimization:**
# $$\boldsymbol{\theta} = \{\boldsymbol{\theta}_{\text{encoder}}, \boldsymbol{\theta}_{\text{classifier}}\}$$
#
# **Learning Rate Scheduling:**
# - **Encoder layers**: $\eta_{\text{enc}} \in [1e-5, 5e-5]$ (lower LR)
# - **Classifier**: $\eta_{\text{cls}} \in [1e-4, 1e-3]$ (higher LR)
#
# **Gradient Flow:**
# $$\frac{\partial L}{\partial \boldsymbol{\theta}_{\text{encoder}}} = \frac{\partial L}{\partial \mathbf{h}} \frac{\partial \mathbf{h}}{\partial \mathbf{H}} \frac{\partial \mathbf{H}}{\partial \boldsymbol{\theta}_{\text{encoder}}}$$
#
# **Regularization Strategies:**
# - **Dropout**: Applied in classifier and attention layers
# - **Weight decay**: L2 regularization on parameters
# - **Early stopping**: Based on validation performance
# - **Layer freezing**: Freeze lower layers, fine-tune upper layers

# %%
# Transformer-based text classifier
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, num_classes, max_len=512):
        super(TransformerClassifier, self).__init__()

        self.transformer = TransformerEncoder(
            vocab_size, d_model, num_heads, d_ff, num_layers, max_len
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, mask=None):
        # Get transformer output
        transformer_output, attention_weights = self.transformer(x, mask)

        # Use [CLS] token (first token) or average pooling for classification
        # Here we'll use average pooling
        if mask is not None:
            # Mask out padding tokens for average
            mask_expanded = mask.unsqueeze(-1).expand(transformer_output.size())
            sum_embeddings = torch.sum(transformer_output * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            # Simple average pooling
            pooled_output = transformer_output.mean(dim=1)

        # Classify
        logits = self.classifier(pooled_output)

        return logits, attention_weights

# Create simple vocabulary and tokenizer for demo
simple_vocab = {
    '<PAD>': 0, '<UNK>': 1, 'good': 2, 'bad': 3, 'great': 4, 'terrible': 5,
    'amazing': 6, 'awful': 7, 'love': 8, 'hate': 9, 'this': 10, 'is': 11,
    'movie': 12, 'book': 13, 'product': 14, 'the': 15, 'a': 16, 'very': 17
}

def simple_tokenize(text, vocab, max_len=10):
    """Simple tokenization for demo"""
    words = text.lower().split()
    tokens = [vocab.get(word, vocab['<UNK>']) for word in words]

    # Pad or truncate
    if len(tokens) < max_len:
        tokens.extend([vocab['<PAD>']] * (max_len - len(tokens)))
    else:
        tokens = tokens[:max_len]

    return tokens

# Create sample classification data
texts = [
    "this movie is great",
    "terrible book very bad",
    "amazing product love this",
    "awful movie hate this",
    "good book this is",
    "bad product terrible"
]
labels = [1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

# Tokenize
tokenized_texts = [simple_tokenize(text, simple_vocab) for text in texts]
X_clf = torch.LongTensor(tokenized_texts)
y_clf = torch.LongTensor(labels)

print(f"Sample tokenized text: {tokenized_texts[0]}")
print(f"Original text: '{texts[0]}'")
print(f"Classification data shape: {X_clf.shape}")

# Create small Transformer classifier
classifier = TransformerClassifier(
    vocab_size=len(simple_vocab),
    d_model=32,
    num_heads=4,
    d_ff=128,
    num_layers=2,
    num_classes=2,
    max_len=10
)

# Test forward pass
logits, attention_weights = classifier(X_clf)
probabilities = F.softmax(logits, dim=1)

print(f"\nClassifier output shape: {logits.shape}")
print(f"Sample probabilities: {probabilities[:3]}")
print(f"Predicted classes: {probabilities.argmax(dim=1)}")
print(f"True labels: {y_clf}")
print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")

# %% [markdown]
# ## Understanding Pre-trained Models
#
# ### Mathematical Foundation of Transfer Learning in NLP
#
# **Pre-trained Language Models** learn rich representations from large-scale unlabeled text:
#
# **Training Paradigm:**
# 1. **Pre-training**: Learn general language representations
# 2. **Fine-tuning**: Adapt to specific downstream tasks
#
# **Pre-training Objectives:**
#
# **1. Masked Language Modeling (BERT):**
# $$L_{\text{MLM}} = -\mathbb{E}_{\mathbf{x} \sim D} \left[ \sum_{i \in \mathcal{M}} \log P(x_i | \mathbf{x}_{\setminus \mathcal{M}}) \right]$$
#
# where $\mathcal{M}$ is the set of masked positions.
#
# **2. Next Sentence Prediction (BERT):**
# $$L_{\text{NSP}} = -\mathbb{E}_{(A,B)} [\log P(IsNext(B|A))]$$
#
# **3. Autoregressive Language Modeling (GPT):**
# $$L_{\text{AR}} = -\mathbb{E}_{\mathbf{x}} \left[ \sum_{t=1}^T \log P(x_t | \mathbf{x}_{<t}) \right]$$
#
# **Model Scale Mathematics:**
#
# **Parameter Scaling Laws:**
# For Transformer models, performance scales as:
# $$L(N) \propto N^{-\alpha}$$
#
# where $N$ is parameter count and $\alpha \approx 0.076$.
#
# **BERT Model Configurations:**
# - **BERT-base**: $L=12$, $H=768$, $A=12$, Parameters=110M
# - **BERT-large**: $L=24$, $H=1024$, $A=16$, Parameters=340M
#
# **Parameter Count Formula:**
# $$\text{Params} = V \times H + L \times (4H^2 + 4H \times \text{FFN\_size}) + \text{Output\_layer}$$
#
# **Computational Requirements:**
#
# **Training FLOPs:**
# For sequence length $n$, batch size $B$, and $T$ training steps:
# $$\text{FLOPs} \approx 6NBT \times (n^2H + nH^2)$$
#
# **Inference FLOPs:**
# $$\text{FLOPs} \approx 2N \times (n^2H + nH^2)$$
#
# **Knowledge Transfer Mathematics:**
#
# **Representation Similarity:**
# Measure similarity between pre-trained and fine-tuned representations:
# $$\text{CKA}(\mathbf{X}, \mathbf{Y}) = \frac{\text{tr}(\mathbf{K}_X \mathbf{K}_Y)}{\sqrt{\text{tr}(\mathbf{K}_X^2) \text{tr}(\mathbf{K}_Y^2)}}$$
#
# where $\mathbf{K}_X = \mathbf{X}\mathbf{X}^T$ is the kernel matrix.
#
# **Feature Transferability:**
# Lower layers learn general features, upper layers learn task-specific features:
# $$\mathbf{h}_l = f_l(\mathbf{h}_{l-1}; \boldsymbol{\theta}_l^{\text{general}} + \Delta\boldsymbol{\theta}_l^{\text{task}})$$

# %%
# Simulate working with pre-trained models (conceptual)
# In practice, you would use libraries like Transformers by Hugging Face

class PretrainedModelSimulator:
    """Simulates the interface of pre-trained models like BERT"""

    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.vocab_size = 30522  # BERT vocab size
        self.hidden_size = 768   # BERT hidden size
        self.num_layers = 12     # BERT layers
        self.num_heads = 12      # BERT attention heads

        # Special tokens
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 100,
            '[CLS]': 101,  # Classification token
            '[SEP]': 102,  # Separator token
            '[MASK]': 103  # Mask token for MLM
        }

        print(f"Simulated {model_name}:")
        print(f"  Vocabulary size: {self.vocab_size:,}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Layers: {self.num_layers}")
        print(f"  Attention heads: {self.num_heads}")

    def tokenize(self, text):
        """Simulate tokenization (simplified)"""
        # In reality, BERT uses WordPiece tokenization
        words = text.lower().split()

        # Add special tokens
        tokens = [self.special_tokens['[CLS]']]

        for word in words:
            # Simulate subword tokenization
            if len(word) > 6:
                # Split long words
                tokens.extend([hash(word[:3]) % 1000 + 1000,
                              hash(word[3:]) % 1000 + 2000])
            else:
                tokens.append(hash(word) % 1000 + 3000)

        tokens.append(self.special_tokens['[SEP]'])
        return tokens

    def get_embeddings(self, tokens):
        """Simulate getting contextual embeddings"""
        # In practice, this would be the output of the transformer
        batch_size = 1
        seq_len = len(tokens)

        # Simulate contextual embeddings
        embeddings = torch.randn(batch_size, seq_len, self.hidden_size)
        return embeddings

    def estimate_parameters(self):
        """Estimate number of parameters"""
        # Rough estimation
        embedding_params = self.vocab_size * self.hidden_size

        # Each transformer layer
        attention_params = 4 * self.hidden_size * self.hidden_size  # Q, K, V, O projections
        ffn_params = 2 * self.hidden_size * (4 * self.hidden_size)  # Two linear layers
        layer_params = attention_params + ffn_params

        total_params = embedding_params + (self.num_layers * layer_params)
        return total_params

# Demonstrate pre-trained model concepts
bert_sim = PretrainedModelSimulator("bert-base-uncased")

# Example usage
sample_text = "The transformer architecture revolutionized natural language processing"
tokens = bert_sim.tokenize(sample_text)
embeddings = bert_sim.get_embeddings(tokens)

print(f"\nSample text: '{sample_text}'")
print(f"Tokenized length: {len(tokens)}")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Estimated parameters: {bert_sim.estimate_parameters():,}")

print("\nCommon Pre-trained Models:")
models_info = {
    'BERT-base': {'params': '110M', 'layers': 12, 'heads': 12, 'hidden': 768},
    'BERT-large': {'params': '340M', 'layers': 24, 'heads': 16, 'hidden': 1024},
    'GPT-2': {'params': '117M-1.5B', 'layers': '12-48', 'heads': '12-25', 'hidden': '768-1600'},
    'RoBERTa': {'params': '125M-355M', 'layers': '12-24', 'heads': '12-16', 'hidden': '768-1024'},
    'T5-base': {'params': '220M', 'layers': 12, 'heads': 12, 'hidden': 768}
}

for model, info in models_info.items():
    print(f"  {model}: {info['params']} parameters, {info['layers']} layers")

# %% [markdown]
# ## Fine-tuning Strategies
#
# ### Mathematical Framework for Model Adaptation
#
# **Fine-tuning** adapts pre-trained models to downstream tasks through various mathematical strategies:
#
# **1. Full Fine-tuning:**
# Update all parameters $\boldsymbol{\theta}$:
# $$\boldsymbol{\theta}_{\text{new}} = \boldsymbol{\theta}_{\text{pretrained}} - \eta \nabla_{\boldsymbol{\theta}} L_{\text{task}}$$
#
# **2. Feature Extraction:**
# Freeze encoder, train only classifier:
# $$\boldsymbol{\theta}_{\text{encoder}} \text{ fixed}, \quad \boldsymbol{\theta}_{\text{classifier}} \text{ trainable}$$
#
# **3. Layer-wise Learning Rates:**
# Different learning rates for different layers:
# $$\eta_l = \eta_0 \cdot \gamma^{L-l}$$
#
# where $l$ is layer index and $\gamma < 1$ (lower layers get smaller LR).
#
# **4. Gradual Unfreezing:**
# Progressively unfreeze layers during training:
# $$\boldsymbol{\theta}_{\text{trainable}}^{(t)} = \boldsymbol{\theta}_{\text{trainable}}^{(t-1)} \cup \{\boldsymbol{\theta}_{\text{layer}_{k(t)}}\}$$
#
# **Parameter-Efficient Fine-tuning:**
#
# **LoRA (Low-Rank Adaptation):**
# Approximate weight updates with low-rank decomposition:
# $$\mathbf{W}_{\text{new}} = \mathbf{W}_0 + \Delta\mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}$$
#
# where $\mathbf{A} \in \mathbb{R}^{r \times d}$, $\mathbf{B} \in \mathbb{R}^{d \times r}$, and $r \ll d$.
#
# **Parameter Reduction:**
# Original parameters: $d^2$
# LoRA parameters: $2rd$ (typically $r = 8$, so $16d \ll d^2$)
#
# **Adapter Layers:**
# Insert small feedforward networks between Transformer layers:
# $$\mathbf{h}_{\text{adapter}} = \mathbf{h} + f(\mathbf{h}; \boldsymbol{\theta}_{\text{adapter}})$$
#
# where $f$ is a bottleneck network: $d \rightarrow d/k \rightarrow d$.
#
# **Prefix Tuning:**
# Prepend learnable prefix vectors:
# $$[\mathbf{P}_K; \mathbf{K}], \quad [\mathbf{P}_V; \mathbf{V}]$$
#
# **BitFit:**
# Fine-tune only bias terms:
# $$\boldsymbol{\theta}_{\text{trainable}} = \{\mathbf{b} : \mathbf{b} \text{ is bias in model}\}$$
#
# **Fine-tuning Stability:**
#
# **Learning Rate Scheduling:**
# $$\eta(t) = \eta_0 \left(1 - \frac{t}{T}\right) \quad \text{(linear decay)}$$
# $$\eta(t) = \eta_0 \cos\left(\frac{\pi t}{2T}\right) \quad \text{(cosine decay)}$$
#
# **Warmup Strategy:**
# $$\eta(t) = \begin{cases}
# \eta_0 \frac{t}{T_{\text{warmup}}} & \text{if } t \leq T_{\text{warmup}} \\
# \eta_{\text{scheduled}}(t) & \text{otherwise}
# \end{cases}$$
#
# **Regularization Mathematics:**
# $$L_{\text{total}} = L_{\text{task}} + \lambda_1 \|\boldsymbol{\theta} - \boldsymbol{\theta}_0\|_2^2 + \lambda_2 \|\boldsymbol{\theta}\|_2^2$$
#
# First term prevents catastrophic forgetting, second term prevents overfitting.

# %%
# Fine-tuning strategies demonstration
class FineTuningStrategies:
    """Demonstrate different fine-tuning approaches"""

    def __init__(self, pretrained_model):
        self.pretrained_model = pretrained_model

    def full_fine_tuning(self, num_classes):
        """Full fine-tuning: update all parameters"""
        print("Strategy 1: Full Fine-tuning")
        print("- All parameters are updated")
        print("- Requires significant computational resources")
        print("- Best performance for sufficient data")

        # Add classification head
        classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.pretrained_model.d_model, num_classes)
        )

        # All parameters trainable
        total_params = sum(p.numel() for p in self.pretrained_model.parameters())
        trainable_params = total_params

        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable percentage: 100%")

        return classifier

    def feature_extraction(self, num_classes):
        """Feature extraction: freeze pre-trained weights"""
        print("\nStrategy 2: Feature Extraction")
        print("- Pre-trained weights are frozen")
        print("- Only classification head is trained")
        print("- Much faster training")

        # Freeze all pretrained parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Add trainable classification head
        classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.pretrained_model.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

        total_params = sum(p.numel() for p in self.pretrained_model.parameters())
        trainable_params = sum(p.numel() for p in classifier.parameters())

        print(f"  Total parameters: {total_params + trainable_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable percentage: {trainable_params/(total_params + trainable_params)*100:.2f}%")

        return classifier

    def gradual_unfreezing(self, num_classes, layers_to_unfreeze=2):
        """Gradual unfreezing: unfreeze top layers"""
        print(f"\nStrategy 3: Gradual Unfreezing (top {layers_to_unfreeze} layers)")
        print("- Freeze most layers, unfreeze top few layers")
        print("- Compromise between speed and performance")
        print("- Good for medium-sized datasets")

        # Freeze all parameters first
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Unfreeze top layers
        unfrozen_params = 0
        for layer in self.pretrained_model.layers[-layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
                unfrozen_params += param.numel()

        # Add classification head
        classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.pretrained_model.d_model, num_classes)
        )

        total_params = sum(p.numel() for p in self.pretrained_model.parameters())
        trainable_params = unfrozen_params + sum(p.numel() for p in classifier.parameters())

        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable percentage: {trainable_params/total_params*100:.2f}%")

        return classifier

    def learning_rate_scheduling(self):
        """Demonstrate learning rate strategies for fine-tuning"""
        print("\nLearning Rate Strategies:")
        print("1. Lower LR for pre-trained layers (1e-5 to 5e-5)")
        print("2. Higher LR for new layers (1e-4 to 1e-3)")
        print("3. Warmup + decay schedule")
        print("4. Discriminative learning rates (different LR per layer)")

        # Example parameter groups
        pretrained_params = list(self.pretrained_model.parameters())

        param_groups = [
            {'params': pretrained_params, 'lr': 2e-5},
            # {'params': classifier.parameters(), 'lr': 1e-4}  # Would add classifier params
        ]

        print(f"\nExample parameter groups:")
        print(f"  Pre-trained layers: {len(pretrained_params)} parameter tensors, LR=2e-5")
        print(f"  New classifier: would use LR=1e-4")

        return param_groups

# Demonstrate fine-tuning strategies
# Use our previously created transformer
finetuning = FineTuningStrategies(transformer)

# Show different strategies
num_classes = 5
classifier1 = finetuning.full_fine_tuning(num_classes)
classifier2 = finetuning.feature_extraction(num_classes)
classifier3 = finetuning.gradual_unfreezing(num_classes, layers_to_unfreeze=2)
param_groups = finetuning.learning_rate_scheduling()

print("\nFine-tuning Best Practices:")
print("1. Start with feature extraction for small datasets")
print("2. Use full fine-tuning for large, task-specific datasets")
print("3. Apply gradual unfreezing for medium datasets")
print("4. Use lower learning rates for pre-trained parameters")
print("5. Monitor validation performance to prevent overfitting")
print("6. Consider early stopping and learning rate scheduling")

# %% [markdown]
# ## Working with Hugging Face Transformers
#
# ### Mathematical Framework for Production NLP Systems
#
# **Hugging Face Transformers** provides standardized interfaces for pre-trained models with mathematical foundations:
#
# **Tokenization Mathematics:**
#
# **WordPiece/BPE Algorithm:**
# 1. Initialize vocabulary with characters: $V_0 = \{c_1, c_2, \ldots, c_n\}$
# 2. Iteratively merge most frequent pairs:
#    $$V_{t+1} = V_t \cup \{\arg\max_{(x,y) \in V_t^2} \text{count}(xy)\}$$
#
# **Subword Regularization:**
# Sample from multiple tokenizations with probability:
# $$P(\mathbf{x} | s) = \prod_{i} \frac{\exp(\text{score}(x_i))}{\sum_{x' \in \text{candidates}} \exp(\text{score}(x'))}$$
#
# **Model Loading Mathematics:**
#
# **Weight Initialization:**
# Pre-trained weights $\boldsymbol{\theta}_{\text{pretrained}}$ loaded with:
# - **Embedding layers**: Direct mapping from vocabulary
# - **Transformer layers**: Exact architecture match required
# - **Task heads**: Random initialization for new tasks
#
# **Architecture Compatibility:**
# Model dimensions must satisfy:
# - $d_{\text{model}}$ consistent across layers
# - Attention heads: $h$ divides $d_{\text{model}}$
# - Vocabulary sizes match for embeddings
#
# **Training Optimization:**
#
# **AdamW Optimizer:**
# $$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t$$
# $$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2$$
# $$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta \left(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda \boldsymbol{\theta}_{t-1}\right)$$
#
# **Learning Rate Scheduling:**
# Linear warmup followed by linear decay:
# $$\eta(t) = \begin{cases}
# \eta_{\max} \frac{t}{t_{\text{warmup}}} & \text{if } t \leq t_{\text{warmup}} \\
# \eta_{\max} \frac{t_{\text{total}} - t}{t_{\text{total}} - t_{\text{warmup}}} & \text{otherwise}
# \end{cases}$$
#
# **Pipeline Architecture:**
#
# **Text Classification Pipeline:**
# $$\text{Input} \xrightarrow{\text{Tokenize}} \text{Token IDs} \xrightarrow{\text{Model}} \text{Logits} \xrightarrow{\text{Softmax}} \text{Probabilities}$$
#
# **Question Answering Pipeline:**
# $$P(\text{start}=i) = \frac{\exp(\mathbf{w}_s^T \mathbf{h}_i)}{\sum_j \exp(\mathbf{w}_s^T \mathbf{h}_j)}$$
# $$P(\text{end}=j) = \frac{\exp(\mathbf{w}_e^T \mathbf{h}_j)}{\sum_k \exp(\mathbf{w}_e^T \mathbf{h}_k)}$$
#
# **Memory and Compute Scaling:**
#
# **Model Size Estimation:**
# For model with $N$ parameters:
# - **FP32**: $4N$ bytes
# - **FP16**: $2N$ bytes
# - **INT8**: $N$ bytes
#
# **Inference Memory:**
# $$\text{Memory} = \text{Model weights} + \text{Activations} + \text{KV cache}$$
#
# **Batch Processing Efficiency:**
# $$\text{Throughput} = \frac{B \times L}{\text{Inference time}}$$
#
# Trade-off between batch size $B$, sequence length $L$, and memory constraints.

# %%
# Simulated Hugging Face workflow (conceptual)
# In practice, you would: pip install transformers

class HuggingFaceWorkflowDemo:
    """Demonstrate typical Hugging Face patterns"""

    def __init__(self):
        print("Hugging Face Transformers Workflow:")
        print("\n# Installation:")
        print("pip install transformers torch datasets")

    def show_model_loading(self):
        """Show how to load pre-trained models"""
        print("\n# Loading Pre-trained Models:")
        print("from transformers import AutoModel, AutoTokenizer, AutoConfig")
        print("")
        print("# Load model and tokenizer")
        print("model_name = 'bert-base-uncased'")
        print("tokenizer = AutoTokenizer.from_pretrained(model_name)")
        print("model = AutoModel.from_pretrained(model_name)")
        print("")
        print("# For specific tasks")
        print("from transformers import AutoModelForSequenceClassification")
        print("classifier = AutoModelForSequenceClassification.from_pretrained(")
        print("    model_name, num_labels=2")

    def show_tokenization(self):
        """Show tokenization patterns"""
        print("\n# Tokenization:")
        print("text = 'Hello, how are you?'")
        print("")
        print("# Basic tokenization")
        print("tokens = tokenizer(text, return_tensors='pt')")
        print("# Returns: {'input_ids': tensor, 'attention_mask': tensor}")
        print("")
        print("# Batch processing")
        print("texts = ['First text', 'Second text']")
        print("batch = tokenizer(texts, padding=True, truncation=True, ")
        print("                  max_length=512, return_tensors='pt')")

    def show_training_loop(self):
        """Show typical training patterns"""
        print("\n# Training Loop:")
        print("from transformers import AdamW, get_linear_schedule_with_warmup")
        print("")
        print("# Setup optimizer")
        print("optimizer = AdamW(model.parameters(), lr=2e-5)")
        print("")
        print("# Learning rate scheduler")
        print("scheduler = get_linear_schedule_with_warmup(")
        print("    optimizer, num_warmup_steps=100, num_training_steps=1000")
        print("")
        print("# Training step")
        print("outputs = model(**batch)")
        print("loss = outputs.loss")
        print("loss.backward()")
        print("optimizer.step()")
        print("scheduler.step()")
        print("optimizer.zero_grad()")

    def show_popular_models(self):
        """Show popular pre-trained models"""
        print("\n# Popular Pre-trained Models:")

        models = {
            "Classification": [
                "bert-base-uncased",
                "roberta-base",
                "distilbert-base-uncased",
                "albert-base-v2"
            ],
            "Generation": [
                "gpt2",
                "gpt2-medium",
                "microsoft/DialoGPT-medium",
                "facebook/blenderbot-400M-distill"
            ],
            "Question Answering": [
                "distilbert-base-cased-distilled-squad",
                "bert-large-uncased-whole-word-masking-finetuned-squad",
                "roberta-base-squad2"
            ],
            "Multilingual": [
                "bert-base-multilingual-cased",
                "xlm-roberta-base",
                "distilbert-base-multilingual-cased"
            ]
        }

        for category, model_list in models.items():
            print(f"\n{category}:")
            for model in model_list:
                print(f"  - {model}")

    def show_pipeline_api(self):
        """Show the simple pipeline API"""
        print("\n# Pipeline API (Quickstart):")
        print("from transformers import pipeline")
        print("")
        print("# Text classification")
        print("classifier = pipeline('sentiment-analysis')")
        print("result = classifier('I love this movie!')")
        print("")
        print("# Question answering")
        print("qa = pipeline('question-answering')")
        print("answer = qa(question='What is AI?', context='AI is...')")
        print("")
        print("# Text generation")
        print("generator = pipeline('text-generation', model='gpt2')")
        print("text = generator('The future of AI is', max_length=50)")
        print("")
        print("# Named entity recognition")
        print("ner = pipeline('ner', aggregation_strategy='simple')")
        print("entities = ner('John works at OpenAI in San Francisco')")

# Run the demonstration
demo = HuggingFaceWorkflowDemo()
demo.show_model_loading()
demo.show_tokenization()
demo.show_training_loop()
demo.show_popular_models()
demo.show_pipeline_api()

print("\n" + "="*60)
print("Key Advantages of Transformers:")
print("1. Parallelizable training (unlike RNNs)")
print("2. Better handling of long-range dependencies")
print("3. Transfer learning capabilities")
print("4. State-of-the-art performance on many NLP tasks")
print("5. Rich ecosystem of pre-trained models")

print("\nChallenges:")
print("1. Quadratic memory complexity with sequence length")
print("2. Large computational requirements")
print("3. Need for large datasets for training from scratch")
print("4. Limited context window (though improving)")