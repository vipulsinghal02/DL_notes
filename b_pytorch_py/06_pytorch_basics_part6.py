# %% [markdown]
# # PyTorch Basics Part 6: Recurrent Neural Networks and NLP
#
# Introduction to sequence modeling, RNNs, LSTMs, and natural language processing with mathematical foundations
#
# ## Mathematical Framework for Sequence Modeling
#
# **Recurrent Neural Networks (RNNs)** process sequential data by maintaining hidden states that evolve over time, enabling models to capture temporal dependencies and variable-length patterns.
#
# ### Core Mathematical Concepts
#
# **1. Sequential Data Representation:**
# A sequence $\mathbf{X} = (\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T)$ where $\mathbf{x}_t \in \mathbb{R}^d$ at time step $t$.
#
# **2. Temporal Dependencies:**
# Current output depends on current input AND previous states:
# $$\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1})$$
# $$\mathbf{y}_t = g(\mathbf{h}_t)$$
#
# **3. RNN Recurrence Relation:**
# $$\mathbf{h}_t = \tanh(\mathbf{W}_{xh}\mathbf{x}_t + \mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{b}_h)$$
# $$\mathbf{y}_t = \mathbf{W}_{hy}\mathbf{h}_t + \mathbf{b}_y$$
#
# Where:
# - $\mathbf{W}_{xh} \in \mathbb{R}^{H \times D}$: Input-to-hidden weights
# - $\mathbf{W}_{hh} \in \mathbb{R}^{H \times H}$: Hidden-to-hidden (recurrent) weights
# - $\mathbf{W}_{hy} \in \mathbb{R}^{O \times H}$: Hidden-to-output weights
# - $\mathbf{h}_t \in \mathbb{R}^H$: Hidden state at time $t$
#
# **4. Backpropagation Through Time (BPTT):**
# Gradients flow backward through the temporal dimension:
# $$\frac{\partial L}{\partial \mathbf{h}_t} = \frac{\partial L}{\partial \mathbf{y}_t} \frac{\partial \mathbf{y}_t}{\partial \mathbf{h}_t} + \frac{\partial L}{\partial \mathbf{h}_{t+1}} \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}$$
#
# **5. Vanishing Gradient Problem:**
# Gradient magnitude decays exponentially with sequence length:
# $$\left\|\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k}\right\| \leq \left\|\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}\right\| \times \cdots \times \left\|\frac{\partial \mathbf{h}_{k+1}}{\partial \mathbf{h}_k}\right\| \propto \gamma^{t-k}$$
#
# Where $\gamma < 1$ leads to vanishing gradients for long sequences.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import matplotlib.pyplot as plt
import string
import re
from collections import Counter, defaultdict
import random

# %% [markdown]
# ## Understanding Sequences and RNN Fundamentals
#
# ### Mathematical Foundation of Sequential Processing
#
# **Sequence modeling** captures patterns in temporally or spatially ordered data through mathematical frameworks that preserve dependency structure:
#
# **Markov Property:**
# $$P(\mathbf{x}_t | \mathbf{x}_{1:t-1}) = P(\mathbf{x}_t | \mathbf{x}_{t-k:t-1})$$
# Many sequences exhibit limited-range dependencies.
#
# **Autoregressive Models:**
# $$P(\mathbf{x}_{1:T}) = \prod_{t=1}^T P(\mathbf{x}_t | \mathbf{x}_{1:t-1})$$
#
# **RNN as Universal Approximator:**
# RNNs can approximate any measurable sequence-to-sequence mapping given sufficient hidden units and proper activation functions.
#
# **Memory Capacity:**
# Information storage capacity of RNN hidden state:
# $$I(\mathbf{h}_t; \mathbf{x}_{1:t}) \leq H \log_2(|\mathcal{A}|)$$
# Where $H$ is hidden size and $\mathcal{A}$ is activation range.
#
# **Sequence Types:**
# - **One-to-one**: $f: \mathbb{R}^d \rightarrow \mathbb{R}^o$ (standard feedforward)
# - **One-to-many**: $f: \mathbb{R}^d \rightarrow (\mathbb{R}^o)^T$ (image captioning)
# - **Many-to-one**: $f: (\mathbb{R}^d)^T \rightarrow \mathbb{R}^o$ (sentiment analysis)
# - **Many-to-many**: $f: (\mathbb{R}^d)^T \rightarrow (\mathbb{R}^o)^T$ (machine translation)
#
# **Computational Complexity:**
# - **Time**: $O(T \cdot H^2 + T \cdot H \cdot D)$ per sequence
# - **Space**: $O(T \cdot H)$ for storing all hidden states (BPTT)

# %%
# Simple sequence data example
# Let's create a toy dataset: predicting the next number in a sequence
def generate_sequence_data(n_samples=1000, seq_length=10):
    """Generate simple arithmetic sequences"""
    sequences = []
    targets = []

    for _ in range(n_samples):
        # Create arithmetic sequence: start + i*step
        start = random.randint(1, 10)
        step = random.randint(1, 5)

        sequence = [start + i * step for i in range(seq_length)]
        target = start + seq_length * step  # Next number in sequence

        sequences.append(sequence)
        targets.append(target)

    return torch.FloatTensor(sequences), torch.FloatTensor(targets)

# Generate data
X, y = generate_sequence_data(1000, 5)
print(f"Sequence shape: {X.shape}")  # [batch_size, sequence_length]
print(f"Target shape: {y.shape}")

# Show examples
for i in range(3):
    seq = X[i].tolist()
    target = y[i].item()
    print(f"Sequence: {seq} -> Next: {target}")

# %% [markdown]
# ## Basic RNN Implementation
#
# ### Mathematical Mechanics of Recurrent Computation
#
# **RNN Forward Pass Mathematics:**
#
# For each time step $t$, the RNN computes:
# $$\mathbf{h}_t = \tanh(\mathbf{W}_{xh}\mathbf{x}_t + \mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{b}_h)$$
#
# **Matrix Dimensions:**
# - Input: $\mathbf{x}_t \in \mathbb{R}^{D}$
# - Hidden state: $\mathbf{h}_t \in \mathbb{R}^{H}$
# - Input weights: $\mathbf{W}_{xh} \in \mathbb{R}^{H \times D}$
# - Recurrent weights: $\mathbf{W}_{hh} \in \mathbb{R}^{H \times H}$
# - Bias: $\mathbf{b}_h \in \mathbb{R}^{H}$
#
# **Activation Function Role:**
# $\tanh$ activation provides:
# - **Bounded output**: $\tanh(x) \in [-1, 1]$
# - **Non-zero gradient**: $\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$
# - **Zero-centered**: Helps with gradient flow
#
# **Unrolled Computation:**
# $$\mathbf{h}_1 = \tanh(\mathbf{W}_{xh}\mathbf{x}_1 + \mathbf{b}_h)$$
# $$\mathbf{h}_2 = \tanh(\mathbf{W}_{xh}\mathbf{x}_2 + \mathbf{W}_{hh}\mathbf{h}_1 + \mathbf{b}_h)$$
# $$\mathbf{h}_3 = \tanh(\mathbf{W}_{xh}\mathbf{x}_3 + \mathbf{W}_{hh}\mathbf{h}_2 + \mathbf{b}_h)$$
# $$\vdots$$
#
# **Parameter Sharing:**
# Same weight matrices $\mathbf{W}_{xh}, \mathbf{W}_{hh}$ used at all time steps:
# - **Benefits**: Fewer parameters, translation invariance
# - **Limitations**: Fixed capacity regardless of sequence length
#
# **Hidden State Interpretation:**
# $\mathbf{h}_t$ serves as a "memory" vector encoding information from $\mathbf{x}_{1:t}$:
# $$\mathbf{h}_t = f(\mathbf{x}_{1:t}; \boldsymbol{\theta})$$

# %%
# Manual RNN implementation to understand the mechanics
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        # RNN weights
        self.i2h = nn.Linear(input_size, hidden_size)  # input to hidden
        self.h2h = nn.Linear(hidden_size, hidden_size)  # hidden to hidden
        self.h2o = nn.Linear(hidden_size, output_size)  # hidden to output

    def forward(self, x):
        batch_size, seq_length = x.size(0), x.size(1)
        hidden = torch.zeros(batch_size, self.hidden_size)

        # Process sequence step by step
        for t in range(seq_length):
            # Get input at time t
            input_t = x[:, t].unsqueeze(1)  # [batch_size, 1]

            # Update hidden state: h_t = tanh(W_ih * x_t + W_hh * h_{t-1})
            hidden = torch.tanh(self.i2h(input_t) + self.h2h(hidden))

        # Output prediction from final hidden state
        output = self.h2o(hidden)
        return output.squeeze()

# Test the manual RNN
manual_rnn = SimpleRNN(input_size=1, hidden_size=20, output_size=1)
test_input = X[:5].unsqueeze(-1)  # Add feature dimension
output = manual_rnn(test_input)
print(f"Manual RNN output shape: {output.shape}")
print(f"Sample predictions: {output[:3]}")
print(f"Actual targets: {y[:3]}")

# %% [markdown]
# ## Using PyTorch's Built-in RNN Layers
#
# ### Mathematical Optimization in PyTorch RNNs
#
# **PyTorch RNN Optimizations:**
#
# **1. Efficient Matrix Operations:**
# PyTorch combines operations for computational efficiency:
# $$[\mathbf{W}_{xh} \; \mathbf{W}_{hh}] \begin{bmatrix} \mathbf{x}_t \\ \mathbf{h}_{t-1} \end{bmatrix} + \mathbf{b}_h$$
#
# **2. Batch Processing:**
# Process multiple sequences simultaneously:
# $$\mathbf{H}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xh}^T + \mathbf{H}_{t-1} \mathbf{W}_{hh}^T + \mathbf{b}_h)$$
#
# Where $\mathbf{X}_t \in \mathbb{R}^{B \times D}$, $\mathbf{H}_t \in \mathbb{R}^{B \times H}$ for batch size $B$.
#
# **3. Multi-layer RNNs:**
# Stack RNN layers for increased representational capacity:
# $$\mathbf{h}_t^{(1)} = \text{RNN}^{(1)}(\mathbf{x}_t, \mathbf{h}_{t-1}^{(1)})$$
# $$\mathbf{h}_t^{(2)} = \text{RNN}^{(2)}(\mathbf{h}_t^{(1)}, \mathbf{h}_{t-1}^{(2)})$$
# $$\vdots$$
# $$\mathbf{h}_t^{(L)} = \text{RNN}^{(L)}(\mathbf{h}_t^{(L-1)}, \mathbf{h}_{t-1}^{(L)})$$
#
# **4. Parameter Count Analysis:**
# For $L$-layer RNN with dimensions $(D, H, O)$:
# - Layer 1: $D \times H + H \times H + H = H(D + H + 1)$
# - Layers 2 to $L$: $(L-1) \times H(H + H + 1) = (L-1) \times H(2H + 1)$
# - Output layer: $H \times O + O$
#
# **Total**: $H(D + H + 1) + (L-1)H(2H + 1) + O(H + 1)$
#
# **5. Computational Graph:**
# PyTorch automatically builds computational graph for backpropagation:
# $$\frac{\partial L}{\partial \mathbf{W}} = \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}}$$

# %%
# Using PyTorch's built-in RNN
class PyTorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PyTorchRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # RNN returns (output, hidden) where output contains all hidden states
        rnn_out, hidden = self.rnn(x)

        # Use the last time step's output
        last_output = rnn_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction.squeeze()

# Create and test PyTorch RNN
pytorch_rnn = PyTorchRNN(input_size=1, hidden_size=20, num_layers=2, output_size=1)
output = pytorch_rnn(test_input)
print(f"PyTorch RNN output shape: {output.shape}")

# Count parameters
total_params = sum(p.numel() for p in pytorch_rnn.parameters())
print(f"Total parameters: {total_params}")

# %% [markdown]
# ## LSTM: Solving the Vanishing Gradient Problem
#
# ### Mathematical Architecture of Long Short-Term Memory
#
# **LSTM** addresses vanishing gradients through gating mechanisms that control information flow:
#
# **LSTM Cell Equations:**
# $$\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$ (Forget gate)
# $$\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$ (Input gate)
# $$\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_C \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C)$$ (Candidate values)
# $$\mathbf{C}_t = \mathbf{f}_t * \mathbf{C}_{t-1} + \mathbf{i}_t * \tilde{\mathbf{C}}_t$$ (Cell state)
# $$\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$ (Output gate)
# $$\mathbf{h}_t = \mathbf{o}_t * \tanh(\mathbf{C}_t)$$ (Hidden state)
#
# **Gate Functions:**
# - **Forget gate** $\mathbf{f}_t$: Decides what information to discard from cell state
# - **Input gate** $\mathbf{i}_t$: Controls which values to update in cell state
# - **Output gate** $\mathbf{o}_t$: Controls which parts of cell state to output
#
# **Gradient Flow Analysis:**
# Cell state gradient: $\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} = \mathbf{f}_t$
#
# Since $\mathbf{f}_t \in [0,1]$, gradient can flow unchanged when $\mathbf{f}_t \approx 1$.
#
# **Long-term Dependencies:**
# $$\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_k} = \prod_{j=k+1}^t \mathbf{f}_j$$
#
# Unlike vanilla RNN where gradients always multiply by $\tanh'$ derivatives.
#
# **GRU (Gated Recurrent Unit) - Simplified LSTM:**
# $$\mathbf{r}_t = \sigma(\mathbf{W}_r \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t])$$ (Reset gate)
# $$\mathbf{z}_t = \sigma(\mathbf{W}_z \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t])$$ (Update gate)
# $$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} \cdot [\mathbf{r}_t * \mathbf{h}_{t-1}, \mathbf{x}_t])$$ (Candidate)
# $$\mathbf{h}_t = (1 - \mathbf{z}_t) * \mathbf{h}_{t-1} + \mathbf{z}_t * \tilde{\mathbf{h}}_t$$ (Hidden state)
#
# **Parameter Efficiency:**
# - **LSTM**: 4 gates × (input + hidden + bias) = $4(D + H + 1)H$
# - **GRU**: 3 gates × (input + hidden + bias) = $3(D + H + 1)H$
#
# GRU has ~25% fewer parameters than LSTM.

# %%
# LSTM implementation
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM returns (output, (hidden, cell))
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use last time step
        prediction = self.fc(lstm_out[:, -1, :])
        return prediction.squeeze()

# GRU: Simplified LSTM
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, hidden = self.gru(x)
        prediction = self.fc(gru_out[:, -1, :])
        return prediction.squeeze()

# Compare architectures
lstm_model = LSTMModel(1, 20, 2, 1)
gru_model = GRUModel(1, 20, 2, 1)

lstm_params = sum(p.numel() for p in lstm_model.parameters())
gru_params = sum(p.numel() for p in gru_model.parameters())

print(f"LSTM parameters: {lstm_params}")
print(f"GRU parameters: {gru_params}")
print(f"GRU is {((lstm_params - gru_params) / lstm_params * 100):.1f}% smaller")

# %% [markdown]
# ## Training RNN Models
#
# ### Mathematical Framework for RNN Training
#
# **Backpropagation Through Time (BPTT):**
#
# **Loss Function:**
# For sequence-to-sequence task:
# $$L = \frac{1}{T} \sum_{t=1}^T \ell(\mathbf{y}_t, \hat{\mathbf{y}}_t)$$
#
# **Gradient Computation:**
# $$\frac{\partial L}{\partial \mathbf{W}} = \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}}$$
#
# **Chain Rule for Recurrent Connections:**
# $$\frac{\partial L}{\partial \mathbf{h}_k} = \frac{\partial L}{\partial \mathbf{h}_T} \prod_{t=k+1}^T \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}$$
#
# **Exploding Gradient Problem:**
# When $\left\|\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}\right\| > 1$, gradients grow exponentially.
#
# **Gradient Clipping:**
# Rescale gradients when norm exceeds threshold:
# $$\mathbf{g} \leftarrow \mathbf{g} \cdot \min\left(1, \frac{\text{clip\_norm}}{\|\mathbf{g}\|}\right)$$
#
# **Truncated BPTT:**
# Process long sequences in chunks of length $K$:
# $$\frac{\partial L}{\partial \mathbf{W}} \approx \sum_{t=T-K+1}^T \frac{\partial L}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}}$$
#
# **Teacher Forcing (for sequence generation):**
# During training, use ground truth as input:
# $$\mathbf{h}_t = \text{RNN}(\mathbf{y}_{t-1}^{\text{true}}, \mathbf{h}_{t-1})$$
#
# **Exposure Bias:**
# Training/inference mismatch leads to error accumulation during generation.
#
# **Optimization Considerations:**
# - **Learning rate**: Often lower than feedforward networks (0.001-0.01)
# - **Batch size**: Memory constraints limit batch size
# - **Initialization**: Xavier/He initialization for gates, orthogonal for recurrent weights

# %%
# Prepare data for training
X_train, y_train = generate_sequence_data(8000, 5)
X_test, y_test = generate_sequence_data(2000, 5)

# Add feature dimension
X_train = X_train.unsqueeze(-1)
X_test = X_test.unsqueeze(-1)

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

# %%
# Training function for RNN models
def train_rnn_model(model, train_loader, test_loader, num_epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        # Testing
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        test_losses.append(test_loss / len(test_loader))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

    return train_losses, test_losses

# Train LSTM model
print("Training LSTM model...")
lstm_model = LSTMModel(1, 32, 2, 1)
lstm_train_losses, lstm_test_losses = train_rnn_model(lstm_model, train_loader, test_loader, 30)

# %% [markdown]
# ## Text Preprocessing and Tokenization
#
# ### Mathematical Foundations of Natural Language Processing
#
# **Text Tokenization** converts raw strings into discrete token sequences for neural network processing:
#
# **Tokenization as Mapping:**
# $$T: \Sigma^* \rightarrow \mathcal{V}^*$$
# Where $\Sigma$ is character alphabet and $\mathcal{V}$ is vocabulary set.
#
# **Vocabulary Construction:**
# Given corpus $\mathcal{C} = \{s_1, s_2, \ldots, s_N\}$:
#
# **1. Word Frequency:**
# $$f(w) = \sum_{s \in \mathcal{C}} \text{count}(w, s)$$
#
# **2. Zipf's Law:**
# Word frequency follows power law:
# $$f(w_r) \propto \frac{1}{r^\alpha}$$
# Where $r$ is rank and $\alpha \approx 1$ for natural languages.
#
# **3. Vocabulary Size vs Coverage:**
# $$\text{Coverage}(V_k) = \frac{\sum_{w \in V_k} f(w)}{\sum_{w \in \mathcal{V}} f(w)}$$
# Where $V_k$ contains top-$k$ most frequent words.
#
# **Out-of-Vocabulary (OOV) Handling:**
# **UNK token probability:**
# $$P(\text{UNK}) = \frac{\text{OOV tokens}}{\text{Total tokens}}$$
#
# **Subword Tokenization:**
# **Byte Pair Encoding (BPE)** learns subword units:
# 1. Initialize with character vocabulary
# 2. Iteratively merge most frequent pairs
# 3. Results in vocabulary balancing word boundary and subword units
#
# **Token-to-Index Mapping:**
# $$\phi: \mathcal{V} \rightarrow \{0, 1, 2, \ldots, |\mathcal{V}|-1\}$$
#
# **Sequence Representation:**
# Text $s = w_1 w_2 \cdots w_T$ becomes index sequence:
# $$\mathbf{x} = [\phi(w_1), \phi(w_2), \ldots, \phi(w_T)]$$
#
# **Special Tokens:**
# - `<PAD>` (0): Padding for batch processing
# - `<UNK>` (1): Unknown/out-of-vocabulary words
# - `<SOS>` (2): Start of sequence
# - `<EOS>` (3): End of sequence

# %%
# Text preprocessing utilities
class SimpleTokenizer:
    def __init__(self):
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.vocab_size = 4

    def preprocess_text(self, text):
        """Basic text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation except periods
        text = re.sub(r'[^\w\s\.]', ' ', text)
        # Split into words
        words = text.split()
        return words

    def build_vocab(self, texts, min_freq=2):
        """Build vocabulary from list of texts"""
        word_counts = Counter()

        for text in texts:
            words = self.preprocess_text(text)
            word_counts.update(words)

        # Add words that appear at least min_freq times
        for word, count in word_counts.items():
            if count >= min_freq and word not in self.word_to_idx:
                self.word_to_idx[word] = self.vocab_size
                self.idx_to_word[self.vocab_size] = word
                self.vocab_size += 1

        print(f"Vocabulary size: {self.vocab_size}")

    def text_to_indices(self, text):
        """Convert text to sequence of indices"""
        words = self.preprocess_text(text)
        indices = []
        for word in words:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx['<UNK>'])
        return indices

    def indices_to_text(self, indices):
        """Convert indices back to text"""
        words = [self.idx_to_word.get(idx, '<UNK>') for idx in indices]
        return ' '.join(words)

# Example text data
sample_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming how we process data.",
    "Neural networks can learn complex patterns from examples.",
    "Deep learning requires large amounts of training data.",
    "The model learns to predict the next word in a sequence."
]

# Build tokenizer
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(sample_texts, min_freq=1)

# Test tokenization
test_text = "The neural network learns patterns."
indices = tokenizer.text_to_indices(test_text)
reconstructed = tokenizer.indices_to_text(indices)

print(f"Original: {test_text}")
print(f"Indices: {indices}")
print(f"Reconstructed: {reconstructed}")

# %% [markdown]
# ## Word Embeddings
#
# ### Mathematical Theory of Distributed Word Representations
#
# **Word embeddings** map discrete tokens to continuous vector spaces that capture semantic relationships:
#
# **Embedding Function:**
# $$E: \mathcal{V} \rightarrow \mathbb{R}^d$$
# $$\mathbf{e}_w = E(w) \in \mathbb{R}^d$$
#
# **Distributional Hypothesis:**
# Words with similar contexts have similar meanings:
# $$\text{similarity}(w_i, w_j) \propto \text{similarity}(\text{context}(w_i), \text{context}(w_j))$$
#
# **Embedding Matrix:**
# $$\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}$$
# where row $i$ corresponds to embedding of word with index $i$.
#
# **One-hot to Dense Mapping:**
# $$\mathbf{e}_w = \mathbf{E}^T \mathbf{v}_w$$
# where $\mathbf{v}_w \in \{0,1\}^{|\mathcal{V}|}$ is one-hot vector.
#
# **Semantic Properties:**
# **Linear relationships** in embedding space:
# $$\mathbf{e}_{\text{king}} - \mathbf{e}_{\text{man}} + \mathbf{e}_{\text{woman}} \approx \mathbf{e}_{\text{queen}}$$
#
# **Similarity Metrics:**
# **Cosine similarity:**
# $$\cos(\mathbf{e}_i, \mathbf{e}_j) = \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\|\mathbf{e}_i\| \|\mathbf{e}_j\|}$$
#
# **Euclidean distance:**
# $$d(\mathbf{e}_i, \mathbf{e}_j) = \|\mathbf{e}_i - \mathbf{e}_j\|_2$$
#
# **Training Objectives:**
#
# **Skip-gram (Word2Vec):**
# Predict context from target word:
# $$\max \sum_{w \in \mathcal{C}} \sum_{c \in \text{context}(w)} \log P(c|w)$$
#
# **CBOW (Continuous Bag of Words):**
# Predict target from context:
# $$\max \sum_{w \in \mathcal{C}} \log P(w|\text{context}(w))$$
#
# **Initialization Strategy:**
# Random initialization: $\mathbf{e}_w \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$ with $\sigma \approx 0.1$

# %%
# Word embedding demonstration
vocab_size = tokenizer.vocab_size
embedding_dim = 50

# Create embedding layer
embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

print(f"Embedding layer shape: {embedding_layer.weight.shape}")
print(f"Each word is represented by a {embedding_dim}-dimensional vector")

# Convert sample text to embeddings
sample_indices = torch.LongTensor(indices)
embeddings = embedding_layer(sample_indices)

print(f"Input indices shape: {sample_indices.shape}")
print(f"Embeddings shape: {embeddings.shape}")

# Show that similar words can have similar embeddings (after training)
word1_idx = tokenizer.word_to_idx.get('the', 1)
word2_idx = tokenizer.word_to_idx.get('neural', 1)

word1_emb = embedding_layer(torch.LongTensor([word1_idx]))
word2_emb = embedding_layer(torch.LongTensor([word2_idx]))

# Compute cosine similarity
similarity = F.cosine_similarity(word1_emb, word2_emb)
print(f"Similarity between 'the' and 'neural': {similarity.item():.4f}")
print("(Note: similarity is random before training)")

# %% [markdown]
# ## Language Modeling: Predicting Next Words
#
# ### Mathematical Framework for Statistical Language Modeling
#
# **Language modeling** estimates probability distributions over word sequences:
#
# **Autoregressive Factorization:**
# $$P(\mathbf{w}_{1:T}) = \prod_{t=1}^T P(w_t | \mathbf{w}_{1:t-1})$$
#
# **Neural Language Model:**
# $$P(w_t | \mathbf{w}_{1:t-1}) = \text{softmax}(\mathbf{W}_{\text{out}} \mathbf{h}_t + \mathbf{b})_{w_t}$$
#
# where $\mathbf{h}_t$ encodes context $\mathbf{w}_{1:t-1}$.
#
# **Cross-Entropy Loss:**
# $$L = -\frac{1}{T} \sum_{t=1}^T \log P(w_t | \mathbf{w}_{1:t-1})$$
#
# **Perplexity:**
# Measure of how well model predicts text:
# $$\text{PPL} = \exp(L) = \exp\left(-\frac{1}{T} \sum_{t=1}^T \log P(w_t | \mathbf{w}_{1:t-1})\right)$$
#
# Lower perplexity indicates better prediction.
#
# **Teacher Forcing:**
# During training, use ground truth previous tokens:
# $$\mathbf{h}_t = f(\mathbf{e}_{w_t^*}, \mathbf{h}_{t-1})$$
#
# **Inference Strategies:**
#
# **Greedy Decoding:**
# $$\hat{w}_t = \arg\max_{w \in \mathcal{V}} P(w | \mathbf{w}_{1:t-1})$$
#
# **Beam Search:**
# Maintain top-$k$ hypotheses at each step:
# $$\text{score}(\mathbf{w}_{1:t}) = \sum_{i=1}^t \log P(w_i | \mathbf{w}_{1:i-1})$$
#
# **Sampling Methods:**
# **Temperature sampling:**
# $$P'(w_t | \mathbf{w}_{1:t-1}) = \frac{\exp(\mathbf{z}_{w_t} / \tau)}{\sum_{w'} \exp(\mathbf{z}_{w'} / \tau)}$$
#
# Lower $\tau$ makes distribution more peaked.
#
# **Top-k sampling:** Only sample from $k$ most likely tokens
# **Nucleus (top-p) sampling:** Sample from tokens with cumulative probability $\geq p$

# %%
# Language model implementation
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_size)
        lstm_out = self.dropout(lstm_out)

        # Reshape for linear layer
        batch_size, seq_len, hidden_size = lstm_out.shape
        lstm_out = lstm_out.reshape(-1, hidden_size)

        # Predict next word for each position
        output = self.fc(lstm_out)  # (batch_size * seq_len, vocab_size)
        return output.reshape(batch_size, seq_len, -1)

# Create language model
lm = LanguageModel(vocab_size, embedding_dim=32, hidden_size=64, num_layers=2)
print(f"Language model parameters: {sum(p.numel() for p in lm.parameters()):,}")

# Test forward pass
test_sequence = torch.LongTensor([[2, 4, 5, 6, 7]])  # Sample word indices
output = lm(test_sequence)
print(f"Input shape: {test_sequence.shape}")
print(f"Output shape: {output.shape}")
print(f"Output represents probability distribution over {vocab_size} words for each position")

# %% [markdown]
# ## Preparing Text Data for Training
#
# ### Mathematical Data Preparation for Sequence Models
#
# **Training Data Construction** for autoregressive language modeling:
#
# **Sliding Window Approach:**
# For sequence $\mathbf{w} = [w_1, w_2, \ldots, w_n]$, create training pairs:
# $$(\mathbf{x}_i, y_i) = ([w_1, \ldots, w_i], w_{i+1})$$
#
# **Input-Target Alignment:**
# - **Input**: $\mathbf{x} = [w_1, w_2, \ldots, w_{T-1}]$
# - **Target**: $\mathbf{y} = [w_2, w_3, \ldots, w_T]$
#
# **Sequence Padding Mathematics:**
#
# **Fixed-Length Sequences:**
# Pad sequences to maximum length $L$:
# $$\mathbf{x}_{\text{pad}} = \begin{cases}
# [\text{PAD}, \ldots, \text{PAD}, \mathbf{x}] & \text{if } |\mathbf{x}| < L \\
# \mathbf{x}[1:L] & \text{if } |\mathbf{x}| \geq L
# \end{cases}$$
#
# **Attention Masks:**
# Binary mask indicating valid positions:
# $$\mathbf{m}_i = \begin{cases}
# 1 & \text{if position } i \text{ is valid} \\
# 0 & \text{if position } i \text{ is padding}
# \end{cases}$$
#
# **Loss Masking:**
# $$L = -\frac{1}{\sum_t m_t} \sum_{t=1}^T m_t \log P(w_t | \mathbf{w}_{1:t-1})$$
#
# **Batch Processing:**
# For batch of sequences $\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(B)}\}$:
# - **Input tensor**: $\mathbf{X} \in \mathbb{Z}^{B \times L}$
# - **Target tensor**: $\mathbf{Y} \in \mathbb{Z}^{B \times L}$
# - **Mask tensor**: $\mathbf{M} \in \{0,1\}^{B \times L}$
#
# **Memory Efficiency:**
# Total memory for batch: $O(B \times L \times H)$ where $H$ is hidden size.
#
# **Data Augmentation for Text:**
# - **Random masking**: Replace tokens with `<UNK>` with probability $p$
# - **Token dropout**: Skip tokens during processing
# - **Sequence shuffling**: Permute sentence order (for document-level models)

# %%
# Prepare training data for language modeling
def create_language_modeling_data(texts, tokenizer, max_length=20):
    """Create input-target pairs for language modeling"""
    input_sequences = []
    target_sequences = []

    for text in texts:
        indices = tokenizer.text_to_indices(text)

        # Create sliding window of sequences
        for i in range(len(indices) - 1):
            # Input: words up to position i
            # Target: word at position i+1
            if i + 1 < max_length:
                input_seq = indices[:i+1]
                target_seq = indices[1:i+2]

                # Pad sequences to max_length
                while len(input_seq) < max_length:
                    input_seq.append(0)  # PAD token
                    target_seq.append(0)

                input_sequences.append(input_seq[:max_length])
                target_sequences.append(target_seq[:max_length])

    return torch.LongTensor(input_sequences), torch.LongTensor(target_sequences)

# Extend our sample texts
extended_texts = sample_texts + [
    "Artificial intelligence systems can understand natural language.",
    "Training deep networks requires careful optimization.",
    "The model generates text by predicting words sequentially.",
    "Recurrent networks process sequences one step at a time.",
    "Word embeddings capture semantic relationships between words."
]

# Rebuild vocabulary with more texts
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(extended_texts, min_freq=1)

# Create training data
X_lm, y_lm = create_language_modeling_data(extended_texts, tokenizer, max_length=15)

print(f"Language modeling data shape: {X_lm.shape}")
print(f"Target shape: {y_lm.shape}")
print(f"Number of training examples: {len(X_lm)}")

# Show example
idx = 5
input_text = tokenizer.indices_to_text(X_lm[idx].tolist())
target_text = tokenizer.indices_to_text(y_lm[idx].tolist())
print(f"\nExample {idx}:")
print(f"Input:  {input_text}")
print(f"Target: {target_text}")

# %% [markdown]
# ## Text Classification with RNNs
#
# ### Mathematical Framework for Sequence Classification
#
# **Text classification** maps variable-length sequences to discrete categories:
#
# **Classification Function:**
# $$f: (\mathcal{V})^* \rightarrow \{1, 2, \ldots, C\}$$
#
# **RNN-based Classifier Architecture:**
# 1. **Encoding**: $\mathbf{h}_{1:T} = \text{RNN}(\mathbf{e}_{w_1}, \ldots, \mathbf{e}_{w_T})$
# 2. **Aggregation**: $\mathbf{s} = g(\mathbf{h}_{1:T})$
# 3. **Classification**: $\hat{y} = \text{softmax}(\mathbf{W}\mathbf{s} + \mathbf{b})$
#
# **Sequence Representation Methods:**
#
# **Last Hidden State:**
# $$\mathbf{s} = \mathbf{h}_T$$
# Simple but may lose early information.
#
# **Mean Pooling:**
# $$\mathbf{s} = \frac{1}{T} \sum_{t=1}^T \mathbf{h}_t$$
# Equal weight to all positions.
#
# **Max Pooling:**
# $$\mathbf{s}_i = \max_{t=1}^T \mathbf{h}_{t,i}$$
# Captures most salient features.
#
# **Attention-based Aggregation:**
# $$\alpha_t = \frac{\exp(\mathbf{v}^T \tanh(\mathbf{W}\mathbf{h}_t))}{\sum_{k=1}^T \exp(\mathbf{v}^T \tanh(\mathbf{W}\mathbf{h}_k))}$$
# $$\mathbf{s} = \sum_{t=1}^T \alpha_t \mathbf{h}_t$$
#
# **Multi-class Cross-Entropy Loss:**
# $$L = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log \hat{y}_{i,c}$$
#
# **Class Imbalance Handling:**
#
# **Weighted Loss:**
# $$L = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C w_c y_{i,c} \log \hat{y}_{i,c}$$
#
# where $w_c = \frac{N}{C \cdot n_c}$ and $n_c$ is count of class $c$.
#
# **Focal Loss:**
# $$L = -\alpha_c (1-\hat{y}_{i,c})^\gamma \log \hat{y}_{i,c}$$
#
# Focuses learning on hard examples ($\gamma > 0$).

# %%
# Text classification model
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, num_layers=2):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                           batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Embed words
        embedded = self.embedding(x)

        # Process with LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)

        # Use last hidden state for classification
        final_hidden = hidden[-1]  # Last layer's hidden state

        # Apply dropout and classify
        output = self.dropout(final_hidden)
        logits = self.classifier(output)

        return logits

# Create synthetic classification data
def create_classification_data():
    """Create simple sentiment classification data"""
    positive_texts = [
        "This is amazing and wonderful.",
        "I love this fantastic product.",
        "Excellent quality and great performance.",
        "Outstanding results and very happy.",
        "Perfect solution for my needs."
    ]

    negative_texts = [
        "This is terrible and disappointing.",
        "I hate this awful product.",
        "Poor quality and bad performance.",
        "Disappointing results and very unhappy.",
        "Useless solution for my needs."
    ]

    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)  # 1=positive, 0=negative

    return texts, labels

# Prepare classification data
class_texts, class_labels = create_classification_data()

# Convert to sequences
class_sequences = []
for text in class_texts:
    indices = tokenizer.text_to_indices(text)
    # Pad to fixed length
    while len(indices) < 10:
        indices.append(0)
    class_sequences.append(indices[:10])

X_class = torch.LongTensor(class_sequences)
y_class = torch.LongTensor(class_labels)

print(f"Classification data shape: {X_class.shape}")
print(f"Labels shape: {y_class.shape}")
print(f"Classes: {torch.unique(y_class)}")

# Create and test classifier
classifier = TextClassifier(vocab_size=tokenizer.vocab_size,
                          embedding_dim=32,
                          hidden_size=64,
                          num_classes=2)

# Test forward pass
test_output = classifier(X_class[:3])
probabilities = F.softmax(test_output, dim=1)

print(f"\nClassifier output shape: {test_output.shape}")
print(f"Sample probabilities:\n{probabilities}")
print(f"Predicted classes: {probabilities.argmax(dim=1)}")
print(f"True labels: {y_class[:3]}")

# %% [markdown]
# ## Handling Variable-Length Sequences
#
# ### Mathematical Framework for Efficient Sequence Processing
#
# **Variable-length sequences** require special handling for efficient batch processing:
#
# **Sequence Length Distribution:**
# Let $\ell_i$ be length of sequence $i$. Batch efficiency depends on:
# $$\text{Efficiency} = \frac{\text{min}(\ell_1, \ldots, \ell_B)}{\text{max}(\ell_1, \ldots, \ell_B)}$$
#
# **Padding Strategies:**
#
# **Static Padding:**
# Pad all sequences to maximum length in dataset:
# $$L = \max_{i} \ell_i$$
# **Memory**: $O(B \times L)$
# **Computation**: $O(B \times L \times H)$
#
# **Dynamic Padding:**
# Pad to maximum length in current batch:
# $$L_{\text{batch}} = \max_{i \in \text{batch}} \ell_i$$
#
# **Packed Sequences (PyTorch):**
#
# **PackedSequence Structure:**
# - **data**: $\mathbb{R}^{\sum_{i=1}^B \ell_i \times H}$ (concatenated valid elements)
# - **batch_sizes**: $\mathbb{Z}^{L_{\max}}$ (number of sequences at each position)
#
# **Memory Efficiency:**
# Packed sequences use exactly $\sum_{i=1}^B \ell_i$ memory vs $B \times L_{\max}$ for padding.
#
# **LSTM with Packed Sequences:**
# $$\text{PackedLSTM}: \text{PackedSequence} \rightarrow \text{PackedSequence}$$
#
# **Masking Mathematics:**
#
# **Attention Mask:**
# $$\mathbf{M}_{i,j} = \begin{cases}
# 0 & \text{if position } j > \ell_i \\
# 1 & \text{otherwise}
# \end{cases}$$
#
# **Masked Computation:**
# $$\mathbf{h}_{i,t} = \begin{cases}
# \text{RNN}(\mathbf{x}_{i,t}, \mathbf{h}_{i,t-1}) & \text{if } t \leq \ell_i \\
# \mathbf{0} & \text{otherwise}
# \end{cases}$$
#
# **Loss Masking:**
# $$L = \frac{\sum_{i,t} M_{i,t} \cdot \ell(y_{i,t}, \hat{y}_{i,t})}{\sum_{i,t} M_{i,t}}$$
#
# **Sorting for Efficiency:**
# Sort sequences by length (descending) for better GPU utilization:
# $$\ell_{\sigma(1)} \geq \ell_{\sigma(2)} \geq \cdots \geq \ell_{\sigma(B)}$$

# %%
# Demonstration of padding and packing
def collate_batch(batch):
    """Custom collate function for variable-length sequences"""
    texts, labels = zip(*batch)

    # Convert to tensors
    texts = [torch.LongTensor(text) for text in texts]
    labels = torch.LongTensor(labels)

    # Get lengths before padding
    lengths = [len(text) for text in texts]

    # Pad sequences
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)

    return padded_texts, labels, lengths

# Variable-length sequence classifier
class VariableLengthClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(VariableLengthClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        # Embed
        embedded = self.embedding(x)

        # Pack padded sequence for efficiency
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        # Process with LSTM
        packed_output, (hidden, _) = self.lstm(packed)

        # Unpack if needed (we just need final hidden state)
        final_hidden = hidden[-1]

        # Classify
        logits = self.classifier(final_hidden)
        return logits

# Create variable length data
variable_texts = [
    [4, 5, 6],  # Short sequence
    [4, 5, 6, 7, 8, 9, 10],  # Medium sequence
    [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # Long sequence
    [4, 5]  # Very short sequence
]
variable_labels = [1, 0, 1, 0]

# Test padding and packing
batch_data = list(zip(variable_texts, variable_labels))
padded_texts, labels, lengths = collate_batch(batch_data)

print(f"Original lengths: {[len(text) for text in variable_texts]}")
print(f"Stored lengths: {lengths}")
print(f"Padded shape: {padded_texts.shape}")
print(f"Padded texts:\n{padded_texts}")

# Test variable length classifier
var_classifier = VariableLengthClassifier(vocab_size=tokenizer.vocab_size,
                                        embedding_dim=16,
                                        hidden_size=32,
                                        num_classes=2)

output = var_classifier(padded_texts, lengths)
print(f"\nClassifier output shape: {output.shape}")
print(f"Predictions: {F.softmax(output, dim=1)}")

# %% [markdown]
# ## Bidirectional RNNs
#
# ### Mathematical Framework for Bidirectional Processing
#
# **Bidirectional RNNs** process sequences in both temporal directions to capture complete contextual information:
#
# **Forward and Backward Processing:**
# $$\overrightarrow{\mathbf{h}}_t = \text{RNN}_f(\mathbf{x}_t, \overrightarrow{\mathbf{h}}_{t-1})$$
# $$\overleftarrow{\mathbf{h}}_t = \text{RNN}_b(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t+1})$$
#
# **Complete Context Representation:**
# $$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{2H}$$
#
# **Mathematical Benefits:**
# - **Forward context**: Information from $\mathbf{x}_{1:t}$
# - **Backward context**: Information from $\mathbf{x}_{t:T}$
# - **Full context**: Information from entire sequence $\mathbf{x}_{1:T}$
#
# **Parameter Count:**
# Bidirectional RNN has approximately **2× parameters** of unidirectional:
# $$\text{Params}_{\text{BiRNN}} \approx 2 \times \text{Params}_{\text{UniRNN}}$$
#
# **Computational Complexity:**
# - **Time**: $O(2 \times T \times H^2)$ (cannot parallelize forward/backward)
# - **Space**: $O(T \times 2H)$ for storing both directions
#
# **Applications:**
#
# **Sequence Labeling:**
# For each position $t$, predict label using full context:
# $$P(y_t | \mathbf{x}_{1:T}) = \text{softmax}(\mathbf{W}[\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t] + \mathbf{b})$$
#
# **Sequence Classification:**
# Aggregate information from both endpoints:
# $$\mathbf{s} = [\overrightarrow{\mathbf{h}}_T; \overleftarrow{\mathbf{h}}_1]$$
#
# **Limitations:**
# - **No streaming**: Requires complete sequence before processing
# - **Increased latency**: Cannot produce output until sequence end
# - **Memory overhead**: Stores hidden states in both directions
#
# **Attention Comparison:**
# Bidirectional RNN vs self-attention:
# - **BiRNN**: Sequential processing, linear complexity in sequence length
# - **Self-attention**: Parallel processing, quadratic complexity, better long-range dependencies

# %%
# Bidirectional LSTM classifier
class BidirectionalClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(BidirectionalClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_size,
                           batch_first=True, bidirectional=True)

        # Classifier takes concatenated forward and backward hidden states
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Embed
        embedded = self.embedding(x)

        # Bidirectional LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)

        # hidden shape: (num_layers * num_directions, batch, hidden_size)
        # For bidirectional: (2, batch, hidden_size)
        forward_hidden = hidden[0]  # Forward direction
        backward_hidden = hidden[1]  # Backward direction

        # Concatenate forward and backward hidden states
        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)

        # Classify
        logits = self.classifier(combined_hidden)
        return logits

# Compare unidirectional vs bidirectional
uni_classifier = TextClassifier(vocab_size=tokenizer.vocab_size,
                               embedding_dim=32,
                               hidden_size=64,
                               num_classes=2)

bi_classifier = BidirectionalClassifier(vocab_size=tokenizer.vocab_size,
                                       embedding_dim=32,
                                       hidden_size=64,
                                       num_classes=2)

uni_params = sum(p.numel() for p in uni_classifier.parameters())
bi_params = sum(p.numel() for p in bi_classifier.parameters())

print(f"Unidirectional classifier parameters: {uni_params:,}")
print(f"Bidirectional classifier parameters: {bi_params:,}")
print(f"Bidirectional has {((bi_params - uni_params) / uni_params * 100):.1f}% more parameters")

# Test both models
test_input = X_class[:3]
uni_output = uni_classifier(test_input)
bi_output = bi_classifier(test_input)

print(f"\nUnidirectional predictions: {F.softmax(uni_output, dim=1).argmax(dim=1)}")
print(f"Bidirectional predictions: {F.softmax(bi_output, dim=1).argmax(dim=1)}")
print(f"True labels: {y_class[:3]}")

# %% [markdown]
# ## RNN Best Practices and Optimization
#
# ### Mathematical Principles for Effective RNN Training
#
# **Training Stability and Performance** requires careful mathematical consideration of gradient dynamics:
#
# **1. Gradient Clipping Mathematics:**
#
# **Gradient Norm:**
# $$\|\nabla_{\boldsymbol{\theta}} L\| = \sqrt{\sum_i \|\nabla_{\boldsymbol{\theta}_i} L\|^2}$$
#
# **Clipping Operation:**
# $$\nabla_{\boldsymbol{\theta}} L \leftarrow \begin{cases}
# \nabla_{\boldsymbol{\theta}} L & \text{if } \|\nabla_{\boldsymbol{\theta}} L\| \leq C \\
# C \frac{\nabla_{\boldsymbol{\theta}} L}{\|\nabla_{\boldsymbol{\theta}} L\|} & \text{otherwise}
# \end{cases}$$
#
# **2. Learning Rate Scheduling:**
#
# **Step Decay:**
# $$\eta_t = \eta_0 \gamma^{\lfloor t/T \rfloor}$$
#
# **Exponential Decay:**
# $$\eta_t = \eta_0 e^{-\lambda t}$$
#
# **Cosine Annealing:**
# $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{\pi t}{T}))$$
#
# **3. Initialization Strategies:**
#
# **Xavier/Glorot Initialization:**
# $$\mathbf{W} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}\right)$$
#
# **Orthogonal Initialization (for recurrent weights):**
# Initialize $\mathbf{W}_{hh}$ as orthogonal matrix to preserve gradient magnitudes.
#
# **4. Regularization Techniques:**
#
# **Dropout Mathematics:**
# $$\mathbf{h}_t = \mathbf{m} \odot \text{RNN}(\mathbf{x}_t, \mathbf{h}_{t-1})$$
# where $\mathbf{m} \sim \text{Bernoulli}(1-p)$ and scale by $\frac{1}{1-p}$.
#
# **Weight Decay:**
# $$L_{\text{total}} = L_{\text{task}} + \lambda \|\boldsymbol{\theta}\|_2^2$$
#
# **5. Architecture Considerations:**
#
# **Skip Connections (Highway Networks):**
# $$\mathbf{h}_t = \mathbf{g}_t \odot \tilde{\mathbf{h}}_t + (1 - \mathbf{g}_t) \odot \mathbf{h}_{t-1}$$
#
# where $\mathbf{g}_t = \sigma(\mathbf{W}_g \mathbf{x}_t + \mathbf{U}_g \mathbf{h}_{t-1})$ is transform gate.
#
# **Layer Normalization:**
# $$\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sigma} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$$
#
# Applied to hidden states for training stability.

# %%
print("RNN and NLP Best Practices:")
print("\n1. Architecture Choices:")
print("   - Use LSTM/GRU instead of vanilla RNN for longer sequences")
print("   - Consider bidirectional RNNs when full context is available")
print("   - Start with 1-2 layers, add more if underfitting")
print("   - Hidden size typically 128-512 for most tasks")

print("\n2. Training Considerations:")
print("   - Use gradient clipping (norm 1.0-5.0) to prevent exploding gradients")
print("   - Apply dropout between layers and before final classifier")
print("   - Use teacher forcing for sequence generation tasks")
print("   - Consider scheduled sampling to reduce exposure bias")

print("\n3. Data Preprocessing:")
print("   - Normalize/lowercase text consistently")
print("   - Handle out-of-vocabulary words with <UNK> tokens")
print("   - Use appropriate padding and packing for efficiency")
print("   - Consider subword tokenization (BPE) for better vocabulary coverage")

print("\n4. Optimization:")
print("   - Adam optimizer often works well for RNNs")
print("   - Learning rate scheduling can improve convergence")
print("   - Batch size affects gradient noise and memory usage")
print("   - Use packed sequences for variable-length inputs")

print("\n5. Evaluation and Analysis:")
print("   - Monitor both training and validation perplexity/loss")
print("   - Visualize attention weights (when applicable)")
print("   - Analyze model predictions on edge cases")
print("   - Check for overfitting with early stopping")

print("\n6. When to Consider Alternatives:")
print("   - Very long sequences: Consider Transformers or memory networks")
print("   - Parallel processing needs: Transformers are more parallelizable")
print("   - Limited data: Pre-trained embeddings or models")
print("   - Real-time inference: Consider model compression techniques")

# Demonstrate gradient clipping importance
print("\nGradient Clipping Demonstration:")
model = nn.LSTM(10, 20, 2)
x = torch.randn(5, 10, 10)
target = torch.randn(5, 10, 20)

output, _ = model(x)
loss = F.mse_loss(output, target)
loss.backward()

# Check gradient norms before clipping
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** (1. / 2)

print(f"Gradient norm before clipping: {total_norm:.4f}")

# Apply gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Check gradient norms after clipping
total_norm_after = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm_after += param_norm.item() ** 2
total_norm_after = total_norm_after ** (1. / 2)

print(f"Gradient norm after clipping: {total_norm_after:.4f}")