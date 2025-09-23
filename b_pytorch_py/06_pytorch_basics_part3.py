# %% [markdown]
# # PyTorch Basics Part 3: Data Loading and Datasets
#
# Working with datasets, data loaders, and data preprocessing for machine learning
#
# ## Mathematical Foundation of Data Loading
#
# **Data loading** is the process of efficiently accessing and preprocessing training examples for machine learning algorithms. The mathematical framework involves:
#
# ### Dataset as a Mathematical Object
# A dataset $\mathcal{D}$ is a collection of input-output pairs:
# $$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$$
#
# Where:
# - $\mathbf{x}_i \in \mathbb{R}^d$ are feature vectors
# - $y_i \in \mathcal{Y}$ are targets (labels for classification, real values for regression)
# - $N$ is the dataset size
#
# ### Sampling and Batching
# **Mini-batch sampling** creates subsets for efficient training:
# $$\mathcal{B}_k = \{(\mathbf{x}_i, y_i)\}_{i \in I_k} \subset \mathcal{D}$$
#
# Where $I_k$ is a random subset of indices with $|I_k| = B$ (batch size).
#
# **Empirical Risk on Mini-batches:**
# $$\hat{R}_{\mathcal{B}}(f) = \frac{1}{|{\mathcal{B}}|} \sum_{(\mathbf{x}, y) \in \mathcal{B}} L(f(\mathbf{x}), y)$$
#
# This approximates the full empirical risk while enabling efficient gradient computation.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
import pandas as pd

# %% [markdown]
# ## Creating Custom Datasets
#
# ### Mathematical Interface Design
# PyTorch's `Dataset` class provides a mathematical interface to data collections:
#
# **Dataset Interface:**
# - `__len__()`: Returns $|\mathcal{D}| = N$ (cardinality)
# - `__getitem__(i)`: Returns $(\mathbf{x}_i, y_i)$ (indexed access)
#
# **Synthetic Data Generation:**
# For regression, we can generate data following:
# $$y_i = f(\mathbf{x}_i) + \epsilon_i$$
#
# Where:
# - $f: \mathbb{R}^d \rightarrow \mathbb{R}$ is the true function
# - $\mathbf{x}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ (feature distribution)
# - $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ (noise term)
#
# This creates a controlled environment for testing algorithms with known ground truth.

# %%
# Simple synthetic dataset
class SyntheticDataset(Dataset):
    def __init__(self, n_samples=1000, n_features=10, noise=0.1):
        # Generate synthetic regression data
        X, y = make_regression(n_samples=n_samples,
                              n_features=n_features,
                              noise=noise,
                              random_state=42)

        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataset
dataset = SyntheticDataset(n_samples=500, n_features=5)
print(f"Dataset size: {len(dataset)}")
print(f"Feature shape: {dataset[0][0].shape}")
print(f"Target shape: {dataset[0][1].shape}")

# Access first few samples
for i in range(3):
    x, y = dataset[i]
    print(f"Sample {i}: x={x[:3]}, y={y.item():.3f}")

# %% [markdown]
# ## Using TensorDataset for Simple Cases
#
# ### Mathematical Convenience
# `TensorDataset` provides a simple wrapper for tensor pairs:
#
# **Tensor Pairs:**
# Given feature tensor $\mathbf{X} \in \mathbb{R}^{N \times d}$ and target tensor $\mathbf{y} \in \mathbb{R}^N$ (or $\mathbb{Z}^N$ for classification):
# $$\text{TensorDataset}(\mathbf{X}, \mathbf{y}) = \{(\mathbf{X}[i], \mathbf{y}[i])\}_{i=0}^{N-1}$$
#
# **Classification Targets:**
# For $K$-class classification:
# $$y_i \in \{0, 1, 2, \ldots, K-1\}$$
#
# This integer encoding enables efficient cross-entropy loss computation via one-hot encoding:
# $$\mathbf{e}_c = [0, \ldots, 0, 1, 0, \ldots, 0] \in \{0,1\}^K$$
# where the 1 is in position $c$.

# %%
# Create data using TensorDataset (easier for simple cases)
X = torch.randn(200, 4)
y = torch.randint(0, 3, (200,))  # 3 classes

tensor_dataset = TensorDataset(X, y)
print(f"TensorDataset size: {len(tensor_dataset)}")

# Access samples
x_sample, y_sample = tensor_dataset[0]
print(f"Sample shape: {x_sample.shape}, Label: {y_sample.item()}")

# %% [markdown]
# ## DataLoader: Batching and Shuffling
#
# ### Mathematical Foundation of Mini-batch Processing
#
# **DataLoader** implements efficient batch sampling with the following mathematical operations:
#
# **Batch Generation:**
# $$\text{DataLoader}(\mathcal{D}, B) \rightarrow \{\mathcal{B}_1, \mathcal{B}_2, \ldots, \mathcal{B}_{\lceil N/B \rceil}\}$$
#
# Where each batch $\mathcal{B}_k$ contains $B$ samples (except possibly the last batch).
#
# **Shuffling (Random Permutation):**
# At each epoch, apply permutation $\pi: \{1, 2, \ldots, N\} \rightarrow \{1, 2, \ldots, N\}$:
# $$\mathcal{D}_{\text{shuffled}} = \{(\mathbf{x}_{\pi(i)}, y_{\pi(i)})\}_{i=1}^{N}$$
#
# **Benefits of Shuffling:**
# 1. **Reduces variance** in gradient estimates
# 2. **Breaks correlations** between consecutive samples
# 3. **Improves convergence** by providing diverse batches
#
# **Batch Size Effects:**
# - **Small batches**: Higher gradient noise, better generalization
# - **Large batches**: Lower gradient noise, faster computation, more memory

# %%
# Create DataLoader
dataloader = DataLoader(tensor_dataset,
                       batch_size=32,
                       shuffle=True,
                       num_workers=0)  # Use 0 for Windows compatibility

print(f"Number of batches: {len(dataloader)}")

# Iterate through batches
for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
    print(f"Batch {batch_idx}: X shape={batch_x.shape}, y shape={batch_y.shape}")
    if batch_idx >= 2:  # Show only first 3 batches
        break

# %% [markdown]
# ## Train/Validation/Test Splits
#
# ### Mathematical Foundation of Data Splitting
#
# **Data Partitioning** divides the dataset for proper model evaluation:
#
# $$\mathcal{D} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}$$
#
# where the sets are **disjoint**: $\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} \cap \mathcal{D}_{\text{test}} = \emptyset$
#
# **Common Split Ratios:**
# - Training: $|\mathcal{D}_{\text{train}}| = 0.7N$ (70%)
# - Validation: $|\mathcal{D}_{\text{val}}| = 0.15N$ (15%)
# - Test: $|\mathcal{D}_{\text{test}}| = 0.15N$ (15%)
#
# **Mathematical Purpose:**
# 1. **Training Set** $\mathcal{D}_{\text{train}}$: Optimize parameters $\boldsymbol{\theta}$
#    $$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{(\mathbf{x}, y) \in \mathcal{D}_{\text{train}}} L(f(\mathbf{x}; \boldsymbol{\theta}), y)$$
#
# 2. **Validation Set** $\mathcal{D}_{\text{val}}$: Select hyperparameters and early stopping
#
# 3. **Test Set** $\mathcal{D}_{\text{test}}$: Unbiased performance estimation
#    $$\text{Generalization Error} \approx \frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{(\mathbf{x}, y) \in \mathcal{D}_{\text{test}}} L(f(\mathbf{x}; \boldsymbol{\theta}^*), y)$$
#
# **Statistical Independence** ensures unbiased estimates.

# %%
# Split dataset into train/val/test
dataset_size = len(tensor_dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    tensor_dataset, [train_size, val_size, test_size]
)

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

# Create data loaders for each split
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# %% [markdown]
# ## Data Transformations
#
# ### Mathematical Foundation of Data Preprocessing
#
# **Data Transformations** are functions $T: \mathbb{R}^d \rightarrow \mathbb{R}^{d'}$ that preprocess input features:
#
# **Common Transformations:**
#
# **1. Normalization (Z-score):**
# $$T_{\text{norm}}(\mathbf{x}) = \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$$
#
# Where:
# - $\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_i$ (sample mean)
# - $\boldsymbol{\sigma} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (\mathbf{x}_i - \boldsymbol{\mu})^2}$ (sample std)
#
# **2. Data Augmentation (Noise Addition):**
# $$T_{\text{noise}}(\mathbf{x}) = \mathbf{x} + \boldsymbol{\epsilon}$$
#
# Where $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$
#
# **Mathematical Benefits:**
# 1. **Improved conditioning**: Normalized features have similar scales
# 2. **Regularization**: Noise injection reduces overfitting
# 3. **Data augmentation**: Artificially increases dataset size
# 4. **Invariance**: Encourages robustness to small perturbations
#
# **Transform Composition:**
# $$T_{\text{composed}} = T_2 \circ T_1 \text{ where } (T_2 \circ T_1)(\mathbf{x}) = T_2(T_1(\mathbf{x}))$$

# %%
# Dataset with transformations
class TransformDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

# Define transforms
def normalize_transform(x):
    return (x - x.mean()) / (x.std() + 1e-8)

def add_noise_transform(x, noise_std=0.1):
    return x + torch.randn_like(x) * noise_std

# Create datasets with transforms
raw_data = torch.randn(100, 5)
targets = torch.randint(0, 2, (100,))

# Without transform
dataset_raw = TransformDataset(raw_data, targets)

# With normalization
dataset_norm = TransformDataset(raw_data, targets, transform=normalize_transform)

# Compare samples
x_raw, _ = dataset_raw[0]
x_norm, _ = dataset_norm[0]

print(f"Raw data stats: mean={x_raw.mean():.3f}, std={x_raw.std():.3f}")
print(f"Normalized data stats: mean={x_norm.mean():.3f}, std={x_norm.std():.3f}")

# %% [markdown]
# ## Working with Real Data: CSV Files
#
# ### Mathematical Data Pipeline
#
# **Structured Data Loading** involves mapping tabular data to tensor representations:
#
# **CSV to Tensor Mapping:**
# $$\text{CSV}_{N \times (d+1)} \rightarrow (\mathbf{X} \in \mathbb{R}^{N \times d}, \mathbf{y} \in \mathbb{R}^N)$$
#
# Where:
# - Each row represents one sample $(\mathbf{x}_i, y_i)$
# - Feature columns → feature matrix $\mathbf{X}$
# - Target column → target vector $\mathbf{y}$
#
# **Data Type Considerations:**
# - **Features**: Usually $\mathbb{R}^d$ (continuous) or $\{0,1\}^d$ (categorical/binary)
# - **Classification targets**: $\{0, 1, \ldots, K-1\}$ (integer labels)
# - **Regression targets**: $\mathbb{R}$ (continuous values)
#
# **Categorical Encoding:**
# For categorical variables with $C$ categories:
# $$\text{OneHot}: \{1, 2, \ldots, C\} \rightarrow \{0,1\}^C$$
#
# This preserves the assumption that features are numerical for tensor operations.

# %%
# Create a sample CSV file
np.random.seed(42)
sample_data = {
    'feature1': np.random.randn(200),
    'feature2': np.random.randn(200),
    'feature3': np.random.randn(200),
    'target': np.random.randint(0, 3, 200)
}
df = pd.DataFrame(sample_data)
df.to_csv('sample_data.csv', index=False)

print("Sample CSV created:")
print(df.head())

# %%
# Dataset class for CSV data
class CSVDataset(Dataset):
    def __init__(self, csv_file, target_column='target'):
        self.data = pd.read_csv(csv_file)
        self.target_column = target_column

        # Separate features and targets
        self.features = self.data.drop(columns=[target_column]).values
        self.targets = self.data[target_column].values

        # Convert to tensors
        self.features = torch.FloatTensor(self.features)
        self.targets = torch.LongTensor(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Load CSV dataset
csv_dataset = CSVDataset('sample_data.csv')
print(f"CSV dataset size: {len(csv_dataset)}")
print(f"Feature shape: {csv_dataset[0][0].shape}")
print(f"Number of classes: {len(torch.unique(csv_dataset.targets))}")

# Create data loader
csv_loader = DataLoader(csv_dataset, batch_size=16, shuffle=True)

# Show a batch
for batch_x, batch_y in csv_loader:
    print(f"Batch features shape: {batch_x.shape}")
    print(f"Batch targets shape: {batch_y.shape}")
    print(f"Target classes in batch: {torch.unique(batch_y)}")
    break

# %% [markdown]
# ## Data Normalization and Standardization
#
# ### Mathematical Foundation of Feature Scaling
#
# **Feature Scaling** ensures all features contribute equally to learning algorithms:
#
# **Standard Normalization (Z-score):**
# $$\tilde{\mathbf{x}}_j = \frac{\mathbf{x}_j - \mu_j}{\sigma_j}$$
#
# Where for feature $j$:
# - $\mu_j = \frac{1}{N} \sum_{i=1}^{N} x_{ij}$ (sample mean)
# - $\sigma_j = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (x_{ij} - \mu_j)^2}$ (sample standard deviation)
#
# **Properties after normalization:**
# - $\mathbb{E}[\tilde{\mathbf{x}}_j] = 0$ (zero mean)
# - $\text{Var}(\tilde{\mathbf{x}}_j) = 1$ (unit variance)
#
# **Critical Rule: Use Training Statistics Only**
# $$\tilde{\mathbf{x}}_{\text{test}} = \frac{\mathbf{x}_{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
#
# This prevents **data leakage** and maintains the i.i.d. assumption.
#
# **Mathematical Benefits:**
# 1. **Gradient stability**: Prevents features with large scales from dominating
# 2. **Convergence speed**: Improves conditioning of optimization problem
# 3. **Numerical stability**: Reduces potential for overflow/underflow

# %%
# Compute statistics from training data
def compute_dataset_stats(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset))
    data, _ = next(iter(loader))
    return data.mean(dim=0), data.std(dim=0)

# Get statistics from training set
train_mean, train_std = compute_dataset_stats(train_dataset)
print(f"Training mean: {train_mean}")
print(f"Training std: {train_std}")

# Normalization function
def normalize_data(x, mean, std):
    return (x - mean) / (std + 1e-8)

# Apply normalization to a batch
for batch_x, batch_y in train_loader:
    normalized_x = normalize_data(batch_x, train_mean, train_std)
    print(f"Original batch mean: {batch_x.mean(dim=0)}")
    print(f"Normalized batch mean: {normalized_x.mean(dim=0)}")
    break

# %% [markdown]
# ## Complete Training Example with Data Loading
#
# ### Mathematical Training Framework
#
# This example demonstrates the complete pipeline from data loading to model training:
#
# **Classification Problem Setup:**
# - Feature space: $\mathbf{x} \in \mathbb{R}^{10}$
# - Label space: $y \in \{0, 1, 2\}$ (3-class classification)
# - Dataset: $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{1000}$
#
# **Model Architecture:**
# $$f(\mathbf{x}; \boldsymbol{\theta}) = \mathbf{W}_3 \sigma(\mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2) + \mathbf{b}_3$$
#
# Where:
# - $\mathbf{W}_1 \in \mathbb{R}^{20 \times 10}$, $\mathbf{W}_2 \in \mathbb{R}^{20 \times 20}$, $\mathbf{W}_3 \in \mathbb{R}^{3 \times 20}$
# - $\sigma$ is ReLU activation
# - Dropout provides regularization: $\text{Dropout}(\mathbf{h}) = \mathbf{h} \odot \mathbf{m}$ where $\mathbf{m} \sim \text{Bernoulli}(p)$
#
# **Objective Function:**
# $$L(\boldsymbol{\theta}) = \frac{1}{|\mathcal{B}|} \sum_{(\mathbf{x}, y) \in \mathcal{B}} -\log p(y|\mathbf{x}; \boldsymbol{\theta})$$
#
# Where $p(y|\mathbf{x}; \boldsymbol{\theta}) = \text{softmax}(f(\mathbf{x}; \boldsymbol{\theta}))_y$

# %%
# Create a classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=3,
                          n_informative=8, random_state=42)

# Convert to tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

# Create dataset and split
full_dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# %%
# Define model
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize model, loss, optimizer
model = Classifier(input_size=10, hidden_size=20, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Model: {model}")

# %%
# Training loop with data loader
num_epochs = 50
train_losses = []

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0

    for batch_x, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# %%
# Evaluation on test set
model.eval()
correct = 0
total = 0
test_loss = 0.0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

accuracy = 100 * correct / total
avg_test_loss = test_loss / len(test_loader)

print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Test Loss: {avg_test_loss:.4f}")

# %%
# Plot training progress
plt.figure(figsize=(8, 4))
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# %% [markdown]
# ## Data Loading Best Practices
#
# ### Mathematical and Computational Considerations
#
# **Batch Size Optimization:**
# The choice of batch size $B$ involves mathematical trade-offs:
#
# **Gradient Estimation Variance:**
# $$\text{Var}[\nabla L_{\mathcal{B}}] = \frac{1}{B} \text{Var}[\nabla L_{(\mathbf{x}, y)}]$$
#
# - **Larger B**: Lower variance, more stable gradients
# - **Smaller B**: Higher variance, better generalization (implicit regularization)
#
# **Memory Complexity:**
# - **Forward pass**: $\mathcal{O}(B \cdot d \cdot h)$ where $h$ is hidden size
# - **Backward pass**: $\mathcal{O}(B \cdot d \cdot h)$ for gradient computation
#
# **Computational Efficiency:**
# - **Vectorization benefits**: Matrix operations scale efficiently with batch size
# - **Memory hierarchy**: Larger batches better utilize cache and GPU memory
# - **Parallelization**: Larger batches enable better parallel computation
#
# **Optimal Batch Size Selection:**
# $$B^* = \arg\min_{B} \left( \frac{\text{computation time}}{B} + \lambda \cdot \text{Var}[\nabla L_{\mathcal{B}}] \right)$$
#
# Where $\lambda$ balances computation speed vs. gradient quality.
#
# **Rule of Thumb**: Powers of 2 (16, 32, 64, 128) often work well due to hardware optimization.

# %%
# Timing data loading with different settings
import time

def time_dataloader(dataloader, num_batches=10):
    start_time = time.time()

    for i, (batch_x, batch_y) in enumerate(dataloader):
        if i >= num_batches:
            break
        # Simulate some processing
        _ = batch_x.mean()

    end_time = time.time()
    return end_time - start_time

# Compare different batch sizes
batch_sizes = [16, 32, 64, 128]
for batch_size in batch_sizes:
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    elapsed = time_dataloader(loader)
    print(f"Batch size {batch_size}: {elapsed:.3f} seconds")

print("\nTips for efficient data loading:")
print("1. Use appropriate batch sizes (powers of 2 often work well)")
print("2. Consider num_workers > 0 for larger datasets")
print("3. Use pin_memory=True when transferring to GPU")
print("4. Precompute expensive transformations when possible")

# %%
# Clean up created files
import os
if os.path.exists('sample_data.csv'):
    os.remove('sample_data.csv')
    print("Cleaned up sample_data.csv")

# %%