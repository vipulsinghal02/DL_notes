# %% [markdown]
# # PyTorch Basics Part 1 - Tensors and Fundamentals
#
# Introduction to PyTorch tensors, operations, and core concepts for scientific computing
#
# ## Mathematical Foundation
#
# **Tensors** are mathematical objects that generalize scalars, vectors, and matrices to higher dimensions. In PyTorch, tensors are the fundamental data structure for all computations.
#
# ### Tensor Hierarchy:
# - **Scalar** (0-tensor): $s \in \mathbb{R}$
# - **Vector** (1-tensor): $\mathbf{v} \in \mathbb{R}^n$
# - **Matrix** (2-tensor): $\mathbf{M} \in \mathbb{R}^{m \times n}$
# - **Higher-order tensor**: $\mathcal{T} \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_k}$
#
# ### Key Properties:
# - **Shape**: The dimensions of a tensor $(d_1, d_2, \ldots, d_k)$
# - **Rank**: Number of dimensions $k$
# - **Size**: Total number of elements $\prod_{i=1}^k d_i$
# - **Dtype**: Data type (float32, int64, etc.)

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Check PyTorch version and CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for Apple Silicon GPU (Metal Performance Shaders)
print(f"MPS (Metal) available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Check device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## Using the Metal GPU (MPS) on your M2 Air is actually very simple.
# Here's what you need to know:
#
# ### Basic MPS Usage
# #### Just change your device:
# ```
#   # Instead of this (CPU):
#   device = torch.device("cpu")
#
#   # Use this (Metal GPU):
#   device = torch.device("mps")
#
#   # Or automatically choose best available:
#   device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# ```
#
# ### Complete Example
# ```
#   import torch
#
#   # Set device
#   device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#   print(f"Using device: {device}")
#
#   # Create tensors on GPU
#   x = torch.randn(1000, 1000, device=device)
#   y = torch.randn(1000, 1000, device=device)
#
#   # Operations run on GPU automatically
#   z = torch.matmul(x, y)  # This runs on Metal GPU!
#
#   # Or move existing tensors to GPU
#   cpu_tensor = torch.randn(100, 100)
#   gpu_tensor = cpu_tensor.to(device)
#
#   For Neural Networks
#
#   import torch.nn as nn
#
#   # Create model
#   model = nn.Linear(784, 10)
#
#   # Move model to GPU
#   model = model.to(device)
#
#   # Now all operations use Metal GPU
#   output = model(input_tensor.to(device))
# ```
#
# ### Important Notes
#
#   What works:
#   - Most PyTorch operations
#   - Neural network training
#   - Matrix operations
#   - Common ML algorithms
#
#   What might not work:
#   - Some newer/experimental features
#   - A few specialized operations (rare)
#
#   Fallback behavior:
#   - If operation isn't supported on MPS, PyTorch automatically falls back to CPU
#   - You'll see a warning but code won't break
#
#   Performance Tip
#
#   For learning, you might want to stick with CPU initially because:
#   - Easier debugging (no device transfers)
#   - Small datasets don't benefit much from GPU
#   - Some operations are actually faster on M2 CPU for small data
#
#   Use GPU when:
#   - Training larger models
#   - Working with bigger datasets
#   - Doing intensive matrix operations
#
# ### Simple Template
#
# ####  At start of notebook
#   ```device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")```
#
# #### Move everything to device
# ```
# model = model.to(device)
# data = data.to(device)
# target = target.to(device)
# ```
#
# Everything else stays the same!

# %%
import torch

if torch.backends.mps.is_available():
    # Check total system memory (unified on M2)
    import psutil
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Total unified memory: {total_memory_gb:.1f} GB")

    # Try to allocate tensors to see GPU memory
    try:
        # Allocate a large tensor to test
        device = torch.device("mps")
        x = torch.randn(1000, 1000, device=device)
        print("✅ MPS allocation successful")

        # Get current memory usage
        print(f"Tensor size: {x.element_size() * x.nelement() / 1024**2:.1f} MB")
    except Exception as e:
        print(f"MPS allocation failed: {e}")
else:
    print("MPS not available")

# %% [markdown]
# ## Creating Tensors
#
# ### Mathematical Context
# Tensor creation involves mapping discrete data structures to continuous mathematical objects. The fundamental operations include:
#
# **Tensor Construction from Data:**
# $$\mathcal{T}: \text{data} \mapsto \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_k}$$
#
# **Random Tensor Generation:**
# - Uniform distribution: $X \sim \mathcal{U}(0, 1)$
# - Normal distribution: $X \sim \mathcal{N}(0, 1)$
# - Discrete uniform: $X \sim \text{DiscreteUniform}(a, b)$
#
# **Special Tensors:**
# - Zero tensor: $\mathbf{0} \in \mathbb{R}^{m \times n}$ where $[\mathbf{0}]_{ij} = 0$
# - Ones tensor: $\mathbf{1} \in \mathbb{R}^{m \times n}$ where $[\mathbf{1}]_{ij} = 1$
# - Identity matrix: $\mathbf{I} \in \mathbb{R}^{n \times n}$ where $[\mathbf{I}]_{ij} = \delta_{ij}$ (Kronecker delta)

# %%
# Creating tensors from scratch
print("=== Creating Tensors ===")

# From Python lists
data_list = [1, 2, 3, 4, 5]
tensor_from_list = torch.tensor(data_list)
print(f"From list: {tensor_from_list}")
print(f"Shape: {tensor_from_list.shape}, dtype: {tensor_from_list.dtype}")

# 2D tensor from nested lists
data_2d = [[1, 2, 3], [4, 5, 6]]
tensor_2d = torch.tensor(data_2d)
print(f"\n2D tensor: \n{tensor_2d}")
print(f"Shape: {tensor_2d.shape}, dtype: {tensor_2d.dtype}")

# %%
# From NumPy arrays
numpy_array = np.array([1.5, 2.5, 3.5])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"\nFrom NumPy: {tensor_from_numpy}")
print(f"dtype: {tensor_from_numpy.dtype}")

# Creating with specific data types
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
int_tensor = torch.tensor([1.5, 2.5, 3.5], dtype=torch.int64)
print(f"\nFloat tensor: {float_tensor}, dtype: {float_tensor.dtype}")
print(f"Int tensor: {int_tensor}, dtype: {int_tensor.dtype}")

# Special tensor creation functions
print("\n=== Special Tensor Creation ===")

# Zeros and ones
zeros_tensor = torch.zeros(3, 4)
ones_tensor = torch.ones(2, 3, 5)
print(f"Zeros (3x4): \n{zeros_tensor}")
print(f"Ones shape: {ones_tensor.shape}")

# %%
# Identity matrix
identity = torch.eye(4)
print(f"\nIdentity matrix: \n{identity}")

# Random tensors
random_uniform = torch.rand(2, 3)  # Uniform [0, 1)
random_normal = torch.randn(2, 3)  # Standard normal
random_int = torch.randint(0, 10, (2, 3))  # Random integers

print(f"\nRandom uniform: \n{random_uniform}")
print(f"Random normal: \n{random_normal}")
print(f"Random int: \n{random_int}")

# Range tensors
range_tensor = torch.arange(0, 10, 2)  # start, end, step
linspace_tensor = torch.linspace(0, 1, 5)  # start, end, steps
print(f"\nRange tensor: {range_tensor}")
print(f"Linspace tensor: {linspace_tensor}")

# Like operations (same shape as existing tensor)
template = torch.tensor([[1, 2], [3, 4]])
zeros_like = torch.zeros_like(template)
ones_like = torch.ones_like(template)
rand_like = torch.rand_like(template.float())  # Need float for rand_like

print(f"\nTemplate: \n{template}")
print(f"Zeros like: \n{zeros_like}")
print(f"Random like: \n{rand_like}")

# %% [markdown]
# ## Tensor Properties and Information
#
# ### Mathematical Properties
# Understanding tensor properties is crucial for mathematical operations:
#
# **Shape and Dimensionality:**
# - Shape: $\text{shape}(\mathcal{T}) = (d_1, d_2, \ldots, d_k)$
# - Rank/Order: $\text{rank}(\mathcal{T}) = k$ (number of indices needed)
# - Total elements: $|\mathcal{T}| = \prod_{i=1}^k d_i$
#
# **Memory Layout and Strides:**
# For a tensor $\mathcal{T} \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_k}$, the memory address mapping is:
# $$\text{addr}(i_1, i_2, \ldots, i_k) = \text{base} + \sum_{j=1}^k i_j \cdot s_j$$
#
# Where $s_j$ are the strides: $s_k = 1$, $s_{k-1} = d_k$, $s_{k-2} = d_k \cdot d_{k-1}$, etc.
#
# **Data Type Mapping:**
# - $\mathbb{Z}$ → `torch.int64`
# - $\mathbb{R}$ → `torch.float32`
# - $\{0, 1\}$ → `torch.bool`

# %%
# Tensor properties
print("=== Tensor Properties ===")

# Create a sample tensor
sample_tensor = torch.randn(2, 3, 4)
print(f"Sample tensor shape: {sample_tensor.shape}")

# Basic properties
print(f"Shape: {sample_tensor.shape}")
print(f"Size: {sample_tensor.size()}")
print(f"Number of dimensions: {sample_tensor.ndim}")
print(f"Number of elements: {sample_tensor.numel()}")
print(f"Data type: {sample_tensor.dtype}")
print(f"Device: {sample_tensor.device}")
print(f"Requires gradient: {sample_tensor.requires_grad}")

# Memory layout
print(f"\nIs contiguous: {sample_tensor.is_contiguous()}")
print(f"Memory format: {sample_tensor.stride()}")

# %%
# Data type conversions
print("\n=== Data Type Conversions ===")

int_tensor = torch.tensor([1, 2, 3])
print(f"Original: {int_tensor}, dtype: {int_tensor.dtype}")

# Convert to different types
float_version = int_tensor.float()
double_version = int_tensor.double()
bool_version = int_tensor.bool()

print(f"Float: {float_version}, dtype: {float_version.dtype}")
print(f"Double: {double_version}, dtype: {double_version.dtype}")
print(f"Bool: {bool_version}, dtype: {bool_version.dtype}")

# Using .to() method
converted = int_tensor.to(torch.float32)
print(f"Using .to(): {converted}, dtype: {converted.dtype}")

# Device placement
print("\n=== Device Management ===")

cpu_tensor = torch.randn(3, 3)
print(f"CPU tensor device: {cpu_tensor.device}")

# Move to GPU if available
if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.cuda()  # or .to('cuda')
    print(f"GPU tensor device: {gpu_tensor.device}")

    # Move back to CPU
    back_to_cpu = gpu_tensor.cpu()
    print(f"Back to CPU device: {back_to_cpu.device}")
elif torch.backends.mps.is_available():
    mps_tensor = cpu_tensor.to('mps')
    print(f"Tensor moved to {mps_tensor.device}.")
    
    back_to_cpu = mps_tensor.cpu()
    print(f"Tensor moved back to {back_to_cpu.device}.")
else:
    print("CUDA or mps not available.")

# Alternative device specification
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tensor_on_device = torch.randn(2, 2, device=device)
print(f"Tensor created on device: {tensor_on_device.device}")

# %% [markdown]
# ## Tensor Indexing and Slicing
#
# ### Mathematical Foundation of Indexing
# Tensor indexing is a fundamental operation for extracting subtensors:
#
# **Basic Indexing:**
# For tensor $\mathcal{T} \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_k}$:
# - Element access: $\mathcal{T}[i_1, i_2, \ldots, i_k] \in \mathbb{R}$
# - Slice notation: $\mathcal{T}[a:b] = \{\mathcal{T}[i] : a \leq i < b\}$
#
# **Boolean Indexing:**
# Given predicate $P: \mathbb{R} \rightarrow \{0, 1\}$:
# $$\mathcal{T}[P(\mathcal{T})] = \{t \in \mathcal{T} : P(t) = 1\}$$
#
# **Advanced Indexing (Fancy Indexing):**
# For index tensors $\mathbf{I}_1, \mathbf{I}_2, \ldots$:
# $$\mathcal{T}[\mathbf{I}_1, \mathbf{I}_2, \ldots] = \{\mathcal{T}[i_1, i_2, \ldots] : i_1 \in \mathbf{I}_1, i_2 \in \mathbf{I}_2, \ldots\}$$
#
# This enables powerful data manipulation and selection operations.

# %%
# Tensor indexing and slicing
print("=== Tensor Indexing and Slicing ===")

# Create a sample tensor
data = torch.arange(24).reshape(4, 6)
print(f"Original tensor (4x6): \n{data}")

# Basic indexing
print(f"\nElement at [2, 3]: {data[2, 3]}")
print(f"First row: {data[0]}")
print(f"First column: {data[:, 0]}")
print(f"Last row: {data[-1]}")

# Slicing
print(f"\nFirst 2 rows: \n{data[:2]}")
print(f"Last 3 columns: \n{data[:, -3:]}")
print(f"Middle section [1:3, 2:5]: \n{data[1:3, 2:5]}")

# Step slicing
print(f"\nEvery other row: \n{data[::2]}")
print(f"Every other column: \n{data[:, ::2]}")
try: 
    print(f"Reverse rows: \n{data[::-1]}") # this is not allowed! 
    # https://github.com/pytorch/pytorch/issues/59786
    # ValueError: step must be greater than zero
except ValueError:
    print(f"Reverse rows: \n{torch.flip(data, dims=[0])}") # flip along rows 0
    print(f"Reverse both rows and cols: \n{torch.flip(data, dims=[0, 1])}") # flip along both rows 0 and cols 1. 
    

# Advanced indexing
print("\n=== Advanced Indexing ===")

# Boolean indexing
bool_mask = data > 10
print(f"Boolean mask (>10): \n{bool_mask}")
print(f"Elements > 10: {data[bool_mask]}")

# Fancy indexing with lists/tensors
row_indices = torch.tensor([0, 2, 3])
col_indices = torch.tensor([1, 3, 5])
print(f"\nSelected rows {row_indices}: \n{data[row_indices]}")
print(f"Selected elements at specific indices: {data[row_indices, col_indices]}")
# remember, take three elements, with i j indices taken elementwise from the two (3-element) 1d arrays. 

# Where function (like numpy.where)
condition = data > 15
result = torch.where(condition, data, torch.zeros_like(data))
print(f"\nWhere condition (>15, else 0): \n{result}")

# Masked select
mask = data % 3 == 0  # Divisible by 3
selected = torch.masked_select(data, mask)
print(f"\nElements divisible by 3: {selected}")

# Non-zero indices
nonzero_indices = torch.nonzero(data > 20)
print(f"\nIndices where data > 20: \n{nonzero_indices}")
print(f"Elements where data > 20: {data[nonzero_indices]}.")

# torch.nonzero() is useful when you need the spatial locations of elements, not just their values. Here are the main use cases:

#   1. Sparse Operations & Memory Efficiency

#   # Large mostly-zero tensor (common in deep learning)
#   sparse_tensor = torch.zeros(1000, 1000)
#   sparse_tensor[100:110, 200:210] = torch.randn(10, 10)

#   # Get only non-zero coordinates for efficient storage
#   coords = torch.nonzero(sparse_tensor)
#   values = sparse_tensor[tuple(coords.T)]

#   print(f"Only {len(coords)} non-zero elements out of {sparse_tensor.numel()}")
#   # Much more memory efficient than storing the full tensor

#   2. Image Processing & Computer Vision

#   # Find pixel locations of objects in a binary mask
#   image_mask = torch.zeros(100, 100)
#   image_mask[30:70, 40:80] = 1  # Object region

#   # Get bounding box coordinates
#   object_pixels = torch.nonzero(image_mask)
#   min_row, max_row = object_pixels[:, 0].min(), object_pixels[:, 0].max()
#   min_col, max_col = object_pixels[:, 1].min(), object_pixels[:, 1].max()

#   print(f"Bounding box: rows {min_row}-{max_row}, cols {min_col}-{max_col}")

#   3. Graph Neural Networks

#   # Adjacency matrix - find which nodes are connected
#   adjacency = torch.tensor([[0, 1, 0, 1],
#                            [1, 0, 1, 0],
#                            [0, 1, 0, 1],
#                            [1, 0, 1, 0]])

#   # Get edge list (pairs of connected nodes)
#   edges = torch.nonzero(adjacency)
#   print(f"Edges: {edges}")  # [[0,1], [0,3], [1,0], [1,2], ...]
#   # This is exactly what you need for graph algorithms

#   4. Attention Mechanisms & Masking

#   # In transformers, find which positions have valid tokens
#   attention_mask = torch.tensor([[1, 1, 1, 0, 0],  # Sentence 1: 3 tokens
#                                 [1, 1, 1, 1, 0]])  # Sentence 2: 4 tokens

#   valid_positions = torch.nonzero(attention_mask)
#   print(f"Valid token positions: {valid_positions}")
#   # Used to apply attention only to valid positions

#   5. Scientific Computing - Finding Peaks/Events

#   # Time series analysis - find when signal exceeds threshold
#   signal = torch.tensor([0.1, 0.3, 0.8, 1.2, 0.9, 0.2, 1.5, 0.4])
#   threshold = 1.0

#   peak_times = torch.nonzero(signal > threshold).squeeze()
#   print(f"Signal peaks at times: {peak_times}")  # Times when events occur

#   6. Loss Function Masking

#   # Only compute loss on certain positions (e.g., non-padding tokens)
#   predictions = torch.randn(32, 100, 10000)  # batch, seq_len, vocab
#   targets = torch.randint(0, 10000, (32, 100))
#   padding_mask = targets != 0  # 0 is padding token

#   # Get valid (non-padding) positions for loss computation
#   valid_positions = torch.nonzero(padding_mask)
#   loss = F.cross_entropy(
#       predictions[valid_positions[:, 0], valid_positions[:, 1]],
#       targets[valid_positions[:, 0], valid_positions[:, 1]]
#   )
# %% [markdown]
# ## Basic Tensor Operations
#
# ### Mathematical Operations on Tensors
#
# **Element-wise Operations (Hadamard Operations):**
# For tensors $\mathcal{A}, \mathcal{B} \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_k}$:
# - Addition: $(\mathcal{A} + \mathcal{B})_{i_1,i_2,\ldots,i_k} = \mathcal{A}_{i_1,i_2,\ldots,i_k} + \mathcal{B}_{i_1,i_2,\ldots,i_k}$
# - Hadamard product: $(\mathcal{A} \odot \mathcal{B})_{i_1,i_2,\ldots,i_k} = \mathcal{A}_{i_1,i_2,\ldots,i_k} \cdot \mathcal{B}_{i_1,i_2,\ldots,i_k}$
#
# **Broadcasting:**
# Automatic shape alignment following rules:
# 1. Align shapes from rightmost dimension
# 2. Dimensions of size 1 are "stretched" to match
# 3. Missing dimensions are assumed to be 1
#
# **Mathematical Functions:**
# - Exponential: $\exp(\mathcal{T}) = e^{\mathcal{T}}$ (element-wise)
# - Logarithm: $\log(\mathcal{T})$ with domain $\mathcal{T} > 0$
# - Trigonometric: $\sin(\mathcal{T}), \cos(\mathcal{T}), \tan(\mathcal{T})$
# - Power: $\mathcal{T}^p$ for scalar $p$
#
# **Clamping/Clipping:**
# $$\text{clamp}(x, a, b) = \begin{cases}
# a & \text{if } x < a \\
# x & \text{if } a \leq x \leq b \\
# b & \text{if } x > b
# \end{cases}$$

# %%
# Basic tensor operations
print("=== Basic Tensor Operations ===")

# Create sample tensors
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
print(f"Tensor a: \n{a}")
print(f"Tensor b: \n{b}")

# Element-wise operations
print("\n=== Element-wise Operations ===")
print(f"Addition (a + b): \n{a + b}")
print(f"Subtraction (a - b): \n{a - b}")
print(f"Multiplication (a * b): \n{a * b}")
print(f"Division (a / b): \n{a / b}")
print(f"Power (a ** 2): \n{a ** 2}")

# In-place operations (memory efficient)
print("\n=== In-place Operations ===")
c = a.clone()  # Make a copy
print(f"Original c: \n{c}")
c.add_(b)  # In-place addition
print(f"After c.add_(b): \n{c}")

# Other in-place operations
d = torch.ones(2, 2)
d.mul_(5)  # d *= 5
d.sub_(2)  # d -= 2
print(f"After in-place operations: \n{d}")

# Broadcasting
print("\n=== Broadcasting ===")
matrix = torch.randn(3, 4)
vector = torch.randn(4)
scalar = 5

print(f"Matrix shape: {matrix.shape}")
print(f"Vector shape: {vector.shape}")

# Broadcasting examples
result1 = matrix + vector  # Vector broadcasted to each row
result2 = matrix + scalar  # Scalar broadcasted to all elements
result3 = matrix + vector.unsqueeze(0)  # Explicit dimension addition
# unsqueeze(0) adds a new dimension of size 1 at position 0 (the beginning).

#   Let's break it down:

#   matrix = torch.randn(3, 4)     # Shape: [3, 4]
#   vector = torch.randn(4)        # Shape: [4]

#   # Without unsqueeze - this works due to broadcasting
#   result1 = matrix + vector      # vector broadcasts to each row

#   # With unsqueeze(0) - explicit dimension addition
#   vector_unsqueezed = vector.unsqueeze(0)  # Shape: [1, 4]
#   result3 = matrix + vector_unsqueezed     # [3, 4] + [1, 4] = [3, 4]

#   What unsqueeze(0) does:
#   - Original vector: [4] → becomes [1, 4]
#   - Adds a dimension of size 1 at position 0
#   - Turns a 1D vector into a 2D "row vector"

#   Visual example:
#   vector = torch.tensor([1, 2, 3, 4])           # Shape: [4]
#   print(vector.shape)                           # torch.Size([4])

#   vector_unsqueezed = vector.unsqueeze(0)       # Shape: [1, 4] 
#   print(vector_unsqueezed.shape)                # torch.Size([1, 4])
#   print(vector_unsqueezed)                      # tensor([[1, 2, 3, 4]])

#   Why use it here?
#   Both result1 and result3 give the same answer, but:
#   - result1: PyTorch automatically broadcasts [4] to work with [3, 4]
#   - result3: You explicitly make the vector [1, 4] so the broadcasting is more obvious

#   It's mainly for clarity - making the broadcasting behavior explicit rather than relying on PyTorch's automatic
#   broadcasting rules. Some people prefer this style to make their intent clearer in the code.

#   Other unsqueeze positions:
#   vector.unsqueeze(0)  # [4] → [1, 4] (row vector)
#   vector.unsqueeze(1)  # [4] → [4, 1] (column vector)



print(f"Matrix + vector result shape: {result1.shape}")
print(f"Matrix + scalar result shape: {result2.shape}")

# Mathematical functions
print("\n=== Mathematical Functions ===")
x = torch.tensor([0., 1., 2., 3.])
print(f"Original: {x}")
print(f"Exponential: {torch.exp(x)}")
print(f"Logarithm: {torch.log(x + 1)}")
print(f"Square root: {torch.sqrt(x)}")
print(f"Sine: {torch.sin(x)}")
print(f"Cosine: {torch.cos(x)}")

# Trigonometric functions
angles = torch.tensor([0., np.pi/4, np.pi/2, np.pi])
print(f"\nAngles: {angles}")
print(f"Sin(angles): {torch.sin(angles)}")
print(f"Cos(angles): {torch.cos(angles)}")

# Rounding and clipping
values = torch.tensor([-2.5, -1.3, 0.7, 1.9, 3.2])
print(f"\nOriginal values: {values}")
print(f"Floor: {torch.floor(values)}")
print(f"Ceil: {torch.ceil(values)}")
print(f"Round: {torch.round(values)}")
print(f"Clamp(-1, 2): {torch.clamp(values, -1, 2)}")

# %% [markdown]
# ## Tensor Reshaping and Manipulation
#
# ### Mathematical Basis of Tensor Transformations
#
# **Reshaping (View Transformation):**
# Given tensor $\mathcal{T} \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_k}$ with $|\mathcal{T}| = \prod_{i=1}^k d_i$ elements, reshaping creates a new view:
# $$\text{reshape}(\mathcal{T}, (m_1, m_2, \ldots, m_\ell)) \in \mathbb{R}^{m_1 \times m_2 \times \cdots \times m_\ell}$$
# where $\prod_{i=1}^k d_i = \prod_{j=1}^\ell m_j$ (conservation of elements).
#
# **Transpose Operations:**
# For matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$:
# $$(\mathbf{A}^T)_{ij} = \mathbf{A}_{ji}$$
#
# For higher-order tensors, transpose swaps specified dimensions:
# $$\text{transpose}(\mathcal{T}, \text{dim}_1, \text{dim}_2)$$
#
# **Dimension Manipulation:**
# - **Squeeze**: Remove dimensions of size 1: $\mathbb{R}^{d_1 \times 1 \times d_3} \rightarrow \mathbb{R}^{d_1 \times d_3}$
# - **Unsqueeze**: Add dimensions of size 1: $\mathbb{R}^{d_1 \times d_2} \rightarrow \mathbb{R}^{d_1 \times 1 \times d_2}$
#
# **Tensor Concatenation and Stacking:**
# - **Concatenation**: Join along existing dimension
#   $$\text{cat}([\mathcal{A}, \mathcal{B}], \text{dim}) \text{ where } \mathcal{A}, \mathcal{B} \text{ have compatible shapes}$$
# - **Stacking**: Join along new dimension
#   $$\text{stack}([\mathcal{A}, \mathcal{B}], \text{dim}) \text{ creates new dimension}$$

# %%
# Tensor reshaping and manipulation
print("=== Tensor Reshaping and Manipulation ===")

# Create a sample tensor
original = torch.arange(24)
print(f"Original tensor: {original}")
print(f"Shape: {original.shape}")

# Reshaping
print("\n=== Reshaping ===")
reshaped_2d = original.reshape(4, 6)
reshaped_3d = original.reshape(2, 3, 4)
reshaped_infer = original.reshape(-1, 8)  # Infer one dimension

print(f"Reshaped to 4x6: \n{reshaped_2d}")
print(f"Reshaped to 2x3x4 shape: {reshaped_3d.shape}")
print(f"Reshaped with -1 (inferred): {reshaped_infer.shape}")

# View vs reshape
print("\n=== View vs Reshape ===")
viewed = original.view(4, 6)  # Must be contiguous
print(viewed)
print(f"View (shares memory): {viewed.shape}")
print(f"Shares memory with original: {viewed.storage().data_ptr() == original.storage().data_ptr()}")

# Transpose and permute
print("\n=== Transpose and Permute ===")
matrix = torch.randn(3, 4)
print(f"Original matrix shape: {matrix.shape}")

# Transpose (2D)
transposed = matrix.t()  # or matrix.T
print(f"Transposed shape: {transposed.shape}")

# Transpose specific dimensions
tensor_3d = torch.randn(2, 3, 4)
transposed_3d = tensor_3d.transpose(0, 2)  # Swap dim 0 and 2
print(f"3D tensor original: {tensor_3d.shape}")
print(f"After transpose(0,2): {transposed_3d.shape}")
print(f"Before transposition: {tensor_3d}.")
print(f"After transposition: {transposed_3d}.")


# 3D tensor original: torch.Size([2, 3, 4])
# After transpose(0,2): torch.Size([4, 3, 2])
# Before transposition: tensor([[[ 0.3559, -0.6866, -0.4934,  0.2415],
#          [-1.1109,  0.0915, -2.3169, -0.2168],
#          [-0.9138, -0.6581,  0.0780,  0.5258]],

#         [[-0.4880,  1.1914, -0.8140, -0.7360],
#          [-1.4032,  0.0360, -0.0635,  0.6756],
#          [-0.0978,  1.8446, -1.1845,  1.3835]]]).
# After transposition: tensor([[[ 0.3559, -0.4880],
#          [-1.1109, -1.4032],
#          [-0.9138, -0.0978]],

#         [[-0.6866,  1.1914],
#          [ 0.0915,  0.0360],
#          [-0.6581,  1.8446]],

#         [[-0.4934, -0.8140],
#          [-2.3169, -0.0635],
#          [ 0.0780, -1.1845]],

#         [[ 0.2415, -0.7360],
#          [-0.2168,  0.6756],
#          [ 0.5258,  1.3835]]]).
# mind numbing!

# Permute (rearrange all dimensions)
permuted = tensor_3d.permute(2, 0, 1)
print(f"After permute(2,0,1): {permuted.shape}")

# Squeeze and unsqueeze
print("\n=== Squeeze and Unsqueeze ===")
tensor_with_ones = torch.randn(1, 3, 1, 4)
print(f"Original with size-1 dims: {tensor_with_ones.shape}")

# Remove size-1 dimensions
squeezed = tensor_with_ones.squeeze()
print(f"After squeeze(): {squeezed.shape}")

# Remove specific dimension
squeezed_dim = tensor_with_ones.squeeze(0)
print(f"After squeeze(0): {squeezed_dim.shape}")

# Add dimensions
unsqueezed = squeezed.unsqueeze(1)
print(f"After unsqueeze(1): {unsqueezed.shape}")

# Flatten
print("\n=== Flatten ===")
to_flatten = torch.randn(2, 3, 4)
print(f"Original shape: {to_flatten.shape}")

flattened = to_flatten.flatten()
print(f"Flattened: {flattened.shape}")

# Flatten starting from specific dimension
partial_flatten = to_flatten.flatten(start_dim=1)
print(f"Flattened from dim 1: {partial_flatten.shape}")

# Stack and concatenate
print("\n=== Stack and Concatenate ===")
a = torch.randn(2, 3)
b = torch.randn(2, 3)
c = torch.randn(2, 3)


# Stack (creates new dimension)
stacked = torch.stack([a, b, c])
print(f"Stacked shape: {stacked.shape}")

stacked_dim1 = torch.stack([a, b, c], dim=1)
print(f"Stacked along dim 1: {stacked_dim1.shape}")
print(f"Individual tensors: \n{a}, \n{b}, \n{c}")
print(f"Tensor after stacking along dim 1: {stacked_dim1}.")
#it makes sense after staring at it, but def need to review it #!!

# Concatenate (along existing dimension)
concat_dim0 = torch.cat([a, b, c], dim=0)
concat_dim1 = torch.cat([a, b, c], dim=1)
print(f"Concatenated along dim 0: {concat_dim0}")
print(f"Concatenated along dim 1: {concat_dim1}")

# Split and chunk
print("\n=== Split and Chunk ===")
large_tensor = torch.randn(6, 4)
print(f"Large tensor shape: {large_tensor.shape}")

# Split into equal parts
chunks = torch.chunk(large_tensor, 3, dim=0)  # 3 chunks along dim 0
print(f"Number of chunks: {len(chunks)}")
print(f"First chunk shape: {chunks[0].shape}")

# Split with specific sizes
splits = torch.split(large_tensor, [2, 3, 1], dim=0)
print(f"Split sizes: {[s.shape for s in splits]}")

# %% [markdown]
# ## Reduction Operations and Statistics
#
# ### Mathematical Foundation of Reductions
#
# **Reduction Operations** collapse tensor dimensions by applying functions across specified axes.
#
# **Basic Reductions:**
# For tensor $\mathcal{T} \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_k}$:
#
# - **Sum**: $\sum_{\mathcal{T}} = \sum_{i_1=1}^{d_1} \sum_{i_2=1}^{d_2} \cdots \sum_{i_k=1}^{d_k} \mathcal{T}_{i_1,i_2,\ldots,i_k}$
#
# - **Mean**: $\bar{\mathcal{T}} = \frac{1}{|\mathcal{T}|} \sum_{\mathcal{T}}$ where $|\mathcal{T}| = \prod_{i=1}^k d_i$
#
# - **Variance**: $\text{Var}(\mathcal{T}) = \frac{1}{|\mathcal{T}|} \sum (\mathcal{T} - \bar{\mathcal{T}})^2$
#
# - **Standard Deviation**: $\sigma(\mathcal{T}) = \sqrt{\text{Var}(\mathcal{T})}$
#
# **Dimensional Reductions:**
# Reducing along dimension $j$:
# $$\text{reduce}_j(\mathcal{T}) \in \mathbb{R}^{d_1 \times \cdots \times d_{j-1} \times d_{j+1} \times \cdots \times d_k}$$
#
# **Order Statistics:**
# - **Minimum**: $\min(\mathcal{T}) = \min_{i_1,\ldots,i_k} \mathcal{T}_{i_1,\ldots,i_k}$
# - **Maximum**: $\max(\mathcal{T}) = \max_{i_1,\ldots,i_k} \mathcal{T}_{i_1,\ldots,i_k}$
# - **Quantile**: $Q_p(\mathcal{T})$ such that $P(X \leq Q_p) = p$
#
# **Norms:**
# - **L1 (Manhattan)**: $\|\mathcal{T}\|_1 = \sum |t_i|$
# - **L2 (Euclidean)**: $\|\mathcal{T}\|_2 = \sqrt{\sum t_i^2}$
# - **L∞ (Maximum)**: $\|\mathcal{T}\|_\infty = \max |t_i|$

# %%
# Reduction operations and statistics
print("=== Reduction Operations and Statistics ===")

# Create sample data
data = torch.randn(4, 5)
print(f"Sample data (4x5): \n{data}")

# Basic reductions
print("\n=== Basic Reductions ===")
print(f"Sum (all elements): {data.sum()}")
print(f"Mean (all elements): {data.mean()}")
print(f"Standard deviation: {data.std()}")
print(f"Variance: {data.var()}")
print(f"Min: {data.min()}")
print(f"Max: {data.max()}")
print(f"Product: {data.prod()}")

# Reductions along specific dimensions
print("\n=== Dimension-wise Reductions ===")
print(f"Sum along rows (dim=0): {data.sum(dim=0)}")
print(f"Sum along columns (dim=1): {data.sum(dim=1)}")
print(f"Mean along rows: {data.mean(dim=0)}")
print(f"Mean along columns: {data.mean(dim=1)}")

# Keep dimensions
sum_keepdim = data.sum(dim=1, keepdim=True)
print(f"\nSum with keepdim=True: {sum_keepdim.shape}")
print(f"Values: \n{sum_keepdim}")

# Min/Max with indices
print("\n=== Min/Max with Indices ===")
min_values, min_indices = data.min(dim=1)
max_values, max_indices = data.max(dim=1)

print(f"Min values per row: {min_values}")
print(f"Min indices per row: {min_indices}")
print(f"Max values per row: {max_values}")
print(f"Max indices per row: {max_indices}")

# Quantiles and percentiles
print("\n=== Quantiles and Percentiles ===")
flattened = data.flatten()
print(f"Median: {torch.median(flattened)}")
print(f"25th percentile: {torch.quantile(flattened, 0.25)}")
print(f"75th percentile: {torch.quantile(flattened, 0.75)}")

print(f"Quantiles on non flattened data: {torch.quantile(data, 0.25)}.")
print("Works just fine!")
# Multiple quantiles
quantiles = torch.quantile(data, torch.tensor([0.1, 0.5, 0.9])) # changed from flattened. 
print(f"10th, 50th, 90th percentiles: {quantiles}")

# Sorting and ranking
print("\n=== Sorting and Ranking ===")
sample_row = data[0]
print(f"Sample row: {sample_row}")

# Sort
sorted_values, sorted_indices = torch.sort(sample_row)
print(f"Sorted values: {sorted_values}")
print(f"Sort indices: {sorted_indices}")

# Sort in descending order
sorted_desc, indices_desc = torch.sort(sample_row, descending=True)
print(f"Sorted descending: {sorted_desc}")
print(f"Sort descending indices: {indices_desc}")

# Top-k values
top_k_values, top_k_indices = torch.topk(sample_row, k=3)
print(f"Top 3 values: {top_k_values}")
print(f"Top 3 indices: {top_k_indices}")

# Unique values
print("\n=== Unique Values ===")
repeated_data = torch.tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
unique_values = torch.unique(repeated_data)
unique_with_counts = torch.unique(repeated_data, return_counts=True)

print(f"Original: {repeated_data}")
print(f"Unique values: {unique_values}")
print(f"Unique values with counts: {unique_with_counts}")

# Statistical measures
print("\n=== Advanced Statistics ===")
sample_data = torch.randn(1000)

# Moments
mean = sample_data.mean()
std = sample_data.std()
var = sample_data.var()

print(f"Sample statistics (n=1000):")
print(f"Mean: {mean:.4f}")
print(f"Std: {std:.4f}")
print(f"Variance: {var:.4f}")

# Manual calculation of higher moments
centered = sample_data - mean
skewness = (centered**3).mean() / (std**3)
kurtosis = (centered**4).mean() / (std**4) - 3  # Excess kurtosis

print(f"Skewness: {skewness:.4f}")
print(f"Excess kurtosis: {kurtosis:.4f}")

# Histogram
print("\n=== Histogram ===")
hist = torch.histc(sample_data, bins=10, min=-3, max=3)
print(f"Histogram counts: {hist}")

# %% [markdown]
# ## Linear Algebra Operations
#
# ### Mathematical Foundation of Linear Algebra
#
# **Matrix Multiplication:**
# For matrices $\mathbf{A} \in \mathbb{R}^{m \times k}$ and $\mathbf{B} \in \mathbb{R}^{k \times n}$:
# $$(\mathbf{A}\mathbf{B})_{ij} = \sum_{\ell=1}^k \mathbf{A}_{i\ell} \mathbf{B}_{\ell j}$$
#
# **Batch Matrix Multiplication:**
# For batch tensors $\mathcal{A} \in \mathbb{R}^{b \times m \times k}$ and $\mathcal{B} \in \mathbb{R}^{b \times k \times n}$:
# $$(\mathcal{A} \otimes \mathcal{B})_{[i]} = \mathcal{A}_{[i]} \mathbf{B}_{[i]} \text{ for each batch } i$$
#
# **Vector Operations:**
# - **Dot Product**: $\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i = \mathbf{u}^T \mathbf{v}$
# - **Outer Product**: $\mathbf{u} \otimes \mathbf{v} = \mathbf{u}\mathbf{v}^T \in \mathbb{R}^{m \times n}$
# - **Cross Product**: $\mathbf{u} \times \mathbf{v} = (u_2v_3 - u_3v_2, u_3v_1 - u_1v_3, u_1v_2 - u_2v_1)$
#
# **Matrix Decompositions:**
#
# **Eigendecomposition** (for symmetric $\mathbf{A} \in \mathbb{R}^{n \times n}$):
# $$\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T$$
# where $\mathbf{Q}$ contains eigenvectors, $\mathbf{\Lambda}$ is diagonal with eigenvalues.
#
# **Singular Value Decomposition (SVD):**
# $$\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$$
# where $\mathbf{U}, \mathbf{V}$ are orthogonal, $\mathbf{\Sigma}$ contains singular values.
#
# **QR Decomposition:**
# $$\mathbf{A} = \mathbf{Q}\mathbf{R}$$
# where $\mathbf{Q}$ is orthogonal, $\mathbf{R}$ is upper triangular.
#
# **Linear System Solution:**
# For $\mathbf{A}\mathbf{x} = \mathbf{b}$:
# - **Exact solution**: $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$ (when $\mathbf{A}$ is invertible)
# - **Least squares**: $\mathbf{x} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b}$ (when overdetermined)

# %%
# Linear algebra operations
print("=== Linear Algebra Operations ===")

# Matrix multiplication
print("=== Matrix Multiplication ===")
A = torch.randn(3, 4)
B = torch.randn(4, 5)
print(f"Matrix A shape: {A.shape}")
print(f"Matrix B shape: {B.shape}")

# Matrix multiplication
C = torch.mm(A, B)  # or A @ B
print(f"A @ B shape: {C.shape}")

# Batch matrix multiplication
batch_A = torch.randn(10, 3, 4)
batch_B = torch.randn(10, 4, 5)
batch_C = torch.bmm(batch_A, batch_B)
print(f"Batch multiplication result shape: {batch_C.shape}")

# More general matrix multiplication (handles broadcasting)
general_A = torch.randn(2, 3, 4)
general_B = torch.randn(4, 5)
general_C = torch.matmul(general_A, general_B)
print(f"General matmul result shape: {general_C.shape}")

# Vector operations
print("\n=== Vector Operations ===")
v1 = torch.randn(5)
v2 = torch.randn(5)

# Dot product
dot_product = torch.dot(v1, v2)
print(f"Dot product: {dot_product}")

# Outer product
outer_product = torch.outer(v1, v2)
print(f"Outer product shape: {outer_product.shape}")

# Cross product (3D vectors)
v3d_1 = torch.randn(3)
v3d_2 = torch.randn(3)
cross_product = torch.cross(v3d_1, v3d_2)
print(f"Cross product: {cross_product}")

# Norms
print("\n=== Vector and Matrix Norms ===")
vector = torch.tensor([3., 4., 5.])
matrix = torch.randn(3, 3)

# Vector norms
l1_norm = torch.norm(vector, p=1)
l2_norm = torch.norm(vector, p=2)  # Euclidean norm
inf_norm = torch.norm(vector, p=float('inf'))

print(f"Vector: {vector}")
print(f"L1 norm: {l1_norm}")
print(f"L2 norm: {l2_norm}")
print(f"Infinity norm: {inf_norm}")

# Matrix norms
frobenius_norm = torch.norm(matrix, p='fro')
nuclear_norm = torch.norm(matrix, p='nuc')
print(f"\nMatrix Frobenius norm: {frobenius_norm}")
print(f"Matrix nuclear norm: {nuclear_norm}")

# Matrix decompositions
print("\n=== Matrix Decompositions ===")

# Create a symmetric positive definite matrix
random_matrix = torch.randn(4, 4)
symmetric_matrix = random_matrix @ random_matrix.T + torch.eye(4)

# Eigendecomposition
eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_matrix)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors shape: {eigenvectors.shape}")

# SVD
U, S, Vt = torch.linalg.svd(random_matrix)
print(f"\nSVD shapes - U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")
print(f"Singular values: {S}")

# QR decomposition
Q, R = torch.linalg.qr(random_matrix)
print(f"\nQR shapes - Q: {Q.shape}, R: {R.shape}")

# Cholesky decomposition
try:
    L = torch.linalg.cholesky(symmetric_matrix)
    print(f"Cholesky decomposition shape: {L.shape}")
except Exception as e:
    print(f"Cholesky failed: {e}")

# Matrix properties
print("\n=== Matrix Properties ===")

# Determinant
det = torch.linalg.det(symmetric_matrix)
print(f"Determinant: {det}")

# Matrix rank
rank = torch.linalg.matrix_rank(random_matrix)
print(f"Matrix rank: {rank}")

# Condition number
cond = torch.linalg.cond(symmetric_matrix)
print(f"Condition number: {cond}")

# Trace
trace = torch.trace(symmetric_matrix)
print(f"Trace: {trace}")

# Matrix inverse
try:
    inverse = torch.linalg.inv(symmetric_matrix)
    print(f"Inverse computed successfully")

    # Verify inverse
    identity_check = symmetric_matrix @ inverse
    error = torch.norm(identity_check - torch.eye(4))
    print(f"Inverse verification error: {error}")
except Exception as e:
    print(f"Matrix inversion failed: {e}")

# Solving linear systems
print("\n=== Solving Linear Systems ===")
A_system = torch.randn(4, 4)
b_system = torch.randn(4)

# Solve Ax = b
try:
    x_solution = torch.linalg.solve(A_system, b_system)
    print(f"Solution x shape: {x_solution.shape}")

    # Verify solution
    residual = A_system @ x_solution - b_system
    residual_norm = torch.norm(residual)
    print(f"Residual norm: {residual_norm}")
except Exception as e:
    print(f"Linear system solving failed: {e}")

# Least squares solution
A_overdetermined = torch.randn(6, 4)  # More equations than unknowns
b_overdetermined = torch.randn(6)

x_lstsq = torch.linalg.lstsq(A_overdetermined, b_overdetermined).solution
print(f"\nLeast squares solution shape: {x_lstsq.shape}")

# Compute residual
residual_lstsq = A_overdetermined @ x_lstsq - b_overdetermined
residual_norm_lstsq = torch.norm(residual_lstsq)
print(f"Least squares residual norm: {residual_norm_lstsq}")

# %% [markdown]
# ## Working with NumPy and Data Conversion
#
# ### Mathematical Interoperability
#
# **Memory Sharing and Zero-Copy Operations:**
# PyTorch tensors and NumPy arrays can share the same memory buffer, enabling efficient data transfer without copying:
#
# $$\text{Memory Layout: } \text{base\_ptr} + \sum_{i=1}^k \text{index}_i \times \text{stride}_i$$
#
# **Data Type Mapping:**
# The conversion preserves mathematical properties:
# - $\mathbb{Z}_{64} \leftrightarrow$ `np.int64` $\leftrightarrow$ `torch.int64`
# - $\mathbb{R}_{32} \leftrightarrow$ `np.float32` $\leftrightarrow$ `torch.float32`
# - $\{0,1\} \leftrightarrow$ `np.bool` $\leftrightarrow$ `torch.bool`
#
# **Important Considerations:**
# - **Shared Memory**: `torch.from_numpy()` creates views, not copies
# - **Device Constraints**: GPU tensors must be moved to CPU before NumPy conversion
# - **Gradient Tracking**: Conversion may affect autograd functionality
#
# **Data Flow Pipeline:**
# $$\text{Raw Data} \xrightarrow{\text{NumPy}} \text{Preprocessed} \xrightarrow{\text{PyTorch}} \text{ML Pipeline} \xrightarrow{\text{Results}} \text{Analysis}$$
#
# This enables seamless integration between scientific computing ecosystems while maintaining mathematical consistency and computational efficiency.

# %%
# Working with NumPy and data conversion
print("=== Working with NumPy and Data Conversion ===")

# NumPy to PyTorch conversion
print("=== NumPy to PyTorch ===")
numpy_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
print(f"NumPy array: \n{numpy_array}")
print(f"NumPy dtype: {numpy_array.dtype}")

# Convert to PyTorch
torch_from_numpy = torch.from_numpy(numpy_array)
print(f"\nPyTorch tensor: \n{torch_from_numpy}")
print(f"PyTorch dtype: {torch_from_numpy.dtype}")

# Note: from_numpy() shares memory!
print(f"\nShares memory: {torch_from_numpy.data_ptr() == numpy_array.__array_interface__['data'][0]}")

# Modify numpy array and see effect on tensor
numpy_array[0, 0] = 999
print(f"After modifying NumPy array: \n{torch_from_numpy}")

# PyTorch to NumPy conversion
print("\n=== PyTorch to NumPy ===")
torch_tensor = torch.randn(3, 4)
print(f"Original PyTorch tensor: \n{torch_tensor}")

# Convert to NumPy
numpy_from_torch = torch_tensor.numpy()
print(f"\nConverted to NumPy: \n{numpy_from_torch}")
print(f"NumPy dtype: {numpy_from_torch.dtype}")

# Also shares memory
torch_tensor[0, 0] = 777
print(f"After modifying PyTorch tensor: \n{numpy_from_torch}")

# Safe conversion (copy)
print("\n=== Safe Conversion (Copy) ===")
torch_original = torch.randn(2, 3)
numpy_copy = torch_original.detach().numpy().copy()
torch_copy = torch.tensor(numpy_copy)  # Creates copy

print(f"Original: \n{torch_original}")
torch_original[0, 0] = 555
print(f"After modification - copy unchanged: \n{torch_copy}")

# GPU tensors require CPU conversion first
if torch.backends.mps.is_available():
    print("\n=== GPU Tensor Conversion ===")
    gpu_tensor = torch.randn(2, 2, device='mps')
    print(f"GPU tensor device: {gpu_tensor.device}")

    # Must move to CPU first
    cpu_tensor = gpu_tensor.cpu()
    numpy_from_gpu = cpu_tensor.numpy()
    print(f"Converted GPU tensor to NumPy: \n{numpy_from_gpu}")

# Working with different data types
print("\n=== Data Type Conversions ===")

# Create tensors with different types
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
double_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

print(f"Int tensor dtype: {int_tensor.dtype}")
print(f"Float tensor dtype: {float_tensor.dtype}")
print(f"Double tensor dtype: {double_tensor.dtype}")

# Convert to NumPy and see dtypes
print(f"\nNumPy dtypes:")
print(f"From int tensor: {int_tensor.numpy().dtype}")
print(f"From float tensor: {float_tensor.numpy().dtype}")
print(f"From double tensor: {double_tensor.numpy().dtype}")

# Working with pandas
print("\n=== Working with Pandas ===")

# Create a pandas DataFrame
df = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'target': np.random.randint(0, 2, 100)
})

print(f"DataFrame shape: {df.shape}")
print(f"DataFrame head: \n{df.head()}")

# Convert DataFrame to tensor
features = torch.tensor(df[['feature1', 'feature2']].values, dtype=torch.float32)
targets = torch.tensor(df['target'].values, dtype=torch.long)

print(f"\nFeatures tensor shape: {features.shape}")
print(f"Targets tensor shape: {targets.shape}")
print(f"Features dtype: {features.dtype}")
print(f"Targets dtype: {targets.dtype}")

# Convert back to pandas
features_numpy = features.numpy()
targets_numpy = targets.numpy()

df_reconstructed = pd.DataFrame({
    'feature1': features_numpy[:, 0],
    'feature2': features_numpy[:, 1],
    'target': targets_numpy
})

print(f"\nReconstructed DataFrame head: \n{df_reconstructed.head()}")

# Memory considerations
print("\n=== Memory Considerations ===")
large_numpy = np.random.randn(1000, 1000)
print(f"NumPy array memory: {large_numpy.nbytes / 1024**2:.2f} MB")

# Shared memory conversion
large_torch_shared = torch.from_numpy(large_numpy)
print(f"Shared conversion - additional memory: ~0 MB")

# Copy conversion
large_torch_copy = torch.tensor(large_numpy)
print(f"Copy conversion - doubles memory usage")

print(f"\nBoth tensors equal: {torch.equal(large_torch_shared, large_torch_copy)}")
print(f"Share memory: {large_torch_shared.data_ptr() == large_torch_copy.data_ptr()}")

# %%