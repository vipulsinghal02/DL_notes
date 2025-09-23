# %% [markdown]
# # PyTorch Basics Part 5: Advanced Topics and Production
#
# Advanced PyTorch features, model deployment, and production considerations with mathematical foundations
#
# ## Mathematical Framework for Advanced Training
#
# **Advanced training techniques** optimize computational efficiency and numerical stability through mathematical innovations:
#
# ### Core Mathematical Concepts
#
# **1. Numerical Precision and Stability:**
# For floating-point representation with mantissa $m$, exponent $e$, and bias $b$:
# $$\text{FP32}: (-1)^s \times 1.m \times 2^{e-127}, \quad e \in [0, 255]$$
# $$\text{FP16}: (-1)^s \times 1.m \times 2^{e-15}, \quad e \in [0, 31]$$
#
# **2. Mixed Precision Training Mathematics:**
# Forward pass in reduced precision: $\mathbf{y} = f(\mathbf{x}; \boldsymbol{\theta}_{\text{FP16}})$
# Gradient computation with scaling: $\mathbf{g}_{\text{scaled}} = S \cdot \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta})$
# Where $S$ is a loss scaling factor to prevent gradient underflow.
#
# **3. Model Compression Theory:**
# **Quantization**: Maps continuous weights $\mathbf{W} \in \mathbb{R}^{m \times n}$ to discrete set:
# $$Q(\mathbf{W}) = \text{round}\left(\frac{\mathbf{W} - Z}{S}\right) \cdot S + Z$$
# Where $S$ is scale factor and $Z$ is zero-point.
#
# **4. Production Optimization:**
# **Inference Efficiency**: For model $f$ with parameters $\boldsymbol{\theta}$:
# - **Latency**: $T_{\text{inference}} = T_{\text{compute}} + T_{\text{memory}} + T_{\text{io}}$
# - **Throughput**: $\tau = \frac{\text{batch\_size}}{T_{\text{inference}}}$
# - **Memory usage**: $M = M_{\text{model}} + M_{\text{activations}} + M_{\text{gradients}}$

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.jit
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from pathlib import Path

# %% [markdown]
# ## Mixed Precision Training
#
# ### Mathematical Foundation of Mixed Precision
#
# **Mixed Precision Training** uses both FP16 and FP32 precision to optimize memory and speed:
#
# **Numerical Stability Analysis:**
# - **FP16 range**: $\approx [6 \times 10^{-5}, 65504]$
# - **FP32 range**: $\approx [1.4 \times 10^{-45}, 3.4 \times 10^{38}]$
#
# **Gradient Scaling Mathematics:**
# To prevent gradient underflow in FP16:
# $$\mathbf{g}_{\text{scaled}} = S \cdot \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta})$$
# $$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot \frac{\mathbf{g}_{\text{scaled}}}{S}$$
#
# **Dynamic Loss Scaling:**
# $$S_{t+1} = \begin{cases}
# S_t \times 2 & \text{if no overflow for } N \text{ steps} \\
# S_t / 2 & \text{if overflow detected} \\
# S_t & \text{otherwise}
# \end{cases}$$
#
# **Memory Reduction:**
# - **Activations**: FP16 storage reduces memory by ~50%
# - **Gradients**: FP16 communication reduces bandwidth
# - **Master weights**: FP32 maintains training stability
#
# **Performance Benefits:**
# - **Tensor Cores**: Specialized hardware for FP16 matrix operations
# - **Bandwidth**: Reduced memory transfers
# - **Compute**: Faster arithmetic operations on modern GPUs

# %%
# Mixed precision training (requires CUDA)
from torch.cuda.amp import autocast, GradScaler

# Simple model for demonstration
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Create synthetic data
x = torch.randn(1000, 100)
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"Device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print("Mixed precision training will be demonstrated")
    scaler = GradScaler()
else:
    print("Mixed precision requires CUDA, will show regular training")

# %%
# Training with mixed precision (if CUDA available)
def train_with_mixed_precision(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Regular training function
def train_regular(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Train for a few epochs
if torch.cuda.is_available():
    for epoch in range(3):
        loss = train_with_mixed_precision(model, dataloader, optimizer, criterion, scaler, device)
        print(f"Epoch {epoch+1}, Mixed Precision Loss: {loss:.4f}")
else:
    for epoch in range(3):
        loss = train_regular(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Regular Loss: {loss:.4f}")

# %% [markdown]
# ## Model Quantization
#
# ### Mathematical Theory of Neural Network Quantization
#
# **Quantization** reduces model precision while maintaining accuracy through careful mathematical mapping:
#
# **Uniform Quantization Mathematics:**
# For weights $\mathbf{W} \in \mathbb{R}^{m \times n}$, quantization maps to discrete set:
# $$Q(\mathbf{W}) = \text{clip}\left(\text{round}\left(\frac{\mathbf{W}}{S}\right), q_{\min}, q_{\max}\right)$$
#
# Where:
# - **Scale factor**: $S = \frac{\max(\mathbf{W}) - \min(\mathbf{W})}{q_{\max} - q_{\min}}$
# - **Zero point**: $Z = q_{\min} - \text{round}\left(\frac{\min(\mathbf{W})}{S}\right)$
#
# **Asymmetric Quantization:**
# $$Q(\mathbf{W}) = \text{round}\left(\frac{\mathbf{W}}{S} + Z\right)$$
# $$\text{Dequantize}: \tilde{\mathbf{W}} = S(Q(\mathbf{W}) - Z)$$
#
# **Quantization Error Analysis:**
# Approximation error: $\epsilon = \mathbf{W} - \tilde{\mathbf{W}}$
# Mean squared error: $\text{MSE} = \mathbb{E}[(\mathbf{W} - \tilde{\mathbf{W}})^2]$
#
# **Post-Training Quantization:**
# 1. **Calibration**: Use representative data to compute optimal scales
# 2. **Statistical analysis**: $S = \frac{\text{percentile}(\mathbf{W}, 99.9)}{\text{MAX\_INT}}$
# 3. **Layer-wise optimization**: Minimize $\|\mathbf{W} - Q(\mathbf{W})\|_2^2$
#
# **Quantization-Aware Training:**
# Include quantization in forward pass:
# $$\mathbf{y} = f(\mathbf{x}; Q(\boldsymbol{\theta}))$$
# Straight-through estimator for gradients:
# $$\frac{\partial L}{\partial \boldsymbol{\theta}} = \frac{\partial L}{\partial Q(\boldsymbol{\theta})} \cdot \mathbf{1}$$

# %%
# Post-training quantization
import torch.quantization as quantization

# Create a model for quantization
class QuantizableNet(nn.Module):
    def __init__(self):
        super(QuantizableNet, self).__init__()
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create and train a simple model
model_fp32 = QuantizableNet()
model_fp32.eval()

# Create calibration data
calibration_data = torch.randn(100, 20)

# Post-training quantization
model_fp32.qconfig = quantization.get_default_qconfig('fbgemm')
model_prepared = quantization.prepare(model_fp32)

# Calibrate with sample data
with torch.no_grad():
    for i in range(10):
        model_prepared(calibration_data[i:i+10])

# Convert to quantized model
model_quantized = quantization.convert(model_prepared)

print("Quantization completed")
print(f"Original model size: {sum(p.numel() * 4 for p in model_fp32.parameters())} bytes (FP32)")
print(f"Quantized model created (approximately 4x smaller)")

# %%
# Compare inference speed and accuracy
test_input = torch.randn(100, 20)

# Measure FP32 model
start_time = time.time()
with torch.no_grad():
    output_fp32 = model_fp32(test_input)
fp32_time = time.time() - start_time

# Measure quantized model
start_time = time.time()
with torch.no_grad():
    output_quantized = model_quantized(test_input)
quantized_time = time.time() - start_time

# Compare outputs
mse_error = F.mse_loss(output_fp32, output_quantized)

print(f"FP32 inference time: {fp32_time:.4f} seconds")
print(f"Quantized inference time: {quantized_time:.4f} seconds")
print(f"Speedup: {fp32_time/quantized_time:.2f}x")
print(f"MSE between outputs: {mse_error:.6f}")

# %% [markdown]
# ## TorchScript and Model Serialization
#
# ### Mathematical Foundation of Graph Compilation
#
# **TorchScript** converts dynamic PyTorch models into static computational graphs for optimization:
#
# **Computational Graph Representation:**
# A neural network as directed acyclic graph $G = (V, E)$ where:
# - **Vertices**: $V = \{v_1, v_2, \ldots, v_n\}$ (operations/tensors)
# - **Edges**: $E \subseteq V \times V$ (data dependencies)
# - **Execution order**: Topological sort of $G$
#
# **Tracing Mathematics:**
# Given input $\mathbf{x}_0$, record operations:
# $$\mathbf{x}_1 = f_1(\mathbf{x}_0), \quad \mathbf{x}_2 = f_2(\mathbf{x}_1), \quad \ldots, \quad \mathbf{y} = f_n(\mathbf{x}_{n-1})$$
#
# Build graph: $G_{\text{trace}} = \{(f_i, \text{input\_shapes}_i, \text{output\_shapes}_i)\}$
#
# **Scripting vs Tracing:**
#
# **Tracing**: Records actual execution path
# - **Pros**: Captures exact computation for given input
# - **Cons**: Cannot handle dynamic control flow
#
# **Scripting**: Analyzes Python AST
# - **Pros**: Preserves control flow (if/else, loops)
# - **Cons**: Limited Python subset support
#
# **Optimization Passes:**
# 1. **Dead code elimination**: Remove unused operations
# 2. **Constant propagation**: Pre-compute constant expressions
# 3. **Operator fusion**: Combine multiple ops into efficient kernels
# 4. **Memory optimization**: Reduce intermediate allocations
#
# **Serialization Format:**
# TorchScript uses Protocol Buffers for:
# $$\text{Model} = (\text{Graph}, \text{Parameters}, \text{Metadata})$$
#
# **Just-In-Time Compilation:**
# - **Graph optimization**: $G_{\text{optimized}} = \text{optimize}(G_{\text{original}})$
# - **Kernel selection**: Choose optimal implementation for hardware
# - **Memory planning**: Minimize memory footprint

# %%
# TorchScript for model serialization and optimization
class ScriptableNet(nn.Module):
    def __init__(self):
        super(ScriptableNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create model
model = ScriptableNet()
model.eval()

# Method 1: Tracing
example_input = torch.randn(1, 3, 32, 32)
traced_model = torch.jit.trace(model, example_input)

# Method 2: Scripting
scripted_model = torch.jit.script(model)

print("TorchScript models created")
print(f"Original model type: {type(model)}")
print(f"Traced model type: {type(traced_model)}")
print(f"Scripted model type: {type(scripted_model)}")

# %%
# Save and load TorchScript models
traced_model.save('traced_model.pt')
scripted_model.save('scripted_model.pt')

# Load models
loaded_traced = torch.jit.load('traced_model.pt')
loaded_scripted = torch.jit.load('scripted_model.pt')

# Test inference
test_input = torch.randn(2, 3, 32, 32)

with torch.no_grad():
    original_output = model(test_input)
    traced_output = loaded_traced(test_input)
    scripted_output = loaded_scripted(test_input)

print("Inference test:")
print(f"Original output shape: {original_output.shape}")
print(f"Traced output shape: {traced_output.shape}")
print(f"Scripted output shape: {scripted_output.shape}")
print(f"Outputs match (traced): {torch.allclose(original_output, traced_output)}")
print(f"Outputs match (scripted): {torch.allclose(original_output, scripted_output)}")

# Clean up files
import os
os.remove('traced_model.pt')
os.remove('scripted_model.pt')

# %% [markdown]
# ## Custom CUDA Kernels and Autograd
#
# ### Mathematical Foundation of Custom Operations
#
# **Custom autograd functions** extend PyTorch's automatic differentiation for specialized operations:
#
# **Autograd System Mathematics:**
# For function $y = f(x)$, autograd maintains:
# - **Forward pass**: $y = f(x)$
# - **Backward pass**: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$
#
# **Chain Rule Implementation:**
# For composite function $L = g(f(x))$:
# $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial g} \cdot \frac{\partial g}{\partial f} \cdot \frac{\partial f}{\partial x}$$
#
# **Custom Function Design:**
# Define $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ with:
# $$\text{forward}(\mathbf{x}) = f(\mathbf{x})$$
# $$\text{backward}(\mathbf{g}) = \mathbf{J}^T \mathbf{g}$$
#
# Where $\mathbf{J} = \frac{\partial f}{\partial \mathbf{x}} \in \mathbb{R}^{m \times n}$ is the Jacobian.
#
# **CUDA Kernel Mathematics:**
# For element-wise operation $y_i = f(x_i)$:
# - **Thread mapping**: $\text{thread\_id} \rightarrow \text{array\_index}$
# - **Memory coalescing**: Adjacent threads access contiguous memory
# - **Parallel execution**: $N$ threads compute $N$ elements simultaneously
#
# **Memory Hierarchy:**
# - **Global memory**: Large, high latency
# - **Shared memory**: Fast, limited size per block
# - **Registers**: Fastest, limited per thread
#
# **Performance Optimization:**
# - **Occupancy**: $\frac{\text{active\_warps}}{\text{max\_warps}}$
# - **Memory bandwidth**: $\frac{\text{bytes\_accessed}}{\text{time}}$
# - **Arithmetic intensity**: $\frac{\text{FLOPs}}{\text{bytes\_accessed}}$

# %%
# Custom autograd function
class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Save input for backward pass
        ctx.save_for_backward(input)
        return input ** 2

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved input
        input, = ctx.saved_tensors
        # Gradient of x^2 is 2x
        return grad_output * 2 * input

# Use custom function
def custom_square(x):
    return SquareFunction.apply(x)

# Test custom function
x = torch.randn(5, requires_grad=True)
y = custom_square(x)
loss = y.sum()
loss.backward()

print(f"Input: {x}")
print(f"Output: {y}")
print(f"Gradient: {x.grad}")
print(f"Expected gradient (2x): {2 * x}")
print(f"Gradients match: {torch.allclose(x.grad, 2 * x.data)}")

# %% [markdown]
# ## Model Checkpointing and State Management
#
# ### Mathematical Framework for Training State
#
# **Model checkpointing** preserves complete training state for reproducible deep learning:
#
# **Training State Representation:**
# Complete state $S_t$ at epoch $t$ includes:
# $$S_t = (\boldsymbol{\theta}_t, \mathbf{m}_t, \mathbf{v}_t, \eta_t, \mathcal{R}_t)$$
#
# Where:
# - $\boldsymbol{\theta}_t$: Model parameters
# - $\mathbf{m}_t, \mathbf{v}_t$: Optimizer momentum/velocity states
# - $\eta_t$: Learning rate schedule state
# - $\mathcal{R}_t$: Random number generator state
#
# **Optimizer State Mathematics:**
#
# **Adam optimizer** maintains:
# $$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t$$
# $$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2$$
# $$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}$$
#
# **Learning Rate Scheduling:**
# Step decay: $\eta_t = \eta_0 \gamma^{\lfloor t/T \rfloor}$
# Exponential decay: $\eta_t = \eta_0 e^{-\lambda t}$
# Cosine annealing: $\eta_t = \eta_{\min} + (\eta_{\max} - \eta_{\min})\frac{1 + \cos(\pi t/T)}{2}$
#
# **Checkpoint Validation:**
# Ensure mathematical consistency:
# 1. **Parameter integrity**: $\|\boldsymbol{\theta}_{\text{loaded}} - \boldsymbol{\theta}_{\text{saved}}\|_2 < \epsilon$
# 2. **Optimizer state**: Verify momentum terms match
# 3. **Reproducibility**: Same random seed → same results
#
# **Distributed Checkpointing:**
# For model parallelism across $N$ devices:
# $$\boldsymbol{\theta} = [\boldsymbol{\theta}_1, \boldsymbol{\theta}_2, \ldots, \boldsymbol{\theta}_N]$$
# Each device saves: $\{\boldsymbol{\theta}_i, \text{rank}_i, \text{world\_size}\}$

# %%
# Comprehensive checkpointing
def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'timestamp': time.time()
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(filename, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Checkpoint loaded: epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")
    return checkpoint['epoch'], checkpoint['loss']

# Example usage
model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Save checkpoint
save_checkpoint(model, optimizer, scheduler, epoch=5, loss=0.123, filename='checkpoint.pth')

# Create new instances
new_model = SimpleNet()
new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=10, gamma=0.1)

# Load checkpoint
epoch, loss = load_checkpoint('checkpoint.pth', new_model, new_optimizer, new_scheduler)

# Clean up
os.remove('checkpoint.pth')

# %% [markdown]
# ## Model Profiling and Performance Analysis
#
# ### Mathematical Framework for Performance Optimization
#
# **Performance profiling** quantifies computational bottlenecks through systematic measurement:
#
# **Performance Metrics:**
#
# **Computational Complexity:**
# - **Time complexity**: $T(n) = O(f(n))$ for input size $n$
# - **Space complexity**: $S(n) = O(g(n))$ for memory usage
# - **FLOPs count**: Floating-point operations per forward/backward pass
#
# **Hardware Utilization:**
# - **GPU occupancy**: $\text{Occupancy} = \frac{\text{Active warps}}{\text{Max warps per SM}}$
# - **Memory bandwidth**: $\text{BW} = \frac{\text{Bytes transferred}}{\text{Time}}$
# - **Arithmetic intensity**: $\text{AI} = \frac{\text{FLOPs}}{\text{Bytes accessed}}$
#
# **Roofline Model:**
# Performance bound: $\text{Performance} \leq \min(\text{Peak FLOPS}, \text{AI} \times \text{Peak BW})$
#
# **Profiling Mathematics:**
#
# **Statistical Profiling:**
# Sample execution times $\{t_1, t_2, \ldots, t_n\}$:
# - **Mean**: $\bar{t} = \frac{1}{n}\sum_{i=1}^n t_i$
# - **Variance**: $\sigma^2 = \frac{1}{n-1}\sum_{i=1}^n (t_i - \bar{t})^2$
# - **Confidence interval**: $\bar{t} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$
#
# **Bottleneck Analysis:**
# For operations $\{O_1, O_2, \ldots, O_k\}$ with times $\{t_1, t_2, \ldots, t_k\}$:
# - **Critical path**: $T_{\text{total}} = \max_i(t_i)$ (parallel) or $\sum_i t_i$ (serial)
# - **Utilization**: $U_i = \frac{t_i}{T_{\text{total}}}$
# - **Amdahl's law**: $S = \frac{1}{(1-P) + \frac{P}{N}}$ where $P$ is parallelizable fraction
#
# **Memory Analysis:**
# - **Cache hit ratio**: $H = \frac{\text{Cache hits}}{\text{Total accesses}}$
# - **Memory throughput**: $\text{Throughput} = \frac{\text{Data size} \times \text{Batch size}}{\text{Time}}$
# - **Memory efficiency**: $\eta_{\text{mem}} = \frac{\text{Useful bytes}}{\text{Total bytes accessed}}$

# %%
# Profiling PyTorch operations
from torch.profiler import profile, record_function, ProfilerActivity

# Model for profiling
class ProfileNet(nn.Module):
    def __init__(self):
        super(ProfileNet, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x):
        with record_function("conv_layer"):
            x = F.relu(self.conv(x))

        with record_function("pooling"):
            x = self.pool(x)

        with record_function("flatten"):
            x = x.view(x.size(0), -1)

        with record_function("fc_layer"):
            x = self.fc(x)

        return x

model = ProfileNet()
inputs = torch.randn(8, 3, 32, 32)

# Profile the model
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        output = model(inputs)

# Print profiling results
print("Profiling Results:")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# %% [markdown]
# ## Distributed Training Mathematics
#
# ### Mathematical Foundation of Parallel Deep Learning
#
# **Distributed training** scales deep learning across multiple devices through mathematical parallelization strategies:
#
# **Data Parallelism:**
# Model replicated across $N$ devices, batch split:
# $$\mathcal{B} = \mathcal{B}_1 \cup \mathcal{B}_2 \cup \cdots \cup \mathcal{B}_N, \quad |\mathcal{B}_i| = \frac{|\mathcal{B}|}{N}$$
#
# Local gradients: $\mathbf{g}_i = \nabla_{\boldsymbol{\theta}} L(\mathcal{B}_i; \boldsymbol{\theta})$
# Global gradient: $\mathbf{g} = \frac{1}{N}\sum_{i=1}^N \mathbf{g}_i$
#
# **Model Parallelism:**
# Parameters split across devices: $\boldsymbol{\theta} = [\boldsymbol{\theta}_1, \boldsymbol{\theta}_2, \ldots, \boldsymbol{\theta}_N]$
# Sequential computation: $\mathbf{h}_i = f_i(\mathbf{h}_{i-1}; \boldsymbol{\theta}_i)$
#
# **Communication Patterns:**
#
# **AllReduce Algorithm:**
# Reduces communication complexity from $O(N^2)$ to $O(N \log N)$:
# 1. **Reduce-scatter**: Each device gets partial sum
# 2. **AllGather**: Broadcast complete result
#
# **Ring AllReduce:**
# Communication time: $T_{\text{comm}} = \alpha (N-1) + \frac{\beta M (N-1)}{N}$
# Where $\alpha$ is latency, $\beta$ is bandwidth, $M$ is message size.
#
# **Gradient Compression:**
# Reduce communication by compressing gradients:
# $$\tilde{\mathbf{g}} = Q(\mathbf{g}) \text{ where } \mathbb{E}[Q(\mathbf{g})] = \mathbf{g}$$
#
# **Scaling Analysis:**
#
# **Linear Speedup**: $S(N) = N$ (ideal case)
# **Strong scaling**: Fixed problem size, increase processors
# **Weak scaling**: Fixed problem size per processor
#
# **Efficiency**: $E(N) = \frac{S(N)}{N} = \frac{T(1)}{N \cdot T(N)}$
#
# **Communication-Computation Overlap:**
# Overlap gradient computation with communication:
# $$T_{\text{total}} = \max(T_{\text{compute}}, T_{\text{communication}})$$

# %%
# Distributed training setup (conceptual - requires multiple GPUs/nodes)
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    # This is conceptual - actual implementation depends on your setup
    print(f"Setting up distributed training: rank {rank}, world_size {world_size}")
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    """Clean up distributed training"""
    # dist.destroy_process_group()
    pass

# Data parallel training (single machine, multiple GPUs)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
else:
    print("Single GPU/CPU training")

print("Distributed training concepts demonstrated")
print("For actual multi-GPU training:")
print("1. Use torch.nn.DataParallel for single-machine multi-GPU")
print("2. Use torch.nn.parallel.DistributedDataParallel for multi-machine")
print("3. Initialize process groups with torch.distributed")
print("4. Use torch.multiprocessing for spawning processes")

# %% [markdown]
# ## Production Deployment Mathematics
#
# ### Mathematical Framework for Model Serving
#
# **Model deployment** requires mathematical analysis of performance, scalability, and reliability:
#
# **Inference Serving Mathematics:**
#
# **Latency Analysis:**
# Total inference time: $T_{\text{inference}} = T_{\text{preprocess}} + T_{\text{forward}} + T_{\text{postprocess}}$
#
# **Batch Processing:**
# For batch size $B$:
# - **Throughput**: $\tau(B) = \frac{B}{T_{\text{batch}}(B)}$
# - **Optimal batch size**: $B^* = \arg\max_B \tau(B)$ subject to memory constraints
#
# **Queue Theory:**
# Using M/M/1 queue model for request serving:
# - **Arrival rate**: $\lambda$ requests/second
# - **Service rate**: $\mu$ requests/second
# - **Utilization**: $\rho = \frac{\lambda}{\mu} < 1$ for stability
# - **Average latency**: $L = \frac{1}{\mu - \lambda}$
#
# **Load Balancing:**
# For $N$ servers with capacities $\{\mu_1, \mu_2, \ldots, \mu_N\}$:
# Optimal routing probabilities: $p_i = \frac{\mu_i}{\sum_{j=1}^N \mu_j}$
#
# **Resource Allocation:**
#
# **Auto-scaling Mathematics:**
# Scale replicas based on metrics:
# $$R(t+1) = \max\left(R_{\min}, \min\left(R_{\max}, R(t) \cdot \frac{M(t)}{M_{\text{target}}}\right)\right)$$
#
# Where $M(t)$ is current metric (CPU, memory, latency).
#
# **Cost Optimization:**
# Total cost: $C = C_{\text{compute}} + C_{\text{storage}} + C_{\text{network}}$
# Minimize: $\min C$ subject to SLA constraints $L \leq L_{\max}, A \geq A_{\min}$
#
# **A/B Testing Mathematics:**
#
# **Statistical Significance:**
# For metrics $X_A, X_B$ with means $\mu_A, \mu_B$:
# $$t = \frac{\bar{X}_A - \bar{X}_B}{\sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}}$$
#
# **Sample Size Calculation:**
# Required samples: $n = \frac{2(z_{\alpha/2} + z_\beta)^2 \sigma^2}{(\mu_1 - \mu_2)^2}$

# %%
# Model versioning and metadata
class ModelMetadata:
    def __init__(self, model_name, version, input_shape, output_shape, classes=None):
        self.model_name = model_name
        self.version = version
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.classes = classes
        self.created_at = time.time()

    def to_dict(self):
        return {
            'model_name': self.model_name,
            'version': self.version,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'classes': self.classes,
            'created_at': self.created_at
        }

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

# Create model package
def package_model(model, metadata, model_path='model_package'):
    """
    Package model with metadata for deployment
    """
    Path(model_path).mkdir(exist_ok=True)

    # Save model
    torch.save(model.state_dict(), f'{model_path}/model.pth')

    # Save TorchScript version
    scripted = torch.jit.script(model)
    scripted.save(f'{model_path}/model_scripted.pt')

    # Save metadata
    metadata.save(f'{model_path}/metadata.json')

    print(f"Model packaged in {model_path}/")
    print(f"Contents: model.pth, model_scripted.pt, metadata.json")

# Example packaging
model = SimpleNet()
metadata = ModelMetadata(
    model_name='SimpleNet',
    version='1.0.0',
    input_shape=[None, 100],
    output_shape=[None, 10],
    classes=list(range(10))
)

package_model(model, metadata)

# Clean up
import shutil
shutil.rmtree('model_package')

# %% [markdown]
# ## Model Serving Infrastructure
#
# ### Mathematical Foundation of Scalable Inference
#
# **Model serving infrastructure** requires mathematical optimization of throughput, latency, and resource utilization:
#
# **Batching Mathematics:**
#
# **Dynamic Batching:**
# Collect requests for time $T$ or until batch size $B$:
# $$\text{Batch ready when: } |\mathcal{B}(t)| \geq B \text{ OR } t - t_{\text{first}} \geq T$$
#
# **Optimal Batch Size:**
# Balance latency vs throughput:
# $$B^* = \arg\max_B \frac{B \cdot \text{QPS}(B)}{\text{Latency}(B)}$$
#
# **Memory Management:**
# GPU memory usage: $M_{\text{GPU}} = M_{\text{model}} + B \times M_{\text{activation}}$
# Constraint: $M_{\text{GPU}} \leq M_{\text{available}}$
#
# **Performance Modeling:**
#
# **Little's Law:**
# Average latency related to throughput and concurrency:
# $$L = \lambda \times W$$
# Where $L$ = average requests in system, $\lambda$ = arrival rate, $W$ = average response time.
#
# **Utilization Theory:**
# For server with capacity $C$ and load $\rho$:
# - **Response time**: $T = \frac{T_{\text{service}}}{1-\rho}$ where $\rho = \frac{\lambda}{C}$
# - **Queue length**: $E[N] = \frac{\rho^2}{1-\rho}$
#
# **Caching Mathematics:**
#
# **Cache Hit Rate:**
# For LRU cache with capacity $C$ and request pattern:
# $$\text{Hit Rate} = 1 - \frac{\text{Unique requests in window}}{\text{Total requests}}$$
#
# **Cache Performance:**
# Effective latency: $L_{\text{eff}} = h \times L_{\text{cache}} + (1-h) \times L_{\text{compute}}$
# Where $h$ is hit rate.
#
# **Multi-Level Serving:**
#
# **Cascade Architecture:**
# Route requests based on complexity:
# $$f(\mathbf{x}) = \begin{cases}
# f_{\text{fast}}(\mathbf{x}) & \text{if confidence} > \theta \\
# f_{\text{accurate}}(\mathbf{x}) & \text{otherwise}
# \end{cases}$$
#
# **Cost-Performance Trade-off:**
# Total cost: $C = C_1 \times p_1 + C_2 \times (1-p_1)$
# Where $p_1$ is fraction routed to fast model, $C_i$ is cost per request.

# %%
# Simple model server class
class ModelServer:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

    def preprocess(self, input_data):
        """Preprocess input data"""
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data).float()
        return input_data.to(self.device)

    def predict(self, input_data):
        """Make prediction"""
        processed_input = self.preprocess(input_data)

        with torch.no_grad():
            output = self.model(processed_input)

        return output.cpu().numpy()

    def predict_batch(self, batch_data):
        """Batch prediction"""
        results = []
        for data in batch_data:
            result = self.predict(data)
            results.append(result)
        return results

# Example usage (conceptual)
print("Model Server Example:")
print("""# Usage:
server = ModelServer('model.pt', device='cuda')
predictions = server.predict(input_data)
batch_predictions = server.predict_batch([data1, data2, data3])
""")

print("\nDeployment considerations:")
print("1. Model format: TorchScript for production")
print("2. Device management: CPU vs GPU")
print("3. Batch processing for efficiency")
print("4. Input validation and preprocessing")
print("5. Error handling and logging")
print("6. Model versioning and A/B testing")
print("7. Monitoring and metrics")

# %% [markdown]
# ## Performance Optimization Theory
#
# ### Mathematical Principles of Deep Learning Optimization
#
# **Performance optimization** applies mathematical analysis to identify and eliminate computational bottlenecks:
#
# **Complexity Analysis:**
#
# **Computational Complexity:**
# For neural network layers:
# - **Linear**: $O(n \times m)$ for $n \times m$ weight matrix
# - **Convolution**: $O(C_{\text{in}} \times C_{\text{out}} \times K^2 \times H \times W)$
# - **Attention**: $O(n^2 \times d)$ for sequence length $n$, dimension $d$
#
# **Memory Complexity:**
# Total memory: $M = M_{\text{parameters}} + M_{\text{activations}} + M_{\text{gradients}} + M_{\text{optimizer}}$
#
# **Roofline Performance Model:**
# Performance bound by compute or memory:
# $$\text{Performance} \leq \min\left(\text{Peak FLOPS}, \text{Arithmetic Intensity} \times \text{Peak Bandwidth}\right)$$
#
# **Optimization Strategies:**
#
# **Operator Fusion:**
# Combine operations to reduce memory transfers:
# $$y = \text{ReLU}(\text{BatchNorm}(\text{Conv}(x))) \rightarrow y = \text{FusedConvBNReLU}(x)$$
#
# **Memory Pool Optimization:**
# Minimize fragmentation through optimal allocation:
# $$\text{Total Memory} = \max_{t} \sum_{i \text{ alive at } t} \text{size}(i)$$
#
# **Kernel Optimization:**
# Optimize CUDA kernels for:
# - **Memory coalescing**: Adjacent threads access contiguous memory
# - **Occupancy**: $\frac{\text{Active warps}}{\text{Max warps per SM}}$
# - **Register usage**: Balance between occupancy and performance
#
# **Algorithmic Optimizations:**
#
# **Gradient Accumulation:**
# Effective batch size $B_{\text{eff}} = B \times N_{\text{acc}}$:
# $$\mathbf{g}_{\text{eff}} = \frac{1}{N_{\text{acc}}} \sum_{i=1}^{N_{\text{acc}}} \mathbf{g}_i$$
#
# **Checkpointing:**
# Trade compute for memory:
# $$\text{Memory} = O(\sqrt{n}), \text{Compute} = O(n) + O(\sqrt{n})$$
# where $n$ is number of layers.
#
# **Mixed Precision Benefits:**
# - **Memory**: ~50% reduction in activation storage
# - **Bandwidth**: 2× improvement in data transfer
# - **Compute**: Tensor Core acceleration for matrix operations

# %%
# Performance optimization demonstrations
def benchmark_operations():
    """Benchmark different operations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test data
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)

    # Benchmark matrix multiplication
    start_time = time.time()
    for _ in range(100):
        z = torch.mm(x, y)
    mm_time = time.time() - start_time

    # Benchmark element-wise operations
    start_time = time.time()
    for _ in range(100):
        z = x * y
    ew_time = time.time() - start_time

    print(f"Device: {device}")
    print(f"Matrix multiplication (100 iterations): {mm_time:.4f} seconds")
    print(f"Element-wise multiplication (100 iterations): {ew_time:.4f} seconds")

benchmark_operations()

print("\nPerformance Optimization Tips:")
print("\n1. Memory Management:")
print("   - Use torch.no_grad() for inference")
print("   - Clear cache with torch.cuda.empty_cache()")
print("   - Use inplace operations when possible")

print("\n2. Data Loading:")
print("   - Use multiple workers in DataLoader")
print("   - Pin memory for GPU transfers")
print("   - Preprocess data offline when possible")

print("\n3. Model Optimization:")
print("   - Use TorchScript for production")
print("   - Consider model quantization")
print("   - Use mixed precision training")
print("   - Optimize batch sizes")

print("\n4. GPU Utilization:")
print("   - Ensure sufficient batch sizes")
print("   - Use tensor cores (mixed precision)")
print("   - Minimize CPU-GPU transfers")
print("   - Use asynchronous operations")

# %% [markdown]
# ## Production Checklist

# %%
print("PyTorch Production Deployment Checklist:")
print("\n✓ Model Development:")
print("  □ Model architecture finalized")
print("  □ Training pipeline validated")
print("  □ Model performance meets requirements")
print("  □ Hyperparameters tuned")

print("\n✓ Model Optimization:")
print("  □ TorchScript conversion tested")
print("  □ Quantization applied (if needed)")
print("  □ Model size optimized")
print("  □ Inference speed benchmarked")

print("\n✓ Data Pipeline:")
print("  □ Input validation implemented")
print("  □ Preprocessing pipeline tested")
print("  □ Data format standardized")
print("  □ Error handling for malformed inputs")

print("\n✓ Infrastructure:")
print("  □ Serving infrastructure chosen")
print("  □ Scaling strategy defined")
print("  □ Load balancing configured")
print("  □ Health checks implemented")

print("\n✓ Monitoring & Logging:")
print("  □ Model performance metrics")
print("  □ Inference latency monitoring")
print("  □ Error rate tracking")
print("  □ Resource utilization monitoring")

print("\n✓ Testing & Validation:")
print("  □ Unit tests for all components")
print("  □ Integration tests")
print("  □ Load testing completed")
print("  □ Model drift detection")

print("\n✓ Deployment & Maintenance:")
print("  □ Model versioning strategy")
print("  □ Rollback procedures")
print("  □ A/B testing framework")
print("  □ Continuous integration setup")
print("  □ Documentation completed")

print("\n✓ Security & Compliance:")
print("  □ Input sanitization")
print("  □ Model security assessment")
print("  □ Data privacy compliance")
print("  □ Access control implemented")

# %% [markdown]
# ## Summary and Next Steps

# %%
print("PyTorch Learning Journey Summary:")
print("\nPart 1: Tensors and Fundamentals")
print("  - Tensor creation, operations, and manipulation")
print("  - GPU acceleration and device management")
print("  - NumPy integration")

print("\nPart 2: Autograd and Neural Networks")
print("  - Automatic differentiation")
print("  - Building neural networks with nn.Module")
print("  - Loss functions and optimizers")
print("  - Training loops")

print("\nPart 3: Data Loading and Datasets")
print("  - Custom datasets and data loaders")
print("  - Data preprocessing and augmentation")
print("  - Train/validation/test splits")
print("  - Best practices for data handling")

print("\nPart 4: CNNs and Computer Vision")
print("  - Convolutional layers and operations")
print("  - CNN architectures and training")
print("  - Image preprocessing and augmentation")
print("  - Transfer learning")

print("\nPart 5: Advanced Topics and Production")
print("  - Mixed precision training")
print("  - Model quantization and optimization")
print("  - TorchScript and deployment")
print("  - Production considerations")

print("\nNext Steps for Advanced PyTorch:")
print("1. Explore specific domains (NLP, RL, GNNs)")
print("2. Learn distributed training for large models")
print("3. Master deployment frameworks (TorchServe, ONNX)")
print("4. Dive into PyTorch ecosystem libraries")
print("5. Contribute to open-source PyTorch projects")
print("6. Stay updated with latest PyTorch releases")

print("\nUseful Resources:")
print("- PyTorch Documentation: pytorch.org/docs")
print("- PyTorch Tutorials: pytorch.org/tutorials")
print("- Papers With Code: paperswithcode.com")
print("- PyTorch Forums: discuss.pytorch.org")
print("- GitHub: github.com/pytorch/pytorch")