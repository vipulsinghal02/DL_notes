# %% [markdown]
# # PyTorch Basics Part 4: Convolutional Neural Networks and Computer Vision
#
# CNNs, image processing, and computer vision fundamentals with PyTorch
#
# ## Mathematical Foundation of Convolutional Neural Networks
#
# **Convolutional Neural Networks (CNNs)** are specialized neural networks designed for processing grid-like data, particularly images. They leverage mathematical operations that preserve spatial relationships and reduce parameter count through weight sharing.
#
# ### Core Mathematical Concepts
#
# **1. Convolution Operation:**
# For 2D discrete convolution between input $I$ and kernel $K$:
# $$S(i,j) = (I * K)(i,j) = \sum_{m} \sum_{n} I(m,n) \cdot K(i-m, j-n)$$
#
# **2. Cross-Correlation (CNN Implementation):**
# $$S(i,j) = \sum_{m} \sum_{n} I(i+m, j+n) \cdot K(m,n)$$
#
# **3. Multi-channel Convolution:**
# For input $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$ and filters $\mathbf{W} \in \mathbb{R}^{F \times C \times K_H \times K_W}$:
# $$Y_{f,i,j} = \sum_{c=1}^{C} \sum_{u=0}^{K_H-1} \sum_{v=0}^{K_W-1} X_{c,i+u,j+v} \cdot W_{f,c,u,v} + b_f$$
#
# **4. Spatial Dimensions:**
# Output size: $H_{out} = \lfloor \frac{H_{in} + 2P - K_H}{S} \rfloor + 1$, $W_{out} = \lfloor \frac{W_{in} + 2P - K_W}{S} \rfloor + 1$
#
# Where $P$ = padding, $S$ = stride, $K$ = kernel size.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# %% [markdown]
# ## Understanding Convolutions
#
# ### Mathematical Intuition of Convolution Kernels
#
# **Convolution kernels** are learned feature detectors that respond to specific patterns in the input:
#
# **Feature Detection Mathematics:**
# Each kernel $\mathbf{K} \in \mathbb{R}^{K_H \times K_W}$ computes:
# $$\text{response}_{i,j} = \sum_{u,v} I_{i+u,j+v} \cdot K_{u,v}$$
#
# **Common Kernel Types:**
# - **Edge Detection**: $\begin{bmatrix} -1 & -1 & -1 \\ -1 & 8 & -1 \\ -1 & -1 & -1 \end{bmatrix}$ (high response to edges)
# - **Blur**: $\frac{1}{9}\begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}$ (local averaging)
# - **Sharpen**: $\begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}$ (emphasizes differences)
#
# **Translation Equivariance:**
# If $f$ is the convolution operation: $f(T_{\mathbf{v}}(\mathbf{x})) = T_{\mathbf{v}}(f(\mathbf{x}))$
# where $T_{\mathbf{v}}$ is translation by vector $\mathbf{v}$.
#
# This means convolution preserves spatial relationships - a pattern detected at one location will be detected if it appears elsewhere.

# %%
# Create a simple 2D convolution
conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

# Create a sample image (1 channel, 5x5)
image = torch.tensor([[[[1, 0, 1, 0, 1],
                        [0, 1, 0, 1, 0],
                        [1, 0, 1, 0, 1],
                        [0, 1, 0, 1, 0],
                        [1, 0, 1, 0, 1]]]], dtype=torch.float32)

print(f"Input shape: {image.shape}")
print(f"Input image:\n{image.squeeze()}")

# Apply convolution
output = conv_layer(image)
print(f"\nOutput shape: {output.shape}")
print(f"Conv weight shape: {conv_layer.weight.shape}")
print(f"Conv bias shape: {conv_layer.bias.shape}")

# %%
# Visualize different kernel effects
def apply_kernel(image, kernel):
    """Apply a custom kernel to an image"""
    conv = nn.Conv2d(1, 1, kernel_size=kernel.shape[0], padding=1, bias=False)
    with torch.no_grad():
        conv.weight[0, 0] = kernel
    return conv(image)

# Create a larger test image
test_image = torch.zeros(1, 1, 10, 10)
test_image[0, 0, 3:7, 3:7] = 1  # White square in the middle

# Define different kernels
edge_kernel = torch.tensor([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]], dtype=torch.float32)

blur_kernel = torch.ones(3, 3) / 9

sharpen_kernel = torch.tensor([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]], dtype=torch.float32)

# Apply kernels
edge_result = apply_kernel(test_image, edge_kernel)
blur_result = apply_kernel(test_image, blur_kernel)
sharpen_result = apply_kernel(test_image, sharpen_kernel)

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(test_image.squeeze(), cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 1].imshow(edge_result.squeeze(), cmap='gray')
axes[0, 1].set_title('Edge Detection')
axes[1, 0].imshow(blur_result.squeeze(), cmap='gray')
axes[1, 0].set_title('Blur')
axes[1, 1].imshow(sharpen_result.squeeze(), cmap='gray')
axes[1, 1].set_title('Sharpen')

for ax in axes.flat:
    ax.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## CNN Layers and Operations
#
# ### Mathematical Framework of CNN Components
#
# **1. Convolution Layer Parameters:**
# For `Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=K, stride=S, padding=P)`:
#
# - **Weight tensor**: $\mathbf{W} \in \mathbb{R}^{C_{out} \times C_{in} \times K \times K}$
# - **Bias vector**: $\mathbf{b} \in \mathbb{R}^{C_{out}}$
# - **Parameter count**: $C_{out} \times C_{in} \times K^2 + C_{out}$
#
# **2. Stride Effect:**
# Stride $S$ subsamples the output by factor $S$:
# $$Y_{i,j} = \sum_{c,u,v} X_{c, S \cdot i + u, S \cdot j + v} \cdot W_{c,u,v}$$
#
# **3. Padding Mathematics:**
# Zero-padding extends input dimensions:
# - **'SAME' padding**: $P = \lfloor K/2 \rfloor$ (output size = input size when $S=1$)
# - **'VALID' padding**: $P = 0$ (no padding)
#
# **4. Receptive Field:**
# The receptive field grows with depth:
# $$RF_{\ell} = RF_{\ell-1} + (K_{\ell} - 1) \prod_{i=1}^{\ell-1} S_i$$
#
# Starting with $RF_0 = 1$ and accumulating kernel sizes weighted by stride products.

# %%
# Different convolution parameters
input_tensor = torch.randn(1, 3, 32, 32)  # Batch=1, Channels=3, Height=32, Width=32

# Different kernel sizes
conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
conv2 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
conv3 = nn.Conv2d(3, 16, kernel_size=7, padding=3)

out1 = conv1(input_tensor)
out2 = conv2(input_tensor)
out3 = conv3(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Conv 3x3 output: {out1.shape}")
print(f"Conv 5x5 output: {out2.shape}")
print(f"Conv 7x7 output: {out3.shape}")

# Stride and padding effects
conv_stride = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
out_stride = conv_stride(input_tensor)
print(f"Conv with stride=2: {out_stride.shape}")

# %% [markdown]
# ### Pooling Operations - Mathematical Foundation
#
# **Pooling** provides spatial downsampling and translation invariance:
#
# **1. Max Pooling:**
# $$P_{\text{max}}(i,j) = \max_{(u,v) \in \mathcal{R}_{i,j}} X(u,v)$$
#
# Where $\mathcal{R}_{i,j}$ is the pooling window at position $(i,j)$.
#
# **2. Average Pooling:**
# $$P_{\text{avg}}(i,j) = \frac{1}{|\mathcal{R}_{i,j}|} \sum_{(u,v) \in \mathcal{R}_{i,j}} X(u,v)$$
#
# **3. Adaptive Pooling:**
# Forces output to specific size regardless of input dimensions:
# $$\text{AdaptivePool}: \mathbb{R}^{C \times H_{in} \times W_{in}} \rightarrow \mathbb{R}^{C \times H_{out} \times W_{out}}$$
#
# **Mathematical Benefits:**
# - **Translation invariance**: Small translations don't affect pooled output
# - **Dimensionality reduction**: Reduces spatial dimensions by factor of kernel size
# - **Computational efficiency**: Fewer parameters in subsequent layers
#
#
# ## Building a Simple CNN
#
# ### Mathematical Architecture Design
#
# **CNN Architecture** combines convolution, activation, and pooling layers:
#
# $$\mathbf{h}^{(\ell+1)} = \text{Pool}(\sigma(\text{Conv}(\mathbf{h}^{(\ell)})))$$
#
# **Layer-wise Transformations:**
# 1. **Convolution**: $\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)} * \mathbf{h}^{(\ell-1)} + \mathbf{b}^{(\ell)}$
# 2. **Activation**: $\mathbf{a}^{(\ell)} = \sigma(\mathbf{z}^{(\ell)})$
# 3. **Pooling**: $\mathbf{h}^{(\ell)} = \text{Pool}(\mathbf{a}^{(\ell)})$
#
# **Spatial Dimension Tracking:**
# For input $(32 \times 32)$ through our CNN:
# - Conv1 (3×3, pad=1): $32 \times 32 \rightarrow 32 \times 32$
# - Pool1 (2×2): $32 \times 32 \rightarrow 16 \times 16$
# - Conv2 (3×3, pad=1): $16 \times 16 \rightarrow 16 \times 16$
# - Pool2 (2×2): $16 \times 16 \rightarrow 8 \times 8$
# - Conv3 (3×3, pad=1): $8 \times 8 \rightarrow 8 \times 8$
# - Pool3 (2×2): $8 \times 8 \rightarrow 4 \times 4$
#
# **Parameter Count Analysis:**
# $$\text{Total params} = \sum_{\ell} (C_{in}^{(\ell)} \times C_{out}^{(\ell)} \times K^2 + C_{out}^{(\ell)}) + \text{FC params}$$

# %%
# Pooling operations
pooling_input = torch.randn(1, 16, 32, 32)

# Max pooling
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
max_pooled = max_pool(pooling_input)

# Average pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
avg_pooled = avg_pool(pooling_input)

# Adaptive pooling (output size fixed regardless of input)
adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
adaptive_pooled = adaptive_pool(pooling_input)

print(f"Input shape: {pooling_input.shape}")
print(f"Max pooled: {max_pooled.shape}")
print(f"Avg pooled: {avg_pooled.shape}")
print(f"Adaptive pooled: {adaptive_pooled.shape}")

# %% [markdown]
# Create a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # After 3 pooling operations: 32->16->8->4
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Create model instance
model = SimpleCNN(num_classes=10)

# Check model parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")

# Print model architecture
print("\nModel Architecture:")
print(model)

# %%
# Test forward pass
test_input = torch.randn(4, 3, 32, 32)  # Batch of 4 images
output = model(test_input)
print(f"Input shape: {test_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Output logits for first sample: {output[0]}")

# %% [markdown]
# ## Working with CIFAR-10 Dataset
#
# ### Mathematical Dataset Characteristics
#
# **CIFAR-10 Dataset Structure:**
# - **Input space**: $\mathbf{X} \in \mathbb{R}^{3 \times 32 \times 32}$ (RGB images)
# - **Label space**: $y \in \{0, 1, 2, \ldots, 9\}$ (10 classes)
# - **Dataset size**: $|\mathcal{D}| = 60,000$ (50k train + 10k test)
#
# **Pixel Value Normalization:**
# $$\tilde{x}_{i,j,c} = \frac{x_{i,j,c} - \mu_c}{\sigma_c}$$
#
# Where $\mu_c, \sigma_c$ are channel-wise mean and standard deviation.
#
# **Classification Objective:**
# $$P(y = k | \mathbf{x}) = \frac{e^{f_k(\mathbf{x})}}{\sum_{j=1}^{10} e^{f_j(\mathbf{x})}}$$
#
# Where $f(\mathbf{x}) \in \mathbb{R}^{10}$ are the network logits.
#
# **Cross-Entropy Loss:**
# $$L = -\frac{1}{N} \sum_{i=1}^{N} \log P(y_i | \mathbf{x}_i) = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{f_{y_i}(\mathbf{x}_i)}}{\sum_{k=1}^{10} e^{f_k(\mathbf{x}_i)}}$$
#
# This loss encourages the correct class logit to be larger than others.

# %%
# Create synthetic CIFAR-10-like data for demonstration
# (In practice, you would use torchvision.datasets.CIFAR10)

def create_synthetic_cifar():
    # Create random images and labels
    images = torch.randn(1000, 3, 32, 32)
    labels = torch.randint(0, 10, (1000,))
    return images, labels

# Create datasets
train_images, train_labels = create_synthetic_cifar()
test_images, test_labels = create_synthetic_cifar()

# Create datasets and loaders
train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# CIFAR-10 class names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Classes: {classes}")

# %% [markdown]
# ## Data Augmentation for Images
#
# ### Mathematical Foundation of Data Augmentation
#
# **Data augmentation** applies transformation $T$ to create additional training examples:
# $$\mathcal{D}_{\text{aug}} = \mathcal{D} \cup \{(T(\mathbf{x}_i), y_i) : (\mathbf{x}_i, y_i) \in \mathcal{D}, T \in \mathcal{T}\}$$
#
# **Common Transformations:**
#
# **1. Geometric Transformations:**
# - **Horizontal Flip**: $T_{\text{flip}}(x_{i,j}) = x_{i, W-1-j}$
# - **Rotation**: $T_{\text{rot}}(\mathbf{x})$ using rotation matrix $\mathbf{R}(\theta)$
# - **Translation**: $T_{\text{trans}}(x_{i,j}) = x_{i-\Delta_y, j-\Delta_x}$
#
# **2. Photometric Transformations:**
# - **Brightness**: $T_{\text{bright}}(\mathbf{x}) = \mathbf{x} + \beta$
# - **Contrast**: $T_{\text{contrast}}(\mathbf{x}) = \alpha \cdot \mathbf{x}$
# - **Color Jitter**: Independent channel scaling
#
# **Mathematical Benefits:**
# 1. **Increased dataset size**: $|\mathcal{D}_{\text{aug}}| > |\mathcal{D}|$
# 2. **Invariance learning**: Model learns $f(T(\mathbf{x})) \approx f(\mathbf{x})$
# 3. **Regularization**: Prevents overfitting to specific orientations/lighting
# 4. **Better generalization**: Augmented data approximates real-world variations
#
# **Normalization Post-Augmentation:**
# $$\tilde{\mathbf{x}} = \frac{T(\mathbf{x}) - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$$
#
# Applied after geometric transformations but before training.

# %%
# Image transformations
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Demonstrate augmentation
sample_image = torch.rand(3, 32, 32)

# Apply transformations multiple times
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(sample_image.permute(1, 2, 0))
axes[0, 0].set_title('Original')

for i in range(5):
    row = (i + 1) // 3
    col = (i + 1) % 3
    augmented = transform_train(sample_image)
    # Denormalize for display
    augmented = augmented * 0.5 + 0.5
    axes[row, col].imshow(augmented.permute(1, 2, 0))
    axes[row, col].set_title(f'Augmented {i+1}')

for ax in axes.flat:
    ax.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Training the CNN
#
# ### Mathematical Training Framework
#
# **CNN Training** optimizes parameters via stochastic gradient descent:
#
# $$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}_t)$$
#
# **Learning Rate Scheduling:**
# $$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/T \rfloor}$$
#
# Where $\gamma < 1$ is decay factor, $T$ is step size.
#
# **Backpropagation in CNNs:**
# Gradients flow backward through:
# 1. **Fully connected layers**: Standard matrix multiplication gradients
# 2. **Pooling layers**: Gradient routing (max pool) or averaging (avg pool)
# 3. **Convolution layers**: Convolution of gradients with flipped kernels
#
# **Convolution Gradient:**
# $$\frac{\partial L}{\partial W_{f,c,u,v}} = \sum_{i,j} \frac{\partial L}{\partial Y_{f,i,j}} \cdot X_{c,i+u,j+v}$$
#
# **Device Optimization:**
# - **GPU acceleration**: Parallel convolution computation
# - **Memory management**: Activation checkpointing for large models
# - **Mixed precision**: FP16 computation with FP32 accumulation

# %%
# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = SimpleCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

print(f"Model moved to {device}")

# %%
# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    return avg_loss, accuracy

# Evaluation function
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(test_loader)
    return avg_loss, accuracy

# %%
# Training loop
num_epochs = 20
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

    # Evaluate
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    # Update learning rate
    scheduler.step()

    # Store metrics
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print()

# %%
# Plot training progress
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
ax1.plot(train_losses, label='Train Loss')
ax1.plot(test_losses, label='Test Loss')
ax1.set_title('Training and Test Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Accuracy plot
ax2.plot(train_accuracies, label='Train Accuracy')
ax2.plot(test_accuracies, label='Test Accuracy')
ax2.set_title('Training and Test Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Advanced CNN Architectures

# %%
# ResNet-like residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        out = F.relu(out)
        return out

# Simple ResNet
class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create and test ResNet
resnet = SimpleResNet(num_classes=10)
test_input = torch.randn(2, 3, 32, 32)
output = resnet(test_input)

print(f"ResNet output shape: {output.shape}")
print(f"ResNet parameters: {sum(p.numel() for p in resnet.parameters()):,}")

# %% [markdown]
# ## Feature Visualization

# %%
# Visualize learned filters
def visualize_filters(model, layer_name='conv1'):
    # Get the first convolutional layer
    conv_layer = getattr(model, layer_name)
    weights = conv_layer.weight.data

    # Normalize weights for visualization
    weights = (weights - weights.min()) / (weights.max() - weights.min())

    # Plot first 8 filters
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        row = i // 4
        col = i % 4

        # For RGB filters, show as RGB or grayscale
        if weights.shape[1] == 3:  # RGB input
            filter_img = weights[i].permute(1, 2, 0)
        else:  # Grayscale or single channel
            filter_img = weights[i, 0]

        axes[row, col].imshow(filter_img, cmap='gray' if len(filter_img.shape) == 2 else None)
        axes[row, col].set_title(f'Filter {i+1}')
        axes[row, col].axis('off')

    plt.suptitle(f'Learned Filters in {layer_name}')
    plt.tight_layout()
    plt.show()

# Visualize filters from the trained model
visualize_filters(model, 'conv1')

# %% [markdown]
# ## Transfer Learning Example
#
# ### Mathematical Foundation of Transfer Learning
#
# **Transfer Learning** leverages pre-trained features for new tasks:
#
# $$f_{\text{new}}(\mathbf{x}) = g(h_{\text{pretrained}}(\mathbf{x}))$$
#
# Where:
# - $h_{\text{pretrained}}: \mathbb{R}^{H \times W \times C} \rightarrow \mathbb{R}^d$ (frozen feature extractor)
# - $g: \mathbb{R}^d \rightarrow \mathbb{R}^K$ (trainable classifier)
#
# **Mathematical Justification:**
# If source and target domains share low-level features, then:
# $$P_{\text{source}}(\text{features}) \approx P_{\text{target}}(\text{features})$$
#
# **Training Strategies:**
#
# **1. Feature Extraction:**
# $$\boldsymbol{\theta}_{\text{features}} \text{ frozen}, \quad \min_{\boldsymbol{\theta}_{\text{classifier}}} L(\boldsymbol{\theta}_{\text{classifier}})$$
#
# **2. Fine-tuning:**
# $$\min_{\boldsymbol{\theta}_{\text{all}}} L(\boldsymbol{\theta}_{\text{features}}, \boldsymbol{\theta}_{\text{classifier}})$$
#
# With reduced learning rate: $\eta_{\text{features}} \ll \eta_{\text{classifier}}$
#
# **Parameter Efficiency:**
# - **Full training**: $|\boldsymbol{\theta}_{\text{all}}|$ parameters
# - **Transfer learning**: $|\boldsymbol{\theta}_{\text{classifier}}| \ll |\boldsymbol{\theta}_{\text{all}}|$
#
# **Mathematical Benefits:**
# 1. **Reduced overfitting**: Fewer trainable parameters
# 2. **Faster convergence**: Good initialization from pre-training
# 3. **Data efficiency**: Effective with small target datasets

# %%
# Example of transfer learning with a pre-trained model
# (Using a simplified version - normally you'd use torchvision.models)

class PretrainedFeatureExtractor(nn.Module):
    def __init__(self):
        super(PretrainedFeatureExtractor, self).__init__()
        # Simulating pre-trained convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

    def forward(self, x):
        return self.features(x)

class TransferLearningModel(nn.Module):
    def __init__(self, num_classes=10, freeze_features=True):
        super(TransferLearningModel, self).__init__()

        # Pre-trained feature extractor
        self.feature_extractor = PretrainedFeatureExtractor()

        # Freeze pre-trained parameters if specified
        if freeze_features:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # New classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

# Create transfer learning model
transfer_model = TransferLearningModel(num_classes=10, freeze_features=True)

# Count trainable parameters
trainable_params = sum(p.numel() for p in transfer_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in transfer_model.parameters())

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {total_params - trainable_params:,}")
print(f"Fraction trainable: {trainable_params/total_params:.3f}")

# Test forward pass
test_input = torch.randn(2, 3, 32, 32)
output = transfer_model(test_input)
print(f"Output shape: {output.shape}")

# %% [markdown]
# ## CNN Best Practices
#
# ### Mathematical Principles Behind Best Practices
#
# **1. Data Preprocessing Mathematics:**
# - **Normalization**: $\tilde{\mathbf{x}} = \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$ improves optimization conditioning
# - **Augmentation**: $|\mathcal{D}_{\text{aug}}| \gg |\mathcal{D}|$ provides implicit regularization
# - **Batch size**: Trade-off between gradient noise ($\propto 1/\sqrt{B}$) and memory
#
# **2. Architecture Design Mathematics:**
# - **Batch Normalization**: $\text{BN}(\mathbf{x}) = \gamma \frac{\mathbf{x} - \mu_{\mathcal{B}}}{\sigma_{\mathcal{B}}} + \beta$ stabilizes training
# - **Dropout**: $\mathbf{h}_{\text{drop}} = \mathbf{h} \odot \mathbf{m}$, $\mathbf{m} \sim \text{Bernoulli}(p)$ prevents overfitting
# - **Skip connections**: $\mathbf{h}^{(\ell+1)} = \mathbf{h}^{(\ell)} + \mathbf{F}(\mathbf{h}^{(\ell)})$ enables deep training
#
# **3. Optimization Mathematics:**
# - **Adam**: Combines momentum and adaptive learning rates for stable convergence
# - **Learning rate scheduling**: $\eta_t = \eta_0 \gamma^{t/T}$ prevents overshooting minima
# - **Early stopping**: Monitors $L_{\text{val}}$ to prevent overfitting
#
# **4. Evaluation Mathematics:**
# - **Multiple metrics**: Accuracy, precision, recall provide comprehensive assessment
# - **Confusion matrix**: $C_{ij} = |\{k: y_k = i, \hat{y}_k = j\}|$ reveals failure modes
# - **Cross-validation**: $\text{CV} = \frac{1}{K} \sum_{k=1}^K L_{\text{fold}_k}$ reduces evaluation variance
#
# **Theoretical Guarantees:**
# - **Universal approximation**: CNNs can approximate any continuous function
# - **Translation equivariance**: $T(f(\mathbf{x})) = f(T(\mathbf{x}))$ for translations $T$
# - **Parameter sharing**: Reduces effective parameter count from $O(H \cdot W)$ to $O(K^2)$

# %% [markdown]
# ## CNN Best Practices

# %%
print("CNN Best Practices:")
print("\n1. Data Preprocessing:")
print("   - Normalize pixel values (0-1 or standardize)")
print("   - Use data augmentation to increase dataset diversity")
print("   - Consider image size and computational constraints")

print("\n2. Architecture Design:")
print("   - Start with smaller networks and gradually increase complexity")
print("   - Use batch normalization for training stability")
print("   - Add dropout for regularization")
print("   - Consider skip connections (ResNet-style) for deeper networks")

print("\n3. Training:")
print("   - Use Adam or AdamW optimizers as starting point")
print("   - Implement learning rate scheduling")
print("   - Monitor both training and validation metrics")
print("   - Use early stopping to prevent overfitting")

print("\n4. Transfer Learning:")
print("   - Use pre-trained models when possible")
print("   - Fine-tune carefully (lower learning rates)")
print("   - Consider which layers to freeze/unfreeze")

print("\n5. Evaluation:")
print("   - Use multiple metrics (accuracy, precision, recall, F1)")
print("   - Visualize predictions and failure cases")
print("   - Analyze feature maps and learned filters")
print("   - Test on diverse, representative data")