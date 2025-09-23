# %% [markdown]
# # PyTorch Basics Part 2: Autograd and Neural Networks
#
# Automatic differentiation, gradients, and basic neural network building blocks
#
# ## Mathematical Foundation of Automatic Differentiation
#
# **Automatic Differentiation (Autograd)** is the computational implementation of the chain rule from calculus. For composite functions, it enables efficient computation of derivatives without symbolic differentiation.
#
# ### The Chain Rule
# For composite function $f(g(x))$:
# $$\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$
#
# ### Computational Graph
# Each operation creates nodes in a **directed acyclic graph (DAG)**:
# - **Forward pass**: Compute function values $f(x)$
# - **Backward pass**: Compute gradients $\frac{\partial f}{\partial x}$ using reverse-mode differentiation
#
# ### Gradient Computation
# For scalar output $y = f(x_1, x_2, \ldots, x_n)$:
# $$\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)$$
#




# PyTorch implements **reverse-mode AD**, which is efficient for functions $\mathbb{R}^n \rightarrow \mathbb{R}$ (many inputs, scalar output) - ideal for loss functions in machine learning.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Autograd: Automatic Differentiation
#
# ### Mathematical Example
# Consider the function: $z = x^2 + y^3$, $\text{loss} = \sum z$
#
# **Manual Computation:**
# - $\frac{\partial z}{\partial x} = 2x$
# - $\frac{\partial z}{\partial y} = 3y^2$
# - $\frac{\partial \text{loss}}{\partial x} = \frac{\partial \text{loss}}{\partial z} \frac{\partial z}{\partial x} = 1 \cdot 2x = 2x$
#
# **Reverse-Mode Algorithm:**
# 1. **Forward pass**: Compute function values and store intermediate results
# 2. **Backward pass**: Apply chain rule from output to inputs
#
# This demonstrates how PyTorch autograd implements mathematical differentiation computationally.

# %%


# %%

# Computational Graph & Tensor Properties

#   Computational Graph

#   A computational graph is a directed acyclic graph (DAG) where:
#   - Nodes = tensors and operations
#   - Edges = data flow between operations
#   - Built dynamically as you perform operations on tensors with requires_grad=True

#   When built: Every time you do operations on tensors with gradient tracking enabled.

#   Tensor Properties

#   | Property      | Description                            | When Used                                    |
#   |---------------|----------------------------------------|----------------------------------------------|
#   | requires_grad | Whether to track gradients             | Set manually or inherited from operations    |
#   | grad          | Stores computed gradients              | Populated after .backward()                  |
#   | grad_fn       | Points to function that created tensor | Automatic when operations create new tensors |
#   | is_leaf       | True if tensor has no grad_fn          | User-created tensors or after .detach()      |

#   Simple Example
import torch

# 1. Create leaf tensors (user-created, no grad_fn)
x = torch.tensor([2.0], requires_grad=True)  # Leaf tensor
w = torch.tensor([3.0], requires_grad=True)  # Leaf tensor
b = torch.tensor([1.0], requires_grad=True)  # Leaf tensor

print(f"x: grad_fn={x.grad_fn}, is_leaf={x.is_leaf}, requires_grad={x.requires_grad}")
# Output: x: grad_fn=None, is_leaf=True, requires_grad=True

# 2. Forward pass - builds computational graph
y = w * x + b  # y = 3*2 + 1 = 7
z = y ** 2     # z = 7^2 = 49

print(f"y: grad_fn={y.grad_fn}, is_leaf={y.is_leaf}")
# Output: y: grad_fn=<AddBackward0>, is_leaf=False

print(f"z: grad_fn={z.grad_fn}, is_leaf={z.is_leaf}")
# Output: z: grad_fn=<PowBackward0>, is_leaf=False

# 3. Backward pass - compute gradients
z.backward()  # Compute dz/dx, dz/dw, dz/db

print(f"x.grad = {x.grad}")  # dz/dx = 2*y*w = 2*7*3 = 42
print(f"w.grad = {w.grad}")  # dz/dw = 2*y*x = 2*7*2 = 28  
print(f"b.grad = {b.grad}")  # dz/db = 2*y*1 = 2*7*1 = 14

# 4. Detaching breaks the graph
x_detached = x.detach()
print(f"x_detached: requires_grad={x_detached.requires_grad}, is_leaf={x_detached.is_leaf}")
# Output: x_detached: requires_grad=False, is_leaf=True

# 5. no_grad() context prevents graph building
with torch.no_grad():
    y_no_grad = w * x + b
    print(f"y_no_grad: grad_fn={y_no_grad.grad_fn}")
    # Output: y_no_grad: grad_fn=None
    
#   Key Insights

#   - Leaf tensors: User-created (grad_fn=None, is_leaf=True)
#   - Intermediate tensors: Created by operations (grad_fn=<SomeBackward>, is_leaf=False)
#   - Graph building: Happens automatically during forward pass
#   - .backward(): Only works on scalar tensors (or with gradient argument)
#   - Memory: Graph is freed after .backward() unless retain_graph=True


# %%
# so when backward is run, we compute gradients starting from the function its being called from (z in this case), all the 
#   way to all the leaf nodes where requires grad is true? what if the functions were built with no grad? then will requires 
#   grad be true?

# ⏺ Backward Pass & Graph Traversal

#   How .backward() Works

#   Yes, exactly! When you call z.backward():

#   1. Starts from z (the tensor you call .backward() on)
#   2. Traverses backward through the computational graph via grad_fn links
#   3. Computes gradients for all leaf nodes where requires_grad=True
#   4. Stores gradients in the .grad attribute of those leaf tensors

#   Operations Built with no_grad()

#   If operations are performed within torch.no_grad():
#   - No computational graph is built
#   - No grad_fn is created
#   - requires_grad is False for the result (even if inputs had requires_grad=True)

#   Example

import torch

# Leaf tensors with gradient tracking
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([3.0], requires_grad=True)

# Case 1: Normal operations (graph built)
y1 = w * x  # y1.requires_grad=True, y1.grad_fn=<MulBackward0>
print(f"y1: requires_grad={y1.requires_grad}, grad_fn={y1.grad_fn}")

# Case 2: Operations with no_grad (no graph)
with torch.no_grad():
    y2 = w * x  # y2.requires_grad=False, y2.grad_fn=None
    print(f"y2: requires_grad={y2.requires_grad}, grad_fn={y2.grad_fn}")

# Case 3: Mixed operations
z1 = y1 ** 2  # Connected to graph
z2 = y2 ** 2  # NOT connected to graph

print(f"z1: requires_grad={z1.requires_grad}, grad_fn={z1.grad_fn}")
print(f"z2: requires_grad={z2.requires_grad}, grad_fn={z2.grad_fn}")

# Backward pass
z1.backward()  # Works - traverses back to x and w
# z2.backward()  # Would ERROR - no graph exists!

print(f"x.grad = {x.grad}")  # Has gradients from z1 path
print(f"w.grad = {w.grad}")  # Has gradients from z1 path

# Key Rules

# 1. requires_grad inheritance: Result has requires_grad=True only if:
# - At least one input has requires_grad=True AND
# - Operation is NOT in torch.no_grad() context
# 2. Graph connectivity: Only tensors connected via grad_fn links participate in backprop
# 3. Leaf node gradients: .backward() only populates .grad for leaf tensors with requires_grad=True

# Memory Efficiency

# During inference, use no_grad() to save memory
# with torch.no_grad():
#     predictions = model(inputs)  # No graph built, faster & less memory
    
# %%
# if I have the following computational graph: \
#   \
#   a and b are initial leaf tensors with requires_grad true, and we multiply them to get c = a*b\
#   \
#   then we have another leaf node d, and we multiply c and d to get e, that is: e = c*d\
#   \
#   now when i compute c.backward() i have the grad attribute of a and b store the gradients of c with respect to a and b. 
#   Then when i do e.backward, the grad attribute gets updated to store the values for e with respect to a and b. \
#   \
#   so the ones with respect to c get overwritten? what if i need them again? how is the situation handled? are the gradients
#    recomputed any time they are needed?

# ⏺ Gradient Accumulation & Overwriting in PyTorch

#   You've identified a key behavior in PyTorch: gradients accumulate by default, they don't get overwritten.

#   What Actually Happens

#   When you call .backward() multiple times, gradients accumulate (add up) in the .grad attribute:



# Create leaf tensors
a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)
d = torch.tensor([4.0], requires_grad=True)

# Build computational graph
c = a * b  # c = 6
e = c * d  # e = 24

# First backward: c.backward()
c.backward(retain_graph=True)  # Need retain_graph=True to keep graph
print(f"After c.backward():")
print(f"a.grad = {a.grad}")  # dc/da = b = 3
print(f"b.grad = {b.grad}")  # dc/db = a = 2

# Second backward: e.backward()
e.backward()
print(f"\nAfter e.backward():")
print(f"a.grad = {a.grad}")  # ACCUMULATED: dc/da + de/da = 3 + 12 = 15
print(f"b.grad = {b.grad}")  # ACCUMULATED: dc/db + de/db = 2 + 8 = 10
print(f"d.grad = {d.grad}")  # de/dd = c = 6

# Why Gradients Accumulate

# This design enables:
# - Mini-batch gradient accumulation: Accumulate gradients across multiple small batches
# - RNN training: Gradients accumulate across time steps
# - Multi-loss optimization: Combine gradients from multiple loss functions

# How to Handle Different Scenarios

# 1. Separate Gradients (Clear between backward calls)

# Get gradients for c
c.backward(retain_graph=True)
dc_da, dc_db = a.grad.clone(), b.grad.clone()  # Save gradients

# Clear gradients
a.grad.zero_()
b.grad.zero_()

# Get gradients for e
e.backward()
de_da, de_db = a.grad.clone(), b.grad.clone()

print(f"dc/da = {dc_da}, de/da = {de_da}")

# 2. Use torch.autograd.grad() (No accumulation)

# Compute gradients without storing in .grad
dc_da, dc_db = torch.autograd.grad(c, [a, b], retain_graph=True)
de_da, de_db = torch.autograd.grad(e, [a, b])

print(f"dc/da = {dc_da}, de/da = {de_da}")

# 3. Standard Training Loop (Clear each iteration)

# # Typical training pattern
# for epoch in range(num_epochs):
#     optimizer.zero_grad()  # Clear gradients
#     loss = model(x)
#     loss.backward()        # Compute gradients
#     optimizer.step()       # Update parameters

#   Key Points

#   - Default behavior: Gradients accumulate, not overwrite
#   - Why: Enables gradient accumulation patterns in ML training
#   - Solutions: Use .zero_grad() or torch.autograd.grad() for separate computations
#   - Graph lifecycle: Graph is destroyed after .backward() unless retain_graph=True

#   The accumulation behavior is intentional and powerful for ML workflows, but requires explicit management when you need
#   separate gradient computations.


# %%
# Tensors with gradient tracking
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = torch.tensor([1.0, 4.0], requires_grad=True)

print(f"x: {x}")
print(f"y: {y}")
print(f"x.requires_grad: {x.requires_grad}")
print(f"y.requires_grad: {y.requires_grad}")

# %%
# Compute a function
z = x**2 + y**3
loss = z.sum()

print(f"z: {z}")
print(f"loss: {loss}")
print(f"loss.requires_grad: {loss.requires_grad}")

# %%
# so grad is both an attribute of a tensor, and a method. also, retain graph was set to true so that the c to a, b graph 
#   could be used later in the e.backward calculation?

# ⏺ .grad Attribute vs torch.autograd.grad() Function

#   You're mixing up two different things:

#   1. .grad - Tensor Attribute

x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # Tensor attribute storing gradients

#   2. torch.autograd.grad() - Function

x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
grad_x = torch.autograd.grad(y, x)  # Function call, returns tuple
print(grad_x[0])  # Same value as x.grad would be

#   Key differences:
#   - .grad is an attribute that accumulates gradients
#   - torch.autograd.grad() is a function that computes and returns gradients without storing them

# retain_graph=True Explanation

#   You're partially correct but not quite. Let me clarify:

a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)
d = torch.tensor([4.0], requires_grad=True)

c = a * b  # Graph: a,b → c
e = c * d  # Graph: a,b → c → e (extended graph)

#   Without retain_graph=True:

c.backward()  # This DESTROYS the graph after computing gradients
# e.backward()  # ERROR! Graph no longer exists

#   With retain_graph=True:

c.backward(retain_graph=True)  # Keeps graph alive
e.backward()  # Works because graph still exists

#   Important Clarification

#   When you do e.backward(), it doesn't need the c→a,b portion specifically. The graph looks like:

#   a ──┐
#       ├→ c ──┐
#   b ──┘      ├→ e
#   d ─────────┘

#   e.backward() traverses the entire path: e → c → a,b and e → d.

#   The retain_graph=True on c.backward() is needed because:
#   1. c.backward() would normally destroy the entire computational graph
#   2. This would make e.backward() impossible since e's graph includes the path through c

#   So yes, you need retain_graph=True to keep the graph alive for the subsequent e.backward() call.


# %%
# Compute gradients
loss.backward()

print(f"x.grad: {x.grad}")
print(f"y.grad: {y.grad}")

# Manual verification:
# dz/dx = 2x, so at x=[2,3]: [4, 6]
# dz/dy = 3y^2, so at y=[1,4]: [3, 48]
print(f"Expected x.grad: {2 * x}")
print(f"Expected y.grad: {3 * y**2}")



# %% [markdown]
# ## Gradient Accumulation and Zeroing
#
# ### Mathematical Principle
# **Gradient Accumulation** follows the linearity of differentiation:
#
# For functions $f_1, f_2, \ldots, f_n$ and scalar $c$:
# $$\frac{\partial}{\partial x}[f_1(x) + f_2(x)] = \frac{\partial f_1}{\partial x} + \frac{\partial f_2}{\partial x}$$
# $$\frac{\partial}{\partial x}[c \cdot f(x)] = c \cdot \frac{\partial f}{\partial x}$$
#
# **Why Gradients Accumulate:**
# - Each `.backward()` call adds to existing gradients: $\text{grad} \leftarrow \text{grad} + \nabla_{\text{new}} f$
# - This enables **gradient accumulation** across multiple loss terms or batches
# - **Must manually zero** gradients between independent computations
#
# **Mathematical Interpretation:**
# If we compute losses $L_1, L_2$ separately:
# $$\nabla (L_1 + L_2) = \nabla L_1 + \nabla L_2$$
#
# PyTorch implements this by accumulating gradients, allowing flexible gradient computation strategies.

# %%
# Gradients accumulate by default
x = torch.tensor([1.0, 2.0], requires_grad=True)

# First computation
y1 = x**2
y1.sum().backward()
print(f"After first backward: x.grad = {x.grad}")

# Second computation (gradients accumulate)
y2 = x**3
y2.sum().backward()
print(f"After second backward: x.grad = {x.grad}")

# Zero gradients
x.grad.zero_()
print(f"After zeroing: x.grad = {x.grad}")

# %% [markdown]
# ## Neural Network Modules
#
# ### Mathematical Foundation of Linear Layers
#
# **Affine Transformation:**
# A linear layer implements the affine transformation:
# $$\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$$
#
# Where:
# - $\mathbf{x} \in \mathbb{R}^{n}$ is the input vector
# - $\mathbf{W} \in \mathbb{R}^{m \times n}$ is the weight matrix
# - $\mathbf{b} \in \mathbb{R}^{m}$ is the bias vector
# - $\mathbf{y} \in \mathbb{R}^{m}$ is the output vector
#
# **Batch Processing:**
# For batch input $\mathbf{X} \in \mathbb{R}^{B \times n}$ (B samples):
# $$\mathbf{Y} = \mathbf{X}\mathbf{W}^T + \mathbf{b}$$
#
# **Parameter Initialization:**
# - **Xavier/Glorot**: $\mathcal{W} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$
# - **He initialization**: $\mathcal{W} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$ (for ReLU)
#
# The linear layer forms the fundamental building block for deep neural networks, implementing learnable linear transformations.

# %%
# Linear layer (fully connected)
linear = nn.Linear(in_features=3, out_features=2)

print(f"Weight shape: {linear.weight.shape}")
print(f"Bias shape: {linear.bias.shape}")
print(f"Weight:\n{linear.weight}")
print(f"Bias: {linear.bias}")

# %%
# Forward pass through linear layer
x = torch.randn(5, 3)  # batch_size=5, input_features=3
output = linear(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Output:\n{output}")

# %% [markdown]
# ## Activation Functions
#
# ### Mathematical Foundation of Non-linearity
#
# **Activation functions** introduce non-linearity into neural networks, enabling them to approximate complex functions. Without activation functions, multiple linear layers would collapse to a single linear transformation.
#
# **Common Activation Functions:**
#
# **ReLU (Rectified Linear Unit):**
# $$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$
# - **Derivative**: $\frac{d}{dx}\text{ReLU}(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$
#
# **Sigmoid:**
# $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
# - **Properties**: $\sigma(x) \in (0, 1)$, $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
# - **Issues**: Vanishing gradients for large $|x|$
#
# **Tanh (Hyperbolic Tangent):**
# $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{e^{2x} - 1}{e^{2x} + 1}$$
# - **Properties**: $\tanh(x) \in (-1, 1)$, $\tanh'(x) = 1 - \tanh^2(x)$
# - **Advantage**: Zero-centered output
#
# **Universal Approximation Theorem**: Neural networks with at least one hidden layer and non-linear activation can approximate any continuous function on compact sets.

# %%
# Common activation functions
x = torch.linspace(-3, 3, 100)

# ReLU
relu_output = F.relu(x)

# Sigmoid
sigmoid_output = torch.sigmoid(x)

# Tanh
tanh_output = torch.tanh(x)

# Plot activations
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x.numpy(), relu_output.numpy())
plt.title('ReLU')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x.numpy(), sigmoid_output.numpy())
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x.numpy(), tanh_output.numpy())
plt.title('Tanh')
plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Building a Simple Neural Network
#
# ### Mathematical Architecture
#
# **Multi-Layer Perceptron (MLP):**
# For a 2-layer network with hidden layer:
# $$\mathbf{h} = \sigma_1(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1)$$
# $$\mathbf{y} = \sigma_2(\mathbf{W}_2\mathbf{h} + \mathbf{b}_2)$$
#
# Where:
# - $\mathbf{x} \in \mathbb{R}^{d}$ is input
# - $\mathbf{W}_1 \in \mathbb{R}^{h \times d}$, $\mathbf{b}_1 \in \mathbb{R}^{h}$ (input to hidden)
# - $\mathbf{W}_2 \in \mathbb{R}^{k \times h}$, $\mathbf{b}_2 \in \mathbb{R}^{k}$ (hidden to output)
# - $\sigma_1, \sigma_2$ are activation functions
#
# **Parameter Count:**
# Total parameters = $(d \times h + h) + (h \times k + k) = h(d + k) + (h + k)$
#
# **Forward Propagation:**
# The network computes the composition:
# $$f(\mathbf{x}; \boldsymbol{\theta}) = \sigma_2(\mathbf{W}_2 \sigma_1(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2)$$
#
# Where $\boldsymbol{\theta} = \{\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2\}$ are learnable parameters.

# %%
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Create network
net = SimpleNet(input_size=4, hidden_size=10, output_size=3)
print(net)

# Count parameters
total_params = sum(p.numel() for p in net.parameters())
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# %%
# Forward pass through network
x = torch.randn(8, 4)  # batch_size=8, input_features=4
output = net(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Output:\n{output}")

# %% [markdown]
# ## Loss Functions
#
# ### Mathematical Foundation of Loss Functions
#
# Loss functions quantify the discrepancy between predictions and targets, providing the objective to minimize during training.
#
# **Mean Squared Error (MSE) - Regression:**
# $$L_{\text{MSE}}(\mathbf{y}, \hat{\mathbf{y}}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
#
# **Properties:**
# - Differentiable everywhere
# - Penalizes large errors quadratically
# - Assumes Gaussian noise in targets
#
# **Cross-Entropy Loss - Classification:**
# For multiclass classification with true class $c$ and predicted probabilities $\mathbf{p}$:
# $$L_{\text{CE}} = -\log p_c = -\log\left(\frac{e^{z_c}}{\sum_{j=1}^{K} e^{z_j}}\right)$$
#
# Where $z_j$ are logits (pre-softmax outputs).
#
# **Softmax Function:**
# $$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$
#
# **Gradient of Cross-Entropy + Softmax:**
# $$\frac{\partial L_{\text{CE}}}{\partial z_i} = p_i - \delta_{ic}$$
# where $\delta_{ic}$ is 1 if $i = c$ (true class), 0 otherwise.
#
# This combination has a clean gradient that drives predictions toward the correct distribution.

# %%
# Mean Squared Error (MSE) for regression
predictions = torch.randn(10, 1)
targets = torch.randn(10, 1)

mse_loss = nn.MSELoss()
loss = mse_loss(predictions, targets)
print(f"MSE Loss: {loss.item():.4f}")

# Manual calculation
manual_mse = ((predictions - targets)**2).mean()
print(f"Manual MSE: {manual_mse.item():.4f}")

# %%
# Cross-Entropy Loss for classification
# Raw logits (before softmax)
logits = torch.randn(5, 3)  # 5 samples, 3 classes
targets = torch.tensor([0, 1, 2, 1, 0])  # class indices

ce_loss = nn.CrossEntropyLoss()
loss = ce_loss(logits, targets)
print(f"Cross-Entropy Loss: {loss.item():.4f}")

# Convert to probabilities
probabilities = F.softmax(logits, dim=1)
print(f"Probabilities:\n{probabilities}")
print(f"Predicted classes: {probabilities.argmax(dim=1)}")

# %% [markdown]
# ## Optimizers
#
# ### Mathematical Foundation of Optimization
#
# **Gradient Descent** is the fundamental optimization algorithm for minimizing loss functions.
#
# **Vanilla Gradient Descent:**
# $$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}_t)$$
#
# Where:
# - $\boldsymbol{\theta}$ are parameters
# - $\eta$ is the learning rate
# - $\nabla_{\boldsymbol{\theta}} L$ is the gradient of loss w.r.t. parameters
#
# **Stochastic Gradient Descent (SGD):**
# Instead of using the full dataset, SGD uses mini-batches:
# $$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla_{\boldsymbol{\theta}} L_{\text{batch}}(\boldsymbol{\theta}_t)$$
#
# **Advanced Optimizers:**
#
# **Adam (Adaptive Moment Estimation):**
# $$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
# $$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
# $$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
# $$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
#
# Adam combines momentum (first moment) with adaptive learning rates (second moment).

# %%
# Create a simple optimization example
net = SimpleNet(2, 5, 1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# Generate some dummy data
x = torch.randn(20, 2)
y = torch.randn(20, 1)

print("Initial loss:")
initial_output = net(x)
initial_loss = F.mse_loss(initial_output, y)
print(f"Loss: {initial_loss.item():.4f}")

# Training step
optimizer.zero_grad()  # Clear gradients
output = net(x)        # Forward pass
loss = F.mse_loss(output, y)  # Compute loss
loss.backward()        # Backward pass
optimizer.step()       # Update parameters

print("\nAfter one optimization step:")
new_output = net(x)
new_loss = F.mse_loss(new_output, y)
print(f"Loss: {new_loss.item():.4f}")
print(f"Loss change: {new_loss.item() - initial_loss.item():.4f}")

# %% [markdown]
# ## Simple Training Loop
#
# ### Mathematical Training Process
#
# **Training Loop** implements the empirical risk minimization:
# $$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \frac{1}{n} \sum_{i=1}^{n} L(f(\mathbf{x}_i; \boldsymbol{\theta}), y_i)$$
#
# **Algorithm:**
# 1. **Forward Pass**: Compute predictions $\hat{\mathbf{y}} = f(\mathbf{X}; \boldsymbol{\theta})$
# 2. **Loss Computation**: $L = \frac{1}{B} \sum_{i=1}^{B} \ell(\hat{y}_i, y_i)$
# 3. **Backward Pass**: Compute $\nabla_{\boldsymbol{\theta}} L$ via backpropagation
# 4. **Parameter Update**: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} L$
# 5. **Repeat** until convergence
#
# **Example: Linear Regression**
# True model: $y = \mathbf{w}^T\mathbf{x} + b + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma^2)$
#
# **Maximum Likelihood Estimation** under Gaussian noise leads to MSE loss:
# $$L(\mathbf{w}, b) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \mathbf{w}^T\mathbf{x}_i - b)^2$$
#
# The training process finds parameters that minimize this empirical risk.

# %%
# Simple regression problem: y = 2x1 + 3x2 + noise
torch.manual_seed(42)

# Generate data
n_samples = 100
x = torch.randn(n_samples, 2)
true_weights = torch.tensor([[2.0], [3.0]])
y = x @ true_weights + 0.1 * torch.randn(n_samples, 1)

# Create model
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

# Training loop
losses = []
for epoch in range(100):
    # Forward pass
    predictions = model(x)
    loss = criterion(predictions, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print(f"\nTrue weights: {true_weights.flatten()}")
print(f"Learned weights: {model.weight.data.flatten()}")
print(f"True bias: 0.0")
print(f"Learned bias: {model.bias.data.item():.4f}")

# %%
# Plot training loss
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)
plt.show()

# %% [markdown]
# ## Working with Different Optimizers

# %%
# Compare different optimizers
def train_with_optimizer(optimizer_class, **kwargs):
    model = nn.Linear(2, 1)
    optimizer = optimizer_class(model.parameters(), **kwargs)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(50):
        predictions = model(x)
        loss = criterion(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses, model.weight.data.flatten()

# Test different optimizers
sgd_losses, sgd_weights = train_with_optimizer(torch.optim.SGD, lr=0.1)
adam_losses, adam_weights = train_with_optimizer(torch.optim.Adam, lr=0.1)
rmsprop_losses, rmsprop_weights = train_with_optimizer(torch.optim.RMSprop, lr=0.1)

plt.figure(figsize=(10, 4))
plt.plot(sgd_losses, label='SGD')
plt.plot(adam_losses, label='Adam')
plt.plot(rmsprop_losses, label='RMSprop')
plt.title('Optimizer Comparison')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()

print(f"True weights: {true_weights.flatten()}")
print(f"SGD weights: {sgd_weights}")
print(f"Adam weights: {adam_weights}")
print(f"RMSprop weights: {rmsprop_weights}")

# %% [markdown]
# ## Saving and Loading Models

# %%
# Save model state
model = SimpleNet(3, 5, 2)
torch.save(model.state_dict(), 'model_weights.pth')

# Save entire model
torch.save(model, 'complete_model.pth')

print("Model saved successfully")

# Load model state (recommended approach)
new_model = SimpleNet(3, 5, 2)
new_model.load_state_dict(torch.load('model_weights.pth'))
new_model.eval()  # Set to evaluation mode

print("Model loaded successfully")

# Clean up files
import os
os.remove('model_weights.pth')
os.remove('complete_model.pth')

# %%