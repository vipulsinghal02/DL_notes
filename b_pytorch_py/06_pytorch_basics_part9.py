# %% [markdown]
# # PyTorch Basics Part 9: Advanced Architectures and Specialized Domains
#
# Exploring Graph Neural Networks, Vision Transformers, advanced computer vision, time series modeling, and multi-modal learning with mathematical foundations
#
# ## Mathematical Framework for Advanced Architectures
#
# Advanced neural architectures extend beyond standard feedforward networks to handle structured data, complex relationships, and multi-modal inputs:
#
# ### Core Mathematical Concepts
#
# **1. Graph Neural Networks:**
# - **Message Passing**: $\mathbf{h}_v^{(l+1)} = \text{UPDATE}^{(l)}\left(\mathbf{h}_v^{(l)}, \text{AGGREGATE}^{(l)}\left(\{\mathbf{h}_u^{(l)} : u \in \mathcal{N}(v)\}\right)\right)$
# - **Graph Convolution**: $\mathbf{H}^{(l+1)} = \sigma\left(\tilde{\mathbf{A}}\mathbf{H}^{(l)}\mathbf{W}^{(l)}\right)$
#   where $\tilde{\mathbf{A}} = \mathbf{D}^{-1/2}(\mathbf{A} + \mathbf{I})\mathbf{D}^{-1/2}$ is normalized adjacency
#
# **2. Vision Transformers:**
# - **Patch Embedding**: $\mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{x}_{patch}^1\mathbf{E}; \ldots; \mathbf{x}_{patch}^N\mathbf{E}] + \mathbf{E}_{pos}$
# - **Multi-Head Self-Attention**: $\text{MSA}(\mathbf{z}) = [\text{head}_1; \ldots; \text{head}_h]\mathbf{W}^O$
#   where $\text{head}_i = \text{Attention}(\mathbf{z}\mathbf{W}_i^Q, \mathbf{z}\mathbf{W}_i^K, \mathbf{z}\mathbf{W}_i^V)$
#
# **3. Object Detection:**
# - **Bounding Box Regression**: $t_x = (x - x_a)/w_a$, $t_y = (y - y_a)/h_a$
#   $t_w = \log(w/w_a)$, $t_h = \log(h/h_a)$
# - **IoU Loss**: $\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$
#
# **4. Time Series Modeling:**
# - **LSTM Gates**:
#   - $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ (forget gate)
#   - $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ (input gate)
#   - $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ (output gate)
# - **Attention in Time**: $\alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^T \exp(e_k)}$ where $e_t = \text{score}(h_t, s)$
#
# **5. Multi-Modal Fusion:**
# - **Early Fusion**: $\mathbf{h} = f([\mathbf{x}_1; \mathbf{x}_2; \ldots; \mathbf{x}_m])$
# - **Late Fusion**: $\mathbf{h} = g(f_1(\mathbf{x}_1), f_2(\mathbf{x}_2), \ldots, f_m(\mathbf{x}_m))$
# - **Cross-Modal Attention**: $\mathbf{a}_{ij} = \frac{\exp(\mathbf{h}_i^T\mathbf{W}\mathbf{h}_j)}{\sum_k \exp(\mathbf{h}_i^T\mathbf{W}\mathbf{h}_k)}$
#
# **6. Inductive Biases:**
# - **Translation Equivariance** (CNNs): $f(T_g(x)) = T_g(f(x))$
# - **Permutation Invariance** (GNNs): $f(\pi(\mathbf{X})) = \pi(f(\mathbf{X}))$
# - **Sequence Modeling** (RNNs): Temporal dependencies through recurrent connections

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

# Try to import PyTorch Geometric (optional)
try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    import networkx as nx
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False
    print("PyTorch Geometric not available. Some GNN examples will be simplified.")

# %% [markdown]
# ## Graph Neural Networks (GNNs)
#
# **Mathematical Foundation:**
#
# GNNs operate on graph-structured data $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where $\mathcal{V}$ is the set of nodes and $\mathcal{E}$ is the set of edges. The core idea is **message passing**:
#
# $$\mathbf{m}_{u \rightarrow v}^{(l)} = \text{MESSAGE}^{(l)}(\mathbf{h}_u^{(l)}, \mathbf{h}_v^{(l)}, \mathbf{e}_{u,v})$$
#
# $$\mathbf{h}_v^{(l+1)} = \text{UPDATE}^{(l)}\left(\mathbf{h}_v^{(l)}, \text{AGGREGATE}^{(l)}\left(\{\mathbf{m}_{u \rightarrow v}^{(l)} : u \in \mathcal{N}(v)\}\right)\right)$$
#
# **Graph Convolutional Networks (GCN):**
# $$\mathbf{H}^{(l+1)} = \sigma\left(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\mathbf{H}^{(l)}\mathbf{W}^{(l)}\right)$$
#
# where:
# - $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ (add self-loops)
# - $\tilde{\mathbf{D}}_{ii} = \sum_j \tilde{\mathbf{A}}_{ij}$ (degree matrix)
# - $\mathbf{H}^{(l)} \in \mathbb{R}^{N \times d_l}$ (node features)
# - $\mathbf{W}^{(l)} \in \mathbb{R}^{d_l \times d_{l+1}}$ (learnable weights)
#
# GNNs are used for social networks, molecular analysis, knowledge graphs, and recommendation systems.

# %%
if PYTORCH_GEOMETRIC_AVAILABLE:
    # Simple Graph Neural Network implementation
    class SimpleGNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
            super(SimpleGNN, self).__init__()

            # Graph convolutional layers
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(input_dim, hidden_dim))

            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))

            self.convs.append(GCNConv(hidden_dim, output_dim))

            self.dropout = nn.Dropout(0.2)

        def forward(self, x, edge_index, batch=None):
            # Apply graph convolutions
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = F.relu(x)
                x = self.dropout(x)

            # Final layer
            x = self.convs[-1](x, edge_index)

            # Global pooling for graph-level predictions
            if batch is not None:
                x = global_mean_pool(x, batch)

            return x

    # Create synthetic graph data
    def create_synthetic_graphs(num_graphs=100, num_nodes_range=(10, 30), num_features=5):
        """Create synthetic graph dataset"""
        graphs = []
        labels = []

        for i in range(num_graphs):
            # Random number of nodes
            num_nodes = np.random.randint(num_nodes_range[0], num_nodes_range[1])

            # Create random graph
            G = nx.erdos_renyi_graph(num_nodes, 0.3)

            # Node features
            node_features = torch.randn(num_nodes, num_features)

            # Edge indices
            edge_index = torch.LongTensor(list(G.edges)).t().contiguous()
            if edge_index.numel() == 0:  # Handle graphs with no edges
                edge_index = torch.LongTensor(2, 0)

            # Create graph label based on some property (e.g., number of nodes)
            label = 1 if num_nodes > 20 else 0

            # Create PyTorch Geometric data object
            graph_data = Data(x=node_features, edge_index=edge_index, y=torch.LongTensor([label]))

            graphs.append(graph_data)
            labels.append(label)

        return graphs, labels

    # Create synthetic graph dataset
    graphs, graph_labels = create_synthetic_graphs(50, (5, 25), 3)

    print(f"Created {len(graphs)} graphs")
    print(f"Sample graph:")
    print(f"  Nodes: {graphs[0].x.shape[0]}")
    print(f"  Features per node: {graphs[0].x.shape[1]}")
    print(f"  Edges: {graphs[0].edge_index.shape[1]}")
    print(f"  Label: {graphs[0].y.item()}")

    # Create GNN model
    gnn_model = SimpleGNN(input_dim=3, hidden_dim=16, output_dim=2)
    print(f"\nGNN parameters: {sum(p.numel() for p in gnn_model.parameters()):,}")

    # Test forward pass
    batch = Batch.from_data_list(graphs[:4])
    output = gnn_model(batch.x, batch.edge_index, batch.batch)
    print(f"Batch output shape: {output.shape}")

else:
    print("PyTorch Geometric not available. Install with: pip install torch-geometric")
    print("Showing conceptual GNN implementation instead:")

    # Conceptual GNN without PyTorch Geometric
    class ConceptualGNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(ConceptualGNN, self).__init__()
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, node_features, adjacency_matrix):
            # Simple message passing: aggregate neighbor features
            # H^(l+1) = σ(A * H^(l) * W^(l))
            x = torch.mm(adjacency_matrix, node_features)
            x = F.relu(self.linear1(x))
            x = self.linear2(x)
            return x

    # Example usage
    conceptual_gnn = ConceptualGNN(5, 16, 2)
    node_features = torch.randn(10, 5)  # 10 nodes, 5 features each
    adjacency = torch.randn(10, 10)   # Adjacency matrix
    output = conceptual_gnn(node_features, adjacency)

    print(f"Conceptual GNN output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in conceptual_gnn.parameters()):,}")

# %% [markdown]
# ## Vision Transformers (ViTs)
#
# **Mathematical Foundation:**
#
# Vision Transformers adapt the transformer architecture to images by treating image patches as sequence tokens:
#
# **1. Patch Embedding:**
# $$\mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{x}_{patch}^1\mathbf{E}; \mathbf{x}_{patch}^2\mathbf{E}; \ldots; \mathbf{x}_{patch}^N\mathbf{E}] + \mathbf{E}_{pos}$$
#
# where:
# - $\mathbf{x}_{patch}^i \in \mathbb{R}^{P^2 \cdot C}$ (flattened patch)
# - $\mathbf{E} \in \mathbb{R}^{P^2 \cdot C \times D}$ (embedding matrix)
# - $\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$ (position embeddings)
#
# **2. Multi-Head Self-Attention:**
# $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$
#
# $$\text{MSA}(\mathbf{z}) = [\text{head}_1; \text{head}_2; \ldots; \text{head}_h]\mathbf{W}^O$$
#
# **3. Transformer Block:**
# $$\mathbf{z}'_l = \text{MSA}(\text{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}$$
# $$\mathbf{z}_l = \text{MLP}(\text{LN}(\mathbf{z}'_l)) + \mathbf{z}'_l$$
#
# Vision Transformers excel with large datasets and global context understanding, competing with CNNs while being more scalable.

# %%
# Vision Transformer implementation
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Convolutional layer to create patch embeddings
        self.projection = nn.Conv2d(in_channels, embed_dim,
                                   kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)        # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)   # (batch_size, num_patches, embed_dim)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, num_patches, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention computation
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).reshape(batch_size, num_patches, embed_dim)

        # Final projection
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, dropout=0.1):
        super(VisionTransformer, self).__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Use class token for classification

        return self.head(cls_token_final)

# Create a smaller ViT for demonstration
vit_model = VisionTransformer(
    img_size=32,      # Smaller image size
    patch_size=4,     # Smaller patches
    in_channels=3,
    num_classes=10,
    embed_dim=192,    # Smaller embedding dimension
    depth=6,          # Fewer layers
    num_heads=3,      # Fewer attention heads
    mlp_ratio=4
)

# Test the model
test_input = torch.randn(2, 3, 32, 32)  # Batch of 2 images
output = vit_model(test_input)

print(f"Vision Transformer:")
print(f"  Input shape: {test_input.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Parameters: {sum(p.numel() for p in vit_model.parameters()):,}")
print(f"  Number of patches: {vit_model.patch_embed.num_patches}")

# Visualize attention (conceptual)
print(f"\nViT Architecture Details:")
print(f"  Patch size: {vit_model.patch_embed.patch_size}x{vit_model.patch_embed.patch_size}")
print(f"  Embedding dim: {192}")
print(f"  Transformer layers: {6}")
print(f"  Attention heads per layer: {3}")

# %% [markdown]
# ## Object Detection with YOLO-Style Architecture
#
# Object detection involves both classification and localization of objects in images. YOLO (You Only Look Once) approaches this as a single regression problem, predicting bounding boxes and class probabilities directly.

# %%
# Simplified YOLO-style object detection
class SimpleYOLO(nn.Module):
    def __init__(self, num_classes=20, grid_size=7, num_boxes=2):
        super(SimpleYOLO, self).__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_boxes = num_boxes

        # Backbone network (simplified)
        self.backbone = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((grid_size, grid_size))
        )

        # Detection head
        # Each grid cell predicts: (x, y, w, h, confidence) * num_boxes + class_probs
        output_size = num_boxes * 5 + num_classes
        self.detection_head = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1), nn.ReLU(),
            nn.Conv2d(1024, output_size, 1)
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)

        # Predict detections
        detections = self.detection_head(features)

        # Reshape to (batch_size, grid_size, grid_size, output_size)
        batch_size = x.size(0)
        detections = detections.permute(0, 2, 3, 1)

        return detections

    def decode_predictions(self, predictions, conf_threshold=0.5):
        """Decode predictions to bounding boxes"""
        batch_size, grid_size, grid_size, _ = predictions.shape
        detections = []

        for b in range(batch_size):
            batch_detections = []

            for i in range(grid_size):
                for j in range(grid_size):
                    cell_pred = predictions[b, i, j]

                    # Extract box predictions and class probabilities
                    for box in range(self.num_boxes):
                        start_idx = box * 5

                        # Box coordinates (x, y, w, h)
                        x = (j + torch.sigmoid(cell_pred[start_idx])) / grid_size
                        y = (i + torch.sigmoid(cell_pred[start_idx + 1])) / grid_size
                        w = torch.sigmoid(cell_pred[start_idx + 2])
                        h = torch.sigmoid(cell_pred[start_idx + 3])

                        # Confidence
                        confidence = torch.sigmoid(cell_pred[start_idx + 4])

                        # Class probabilities
                        class_probs = F.softmax(cell_pred[self.num_boxes * 5:], dim=0)
                        class_confidence = confidence * class_probs.max()

                        if class_confidence > conf_threshold:
                            batch_detections.append({
                                'x': x.item(), 'y': y.item(),
                                'w': w.item(), 'h': h.item(),
                                'confidence': confidence.item(),
                                'class': class_probs.argmax().item(),
                                'class_prob': class_probs.max().item()
                            })

            detections.append(batch_detections)

        return detections

# Create YOLO model
yolo_model = SimpleYOLO(num_classes=10, grid_size=7, num_boxes=2)

# Test the model
test_images = torch.randn(2, 3, 224, 224)
predictions = yolo_model(test_images)

print(f"YOLO Object Detection:")
print(f"  Input shape: {test_images.shape}")
print(f"  Predictions shape: {predictions.shape}")
print(f"  Parameters: {sum(p.numel() for p in yolo_model.parameters()):,}")

# Decode predictions
with torch.no_grad():
    detections = yolo_model.decode_predictions(predictions, conf_threshold=0.1)
    print(f"  Detections in first image: {len(detections[0])}")
    if detections[0]:
        det = detections[0][0]
        print(f"  Sample detection: class={det['class']}, conf={det['confidence']:.3f}, bbox=({det['x']:.3f},{det['y']:.3f},{det['w']:.3f},{det['h']:.3f})")

# YOLO Loss function (simplified)
class YOLOLoss(nn.Module):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20):
        super(YOLOLoss, self).__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        # Loss weights
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, predictions, targets):
        """Simplified YOLO loss computation"""
        # This is a conceptual implementation
        # Real YOLO loss is more complex with IoU calculations

        batch_size = predictions.size(0)

        # Separate box coordinates, confidence, and class predictions
        coord_loss = F.mse_loss(predictions[:, :, :, :4], targets[:, :, :, :4])
        conf_loss = F.mse_loss(predictions[:, :, :, 4], targets[:, :, :, 4])
        class_loss = F.cross_entropy(predictions[:, :, :, 5:].view(-1, self.num_classes),
                                   targets[:, :, :, 5].view(-1).long())

        total_loss = self.lambda_coord * coord_loss + conf_loss + class_loss
        return total_loss

print(f"\nYOLO Architecture Summary:")
print(f"  Grid size: {7}x{7} = {49} cells")
print(f"  Boxes per cell: {2}")
print(f"  Total possible detections: {7*7*2} = {98}")
print(f"  Output per cell: {2*5 + 10} = {20} values (2 boxes * 5 params + 10 classes)")

# %% [markdown]
# ## Time Series Forecasting with Deep Learning
#
# Time series forecasting predicts future values based on historical data. Deep learning approaches can capture complex temporal patterns and non-linear relationships in sequential data.

# %%
# Time Series Forecasting Models
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMForecaster, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Use last time step for prediction
        last_output = lstm_out[:, -1, :]

        # Apply dropout and linear layer
        output = self.dropout(last_output)
        predictions = self.linear(output)

        return predictions

class TransformerForecaster(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.1):
        super(TransformerForecaster, self).__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead,
                                                  dim_feedforward=d_model*4,
                                                  dropout=dropout,
                                                  batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_projection = nn.Linear(d_model, output_size)

    def forward(self, x):
        seq_len = x.size(1)

        # Project input and add positional encoding
        x = self.input_projection(x)
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)

        # Transformer encoding
        x = self.transformer(x)

        # Use last time step for prediction
        predictions = self.output_projection(x[:, -1, :])

        return predictions

# Generate synthetic time series data
def generate_time_series(n_samples=1000, seq_length=50, n_features=1):
    """Generate synthetic time series with trend and seasonality"""
    time = np.arange(n_samples + seq_length)

    # Multiple components
    trend = 0.02 * time
    seasonal = 2 * np.sin(2 * np.pi * time / 50) + np.sin(2 * np.pi * time / 10)
    noise = 0.5 * np.random.randn(len(time))

    series = trend + seasonal + noise

    # Create sequences
    X, y = [], []
    for i in range(n_samples):
        X.append(series[i:i+seq_length])
        y.append(series[i+seq_length])

    X = np.array(X).reshape(n_samples, seq_length, n_features)
    y = np.array(y).reshape(n_samples, n_features)

    return torch.FloatTensor(X), torch.FloatTensor(y), series

# Generate time series data
X_ts, y_ts, full_series = generate_time_series(1000, 50, 1)

print(f"Time Series Data:")
print(f"  Input sequences shape: {X_ts.shape}")
print(f"  Targets shape: {y_ts.shape}")

# Visualize the time series
plt.figure(figsize=(15, 10))

# Plot full series
plt.subplot(3, 2, 1)
plt.plot(full_series[:200])
plt.title('Synthetic Time Series (First 200 points)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)

# Show sample sequence
plt.subplot(3, 2, 2)
sample_idx = 100
plt.plot(range(50), X_ts[sample_idx].squeeze(), 'b-', label='Input Sequence')
plt.plot(50, y_ts[sample_idx].item(), 'ro', label='Target', markersize=8)
plt.title('Sample Input Sequence and Target')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Create and test models
lstm_forecaster = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2, output_size=1)
transformer_forecaster = TransformerForecaster(input_size=1, d_model=64, nhead=4,
                                             num_layers=3, output_size=1)

# Test forward passes
test_input = X_ts[:5]
lstm_pred = lstm_forecaster(test_input)
transformer_pred = transformer_forecaster(test_input)

print(f"\nModel Comparison:")
print(f"LSTM Forecaster:")
print(f"  Parameters: {sum(p.numel() for p in lstm_forecaster.parameters()):,}")
print(f"  Prediction shape: {lstm_pred.shape}")

print(f"Transformer Forecaster:")
print(f"  Parameters: {sum(p.numel() for p in transformer_forecaster.parameters()):,}")
print(f"  Prediction shape: {transformer_pred.shape}")

# Simple training loop for LSTM
def train_forecaster(model, X_train, y_train, epochs=50, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(dataloader))
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {losses[-1]:.6f}")

    return losses

# Train LSTM model
print("\nTraining LSTM forecaster...")
lstm_losses = train_forecaster(lstm_forecaster, X_ts[:800], y_ts[:800], epochs=30)

# Evaluate on test set
lstm_forecaster.eval()
with torch.no_grad():
    test_predictions = lstm_forecaster(X_ts[800:900])
    test_targets = y_ts[800:900]
    test_mse = F.mse_loss(test_predictions, test_targets)
    print(f"Test MSE: {test_mse.item():.6f}")

# Plot results
plt.subplot(3, 2, 3)
plt.plot(lstm_losses)
plt.title('LSTM Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(test_targets.numpy(), 'b-', label='Actual', alpha=0.7)
plt.plot(test_predictions.numpy(), 'r-', label='Predicted', alpha=0.7)
plt.title('LSTM Predictions vs Actual')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Multi-step forecasting
def multi_step_forecast(model, initial_sequence, steps=20):
    """Forecast multiple steps ahead"""
    model.eval()
    forecasts = []
    current_seq = initial_sequence.clone()

    with torch.no_grad():
        for _ in range(steps):
            # Predict next value
            next_pred = model(current_seq.unsqueeze(0))
            forecasts.append(next_pred.item())

            # Update sequence (slide window)
            current_seq = torch.cat([current_seq[1:], next_pred.unsqueeze(0)])

    return forecasts

# Multi-step forecasting example
initial_seq = X_ts[900]
multi_forecasts = multi_step_forecast(lstm_forecaster, initial_seq, steps=20)

plt.subplot(3, 2, 5)
plt.plot(range(50), initial_seq.squeeze(), 'b-', label='Initial Sequence')
plt.plot(range(50, 70), multi_forecasts, 'r-', label='Multi-step Forecast', marker='o')
plt.plot(range(50, 70), full_series[950:970], 'g--', label='True Future', alpha=0.7)
plt.title('Multi-step Forecasting')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 6)
# Compare different sequence lengths
sequence_lengths = [10, 20, 30, 50, 100]
test_errors = []

for seq_len in sequence_lengths:
    # Create data with different sequence lengths
    X_test, y_test, _ = generate_time_series(200, seq_len, 1)

    # Create and quickly train model
    temp_model = LSTMForecaster(1, 32, 1, 1)
    temp_losses = train_forecaster(temp_model, X_test[:150], y_test[:150], epochs=20, lr=0.01)

    # Test
    temp_model.eval()
    with torch.no_grad():
        test_pred = temp_model(X_test[150:])
        test_mse = F.mse_loss(test_pred, y_test[150:]).item()
        test_errors.append(test_mse)

plt.plot(sequence_lengths, test_errors, 'bo-')
plt.title('Sequence Length vs Test Error')
plt.xlabel('Sequence Length')
plt.ylabel('Test MSE')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"\nTime Series Forecasting Summary:")
print(f"  LSTM works well for sequential patterns")
print(f"  Transformers can capture long-range dependencies")
print(f"  Multi-step forecasting becomes less accurate over time")
print(f"  Sequence length affects model performance")

# %% [markdown]
# ## Multi-Modal Learning
#
# Multi-modal learning combines information from different types of data (text, images, audio, etc.) to make better predictions. This is increasingly important for applications like image captioning, visual question answering, and cross-modal retrieval.

# %%
# Multi-modal fusion model (Vision + Text)
class MultiModalFusion(nn.Module):
    def __init__(self, image_features=512, text_features=256,
                 hidden_dim=128, num_classes=10, fusion_method='concat'):
        super(MultiModalFusion, self).__init__()
        self.fusion_method = fusion_method

        # Image encoder (simplified CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, image_features)
        )

        # Text encoder (simplified)
        self.text_encoder = nn.Sequential(
            nn.Embedding(1000, 128),  # Vocabulary size 1000
            nn.LSTM(128, text_features//2, batch_first=True, bidirectional=True)
        )

        # Fusion layers
        if fusion_method == 'concat':
            fusion_input_dim = image_features + text_features
        elif fusion_method == 'hadamard':  # Element-wise product
            # Project to same dimension first
            self.image_proj = nn.Linear(image_features, hidden_dim)
            self.text_proj = nn.Linear(text_features, hidden_dim)
            fusion_input_dim = hidden_dim
        elif fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(image_features, num_heads=4, batch_first=True)
            fusion_input_dim = image_features + text_features

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, images, text):
        # Encode images
        image_features = self.image_encoder(images)

        # Encode text
        text_embedded = self.text_encoder[0](text)
        text_lstm_out, (text_hidden, _) = self.text_encoder[1](text_embedded)
        # Use last hidden state
        text_features = text_hidden[-1]  # Take last layer, forward direction

        # Fusion
        if self.fusion_method == 'concat':
            fused_features = torch.cat([image_features, text_features], dim=1)
        elif self.fusion_method == 'hadamard':
            img_proj = self.image_proj(image_features)
            txt_proj = self.text_proj(text_features)
            fused_features = img_proj * txt_proj  # Element-wise product
        elif self.fusion_method == 'attention':
            # Use text as query, image as key/value
            text_expanded = text_features.unsqueeze(1)  # Add sequence dimension
            img_expanded = image_features.unsqueeze(1)

            attended_features, _ = self.attention(text_expanded, img_expanded, img_expanded)
            fused_features = torch.cat([attended_features.squeeze(1), text_features], dim=1)

        # Classification
        output = self.classifier(fused_features)
        return output

# Create synthetic multi-modal data
def create_multimodal_data(n_samples=1000, img_size=32, max_text_length=20):
    """Create synthetic image-text pairs"""
    # Random images
    images = torch.randn(n_samples, 3, img_size, img_size)

    # Random text (word indices)
    texts = torch.randint(1, 1000, (n_samples, max_text_length))

    # Labels based on some combination of image and text properties
    # (This is synthetic - in practice, labels would be meaningful)
    image_sum = images.mean(dim=(2, 3)).sum(dim=1)
    text_sum = texts.float().mean(dim=1)
    labels = ((image_sum + text_sum) > 0).long()

    return images, texts, labels

# Generate multi-modal data
mm_images, mm_texts, mm_labels = create_multimodal_data(500, 32, 15)

print(f"Multi-Modal Data:")
print(f"  Images shape: {mm_images.shape}")
print(f"  Texts shape: {mm_texts.shape}")
print(f"  Labels shape: {mm_labels.shape}")
print(f"  Label distribution: {torch.bincount(mm_labels)}")

# Test different fusion methods
fusion_methods = ['concat', 'hadamard', 'attention']
models = {}

for method in fusion_methods:
    model = MultiModalFusion(image_features=256, text_features=128,
                            hidden_dim=64, num_classes=2, fusion_method=method)
    models[method] = model

    # Test forward pass
    test_images = mm_images[:3]
    test_texts = mm_texts[:3]

    output = model(test_images, test_texts)

    print(f"\n{method.capitalize()} Fusion:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Output shape: {output.shape}")

# Cross-modal attention mechanism
class CrossModalAttention(nn.Module):
    def __init__(self, dim1, dim2, hidden_dim):
        super(CrossModalAttention, self).__init__()
        self.proj1 = nn.Linear(dim1, hidden_dim)
        self.proj2 = nn.Linear(dim2, hidden_dim)
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)

    def forward(self, features1, features2):
        # Project both modalities to same dimension
        proj1 = self.proj1(features1)
        proj2 = self.proj2(features2)

        # Compute attention weights
        combined = torch.cat([proj1, proj2], dim=1)
        attention = torch.sigmoid(self.attention_weights(combined))

        # Weighted combination
        fused = attention * proj1 + (1 - attention) * proj2

        return fused, attention

# Test cross-modal attention
cross_attention = CrossModalAttention(256, 128, 64)
feat1 = torch.randn(5, 256)
feat2 = torch.randn(5, 128)

fused_features, attention_weights = cross_attention(feat1, feat2)

print(f"\nCross-Modal Attention:")
print(f"  Input 1 shape: {feat1.shape}")
print(f"  Input 2 shape: {feat2.shape}")
print(f"  Fused features shape: {fused_features.shape}")
print(f"  Attention weights shape: {attention_weights.shape}")
print(f"  Sample attention weights: {attention_weights[:3, 0]}")

# Multi-modal training example
def train_multimodal(model, images, texts, labels, epochs=30):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(images, texts, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    losses = []
    accuracies = []

    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_images, batch_texts, batch_labels in dataloader:
            optimizer.zero_grad()

            outputs = model(batch_images, batch_texts)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100 * correct / total

        losses.append(avg_loss)
        accuracies.append(accuracy)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return losses, accuracies

# Train concat fusion model
print("\nTraining concatenation fusion model...")
concat_model = models['concat']
concat_losses, concat_accuracies = train_multimodal(
    concat_model, mm_images[:400], mm_texts[:400], mm_labels[:400], epochs=20
)

# Evaluate
concat_model.eval()
with torch.no_grad():
    test_outputs = concat_model(mm_images[400:], mm_texts[400:])
    _, predicted = torch.max(test_outputs, 1)
    test_accuracy = (predicted == mm_labels[400:]).float().mean() * 100
    print(f"Test accuracy: {test_accuracy:.2f}%")

# Visualize training
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(concat_losses)
plt.title('Multi-Modal Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(concat_accuracies)
plt.title('Multi-Modal Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"\nMulti-Modal Learning Summary:")
print(f"  Fusion methods: concatenation, element-wise product, attention")
print(f"  Cross-modal attention allows dynamic weighting")
print(f"  Each modality contributes complementary information")
print(f"  Applications: image captioning, VQA, multimodal retrieval")

# %% [markdown]
# ## Advanced Architecture Best Practices
#
# Working with advanced architectures requires understanding their specific characteristics, training requirements, and application domains. Here's a comprehensive guide to best practices.

# %%
print("Advanced Architectures: Best Practices and Guidelines")
print("="*70)

architecture_guide = {
    "Graph Neural Networks (GNNs)": {
        "When to Use": [
            "Node/graph classification tasks",
            "Social network analysis",
            "Molecular property prediction",
            "Knowledge graph reasoning",
            "Recommendation systems"
        ],
        "Key Considerations": [
            "Choose appropriate graph convolution (GCN, GraphSAGE, GAT)",
            "Handle variable graph sizes with batching",
            "Consider over-smoothing in deep GNNs",
            "Use appropriate pooling for graph-level tasks",
            "Be aware of scalability with large graphs"
        ],
        "Common Pitfalls": [
            "Ignoring graph structure in data preprocessing",
            "Using too many layers (over-smoothing)",
            "Not handling isolated nodes properly",
            "Inadequate graph augmentation strategies"
        ]
    },

    "Vision Transformers (ViTs)": {
        "When to Use": [
            "Large-scale image classification",
            "When you have abundant training data",
            "Transfer learning from large pre-trained models",
            "Tasks requiring global context"
        ],
        "Key Considerations": [
            "Need large datasets or pre-trained models",
            "Patch size affects model granularity",
            "Position embeddings are crucial",
            "Class token design for classification",
            "Computational cost scales quadratically"
        ],
        "Common Pitfalls": [
            "Training from scratch on small datasets",
            "Using inappropriate patch sizes",
            "Neglecting proper data augmentation",
            "Not leveraging pre-trained models"
        ]
    },

    "Object Detection (YOLO-style)": {
        "When to Use": [
            "Real-time object detection",
            "Multiple object localization",
            "Autonomous driving applications",
            "Surveillance systems"
        ],
        "Key Considerations": [
            "Anchor box design affects performance",
            "Non-maximum suppression for post-processing",
            "Multi-scale detection for different object sizes",
            "Loss function balances classification and localization",
            "Data augmentation specific to bounding boxes"
        ],
        "Common Pitfalls": [
            "Imbalanced positive/negative samples",
            "Inappropriate anchor box sizes",
            "Ignoring IoU thresholds in evaluation",
            "Not handling small objects properly"
        ]
    },

    "Time Series Models": {
        "When to Use": [
            "Sequential data prediction",
            "Financial forecasting",
            "Weather prediction",
            "IoT sensor data analysis"
        ],
        "Key Considerations": [
            "Sequence length affects model memory",
            "Stationarity assumptions in traditional methods",
            "Multi-variate vs univariate forecasting",
            "Seasonality and trend handling",
            "Multi-step vs single-step predictions"
        ],
        "Common Pitfalls": [
            "Data leakage in train/test splits",
            "Not accounting for temporal dependencies",
            "Ignoring data preprocessing (scaling, differencing)",
            "Overfitting to recent patterns"
        ]
    },

    "Multi-Modal Learning": {
        "When to Use": [
            "Image captioning",
            "Visual question answering",
            "Cross-modal retrieval",
            "Multimodal sentiment analysis"
        ],
        "Key Considerations": [
            "Choose appropriate fusion strategy",
            "Handle modality imbalance",
            "Alignment between modalities",
            "Missing modality robustness",
            "Cross-modal attention mechanisms"
        ],
        "Common Pitfalls": [
            "One modality dominating the learning",
            "Poor temporal/spatial alignment",
            "Inadequate fusion strategy",
            "Not handling missing modalities"
        ]
    }
}

for architecture, info in architecture_guide.items():
    print(f"\n{architecture}:")
    print(f"  When to Use:")
    for use_case in info["When to Use"]:
        print(f"    • {use_case}")

    print(f"  Key Considerations:")
    for consideration in info["Key Considerations"]:
        print(f"    • {consideration}")

    print(f"  Common Pitfalls:")
    for pitfall in info["Common Pitfalls"]:
        print(f"    ⚠  {pitfall}")

print("\n" + "="*70)
print("GENERAL BEST PRACTICES FOR ADVANCED ARCHITECTURES:")
print("="*70)

general_practices = {
    "Data Preparation": [
        "Understand data structure and relationships",
        "Apply domain-specific preprocessing",
        "Use appropriate data augmentation",
        "Handle missing or corrupted data gracefully",
        "Ensure proper train/validation/test splits"
    ],

    "Model Design": [
        "Start with established architectures",
        "Gradually increase model complexity",
        "Use appropriate inductive biases",
        "Consider computational constraints",
        "Design for interpretability when needed"
    ],

    "Training Strategies": [
        "Use transfer learning when possible",
        "Implement proper regularization",
        "Monitor multiple metrics",
        "Use learning rate scheduling",
        "Apply gradient clipping for stability"
    ],

    "Evaluation & Debugging": [
        "Use domain-appropriate metrics",
        "Validate on realistic test cases",
        "Visualize intermediate representations",
        "Perform ablation studies",
        "Test edge cases and failure modes"
    ],

    "Production Considerations": [
        "Profile computational requirements",
        "Plan for model versioning",
        "Implement monitoring and alerting",
        "Consider model compression techniques",
        "Design for graceful degradation"
    ]
}

for category, practices in general_practices.items():
    print(f"\n{category}:")
    for practice in practices:
        print(f"  • {practice}")

print("\n" + "="*70)
print("ARCHITECTURE SELECTION GUIDE:")
print("="*70)

selection_guide = {
    "Data Type": {
        "Images": "CNNs, Vision Transformers, Object Detection models",
        "Text": "RNNs, LSTMs, Transformers, BERT-style models",
        "Graphs": "Graph Neural Networks (GCN, GraphSAGE, GAT)",
        "Time Series": "RNNs, LSTMs, Temporal CNNs, Transformers",
        "Multi-Modal": "Fusion architectures, Cross-modal attention"
    },

    "Task Type": {
        "Classification": "Standard architectures with classification heads",
        "Regression": "Similar to classification but with regression outputs",
        "Generation": "Generative models (GANs, VAEs, Autoregressive)",
        "Detection": "Object detection architectures (YOLO, R-CNN)",
        "Segmentation": "U-Net, FCN, DeepLab architectures"
    },

    "Dataset Size": {
        "Small (<1K)": "Transfer learning, data augmentation, simple models",
        "Medium (1K-100K)": "Fine-tuning, moderate complexity models",
        "Large (100K-1M)": "Custom architectures, from-scratch training",
        "Very Large (>1M)": "Large-scale architectures, distributed training"
    },

    "Computational Budget": {
        "Limited": "Lightweight models, pruning, quantization",
        "Moderate": "Standard architectures, efficient implementations",
        "High": "Large models, ensemble methods, extensive search",
        "Unlimited": "Massive models, neural architecture search"
    }
}

for criterion, recommendations in selection_guide.items():
    print(f"\n{criterion}:")
    for key, value in recommendations.items():
        print(f"  {key}: {value}")

print("\n" + "="*70)
print("Remember: The best architecture depends on your specific problem,")
print("data characteristics, computational constraints, and performance requirements.")
print("Start simple, iterate based on results, and always validate thoroughly!")
print("="*70)

# %%