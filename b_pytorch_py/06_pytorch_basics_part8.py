# %% [markdown]
# # PyTorch Basics Part 8: Generative Models and Unsupervised Learning
#
# Exploring autoencoders, GANs, VAEs, and self-supervised learning with mathematical foundations for generating and learning from unlabeled data
#
# ## Mathematical Framework for Generative Modeling
#
# **Generative models** learn probability distributions over data to generate new samples and discover latent structure:
#
# ### Core Mathematical Concepts
#
# **1. Generative vs Discriminative Models:**
# - **Discriminative**: $P(y|\mathbf{x})$ - predict labels given data
# - **Generative**: $P(\mathbf{x})$ or $P(\mathbf{x}, y)$ - model data distribution
#
# **2. Density Estimation:**
# Learn probability density function $p_{\boldsymbol{\theta}}(\mathbf{x})$ that approximates true data distribution $p_{\text{data}}(\mathbf{x})$.
#
# **3. Latent Variable Models:**
# $$p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z})d\mathbf{z}$$
#
# where $\mathbf{z} \in \mathbb{R}^d$ is latent variable, $\mathbf{x} \in \mathbb{R}^D$ is observed data.
#
# **4. Maximum Likelihood Estimation:**
# $$\boldsymbol{\theta}^* = \arg\max_{\boldsymbol{\theta}} \sum_{i=1}^N \log p_{\boldsymbol{\theta}}(\mathbf{x}_i)$$
#
# **5. Information-Theoretic Measures:**
# - **KL Divergence**: $D_{KL}(p||q) = \int p(\mathbf{x}) \log \frac{p(\mathbf{x})}{q(\mathbf{x})} d\mathbf{x}$
# - **Jensen-Shannon Divergence**: $D_{JS}(p||q) = \frac{1}{2}D_{KL}(p||m) + \frac{1}{2}D_{KL}(q||m)$
#   where $m = \frac{1}{2}(p+q)$
#
# **6. Representation Learning Objectives:**
# - **Reconstruction**: Minimize $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$
# - **Regularization**: Constrain latent space structure
# - **Disentanglement**: Learn interpretable factors of variation
#
# **7. Sampling Methods:**
# - **Ancestral sampling**: Sample $\mathbf{z} \sim p(\mathbf{z})$, then $\mathbf{x} \sim p(\mathbf{x}|\mathbf{z})$
# - **MCMC methods**: For complex posteriors
# - **Variational approximation**: Tractable inference

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Autoencoders: Learning Compressed Representations
#
# Autoencoders learn to compress data into a lower-dimensional representation (encoding) and then reconstruct the original data (decoding). They're useful for dimensionality reduction, denoising, and feature learning.

# %%
# Basic Autoencoder implementation
class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(SimpleAutoencoder, self).__init__()

        # Encoder: compress input to latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

        # Decoder: reconstruct input from latent representation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Assuming input is normalized to [0,1]
        )

    def forward(self, x):
        # Encode
        latent = self.encoder(x)
        # Decode
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# Create synthetic high-dimensional data
def create_synthetic_data(n_samples=1000, n_features=50, n_informative=5):
    """Create high-dimensional data with underlying low-dimensional structure"""
    # Generate low-dimensional structure
    base_data = np.random.randn(n_samples, n_informative)

    # Create high-dimensional data by linear combination + noise
    mixing_matrix = np.random.randn(n_informative, n_features)
    high_dim_data = base_data @ mixing_matrix

    # Add some noise
    noise = 0.1 * np.random.randn(n_samples, n_features)
    high_dim_data += noise

    # Normalize to [0, 1]
    high_dim_data = (high_dim_data - high_dim_data.min()) / (high_dim_data.max() - high_dim_data.min())

    return torch.FloatTensor(high_dim_data), torch.FloatTensor(base_data)

# Generate data
X_high, X_true = create_synthetic_data(1000, 50, 3)
print(f"High-dimensional data shape: {X_high.shape}")
print(f"True low-dimensional structure shape: {X_true.shape}")

# Create autoencoder
autoencoder = SimpleAutoencoder(input_dim=50, hidden_dim=128, latent_dim=3)
print(f"Autoencoder parameters: {sum(p.numel() for p in autoencoder.parameters()):,}")

# Test forward pass
sample_input = X_high[:5]
reconstructed, latent = autoencoder(sample_input)
print(f"Input shape: {sample_input.shape}")
print(f"Latent shape: {latent.shape}")
print(f"Reconstructed shape: {reconstructed.shape}")

# %% [markdown]
# ## Training Autoencoders
#
# Autoencoders are trained to minimize reconstruction error - the difference between the input and the reconstructed output. This forces the network to learn meaningful representations in the bottleneck layer.

# %%
# Training function for autoencoders
def train_autoencoder(model, data_loader, num_epochs=100, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch_x in data_loader:
            # Forward pass
            reconstructed, latent = model(batch_x)

            # Compute reconstruction loss
            loss = criterion(reconstructed, batch_x)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)
        losses.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

    return losses

# Create data loader
dataset = TensorDataset(X_high)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train autoencoder
print("Training autoencoder...")
losses = train_autoencoder(autoencoder, data_loader, num_epochs=100)

# Plot training loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)

# Evaluate reconstruction quality
autoencoder.eval()
with torch.no_grad():
    test_reconstructed, test_latent = autoencoder(X_high[:100])
    reconstruction_error = F.mse_loss(test_reconstructed, X_high[:100])
    print(f"Final reconstruction error: {reconstruction_error:.6f}")

# Visualize latent representations
plt.subplot(1, 2, 2)
latent_np = test_latent.numpy()
plt.scatter(latent_np[:, 0], latent_np[:, 1], alpha=0.6)
plt.title('Learned Latent Representation')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Denoising Autoencoders
#
# Denoising autoencoders learn robust representations by training to reconstruct clean data from corrupted inputs. This helps the model learn more meaningful features and improves generalization.

# %%
# Denoising Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate=0.2):
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Dropout(dropout_rate),  # Add noise through dropout
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x, add_noise=True):
        if add_noise and self.training:
            # Add Gaussian noise during training
            noise = 0.1 * torch.randn_like(x)
            noisy_x = torch.clamp(x + noise, 0, 1)
        else:
            noisy_x = x

        latent = self.encoder(noisy_x)
        reconstructed = self.decoder(latent)

        return reconstructed, latent

# Create and train denoising autoencoder
denoising_ae = DenoisingAutoencoder(input_dim=50, hidden_dim=128, latent_dim=3)

def train_denoising_ae(model, data_loader, num_epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch_x in data_loader:
            # Forward pass with noise
            reconstructed, _ = model(batch_x, add_noise=True)

            # Loss: reconstruct clean data from noisy input
            loss = criterion(reconstructed, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / len(data_loader))

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {losses[-1]:.6f}")

    return losses

print("Training denoising autoencoder...")
denoising_losses = train_denoising_ae(denoising_ae, data_loader, 80)

# Compare denoising performance
denoising_ae.eval()
with torch.no_grad():
    # Add noise to test data
    test_data = X_high[:20]
    noise = 0.2 * torch.randn_like(test_data)
    noisy_test_data = torch.clamp(test_data + noise, 0, 1)

    # Reconstruct
    denoised, _ = denoising_ae(noisy_test_data, add_noise=False)

    # Compute metrics
    original_mse = F.mse_loss(noisy_test_data, test_data)
    denoised_mse = F.mse_loss(denoised, test_data)

    print(f"\nDenoising Results:")
    print(f"Noisy data MSE: {original_mse:.6f}")
    print(f"Denoised data MSE: {denoised_mse:.6f}")
    print(f"Improvement: {((original_mse - denoised_mse) / original_mse * 100):.1f}%")

# %% [markdown]
# ## Variational Autoencoders (VAEs)
#
# VAEs extend autoencoders by learning a probabilistic latent space. Instead of encoding to a fixed point, they encode to a distribution (mean and variance), enabling generation of new samples.

# %%
# Variational Autoencoder implementation
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder network
        self.encoder_hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)     # Mean
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim) # Log variance

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        """Encode input to latent parameters"""
        hidden = self.encoder_hidden(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent variable to reconstruction"""
        return self.decoder(z)

    def forward(self, x):
        # Encode
        mu, logvar = self.encode(x)

        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)

        # Decode
        reconstructed = self.decode(z)

        return reconstructed, mu, logvar

    def generate(self, num_samples, device='cpu'):
        """Generate new samples from the learned distribution"""
        self.eval()
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_samples, self.fc_mu.out_features).to(device)
            # Decode to generate new samples
            generated = self.decode(z)
        return generated

# VAE loss function combines reconstruction loss and KL divergence
def vae_loss_function(reconstructed, original, mu, logvar, beta=1.0):
    """VAE loss: reconstruction loss + KL divergence"""
    # Reconstruction loss (binary cross-entropy)
    reconstruction_loss = F.binary_cross_entropy(reconstructed, original, reduction='sum')

    # KL divergence loss (regularizes latent space to be close to standard normal)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = reconstruction_loss + beta * kl_loss

    return total_loss, reconstruction_loss, kl_loss

# Create VAE
vae = VAE(input_dim=50, hidden_dim=128, latent_dim=3)
print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")

# Test VAE forward pass
test_input = X_high[:5]
recon, mu, logvar = vae(test_input)
total_loss, recon_loss, kl_loss = vae_loss_function(recon, test_input, mu, logvar)

print(f"Test reconstruction shape: {recon.shape}")
print(f"Test mu shape: {mu.shape}")
print(f"Test logvar shape: {logvar.shape}")
print(f"Total loss: {total_loss.item():.4f}")
print(f"Reconstruction loss: {recon_loss.item():.4f}")
print(f"KL loss: {kl_loss.item():.4f}")

# %% [markdown]
# ## Training Variational Autoencoders
#
# VAE training balances reconstruction quality with regularization of the latent space. The KL divergence term ensures the latent space follows a standard normal distribution, enabling generation.

# %%
# Training function for VAE
def train_vae(model, data_loader, num_epochs=100, beta=1.0, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = {'total': [], 'reconstruction': [], 'kl': []}

    model.train()
    for epoch in range(num_epochs):
        epoch_losses = {'total': 0, 'reconstruction': 0, 'kl': 0}

        for batch_x in data_loader:
            # Forward pass
            reconstructed, mu, logvar = model(batch_x)

            # Compute loss
            total_loss, recon_loss, kl_loss = vae_loss_function(
                reconstructed, batch_x, mu, logvar, beta
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['reconstruction'] += recon_loss.item()
            epoch_losses['kl'] += kl_loss.item()

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(data_loader)
            losses[key].append(epoch_losses[key])

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Total: {epoch_losses['total']:.4f}")
            print(f"  Reconstruction: {epoch_losses['reconstruction']:.4f}")
            print(f"  KL: {epoch_losses['kl']:.4f}")

    return losses

# Train VAE
print("Training VAE...")
vae_losses = train_vae(vae, data_loader, num_epochs=100, beta=1.0)

# Plot training losses
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(vae_losses['total'])
plt.title('Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(vae_losses['reconstruction'], label='Reconstruction')
plt.plot(vae_losses['kl'], label='KL Divergence')
plt.title('Loss Components')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Generate new samples
vae.eval()
generated_samples = vae.generate(100)

# Visualize latent space
with torch.no_grad():
    test_mu, test_logvar = vae.encode(X_high[:200])

plt.subplot(1, 3, 3)
plt.scatter(test_mu[:, 0].numpy(), test_mu[:, 1].numpy(), alpha=0.6, label='Encoded')
# Sample from prior
prior_samples = torch.randn(100, 2)
plt.scatter(prior_samples[:, 0].numpy(), prior_samples[:, 1].numpy(),
           alpha=0.6, label='Prior', marker='x')
plt.title('Latent Space')
plt.xlabel('Latent Dim 1')
plt.ylabel('Latent Dim 2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Generated samples shape: {generated_samples.shape}")
print(f"Generated samples range: [{generated_samples.min():.3f}, {generated_samples.max():.3f}]")

# %% [markdown]
# ## Generative Adversarial Networks (GANs)
#
# GANs consist of two competing networks: a generator that creates fake data and a discriminator that tries to distinguish real from fake data. This adversarial training often produces high-quality generated samples.

# %%
# Generator Network
class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, noise):
        return self.network(noise)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )

    def forward(self, x):
        return self.network(x).view(-1)

# Create simple 2D dataset for visualization
def create_2d_data(n_samples=1000, dataset_type='moons'):
    """Create 2D datasets for GAN visualization"""
    if dataset_type == 'moons':
        data, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    elif dataset_type == 'blobs':
        data, _ = make_blobs(n_samples=n_samples, centers=4, cluster_std=0.3, random_state=42)
    else:  # circular
        angles = np.random.uniform(0, 2*np.pi, n_samples)
        radii = 2 + 0.5 * np.random.randn(n_samples)
        data = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    # Normalize to [-1, 1] for tanh activation
    data = 2 * (data - data.min()) / (data.max() - data.min()) - 1
    return torch.FloatTensor(data)

# Create 2D dataset
real_data = create_2d_data(1000, 'moons')
data_dim = real_data.shape[1]

# Visualize real data
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.6)
plt.title('Real Data Distribution')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)

# Create GAN
noise_dim = 2
hidden_dim = 128

generator = Generator(noise_dim, hidden_dim, data_dim)
discriminator = Discriminator(data_dim, hidden_dim)

print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

# Test generator
test_noise = torch.randn(100, noise_dim)
fake_data = generator(test_noise)

plt.subplot(1, 2, 2)
plt.scatter(fake_data[:, 0].detach().numpy(), fake_data[:, 1].detach().numpy(), alpha=0.6)
plt.title('Initial Generated Data (Untrained)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Real data shape: {real_data.shape}")
print(f"Generated data shape: {fake_data.shape}")

# %% [markdown]
# ## Training GANs
#
# GAN training involves alternating between training the discriminator (to better distinguish real from fake) and the generator (to better fool the discriminator). This adversarial process can be unstable and requires careful tuning.

# %%
# GAN training function
def train_gan(generator, discriminator, real_data, num_epochs=200,
              batch_size=32, noise_dim=2, lr=0.0002):

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCELoss()

    # Data loader
    dataset = TensorDataset(real_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training metrics
    d_losses = []
    g_losses = []

    # Labels
    real_label = 1.0
    fake_label = 0.0

    for epoch in range(num_epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0

        for batch_real in dataloader:
            batch_real = batch_real[0]  # Extract from tuple
            batch_size_actual = batch_real.size(0)

            # Train Discriminator
            discriminator.zero_grad()

            # Real data
            real_labels = torch.full((batch_size_actual,), real_label, dtype=torch.float)
            real_output = discriminator(batch_real)
            d_loss_real = criterion(real_output, real_labels)

            # Fake data
            noise = torch.randn(batch_size_actual, noise_dim)
            fake_data = generator(noise)
            fake_labels = torch.full((batch_size_actual,), fake_label, dtype=torch.float)
            fake_output = discriminator(fake_data.detach())
            d_loss_fake = criterion(fake_output, fake_labels)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            generator.zero_grad()

            # Generate fake data and try to fool discriminator
            fake_output = discriminator(fake_data)
            g_loss = criterion(fake_output, real_labels)  # Want discriminator to think it's real
            g_loss.backward()
            g_optimizer.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

        # Average losses
        d_losses.append(epoch_d_loss / len(dataloader))
        g_losses.append(epoch_g_loss / len(dataloader))

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  D Loss: {d_losses[-1]:.4f}, G Loss: {g_losses[-1]:.4f}")
            print(f"  D(real): {real_output.mean():.4f}, D(fake): {fake_output.mean():.4f}")

    return d_losses, g_losses

# Train GAN
print("Training GAN...")
d_losses, g_losses = train_gan(generator, discriminator, real_data,
                              num_epochs=200, batch_size=32)

# Plot training losses
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(d_losses, label='Discriminator')
plt.plot(g_losses, label='Generator')
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Generate samples after training
generator.eval()
discriminator.eval()

with torch.no_grad():
    # Generate new samples
    test_noise = torch.randn(1000, noise_dim)
    generated_samples = generator(test_noise)

    # Evaluate discriminator on real and fake data
    real_scores = discriminator(real_data[:1000])
    fake_scores = discriminator(generated_samples)

# Plot generated samples
plt.subplot(1, 3, 2)
plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.6, label='Real', s=20)
plt.scatter(generated_samples[:, 0], generated_samples[:, 1],
           alpha=0.6, label='Generated', s=20)
plt.title('Real vs Generated Data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)

# Plot discriminator scores
plt.subplot(1, 3, 3)
plt.hist(real_scores.numpy(), bins=30, alpha=0.7, label='Real', density=True)
plt.hist(fake_scores.numpy(), bins=30, alpha=0.7, label='Generated', density=True)
plt.title('Discriminator Scores')
plt.xlabel('Score (Real Probability)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Real data scores - Mean: {real_scores.mean():.3f}, Std: {real_scores.std():.3f}")
print(f"Generated data scores - Mean: {fake_scores.mean():.3f}, Std: {fake_scores.std():.3f}")

# %% [markdown]
# ## Self-Supervised Learning
#
# Self-supervised learning creates supervisory signals from the data itself, without external labels. Common approaches include predicting masked tokens, next frames, or transformations applied to the data.

# %%
# Self-supervised learning: Predicting rotations
class RotationPredictor(nn.Module):
    """Predict rotation angles applied to data (simplified example)"""

    def __init__(self, input_dim, hidden_dim, num_rotations=4):
        super(RotationPredictor, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        self.rotation_classifier = nn.Linear(hidden_dim // 2, num_rotations)

    def forward(self, x):
        features = self.encoder(x)
        rotation_pred = self.rotation_classifier(features)
        return rotation_pred, features

# Create augmented data with rotations
def create_rotation_task(data, rotation_angles=[0, 90, 180, 270]):
    """Create self-supervised rotation prediction task"""
    augmented_data = []
    labels = []

    for i, angle in enumerate(rotation_angles):
        # Simulate rotation by applying transformation matrix
        # (This is a simplified 2D rotation for demonstration)
        angle_rad = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])

        # Apply rotation to 2D data
        if data.shape[1] == 2:
            rotated = data @ rotation_matrix.T
        else:
            # For higher dimensional data, apply rotation to first 2 dimensions
            rotated = data.clone()
            rotated[:, :2] = data[:, :2] @ rotation_matrix.T

        augmented_data.append(rotated)
        labels.extend([i] * len(data))

    all_data = torch.cat(augmented_data, dim=0)
    all_labels = torch.LongTensor(labels)

    return all_data, all_labels

# Create self-supervised learning task
ssl_data, ssl_labels = create_rotation_task(real_data[:200])

print(f"Self-supervised data shape: {ssl_data.shape}")
print(f"Self-supervised labels shape: {ssl_labels.shape}")
print(f"Label distribution: {torch.bincount(ssl_labels)}")

# Visualize rotated data
plt.figure(figsize=(12, 3))
rotation_names = ['0°', '90°', '180°', '270°']

for i in range(4):
    plt.subplot(1, 4, i+1)
    mask = ssl_labels == i
    plt.scatter(ssl_data[mask, 0], ssl_data[mask, 1], alpha=0.6, s=10)
    plt.title(f'Rotation {rotation_names[i]}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True)
    plt.axis('equal')

plt.tight_layout()
plt.show()

# Train rotation predictor
rotation_model = RotationPredictor(input_dim=2, hidden_dim=64, num_rotations=4)

def train_self_supervised(model, data, labels, num_epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_data, batch_labels in dataloader:
            # Forward pass
            rotation_pred, features = model(batch_data)
            loss = criterion(rotation_pred, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(rotation_pred, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100 * correct / total

        losses.append(avg_loss)
        accuracies.append(accuracy)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return losses, accuracies

print("Training self-supervised rotation predictor...")
ssl_losses, ssl_accuracies = train_self_supervised(rotation_model, ssl_data, ssl_labels, 100)

# Plot results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(ssl_losses)
plt.title('Self-Supervised Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(ssl_accuracies)
plt.title('Rotation Prediction Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Final rotation prediction accuracy: {ssl_accuracies[-1]:.2f}%")

# %% [markdown]
# ## Contrastive Learning
#
# Contrastive learning learns representations by pulling similar samples together and pushing dissimilar samples apart in the embedding space. This is particularly powerful for learning from unlabeled data.

# %%
# Simple contrastive learning implementation
class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ContrastiveEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        # L2 normalize for cosine similarity
        return F.normalize(features, p=2, dim=1)

# Contrastive loss (simplified InfoNCE)
def contrastive_loss(features1, features2, temperature=0.5):
    """Compute contrastive loss between two sets of features"""
    batch_size = features1.shape[0]

    # Compute similarities
    similarity_matrix = torch.mm(features1, features2.t()) / temperature

    # Create labels (diagonal elements are positive pairs)
    labels = torch.arange(batch_size).long()

    # Compute cross-entropy loss
    loss = F.cross_entropy(similarity_matrix, labels)

    return loss

# Data augmentation for contrastive learning
def augment_data(data, noise_std=0.1):
    """Apply simple augmentations"""
    # Add Gaussian noise
    augmented = data + torch.randn_like(data) * noise_std
    return augmented

# Create contrastive learning setup
contrastive_encoder = ContrastiveEncoder(input_dim=2, hidden_dim=64, output_dim=16)

def train_contrastive(model, data, num_epochs=200, batch_size=32, temperature=0.5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch_data in dataloader:
            batch_data = batch_data[0]

            # Create two augmented views of the same data
            view1 = augment_data(batch_data, 0.1)
            view2 = augment_data(batch_data, 0.1)

            # Encode both views
            features1 = model(view1)
            features2 = model(view2)

            # Compute contrastive loss
            loss = contrastive_loss(features1, features2, temperature)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / len(dataloader))

        if (epoch + 1) % 40 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {losses[-1]:.4f}")

    return losses

# Train contrastive model
print("Training contrastive learning model...")
contrastive_losses = train_contrastive(contrastive_encoder, real_data, num_epochs=200)

# Evaluate learned representations
contrastive_encoder.eval()
with torch.no_grad():
    embeddings = contrastive_encoder(real_data)

    # Apply k-means clustering on learned embeddings
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings.numpy())

# Visualize results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(contrastive_losses)
plt.title('Contrastive Learning Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(real_data[:, 0], real_data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.title('Clustering on Original Data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.colorbar()
plt.grid(True)

plt.subplot(1, 3, 3)
# Visualize embeddings (first 2 dimensions)
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.title('Learned Embeddings (2D projection)')
plt.xlabel('Embedding Dim 1')
plt.ylabel('Embedding Dim 2')
plt.colorbar()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Learned embedding dimension: {embeddings.shape[1]}")
print(f"Embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
print(f"Number of clusters found: {len(np.unique(cluster_labels))}")

# %% [markdown]
# ## Generative Model Comparison and Best Practices
#
# Each generative model has different strengths, use cases, and training considerations. Understanding when to use each approach is crucial for successful applications.

# %%
print("Generative Models and Unsupervised Learning: Summary and Best Practices")

print("\n" + "="*60)
print("MODEL COMPARISON:")
print("="*60)

models_comparison = {
    "Autoencoders": {
        "Strengths": [
            "Simple to implement and train",
            "Good for dimensionality reduction",
            "Stable training process",
            "Interpretable latent space"
        ],
        "Weaknesses": [
            "Limited generation quality",
            "Deterministic encoding",
            "May not capture data distribution well"
        ],
        "Use Cases": [
            "Dimensionality reduction",
            "Anomaly detection",
            "Data denoising",
            "Feature learning"
        ]
    },

    "VAEs": {
        "Strengths": [
            "Probabilistic latent space",
            "Principled framework",
            "Good for interpolation",
            "Stable training"
        ],
        "Weaknesses": [
            "Blurry generations",
            "KL divergence can be tricky",
            "May not capture sharp details"
        ],
        "Use Cases": [
            "Data generation",
            "Latent space exploration",
            "Semi-supervised learning",
            "Representation learning"
        ]
    },

    "GANs": {
        "Strengths": [
            "High-quality generations",
            "Sharp, realistic outputs",
            "No assumptions about data distribution",
            "Powerful for image generation"
        ],
        "Weaknesses": [
            "Unstable training",
            "Mode collapse issues",
            "Difficult to evaluate",
            "No explicit likelihood"
        ],
        "Use Cases": [
            "High-quality image generation",
            "Data augmentation",
            "Style transfer",
            "Super-resolution"
        ]
    }
}

for model_name, info in models_comparison.items():
    print(f"\n{model_name}:")
    print(f"  Strengths:")
    for strength in info["Strengths"]:
        print(f"    + {strength}")
    print(f"  Weaknesses:")
    for weakness in info["Weaknesses"]:
        print(f"    - {weakness}")
    print(f"  Best for: {', '.join(info['Use Cases'])}")

print("\n" + "="*60)
print("TRAINING BEST PRACTICES:")
print("="*60)

training_tips = {
    "General": [
        "Start with simple architectures and gradually increase complexity",
        "Use appropriate learning rates (often lower than supervised learning)",
        "Monitor training stability and convergence",
        "Use validation data to prevent overfitting",
        "Consider data preprocessing and normalization carefully"
    ],

    "Autoencoders": [
        "Choose bottleneck size carefully (not too small, not too large)",
        "Use skip connections for better reconstruction",
        "Add noise for denoising autoencoders",
        "Consider sparse regularization for interpretability"
    ],

    "VAEs": [
        "Balance reconstruction and KL losses with β parameter",
        "Use β-annealing (start with β=0, gradually increase)",
        "Monitor KL collapse (posterior collapse)",
        "Consider different prior distributions"
    ],

    "GANs": [
        "Use different learning rates for G and D",
        "Apply spectral normalization for stability",
        "Use gradient penalty instead of weight clipping",
        "Monitor discriminator/generator balance",
        "Use progressive growing for high-resolution images"
    ],

    "Self-Supervised Learning": [
        "Design pretext tasks relevant to downstream tasks",
        "Use strong data augmentations",
        "Consider contrastive learning for representation quality",
        "Evaluate learned representations on downstream tasks"
    ]
}

for category, tips in training_tips.items():
    print(f"\n{category}:")
    for tip in tips:
        print(f"  • {tip}")

print("\n" + "="*60)
print("EVALUATION METRICS:")
print("="*60)

evaluation_metrics = {
    "Generation Quality": [
        "Inception Score (IS) - for image quality",
        "Fréchet Inception Distance (FID) - distribution similarity",
        "Perceptual metrics - human evaluation",
        "BLEU scores - for text generation"
    ],
    "Representation Quality": [
        "Downstream task performance",
        "Linear probe accuracy",
        "Clustering metrics (silhouette score, ARI)",
        "t-SNE/UMAP visualizations"
    ],
    "Model-Specific": [
        "Reconstruction error (AE, VAE)",
        "Log-likelihood (VAE)",
        "KL divergence (VAE)",
        "Mode coverage (GAN)"
    ]
}

for category, metrics in evaluation_metrics.items():
    print(f"\n{category}:")
    for metric in metrics:
        print(f"  • {metric}")

print("\n" + "="*60)
print("CHOOSING THE RIGHT APPROACH:")
print("="*60)

decision_guide = {
    "Need high-quality generations?": "Consider GANs or advanced VAEs",
    "Want interpretable latent space?": "Use VAEs or β-VAEs",
    "Limited computational resources?": "Start with autoencoders",
    "Need stable training?": "Avoid GANs, prefer VAEs or autoencoders",
    "Working with images?": "GANs often work best",
    "Working with time series?": "Consider RNN-based VAEs or autoencoders",
    "Need to generate diverse samples?": "Check for mode collapse in GANs",
    "Want to learn representations?": "Self-supervised or contrastive learning"
}

for question, answer in decision_guide.items():
    print(f"\n{question}")
    print(f"  → {answer}")

print("\n" + "="*60)
print("Remember: The choice of generative model depends heavily on your specific")
print("use case, data type, computational constraints, and quality requirements.")
print("Always start simple and iterate based on results!")
print("="*60)

# %%