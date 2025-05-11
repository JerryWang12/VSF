# Test NonStructuralVSFModel
from models.non_structural_vsf_model import NonStructuralVSFModel
import torch

# Simulate some sample data
batch_size = 8
seq_len = 12
num_vars = 137
sample_data = torch.randn(batch_size, seq_len, num_vars)
sample_mask = torch.randint(0, 2, (batch_size, seq_len, num_vars)).float()

# Initialize model
model = NonStructuralVSFModel(input_dim=num_vars)
output, mu, logvar = model(sample_data, sample_mask)

print(f"Output shape: {output.shape}")  # (B, T, D)
print(f"Latent mean shape: {mu.shape}")  # (B, latent_dim)
print(f"Latent logvar shape: {logvar.shape}")  # (B, latent_dim)

# Calculate loss
loss, recon_loss, kl_loss = model.compute_loss(output, sample_data, mu, logvar, sample_mask)
print(f"Total loss: {loss.item()}, Recon loss: {recon_loss.item()}, KL loss: {kl_loss.item()}")
