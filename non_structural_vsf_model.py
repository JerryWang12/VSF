import torch
import torch.nn as nn
import torch.nn.functional as F


class NonStructuralVSFModel(nn.Module):
    def __init__(self, input_dim, embed_dim=64, latent_dim=32, num_layers=4, num_heads=4, max_len=5000, dropout=0.1):
        super(NonStructuralVSFModel, self).__init__()

        # Variable Embedding
        self.var_embedding = nn.Parameter(torch.randn(input_dim, embed_dim))
        self.time_embedding = nn.Parameter(torch.randn(max_len, embed_dim))

        # Cross-dimensional self-attention for variable interaction
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        # CVAE components
        self.encoder_mu = nn.Linear(embed_dim, latent_dim)
        self.encoder_logvar = nn.Linear(embed_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim + embed_dim, embed_dim)

        # Output layer
        self.output_layer = nn.Linear(embed_dim, input_dim)

    def forward(self, x, mask):
        # Apply variable and time embeddings
        batch_size, seq_len, num_vars = x.shape
        var_embed = self.var_embedding.unsqueeze(0).expand(batch_size, seq_len, -1)  # (B, T, D)
        time_embed = self.time_embedding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)  # (B, T, E)

        # Combine variable and time embeddings
        x_embed = x + var_embed + time_embed

        # Apply multi-layer cross-dimensional self-attention
        for layer in self.attention_layers:
            x_embed = layer(x_embed)

        # Encoder (CVAE)
        x_pooled = x_embed.mean(dim=1)  # (B, embed_dim)
        mu = self.encoder_mu(x_pooled)  # (B, latent_dim)
        logvar = self.encoder_logvar(x_pooled)  # (B, latent_dim)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        # Decoder (CVAE)
        z_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, latent_dim)
        decoder_input = torch.cat([z_expanded, x_embed], dim=-1)  # (B, T, latent_dim + embed_dim)
        decoded = self.decoder(decoder_input)  # (B, T, embed_dim)

        # Project to original variable space
        output = self.output_layer(decoded)  # (B, T, num_vars)

        return output, mu, logvar

    def compute_loss(self, recon_x, x, mu, logvar, mask, beta=0.1):
        # MSE reconstruction loss (only for masked positions)
        recon_loss = F.mse_loss(recon_x * (1 - mask), x * (1 - mask), reduction="mean")

        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss
