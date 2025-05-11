import torch
import torch.nn as nn


class NonStructureAwareModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=4, num_heads=4, output_dim=None, dropout=0.3):
        super(NonStructureAwareModel, self).__init__()

        self.time_embedding = nn.Linear(input_dim, hidden_dim)
        self.var_embedding = nn.Linear(input_dim, hidden_dim)

        self.self_attention_time = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.self_attention_var = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim or input_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, V = x.size()

        # Time and variable encoding
        time_features = self.time_embedding(x).permute(1, 0, 2)  # (T, B, hidden_dim)
        var_features = self.var_embedding(x.transpose(1, 2)).permute(1, 0, 2)  # (V, B, hidden_dim)

        # Self-attention
        time_out, _ = self.self_attention_time(time_features, time_features, time_features)
        var_out, _ = self.self_attention_var(var_features, var_features, var_features)

        # Fuse and project
        fused_features = time_out.permute(1, 0, 2) + var_out.permute(1, 0, 2)
        out = self.fc(self.dropout(fused_features))

        return out
