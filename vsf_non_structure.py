import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


# 1. 数据预处理与变量/时间嵌入模块
class VariatesEmbedding(nn.Module):
    def __init__(self, num_vars, embedding_dim, max_len=5000):
        super().__init__()
        self.var_embedding = nn.Embedding(num_vars, embedding_dim)
        self.position_encoding = self._generate_positional_encoding(embedding_dim, max_len)

    def _generate_positional_encoding(self, embedding_dim, max_len):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        B, T, D = x.size()
        device = x.device

        # 变量嵌入 (D, E)
        var_embed = self.var_embedding(torch.arange(D, device=device))  # (D, E)

        # 时间位置信息 (T, E)
        pos_embed = self.position_encoding[:T, :].to(device)  # (T, E)

        # 扩展维度到 (B, T, D, E)
        var_embed = var_embed[None, None, :, :].expand(B, T, -1, -1) 
        pos_embed = pos_embed[None, :, None, :].expand(B, -1, D, -1) 

        # 加入时间位置信息
        return var_embed + pos_embed


# 2. 交互关系编码模块
class DualBranchAttention(nn.Module):
    def __init__(self, num_vars, seq_len, embed_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.num_vars = num_vars
        self.embed_dim = embed_dim
        
        # ===== 时间注意力分支 =====
        self.time_proj = nn.Linear(
            num_vars * embed_dim,  # 输入维度 D*E (137 * 64=8768)
            embed_dim             # 输出维度 E (64)
        )
        self.time_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # ===== 变量注意力分支 =====
        self.var_proj = nn.Linear(
            seq_len * embed_dim,   # 输入维度 T*E (12 * 64=768)
            embed_dim             # 输出维度 E (64)
        )
        self.var_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x):
        B, T, D, E = x.size()
        
        # === 时间维度处理 ===
        # 输入形状: (B, T, D, E) → (B, T, D*E)
        x_time = x.reshape(B, T, -1)
        # 投影: (B, T, 8768) → (B, T, 64)
        x_time_proj = self.time_proj(x_time)
        # 注意力计算
        time_attn, _ = self.time_attention(x_time_proj, x_time_proj, x_time_proj)
        # 恢复形状: (B, T, 64) → (B, T, 1, 64) → 广播到 (B, T, 137, 64)
        time_attn = time_attn.unsqueeze(2).expand(-1, -1, D, -1)
        
        # === 变量维度处理 ===
        # 输入形状: (B, T, D, E) → (B, D, T, E)
        x_var = x.permute(0, 2, 1, 3)
        # 展平: (B, D, 12, 64) → (B, D, 768)
        x_var_flat = x_var.reshape(B, D, -1)
        # 投影: (B, D, 768) → (B, D, 64)
        x_var_proj = self.var_proj(x_var_flat)
        # 注意力计算
        var_attn, _ = self.var_attention(x_var_proj, x_var_proj, x_var_proj)
        # 恢复形状: (B, D, 64) → (B, D, 1, 64) → 广播到 (B, T, 137, 64)
        var_attn = var_attn.unsqueeze(2).permute(0, 2, 1, 3).expand(-1, T, -1, -1)
        
        # 特征融合
        return time_attn + var_attn


# 3. 生成式补全模块
class CVAE(nn.Module):
    def __init__(self, embed_dim, latent_dim=32):
        super().__init__()
        # 编码器保持原样
        self.encoder_mu = nn.Linear(embed_dim, latent_dim)
        self.encoder_logvar = nn.Linear(embed_dim, latent_dim)
        
        # 解码器输入维度调整为 latent_dim + embed_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, x):
        B, T, D, E = x.size()
        
        # === 编码阶段 ===
        # 沿时间维度取均值 (B, T, D, E) -> (B, D, E)
        x_flat = x.mean(dim=1)
        
        # 生成潜在变量 (B, D, latent_dim)
        mu = self.encoder_mu(x_flat)
        logvar = self.encoder_logvar(x_flat)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)  # (B, D, latent_dim)
        
        # === 解码阶段 ===
        # 扩展潜在变量到时间维度 (B, D, latent_dim) -> (B, T, D, latent_dim)
        z_expanded = z.unsqueeze(1).expand(-1, T, -1, -1)
        
        # 获取时序特征 (B, T, D, E)
        time_feature = x  # 原始输入已包含时序信息
        
        # 拼接特征 (B, T, D, latent_dim + E)
        combined = torch.cat([time_feature, z_expanded], dim=-1)
        
        # 解码重构 (B, T, D, E)
        x_decoded = self.decoder(combined)
        
        return x_decoded, mu, logvar

    def compute_loss(self, recon_x, x, mu, logvar, mask):
        # 仅计算非掩码部分的重构损失
        recon_loss = F.mse_loss(recon_x * (1 - mask), x * (1 - mask), reduction="mean")

        # KL散度
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + 0.1 * kl_loss

        return total_loss, recon_loss, kl_loss


# 4. 自监督训练模块
class NonStructureVSFModel(nn.Module):
    def __init__(self, num_vars, seq_len, embed_dim=64, latent_dim=32, num_heads=4, dropout=0.1):
        super().__init__()
        self.embedding = VariatesEmbedding(num_vars, embed_dim)
        self.attention = DualBranchAttention(
            num_vars=num_vars,
            seq_len=seq_len,  # 新增seq_len参数
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.cvae = CVAE(embed_dim, latent_dim)

    def forward(self, x, mask):
        # 数据嵌入
        x_embed = self.embedding(x)

        # 交互关系编码
        context = self.attention(x_embed)

        # 生成式补全
        recon_x, mu, logvar = self.cvae(context)
        return recon_x, mu, logvar

    def compute_loss(self, recon_x, x, mu, logvar, mask):
        return self.cvae.compute_loss(recon_x, x, mu, logvar, mask)


# 5. 推理与部署模块 (修正版)
class NonStructuralVSFDataset(Dataset):
    def __init__(self, data_dir, split='train', seq_len=12, pred_len=12, mask_ratio=0.15, seed=42):
        np.random.seed(seed)

        # 读取 .npz 数据
        data_file = os.path.join(data_dir, f"{split}.npz")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"数据文件 {data_file} 不存在，请检查路径和文件名是否正确。")
        
        # 加载数据
        data = np.load(data_file)
        self.data = data['x']  # (samples, seq_len, num_vars, 1)
        self.target = data['y']  # (samples, seq_len, num_vars, 1)

        # 移除可能存在的额外通道维度
        if self.data.ndim == 4 and self.data.shape[-1] == 1:
            self.data = self.data.squeeze(-1)
            self.target = self.target.squeeze(-1)

        print(f"Loaded {split} data with shape: {self.data.shape}, {self.target.shape}")

        # 生成 masks
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.masks = self._generate_masks(mask_ratio)

    def _generate_masks(self, mask_ratio):
        masks = np.ones_like(self.data, dtype=np.float32)
        num_samples, seq_len, num_vars = self.data.shape
        for i in range(num_samples):
            for var in range(num_vars):
                mask_indices = np.random.choice(seq_len, int(seq_len * mask_ratio), replace=False)
                masks[i, mask_indices, var] = 0.0
        return masks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.masks[idx], dtype=torch.float32),
            torch.tensor(self.target[idx], dtype=torch.float32)
        )


def get_dataloader(data_dir, batch_size=64, split='train'):
    dataset = NonStructuralVSFDataset(data_dir, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
