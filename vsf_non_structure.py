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
        pos_embed = self.position_encoding[:T, :].to(device)

        # 复制到 B, T, D, E
        var_embed = var_embed.unsqueeze(0).unsqueeze(0).expand(B, T, D, -1)
        pos_embed = pos_embed.unsqueeze(0).unsqueeze(1).expand(B, -1, D, -1)

        # 加入时间位置信息
        return var_embed + pos_embed


# 2. 交互关系编码模块
class DualBranchAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.time_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.var_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        B, T, D, E = x.size()

        # 时间维度自注意力
        x_time = x.view(B, T, D * E)
        time_attn_output, _ = self.time_attention(x_time, x_time, x_time)
        time_attn_output = time_attn_output.view(B, T, D, E)

        # 变量维度自注意力
        x_var = x.permute(0, 2, 1, 3).reshape(B, D, T * E)
        var_attn_output, _ = self.var_attention(x_var, x_var, x_var)
        var_attn_output = var_attn_output.view(B, D, T, E).permute(0, 2, 1, 3)

        # 融合时间和变量特征
        return time_attn_output + var_attn_output


# 3. 生成式补全模块
class CVAE(nn.Module):
    def __init__(self, embed_dim, latent_dim=32):
        super().__init__()
        self.encoder_mu = nn.Linear(embed_dim, latent_dim)
        self.encoder_logvar = nn.Linear(embed_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim + embed_dim, embed_dim)

    def forward(self, x):
        B, T, D, E = x.size()
        # 压缩到 (B, D, E)
        x_flat = x.mean(dim=1)

        # CVAE编码
        mu = self.encoder_mu(x_flat)
        logvar = self.encoder_logvar(x_flat)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        # CVAE解码
        z_expanded = z.unsqueeze(1).expand(-1, T, -1)
        x_decoded = self.decoder(torch.cat([x.mean(dim=2), z_expanded], dim=-1))

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
    def __init__(self, num_vars, embed_dim=64, latent_dim=32, num_heads=4, dropout=0.1):
        super().__init__()
        self.embedding = VariatesEmbedding(num_vars, embed_dim)
        self.attention = DualBranchAttention(embed_dim, num_heads, dropout)
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

        # 修改数据文件名
        data_file = os.path.join(data_dir, f"{split}.npz")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"数据文件 {data_file} 不存在，请检查路径和文件名是否正确。")
        
        # 读取 .npz 数据
        data = np.load(data_file)
        self.data = data['x']
        self.target = data['y']
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.masks = self._generate_masks(mask_ratio)

        print(f"Loaded {split} data with shape: {self.data.shape}, {self.target.shape}")

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



# 测试完整流程
if __name__ == "__main__":
    model = NonStructureVSFModel(num_vars=137).cuda()
    dataloader = get_dataloader('./data/SOLAR_non_structural', batch_size=32, split='train')

    for x, mask, target in dataloader:
        x, mask, target = x.cuda(), mask.cuda(), target.cuda()
        recon_x, mu, logvar = model(x, mask)
        loss, recon_loss, kl_loss = model.compute_loss(recon_x, x, mu, logvar, mask)
        print(f"Loss: {loss.item()}, Recon Loss: {recon_loss.item()}, KL Loss: {kl_loss.item()}")
        break
