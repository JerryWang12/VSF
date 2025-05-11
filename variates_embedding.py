import torch
import torch.nn as nn
import math


class VariatesEmbedding(nn.Module):
    def __init__(self, num_vars, embedding_dim, max_len=5000):
        super().__init__()
        # 可学习的变量嵌入
        self.var_embedding = nn.Embedding(num_vars, embedding_dim)

        # 时间位置信息
        self.position_encoding = self._generate_positional_encoding(embedding_dim, max_len)

    def _generate_positional_encoding(self, embedding_dim, max_len):
        # 生成位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.size()
        device = x.device

        # 获取变量嵌入 (D, E)
        var_embed = self.var_embedding(torch.arange(D, device=device))  # (D, E)

        # 复制到时间维度 (B, T, D, E)
        var_embed = var_embed.unsqueeze(0).unsqueeze(0).expand(B, T, D, -1)

        # 加入时间位置信息
        pos_embed = self.position_encoding[:T, :].to(device).unsqueeze(1).expand(-1, D, -1)  # (T, D, E)
        pos_embed = pos_embed.unsqueeze(0).expand(B, -1, -1, -1)  # (B, T, D, E)

        # 最终输出 (B, T, D, E)
        return var_embed + pos_embed
