import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class NonStructureVSFDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.data, self.target = self.load_data()

    def load_data(self):
        # 加载数据
        data_file = os.path.join(self.data_dir, f"{self.split}_data.npy")
        target_file = os.path.join(self.data_dir, f"{self.split}_target.npy")

        # 检查文件是否存在
        if not os.path.exists(data_file) or not os.path.exists(target_file):
            raise FileNotFoundError(f"Data or target file not found in {self.data_dir}!")

        # 加载数据
        data = np.load(data_file)
        target = np.load(target_file)

        # 数据完整性检查
        assert data.shape[0] == target.shape[0], "Data and target size mismatch!"
        assert data.ndim == 3 and target.ndim == 3, "Expected 3D tensors (B, T, D)"
        assert data.shape[2] == target.shape[2], "Input and target must have the same number of variables"

        print(f"Loaded {self.split} data: {data.shape}, target: {target.shape}")
        return data, target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回一个样本 (B, T, D) 和对应的标签
        sample = self.data[idx]
        target = self.target[idx]

        # 转换为 PyTorch 张量
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def get_non_structure_data_loader(data_dir, batch_size=64, split='train', shuffle=True):
    dataset = NonStructureVSFDataset(data_dir, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
