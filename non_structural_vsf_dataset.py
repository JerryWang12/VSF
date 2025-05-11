# data/non_structural_vsf_dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random


class NonStructuralVSFDataset(Dataset):
    def __init__(self, data_path, seq_len=12, pred_len=12, mask_ratio=0.15, split='train', seed=42):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mask_ratio = mask_ratio
        self.data_path = data_path
        self.split = split
        self.seed = seed

        # Set random seed for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Load data
        data_file = os.path.join(data_path, f"{split}_data.npy")
        target_file = os.path.join(data_path, f"{split}_target.npy")
        self.data = np.load(data_file)  # (N, T, D)
        self.target = np.load(target_file)  # (N, T, D)

        # Verify data consistency
        assert self.data.shape[0] == self.target.shape[0], "Mismatch in data and target size"
        assert self.data.shape[1] == self.seq_len, f"Expected sequence length {self.seq_len}, got {self.data.shape[1]}"
        assert self.target.shape[
                   1] == self.pred_len, f"Expected prediction length {self.pred_len}, got {self.target.shape[1]}"
        assert self.data.shape[2] == self.target.shape[2], "Mismatch in variable dimensions"

        # Generate masks
        self.masks = self._generate_masks()

    def _generate_masks(self):
        """
        Generate masks where a fixed ratio of variables are masked entirely.
        """
        num_samples, _, num_vars = self.data.shape
        masks = np.ones((num_samples, self.seq_len, num_vars), dtype=np.float32)

        for i in range(num_samples):
            # Randomly select a subset of variables to mask
            mask_indices = np.random.choice(num_vars, int(num_vars * self.mask_ratio), replace=False)
            masks[i, :, mask_indices] = 0.0  # Mask entire variables

        return masks

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]  # (T, D)
        y = self.target[idx]  # (T, D)
        mask = self.masks[idx]  # (T, D)

        # Apply mask to the input data
        x_masked = x * mask

        # Convert to PyTorch tensors
        return torch.tensor(x_masked, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32), torch.tensor(y,
                                                                                                                  dtype=torch.float32)
