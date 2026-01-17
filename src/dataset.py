import torch
from torch.utils.data import Dataset
import numpy as np

class HCALDataset(Dataset):
    def __init__(self, data, label):
        self.data = data.astype(np.float32)
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = torch.tensor(x).unsqueeze(0)      # (1, 64, 72)
        x = x.repeat(3, 1, 1)                  # (3, 64, 72)
        y = torch.tensor(self.label).long()
        return x, y
