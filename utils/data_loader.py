# utils/data_loader.py

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class TrainDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = self.seq_len + self.pred_len
        self.valid_indices = self._get_valid_indices()

    def _get_valid_indices(self):
        valid_indices = []
        for i in range(len(self.data) - self.total_len + 1):
            start_block = i // 1000
            end_block = (i + self.total_len - 1) // 1000
            if start_block == end_block and i % 1000 >= self.total_len:
                valid_indices.append(i)
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        start_idx = self.valid_indices[index]
        x_enc = self.data[start_idx:start_idx + self.seq_len]
        y = self.data[start_idx + self.seq_len:start_idx + self.total_len]
        return x_enc, y

class TestDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return (len(self.data) - self.seq_len - self.pred_len + 1) // 1000 + 1

    def __getitem__(self, index):
        start_idx = index * 1000
        x_enc = self.data[start_idx:start_idx + self.seq_len]
        y = self.data[start_idx + self.seq_len:start_idx + self.seq_len + self.pred_len]
        return x_enc, y

def load_data(csv_file, target_column):
    data = pd.read_csv(csv_file)
    target_data = data[target_column].values.reshape(-1, 1)
    return torch.FloatTensor(target_data)

def create_dataloaders(data, seq_len, pred_len, batch_size, num_workers):
    train_dataset = TrainDataset(data, seq_len, pred_len)
    test_dataset = TestDataset(data, seq_len, pred_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader