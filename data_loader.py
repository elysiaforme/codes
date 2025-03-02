import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader


class TipDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.data['tip'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npy_location = self.data.iloc[idx]['npy_location']
        features = np.load(npy_location)
        label = self.labels[idx]
        if self.transform:
            features = self.transform(features)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def get_data_loader(csv_path, batch_size=32, shuffle=True, transform=None, num_workers=0):
    dataset = TipDataset(csv_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader
