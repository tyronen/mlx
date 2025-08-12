import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from utils import log_transform_plus1, time_transform

class PrecomputedNPZDataset(Dataset):
    def __init__(self, npz_path, time_decay=None):
        
        data = np.load(npz_path, allow_pickle=True)

        self.features_num = torch.tensor(data['features_num'], dtype=torch.float32)
        # Load title embeddings (precomputed)
        self.title_embeddings = torch.tensor(data['title_embeddings'], dtype=torch.float32)

        # Load categorical indices
        self.domain_indices = torch.tensor(data['domain_index'], dtype=torch.long)
        self.tld_indices = torch.tensor(data['tld_index'], dtype=torch.long)
        self.user_indices = torch.tensor(data['user_index'], dtype=torch.long)

        # Targets (with log1p transform)
        self.targets = torch.log10(torch.tensor(data['targets'], dtype=torch.float32) + 1)

        # # Delta_t and time decay weights (optional)
        # self.has_delta_t = 'delta_t' in data.files
        # self.time_decay = time_decay

        # if self.has_delta_t:
        #     self.delta_t = torch.tensor(data['delta_t'].astype(np.float32), dtype=torch.float32)

        #     if self.time_decay is not None:
        #         self.weights = torch.exp(-self.time_decay * self.delta_t)
        #     else:
        #         self.weights = torch.ones_like(self.targets)
        # else:
        #     self.delta_t = None
        #     self.weights = torch.ones_like(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        features_num = self.features_num[idx]
        title_emb = self.title_embeddings[idx]
        domain_idx = self.domain_indices[idx]
        tld_idx = self.tld_indices[idx]
        user_idx = self.user_indices[idx]
        target = self.targets[idx]
        # weight = self.weights[idx]

        return features_num, title_emb, domain_idx, tld_idx, user_idx, target


